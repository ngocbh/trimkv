# HeadKV: head-level KV cache compression with static, importance-weighted budgets.
# Paper: arXiv 2410.19258 (Fu et al., 2024) — "Not All Heads Matter".
#
# Ported from third_party/HeadKV/headkv/snapkv_utils.py (ReasonSnapKVCluster) and
# adapted to TrimKV's rkv/ AdaKV integration: it reuses ALL of AdaKV's flattened
# per-head variable-length machinery (AdaKVDynamicCache, the adakv attention
# forward, flash_attn_varlen), and only changes the SOURCE of the per-head budget:
#   AdaKV  -> per-head budget chosen DYNAMICALLY from attention (global top-k + floor)
#   HeadKV -> per-head budget fixed STATICALLY from precomputed head-importance scores
#
# Head-importance scores are precomputed per model by a reason/retrieval-in-a-haystack
# detection (see rkv/headkv_detect.py) and stored as {"L-H": [per-probe scores]} JSON.
# Qwen3-VL-8B-Thinking is GQA (32 Q heads, 8 KV heads): scores are per Q-head and are
# mean-aggregated over each group of 4 Q-heads to a per-KV-head budget (matching
# gqa_func='mean' and conserving the per-layer total).

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadKV:
    def __init__(
        self,
        budget=1024,
        window_size=8,
        kernel_size=7,
        pooling="maxpool",
        beta=1.2,
        temp=1.0,
        head_score_path=None,
        head_choice="reason",
        gqa_func="mean",
        layer_idx=None,
        num_hidden_layers=None,
        num_attention_heads=32,
        num_key_value_heads=8,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        # self.budget is the per-head budget EXCLUDING the window (matches AdaKV).
        self.budget = budget - window_size
        self.base_capacity = budget
        self.beta = float(beta)
        self.temp = float(temp)
        self.gqa_func = gqa_func
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.head_score_path = self._resolve_score_path(head_score_path, head_choice)
        # Per-(layer, KV-head) static budget, EXCLUDING the window. Shape [L, H_kv].
        self.kv_head_capacity = self._compute_kv_head_capacity()

    def _resolve_score_path(self, head_score_path, head_choice):
        if head_score_path and os.path.exists(head_score_path):
            return head_score_path
        # Default location: rkv/head_score/Qwen3-VL-8B-Thinking_<choice>.json
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rkv/
        cand = os.path.join(here, "head_score", f"Qwen3-VL-8B-Thinking_{head_choice}.json")
        return cand

    def _compute_kv_head_capacity(self):
        """Replicates ReasonSnapKVCluster.__init__ score->budget (snapkv_utils.py:379-389),
        then mean-aggregates 32 Q-head budgets into 8 KV-head budgets (GQA)."""
        L = self.num_hidden_layers
        Hq = self.num_attention_heads
        Hkv = self.num_key_value_heads
        groups = self.num_key_value_groups

        with open(self.head_score_path, "r") as f:
            head_list = json.loads(f.readline())
        # Per-Q-head scalar = mean of its per-probe scores. Keys ordered 0-0..(L-1)-(Hq-1).
        head_score = np.array([np.mean(v) for v in head_list.values()], dtype=np.float64)
        assert head_score.shape[0] == L * Hq, (
            f"head_score has {head_score.shape[0]} entries, expected L*Hq={L*Hq}"
        )
        head_score = torch.tensor(head_score / head_score.sum())
        head_score = torch.pow(head_score, self.temp)
        head_score = head_score / torch.sum(head_score)
        total_attention = head_score.reshape(L, Hq)  # distribution over all L*Hq heads

        base = self.base_capacity - self.window_size  # per-head budget excl. window
        total_pool_capacity = (base // self.beta) * L * Hq
        min_num = base - base // self.beta
        head_cap_q = torch.round(total_attention * total_pool_capacity + min_num).int()  # [L, Hq]

        # GQA: aggregate 4 Q-heads -> 1 KV-head (mean conserves the per-layer total).
        kv_cap = head_cap_q.reshape(L, Hkv, groups).float().mean(dim=-1).round().int()  # [L, Hkv]
        kv_cap = torch.clamp(kv_cap, min=1)
        return kv_cap

    def update_compression_config(self, **compression_config):
        changed = False
        if "budget" in compression_config:
            self.base_capacity = compression_config["budget"]
            self.budget = self.base_capacity - self.window_size
            changed = True
        for k in ("beta", "temp"):
            if k in compression_config:
                setattr(self, k, float(compression_config[k]))
                changed = True
        if "window_size" in compression_config:
            self.window_size = compression_config["window_size"]
            changed = True
        if changed:
            self.kv_head_capacity = self._compute_kv_head_capacity()

    # ---- attention scoring (flattened, GQA-aware): copied from AdaKV.calc_attn_score ----
    def calc_attn_score(self, key_states, query_states, head_lens, cu_klen):
        bsz, num_heads, q_len, head_dim = query_states.shape
        assert bsz == 1, "Only batch size 1 is supported in calc_attn_score"
        assert key_states.dim() == 2, "key_states should be in flatten view"
        num_kv_heads = head_lens.shape[0]
        num_heads_per_kv = num_heads // num_kv_heads

        cur_query_states = query_states[:, :, -self.window_size:, :]

        attn_weights = []
        for head_idx in range(num_kv_heads):
            head_key_states = key_states[cu_klen[head_idx]:cu_klen[head_idx + 1], :].view(1, 1, -1, head_dim)
            head_key_states = head_key_states.expand(1, num_heads_per_kv, -1, -1)
            head_query_states = cur_query_states[:, head_idx * num_heads_per_kv:(head_idx + 1) * num_heads_per_kv, :, :]
            head_attn_weights = torch.matmul(head_query_states, head_key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(head_attn_weights.dtype).min,
                              device=head_attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=head_attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(head_attn_weights.device)
            attention_mask = mask[None, None, :, :]
            head_attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            head_attn_weights = nn.functional.softmax(head_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            head_attn_weights_mean = head_attn_weights.mean(dim=-2)

            if self.gqa_func == "max":
                head_attn_weights_mean = head_attn_weights_mean.max(dim=-2, keepdim=True).values
            elif self.gqa_func == "mean":
                head_attn_weights_mean = head_attn_weights_mean.mean(dim=-2, keepdim=True)
            else:
                raise ValueError("gqa_func not supported")

            if self.pooling == "avgpool":
                head_attn_weights_mean = F.avg_pool1d(head_attn_weights_mean, kernel_size=self.kernel_size,
                                                      padding=self.kernel_size // 2, stride=1)
            elif self.pooling == "maxpool":
                head_attn_weights_mean = F.max_pool1d(head_attn_weights_mean, kernel_size=self.kernel_size,
                                                      padding=self.kernel_size // 2, stride=1)
            else:
                raise ValueError("Pooling method not supported")
            attn_weights.append(head_attn_weights_mean)

        attn_weights_mean = torch.cat(attn_weights, dim=2).squeeze()
        return attn_weights_mean

    def update_kv(self, origin_key_states, query_states, origin_value_states, head_lens, cu_klen):
        """Like AdaKV.update_kv_gqa, but the per-head kept-token count is the STATIC
        importance-weighted budget self.kv_head_capacity[layer_idx][head] instead of a
        dynamic global-top-k allocation."""
        _device = origin_key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = head_lens.shape[0]

        # Static per-KV-head budgets for this layer (excludes window).
        head_capacity = self.kv_head_capacity[self.layer_idx].to(_device)

        # If nothing to compress (cache already within total budget), no-op.
        if int(head_capacity.sum().item()) + num_kv_heads * self.window_size >= origin_key_states.shape[0]:
            return origin_key_states, origin_value_states, head_lens, cu_klen

        attn_score = self.calc_attn_score(origin_key_states, query_states, head_lens, cu_klen)

        heads_key_states = []
        heads_value_states = []
        new_head_lens = []
        assert bsz == 1

        for head_idx in range(num_kv_heads):
            head_attn_score = attn_score[cu_klen[head_idx]:cu_klen[head_idx + 1] - self.window_size]
            head_key_states = origin_key_states[cu_klen[head_idx]:cu_klen[head_idx + 1], :]
            head_value_states = origin_value_states[cu_klen[head_idx]:cu_klen[head_idx + 1], :]

            cap = int(head_capacity[head_idx].item())
            if cap >= head_attn_score.shape[-1]:
                # keep this head uncompressed
                new_head_lens.append(head_key_states.shape[0])
                heads_key_states.append(head_key_states.view(-1, head_dim))
                heads_value_states.append(head_value_states.view(-1, head_dim))
                continue

            cache_index = torch.topk(head_attn_score, k=cap, dim=-1).indices
            new_head_lens.append(cache_index.shape[-1] + self.window_size)

            cache_index = cache_index.to(torch.long).unsqueeze(-1).expand(-1, head_dim)
            top_Kcache = head_key_states.gather(dim=0, index=cache_index)
            top_Vcache = head_value_states.gather(dim=0, index=cache_index)
            selected_k = torch.cat([top_Kcache, head_key_states[-self.window_size:, :]], dim=0)
            selected_v = torch.cat([top_Vcache, head_value_states[-self.window_size:, :]], dim=0)

            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        head_lens = torch.tensor(new_head_lens, device=_device, dtype=torch.int32)
        cu_klen = torch.cat(
            [torch.tensor([0], device=_device, dtype=torch.int32), torch.cumsum(head_lens, dim=0)], dim=0
        ).to(torch.int32)

        return heads_key_states, heads_value_states, head_lens, cu_klen
