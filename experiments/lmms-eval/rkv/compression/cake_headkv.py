# CakeHeadKV: training-free LAYER + HEAD KV-budget allocation — CAKE x HeadKV.
#
# This is the training-free analog of DBTrimKV's layer+head allocation, added to
# isolate what the *allocation axes* contribute (vs training). It composes:
#   - CAKE (arXiv 2503.12491): a DYNAMIC per-LAYER budget from this prompt's
#     attention (entropy x variance preference), summing to a fixed global total.
#   - HeadKV (arXiv 2410.19258): STATIC per-HEAD importance from precomputed
#     retrieval-head scores, used here to distribute each layer's budget across its
#     KV heads.
# cap[l,h] = within_layer_head_weight[l,h] * (cake_layer_budget[l] * num_kv_heads)
#
# Runs on the AdaKV flattened per-head cache (heads hold different counts) with
# CAKE-style deferred eviction (all layers evicted at the last prefill layer, once
# every layer's preference score is known). batch_size=1.

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cake import adjust_budgets, calculate_entropy


class CakeHeadKVAllocator:
    """Shared across all layers. Accumulates per-layer CAKE preference + per-head
    ranking scores during prefill, then allocates cap[l,h] = layer(CAKE) x head(HeadKV)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pref = {}
        self.hh = {}
        self.head_lens = {}
        self.cu = {}
        self.layer_budget = {}  # layer_idx -> per-KV-head cap tensor (excl. window)

    def register(self, layer_idx, pref, hh, head_lens, cu):
        self.pref[layer_idx] = pref
        self.hh[layer_idx] = hh
        self.head_lens[layer_idx] = head_lens
        self.cu[layer_idx] = cu

    def allocate(self, total_size, num_layers, num_kv_heads, seq_len_minus_window, head_weight):
        prefs = [self.pref[i] for i in range(num_layers)]
        denom = sum(prefs) or 1.0
        budgets = [p / denom * total_size for p in prefs]  # CAKE per-layer, per-head avg
        budgets = adjust_budgets(budgets, total_size, seq_len_minus_window, num_layers)
        caps = {}
        for l in range(num_layers):
            layer_total = budgets[l] * num_kv_heads  # tokens for layer l across heads (excl window)
            w = head_weight[l].to(torch.float64)  # (num_kv_heads,), sums to 1 within layer
            cap = torch.clamp(torch.round(w * layer_total), min=1).int()
            caps[l] = cap
        self.layer_budget = caps
        return caps


class CakeHeadKV:
    def __init__(
        self,
        budget=1024,
        window_size=32,
        kernel_size=5,
        tau1=1.6,
        tau2=0.6,
        gamma=200.0,
        temp=1.0,
        head_score_path=None,
        head_choice="reason",
        layer_idx=None,
        num_hidden_layers=None,
        num_attention_heads=32,
        num_key_value_heads=8,
        allocator=None,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.gamma = float(gamma)
        self.temp = float(temp)
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.allocator = allocator if allocator is not None else CakeHeadKVAllocator()
        self.head_score_path = self._resolve_score_path(head_score_path, head_choice)
        # Static within-layer head-importance weights [L, H_kv] (each row sums to 1).
        self.head_weight = self._compute_head_weight()

    def _resolve_score_path(self, head_score_path, head_choice):
        if head_score_path and os.path.exists(head_score_path):
            return head_score_path
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rkv/
        return os.path.join(here, "head_score", f"Qwen3-VL-8B-Thinking_{head_choice}.json")

    def _compute_head_weight(self):
        L, Hq, Hkv, g = self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads, self.num_key_value_groups
        with open(self.head_score_path, "r") as f:
            head_list = json.loads(f.readline())
        score = np.array([np.mean(v) for v in head_list.values()], dtype=np.float64)
        assert score.shape[0] == L * Hq, f"{score.shape[0]} != L*Hq={L*Hq}"
        score = torch.tensor(score / score.sum())
        score = torch.pow(score, self.temp)
        kv_score = score.reshape(L, Hkv, g).mean(dim=-1)  # aggregate Q->KV heads [L, Hkv]
        w = kv_score / kv_score.sum(dim=1, keepdim=True)  # within-layer normalize
        return w

    def update_compression_config(self, **compression_config):
        if "budget" in compression_config:
            self.budget = compression_config["budget"]
        if "window_size" in compression_config:
            self.window_size = compression_config["window_size"]

    @property
    def total_size(self):
        return (self.budget - self.window_size) * self.num_hidden_layers

    def compute_scores(self, key_flat, query_states, head_lens, cu_klen, need_pref=True):
        """Per-KV-head window attention on the flattened cache. Returns:
        - pref: CAKE layer preference (entropy^(1/tau1) * var^(1/tau2)), or None
        - hh:   list of per-head ranking scores (mean + gamma*var, avgpool'd)."""
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        num_kv = head_lens.shape[0]
        g = num_q_heads // num_kv
        W = self.window_size
        cur_q = query_states[:, :, -W:, :]
        pref_disp = pref_var = None
        hh = []
        for h in range(num_kv):
            s, e = int(cu_klen[h]), int(cu_klen[h + 1])
            hk = key_flat[s:e, :].view(1, 1, -1, head_dim).expand(1, g, -1, -1)
            hq = cur_q[:, h * g:(h + 1) * g, :, :]
            A = torch.matmul(hq, hk.transpose(2, 3)) / math.sqrt(head_dim)  # (1,g,W,S_h)
            mask = torch.full((W, W), torch.finfo(A.dtype).min, device=A.device)
            mc = torch.arange(W, device=A.device)
            mask.masked_fill_(mc < (mc + 1).view(W, 1), 0)
            A[:, :, -W:, -W:] += mask[None, None, :, :]
            A = F.softmax(A, dim=-1, dtype=torch.float32).to(query_states.dtype)
            Am = A.mean(dim=1)  # (1, W, S_h) GQA-mean over groups
            if need_pref:
                past = Am[:, :, :-W]
                d = calculate_entropy(past)
                v = torch.var(past, dim=-2).sum()
                pref_disp = d if pref_disp is None else pref_disp + d
                pref_var = v if pref_var is None else pref_var + v
            hh_h = Am.mean(dim=-2) + self.gamma * Am.var(dim=-2)  # (1, S_h)
            hh_h = hh_h[:, :-W]
            hh_h = F.avg_pool1d(hh_h, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1).squeeze(0)
            hh.append(hh_h)
        pref = None
        if need_pref:
            pref = float((pref_disp ** (1.0 / self.tau1) * pref_var ** (1.0 / self.tau2)).cpu())
        return pref, hh

    def evict_layer(self, key_flat, value_flat, hh, head_lens, cu_klen, cap):
        """Evict a layer's flattened cache to per-head budgets cap[h] (+ window each)."""
        dev = key_flat.device
        head_dim = key_flat.shape[-1]
        num_kv = head_lens.shape[0]
        W = self.window_size
        ks, vs, lens = [], [], []
        for h in range(num_kv):
            s, e = int(cu_klen[h]), int(cu_klen[h + 1])
            hk = key_flat[s:e, :]
            hv = value_flat[s:e, :]
            hh_h = hh[h]
            c = int(cap[h].item())
            if c >= hh_h.shape[-1]:
                lens.append(hk.shape[0])
                ks.append(hk)
                vs.append(hv)
                continue
            idx = torch.topk(hh_h, c).indices.to(torch.long).unsqueeze(-1).expand(-1, head_dim)
            tk = hk[:-W].gather(0, idx)
            tv = hv[:-W].gather(0, idx)
            sk = torch.cat([tk, hk[-W:]], dim=0)
            sv = torch.cat([tv, hv[-W:]], dim=0)
            lens.append(sk.shape[0])
            ks.append(sk)
            vs.append(sv)
        ko = torch.cat(ks, dim=0)
        vo = torch.cat(vs, dim=0)
        hl = torch.tensor(lens, device=dev, dtype=torch.int32)
        cu = torch.cat([torch.tensor([0], device=dev, dtype=torch.int32), torch.cumsum(hl, dim=0).to(torch.int32)])
        return ko, vo, hl, cu
