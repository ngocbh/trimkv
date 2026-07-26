# CAKE: Cascading and Adaptive KV cache Eviction with layer preferences.
# Paper: arXiv 2503.12491 (Qin et al., 2025).
#
# Ported from third_party/cakekv/cake/{cake_cache.py,model/modify_qwen2.py,utils.py}
# and adapted to TrimKV's rkv/ integration (SnapKV-style, standard RKVDynamicCache,
# GQA-aware, per-KV-head eviction, compresses at both prefill and decode).
#
# CAKE differs from SnapKV by allocating a DIFFERENT budget per LAYER, derived from
# each layer's attention statistics (dispersion=entropy, temporal-shift=variance).
# Because a layer's budget depends on all layers' scores, the per-layer CAKE objects
# share a single `CakeAllocator`; the attention forward defers prefill eviction to the
# last layer, where the allocator splits a fixed global budget across layers.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Copied from transformers.models.llama.modeling_llama.repeat_kv (GQA support).
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Ported verbatim from third_party/cakekv/cake/utils.py:15-19
def calculate_entropy(attention_scores: torch.Tensor) -> torch.Tensor:
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))
    return entropy.to(dtype=torch.float32)


# Ported verbatim from third_party/cakekv/cake/utils.py:21-67
def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):
    budget_list = np.array(budget_list, dtype=int)
    # Limit the budget of all layers to not exceed seq_len
    excess = np.maximum(budget_list - seq_len, 0)
    budget_list = np.minimum(budget_list, seq_len)

    total_excess = np.sum(excess)
    if total_excess > 0:
        valid_indices = budget_list < seq_len
        num_valid = np.sum(valid_indices)
        if num_valid > 0:
            distribute_per_layer = total_excess // num_valid
            remainder = total_excess % num_valid
            budget_list[valid_indices] += distribute_per_layer
            budget_list[np.where(valid_indices)[0][:remainder]] += 1

    # Ensure total budget equals total_budget
    current_total_budget = np.sum(budget_list)
    budget_diff = total_budget - current_total_budget
    if budget_diff != 0:
        if budget_diff > 0:
            valid_indices = budget_list < seq_len
        else:
            valid_indices = budget_list > 1
        num_valid = np.sum(valid_indices)
        if num_valid > 0:
            adjust_per_layer = abs(budget_diff) // num_valid
            remainder = abs(budget_diff) % num_valid
            if budget_diff > 0:
                budget_list[valid_indices] += adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1
            else:
                budget_list[valid_indices] -= adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] -= 1

    return budget_list.tolist()


class CakeAllocator:
    """Shared across all layers of a model (one instance passed into every layer's
    CAKE object). Accumulates each layer's preference score + eviction score during
    prefill, then splits a fixed global budget across layers proportionally to the
    preference scores. Reset at the start of every new sample's prefill."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pref_scores = {}   # layer_idx -> float
        self.hh_scores = {}     # layer_idx -> tensor (bsz, num_kv_heads, seqlen - window)
        self.layer_budget = {}  # layer_idx -> int (per-KV-head budget, excluding window)

    def register(self, layer_idx, pref_score, hh_score):
        self.pref_scores[layer_idx] = pref_score
        self.hh_scores[layer_idx] = hh_score

    def allocate(self, total_size, seq_len_minus_window, num_layers):
        prefs = [self.pref_scores[i] for i in range(num_layers)]
        denom = sum(prefs)
        if denom <= 0:
            budgets = [total_size / num_layers] * num_layers
        else:
            budgets = [p / denom * total_size for p in prefs]
        budgets = adjust_budgets(budgets, total_size, seq_len_minus_window, num_layers)
        for i, b in enumerate(budgets):
            self.layer_budget[i] = int(b)
        return self.layer_budget


class CAKE:
    def __init__(
        self,
        budget=1024,
        window_size=32,
        kernel_size=5,
        tau1=1.6,
        tau2=0.6,
        gamma=200.0,
        layer_idx=None,
        num_hidden_layers=None,
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
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        # Shared allocator (same object for every layer). Created in load_cake_model.
        self.allocator = allocator if allocator is not None else CakeAllocator()

    def update_compression_config(self, **compression_config):
        self.budget = compression_config.get("budget", self.budget)
        self.window_size = compression_config.get("window_size", self.window_size)
        self.kernel_size = compression_config.get("kernel_size", self.kernel_size)

    @property
    def total_size(self):
        # Global budget shared across layers = average per-layer per-head budget * L,
        # matching the paper's M x #layers x #heads global budget (window kept on top).
        return (self.budget - self.window_size) * self.num_hidden_layers

    def _window_attn(self, key_states, query_states, causal):
        """Observation-window attention: last-window queries vs all keys, GQA-repeated
        to the query-head count. Returns softmaxed (bsz, num_q_heads, W, S)."""
        bsz, num_kv_heads, S, head_dim = key_states.shape
        num_q_heads = query_states.shape[1]
        W = query_states.shape[-2]
        rep_k = repeat_kv(key_states, num_q_heads // num_kv_heads)
        attn = torch.matmul(query_states, rep_k.transpose(2, 3)) / math.sqrt(head_dim)
        if causal:
            mask = torch.full((W, W), torch.finfo(attn.dtype).min, device=attn.device)
            mask_cond = torch.arange(mask.size(-1), device=attn.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            attn[:, :, -W:, -W:] += mask[None, None, :, :]
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn

    def compute_scores(self, key_states, query_states, key_padding=None):
        """Prefill scoring (ref modify_qwen2.py:114-144). Returns:
        - pref_score: scalar float, layer preference = disp^(1/tau1) * var^(1/tau2)
        - hh_score:   (bsz, num_kv_heads, S - window) eviction indicator (mean + gamma*var)
        key_padding: optional (bsz, S) bool mask (True = padding key); used for bs>1 so
        padded tokens are never kept. When None (the bs=1 path) behaviour is unchanged.
        """
        bsz, num_kv_heads, S, head_dim = key_states.shape
        num_q_heads = query_states.shape[1]
        groups = num_q_heads // num_kv_heads
        W = self.window_size

        attn = self._window_attn(key_states, query_states, causal=True)  # (b, q, W, S)

        # Preference score (a single scalar per layer).
        disp = calculate_entropy(attn[:, :, -W:, :-W])
        var = torch.var(attn[:, :, -W:, :-W], dim=-2).sum()
        pref_score = float((disp ** (1.0 / self.tau1) * var ** (1.0 / self.tau2)).cpu())

        # Eviction indicator: mean + gamma * variance over the window queries.
        attention_score = attn[:, :, -W:, :]
        attn_cache = attention_score.mean(dim=-2) + self.gamma * attention_score.var(dim=-2)  # (b, q, S)
        attn_cache = attn_cache[:, :, :-W]
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_kv_heads, groups, -1).mean(dim=-2)  # (b, kv, S - W)
        if key_padding is not None:
            pad = key_padding[:, : attn_cache.shape[-1]]  # (bsz, S - W), True = padding
            attn_cache = attn_cache.masked_fill(pad[:, None, :], torch.finfo(attn_cache.dtype).min)
        return pref_score, attn_cache

    def _select(self, key_states, value_states, hh_score, budget):
        """Keep top-`budget` past tokens per KV head (by hh_score) + the last
        `window_size` tokens verbatim (ref cake_cache.py:249-275 / 294-324)."""
        bsz, num_kv_heads, S, head_dim = key_states.shape
        budget = int(min(budget, hh_score.shape[-1]))
        indices = hh_score.topk(budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past = key_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        v_past = value_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        return torch.cat([k_past, k_cur], dim=2), torch.cat([v_past, v_cur], dim=2)

    def evict_prefill(self, key_states, value_states, hh_score, budget):
        """Evict a single layer's stored KV using its precomputed prefill hh_score."""
        return self._select(key_states, value_states, hh_score, budget)

    def decode_evict(self, key_states, value_states, query_states, budget):
        """Decode-time eviction (ref cake_cache.py:294-324): recompute the window
        indicator (mean over window, no gamma term) and cap the layer at budget+window."""
        bsz, num_kv_heads, S, head_dim = key_states.shape
        groups = query_states.shape[1] // num_kv_heads
        if S <= budget + self.window_size:
            return key_states, value_states
        attn = self._window_attn(key_states, query_states, causal=False)  # (b, q, W, S)
        attn_cache = attn[:, :, :, : -self.window_size].mean(dim=-2)  # (b, q, S - W)
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_kv_heads, groups, -1).mean(dim=-2)  # (b, kv, S - W)
        return self._select(key_states, value_states, attn_cache, budget)
