from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
from einops import rearrange
from functools import partial
# from tiny_api_cuda import update_flatten_view
from trimkv.triton import update_flatten_view_triton

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache


import math
import torch


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    log_half = math.log(0.5)
    return torch.where(
        x < log_half,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def compute_log_G(
    log_beta: torch.Tensor,
    t: int | torch.Tensor,
    i: int | torch.Tensor,
    n: int | torch.Tensor, # number of lookahead tokens, used to compute the multi-step score; if n=1, multi-step score is equivalent to single-step score
    eps: float = 1e-9,
) -> torch.Tensor:
    dtype = log_beta.dtype
    device = log_beta.device

    t = torch.tensor(t, device=device, dtype=dtype) if not isinstance(t, torch.Tensor) else t.to(device=device, dtype=dtype)
    i = torch.tensor(i, device=device, dtype=dtype) if not isinstance(i, torch.Tensor) else i.to(device=device, dtype=dtype)
    n = torch.tensor(n, device=device, dtype=dtype) if not isinstance(n, torch.Tensor) else n.to(device=device, dtype=dtype)

    a = t + 1 - i

    x_num = n * log_beta
    x_den = log_beta

    # Only clamp accidental positive roundoff, not exact zeros.
    x_num = torch.where(x_num > 0, -torch.as_tensor(eps, dtype=dtype, device=device), x_num)
    x_den = torch.where(x_den > 0, -torch.as_tensor(eps, dtype=dtype, device=device), x_den)

    log_G = a * log_beta + _log1mexp(x_num) - _log1mexp(x_den)
    return log_G


class TrimKVCache(DynamicCache):
    def __init__(
        self,
        max_seq_len: int = None,
        memory_size: Optional[int] = None,
        buffer_size: int = 1,
        sliding_window_size: int = 0,
        strategy: str = 'fixed_budget',
        lookahead_steps: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.memory_size = memory_size
        self.buffer_size = buffer_size
        self.strategy = strategy
        assert strategy in ['fixed_budget'], "Only 'fix_budget' strategy are supported"
        self.sliding_window_size = sliding_window_size
        self.do_compress = (self.memory_size is not None)

        self.lookahead_steps = lookahead_steps

        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []
        self.attention_mask = None
        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None
        
        # log peak cached tokens and full memory
        self.peak_cached_tokens = None
        self.layers = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def get_seq_length(self) -> int:
        return self._seen_tokens

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        cached_length = self.get_cache_length()

        kv_offset = max(self._seen_tokens - cached_length + 1, 0)
        kv_length = cached_length + query_length

        return kv_length, kv_offset

    def get_cache_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_total_cached_tokens(self, num_key_value_heads: Optional[int] = None) -> int:
        if num_key_value_heads is None:
            num_key_value_heads = self.key_cache[0].shape[1] if self.key_cache else 0

        return sum(self.get_cache_length(layer_idx) * num_key_value_heads for layer_idx in range(len(self)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        retention_weights = cache_kwargs.get("retention_weights")
        attention_mask = cache_kwargs.get("attention_mask", None)
        # Update the cache
        if key_states is not None:
            bsz, num_heads, seq_len, dim = key_states.shape
            assert bsz == 1 or (attention_mask is None or attention_mask.dim() == 2), "Batch size greater than 1 is only supported when attention_mask is provided with shape (B, S) or None"
            self.attention_mask = attention_mask
            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += seq_len

            cache_positions = cache_position[None, None, :].expand_as(retention_weights) if cache_position.dim() == 1 else cache_position

            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))

                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.retention_weights.append(retention_weights)
                self.kv_positions.append(cache_positions)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                self.retention_weights[layer_idx] = retention_weights
                self.kv_positions[layer_idx] = cache_positions
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.retention_weights[layer_idx] = torch.cat([self.retention_weights[layer_idx], retention_weights], dim=-1)
                self.kv_positions[layer_idx] = torch.cat([self.kv_positions[layer_idx], cache_positions], dim=-1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], {}

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            self.retention_weights[layer_idx] = self.retention_weights[layer_idx][indices, ...]
            self.kv_positions[layer_idx] = self.kv_positions[layer_idx][indices, ...]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = TrimKVCache(device=self.device)
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.retention_weights = [tensor[i : i + split_size] for tensor in self.retention_weights]
            current_split.kv_positions = [tensor[i : i + split_size] for tensor in self.kv_positions]
            out.append(current_split)
        return out

    def compress(self):
        if not self.do_compress:
            return

        assert len(self.key_cache) > 0 and self.key_cache[0].numel() > 0, "Cache is empty, cannot compress"
        num_layers = len(self.key_cache)
        memory_size = self.memory_size
        buffer_size = self.buffer_size

        # layer wise compression
        for layer_idx in range(num_layers):
            if memory_size + buffer_size <= self.get_cache_length(layer_idx):
                key_states = self.key_cache[layer_idx]
                value_states = self.value_cache[layer_idx]
                kv_positions = self.kv_positions[layer_idx]
                retention_weights = self.retention_weights[layer_idx]
                
                log_beta = retention_weights.to(torch.float32)
                q_idx = self.get_seq_length() + 1
                if self.lookahead_steps == 1:
                    scores = (log_beta * (q_idx - kv_positions))
                else:
                    scores = compute_log_G(
                        log_beta,
                        q_idx,
                        kv_positions,
                        n=self.lookahead_steps,
                    )


                if self.sliding_window_size > 0:
                    scores[:, :, -self.sliding_window_size:] = float('inf')

                if self.attention_mask is not None:
                    # set the scores of the masked positions to -inf
                    mask = self.attention_mask[:, None, :key_states.shape[-2]]
                    scores = scores.masked_fill(~mask, float('-inf'))

                # get top-k (memory size) indices with highest alpha values to keep
                # top_k_indices = torch.topk(log_alpha, memory_size, dim=-1).indices
                top_k_indices = torch.topk(scores, memory_size, dim=-1).indices
                # sort the top-k indices to maintain order
                top_k_indices, _ = torch.sort(top_k_indices, dim=-1)

                # gather the top-k key and value states to the first position, using gather
                self.key_cache[layer_idx] = key_states.gather(-2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1]))
                self.value_cache[layer_idx] = value_states.gather(-2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, value_states.shape[-1]))
                self.retention_weights[layer_idx] = self.retention_weights[layer_idx].gather(-1, top_k_indices)
                self.kv_positions[layer_idx] = kv_positions.gather(-1, top_k_indices)
        
    def log(self, layer_idx: int = None):
        logs = {}
        if layer_idx is None:
            for layer_idx in range(len(self.key_cache)):
                logs[layer_idx] = {
                    "seen_tokens": self.get_seq_length(),
                    "kv_positions": self.kv_positions[layer_idx].detach().cpu(),
                }
        else:
            logs["seen_tokens"] = self.get_seq_length()
            logs["kv_positions"] = self.kv_positions[layer_idx].detach().cpu()
        return logs

    def copy_to_device(self, device: str):
        new_cache = TrimKVCache(
            memory_size=self.memory_size,
            buffer_size=self.buffer_size,
            sliding_window_size=self.sliding_window_size,
        )

        new_cache._seen_tokens = self._seen_tokens
        new_cache.offset = self.offset.to(device)
        new_cache.block_mask = self.block_mask.to(device) if self.block_mask is not None else None
        new_cache.key_cache = [tensor.to(device) for tensor in self.key_cache]
        new_cache.value_cache = [tensor.to(device) for tensor in self.value_cache]
        new_cache.retention_weights = [tensor.to(device) for tensor in self.retention_weights]
        new_cache.kv_positions = [tensor.to(device) for tensor in self.kv_positions]
        return new_cache


class DynamicBudgetTrimKVCache(TrimKVCache):
    def __init__(
        self,
        max_seq_len: int = None,
        memory_size: Optional[int] = None,
        alpha_threshold: float = None,
        buffer_size: int = 1,
        sliding_window_size: int = 0,
        min_tokens_per_head: int = 0,
        strategy: str = 'fixed_budget',
        lookahead_steps: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self.memory_size = memory_size
        self.buffer_size = buffer_size
        self.max_seq_len = max_seq_len
        self.sliding_window_size = sliding_window_size
        self.min_tokens_per_head = min_tokens_per_head # the minimum number of tokens to keep for each head, used to prevent some heads from being completely wiped out when the cache is very large and all tokens have small alpha values
        self.alpha_threshold = alpha_threshold # use with "threshold" strategy, keys with alpha value smaller than the threshold will be removed
        self.strategy = strategy
        assert strategy in ['fixed_budget', 'threshold'], "Only 'fix_budget' and 'threshold' strategies are supported"
        self.do_compress = (self.strategy == "threshold" and self.alpha_threshold is not None) or (self.strategy == "fixed_budget" and self.memory_size is not None)

        self.lookahead_steps = lookahead_steps

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.retention_weights: List[torch.Tensor] = []
        self.kv_positions: List[torch.Tensor] = []
        self.attention_mask = None

        # Running parameters for update_flatten_view and flash_attn_varlen_func
        self.head_lens: List[torch.Tensor] = []
        self.cu_seqlens_k: List[torch.Tensor] = []

        self.device = device
        self.offset = torch.tensor(0, dtype=torch.int64)
        self.block_mask = None
        
        self._seen_tokens = 0
        self.peak_cached_tokens = None
        self.layers = []

    def get_total_cached_tokens(self) -> int:
        return sum(self.key_cache[layer_idx].shape[0] for layer_idx in range(len(self)))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        retention_weights = cache_kwargs.get("retention_weights")
        attention_mask = cache_kwargs.get("attention_mask", None)
        # Update the cache
        if key_states is not None:
            # Update the number of seen tokens
            bz, num_heads, seq_len, dim = key_states.shape

            if layer_idx == 0:
                self._seen_tokens += seq_len

            cache_positions = cache_position[None, None, :].expand_as(retention_weights) if cache_position.dim() == 1 else cache_position

            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    self.retention_weights.append(torch.tensor([]))
                    self.kv_positions.append(torch.tensor([]))

                key_states = key_states.contiguous().view(-1, dim) # (B*H*S,D)
                value_states = value_states.contiguous().view(-1, dim) # (B*H*S,D)

                # use (-1, 1) just to be compatible with update_flatten_view
                cache_positions = cache_positions.contiguous().view(-1, 1) # (B*H*S, 1)
                retention_weights = retention_weights.contiguous().view(-1, 1) # (B*H*S, 1)

                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.retention_weights.append(retention_weights)
                self.kv_positions.append(cache_positions)
                # Update head_lens and cu_seqlens_k
                self.head_lens.append(torch.tensor([seq_len] * num_heads, device=self.device, dtype=torch.int32))
                self.cu_seqlens_k.append(torch.arange(0, (seq_len * num_heads) + 1, step=seq_len, device=self.device, dtype=torch.int32))
            else:
                self.key_cache[layer_idx] = update_flatten_view_triton(
                    self.key_cache[layer_idx], key_states.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )
                self.value_cache[layer_idx] = update_flatten_view_triton(
                    self.value_cache[layer_idx], value_states.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )

                retention_weights = retention_weights.unsqueeze(-1)
                cache_positions = cache_positions.unsqueeze(-1)
                self.retention_weights[layer_idx] = update_flatten_view_triton(
                    self.retention_weights[layer_idx], retention_weights.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )
                self.kv_positions[layer_idx] = update_flatten_view_triton(
                    self.kv_positions[layer_idx], cache_positions.contiguous(), self.head_lens[layer_idx], self.cu_seqlens_k[layer_idx]
                )

                # Update head_lens and cu_seqlens_k
                self.head_lens[layer_idx] = self.head_lens[layer_idx] + seq_len
                cu_offset = torch.arange(0, (num_heads * seq_len) + 1, step=seq_len, device=self.device, dtype=torch.int32)
                self.cu_seqlens_k[layer_idx] = self.cu_seqlens_k[layer_idx] + cu_offset

        flash_attn_kwargs = {
            "head_lens": self.head_lens[layer_idx],
            "cu_seqlens_k": self.cu_seqlens_k[layer_idx],
        }

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.retention_weights[layer_idx], self.kv_positions[layer_idx], flash_attn_kwargs

    def get_cache_length(self, layer_idx: int = 0, head_idx: int = 0) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.head_lens[layer_idx][head_idx].item()

    @torch.inference_mode()
    def compress(self):
        device = self.device
        assert len(self.key_cache) > 0 and self.key_cache[0].numel() > 0, "Cache is empty, cannot compress"
        num_layers = len(self.key_cache)
        num_key_value_heads = self.head_lens[0].shape[0] if self.head_lens else 0

        if not self.do_compress:
            return

        total_memory_size = num_layers * num_key_value_heads * self.memory_size
        if self.strategy == "fixed_budget" and num_layers * num_key_value_heads * (self.memory_size + self.buffer_size) > self.get_total_cached_tokens():
            # if the total number of tokens in the cache is smaller than the budget, do not compress
            return

        # compute the scores for all tokens in the cache across all layers and heads, and select the top-k tokens with highest scores to keep in the cache
        rw = torch.cat([self.retention_weights[l] for l in range(num_layers)], dim=0).squeeze(-1)  # (T,)
        kv_pos = torch.cat([self.kv_positions[l] for l in range(num_layers)], dim=0).squeeze(-1)  # (T,)
        layer_lens = torch.tensor([self.retention_weights[l].shape[0] for l in range(num_layers)])
        cu_layer_lens = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), layer_lens], dim=0), dim=0)

        q_idx = self.get_seq_length() + 1
        if self.lookahead_steps == 1:
            scores = (q_idx - kv_pos) * rw
        else:
            scores = compute_log_G(
                rw.to(torch.float32),
                q_idx,
                kv_pos,
                n=self.lookahead_steps,
            )

        if self.min_tokens_per_head == 0:
            if self.strategy == "fixed_budget":
                topk_idx = torch.topk(scores, total_memory_size, largest=True, sorted=False).indices  # (K,)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.index_fill_(0, topk_idx, True)
            else: # thresholding strategy
                topk_mask = scores >= torch.log(torch.tensor(self.alpha_threshold, device=device))

            # compute budget for each head and layer
            for l in range(num_layers):
                start = cu_layer_lens[l].item()
                end = cu_layer_lens[l + 1].item()
                layer_mask = topk_mask[start:end]  # (S_l,)
                self.key_cache[l] = self.key_cache[l][layer_mask, ...]
                self.value_cache[l] = self.value_cache[l][layer_mask, ...]
                self.retention_weights[l] = self.retention_weights[l][layer_mask, ...]
                self.kv_positions[l] = self.kv_positions[l][layer_mask, ...]

                self.head_lens[l] = torch.tensor(
                    [layer_mask[self.cu_seqlens_k[l][h]:self.cu_seqlens_k[l][h+1]].sum() for h in range(num_key_value_heads)],
                    device=device,
                    dtype=torch.int32,
                )

                self.cu_seqlens_k[l] = torch.cumsum(
                    torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[l]], dim=0),
                    dim=0,
                    dtype=torch.int32,
                )
        else:
            head_lens = torch.cat([self.head_lens[l] for l in range(num_layers)], dim=0) # (L*H,)
            cu_head_lens = torch.cumsum(head_lens, dim=0)

            assert self.strategy == "fixed_budget", "Current implementation of DynamicBudgetTrimKVCache with min_tokens_per_head > 0 is not supported for thresholding strategy"

            adaptive_memory_size = int(total_memory_size - num_layers * num_key_value_heads * self.min_tokens_per_head)
            topk_idx = torch.topk(scores, adaptive_memory_size, largest=True, sorted=False).indices  # (K,)

            # use bucketize to compute the number of keys selected in each head
            cu_head_lens = torch.cumsum(head_lens, dim=0)
            lh_idx = torch.bucketize(topk_idx, cu_head_lens)
            head_cnt = torch.zeros_like(head_lens, dtype=torch.long)
            head_cnt.index_add_(0, lh_idx, torch.ones_like(topk_idx))
            head_cnt = head_cnt + self.min_tokens_per_head
            # reshape head_cnt to (L, H)
            head_cnt = head_cnt.view(num_layers, num_key_value_heads)
            # compute topk mask for each layer, head
            for l in range(num_layers):
                start = cu_layer_lens[l].item()
                end = cu_layer_lens[l + 1].item()
                layer_scores = scores[start:end]
                layer_topk_mask = []
                for h in range(num_key_value_heads):
                    head_start = self.cu_seqlens_k[l][h].item()
                    head_end = self.cu_seqlens_k[l][h + 1].item()
                    head_scores = layer_scores[head_start:head_end]
                    k = head_cnt[l, h].item()

                    if k >= head_scores.shape[0]:
                        layer_topk_mask.append(torch.ones_like(head_scores, dtype=torch.bool))
                        continue

                    head_topk_idx = torch.topk(head_scores, k, largest=True, sorted=False).indices
                    head_topk_mask = torch.zeros_like(head_scores, dtype=torch.bool)
                    head_topk_mask.index_fill_(0, head_topk_idx, True)
                    layer_topk_mask.append(head_topk_mask)
                layer_topk_mask = torch.cat(layer_topk_mask, dim=0)
                self.key_cache[l] = self.key_cache[l][layer_topk_mask, ...]
                self.value_cache[l] = self.value_cache[l][layer_topk_mask, ...]
                self.retention_weights[l] = self.retention_weights[l][layer_topk_mask, ...]
                self.kv_positions[l] = self.kv_positions[l][layer_topk_mask, ...]

                self.head_lens[l] = torch.tensor(
                    [layer_topk_mask[self.cu_seqlens_k[l][h]:self.cu_seqlens_k[l][h+1]].sum() for h in range(num_key_value_heads)],
                    device=device,
                    dtype=torch.int32,
                )

                self.cu_seqlens_k[l] = torch.cumsum(
                    torch.cat([torch.zeros(1, device=device, dtype=torch.int32), self.head_lens[l]], dim=0),
                    dim=0,
                    dtype=torch.int32,
                )

    def log(self):
        head_wise_kv_positions = []
        for l in range(self.num_layers):
            for h in range(self.num_key_value_heads):
                head_start = self.cu_seqlens_k[l][h].item()
                head_end = self.cu_seqlens_k[l][h + 1].item()
                head_kv_positions = self.kv_positions[l][head_start:head_end].detach().cpu()
                head_wise_kv_positions.append(head_kv_positions)
            
        logs = {
            "head_wise_kv_positions": head_wise_kv_positions, # list of token positions cached for each head
            "flat_head_lens": torch.cat(self.head_lens).to('cpu').numpy(), # flattened head lens for all layers * heads
        }
        return logs

    def copy_to_device(self, device):
        raise NotImplementedError("copy_to_device not implemented for DynamicBudgetTrimKVCache")


class PagedCache:
    def __init__(self, batch_size: int, num_layers: int, num_heads: int, max_blocks_per_head: int, head_dim: int, block_size: int, num_blocks_ratio: float = 1.0, device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_blocks_per_head = max_blocks_per_head
        self.num_blocks = int(num_layers * num_heads * max_blocks_per_head * batch_size * num_blocks_ratio)

        self.block_table = torch.full(
            (num_layers, batch_size, num_heads, max_blocks_per_head),
            fill_value=self.num_blocks,
            dtype=torch.int32,
            device=device,
        )
        self.cache_seqlens = torch.zeros(
            (num_layers, batch_size, num_heads),
            dtype=torch.int32,
            device=device,
        )

        # Allocate one extra null-block (index = self.num_blocks = sentinel) so that any
        # flash_attn access to the sentinel block ID reads safe zeros instead of OOB memory.
        self.key = torch.zeros((self.num_blocks + 1) * block_size, head_dim, dtype=dtype, device=device)
        self.value = torch.zeros((self.num_blocks + 1) * block_size, head_dim, dtype=dtype, device=device)
        self.pos = torch.zeros((self.num_blocks + 1) * block_size, 1, dtype=torch.int64, device=device)
        self.retention = torch.zeros((self.num_blocks + 1) * block_size, 1, dtype=dtype, device=device)

        self.device = device

        self.free_block_indices = torch.arange(
            self.num_blocks - 1, -1, -1,
            dtype=torch.int32,
            device=device,
        )
        self.num_free_blocks = self.num_blocks
        self.block_arange = torch.arange(self.block_size, dtype=torch.int32, device=device)
        self.layer_arange = torch.arange(self.num_layers, dtype=torch.int32, device=device)
        self.head_arange = torch.arange(self.num_heads, dtype=torch.int32, device=device)

    def reset(self):
        self.block_table.fill_(self.num_blocks)
        self.cache_seqlens.zero_()
        self.num_free_blocks = self.num_blocks
        self.free_block_indices = torch.arange(
            self.num_blocks - 1, -1, -1,
            dtype=torch.int32,
            device=self.device,
        )

    def get_free_blocks(self, num_blocks: int) -> torch.Tensor:
        if num_blocks > self.num_free_blocks:
            additional_blocks = max(self.num_blocks, num_blocks - self.num_free_blocks)
            print(f"Expanding physical cache from {self.num_blocks} blocks to {self.num_blocks + additional_blocks} blocks to accommodate {num_blocks} requested blocks (only {self.num_free_blocks} free)")
            self._expand_physical_cache(additional_blocks)
            print(f"Expanded physical cache: now have {self.num_blocks} blocks, {self.num_free_blocks} free blocks")
       
        free_blocks = self.free_block_indices[self.num_free_blocks - num_blocks:self.num_free_blocks]
        self.num_free_blocks -= num_blocks
        return free_blocks

    def release_blocks(self, block_indices: torch.Tensor) -> None:
        num_blocks = block_indices.shape[0]
        self.free_block_indices[self.num_free_blocks:self.num_free_blocks + num_blocks] = block_indices
        self.num_free_blocks += num_blocks

    def _ensure_token_arange(self, seq_len: int) -> torch.Tensor:
        if not hasattr(self, "_token_arange") or self._token_arange.numel() < seq_len:
            self._token_arange = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        return self._token_arange[:seq_len]

    def get_items(self, global_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = self.key[global_indices]
        values = self.value[global_indices]
        retention_weights = self.retention[global_indices]
        kv_positions = self.pos[global_indices]
        return keys, values, retention_weights, kv_positions

    def _ensure_block_capacity(self, total_blocks: torch.Tensor, maxB: int, layer_idx: int, batch_slice: slice, block_table: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if torch.any(total_blocks > maxB):
            needed_max = int(total_blocks.max().item())
            self._expand_block_table(needed_max)
            maxB = self.max_blocks_per_head
            block_table = self.block_table[layer_idx, batch_slice]
        return block_table, maxB

    def _expand_physical_cache(self, additional_blocks: int):
        old_num_blocks = self.num_blocks
        new_num_blocks = old_num_blocks + additional_blocks
        old_size = old_num_blocks * self.block_size
        new_size = new_num_blocks * self.block_size

        def expand_tensor(t):
            # Allocate new_size + block_size to keep the extra null sentinel block at the end.
            new_t = torch.zeros((new_size + self.block_size, *t.shape[1:]), dtype=t.dtype, device=t.device)
            new_t[:old_size] = t[:old_size]  # copy valid data only (exclude old sentinel block)
            return new_t

        self.key = expand_tensor(self.key)
        self.value = expand_tensor(self.value)
        self.pos = expand_tensor(self.pos)
        self.retention = expand_tensor(self.retention)

        new_free_indices = torch.arange(
            new_num_blocks - 1, old_num_blocks - 1, -1, 
            dtype=torch.int32, device=self.device
        )
        self.free_block_indices = torch.cat([new_free_indices, self.free_block_indices])
        self.block_table[self.block_table == old_num_blocks] = new_num_blocks

        self.num_blocks = new_num_blocks
        self.num_free_blocks += additional_blocks

    def _expand_block_table(self, needed_max_blocks: int):
        pad_size = needed_max_blocks - self.max_blocks_per_head
        padding = torch.full(
            (self.num_layers, self.batch_size, self.num_heads, pad_size),
            fill_value=self.num_blocks,
            dtype=self.block_table.dtype,
            device=self.device
        )
        self.block_table = torch.cat([self.block_table, padding], dim=-1)
        self.max_blocks_per_head = needed_max_blocks

    def add(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        retention_weights: torch.Tensor,
        kv_positions: torch.Tensor,
        layer_idx: int,
        batch_idx: int = None,
    ) -> None:
        bsz, num_heads, seq_len, dim = key_states.shape
        assert num_heads == self.num_heads
        assert dim == self.head_dim

        if batch_idx is not None:
            assert bsz == 1, "When batch_idx is set, pass tensors with bsz==1 (the selected item)."
            batch_slice = slice(batch_idx, batch_idx + 1)
            B = 1
        else:
            batch_slice = slice(0, bsz)
            B = bsz

        H = num_heads
        BS = self.block_size
        maxB = self.max_blocks_per_head
        device = self.device

        key_flat = rearrange(key_states,        "b h s d -> (b h s) d")
        val_flat = rearrange(value_states,      "b h s d -> (b h s) d")
        rw_flat  = rearrange(retention_weights, "b h s   -> (b h s) 1")
        pos_flat = rearrange(kv_positions,      "b h s   -> (b h s) 1")

        block_table = self.block_table[layer_idx, batch_slice]
        s0 = self.cache_seqlens[layer_idx, batch_slice].to(torch.int64)

        e0 = s0 + seq_len

        existing_blocks = (s0 + (BS - 1)) // BS
        total_blocks    = (e0 + (BS - 1)) // BS
        new_blocks      = total_blocks - existing_blocks

        block_table, maxB = self._ensure_block_capacity(total_blocks, maxB, layer_idx, batch_slice, block_table)

        counts = new_blocks.reshape(-1).to(torch.int64)
        total_new = int(counts.sum().item())

        if total_new > 0:
            free = self.get_free_blocks(total_new).to(torch.int64)

            BH = counts.numel()
            bh = torch.arange(BH, device=device, dtype=torch.int64)
            bh_rep = torch.repeat_interleave(bh, counts)

            start = torch.cumsum(counts, 0) - counts
            offset = torch.arange(total_new, device=device, dtype=torch.int64) - start[bh_rep]
            slot = existing_blocks.reshape(-1).to(torch.int64)[bh_rep] + offset

            b_rep = bh_rep // H
            h_rep = bh_rep % H

            block_table[b_rep, h_rep, slot] = free.to(torch.int32)

        self.cache_seqlens[layer_idx, batch_slice] = e0.to(torch.int32)

        t = self._ensure_token_arange(seq_len)
        logical = s0[..., None] + t[None, None, :]
        block_slots = (logical // BS).to(torch.int64)
        in_block = (logical % BS).to(torch.int64)

        block_ids = torch.gather(block_table.to(torch.int64), dim=-1, index=block_slots)
        phys = block_ids * BS + in_block
        idx = phys.reshape(-1)

        self.key[idx] = key_flat
        self.value[idx] = val_flat
        self.retention[idx] = rw_flat
        self.pos[idx] = pos_flat

        
        
class PagedTrimKVCache(TrimKVCache):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_seq_len: int | str, # will be deprecated in future
        memory_size: Optional[int] = None,
        alpha_threshold: float = None,
        buffer_size: int = 1,
        min_tokens_per_head: int = 0,
        sliding_window_size: int = 0,
        strategy: str = 'fixed_budget',
        lookahead_steps: int = 1,
        block_size: int = 256,
        num_blocks_ratio: float = 1.0,
        device: str = "cuda",
    ) -> None:
        # For paged attention cache
        self.num_layers = num_layers
        self.num_key_value_heads = num_heads
        self.memory_size = memory_size
        self.alpha_threshold = alpha_threshold
        self.buffer_size = buffer_size
        self.sliding_window_size = sliding_window_size
        self.min_tokens_per_head = min_tokens_per_head
        self.strategy = strategy

        self.max_seq_len = 128 # an initial value, the actual max_seq_len will be determined by the number of blocks allocated in the paged cache, which can grow dynamically as needed
        print("Using auto-paged cache with dynamic block allocation, starting with max_seq_len = {}".format(self.max_seq_len))

        # self.max_seq_len = max_seq_len
        self.device = device
        assert strategy in ['fixed_budget', 'threshold'], "Only 'fix_budget' and 'threshold' strategies are supported"
        self.do_compress = (self.strategy == "threshold" and self.alpha_threshold is not None) or (self.strategy == "fixed_budget" and self.memory_size is not None)

        self.lookahead_steps = lookahead_steps

        self.block_size = block_size
        self.num_blocks_ratio = num_blocks_ratio
        self.max_blocks_per_head = math.ceil(self.max_seq_len / block_size) + 1 # add one extra block for safety, the actual number of blocks will be determined by num_blocks_ratio
        self.paged_cache: Optional[PagedCache] = None

        self.peak_cached_tokens = None
        self._seen_tokens = 0
        self.layers = []
        
    def initialize_paged_cache(self, key_states):
        dtype = key_states.dtype
        device = key_states.device
        bsz, _, _, dim = key_states.shape


        self.paged_cache = PagedCache(
            batch_size=bsz,
            num_layers=self.num_layers,
            num_heads=self.num_key_value_heads,
            max_blocks_per_head=self.max_blocks_per_head,
            head_dim=dim,
            block_size=self.block_size,
            num_blocks_ratio=self.num_blocks_ratio,
            device=device,
            dtype=dtype,
        )
        

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        query_length = cache_position.shape[0]
        cached_length = self.get_cache_length()
        kv_offset = max(self._seen_tokens - cached_length + 1, 0)
        kv_length = cached_length + query_length

        return kv_length, kv_offset

    def get_cache_length(self) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if self.paged_cache is None:
            return 0
        return self.paged_cache.cache_seqlens.max().item()

    def get_total_cached_tokens(self) -> int:
        if self.paged_cache is None:
            raise RuntimeError("Paged cache is not initialized yet.")
        return self.paged_cache.cache_seqlens.sum(dim=(0,2))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        retention_weights = cache_kwargs.get("retention_weights")
        attention_mask = cache_kwargs.get("attention_mask", None)
        if self.paged_cache is None:
            self.initialize_paged_cache(key_states)

        # Update the number of seen tokens
        bz, num_heads, seq_len, dim = key_states.shape

        if layer_idx == 0:
            self._seen_tokens += seq_len

        cache_positions = cache_position[None, None, :].expand_as(retention_weights) if cache_position.dim() == 1 else cache_position
        if attention_mask is not None and self.paged_cache.cache_seqlens[layer_idx].sum() == 0:
            # If there is an attention mask and it's the prefill step, we need to process each sample in the batch separately due to different valid lengths
            for i in range(bz):
                num_new_tokens = attention_mask[i].sum().item()

                self.paged_cache.add(
                    key_states[i:i+1, :, -num_new_tokens:, :],
                    value_states[i:i+1, :, -num_new_tokens:, :],
                    retention_weights[i:i+1, :, -num_new_tokens:],
                    cache_positions[i:i+1, :, -num_new_tokens:],
                    layer_idx,
                    batch_idx=i,
                )
        else:
            self.paged_cache.add(
                key_states,
                value_states,
                retention_weights,
                cache_positions,
                layer_idx,
            )

        flash_attn_kwargs = {
            "cache_seqlens": self.paged_cache.cache_seqlens[layer_idx],
            "block_size": self.block_size,
            "block_table": self.paged_cache.block_table[layer_idx],
        }
        
        num_cached_tokens = self.paged_cache.cache_seqlens.sum(axis=(0,2)) # this is torch tensor of shape [batch_size]
        if self.peak_cached_tokens is None:
            self.peak_cached_tokens = num_cached_tokens
        else:
            self.peak_cached_tokens = torch.max(self.peak_cached_tokens, num_cached_tokens)

        return self.paged_cache.key, self.paged_cache.value, self.paged_cache.retention, self.paged_cache.pos, flash_attn_kwargs

    @torch.inference_mode()
    def compress(self):
        device = self.device
        batch_size = self.paged_cache.cache_seqlens.shape[1]
        BS = self.block_size
        L = self.num_layers
        H = self.num_key_value_heads
        # MB = self.max_blocks_per_head
        MB = self.paged_cache.block_table.shape[-1]
        sentinel = int(self.paged_cache.num_blocks)

        if not self.do_compress:
            return

        for b in range(batch_size):
            block_table = self.paged_cache.block_table[:, b, :, :].contiguous()
            seqlens = self.paged_cache.cache_seqlens[:, b, :]
            total_cached_tokens = seqlens.sum().item()

            if self.strategy == "fixed_budget" and self.num_layers * self.num_key_value_heads * (self.memory_size + self.buffer_size) > total_cached_tokens:
               continue
        
            if self.strategy == "threshold" and self._seen_tokens % self.buffer_size != 0:
                # only compress when the number of seen tokens reaches the buffer size, to avoid too frequent compressions which can be costly
                continue

            lens = seqlens.reshape(-1)  # [L*H]
            offsets = torch.empty((L * H + 1,), device=device, dtype=torch.int64)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(lens, dim=0)

            N = int(lens.sum().item())
            stream_id = torch.repeat_interleave(
                torch.arange(L * H, device=device),
                lens
            )  # [N]
            # starting offset of each stream in the packed array
            starts = torch.cumsum(lens, dim=0) - lens          # [BH]
            pos = torch.arange(N, device=device) - starts[stream_id]  # [N]

            block_slot = (pos // BS)
            in_block   = (pos %  BS)

            # print(block_slot[:1000], in_block[:1000], layer_idx[:1000], head_idx[:1000])
            bt = block_table.reshape(-1)
            bt_index = stream_id * MB + block_slot
            block_id = bt[bt_index]
            # physical global index into paged arrays
            global_idx = block_id * BS + in_block  # [N]

            key, value, ret, kv_pos = self.paged_cache.get_items(global_idx)  # [N,D], [N,D], [N,1], [N,1]

            q_idx = self.get_seq_length() + 1
            if self.lookahead_steps == 1:
                scores =  ret.to(torch.float32) * (q_idx - kv_pos)
            else:
                scores = compute_log_G(
                    ret.to(torch.float32),
                    q_idx,
                    kv_pos,
                    n=self.lookahead_steps,
                )

            scores = scores.squeeze(-1)  # (N,)

            if self.min_tokens_per_head == 0:
                if self.strategy == "fixed_budget":
                    total_memory_size = L * H * self.memory_size
                    k = min(int(total_memory_size), N)
                    keep_indices = torch.topk(scores, k, largest=True, sorted=False).indices  # (K,)
                    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
                    keep_mask.index_fill_(0, keep_indices, True)
                else: # thresholding strategy
                    keep_mask = scores >= torch.log(torch.tensor(self.alpha_threshold, device=device))
                    # keep_mask = scores >= -0.02
                    keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
                    k = keep_mask.sum().item()

                # compute budget for each head and layer
                src_stream = stream_id[keep_mask]
                new_lens = torch.bincount(src_stream, minlength=L * H)
            else:
                if self.strategy == "fixed_budget":
                    adaptive_memory_size = int(L * H * (self.memory_size - self.min_tokens_per_head))
                    topk_indices = torch.topk(scores, adaptive_memory_size, largest=True, sorted=False).indices  # (K,)
                    adaptive_keep_mask = torch.zeros_like(scores, dtype=torch.bool)
                    adaptive_keep_mask.index_fill_(0, topk_indices, True)

                    adaptive_src_stream = stream_id[adaptive_keep_mask]
                    head_lens = torch.bincount(adaptive_src_stream, minlength=L * H)
                    new_lens = head_lens + self.min_tokens_per_head
                    # build final keep mask
                    keep_mask = torch.zeros((N,), device=device, dtype=torch.bool)
                else:
                    keep_mask = scores >= torch.log(torch.tensor(self.alpha_threshold, device=device))
                    src_stream = stream_id[keep_mask]
                    new_lens = torch.bincount(src_stream, minlength=L * H)
                    new_lens = torch.clamp(new_lens, min=self.min_tokens_per_head)
                    # restart the keep_mask to ensure at least min_tokens_per_head are kept for each head
                    keep_mask = torch.zeros((N,), device=device, dtype=torch.bool)

                for lh in range(L * H):
                    head_start = starts[lh].item()
                    head_end = starts[lh].item() + lens[lh].item()
                    head_scores = scores[head_start:head_end]
                    true_len = new_lens[lh].item()

                    if true_len >= head_scores.shape[0]:
                        keep_mask[head_start:head_end] = True
                        true_len = head_scores.shape[0]
                    else:
                        head_topk_idx = torch.topk(head_scores, true_len, largest=True, sorted=False).indices
                        keep_mask[head_start:head_end][head_topk_idx] = True

                    new_lens[lh] = true_len

                src_stream = stream_id[keep_mask]
                k = keep_mask.sum().item()

            # Rebuild block table
            # # --- update block_table by releasing unused tail blocks (keep prefix blocks) ---
            old_blocks = (lens + BS - 1) // BS  # [L * H]
            new_blocks = (new_lens + BS - 1) // BS  # [L * H]
            slots = torch.arange(MB, device=device)[None, :]
            rel_mask = ((slots >= new_blocks[:, None]) & (slots < old_blocks[:, None])).reshape(-1)
            rel_blocks = bt[rel_mask]
            if rel_blocks.numel() > 0:
                self.paged_cache.release_blocks(rel_blocks)
            bt[rel_mask] = sentinel
            # Write back to the original block_table in case block_table[:, b, :, :] was non-contiguous
            # (e.g., batch_size > 1), in which case `bt` is a copy and bt[rel_mask] = sentinel
            # would not propagate to self.paged_cache.block_table automatically.
            self.paged_cache.block_table[:, b, :, :] = block_table
            
            self.paged_cache.cache_seqlens[:, b, :] = new_lens.view(L, H).to(torch.int32)
            # --- compute destination global indices for compacted layout ---
            ar = torch.arange(k, device=device)
            is_start = torch.ones((k,), device=device, dtype=torch.bool)
            is_start[1:] = (src_stream[1:] != src_stream[:-1])
            start_idx = torch.where(is_start, ar, torch.zeros_like(ar))
            start_idx = torch.cummax(start_idx, dim=0).values
            new_pos = ar - start_idx
            dest_block_slot = (new_pos // BS)
            dest_in_block   = (new_pos %  BS)
            dest_block_id = bt[src_stream * MB + dest_block_slot]
            dest_global = dest_block_id * BS + dest_in_block
            self.paged_cache.key[dest_global] = key[keep_mask]
            self.paged_cache.value[dest_global] = value[keep_mask]
            self.paged_cache.pos[dest_global] = kv_pos[keep_mask]
            self.paged_cache.retention[dest_global] = ret[keep_mask]

    
    def log(self):
        num_layers, batch_size, num_heads = self.paged_cache.cache_seqlens.shape
        block_size = self.block_size
        device = self.device
        
        # We need the current query length to calculate "age" and uniform canvas size
        current_seq_len = self.get_seq_length() + 1
        
        head_wise_kv_positions = []
        head_wise_scores = [] 

        for l in range(num_layers):
            for b in range(batch_size):
                for h in range(num_heads):
                    seq_len = self.paged_cache.cache_seqlens[l, b, h].item()
                    
                    if seq_len == 0:
                        # Explicitly place on CPU to match the detach().cpu() logic below
                        head_wise_kv_positions.append(torch.tensor([], dtype=torch.long, device='cpu'))
                        head_wise_scores.append(torch.tensor([], dtype=torch.float32, device='cpu'))
                        continue

                    # 1. Get Blocks
                    num_blocks_needed = (seq_len + block_size - 1) // block_size
                    block_ids = self.paged_cache.block_table[l, b, h, :num_blocks_needed]

                    # 2. Get Global Indices (Cast to long for safe indexing)
                    offsets = torch.arange(block_size, device=device)
                    global_indices = (block_ids.unsqueeze(-1) * block_size + offsets.unsqueeze(0)).view(-1)
                    valid_global_indices = global_indices[:seq_len].long()
                    
                    # 3. Fetch Data
                    pos_data = self.paged_cache.pos[valid_global_indices].squeeze(-1)
                    ret_data = self.paged_cache.retention[valid_global_indices].squeeze(-1)

                    # 4. Calculate Score: Ret * (Current_Time - Token_Time)
                    surviving_scores = ret_data.float() * (current_seq_len - pos_data.float())

                    # 5. Reconstruct Full Score Timeline (Uniform length)
                    full_len = current_seq_len # Replaces max_pos + 1
                    reconstructed_scores = torch.full((full_len,), -1000.0, dtype=torch.float32, device=device)
                    
                    # Scatter surviving scores into their original positions
                    reconstructed_scores[pos_data.long()] = surviving_scores

                    # Store results
                    head_wise_kv_positions.append(pos_data.detach().cpu())
                    head_wise_scores.append(reconstructed_scores.detach().cpu())

        flat_head_lens = self.paged_cache.cache_seqlens.flatten().detach().cpu().numpy()

        logs = {
            "head_wise_kv_positions": head_wise_kv_positions,
            "head_wise_scores": head_wise_scores, 
            "flat_head_lens": flat_head_lens,
        }
        return logs

    def copy_to_device(self, device):
        raise NotImplementedError("copy_to_device not implemented for DynamicBudgetTrimKVCache")


__all__ = [
    "TrimKVCache",
    "PagedTrimKVCache",
]

