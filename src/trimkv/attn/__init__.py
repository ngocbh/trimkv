import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from . import eager_attn
from . import flash_attn

TRIMKV_ATTENTION_IMPLEMENTATIONS = {
    "db_attn_flash": flash_attn.dynamic_kv_budget_attention_forward, # Dynamic-KV-Budget Attention implementation using Flash Attention
    "attn_eager": eager_attn.eager_attention_forward, # Standard Attention implementation using Eager Attention
}


def get_trimkv_wrapper(attn_impl: str):
    attn_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

    def attn_wrapper(*args, **kwargs):
        kwargs.pop("retention_weights", None)
        kwargs.pop("kv_positions", None)
        kwargs.pop("rg_dropout", None)
        kwargs.pop("flash_attn_kwargs", None)
        attn_output, attn_weights = attn_fn(*args, **kwargs)
        return attn_output, attn_weights, None
    
    return attn_wrapper

def get_attention_interface(attn_impl: str, compile=False):
    if attn_impl not in TRIMKV_ATTENTION_IMPLEMENTATIONS:
        attention_inference = get_trimkv_wrapper(attn_impl)
    else:
        attention_inference = TRIMKV_ATTENTION_IMPLEMENTATIONS.get(attn_impl, None)

    if compile:
        attention_inference = torch.compile(attention_inference)

    return attention_inference
