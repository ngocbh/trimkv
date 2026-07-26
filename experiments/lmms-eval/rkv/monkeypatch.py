from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_vl import modeling_qwen3_vl
from .modeling import (
    Qwen3VLTextAttention_init,
    Qwen3VLTextAttention_forward,
    Qwen3VLForConditionalGeneration_forward,
    Qwen3VLTextAttention_adakv_forward,
    Qwen3VLTextAttention_cake_forward,
    Qwen3VLTextAttention_cake_headkv_forward,
)


def replace_qwen3vl(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3VLTextAttention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_vl.Qwen3VLTextAttention.__init__ = init_wrapper
    modeling_qwen3_vl.Qwen3VLTextAttention.forward = Qwen3VLTextAttention_forward
    modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = Qwen3VLForConditionalGeneration_forward


def replace_qwen3vl_adakv(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3VLTextAttention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_vl.Qwen3VLTextAttention.__init__ = init_wrapper
    modeling_qwen3_vl.Qwen3VLTextAttention.forward = Qwen3VLTextAttention_adakv_forward
    modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = Qwen3VLForConditionalGeneration_forward


def replace_qwen3vl_cake(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3VLTextAttention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_vl.Qwen3VLTextAttention.__init__ = init_wrapper
    modeling_qwen3_vl.Qwen3VLTextAttention.forward = Qwen3VLTextAttention_cake_forward
    modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = Qwen3VLForConditionalGeneration_forward


def replace_qwen3vl_cake_headkv(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3VLTextAttention_init(self, config, layer_idx, compression_config)

    modeling_qwen3_vl.Qwen3VLTextAttention.__init__ = init_wrapper
    modeling_qwen3_vl.Qwen3VLTextAttention.forward = Qwen3VLTextAttention_cake_headkv_forward
    modeling_qwen3_vl.Qwen3VLForConditionalGeneration.forward = Qwen3VLForConditionalGeneration_forward

def update_qwen3vl_compression_config(model, **compression_config):
    for layer in model.model.language_model.layers:
        layer.self_attn.kv_cluster.update_compression_config(**compression_config)

