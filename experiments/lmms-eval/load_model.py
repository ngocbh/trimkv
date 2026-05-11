import os
import torch
from functools import partial

from lmms_eval.utils import simple_parse_args_string
from lmms_eval.models import get_model

from trimkv.cache_utils import TrimKVCache, DynamicBudgetTrimKVCache, PagedTrimKVCache
from trimkv.models.qwen3_vl import TrimKVQwen3VLForConditionalGeneration
from trimkv.models.qwen2_5_vl import TrimKVQwen2_5_VLForConditionalGeneration
from trimkv.models.llava import TrimKVLlavaForConditionalGeneration

from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from transformers import LlavaForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from rkv.dynamic_cache import RKVDynamicCache
from rkv.adakv_cache import AdaKVDynamicCache
from rkv.monkeypatch import replace_qwen3vl, update_qwen3vl_compression_config, replace_qwen3vl_adakv


def _compute_qwen_text_tokens(inputs, processor):
    bz = inputs['input_ids'].shape[0]
    assert bz == 1, "Batch size greater than 1 is not supported for Qwen models in _compute_qwen_text_tokens."
    num_non_text_tokens = ((inputs['input_ids'] == processor.video_token_id) 
                           |(inputs['input_ids'] == processor.image_token_id)).sum(axis=1)

    return inputs['input_ids'].shape[1] - num_non_text_tokens.max().item()


def _compute_llava_text_tokens(inputs, processor):
    bz = inputs['input_ids'].shape[0]
    assert bz == 1, "Batch size greater than 1 is not supported for Qwen models in _compute_qwen_text_tokens."
    num_non_text_tokens = (inputs['input_ids'] == processor.image_token_id).sum(axis=1)
    
    return inputs['input_ids'].shape[1] - num_non_text_tokens.max().item()


def update_processor_pixels(processor, data_args):
    # This is to follow https://github.com/QwenLM/Qwen3-VL/blob/599ce6104cab6e111f40a30e786561e41b9731ba/qwen-vl-finetune/qwenvl/data/data_processor.py#L44
    # --- Image Processor ---
    ip = processor.image_processor
    print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    print(f"ip.size: {ip.size}")
    print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def load_trimkv_model(config):
    if config.model_type == "qwen3_vl":
        print("Loading TrimKVQwen3VL model...")
        model_cls = TrimKVQwen3VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
    elif config.model_type == "qwen2_5_vl":
        print("Loading TrimKVQwen2.5VL model...")
        model_cls = TrimKVQwen2_5_VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
    elif config.model_type == "llava_hf":
        print("Loading Llava TrimKV model...")
        model_cls = TrimKVLlavaForConditionalGeneration
        compute_text_tokens_fn = _compute_llava_text_tokens
    else:
        raise ValueError(f"Unsupported model: {config.model_path}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        load_trimkv_weights=True,
        download_from=config.download_from,
        use_cache=True,
        device_map="cuda",
    )

    model.config.text_config.attn_impl = config.attn_implementation
    model.config.text_config._attn_implementation = config.attn_implementation
    print(f"Using TrimKV model with config: {model.config}")

    processor = AutoProcessor.from_pretrained(
        model.config.base_model,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    def prepare_input_for_generation(model, inputs, **kwargs):
        # if fixed_kv_budget is False, the memory size is dynamically adjusted according to input length
        if not config.fixed_kv_budget:
            num_text_tokens = compute_text_tokens_fn(inputs, processor)
            model.config.memory_size = config.kv_budget + num_text_tokens
            model.config.text_config.memory_size = config.kv_budget + num_text_tokens

        past_key_values = TrimKVCache(
            memory_size=model.config.memory_size,
            buffer_size=config.buffer_size,
            device="cuda",
        )
        inputs['past_key_values'] = past_key_values
        return inputs

    return model, processor, prepare_input_for_generation



def load_dbtrimkv_model(config):
    if config.model_type == "qwen3_vl":
        print("Loading TrimKVQwen3VL model...")
        model_cls = TrimKVQwen3VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
    elif config.model_type == "qwen2_5_vl":
        print("Loading TrimKVQwen2.5VL model...")
        model_cls = TrimKVQwen2_5_VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
    elif config.model_type == "llava_hf":
        print("Loading Llava TrimKV model...")
        model_cls = TrimKVLlavaForConditionalGeneration
        compute_text_tokens_fn = _compute_llava_text_tokens
    else:
        raise ValueError(f"Unsupported model: {config.model_path}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        load_trimkv_weights=True,
        download_from=config.download_from,
        use_cache=True,
        device_map="cuda",
    )

    model.config.text_config._attn_implementation = "flash_attention_2"
    print(f"Using DBTrimKV model with config: {model.config}")

    processor = AutoProcessor.from_pretrained(
        model.config.base_model,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    def prepare_input_for_generation(model, inputs, **kwargs):
        # if fixed_kv_budget is False, the memory size is dynamically adjusted according to input length
        if not config.fixed_kv_budget:
            num_text_tokens = compute_text_tokens_fn(inputs, processor)
            memory_size = config.kv_budget + num_text_tokens
        else:
            memory_size = config.kv_budget

        if config.strategy == 'threshold':
            print(f"Using threshold strategy with alpha_threshold: {config.alpha_threshold}")
            past_key_values = PagedTrimKVCache(
                num_layers=model.config.text_config.num_hidden_layers,
                num_heads=model.config.text_config.num_key_value_heads,
                buffer_size=config.buffer_size,
                alpha_threshold=config.alpha_threshold,
                num_blocks_ratio=1.0,
                max_seq_len=32768,
                strategy=config.strategy,
                lookahead_steps=config.lookahead_steps,
            )
        elif config.strategy == 'fixed_budget':
            print(f"Using fixed budget strategy with memory size: {memory_size}")
            past_key_values = PagedTrimKVCache(
                num_layers=model.config.text_config.num_hidden_layers,
                num_heads=model.config.text_config.num_key_value_heads,
                max_seq_len=2048, # deprecated in future, it is already automatically calculated.
                memory_size=memory_size,
                buffer_size=config.buffer_size,
                num_blocks_ratio=0.25,
                strategy=config.strategy,
                lookahead_steps=config.lookahead_steps,
            )
        else:
            raise ValueError(f"Unsupported strategy: {config.strategy}")
        inputs['past_key_values'] = past_key_values

        return inputs
    return model, processor, prepare_input_for_generation

def load_vanilla_model(config):
    print(f"Loading vanilla model from {config.model_path}...")
    if config.model_type == "qwen3_vl":
        print("Loading Qwen3VL model...")
        model_cls = Qwen3VLForConditionalGeneration
    elif config.model_type == "qwen2_5_vl":
        print("Loading Qwen2.5VL model...")
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif config.model_type == "llava_hf":
        print("Loading Llava model...")
        model_cls = LlavaForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {config.model_path}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    def prepare_input_for_generation(model, inputs, **kwargs):
        return inputs

    return model, processor, prepare_input_for_generation


def load_rkv_model(config):
    compression_config = {
        "method": config.method,
        "method_config": {
            "budget": config.kv_budget,
            "window_size": config.window_size,
            "mix_lambda": config.mix_lambda,
            "retain_ratio": config.retain_ratio,
            "retain_direction": config.retain_direction,
            "first_tokens": config.first_tokens,
        },
        "compression": None,
        "update_kv": True
    }
    model_config = {
        "divide_method": config.divide_method,
        "divide_length": config.divide_length,
        "compression_content": config.compression_content,
    }
    # apply monkey patch
    if config.model_type == "qwen3_vl":
        replace_qwen3vl(compression_config)
        model_cls = Qwen3VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
        update_compression_config_fn = update_qwen3vl_compression_config
    else:
        raise ValueError(f"Unsupported model: {config.model_type}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.config.update(model_config)

    model.newline_token_ids = [
        processor.tokenizer.encode("\n")[-1],
        processor.tokenizer.encode(".\n")[-1],
        processor.tokenizer.encode(")\n")[-1],
        processor.tokenizer.encode("\n\n")[-1],
        processor.tokenizer.encode(".\n\n")[-1],
        processor.tokenizer.encode(")\n\n")[-1],
    ]

    model.after_think_token_ids = [
        processor.tokenizer.encode("</think>")[-1],
    ]

    def prepare_input_for_generation(model, inputs, **kwargs):
        # if fixed_kv_budget is False, the memory size is dynamically adjusted according to input length
        if not config.fixed_kv_budget:
            num_text_tokens = compute_text_tokens_fn(inputs, processor)
            budget = config.kv_budget + num_text_tokens
            update_compression_config_fn(model, budget=budget)

        past_key_values = RKVDynamicCache()
        inputs['past_key_values'] = past_key_values

        return inputs

    return model, processor, prepare_input_for_generation

def load_adapyramidkv_model(config):
    compression_config = {
        "method": config.method,
        "method_config": {
            "budget": config.kv_budget,
            "window_size": config.window_size,
            "mix_lambda": config.mix_lambda,
            "retain_ratio": config.retain_ratio,
            "retain_direction": config.retain_direction,
            "first_tokens": config.first_tokens,
            "pyram_mode": True,
        },
        "compression": True,
        "update_kv": True
    }
    model_config = {
        "divide_method": config.divide_method,
        "divide_length": config.divide_length,
        "compression_content": config.compression_content,
    }
    # apply monkey patch
    if config.model_type == "qwen3_vl":
        replace_qwen3vl_adakv(compression_config)
        model_cls = Qwen3VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
        update_compression_config_fn = update_qwen3vl_compression_config
    else:
        raise ValueError(f"Unsupported model: {config.model_type}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.config.update(model_config)

    model.newline_token_ids = [
        processor.tokenizer.encode("\n")[-1],
        processor.tokenizer.encode(".\n")[-1],
        processor.tokenizer.encode(")\n")[-1],
        processor.tokenizer.encode("\n\n")[-1],
        processor.tokenizer.encode(".\n\n")[-1],
        processor.tokenizer.encode(")\n\n")[-1],
    ]

    model.after_think_token_ids = [
        processor.tokenizer.encode("</think>")[-1],
    ]

    def prepare_input_for_generation(model, inputs, **kwargs):
        # if fixed_kv_budget is False, the memory size is dynamically adjusted according to input length
        if not config.fixed_kv_budget:
            num_text_tokens = compute_text_tokens_fn(inputs, processor)
            budget = config.kv_budget + num_text_tokens
            update_compression_config_fn(model, budget=budget)

        past_key_values = AdaKVDynamicCache()
        inputs['past_key_values'] = past_key_values

        return inputs

    return model, processor, prepare_input_for_generation


def load_adakv_model(config):
    compression_config = {
        "method": config.method,
        "method_config": {
            "budget": config.kv_budget,
            "window_size": config.window_size,
            "mix_lambda": config.mix_lambda,
            "retain_ratio": config.retain_ratio,
            "retain_direction": config.retain_direction,
            "first_tokens": config.first_tokens,
        },
        "compression": True,
        "update_kv": True
    }
    model_config = {
        "divide_method": config.divide_method,
        "divide_length": config.divide_length,
        "compression_content": config.compression_content,
    }
    # apply monkey patch
    if config.model_type == "qwen3_vl":
        replace_qwen3vl_adakv(compression_config)
        model_cls = Qwen3VLForConditionalGeneration
        compute_text_tokens_fn = _compute_qwen_text_tokens
        update_compression_config_fn = update_qwen3vl_compression_config
    else:
        raise ValueError(f"Unsupported model: {config.model_type}")

    model = model_cls.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=config.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        config.model_path,
        padding_side="left",
    )
    if 'qwen' in config.model_type:
        update_processor_pixels(processor, config)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model.config.update(model_config)

    model.newline_token_ids = [
        processor.tokenizer.encode("\n")[-1],
        processor.tokenizer.encode(".\n")[-1],
        processor.tokenizer.encode(")\n")[-1],
        processor.tokenizer.encode("\n\n")[-1],
        processor.tokenizer.encode(".\n\n")[-1],
        processor.tokenizer.encode(")\n\n")[-1],
    ]

    model.after_think_token_ids = [
        processor.tokenizer.encode("</think>")[-1],
    ]

    def prepare_input_for_generation(model, inputs, **kwargs):
        # if fixed_kv_budget is False, the memory size is dynamically adjusted according to input length
        if not config.fixed_kv_budget:
            num_text_tokens = compute_text_tokens_fn(inputs, processor)
            budget = config.kv_budget + num_text_tokens
            update_compression_config_fn(model, budget=budget)

        past_key_values = AdaKVDynamicCache()
        inputs['past_key_values'] = past_key_values

        return inputs

    return model, processor, prepare_input_for_generation


LOADER_MAP = {
    "vanilla": load_vanilla_model,
    "trimkv": load_trimkv_model,
    "dbtrimkv": load_dbtrimkv_model,
    "dbtrimkv_threshold": load_dbtrimkv_model,
    "snapkv": load_rkv_model,
    "rkv": load_rkv_model,
    "adakv": load_adakv_model,
    "adapyramidkv": load_adapyramidkv_model,
}



def load_model(config):
    load_model_fn = LOADER_MAP[config.method]

    model_args = simple_parse_args_string(config.model_args)
    model_args['batch_size'] = config.batch_size if config.method not in ['adakv', 'adapyramidkv'] else 1  #  AdaKV does not support batch_size > 1
    model_args['load_model_fn'] = partial(load_model_fn, config=config)

    lmms_model_cls = get_model(config.model_type)
    model = lmms_model_cls(**model_args)
    return model
