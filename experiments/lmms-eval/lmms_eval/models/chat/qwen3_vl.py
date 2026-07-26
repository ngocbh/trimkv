import gc
import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.tail_repeat import TailRepeatCriteria, TailRepeatHashCriteria
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")



@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, group_fn=lambda x: x[2], grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        e2e_latency = 0
        total_tokens = 0
        is_visualization = any(request.visualization for request in requests)
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]

            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            gen_kwargs = all_gen_kwargs[0]

            batched_messages = [chat_message.to_hf_messages() for chat_message in chat_messages]

            inputs = self.processor.apply_chat_template(
                batched_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                return_tensors="pt",
            )
            
            # [MODIFIED] Use the provided prepare_inputs_for_generation function
            inputs = self.prepare_inputs_for_generation_fn(self.model, inputs)
            # [END MODIFIED]

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            stoppers = StoppingCriteriaList([
                # TailRepeatCriteria(repeats=3, pmax=16, eos_token_id=self.tokenizer.eos_token_id),
                TailRepeatHashCriteria(repeats=3, pmax=128, eos_token_id=self.tokenizer.eos_token_id),
            ])

            start_time = time.time()

            if not is_visualization:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    top_k=current_gen_kwargs.get("top_k", None),
                    use_cache=self.use_cache,
                    stopping_criteria=stoppers,)
            else:
                cache_dir = f"logs/visualization_b256/{task[0]}_{split[0]}_{doc_id[0]}"
                cont = generate_and_log_cache_info(self.model, 
                    **inputs, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                    pad_token_id=pad_token_id, 
                    do_sample=current_gen_kwargs["do_sample"], 
                    temperature=current_gen_kwargs["temperature"], 
                    top_p=current_gen_kwargs["top_p"], 
                    num_beams=current_gen_kwargs["num_beams"], 
                    max_new_tokens=current_gen_kwargs["max_new_tokens"], 
                    top_k=current_gen_kwargs.get("top_k", None), 
                    use_cache=self.use_cache, 
                    stopping_criteria=stoppers,
                    cache_dir=cache_dir,)
            
            cache_info = {}
            if hasattr(inputs["past_key_values"], "peak_cached_tokens"):
                cache_info["peak_cached_tokens"] = inputs["past_key_values"].peak_cached_tokens.cpu().numpy()
                cache_info["full_cached_tokens"] = self.model.config.text_config.num_hidden_layers * self.model.config.text_config.num_attention_heads * cont.shape[1]

            end_time = time.time()
            # [END MODIFIED]

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Calculate timing metrics for batch
            e2e_latency += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for ans, context in zip(answers, batched_messages):
                context = "<I dont save context>"
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Raw Response: {ans}")
                eval_logger.debug(f"Model Clean Response: {clean_ans}")

            # [MODIFIED] Free per-sample memory. Video decoding (decord) otherwise
            # accumulates CPU RAM across samples (~tens of GB/video for long clips),
            # OOM-killing long video-eval jobs. Explicit del + gc.collect() fixes it.
            del inputs, cont, generated_ids_trimmed, answers
            gc.collect()
            torch.cuda.empty_cache()
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        # Log metrics
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        return res, cache_info


def generate_and_log_cache_info(model, cache_dir, **generate_kwargs):
    original_forward = model.forward

    special_token_ids = [id for id in [
        getattr(model.config, "vision_start_token_id", None), 
        getattr(model.config, "vision_end_token_id", None),
        getattr(model.config.text_config, "bos_token_id", None) if hasattr(model.config, "text_config") else None,
        getattr(model.config.text_config, "eos_token_id", None) if hasattr(model.config, "text_config") else None,
        generate_kwargs.get("pad_token_id", None)
    ] if id is not None]
    
    vision_token_ids = [id for id in [
        getattr(model.config, "image_token_id", None), 
        getattr(model.config, "video_token_id", None)
    ] if id is not None]
    
    chosen_steps = [15, 50, 100, 500, 1000, 2000, 5000, 10000, 15000]

    v_ids_arr = np.array(vision_token_ids)
    s_ids_arr = np.array(special_token_ids)
    
    # Trackers for on-the-fly calculation
    running_seq = []
    history_visual = []
    history_special = []
    history_total = []
    history_vision_start = []
    history_vision_end = []
    history_eos = []
    history_bos = []
    os.makedirs(cache_dir, exist_ok=True)
    
    def forward_log_paged_cache(*args, **kwargs):
        # 1. Intercept input_ids to maintain the full sequence up to the current timestep
        if 'input_ids' in kwargs:
            curr_ids = kwargs['input_ids'][0].cpu().numpy()
        else:
            curr_ids = args[0][0].cpu().numpy()
            
        running_seq.extend(curr_ids.tolist())
        current_full_seq = np.array(running_seq)

        # 2. Execute forward pass
        outputs = original_forward(*args, **kwargs)
        
        # 3. Extract cache data
        data = outputs.past_key_values.log()
        
        step_visual = []
        step_special = []
        step_total = []
        step_vision_start = []
        step_vision_end = []
        step_eos = []
        step_bos = []
        
        flat_head_lens = data['flat_head_lens']
        head_wise_kv_positions = data['head_wise_kv_positions']
        head_wise_scores = data['head_wise_scores']

        
        # 4. Calculate counts on the fly
        for h in range(len(head_wise_kv_positions)):
            pos_array = head_wise_kv_positions[h]
            step_total.append(flat_head_lens[h])
            
            # Map cached absolute positions to the token IDs seen so far
            # Using pos_array[pos_array < len(current_full_seq)] prevents indexing errors 
            valid_pos = pos_array[pos_array < len(current_full_seq)]
            cached_token_ids = current_full_seq[valid_pos]
            
            step_visual.append(np.isin(cached_token_ids, v_ids_arr).sum())
            step_special.append(np.isin(cached_token_ids, s_ids_arr).sum())
            step_vision_start.append(np.isin(cached_token_ids, [model.config.vision_start_token_id]).sum())
            step_vision_end.append(np.isin(cached_token_ids, [model.config.vision_end_token_id]).sum())
            step_eos.append(np.isin(cached_token_ids, [model.config.text_config.eos_token_id]).sum())
            step_bos.append(np.isin(cached_token_ids, [model.config.text_config.bos_token_id]).sum())

        history_visual.append(step_visual)
        history_special.append(step_special)
        history_total.append(step_total)
        
        history_vision_start.append(step_vision_start)
        history_vision_end.append(step_vision_end)
        history_eos.append(step_eos)
        history_bos.append(step_bos)
        
        if len(history_total) in chosen_steps:
            print(f"Step {len(history_total)}: Visual Tokens per head: {step_visual}, Special Tokens per head: {step_special}, Total Cached Tokens per head: {step_total}")
            np.savez(os.path.join(cache_dir, f"head_wise_kv_positions_at_{len(history_total)}"), *[data.cpu().numpy() for data in head_wise_kv_positions])
            np.savez(os.path.join(cache_dir, f"head_wise_scores_at_{len(history_total)}.npy"), *[data.cpu().numpy() for data in head_wise_scores])
        
        return outputs
    
    model.forward = forward_log_paged_cache
    outputs = model.generate(**generate_kwargs)
    model.forward = original_forward

    

    # Transpose from [generation_length, num_heads] to [num_heads, generation_length]
    num_visual_tokens = np.array(history_visual, dtype=np.int32).T
    num_special_tokens = np.array(history_special, dtype=np.int32).T
    num_total_tokens = np.array(history_total, dtype=np.int32).T
    num_vision_start_tokens = np.array(history_vision_start, dtype=np.int32).T
    num_vision_end_tokens = np.array(history_vision_end, dtype=np.int32).T
    num_eos_tokens = np.array(history_eos, dtype=np.int32).T
    num_bos_tokens = np.array(history_bos, dtype=np.int32).T

    print('visual tokens:', num_visual_tokens[:,-1])
    print('special tokens:', num_special_tokens[:,-1])
    print('total tokens:', num_total_tokens[:,-1])
    print('vision end tokens:', num_vision_end_tokens[:,-1])
    print('vision start tokens:', num_vision_start_tokens[:,-1])
    print('eos tokens:', num_eos_tokens[:,-1])
    print('bos tokens:', num_bos_tokens[:,-1])
    np.save(os.path.join(cache_dir, "generation_outputs.npy"), outputs.cpu().numpy())
    np.save(os.path.join(cache_dir, "num_visual_tokens.npy"), num_visual_tokens)
    np.save(os.path.join(cache_dir, "num_special_tokens.npy"), num_special_tokens)
    np.save(os.path.join(cache_dir, "num_total_tokens.npy"), num_total_tokens)
    np.save(os.path.join(cache_dir, "num_vision_start_tokens.npy"), num_vision_start_tokens)
    np.save(os.path.join(cache_dir, "num_vision_end_tokens.npy"), num_vision_end_tokens)
    np.save(os.path.join(cache_dir, "num_eos_tokens.npy"), num_eos_tokens)
    np.save(os.path.join(cache_dir, "num_bos_tokens.npy"), num_bos_tokens)

    return outputs
