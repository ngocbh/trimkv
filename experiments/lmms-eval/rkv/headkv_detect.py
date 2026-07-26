"""Generate per-head importance scores for Qwen3-VL-8B-Thinking via retrieval-head
detection (needle-in-a-haystack), adapted from HeadKV's Important_Head detection
(third_party/HeadKV) to native HF Qwen3-VL with eager attention.

Method: insert a needle sentence into a Paul-Graham-essay haystack at various
context lengths and depths, ask the retrieval question, and during decoding measure
— per (layer, Q-head) — how much of each head's TOP attention lands on the needle
span (the retrieval-head signal, Wu et al. 2024). Accumulate over all (length,depth)
probes. Output {"L-H": [per-probe scores]} JSON, the format HeadKV expects; GQA
aggregation (32 Q-heads -> 8 KV-heads) happens later in rkv/compression/headkv.py.

Run offline on 1 GPU (model cached, haystack local). ~10-20 min.
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
QUESTION = "What is the best thing to do in San Francisco?"
ANSWER_KEY = "sandwich"  # lenient success check


def read_haystack(haystack_dir, min_words):
    ctx = ""
    files = sorted(glob.glob(os.path.join(haystack_dir, "*.txt")))
    while len(ctx.split()) < min_words:
        for fp in files:
            with open(fp, "r", errors="ignore") as f:
                ctx += f.read()
            if len(ctx.split()) >= min_words:
                break
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    ap.add_argument("--haystack_dir", default="/storage/home/ngocbh/project/trimkv/third_party/HeadKV/data/PaulGrahamEssays")
    ap.add_argument("--out", default="/storage/home/ngocbh/project/trimkv/experiments/lmms-eval/rkv/head_score/Qwen3-VL-8B-Thinking_reason.json")
    ap.add_argument("--ctx_lengths", default="1000,2000,4000")
    ap.add_argument("--depths", default="0,20,40,60,80,100")
    ap.add_argument("--decode_len", type=int, default=32)
    args = ap.parse_args()

    ctx_lengths = [int(x) for x in args.ctx_lengths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]

    print(f"Loading {args.model} (eager attention) ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
    ).eval()
    tok = AutoProcessor.from_pretrained(args.model).tokenizer
    dev = model.device

    tc = model.config.text_config
    L, Hq = tc.num_hidden_layers, tc.num_attention_heads
    print(f"L={L} Hq={Hq}", flush=True)

    needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
    q_ids = tok(f"\nQuestion: {QUESTION}\nAnswer:", add_special_tokens=False).input_ids
    topk = len(needle_ids)

    head_counter = {f"{l}-{h}": [] for l in range(L) for h in range(Hq)}
    haystack = read_haystack(args.haystack_dir, max(ctx_lengths) + 500)
    base_all = tok(haystack, add_special_tokens=False).input_ids

    n_ok = 0
    for clen in ctx_lengths:
        base_ids = base_all[:clen]
        for depth in depths:
            ins = int(len(base_ids) * depth / 100)
            prompt_ids = base_ids[:ins] + needle_ids + base_ids[ins:]
            ns, ne = ins, ins + len(needle_ids)  # needle token span
            full_ids = prompt_ids + q_ids
            input_ids = torch.tensor([full_ids], device=dev)

            run_score = torch.zeros(L, Hq, dtype=torch.float64)
            gen = []
            with torch.no_grad():
                out = model(input_ids=input_ids[:, :-1], use_cache=True)
                past = out.past_key_values
                inp = input_ids[:, -1:]
                for _ in range(args.decode_len):
                    out = model(input_ids=inp, past_key_values=past, use_cache=True, output_attentions=True)
                    past = out.past_key_values
                    for l in range(L):
                        a = out.attentions[l][0, :, -1, :]  # (Hq, kv_len)
                        vals, idx = a.topk(topk, dim=-1)
                        in_span = (idx >= ns) & (idx < ne)
                        run_score[l] += (vals * in_span).sum(dim=-1).double().cpu() / topk
                    nxt = int(out.logits[0, -1].argmax().item())
                    gen.append(nxt)
                    inp = torch.tensor([[nxt]], device=dev)
                    if nxt == tok.eos_token_id:
                        break
            resp = tok.decode(gen, skip_special_tokens=True)
            ok = ANSWER_KEY in resp.lower()
            n_ok += int(ok)
            for l in range(L):
                for h in range(Hq):
                    head_counter[f"{l}-{h}"].append(float(run_score[l, h]))
            print(f"clen={clen} depth={depth} ok={ok} resp[:60]={resp[:60]!r}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(head_counter, f)

    # sanity: top retrieval heads
    means = sorted(((k, sum(v) / len(v)) for k, v in head_counter.items()), key=lambda x: -x[1])
    print(f"\nSaved {args.out}  ({len(head_counter)} heads, {len(next(iter(head_counter.values())))} probes, {n_ok} answered correctly)")
    print("Top-10 retrieval heads (layer-head: score):", [(k, round(s, 4)) for k, s in means[:10]])
    print("HEADKV_DETECT_DONE")


if __name__ == "__main__":
    main()
