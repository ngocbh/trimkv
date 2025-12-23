# TRIM-KV: Token Retention for Memory-Bounded Key-Value Eviction

<a href="https://arxiv.org/pdf/2512.03324"><img src="https://img.shields.io/badge/arxiv-2512.03324-red?style=for-the-badge"></a>

### What is TRIM-KV?

> An efficient and learnable key–value eviction strategy designed to improve the efficiency of large language models (LLMs) in long-horizon inference.

Imagine if our brain worked like a transformer:

<div align="center">
    <img width="1000" alt="teaser" src="assets/fun.gif"/>
</div>

It will explode 🧠💥 and so would our GPU. TRIM-KV lets your model forget the parts that don’t matter much, so it doesn’t melt its VRAM.

The core idea behind TRIM-KV is to learn the intrinsic importance of each key–value pair at creation time, what we call *token retention*, and then decay this importance exponentially over time, mimicking standard inference with eviction.

The retention score is query-agnostic and captures the long-term utility of tokens. This is different from attention scores, which are query-dependent: they capture the short-term utility for predicting the next token and are recomputed at every step, making them local, myopic, and highly dependent on the transient decoding state.

In case you find this useful

```tex
@article{bui2025cache,
  title={Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs},
  author={Bui, Ngoc and Sharma, Shubham and Lamba, Simran and Mishra, Saumitra and Ying, Rex},
  journal={arXiv preprint arXiv:2512.03324},
  year={2025}
}
```


### Why TRIM-KV?

It's fast

<div align="center">
    <img width="1000" alt="teaser" src="assets/speed.png"/>
</div>

It's smart

<div align="center">
    <img width="1000" alt="teaser" src="assets/performance.png"/>
</div>


And it's interpretable

<div align="center">
    <img width="1000" alt="teaser" src="assets/eviction.png"/>
</div>

<div align="center">
    <img width="1000" alt="teaser" src="assets/vis.png"/>
</div>

One of the most interesting observations we found is that, in some layers, TRIM-KV tends to retain only period tokens. This suggests that, in these heads, period tokens may implicitly act as gist tokens that summarize information from the preceding sentence. This contrasts with recent approaches that advocate retaining entire chunks of tokens. Our results indicate that it can be more budget-efficient to keep carefully chosen individual tokens, since they already carry rich contextual information.

Please refer to Section 5.1.2 in our paper for detailed discussions. Check [`examples/visualize.ipynb`](examples/visualize.ipynb) for the visualization script.

---

### Some Notes

- I’m waiting for approval before releasing the code. (We released parts of the code for testing our models. Training and experiments will be released soon...)

- We recently refactored the code to support transformers v4.57.0. This release is to freeze the codebase at a version close to what was used in the paper to ensure all reported results are reproducible. If you encounter any issues, please open a GitHub issue.

- Further refactoring and updates are planned in the near future.

---

## Getting Started

### Requirements

- Python 3.11 or higher (tested with 3.12)
- PyTorch 2.7.0 or higher (tested with 2.8.0)
- FlashAttention 2.7.2.post1 or higher (tested with 2.8.0)
- Transformers 4.57.1

```sh
pip install -r requirements.txt
```

This is a minimal set of requirements for training purposes. Additional dependencies may be needed for running specific experiments. We provided a full example of the environment used in our experiments in [`examples/env.yaml`](examples/env.yaml).

### Installation

From the root of the repo:

```sh
git clone https://github.com/ngocbh/trimkv.git
cd trimkv
pip install -e .
````

---

## Quick Start

```python
import torch
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import TrimKVCache
from transformers import AutoTokenizer

model_path = "<TrimKV model_path here>"
download_from = "huggingface"  # options: "wandb", "local", "huggingface"

model = TrimKVQwen3ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_trimkv_weights=True,
    download_from=download_from,
    use_cache=True,
    device_map="cuda",
)

# Configure TRIM-KV settings
model.config._attn_implementation = "flash_attention_2"
model.config.compress_memory = True
model.config.memory_size = 512
model.config.buffer_size = 128

tokenizer = AutoTokenizer.from_pretrained(
    model.config.base_model,
    use_fast=True,
    padding_side="left",
)

# Use model.generate as normal.
# Note: TRIM-KV uses TrimKVCache under the hood. So please pass TrimKVCache to model.generate
```

For a runnable end-to-end example, see [`examples/test_qwen3.py`](examples/test_qwen3.py).

---

## Training

Training scripts for LLMs with TRIM-KV are available under: [`train/llm/scripts`](train/llm/scripts)

Please refer to the scripts and their arguments for training configurations and usage.

---

## Experiments

Reproduction details, experiment setups, and evaluation instructions are documented in: [`experiments/README.md`](experiments/README.md)

This repo includes implementations of TRIM-KV, along with a set of baselines and benchmarks.

- **Baselines**
  - `TrimKV`, `R-KV`, `SeerAttn-R`, `SnapKV`, `StreamingLLM`, `H2O`, `KeyDiff`, `LocRet`

- **Benchmarks**
  - Long-horizon generation: `GSM8K`, `MATH-500`, `AIME24`, `LongProc`
  - Long-context understanding: `SCBench`, `LongMemEval`, `LongBench`, `LongBenchV2`

---

## Released Models

| Base Model                    | TRIM-KV Checkpoints                           | Training Datasets        | Training Context Len | Training $M$ |
|------------------------------|-----------------------------------------------|--------------------------|-------------------------|--------------|
| Qwen3-1.7B                     | [TRIM-KV-Qwen3-1.7B-Math](https://huggingface.co/ngocbh/TrimKV-Qwen3-1.7B-Math)            | OpenR1-Math-220k          | 16K   | 512     |
| Qwen3-4B                     | [TRIM-KV-Qwen3-4B-Math](https://huggingface.co/ngocbh/TrimKV-Qwen3-4B-Math)            | OpenR1-Math-220k          | 16K   | 512     |
| Qwen3-8B                     | [TRIM-KV-Qwen3-8B-Math](https://huggingface.co/ngocbh/TrimKV-Qwen3-8B-Math)            |  OpenR1-Math-220k          | 16K   | 512     |
| Qwen3-14B                    | [TRIM-KV-Qwen3-14B-Math](https://huggingface.co/ngocbh/TrimKV-Qwen3-14B-Math)           |  OpenR1-Math-220k         | 16K   | 512     |
| Qwen3-4B-Instruct-2507 | [TrimKV-Qwen3-4B-Instruct-2507](https://huggingface.co/ngocbh/TrimKV-Qwen3-4B-Instruct-2507) | Synth-Long, BookSum, Buddhi      |  128K      | 4096     |
| Phi-3-mini-128k-instruct | [TrimKV-Phi-3-mini-128k-instruct](https://huggingface.co/ngocbh/TrimKV-Phi-3-mini-128k-instruct) | LongAlpaca          |  128K  | 2048 |
| DeepSeek-R1-Distill-Llama-8B                    | [TrimKV-DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/ngocbh/TrimKV-DeepSeek-R1-Distill-Llama-8B)           |  OpenR1-Math-220k         | 32K   | 512     |

---

## Acknowledgements

A large portion of this repository is adapted from or built upon the following projects:

* [https://github.com/microsoft/SeerAttention](https://github.com/microsoft/SeerAttention)
* [https://github.com/Zefan-Cai/R-KV](https://github.com/Zefan-Cai/R-KV)
* [https://github.com/microsoft/MInference](https://github.com/microsoft/MInference)
* [https://github.com/huangyuxiang03/Locret](https://github.com/huangyuxiang03/Locret)
* [https://github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
* [https://github.com/princeton-pli/LongProc](https://github.com/princeton-pli/LongProc)
* [https://github.com/THUDM/LongBench](https://github.com/THUDM/LongBench)
