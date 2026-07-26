# SETUP — Running the CAKE / HeadKV / CAKE+HeadKV KV-eviction baselines

This documents everything needed to run the three training-free KV-cache-eviction
baselines added for the rebuttal (AC-2 / UeJL-1) and reproduce their cells of
**Table 2** on **Qwen3-VL-8B-Thinking**:

| method (`--method`) | allocation axis | cache / integration | batch size |
|---|---|---|---|
| `cake`        | per-**layer** (dynamic, attention entropy×variance) | standard cache | **bs>1 OK** |
| `headkv`      | per-**head** (static, precomputed importance)        | flattened per-head (AdaKV) | **bs=1 only** |
| `cake_headkv` | **layer + head** (CAKE × HeadKV)                     | flattened per-head, deferred eviction | **bs=1 only** |

Datasets (5): `mmstar`, `mathvision_testmini`, `video_mmmu_adaptation`,
`video_mmmu_comprehension`, `videomathqa_mcq`. Budget `M = 1024` (global `M×L×H`).
All 5 use **rule-based scoring** (MC / accuracy) — no real LLM judge needed.

Code lives in `experiments/lmms-eval/` (branch **`efficiency`**).

---

## 1. Environment

Python **3.12**, CUDA **12.6** (H100/H200-class GPU). Create a fresh env and install
the pinned stack (this is the exact set the results were produced with):

```bash
conda create -n trimkv python=3.12 -y && conda activate trimkv

# Core (from repo root requirements.txt)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.57.1 accelerate==1.8.1 datasets==3.6.0 numpy==1.26.4 \
            deepspeed==0.16.3 trl==0.19.0 hf_transfer==0.1.9 qwen-vl-utils==0.0.14 \
            fire faiss-cpu pebble wandb word2number timeout_decorator google-genai pydantic==2.11.7

# flash-attn: install the PREBUILT wheel (a source build is slow/unnecessary).
# Pick the wheel matching torch2.8 / cu12 / cp312 / cxx11abiTRUE from
# https://github.com/Dao-AILab/flash-attention/releases  (version 2.8.3.post1):
pip install flash_attn==2.8.3.post1 --no-build-isolation   # or the direct wheel URL

# trimkv package (repo root) — NOTE: `import trimkv` requires flash_attn
pip install -e .

# lmms-eval fork (harness) — installs as editable; may downgrade numpy->1.26.4, httpx->0.23.3 (harmless)
pip install -e experiments/lmms-eval
```

### 1a. torchcodec + FFmpeg (REQUIRED for the video datasets)

Video decoding **must** use `torchcodec`, not `decord`. The decord path leaks CPU
RAM (~tens of GB per long clip) and OOM-kills video jobs; torchcodec seeks frames
(≈2–3 s/video, ~50 MB) and fixes both the OOM and the runtime.

```bash
conda install -n trimkv -y -c conda-forge --freeze-installed "ffmpeg<8"   # provides libav* for torchcodec
pip install "torchcodec==0.7.*" --no-deps                                  # matches torch 2.8; keeps torch pinned
# verify:
python -c "from torchcodec.decoders import VideoDecoder; import torchcodec; print('torchcodec', torchcodec.__version__)"
```

At run time set `FORCE_QWENVL_VIDEO_READER=torchcodec` (see §4) so qwen-vl-utils uses it.

---

## 2. HuggingFace auth, models, datasets

An **HF token is required** (some datasets are token-gated; VideoMMMU is
access-gated — the token's account must be granted access at
`huggingface.co/datasets/lmms-lab/VideoMMMU`).

- **Base model** (baselines run on the *vanilla* base): `Qwen/Qwen3-VL-8B-Thinking`
- Datasets (auto-downloaded on first run): `Lin-Chen/MMStar`, `MathLLMs/MathVision`,
  `lmms-lab/VideoMMMU` (configs **Adaptation** and **Comprehension** cache
  separately — run each online once), `MBZUAI/VideoMathQA`.

Video files are extracted under `$HF_HOME/video_mmmu/` and `$HF_HOME/videomathqa/videos/`.

---

## 3. HeadKV head-importance scores

`headkv` and `cake_headkv` need a per-head importance file at
`experiments/lmms-eval/rkv/head_score/Qwen3-VL-8B-Thinking_reason.json`
(**already committed** on `efficiency`). To regenerate on a new model:

```bash
cd experiments/lmms-eval
python rkv/headkv_detect.py --model Qwen/Qwen3-VL-8B-Thinking   # ~15 min on 1 GPU, offline
```

---

## 4. Environment variables (set before launching)

```bash
export HF_HOME=<big-scratch>/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=<your_hf_token>                     # gated datasets
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=torchcodec         # REQUIRED for video (else OOM)
# The task modules build an OpenAI-compatible judge client AT IMPORT; these 5 tasks
# score rule-based, so a non-empty DUMMY key is enough:
export API_TYPE=openai
export OPENAI_API_KEY=dummy_rule_based_scoring
# clear any inherited/broken proxy vars if your launcher snapshots them
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

`run_benchmark.py` also calls `load_dotenv()`, so these can instead live in
`experiments/lmms-eval/.env` (gitignored — put the real HF token there).

---

## 5. Running the evaluations

Run from `experiments/lmms-eval/`. One GPU per job. Template:

```bash
python run_benchmark.py \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --method <cake|headkv|cake_headkv> \
    --compress_args=kv_budget=1024 \
    --tasks <TASK> \
    --batch_size <BS> \
    --gen_kwargs=max_new_tokens=32768 \
    --output_path ./results/table2/Qwen3-VL-8B-Thinking/<method> \
    --run_name <method>_b1024_bs<BS> \
    --log_samples
```

**Full run matrix (15 jobs = 3 methods × 5 datasets):**

| method | mmstar | mathvision_testmini | video_mmmu_adaptation | video_mmmu_comprehension | videomathqa_mcq |
|---|---|---|---|---|---|
| `cake`        | bs=16 | bs=16 | bs=1–4 | bs=1–4 | bs=1–4 |
| `headkv`      | bs=1  | bs=1  | bs=1   | bs=1   | bs=1 |
| `cake_headkv` | bs=1  | bs=1  | bs=1   | bs=1   | bs=1 |

- **`cake` supports `batch_size>1`** (padding-aware; `bs=1` is byte-identical to the
  reference path). Use bs=16 on image/text; keep video modest (bs=1–4) for GPU memory.
- **`headkv` / `cake_headkv` are `batch_size=1` only** (flattened per-head varlen cache
  hard-asserts `B==1`). `run_benchmark`/`load_model` force bs=1 for them automatically.
- Keep `--gen_kwargs=max_new_tokens=32768` for all (Qwen3-VL-**Thinking** emits long
  reasoning traces, matching the paper's setup).
- **Do NOT pass `--rerun`** → jobs **resume** from their `results/.../<task>_<run_name>.jsonl`
  (skip already-done samples). Important for long/preempted video jobs.

Example (CAKE, MMStar, batched):
```bash
python run_benchmark.py --model Qwen/Qwen3-VL-8B-Thinking --method cake \
  --compress_args=kv_budget=1024 --tasks mmstar --batch_size 16 \
  --gen_kwargs=max_new_tokens=32768 --run_name cake_b1024_bs16 \
  --output_path ./results/table2/Qwen3-VL-8B-Thinking/cake --log_samples
```

---

## 6. Resources & runtime (per job, 1× H100/H200)

| task type | GPU mem | CPU RAM | approx runtime |
|---|---|---|---|
| image/text (mmstar, mathvision) | 1 GPU | ~200 GB | ~1–4 h (cake bs=16 much faster) |
| video (with torchcodec) | 1 GPU | ~200 GB (generous headroom) | ~10–15 h at bs=1 |

Without torchcodec, video jobs OOM — do not skip §1a.

---

## 7. Collecting results & Table-2 comparison

Each run writes `results/table2/Qwen3-VL-8B-Thinking/<method>/<task>_<run_name>_results.json`.
Headline metric per task: `average` (mmstar), `mathvision_standard_eval` (mathvision),
`mmmu_acc` (video_mmmu_*), `videomathqa_perception_score` (videomathqa_mcq); read
`results[<task>]["<metric>,none"]`.

**Paper Table-2 @ budget 1024 (for comparison; `*` = trained):**

| Method | MMStar | MathVision | VMMMU-adapt | VMMMU-compr | VideoMathQA |
|---|---|---|---|---|---|
| Vanilla        | 71.52 | 48.68 | 35.67 | 55.00 | 36.19 |
| SnapKV         | 51.84 | 15.13 | 21.00 | 28.67 | 20.24 |
| R-KV           | 58.42 | 24.67 | 22.00 | 30.33 | 26.43 |
| AdaKV          | 66.89 | 32.89 | 25.42 | 31.67 | 23.57 |
| Ada-Pyramid-KV | 66.84 | 28.29 | 28.96 | 30.74 | 23.09 |
| TrimKV\*       | 70.64 | 45.72 | 35.00 | 59.00 | 35.00 |
| DBTrimKV\*     | 71.50 | 52.63 | 37.00 | 59.33 | 36.43 |

Report each new method's 5 accuracies + `Avg %vsVanilla` = mean over the 5 of
`method/vanilla × 100`. Expected ordering: **CAKE+HeadKV ≳ CAKE ≈ HeadKV > SnapKV**,
all below the trained TrimKV/DBTrimKV — i.e., layer+head allocation helps, but the
learned gate (DBTrimKV) is what closes the gap to full cache.

---

## 8. Gotchas / checklist

- [ ] `torchcodec` + FFmpeg installed and `FORCE_QWENVL_VIDEO_READER=torchcodec` set (else video OOM).
- [ ] `HF_TOKEN` set **and** granted access to `lmms-lab/VideoMMMU` (gated).
- [ ] First video-task run must be **online** (downloads + extracts videos, per config); later runs can be offline (`HF_HUB_OFFLINE=1`).
- [ ] `headkv`/`cake_headkv`: bs=1 only; head-score JSON present at `rkv/head_score/`.
- [ ] Non-empty `OPENAI_API_KEY` (dummy fine) — the task modules construct a judge client at import even though scoring is rule-based.
- [ ] Omit `--rerun` to allow resume from partial `.jsonl` outputs.
