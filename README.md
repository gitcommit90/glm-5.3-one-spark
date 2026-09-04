# GLM-5.3-Flash on One DGX Spark

A reproducible, production-capable vLLM deployment of **Z.ai GLM-5.3-Flash on one NVIDIA DGX Spark**, using [Turboderp's 2.05-bpw EXL3 checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw) and [Inco AI's DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

> **64.1 tok/s structured C1 (K7) · 29.9 tok/s prose / 40.1 tok/s code (K5 default) · 181.9 tok/s C4 active-stream aggregate · 262K context**
>
> ⭐ **2026-09-03: default `num_speculative_tokens` changed from 7 to 5** after a K sweep — best prose and code speed, ~16% slower on pure structured output. K=8 remains the structured-output setting. See [K sweep](#speculative-depth-k-sweep).
>
> ⭐ **2026-09-03: cold start to API-ready cut from ~14 min to 3 min 21 s (−76%)** with no change to throughput. Details in [`docs/startup-time.md`](docs/startup-time.md).

This is a **deployment/runtime recipe**, not a new model or quantization. Model weights remain with their original publishers and are never bundled here.

Hugging Face discovery page: https://huggingface.co/gitcommit90/GLM-5.3-Flash-EXL3-2.05-One-Spark
Hugging Face Collection: https://huggingface.co/collections/gitcommit90/glm-53-one-spark-6a98b70df9981ae425acbc05

## Headline results

One DGX Spark (GB10, 128 GB unified memory), TP1, EXL3 2.05 bpw, DFlash2 **K7** (the default at the time these were collected; shipped default is now K5, see below), FP8 KV, thinking off, direct vLLM backend:

| Workload | Sampling | Result |
|---|---|---:|
| Structured C1, five-run median | temperature 0 | **64.053 tok/s** |
| Structured C1, five-run median | temperature 1.0, top-p 0.95 | **62.637 tok/s** |
| Open-ended prose, five-run median | temperature 1.0, top-p 0.95 | **25.059 tok/s** |
| Structured C4, median active stream | temperature 1.0, top-p 0.95 | **40.958 tok/s/stream** |
| Structured C4, summed active-stream convention | temperature 1.0, top-p 0.95 | **181.944 tok/s** |
| Structured C4, strict submission-to-completion wall | temperature 1.0, top-p 0.95 | **91.475 tok/s** |

**C4 disclosure:** 181.944 tok/s follows MiaAI's sum-of-individual-active-stream-rates convention. The current mixed-prefill scheduler admits C4 in stages, so strict full-batch wall throughput is 91.475 tok/s. Both are reported to avoid conflating active decode with delivered wall throughput.

### Speculative depth (K) sweep

`num_speculative_tokens` is the only knob that moved after release. Spec decode is lossless, so K changes speed only, never output. Measured 2026-09-03 on the live General23 image, temperature 0, thinking off, 400 output tokens, C1, one restart per K (3 runs each; K7 structured baseline 5 runs). Raw: [`benchmarks/raw/k-sweep-20260903`](benchmarks/raw/k-sweep-20260903).

| K | Structured (count 1→200) | Prose (hash-map explainer) | Code (LRU cache) | Structured accept/step | Prose accept/step |
|--:|---:|---:|---:|---:|---:|
| 4 | 48.6 tok/s | 28.8 | 37.7 | 3.99 | 1.99 |
| **5 (default)** | 53.5 | **29.9** | **40.1** | 4.94 | 2.30 |
| 6 | 59.2 | 29.0 | 37.5 | 5.91 | 2.46 |
| 7 (previous default) | 63.9 | 25.8 | 38.3 | 6.86 | 2.29 |
| 8 | **66.9** | 23.9 | 38.0 | 7.70 | 2.23 |

Reading: structured output accepts almost every draft token, so speed rises monotonically with K. Prose and code accept ~2–4 tokens/step regardless of K, so each extra draft token past ~5 is wasted drafter work and a longer step (110 ms at K5 vs 127 ms at K7). K5 is the best all-round default for chat/coding use; set `K=8` if your workload is dominated by lists, JSON, or repetitive structured text.

Override at launch: `ONE_SPARK_K=8 ./start.sh` (or edit `num_speculative_tokens` in `scripts/serve-one-spark.sh`).

### Long-context prefill and prefix cache

| Prompt | Cold prefill | Cold TTFT | Warm TTFT | Cached tokens |
|---:|---:|---:|---:|---:|
| 8,002 tokens | **786.4 tok/s** | 10.175 s | 1.775 s | 7,168 |
| 16,002 tokens | **822.1 tok/s** | 19.464 s | 2.786 s | 14,336 |
| 100,002 tokens | **845.7 tok/s** | 118.250 s | 8.842 s | 93,184 |

All cold requests had zero cache hits. All checksum answers were correct. Raw evidence and harnesses are in [`benchmarks/raw`](benchmarks/raw).

## Quantization Analysis

This deployment uses [Turboderp's GLM-5.3-Flash EXL3 2.05-bpw quant](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw), revision `51058cd551c7e570d87bd32a4adee720edce2349`. The exact checkpoint is **85.23 GB (79.38 GiB)**.

An [independent full-vocabulary measurement](https://github.com/malaiwah/quant-fidelity-suite/blob/794d80fa79db4d30cd0fa8140a07c665dd363251/registry/receipts/malaiwah/stream-turbo-2.05bpw-kld.json) of this exact revision against BF16 teacher logits reported:

| Quant | Size | Top-1 agreement | Mean KLD | Scored positions |
|---|---:|---:|---:|---:|
| Turboderp EXL3 2.05 | 85.23 GB | **88.92%** | **0.121638** | **51,175** |

The measurement used 25 windows, the full 154,880-token vocabulary, teacher forcing, FP64 accumulation, and two cold runs with identical results. The checkpoint was quantized from the official FP8 release; the measurement reference is BF16. Machine-readable summary: [`benchmarks/quality/quantization-analysis.json`](benchmarks/quality/quantization-analysis.json).

## What this project contributes

The two-Spark work by [Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) established the vLLM/EXL3/DFlash foundation. This project adapts and extends that foundation for a very different target:

- Generalized **TP1/full-model EXL3 loading** instead of TP2 sharding
- Support for Turboderp's mixed-rate **2.05-bpw `mul1`** checkpoint
- Correct `mul1` codebook propagation through fused routed MoE
- ARM64/GB10 `sm_121a` ExLlamaV3 build and x86-intrinsic fail-closed fixes
- Vision EXL3 ledger, QKV mapping, and dual-layout repairs
- BF16/FP16 compatibility bridge needed by this checkpoint/runtime combination
- DFlash2 integration with the full target model (K5 default, K sweep published)
- A validated one-Spark memory shape with 262,144-token context
- Reproducible benchmark harnesses and raw evidence
- Persistent OpenAI-compatible serving with reasoning, tools, JSON, vision, and prefix caching

See [PROVENANCE.md](PROVENANCE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing.

## Runtime configuration

- Target: `turboderp/GLM-5.3-Flash-exl3`, branch `2.05bpw`, revision `51058cd551c7e570d87bd32a4adee720edce2349`
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- ExLlamaV3: `e648f1a131365aae15920073e761a3fa5a527654` (v1.4.5)
- vLLM ARM64 base image digest: `sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`
- TP1; DFlash2 K5 (`ONE_SPARK_K` override); FP8 KV; CUDA graphs
- `max_model_len=262144`
- `max_num_seqs=4`
- `max_num_batched_tokens=7168`
- `gpu_memory_utilization=0.90`

Measured startup allocation:

| Component | Memory |
|---|---:|
| Loaded target + DFlash model allocation | 79.96 GiB |
| KV-cache pool | 23.55 GiB (was 20.75 before the 2026-09-03 startup work) |
| CUDA graphs | 0.24 GiB |
| Live model process after startup | approximately 106.7 GiB |

## Quick start

### Requirements

- One NVIDIA DGX Spark
- Docker with NVIDIA GPU support
- Approximately 86 GB disk for model repositories, plus image/build space
- Hugging Face CLI (`pip install -U huggingface_hub`)

### 1. Download the original weights

Read the DFlash2 license first. It is **CC BY-NC-ND 4.0 for research/evaluation**, not a permissive commercial model license.

```bash
export ACCEPT_DFLASH2_NC_LICENSE=1
./download.sh
```

The script pins both repositories to the exact revisions tested here.

### 2. Launch

Use the published, pinned ARM64 runtime image:

```bash
IMAGE=ghcr.io/gitcommit90/glm-5.3-one-spark:general23 ./start.sh
```

Or build the exact image locally from this repository:

```bash
BUILD=1 ./start.sh
```

The API binds to `127.0.0.1:18080` by default:

```bash
curl http://127.0.0.1:18080/v1/models
```

Set `HOST=0.0.0.0` only if you intentionally want network exposure, and protect it with an authenticated reverse proxy.

### Existing model directories

```bash
MODEL_DIR=/path/to/GLM-5.3-Flash-exl3-2.05bpw \
DFLASH_DIR=/path/to/GLM-5.3-Flash-DFlash2 \
./start.sh
```

### Persistent user service

```bash
./scripts/install-systemd.sh
systemctl --user enable --now glm53-one-spark.service
```

## SparkRun / Spark Arena

A native, digest-pinned SparkRun recipe is provided at
[`sparkrun/glm-5.3-flash-exl3-2.05bpw-dflash2-vllm-gitcommit90.yaml`](sparkrun/glm-5.3-flash-exl3-2.05bpw-dflash2-vllm-gitcommit90.yaml).
It reproduces the one-Spark General23 launch without private overlays or bundled
model weights. See [`sparkrun/README.md`](sparkrun/README.md) for usage and the
DFlash2 license notice.

## Validation

Fast source/package checks:

```bash
./scripts/verify-package.sh
```

After the server is healthy:

```bash
python3 benchmarks/raw/default-sampling-pre-prefill-20260902T220641Z/run_pre_prefill_suite.py --help
python3 benchmarks/raw/default-sampling-cold-prefix-20260902T221828Z/run_cold_prefix_suite.py --help
```

Canonical results were collected directly from the backend with the scheduler idle before each suite. See [`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md).

## Known limitations

- Initial model load is about **3.5 minutes** cold (~14 minutes before 2026-09-03; see [`docs/startup-time.md`](docs/startup-time.md)).
- The 23.55-GiB KV pool holds approximately 368,268 tokens total; four full 262K contexts do not fit simultaneously.
- Current mixed-prefill admission stages C2/C4 work. This is why strict C4 wall throughput is lower than summed active-stream throughput.
- The special E2 fat-expert kernel is compiled but this checkpoint lacks the shared-SUH layout it requires; runtime safely uses the sorted fallback for those prefill cases. Core fused EXL3 routed-MoE decode remains active.
- DFlash2's checkpoint license limits it to noncommercial research/evaluation unless Inco AI grants another license.

## Credits

This work would not exist without:

- [Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) — original two-Spark vLLM/EXL3/DFlash recipe, overlays, tests, and benchmark convention
- [Z.ai / GLM-5 Team](https://huggingface.co/zai-org/GLM-5.3-Flash) — GLM-5.3-Flash
- [Turboderp](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) — 2.05-bpw quant and [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — DFlash2 drafter
- [vLLM contributors](https://github.com/vllm-project/vllm) — serving engine and upstream fixes

Prior single-Spark GLM work by vcruz305, Weschera, and Sayyidfareed also helped establish the surrounding state of the art. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license details.

## License

Project-authored integration code is released under MIT. Substantial portions are derived from MiaAI's MIT-licensed repository and retain its copyright. Embedded or downloaded third-party components remain under their own licenses. In particular, **DFlash2 is CC BY-NC-ND 4.0**.
