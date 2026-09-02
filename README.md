# GLM-5.3-Flash on One DGX Spark

A reproducible, production-capable vLLM deployment of **Z.ai GLM-5.3-Flash on one NVIDIA DGX Spark**, using [Turboderp's 2.05-bpw EXL3 checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw) and [Inco AI's DFlash2 drafter](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

> **64.1 tok/s structured C1 · 25.1 tok/s prose · 181.9 tok/s C4 active-stream aggregate · 262K context**

This is a **deployment/runtime recipe**, not a new model or quantization. Model weights remain with their original publishers and are never bundled here.

Hugging Face discovery page: https://huggingface.co/gitcommit90/GLM-5.3-Flash-EXL3-2.05-One-Spark

## Headline results

One DGX Spark (GB10, 128 GB unified memory), TP1, EXL3 2.05 bpw, DFlash2 K7, FP8 KV, thinking off, direct vLLM backend:

| Workload | Sampling | Result |
|---|---|---:|
| Structured C1, five-run median | temperature 0 | **64.053 tok/s** |
| Structured C1, five-run median | temperature 1.0, top-p 0.95 | **62.637 tok/s** |
| Open-ended prose, five-run median | temperature 1.0, top-p 0.95 | **25.059 tok/s** |
| Structured C4, median active stream | temperature 1.0, top-p 0.95 | **40.958 tok/s/stream** |
| Structured C4, summed active-stream convention | temperature 1.0, top-p 0.95 | **181.944 tok/s** |
| Structured C4, strict submission-to-completion wall | temperature 1.0, top-p 0.95 | **91.475 tok/s** |

**C4 disclosure:** 181.944 tok/s follows MiaAI's sum-of-individual-active-stream-rates convention. The current mixed-prefill scheduler admits C4 in stages, so strict full-batch wall throughput is 91.475 tok/s. Both are reported to avoid conflating active decode with delivered wall throughput.

### Long-context prefill and prefix cache

| Prompt | Cold prefill | Cold TTFT | Warm TTFT | Cached tokens |
|---:|---:|---:|---:|---:|
| 8,002 tokens | **786.4 tok/s** | 10.175 s | 1.775 s | 7,168 |
| 16,002 tokens | **822.1 tok/s** | 19.464 s | 2.786 s | 14,336 |
| 100,002 tokens | **845.7 tok/s** | 118.250 s | 8.842 s | 93,184 |

All cold requests had zero cache hits. All checksum answers were correct. Raw evidence and harnesses are in [`benchmarks/raw`](benchmarks/raw).

## What this project contributes

The two-Spark work by [Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) established the vLLM/EXL3/DFlash foundation. This project adapts and extends that foundation for a very different target:

- Generalized **TP1/full-model EXL3 loading** instead of TP2 sharding
- Support for Turboderp's mixed-rate **2.05-bpw `mul1`** checkpoint
- Correct `mul1` codebook propagation through fused routed MoE
- ARM64/GB10 `sm_121a` ExLlamaV3 build and x86-intrinsic fail-closed fixes
- Vision EXL3 ledger, QKV mapping, and dual-layout repairs
- BF16/FP16 compatibility bridge needed by this checkpoint/runtime combination
- DFlash2 K7 integration with the full target model
- A validated one-Spark memory shape with 262,144-token context
- Reproducible benchmark harnesses and raw evidence
- Persistent OpenAI-compatible serving with reasoning, tools, JSON, vision, and prefix caching

See [PROVENANCE.md](PROVENANCE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing.

## Runtime configuration

- Target: `turboderp/GLM-5.3-Flash-exl3`, branch `2.05bpw`, revision `51058cd551c7e570d87bd32a4adee720edce2349`
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- ExLlamaV3: `e648f1a131365aae15920073e761a3fa5a527654` (v1.4.5)
- vLLM ARM64 base image digest: `sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`
- TP1; DFlash2 K7; FP8 KV; CUDA graphs
- `max_model_len=262144`
- `max_num_seqs=4`
- `max_num_batched_tokens=7168`
- `gpu_memory_utilization=0.90`

Measured startup allocation:

| Component | Memory |
|---|---:|
| Loaded target + DFlash model allocation | 79.96 GiB |
| KV-cache pool | 20.75 GiB |
| CUDA graphs | 0.28 GiB |
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

- Initial model load is about **14 minutes** with warm build/JIT caches.
- The 20.75-GiB KV pool holds approximately 324,710 tokens total; four full 262K contexts do not fit simultaneously.
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
