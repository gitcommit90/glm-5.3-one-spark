---
license: mit
library_name: vllm
tags:
- glm-5.3-flash
- dgx-spark
- gb10
- exl3
- vllm
- dflash2
- speculative-decoding
- arm64
- deployment
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: turboderp
---

# GLM-5.3-Flash EXL3 2.05 — One DGX Spark

A reproducible, production-capable deployment of **GLM-5.3-Flash on one NVIDIA DGX Spark**, using:

- [Turboderp's 2.05-bpw EXL3 checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw)
- [Inco AI's DFlash2 drafter (K5 default)](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
- A TP1 ARM64/SM121 vLLM runtime derived from [MiaAI's two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)

> **64.1 tok/s structured C1 (K7) · 29.9 tok/s prose / 40.1 tok/s code (K5 default) · 181.9 tok/s C4 active-stream aggregate · 262K context**

## This repository does not contain model weights

This is a **deployment/runtime discovery page**, not a new model or quant. The software, Docker recipe, benchmark harnesses, and raw evidence live in the linked GitHub repository. Target and draft weights download directly from their original publishers.

## Measured results

One DGX Spark, TP1, EXL3 2.05 bpw, DFlash2 K7 (default at collection time; shipped default is now K5), FP8 KV, thinking disabled:

| Benchmark | Result |
|---|---:|
| Structured C1, five-run median, temperature 0 | **64.053 tok/s** |
| Structured C1, five-run median, temperature 1.0/top-p 0.95 | **62.637 tok/s** |
| Open-ended prose, five-run median | **25.059 tok/s** |
| C4 median active stream | **40.958 tok/s/stream** |
| C4 summed active-stream convention | **181.944 tok/s** |
| C4 strict submission-to-completion wall | **91.475 tok/s** |

**C4 disclosure:** 181.944 tok/s is the sum of each stream's active decode rate, matching MiaAI's reporting convention. Because the current scheduler stages admission, strict full-batch wall throughput is 91.475 tok/s. Both numbers are published intentionally.

## Speculative depth (K) sweep

Default `num_speculative_tokens` moved from 7 to **5** on 2026-09-03. Spec decode is lossless; K only changes speed. Temperature 0, thinking off, 400 tokens, C1:

| K | Structured | Prose | Code |
|--:|---:|---:|---:|
| 4 | 48.6 tok/s | 28.8 | 37.7 |
| **5 (default)** | 53.5 | **29.9** | **40.1** |
| 6 | 59.2 | 29.0 | 37.5 |
| 7 | 63.9 | 25.8 | 38.3 |
| 8 | **66.9** | 23.9 | 38.0 |

Use K=8 for list/JSON-heavy workloads (`ONE_SPARK_K=8`). Raw data in the GitHub repo under `benchmarks/raw/k-sweep-20260903`.

## Long-context results

| Prompt | Cold prefill | Cold TTFT | Warm TTFT |
|---:|---:|---:|---:|
| 8K | **786.4 tok/s** | 10.175 s | 1.775 s |
| 16K | **822.1 tok/s** | 19.464 s | 2.786 s |
| 100K | **845.7 tok/s** | 118.250 s | 8.842 s |

Every cold request had zero prefix-cache hits; every warm checksum response was correct.

## Quantization Analysis

This deployment uses [Turboderp's GLM-5.3-Flash EXL3 2.05-bpw quant](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw), revision `51058cd551c7e570d87bd32a4adee720edce2349`. The exact checkpoint is **85.23 GB (79.38 GiB)**.

An [independent full-vocabulary measurement](https://github.com/malaiwah/quant-fidelity-suite/blob/794d80fa79db4d30cd0fa8140a07c665dd363251/registry/receipts/malaiwah/stream-turbo-2.05bpw-kld.json) of this exact revision against BF16 teacher logits reported:

| Quant | Size | Top-1 agreement | Mean KLD | Scored positions |
|---|---:|---:|---:|---:|
| Turboderp EXL3 2.05 | 85.23 GB | **88.92%** | **0.121638** | **51,175** |

The measurement used 25 windows, the full 154,880-token vocabulary, teacher forcing, FP64 accumulation, and two cold runs with identical results. The checkpoint was quantized from the official FP8 release; the measurement reference is BF16. Machine-readable summary: [`quantization-analysis.json`](https://github.com/gitcommit90/glm-5.3-one-spark/blob/main/benchmarks/quality/quantization-analysis.json).

## Links

- **GitHub source and instructions:** `https://github.com/gitcommit90/glm-5.3-one-spark`
- **Hugging Face Collection:** https://huggingface.co/collections/gitcommit90/glm-53-one-spark-6a98b70df9981ae425acbc05
- **Prebuilt ARM64 runtime image:** `ghcr.io/gitcommit90/glm-5.3-one-spark:general23`
- **Target checkpoint:** https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw
- **DFlash2 checkpoint:** https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- **Z.ai base model:** https://huggingface.co/zai-org/GLM-5.3-Flash

## Credits and licensing

The runtime is derived from Mia's AI Lab's MIT-licensed two-Spark recipe and substantially adapted for TP1, full-model 2.05-bpw `mul1` EXL3, ARM64/SM121, and one-Spark memory limits.

Credits: Z.ai / GLM-5 Team, Turboderp and ExLlamaV3, Inco AI, Mia's AI Lab, and the vLLM contributors.

**DFlash2 is CC BY-NC-ND 4.0 for research/evaluation.** It is not bundled here. Commercial users must obtain appropriate licensing from Inco AI.
