# Third-party notices

This document is a practical attribution inventory, not legal advice.

## Mia's AI Lab — GLM-5.3-Flash EXL3 2x DGX Sparks

- Source: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- Pinned basis: `c707598ebcf02fd827d079a7c47e785069425efe`
- License: MIT
- Copyright: Copyright (c) 2026 Mia's AI Lab
- Use here: substantial derived recipe, overlays, tests, documentation concepts, and benchmark convention

The original MIT notice is retained in `LICENSES/MIAAI-MIT.txt`. The original upstream README is retained at `docs/MIA-UPSTREAM-README.md`.

## Z.ai / GLM-5 Team — GLM-5.3-Flash

- Model: https://huggingface.co/zai-org/GLM-5.3-Flash
- License shown by the source checkpoint: MIT
- Use here: base architecture, tokenizer, configuration, model and vision behavior
- Technical report: https://arxiv.org/abs/2602.15763

The model weights are not distributed by this repository.

## Turboderp — GLM-5.3-Flash EXL3 and ExLlamaV3

- Quant: https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/2.05bpw
- ExLlamaV3: https://github.com/turboderp-org/exllamav3
- ExLlamaV3 pin: `e648f1a131365aae15920073e761a3fa5a527654`
- License: MIT
- Copyright: Copyright (c) 2025 Turboderp
- Use here: unmodified 2.05-bpw target checkpoint; modified ARM64/SM121 build of EXL3 kernels

The target model weights are downloaded from their original repository rather than mirrored here.

## Inco AI — GLM-5.3-Flash DFlash2

- Model: https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- Pin: `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- License: CC BY-NC-ND 4.0, research/evaluation
- Use here: unmodified speculative-decoding draft checkpoint

This repository does not bundle or modify the DFlash2 weights. Commercial users must contact Inco AI for licensing.

Requested citation:

```bibtex
@misc{inco2026dflash2,
  title = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year = {2026},
  month = {August},
  url = {https://inco.ai/blog/dflash2/}
}
```

Original DFlash citation:

```bibtex
@inproceedings{chen2026dflash,
  title = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026}
}
```

## malaiwah — quant-fidelity-suite

- Source: https://github.com/malaiwah/quant-fidelity-suite
- Exact-checkpoint receipt: https://github.com/malaiwah/quant-fidelity-suite/blob/794d80fa79db4d30cd0fa8140a07c665dd363251/registry/receipts/malaiwah/stream-turbo-2.05bpw-kld.json
- Use here: independent full-vocabulary KLD and top-1 measurement for the exact 2.05-bpw checkpoint revision

## vLLM contributors

- Source: https://github.com/vllm-project/vllm
- Base-image revision reported by runtime: `487ecf187`
- License: Apache License 2.0
- Use here: serving engine, model implementation, DFlash integration, scheduler and structured-output backports

The Apache 2.0 license is retained at `LICENSES/VLLM-APACHE-2.0.txt`. Files copied or adapted directly from vLLM keep SPDX headers where present.

## NVIDIA and CUDA container components

The build targets NVIDIA DGX Spark and starts from the pinned vLLM ARM64 CUDA image. NVIDIA CUDA/container components remain under their applicable NVIDIA licenses. Redistribution of the prebuilt image must preserve all notices included in the base image.
