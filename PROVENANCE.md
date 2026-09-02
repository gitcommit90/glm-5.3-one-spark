# Provenance and scope of modifications

GLM-5.3 One-Spark began from Mia's AI Lab repository `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` at commit:

```text
c707598ebcf02fd827d079a7c47e785069425efe
```

The Git history in this repository preserves that ancestry. At the release-preparation audit, 63 inherited files were still byte-identical to that pinned upstream snapshot. The principal inherited files modified for One-Spark were:

- `Dockerfile`
- `overlay/exl3.py`
- `overlay/patch_exl3_ext_aarch64.py`

Important additive One-Spark files include:

- `overlay/patch_full_exl3_loader.py`
- `overlay/patch_exl3_bf16_output.py`
- `overlay/patch_vision_*.py`
- `overlay/patch_qkv_loader_diag.py`
- `tests/test_exl3_mul1_fused_diff.py`
- `scripts/serve-one-spark.sh`

The packaged target weights are not modified by this project. They are downloaded from Turboderp's official 2.05-bpw branch. The DFlash2 weights are not modified or redistributed and remain hosted by Inco AI.

The runtime also contains source-exact behavioral backports from vLLM PRs #52805 and #53046, identified in `overlay/patch_xgrammar_termination.py`, and a vLLM-derived `qwen3_dflash2.py` marked Apache-2.0.
