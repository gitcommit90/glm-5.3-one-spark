# SparkRun recipe

`glm-5.3-flash-exl3-2.05bpw-dflash2-vllm-gitcommit90.yaml` packages the
validated General23 configuration as a one-node SparkRun workload.

The recipe pins the runtime image, target checkpoint, and DFlash2 checkpoint to
immutable digests/revisions. It downloads weights from their original
publishers and does not redistribute them.

```bash
sparkrun show sparkrun/glm-5.3-flash-exl3-2.05bpw-dflash2-vllm-gitcommit90.yaml
sparkrun run sparkrun/glm-5.3-flash-exl3-2.05bpw-dflash2-vllm-gitcommit90.yaml --solo
```

Review the DFlash2 checkpoint's CC BY-NC-ND 4.0 noncommercial
research/evaluation terms before running this recipe.
