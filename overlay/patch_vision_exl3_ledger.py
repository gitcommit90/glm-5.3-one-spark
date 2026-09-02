from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py')
s=p.read_text()
old='''        self.tensor_storage = ledger
        logger.info("EXL3 loaded per-tensor ledger: %d entries from %s", len(ledger), path)
'''
new='''        # ExLlamaV3's exported ledger omits the quantized vision tower even
        # though the safetensors contain packed EXL3 q/k/v, attention output,
        # and MLP matrices. Recover those entries from safetensors headers so
        # vLLM constructs packed parameters instead of dense weight slots.
        from glob import glob
        from safetensors import safe_open
        vision_parts: dict[str, dict[str, dict[str, Any]]] = {}
        required = {"trellis", "suh", "svh", self.codebook}
        for shard_path in sorted(glob(os.path.join(model_name, "*.safetensors"))):
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for tensor_name in sf.keys():
                    if not tensor_name.startswith("model.visual."):
                        continue
                    base, suffix = tensor_name.rsplit(".", 1)
                    if suffix not in required:
                        continue
                    shape = list(sf.get_slice(tensor_name).get_shape())
                    vision_parts.setdefault(base, {})[tensor_name] = {"shape": shape}
        recovered = 0
        for base, stored in vision_parts.items():
            suffixes = {name.rsplit(".", 1)[-1] for name in stored}
            if required.issubset(suffixes):
                ledger[base] = {
                    "stored_tensors": stored,
                    "quant_format": "exl3",
                    "bits_per_weight": int(self.bits),
                }
                recovered += 1
        if recovered == 0:
            raise ValueError("EXL3 vision ledger recovery found no packed linears")
        self.tensor_storage = ledger
        logger.info(
            "EXL3 loaded per-tensor ledger: %d entries (%d recovered vision) from %s",
            len(ledger), recovered, path,
        )
'''
if new not in s:
    if s.count(old)!=1: raise RuntimeError(f'ledger target count={s.count(old)}')
    s=s.replace(old,new)
compile(s,str(p),'exec'); p.write_text(s)
print('glm53: EXL3 vision ledger recovery installed')
