#!/usr/bin/env python3
from pathlib import Path
model=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py')
s=model.read_text()
marker='''            # Handle FP8 indexer WK: dequantize to BF16 for fusion with
'''
insert='''            # Full EXL3 checkpoints keep the KDA short-convolution kernel
            # fused as q|k|v, while this runtime represents it as three
            # ColumnParallelLinear parameters. Split before ordinary loading.
            if name.endswith(".conv1d.weight") and name not in params_dict:
                if loaded_weight.shape[0] % 3:
                    raise RuntimeError(
                        f"GLM5Next fused conv1d rows are not divisible by 3: "
                        f"{name} {tuple(loaded_weight.shape)}"
                    )
                chunks = loaded_weight.chunk(3, dim=0)
                for label, chunk in zip(("q", "k", "v"), chunks):
                    target = name.replace(
                        ".conv1d.weight", f".{label}_conv1d.weight"
                    )
                    if target not in params_dict:
                        raise KeyError(
                            f"GLM5Next fused conv1d target missing: {target}"
                        )
                    param = params_dict[target]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, chunk, **kwargs)
                    loaded_params.add(target)
                continue

'''
if 'GLM5Next fused conv1d rows' not in s:
 if s.count(marker)!=1: raise SystemExit(f'model marker count {s.count(marker)}')
 model.write_text(s.replace(marker,insert+marker))
 print('full EXL3 fused conv loader installed')
else: print('full EXL3 fused conv loader already present')

kda=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py')
s=kda.read_text()
old='''        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
'''
new='''        saved_quant_config = vllm_config.quant_config
        # FP8 checkpoints keep KDA projections native, but full-model EXL3
        # checkpoints carry packed qkv/o projections in these layers. Preserve
        # the EXL3 config; its per-tensor ledger returns UnquantizedLinearMethod
        # for every native KDA tensor.
        keep_exl3 = (
            saved_quant_config is not None
            and saved_quant_config.get_name() == "exl3"
        )
        if not keep_exl3:
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
'''
if 'keep_exl3 = (' not in s:
 if s.count(old)!=1: raise SystemExit(f'kda marker count {s.count(old)}')
 kda.write_text(s.replace(old,new))
 print('full EXL3 KDA quant guard installed')
else: print('full EXL3 KDA quant guard already present')

# Adapt KDA projection fusion for full-model EXL3: packed QKV remains one
# EXL3 linear, while native b/f_a/g_a retain a small merged native GEMM.
kda=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py')
s=kda.read_text()
old='''        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )
'''
new='''        # Full-model EXL3 stores QKV as one packed matrix, while b/f_a/g_a
        # remain native. Keep that split instead of forcing mixed quant/native
        # shards into one fused LinearBase.
        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            3 * projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.in_proj_bfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                self.num_heads,  # b (shard 0)
                self.head_dim,  # f_a (shard 1, replicated)
                self.head_dim,  # g_a (shard 2, replicated)
            ],
            replicated_shard_ids=(1, 2),
            tp_size=self.tp_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.in_proj_bfg_a",
        )
'''
if 'self.in_proj_bfg_a = ' not in s:
 if s.count(old)!=1: raise SystemExit(f'KDA projection init marker count {s.count(old)}')
 s=s.replace(old,new)
oldf='''        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )
'''
newf='''        qkv = self.qkv_proj(hidden_states)[0]
        bfg = self.in_proj_bfg_a(hidden_states)[0]
        beta_raw, f_a, g_a = bfg.split(
            [self.local_num_heads, self.head_dim, self.head_dim], dim=-1
        )
'''
if 'bfg = self.in_proj_bfg_a' not in s:
 if s.count(oldf)!=1: raise SystemExit(f'KDA forward marker count {s.count(oldf)}')
 s=s.replace(oldf,newf)
kda.write_text(s)
print('full EXL3 KDA split QKV + native BFG installed')

model=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py')
s=model.read_text()
oldm='''            # KDA: merge q, k, v, b, f_a, g_a projections into one GEMM
            (".in_proj_qkvbfg_a", ".q_proj", 0),
            (".in_proj_qkvbfg_a", ".k_proj", 1),
            (".in_proj_qkvbfg_a", ".v_proj", 2),
            (".in_proj_qkvbfg_a", ".b_proj", 3),
            (".in_proj_qkvbfg_a", ".f_a_proj", 4),
            (".in_proj_qkvbfg_a", ".g_a_proj", 5),
'''
newm='''            # KDA full EXL3: qkv_proj loads directly; merge native B/F/G.
            (".in_proj_bfg_a", ".b_proj", 0),
            (".in_proj_bfg_a", ".f_a_proj", 1),
            (".in_proj_bfg_a", ".g_a_proj", 2),
'''
if '.in_proj_bfg_a", ".b_proj"' not in s:
 if s.count(oldm)!=1: raise SystemExit(f'model KDA mapping marker count {s.count(oldm)}')
 s=s.replace(oldm,newm)
model.write_text(s)
print('full EXL3 KDA loader mapping installed')

# MLA projections are native in Mia's FP8 source checkpoint but packed in the
# full 2.05 EXL3 checkpoint. Preserve EXL3 only; ledger keeps native tensors native.
model=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py')
s=model.read_text()
old='''                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
'''
new='''                quant_config=(
                    quant_config
                    if quant_config is not None and quant_config.get_name() == "exl3"
                    else None
                ),
                prefix=f"{prefix}.self_attn",
'''
if 'if quant_config is not None and quant_config.get_name() == "exl3"' not in s:
 if s.count(old)!=1: raise SystemExit(f'MLA quant guard marker count {s.count(old)}')
 model.write_text(s.replace(old,new))
 print('full EXL3 MLA quant guard installed')
else: print('full EXL3 MLA quant guard already present')

# Vision tower is BF16 in Mia's FP8 checkpoint but full-model EXL3 in the
# Turboderp 2.05 target. Preserve EXL3 and teach the visual loader q_proj names.
model=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py')
s=model.read_text()
old='''                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
'''
new='''                quant_config=(
                    vllm_config.quant_config
                    if vllm_config.quant_config is not None
                    and vllm_config.quant_config.get_name() == "exl3"
                    else None
                ),
                prefix=maybe_prefix(prefix, "visual"),
'''
if 'vllm_config.quant_config.get_name() == "exl3"' not in s:
 if s.count(old)!=1: raise SystemExit(f'vision quant marker count {s.count(old)}')
 model.write_text(s.replace(old,new))
 print('full EXL3 vision quant guard installed')
else: print('full EXL3 vision quant guard already present')

mm=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/multimodal.py')
s=mm.read_text()
old='''            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
'''
new='''            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".attn.q_proj.": (".attn.qkv_proj.", "q"),
            ".attn.k_proj.": (".attn.qkv_proj.", "k"),
            ".attn.v_proj.": (".attn.qkv_proj.", "v"),
'''
# Patch only Glm5Next mapper, first occurrence.
if '.attn.q_proj.": (".attn.qkv_proj."' not in s:
 if s.count(old)<1: raise SystemExit('vision mapper marker missing')
 s=s.replace(old,new,1)
 mm.write_text(s)
 print('full EXL3 vision qkv mapper installed')
else: print('full EXL3 vision qkv mapper already present')

# _mark_tower_model temporarily clears vllm_config.quant_config. Capture EXL3
# before entering the context so the full packed vision tower is constructed.
model=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py')
s=model.read_text()
old='''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
'''
new='''        vision_quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
'''
if 'vision_quant_config = vllm_config.quant_config' not in s:
 if s.count(old)!=1: raise SystemExit(f'vision capture marker count {s.count(old)}')
 s=s.replace(old,new)
old2='''                quant_config=(
                    vllm_config.quant_config
                    if vllm_config.quant_config is not None
                    and vllm_config.quant_config.get_name() == "exl3"
                    else None
                ),
'''
new2='''                quant_config=(
                    vision_quant_config
                    if vision_quant_config is not None
                    and vision_quant_config.get_name() == "exl3"
                    else None
                ),
'''
if old2 in s: s=s.replace(old2,new2,1)
model.write_text(s)
print('full EXL3 vision config capture installed')
