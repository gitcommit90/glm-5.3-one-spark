from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py')
s=p.read_text()
old='''        ys = [inner.forward(x.contiguous().half(), {}, out_dtype=x.dtype)
              for inner in layer._exl3_linear_inners]
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)
'''
new='''        # ExLlamaV3 hgemm accepts fp16/fp32 output buffers, not bf16. vLLM
        # runs GLM in bf16, so compute packed linears into fp16 and cast the
        # assembled result back before bias/residual math.
        ys = [inner.forward(x.contiguous().half(), {}, out_dtype=torch.float16)
              for inner in layer._exl3_linear_inners]
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)
        if y.dtype != x.dtype:
            y = y.to(x.dtype)
'''
if new not in s:
    if s.count(old)!=1: raise RuntimeError(f'EXL3 apply target count={s.count(old)}')
    s=s.replace(old,new)
compile(s,str(p),'exec'); p.write_text(s)
print('glm53: EXL3 packed linear bf16 output bridge installed')
