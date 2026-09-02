from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py')
s=p.read_text(); a=s.index('class QKVParallelLinear'); b=s.index('class MinimaxM3QKVParallelLinearWithIndexer',a); part=s[a:b]
old='''            if param is None and name == "bias":
                continue
            param.weight_loader(param, loaded_weight, shard_id)
'''
new='''            if param is None and name == "bias":
                continue
            if param is self:
                raise AttributeError(
                    f"QKVParallelLinear missing parameter name={name!r} "
                    f"shard_id={shard_id!r}; registered={tuple(self._parameters)}"
                )
            param.weight_loader(param, loaded_weight, shard_id)
'''
if new not in part:
    if part.count(old)!=1: raise RuntimeError(f'QKV diagnostic target count={part.count(old)}')
    part=part.replace(old,new)
s=s[:a]+part+s[b:]
compile(s,str(p),'exec'); p.write_text(s)
print('glm53: QKV missing-parameter diagnostic installed')
