from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/multimodal.py')
s=p.read_text()
# This checkpoint contains the native fused vision qkv weight/bias AND EXL3
# split q/k/v companion tensors. The vision module consumes the fused copy.
for proj, shard in (("q", "q"), ("k", "k"), ("v", "v")):
    line=f'            ".attn.{proj}_proj.": (".attn.qkv.", "{shard}"),\n'
    if line in s:
        s=s.replace(line,'')
old='''    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
new='''    def load_weights(self, weights) -> set[str]:
        # Turboderp full EXL3 vision shards contain both a native fused qkv
        # weight/bias and split EXL3 q_proj/k_proj/v_proj companions. This
        # implementation registers the fused QKVParallelLinear; consume that
        # canonical copy and ignore the redundant split representation.
        split = (".attn.q_proj.", ".attn.k_proj.", ".attn.v_proj.")
        weights = ((name, weight) for name, weight in weights if not any(
            part in name for part in split
        ))
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
if new not in s:
    if s.count(old)!=1: raise RuntimeError(f'vision load_weights target count={s.count(old)}')
    s=s.replace(old,new)
compile(s,str(p),'exec')
assert 'redundant split representation' in s
assert '.attn.q_proj.' in s  # filter tuple
assert '".attn.q_proj.": (' not in s
p.write_text(s)
print('glm53: vision dual-layout loader uses fused qkv and filters split EXL3 companions')
