from pathlib import Path
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/multimodal.py')
s=p.read_text()
old_map='''            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
'''
new_map='''            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".attn.q_proj.": (".attn.qkv.", "q"),
            ".attn.k_proj.": (".attn.qkv.", "k"),
            ".attn.v_proj.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
'''
if new_map not in s:
    if s.count(old_map)!=1: raise RuntimeError(f'vision mapping block count={s.count(old_map)}')
    s=s.replace(old_map,new_map)
old='''    def load_weights(self, weights) -> set[str]:
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
new='''    def load_weights(self, weights) -> set[str]:
        # Turboderp full EXL3 vision shards contain split packed q/k/v tensors
        # plus a redundant native fused qkv weight/bias. The registered qkv
        # linear is EXL3 (resolved from the split ledger), so retain/map the
        # packed split representation and discard only the native fused copy.
        redundant_native = (".attn.qkv.weight", ".attn.qkv.bias")
        weights = ((name, weight) for name, weight in weights if not name.endswith(
            redundant_native
        ))
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
if new not in s:
    if s.count(old)!=1: raise RuntimeError(f'general18 load method count={s.count(old)}')
    s=s.replace(old,new)
compile(s,str(p),'exec')
assert new_map in s and new in s
p.write_text(s)
print('glm53: vision EXL3 split tensors canonical; redundant native fused qkv filtered')
