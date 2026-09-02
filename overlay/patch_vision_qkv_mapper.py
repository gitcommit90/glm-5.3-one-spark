from pathlib import Path

p = Path('/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/multimodal.py')
s = p.read_text()
repls = {
    '".attn.q_proj.": (".attn.qkv_proj.", "q"),': '".attn.q_proj.": (".attn.qkv.", "q"),',
    '".attn.k_proj.": (".attn.qkv_proj.", "k"),': '".attn.k_proj.": (".attn.qkv.", "k"),',
    '".attn.v_proj.": (".attn.qkv_proj.", "v"),': '".attn.v_proj.": (".attn.qkv.", "v"),',
}
changed = 0
for old, new in repls.items():
    if old in s:
        if s.count(old) != 1:
            raise RuntimeError(f'unexpected duplicate mapping: {old}')
        s = s.replace(old, new)
        changed += 1
    elif new not in s:
        raise RuntimeError(f'mapping target not found: {old}')
p.write_text(s)
compile(s, str(p), 'exec')
for new in repls.values():
    assert s.count(new) == 1
print(f'glm53: vision split q/k/v mapped to registered qkv module ({changed} changed)')
