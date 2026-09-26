#!/usr/bin/env python3
"""Backport two upstream vLLM k-pool tail fixes onto the pinned base image.

The pinned base (``vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...``)
predates both fixes; vLLM ``main`` has them. Applied at image build, like the
other overlays. Fail-closed, idempotent, preflights every pinned anchor before
writing, verifies the exact patched state, atomic per-file replace.

1. vllm-project/vllm#57477 (Jared Wen): the NVIDIA prefill tail-seed kernel
   ``_kpool_tail_seed_kernel`` addressed the tail as dense
   ``[num_blocks, 2, kpool, head_dim]`` blocks. GLM5Next's tail co-owns the
   compressed-indexer storage with pages padded to the indexer page
   (``kv_cache_utils``' ``page_size_padded=idx_page``), and the runner passes a
   strided view. Prefill-seeded K/gate then miss the block the decode kernel
   reads (decode already honours ``tail.stride(0)``/``stride(1)``) and land
   inside another block's page of the shared indexer storage. The seed kernel
   now takes ``TAIL_BLOCK_ELEMS``/``KPOOL_HEAD`` from the tensor's strides.

2. vllm-project/vllm#58454 (Matt Mastracci): with speculative decoding every
   verify token is stashed in the tail before acceptance. With a one-pool ring
   (``block_size == index_kpool``) the drafts behind a rejected pool-completing
   draft overwrite the K/gate its redo reads. The tail block becomes a ring of
   ``index_kpool * next_power_of_2(cdiv(index_kpool + num_speculative_tokens,
   index_kpool))`` slots (16 for kpool=4, 5 draft tokens); seed and decode
   address ``pos % RING`` while pool completion still spans ``POOL_SIZE``.
   The ring must divide the attention block (asserted), so it does not change
   the scheduler block or prefix-cache granularity.

Adapted to this image's layout (tail spec: ``num_kv_heads=1``,
``head_size=2*head_dim``; speculative token count from
``vllm_config.speculative_config``); upstream's ``common/attention.py`` differs.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_VLLM_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
KPC = SITE / "models/glm5next/nvidia/ops/kpool_compress.py"
ATTN = SITE / "models/glm5next/nvidia/attention.py"

# --- kpool_compress.py: seed kernel (#57477 strides + #58454 ring) ----------------
SEED_SIG = (
    "    KPOOL: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n):\n"
    "    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.\n"
)
SEED_SIG_NEW = (
    "    KPOOL: tl.constexpr,\n"
    "    TAIL_BLOCK_ELEMS: tl.constexpr,  # [glm53-kpool-backport] tail.stride(0) (#57477)\n"
    "    KPOOL_HEAD: tl.constexpr,  # tail.stride(1)\n"
    "    RING: tl.constexpr,  # tail ring slots (#58454)\n"
    "    BLOCK_D: tl.constexpr,\n):\n"
    "    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.\n"
)
SEED_BLK = "    blk = t // KPOOL  # t >= 0 here, so trunc == floor\n"
SEED_BLK_NEW = "    blk = t // RING  # t >= 0 here, so trunc == floor\n"
SEED_AHEAD = "    if ahead >= 0 and ahead // KPOOL == blk:\n"
SEED_AHEAD_NEW = "    if ahead >= 0 and ahead // RING == blk:\n"
SEED_BASE = "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n"
SEED_BASE_NEW = "    base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM\n"
SEED_SCORE = "    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)\n"
SEED_SCORE_NEW = "    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)\n"
SEED_CALL = (
    "        KPOOL=kpool,\n        BLOCK_D=triton.next_power_of_2(head_dim),\n    )\n\n\n"
    "# ---------------------------------------------------------------------------\n"
    "# kpool_decode_update_and_maybe_write_cache_batched"
)
SEED_CALL_NEW = (
    "        KPOOL=kpool,\n"
    "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n"
    "        KPOOL_HEAD=tail_kv_cache.stride(1),\n"
    "        RING=tail_kv_cache.shape[2],\n"
    "        BLOCK_D=triton.next_power_of_2(head_dim),\n    )\n\n\n"
    "# ---------------------------------------------------------------------------\n"
    "# kpool_decode_update_and_maybe_write_cache_batched"
)

# --- kpool_compress.py: decode kernel (#58454 ring) ---------------------------------
DEC_SIG = "    POOL_SIZE: tl.constexpr,\n    TAIL_BLOCK_ELEMS: tl.constexpr,\n"
DEC_SIG_NEW = (
    "    POOL_SIZE: tl.constexpr,\n"
    "    RING: tl.constexpr,  # [glm53-kpool-backport] tail ring slots >= POOL_SIZE (#58454)\n"
    "    TAIL_BLOCK_ELEMS: tl.constexpr,\n"
)
DEC_PHYS_SLOT = "        phys_slot = safe_pos % POOL_SIZE\n"
DEC_PHYS_SLOT_NEW = "        phys_slot = safe_pos % RING\n"
DEC_BLOCK = "        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE\n"
DEC_BLOCK_NEW = "        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING\n"
_LOOP = (
    "            for pool_slot in tl.static_range(0, POOL_SIZE):\n"
    "                is_current = pool_slot == slot\n"
    "                phys = (pool_logical_start + pool_slot) % {mod}\n"
)
DEC_MAX = "            max_score = tl.full((BLOCK_D,), -float(\"inf\"), tl.float32)\n" + _LOOP.format(mod="POOL_SIZE")
DEC_MAX_NEW = "            max_score = tl.full((BLOCK_D,), -float(\"inf\"), tl.float32)\n" + _LOOP.format(mod="RING")
DEC_ACC = "            denom = tl.full((BLOCK_D,), 0.0, tl.float32)\n" + _LOOP.format(mod="POOL_SIZE")
DEC_ACC_NEW = "            denom = tl.full((BLOCK_D,), 0.0, tl.float32)\n" + _LOOP.format(mod="RING")
DEC_ASSERT = "    assert tail_kv_cache.shape[2] == pool_size\n"
DEC_ASSERT_NEW = (
    "    ring = tail_kv_cache.shape[2]  # [glm53-kpool-backport] (#58454)\n"
    "    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)\n"
)
DEC_CALL = "        POOL_SIZE=pool_size,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n"
DEC_CALL_NEW = "        POOL_SIZE=pool_size,\n        RING=ring,\n        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n"

# --- attention.py: tail spec becomes a ring (#58454) --------------------------------
ATTN_HELPER = "class Glm5NextTailCache(DeepseekV32IndexerCache):\n"
ATTN_HELPER_NEW = (
    "def _glm53_tail_ring(index_kpool: int, vllm_config, cache_config) -> int:\n"
    "    \"\"\"[glm53-kpool-backport] Tail ring slots (vllm-project/vllm#58454).\n\n"
    "    Drafts are stashed before acceptance; with a one-pool ring the drafts behind a\n"
    "    rejected pool-completing draft overwrite the keys its redo reads.\n"
    "    \"\"\"\n"
    "    spec = getattr(vllm_config, \"speculative_config\", None)\n"
    "    num_spec = int(getattr(spec, \"num_speculative_tokens\", 0) or 0)\n"
    "    groups = -(-(index_kpool + num_spec) // index_kpool)\n"
    "    ring = index_kpool * (1 << (groups - 1).bit_length())\n"
    "    assert cache_config.block_size % ring == 0, (\n"
    "        f\"Glm5NextTailCache: cache_config.block_size ({cache_config.block_size}) \"\n"
    "        f\"must be a multiple of the tail ring ({ring})\"\n"
    "    )\n"
    "    return ring\n\n\n"
    "class Glm5NextTailCache(DeepseekV32IndexerCache):\n"
)
ATTN_SPEC = "        return KpoolTailSpec(\n            block_size=self._index_kpool,\n"
ATTN_SPEC_NEW = (
    "        ring = _glm53_tail_ring(self._index_kpool, vllm_config, self.cache_config)\n"
    "        return KpoolTailSpec(\n            block_size=ring,\n"
)
ATTN_SW = "            sliding_window=self._index_kpool,\n"
ATTN_SW_NEW = "            sliding_window=ring,\n"

SITES = {
    KPC: (
        ("seed signature", SEED_SIG, SEED_SIG_NEW),
        ("seed block", SEED_BLK, SEED_BLK_NEW),
        ("seed ahead", SEED_AHEAD, SEED_AHEAD_NEW),
        ("seed base", SEED_BASE, SEED_BASE_NEW),
        ("seed score store", SEED_SCORE, SEED_SCORE_NEW),
        ("seed launch", SEED_CALL, SEED_CALL_NEW),
        ("decode signature", DEC_SIG, DEC_SIG_NEW),
        ("decode phys slot", DEC_PHYS_SLOT, DEC_PHYS_SLOT_NEW),
        ("decode block", DEC_BLOCK, DEC_BLOCK_NEW),
        ("decode max loop", DEC_MAX, DEC_MAX_NEW),
        ("decode acc loop", DEC_ACC, DEC_ACC_NEW),
        ("decode ring assert", DEC_ASSERT, DEC_ASSERT_NEW),
        ("decode launch", DEC_CALL, DEC_CALL_NEW),
    ),
    ATTN: (
        ("tail ring helper", ATTN_HELPER, ATTN_HELPER_NEW),
        ("tail spec block_size", ATTN_SPEC, ATTN_SPEC_NEW),
        ("tail spec sliding_window", ATTN_SW, ATTN_SW_NEW),
    ),
}
MARK = "[glm53-kpool-backport]"


def is_patched(text: str, sites) -> bool:
    """Exact post-state: every replacement present once."""
    return all(text.count(new) == 1 for _name, _old, new in sites)


def prepare(source: str, sites) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns (text, action)."""
    if MARK in source:
        if not is_patched(source, sites):
            raise ValueError("partial/inconsistent k-pool backport -- refusing to touch a half-patched file")
        return source, "already present"
    out = source
    for name, old, new in sites:
        n = out.count(old)
        if n != 1:
            raise ValueError(f"pinned anchor '{name}' drifted (found {n}, expected 1)")
        out = out.replace(old, new, 1)
    if not is_patched(out, sites):
        raise ValueError("k-pool backport post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-backport.tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{target.stem}*.pyc"):
            pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]
    planned = []
    for target, sites in SITES.items():  # preflight everything before writing anything
        if not target.is_file():
            raise SystemExit(f"missing {target}")
        source = target.read_text()
        try:
            text, action = prepare(source, sites)
        except ValueError as exc:
            raise SystemExit(f"{target.name}: k-pool backport preflight failed: {exc}") from exc
        compile(text, str(target), "exec")
        planned.append((target, source, text, action))
    for target, source, text, action in planned:
        if not preflight_only and text != source:
            replace_file(target, text)
            clear_pyc(target)
        print(f"{target.name}: k-pool backport (vLLM #57477 + #58454) {action}"
              + (" [preflight only]" if preflight_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
