#!/usr/bin/env python3
"""Regression tests for the vLLM #57477 + #58454 k-pool tail backport overlay.

CPU-only: pinned-fixture patching, idempotence, fail-closed refusal of a half-patched file,
and the tail-ring size. Inside the image it also preflights the installed vLLM files.
The kernel-level GPU tests are vLLM's own (tests/kernels/test_kpool_decode_update_batched.py
on vLLM main), which fail on the unpatched base image and pass once this overlay is applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (HERE / "patch_kpool_vllm_backports.py", ROOT / "overlay" / "patch_kpool_vllm_backports.py")
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
import patch_kpool_vllm_backports as bp  # noqa: E402


def _fixture(sites) -> str:
    """A file containing every pinned anchor once, separated by filler lines."""
    return "\n# filler\n".join(old for _name, old, _new in sites)


def _check_file_roundtrip(sites) -> None:
    source = _fixture(sites)
    patched, action = bp.prepare(source, sites)
    assert action == "patched"
    assert bp.is_patched(patched, sites)
    again, action = bp.prepare(patched, sites)
    assert action == "already present" and again == patched


def test_fixture_roundtrip_kpool_compress():
    _check_file_roundtrip(bp.SITES[bp.KPC])


def test_fixture_roundtrip_attention():
    _check_file_roundtrip(bp.SITES[bp.ATTN])


def test_half_patched_file_is_refused():
    sites = bp.SITES[bp.KPC]
    patched, _ = bp.prepare(_fixture(sites), sites)
    name, old, new = sites[7]  # decode phys slot
    broken = patched.replace(new, old)
    try:
        bp.prepare(broken, sites)
    except ValueError as exc:
        assert "half-patched" in str(exc)
    else:
        raise AssertionError("half-patched file was accepted")


def test_drifted_anchor_is_refused():
    sites = bp.SITES[bp.KPC]
    source = _fixture(sites).replace("phys_slot = safe_pos % POOL_SIZE", "phys_slot = safe_pos % 4")
    try:
        bp.prepare(source, sites)
    except ValueError as exc:
        assert "drifted" in str(exc)
    else:
        raise AssertionError("drifted anchor was accepted")


def _ring_fn():
    namespace: dict = {}
    helper = bp.ATTN_HELPER_NEW.split("class Glm5NextTailCache")[0]
    exec(helper, namespace)  # noqa: S102 - the helper text this overlay injects
    return namespace["_glm53_tail_ring"]


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_tail_ring_matches_upstream_sizes():
    ring = _ring_fn()
    cache = _NS(block_size=640)
    # (num_speculative_tokens, ring) pairs from vLLM #58454's test, plus DFlash2 K=5.
    for num_spec, expected in [(0, 4), (1, 8), (4, 8), (5, 16), (7, 16), (13, 32)]:
        cfg = _NS(speculative_config=_NS(num_speculative_tokens=num_spec))
        assert ring(4, cfg, cache) == expected, num_spec
    assert ring(4, _NS(speculative_config=None), _NS(block_size=7168)) == 4


def test_tail_ring_must_divide_attention_block():
    ring = _ring_fn()
    try:
        ring(4, _NS(speculative_config=_NS(num_speculative_tokens=5)), _NS(block_size=1000))
    except AssertionError:
        return
    raise AssertionError("ring that does not divide block_size was accepted")


def test_installed_files_preflight():
    """Inside the image: the pinned anchors match the installed vLLM (or it is already patched)."""
    if not bp.KPC.is_file():
        return  # not in the image
    for target, sites in bp.SITES.items():
        _text, action = bp.prepare(target.read_text(), sites)
        assert action in ("patched", "already present"), target


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} k-pool backport tests passed")
