#!/usr/bin/env python3
"""Stub AVX CPU targets so ExLlamaV3's extension compiles on aarch64/GB10."""

from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")
(root / "avx2_target.cpp").write_text(
    '#include "avx2_target.h"\nbool is_avx2_supported() { return false; }\nbool is_f16c_supported() { return false; }\n'
)
(root / "avx512_target.cpp").write_text(
    '#include "avx512_target.h"\nbool is_avx512_supported() { return false; }\n'
)
(root / "parallel/all_reduce_cpu_avx2.cpp").write_text(
    """#include "all_reduce_cpu_avx2.h"
#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
(root / "parallel/all_reduce_cpu_avx512.cpp").write_text(
    """#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
for hdr, name in (("avx2_target.h", "avx2"), ("avx512_target.h", "avx512")):
    guard = name.upper()
    (root / hdr).write_text(
        "#pragma once\n"
        f"bool is_{name}_supported();\n"
        + ("bool is_f16c_supported();\n" if name == "avx2" else "")
        + f"#define {guard}_TARGET\n"
        f"#define {guard}_TARGET_OPTIONAL\n"
    )
print(f"aarch64 EXL3 CPU-target stubs written in {root}")

# v1.4.5 adds x86-only CPU mul1 MoE sources. GPU fused MoE is required on
# Grace/Blackwell; CPU expert offload is not, so provide ABI-complete fail-closed
# stubs rather than compiling AVX2/AVX-512 intrinsics on aarch64.
moe_mul1 = root / "cpu/moe_mul1.cpp"
if moe_mul1.exists():
    original = moe_mul1.read_text()
    if not original.startswith("#if defined(__aarch64__)"):
        stub = r'''#if defined(__aarch64__)
#include "moe_mul1.h"
#include <torch/extension.h>
int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    int64_t, double, int64_t) {
    TORCH_CHECK(false, "EXL3 CPU MoE offload is unavailable on aarch64");
}
void exl3_moe_cpu_free_layer(int64_t) {}
void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) {
    TORCH_CHECK(false, "EXL3 CPU MoE offload is unavailable on aarch64");
}
void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int) {}
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) {}
void exl3_moe_cpu_set_prof(bool) {}
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
#else
'''
        moe_mul1.write_text(stub + original + "\n#endif\n")

handoff = root / "cpu/moe_handoff.cu"
if handoff.exists():
    text = handoff.read_text()
    old = '''inline void cpu_pause_()\n{\n#ifdef __linux__\n    __builtin_ia32_pause();\n#else\n    _mm_pause();\n#endif\n}'''
    new = '''inline void cpu_pause_()\n{\n#if defined(__aarch64__)\n    asm volatile("yield");\n#elif defined(__linux__)\n    __builtin_ia32_pause();\n#else\n    _mm_pause();\n#endif\n}'''
    if old in text:
        handoff.write_text(text.replace(old, new, 1))
    elif 'asm volatile("yield")' not in text:
        raise RuntimeError("moe_handoff cpu_pause_ source drift")

# v1.4.5 adds x86-only CPU mul1 MoE sources. GPU fused MoE is required on
# Grace/Blackwell; CPU expert offload is not, so provide ABI-complete fail-closed
# stubs rather than compiling AVX2/AVX-512 intrinsics on aarch64.
moe_mul1 = root / "cpu/moe_mul1.cpp"
if moe_mul1.exists():
    original = moe_mul1.read_text()
    if not original.startswith("#if defined(__aarch64__)"):
        stub = r'''#if defined(__aarch64__)
#include "moe_mul1.h"
#include <torch/extension.h>
int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
    int64_t, double, int64_t) {
    TORCH_CHECK(false, "EXL3 CPU MoE offload is unavailable on aarch64");
}
void exl3_moe_cpu_free_layer(int64_t) {}
void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) {
    TORCH_CHECK(false, "EXL3 CPU MoE offload is unavailable on aarch64");
}
void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int) {}
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) {}
void exl3_moe_cpu_set_prof(bool) {}
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
#else
'''
        moe_mul1.write_text(stub + original + "\n#endif\n")

handoff = root / "cpu/moe_handoff.cu"
if handoff.exists():
    text = handoff.read_text()
    old = '''inline void cpu_pause_()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}'''
    new = '''inline void cpu_pause_()
{
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__linux__)
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}'''
    if old in text:
        handoff.write_text(text.replace(old, new, 1))
    elif 'asm volatile("yield")' not in text:
        raise RuntimeError("moe_handoff cpu_pause_ source drift")

all_reduce = root / "parallel/all_reduce_cpu.cu"
if all_reduce.exists():
    text = all_reduce.read_text()
    old = """                #ifdef __linux__
                    __builtin_ia32_pause();
                #else
                    _mm_pause();
                #endif"""
    new = """                #if defined(__aarch64__)
                    asm volatile(\"yield\");
                #elif defined(__linux__)
                    __builtin_ia32_pause();
                #else
                    _mm_pause();
                #endif"""
    if old in text:
        all_reduce.write_text(text.replace(old, new, 1))
    elif 'asm volatile("yield")' not in text:
        raise RuntimeError("all_reduce_cpu pause source drift")
