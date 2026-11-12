#!/usr/bin/env python3
"""native-lane-v1 value shapes: the bridge path, a marked compiler, and MLIR.

The candidate native-lane-v1 profile (src/ir/native_lane_profile/mod.rs) admits
three shapes the historical frozen profile refuses, because the downstream consumer
programs need them and a compiler carrying the `nblane01` mark implements them:

  * a store intrinsic's result read as a value (the `w_set` helper returns it):
    the defined 0 that MLIR materialises and the marked compiler writes;
  * full-width `i64 as f64` / `f64 as i64` (round-to-nearest-even / saturating);
  * an `if` whose other branch returns, breaks or continues (lowering forwards
    the reaching branch's value to the merge).

Every case is built twice through `mindc build` -- `--backend native
--native-profile bounded` with MINDC_NATIVE_ELF naming the marked compiler, and
`--backend mlir` -- and both executables must produce the value computed here
independently of either backend (exit status, or 8-byte little-endian stdout
words). The store cases dirty the stack first, so a compiler that leaves a store
result slot unwritten reads stale data and fails (the pre-fix frozen seed returns
19 instead of 17 on `deep_dirty_then_tail`). `main(argc, argv)` is included
because an explicit-layout image starts with the allocator runtime prefix, and
the bridge must find the argc/argv binding at the program entry behind it.

The smoke names its compiler explicitly and refuses to run without one; CI
passes the committed bounded seed, which carries the mark since the reseed:

  MINDC_BIN=target/release/mindc python3 native_lane_value_shapes_smoke.py \\
      --compiler examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf

Both paths are resolved against the invoking directory before use: every build
runs with its case directory as cwd, where a relative path names nothing.

NATIVE_ONLY_CASES are rows for native-only families (no MLIR twin): built
through the stdin protocol (`direct_feed`) and/or the front door (`front_door`),
then run and checked against the row (exit, stdout, objdump/readelf text, a named
refusal: rc 3 and no ELF). Lanes add rows and `.mind` fixtures, never glue.
Rehearsal modes (never a seed move): `--mint-stage2 OUT [--assert-fixed-point]`
mints frozen seed -> stage1 -> stage2 over the bundle WITH the compiler parts and
prints its sha256; `--direct-compile PROG --compiler ELF --out ELF` feeds one
program to a compiler ELF. `--only FAMILY` runs one row family. A
`compiler_driver` row composes its fixture after main.mind + the parts (the minted
unit) so it can call compiler fns; it compiles the whole compiler, so it runs only
when `--only` names its family.

Exit: 0 every case matches; 1 a case differs; 2 BLOCKED (no mindc, no marked
compiler, or a compiler without the mark).
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from _selfhost_sources import direct_compile, driver_compile, mint_stage2  # noqa: E402
from _w8_native import cast_edges_oracle, cast_exprs_oracle, cast_words  # noqa: E402
FIXTURES = HERE / "testdata" / "native_lane_values"
MARK = b"\x48\xb8nblane01"
PRINT_MARK = b"\x48\xb8nbprint1"  # nb_print.mind; without it the native leg must refuse print
# Used only when the variable is unset AND this path is an executable file. The
# LLVM 23 leg installs no LLVM 20, so there the build finds the tools on PATH.
TOOLS = {
    "MLIR_OPT": "/usr/lib/llvm-20/bin/mlir-opt",
    "MLIR_TRANSLATE": "/usr/lib/llvm-20/bin/mlir-translate",
    "CLANG": "/usr/lib/llvm-20/bin/clang",
}
# llvm-mc / llvm-objcopy: only beside $MLIR_OPT when it is set, else the LLVM 20 dir, then PATH (LLVM 23 leg).
LLVM_PATH = str(pathlib.Path(os.environ["MLIR_OPT"]).parent) if os.environ.get("MLIR_OPT") else os.pathsep.join([str(pathlib.Path(TOOLS["MLIR_OPT"]).parent), os.environ.get("PATH", os.defpath)])

DIRTY = (
    "fn dirty(a: i64) -> i64 { let b: i64 = a + 1; let c: i64 = b * 3; let d: i64 = c + b; "
    "let e: i64 = d - 2; let f: i64 = e * e; let g: i64 = f + d; g }\n"
)
ALLOC = "let p: i64 = __mind_alloc(8);"

# (id, source, expected exit status). A store result is the defined value 0.
EXIT_CASES = [
    ("tail_i64", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i64(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 7 + (z - z) }}", 7),
    ("tail_i8", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i8(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 9 + (z - z) }}", 9),
    ("tail_i32", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i32(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 11 + (z - z) }}", 11),
    ("let_init", DIRTY + f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(3); "
     "let x: i64 = __mind_store_i64(p, 77); x + 5 + (z - z) }", 5),
    ("assign", f"fn main() -> i64 {{ {ALLOC} let mut x: i64 = 40; x = __mind_store_i64(p, 77); "
     "x + __mind_load_i64(p) - 70 }", 7),
    ("return_value", "fn w(p: i64, v: i64) -> i64 { return __mind_store_i64(p, v); }\n"
     f"fn main() -> i64 {{ {ALLOC} w(p, 41) + __mind_load_i64(p) }}", 41),
    ("binop", f"fn main() -> i64 {{ {ALLOC} return 13 + 6 * __mind_store_i8(p, 9); }}", 13),
    ("call_arg", "fn id(v: i64) -> i64 { v + 2 }\n"
     f"fn main() -> i64 {{ {ALLOC} id(__mind_store_i64(p, 50)) }}", 2),
    ("nested_store_arg", "fn main() -> i64 { let p: i64 = __mind_alloc(16); "
     "__mind_store_i64(p + 8, __mind_store_i64(p, 33)); "
     "__mind_load_i64(p + 8) + __mind_load_i64(p) }", 33),
    ("if_cond", f"fn main() -> i64 {{ {ALLOC} if __mind_store_i64(p, 5) == 0 {{ return 21; }} "
     "return 3; }", 21),
    ("while_cond", f"fn main() -> i64 {{ {ALLOC} let mut i: i64 = 0; "
     "while __mind_store_i64(p, i) + i < 6 { i = i + 1; } i + __mind_load_i64(p) }", 12),
    ("value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 0; let y: i64 = if c == 1 {{ 5 }} "
     "else { __mind_store_i8(p, 77) }; y + 3 }", 3),
    ("nested_value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 1; let d: i64 = 1; "
     "let y: i64 = if c == 1 { if d == 1 { __mind_store_i64(p, 70) } else { 2 } } else { 3 }; "
     "y + 5 }", 5),
    ("deep_value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 1; "
     "let y: i64 = 40 + if c == 1 { __mind_store_i64(p, 70) } else { 2 }; y }", 40),
    ("stmt_if_both_stores", "fn f(p: i64, k: i64) -> i64 { if k == 0 { __mind_store_i64(p, 4); } "
     "else { __mind_store_i64(p, 6); } __mind_load_i64(p) }\n"
     f"fn main() -> i64 {{ {ALLOC} f(p, 0) * 10 + f(p, 1) }}", 46),
    ("store_as_f64", f"fn main() -> i64 {{ {ALLOC} let f: f64 = __mind_store_i64(p, 70) as f64; "
     "let g: f64 = f + 2.5; g as i64 }", 2),
    ("logic_and_bits", f"fn main() -> i64 {{ {ALLOC} if !__mind_store_i64(p, 1) && "
     "__mind_load_i64(p) == 1 { return __mind_store_i64(p, 255) | 19; } 2 }", 19),
    ("loop_accumulate", "fn w(p: i64) -> i64 { let mut i: i64 = 0; let mut acc: i64 = 0; "
     "while i < 5 { let x: i64 = __mind_store_i64(p, i + 100); acc = acc + x + 1; i = i + 1; } acc }\n"
     f"fn main() -> i64 {{ {ALLOC} w(p) + __mind_load_i64(p) - 100 }}", 9),
    ("deep_dirty_then_tail", DIRTY + "fn w(p: i64, a: i64, b: i64, c: i64) -> i64 { "
     "let s: i64 = a * b + c; __mind_store_i64(p, s) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(9); let r: i64 = w(p, 3, 4, 5); "
     "r + __mind_load_i64(p) + (z - z) }", 17),
    ("w_set_shape", "fn w_get(w_addr: i64, i: i64) -> i64 { __mind_load_i64(w_addr + i * 8) }\n"
     "fn w_set(w_addr: i64, i: i64, v: i64) -> i64 { __mind_store_i64(w_addr + i * 8, v) }\n"
     "fn main() -> i64 { let w: i64 = __mind_alloc(64); let mut i: i64 = 0; let mut bad: i64 = 0; "
     "while i < 8 { bad = bad + w_set(w, i, i * 3 + 1); i = i + 1; } w_get(w, 7) + bad }", 22),
    ("diverging_else", "fn f(first: i64) -> i64 { let mut width: i64 = 0; let mut cp: i64 = 0; "
     "if first < 128 { width = 1; cp = first; } else { if first >= 194 && first <= 223 { "
     "width = 2; cp = first & 31; } else { return 0 - 1; } } width * 100 + cp }\n"
     "fn main() -> i64 { f(5) + f(200) - f(250) - 309 }", 5),
    ("diverging_loop_exits", "fn f(n: i64) -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; "
     "while i < n { if i == 3 { s = s + 10; } else { if i > 6 { break; } else { i = i + 1; "
     "continue; } } i = i + 1; } s + i }\nfn main() -> i64 { f(20) }", 17),
    ("native_opt_directive", "// mind: native_opt = full\nfn step(i: i64) -> i64 { i * 3 + 1 }\n"
     "fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; "
     "while i < 10 { s = s + step(i); i = i + 1; } s - 100 }", 45),
]
GROUP_SUM = "testdata/native_avx2/group_sum_"
# Group-sum rows (kinds 87/88): every fixture runs here (AVX2 path) and under qemu
# -cpu Westmere (no AVX: the scalar path); both must print the same bytes, from the
# .hex next to the fixture. bank3 / bank4 / acc32 = bytes 0..832 / 1824..3104 /
# 896..1152 (twice) of the scalar oracle native_avx2_i64_scalar.mind's stdout
# (`git show cdc0c306:examples/mindc_mind/native_avx2_i64_scalar.mind > s.mind`,
# `--direct-compile s.mind --compiler <stage2> --out s`, `./s | xxd -p`, sliced);
# remainder = those oracle rows with pixels >= n left at -1; none / alias = the
# sequential call semantics stated in nb_group_sum_bank.mind / nb_group_sum_acc.mind.
# qemu-user runner: QEMU_X86_64, else PATH, else the default install path; an absent runner is BLOCKED.
QEMU = os.environ.get("QEMU_X86_64") or shutil.which("qemu-x86_64") or "/usr/local/bin/qemu-x86_64"
GS_WESTMERE = [QEMU, "-cpu", "Westmere"]


def _gs_hex(name: str) -> bytes:
    return bytes.fromhex((HERE / f"{GROUP_SUM}{name}.hex").read_text())


MEASURE = "testdata/native_measure/"
# M-A4 LICM (nb_opt_licm.mind): each fixture with the native_opt declaration absent and present,
# front door + MLIR; every build must give the value computed independently (the fixture header).
LICM_DIRECTIVE = "// mind: native_opt = full\n"
# Narrow-int fixtures are direct-feed only (the Rust lane refuses an i32 binding, lane.type.int_width).
LICM_NARROW = [("licm_narrow", 44), ("licm_shl_u8", 74), ("licm_shr_i8", 208), ("licm_shl_u16", 220), ("licm_arith_mix", 181)]
LICM_CASES = [("licm_basic", 131), ("licm_nested", 33), ("licm_chain", 242), ("licm_counter", 153),
              ("licm_store", 220), ("licm_if_assign", 165), ("licm_branch_loop", 56), ("licm_for", 126),
              ("licm_float", 34), ("licm_div_zero", 80)]
EXIT_CASES += [(f"{name}_{tag}", pre + (FIXTURES / f"{name}.mind").read_text(), want)
               for name, want in LICM_CASES for tag, pre in (("off", ""), ("on", LICM_DIRECTIVE))]
# Native-only rows. Keys (all optional but fixture/family): fixture (path or
# source text), family, expect_exit, expect_stdout (bytes), objdump_required /
# objdump_forbidden / readelf_required / readelf_forbidden (substrings of
# `objdump -D -b binary -m i386:x86-64` / `readelf -W -l -d` text), runner_prefix
# (e.g. qemu-x86_64 -cpu Westmere), env, fake_driver (build the fake libcuda and
# export LD_LIBRARY_PATH), timeout_s, stderr_required, refusal (named construct
# the compiler prints: rc 3, no ELF), direct_feed, front_door, compiler_driver
# (fixture is a driver for the composed compiler unit), mlir_expect, mark (a
# capability mark; a compiler without it must refuse the fixture), encoder_table +
# parts (the emitter's byte table, checked through the compiled parts; see below).
BANK_ACC8 = bytes.fromhex(  # bank_acc8 / _layer_a / _scalar: the 8 x 32 accumulator bytes
    "9c8e009863ffe67daae54b1a710a7d36b89488eb7f8da524c65bb2fc8d481118d4fa8a169bfbbe13e2310a9ea966ab11"
    "f0c04d86b7490126fe6709ecc564c9cd0ce77636d3771a151afec5f9e142c920286d0c37ef85988736f4b5f5fd00e9c6"
    "445374920b74e9fe524aaf7f199f46026099749327425ba56e00e85e351de0187c3f336b43f01bfc8a16f697517b93b2"
    "984536375f7e3969a68ccfe26db91e17b4ab63677beca1a5c262ca8089d71f4cd0710184973a230bde989c72a5d51413"
    "ec97b553b3686bc2fa2e5c0ec1b35bc7081e8661cf7608d116257ff5dd71321c2404d9e2eb646807327bdb6af90fb7ba"
    "404a74fd0733d9ce4e31a708158ee7bf"
)
BANK_FORMS = bytes.fromhex(  # bank_forms / _scalar: the 14 forms, 448 bytes
    "46c2140f4783a1d0de69dfe64aecbf94745f628ae90805dce20d17073e8228f0828948fd5d6d0f6bd03e1c5874c14386"
    "d62c4724712bdcd1b2d2a59bf2aa370646c1140e4782a1cfde69dee64aebbf94745f618ae90804dbe20d16063e8128ef"
    "46c2140f4883a1d0de69dfe64aecbf94745f628ae90805dce20d17073f8228f0828948fd5c6d0f6bd03e1c5874c14386"
    "d62c4724712bdcd1b2d2a59bf2aa3706825ef9e3e18266982fbc812bedb67820420fab67fd44985495c2eda329d97f7a"
    "51f393349d214ec5ba700f308e32743eed2206279974419145c69192972450b7c2697f194b4a22a37911652049f0f993"
    "64ffbb6c1a41ae7d609ffc06d8e8faabc7d8227a1c1019e87017a8d0025fc17bba428494250231317ac733cf9aa22053"
    "853cc067373873f10e8bb926237ce6487fc33b072388ed7ff6602045d657e852b339a2a3fb32d7d63b8239aac57d58e2"
    "4734e0be49f51978b3c867c747f2c95e7082007469f8000000e36969004b4bf81800a900a90000000000f800001800ce"
    "7082007469f8000000e36969004b4bf81800a900a90000000000f800001800cebc1d362c1fd26012f0a8defa1ebd1367"
    "e0586fc83d6fecf6872092b8d939f540"
)
# bank_forms: every bank op it emits, as objdump decodes it (an independent decoder).
BANK_FORMS_ASM = [
    "vmovdqu (%rax),%ymm8", "vmovdqu (%rax),%ymm9", "vpaddd %ymm9,%ymm8,%ymm10", "vmovdqu %ymm10,(%rax)",
    "vpsubd %ymm9,%ymm8,%ymm11", "vmovdqu %ymm11,0x20(%rax)", "vpaddb %ymm9,%ymm8,%ymm12",
    "vmovdqu %ymm12,0x40(%rax)", "vpaddq %ymm9,%ymm8,%ymm13", "vmovdqu %ymm13,0x60(%rax)",
    "vpsubq %ymm9,%ymm8,%ymm14", "vmovdqu %ymm14,0x80(%rax)", "vmovdqu 0x20(%rax),%ymm1",
    "vpaddd 0x20(%rax),%ymm1,%ymm1", "vmovdqu %ymm1,0xa0(%rax)", "vmovdqu 0x40(%rax),%ymm2",
    "vpsubd 0xc8(%rax),%ymm2,%ymm2", "vmovdqu %ymm2,0xc0(%rax)", "vmovdqu 0x60(%rax),%ymm3",
    "vpaddb 0x60(%rax),%ymm3,%ymm3", "vmovdqu %ymm3,0xe0(%rax)", "vmovdqu 0x80(%rax),%ymm12",
    "vpaddq 0x80(%rax),%ymm12,%ymm12", "vmovdqu %ymm12,0x100(%rax)", "vmovdqu 0xa0(%rax),%ymm13",
    "vpsubq 0xa0(%rax),%ymm13,%ymm13", "vmovdqu %ymm13,0x120(%rax)", "vmovdqu -0x40(%rax),%ymm4",
    "vmovdqu %ymm4,0x140(%rax)", "vmovdqu 0xc0(%rax),%ymm5", "vpshufb 0xe0(%rax),%ymm5,%ymm6",
    "vmovdqu %ymm6,0x160(%rax)", "vmovdqu 0xe0(%rax),%ymm7", "vpshufb %ymm7,%ymm5,%ymm14",
    "vmovdqu %ymm14,0x180(%rax)", "vmovdqu (%rax),%ymm0", "vmovdqu %ymm0,(%rax)",
]
EPI_FORMS = bytes.fromhex(  # bank_epi_forms / bank_epi_scalar: the six R-02e rows, 320 bytes
    "c26902f29bd79b2a70b25e1f75fade1f080ce0608115da250a54a47db455ab513e000801e0e14b21048440210f2c28a9"
    "4008720a0081003288106084180a6401ff8f69dff8f9fffd9cfdfbb3fffffbff66b9f7fff2f7fbfffbbdef9cdd7aff9d"
    "ffffffffffffffffffffffffffffffffffffffffffffffff000000000000000000000000000000000000000000000000"
    "0000000000000000ffffffffffffffff4c3942cf5faa0300171a7d764b6f0700b45b810fde9703002ca7ccd2f8eb0400"
    "00c0470129133efc0080a3323fe6a75700802867de5be016004017468d91538d00000000000000000000000000000000"
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "c0470129133efc3f80a3323fe6a75767802867de5be016e04017468d91538d3c"
)
# bank_epi_forms: every epilogue op it emits, as objdump decodes it.
EPI_FORMS_ASM = ["vpmuludq %ymm8,%ymm7,%ymm0", "vpand  %ymm7,%ymm8,%ymm14", "vpor   %ymm8,%ymm7,%ymm0", "vpcmpgtq %ymm8,%ymm7,%ymm14", "vpcmpgtq %ymm7,%ymm8,%ymm0", "vpsrlq %xmm14,%ymm7,%ymm0", "vpsllq %xmm14,%ymm8,%ymm0", "vpsllq %xmm14,%ymm8,%ymm7", "vpsllq %xmm14,%ymm8,%ymm8"]
BANK_CANARY = bytes.fromhex("fff6c2ffffffffff4239d2ffffffffff857be1ffffffffffc8bdf0ffffffffff50f8540100000000")  # bank_canary / _scalar
NATIVE_ONLY_CASES = [
    {"id": "e1_direct_feed_control", "family": "control", "fixture": "fn main() -> i64 { 7 }",
     "direct_feed": True, "front_door": True, "expect_exit": 7,
     "objdump_required": ["syscall"], "readelf_required": ["LOAD"]},
    {"id": "e1_unknown_name_control", "family": "control", "direct_feed": True,
     "fixture": "fn main() -> i64 { __mind_e1_unregistered_v1(5) }", "refusal": "--backend=mlir"},
    {"id": "e1_unknown_name_front_door", "family": "control", "front_door": True,
     "fixture": "fn main() -> i64 { __mind_e1_unregistered_v1(5) }", "stderr_required": "lane.call.unknown"},
    # R-01b register bank (kinds 70-83); expected bytes also checked by the scalar twins
    # (bank_*_scalar cases below: native and MLIR legs).
    {"id": "bank_acc8", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_acc8.mind", "direct_feed": True,
     "front_door": True, "expect_stdout": BANK_ACC8, "objdump_forbidden": ["%ymm15"],
     "objdump_required": ["vzeroall", "vzeroupper", "vpaddd (%rax),%ymm0,%ymm0", "vpaddd 0xe0(%rax),%ymm7,%ymm7",
                          "cpuid", "xgetbv"]},
    {"id": "bank_acc8_layer_a", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_acc8_layer_a.mind",
     "direct_feed": True, "expect_stdout": BANK_ACC8, "objdump_required": ["vpaddd (%rcx),%ymm15,%ymm15"]},
    {"id": "bank_acc8_scalar", "family": "ymm_bank", "fixture": "testdata/native_avx2/bank_acc8_scalar.mind",  # no bank name: no mark
     "direct_feed": True, "front_door": True, "expect_stdout": BANK_ACC8, "objdump_forbidden": ["%ymm"]},
    {"id": "bank_forms", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_forms.mind", "direct_feed": True,
     "front_door": True, "expect_stdout": BANK_FORMS,
     "objdump_required": BANK_FORMS_ASM},
    {"id": "bank_residue", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_residue.mind",
     "direct_feed": True, "expect_stdout": bytes(32)},
    {"id": "bank_canary", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_canary.mind",
     "direct_feed": True, "expect_stdout": BANK_CANARY},
    {"id": "bank_refuse_nonliteral_reg", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_nonliteral_reg.mind",
     "direct_feed": True, "refusal": "ymm.operand_not_literal"},
    {"id": "bank_twin_nonliteral_reg", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_nonliteral_reg.mind",
     "direct_feed": True, "expect_exit": 48},
    {"id": "bank_refuse_reg15", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_reg15.mind",
     "direct_feed": True, "refusal": "ymm.register_out_of_range"},
    {"id": "bank_twin_reg15", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_reg15.mind",
     "direct_feed": True, "expect_exit": 48},
    {"id": "bank_refuse_user_call", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_user_call.mind",
     "direct_feed": True, "refusal": "ymm.bank_function_not_leaf"},
    {"id": "bank_twin_user_call", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_user_call.mind",
     "direct_feed": True, "expect_exit": 48},
    {"id": "bank_refuse_f64", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_f64.mind",
     "direct_feed": True, "refusal": "ymm.bank_function_not_leaf"},
    {"id": "bank_twin_f64", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_f64.mind",
     "direct_feed": True, "expect_exit": 24},
    {"id": "bank_refuse_layer_a", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_layer_a.mind",
     "direct_feed": True, "refusal": "ymm.memory_form_in_bank_function"},
    {"id": "bank_twin_layer_a", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_layer_a.mind",
     "direct_feed": True, "expect_exit": 67},
    {"id": "bank_refuse_nonliteral_disp", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_nonliteral_disp.mind",
     "direct_feed": True, "refusal": "ymm.operand_not_literal"},
    {"id": "bank_twin_nonliteral_disp", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_nonliteral_disp.mind",
     "direct_feed": True, "expect_exit": 48},
    {"id": "bank_refuse_value_use", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_value_use.mind",
     "direct_feed": True, "refusal": "ymm.discard_result_use"},
    {"id": "bank_twin_value_use", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_value_use.mind",
     "direct_feed": True, "expect_exit": 48},
    # R-02e epilogue rows (kinds 94-99, nb_x86_ymm_epi.mind); scalar twin bank_epi_scalar below.
    {"id": "bank_epi_forms", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_epi_forms.mind",
     "direct_feed": True, "front_door": True, "expect_stdout": EPI_FORMS,
     "objdump_required": ["vzeroall", "vzeroupper", *EPI_FORMS_ASM]},
    {"id": "bank_epi_quad", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_epi_quad.mind",
     "direct_feed": True, "expect_exit": 0, "objdump_required": ["vpmuludq", "vpcmpgtq", "vpsrlq", "vpsllq", "vpand", "vpor"]},
    {"id": "bank_refuse_epi_reg15", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_refuse_epi_reg15.mind",
     "direct_feed": True, "refusal": "ymm.register_out_of_range"},
    {"id": "bank_twin_epi_reg15", "family": "ymm_bank", "mark": "nbbank01", "fixture": "testdata/native_avx2/bank_twin_epi_reg15.mind",
     "direct_feed": True, "expect_exit": 4},
    # native_opt: the line-1 directive sets bit 3 of the flags word (nb_opt_level.mind) and moves no
    # byte yet; bit 0 still captures argv beside it; the attribute form is refused on both fences.
    {"id": "opt_level_recognizer", "family": "native_opt", "direct_feed": True, "expect_exit": 0,
     "fixture": (HERE / "nb_opt_level.mind").read_text() + (FIXTURES / "nb_opt_level_cases.mind").read_text()},
    {"id": "opt_full_argc", "family": "native_opt", "direct_feed": True, "expect_exit": 5,
     "fixture": "// mind: native_opt = full\nfn main() -> i64 { __mind_argc() + 4 }"},
    {"id": "opt_near_miss_argc", "family": "native_opt", "direct_feed": True, "expect_exit": 5,
     "fixture": "// mind: native_opt = fast\nfn main() -> i64 { __mind_argc() + 4 }"},
    {"id": "opt_attr_form_direct", "family": "native_opt", "direct_feed": True, "refusal": "--backend=mlir",
     "fixture": "#[native_opt(full)]\nfn main() -> i64 { 7 }"},
    {"id": "opt_attr_form_front_door", "family": "native_opt", "front_door": True,
     "fixture": "#[native_opt(full)]\nfn main() -> i64 { 7 }", "stderr_required": "lane.source.function_contract"},
    # M-A4 LICM on the self-host, declaration absent and present.
    # A loop-body shadow exits 164. A narrow i32 let exits 44 (the Rust lane
    # refuses that binding). Prefix `&` is outside the native lowering: both
    # declarations refuse, rc 3, and the message names --backend=mlir.
    *[{"id": f"licm_shadow_{tag}", "family": "licm", "direct_feed": True, "expect_exit": 164,
       "fixture": pre + (FIXTURES / "licm_shadow.mind").read_text()}
      for tag, pre in (("off", ""), ("on", LICM_DIRECTIVE))],
    *[{"id": f"{nm}_{tag}", "family": "licm", "direct_feed": True, "expect_exit": want,
       "fixture": pre + (FIXTURES / f"{nm}.mind").read_text()}
      for nm, want in LICM_NARROW for tag, pre in (("off", ""), ("on", LICM_DIRECTIVE))],
    {"id": "narrow_wrap_edges", "family": "values", "direct_feed": True, "expect_exit": 63,
     "fixture": "testdata/native_lane_values/narrow_wrap_edges.mind"},  # one bit per W0 descriptor width
    {"id": "i32_param_range", "family": "values", "direct_feed": True, "expect_exit": 63,
     "fixture": "testdata/native_lane_values/i32_param_range.mind"},  # i32/u32 params: no entry wrap needed
    *[{"id": f"licm_addr_{tag}", "family": "licm", "direct_feed": True, "refusal": "--backend=mlir",
       "fixture": pre + (FIXTURES / "licm_addr.mind").read_text()}
      for tag, pre in (("off", ""), ("on", LICM_DIRECTIVE))],
    {"id": "e1_part_dispatch_check", "family": "parts", "compiler_driver": True, "expect_exit": 0,
     "fixture": "testdata/compiler_parts/part_dispatch_check_driver.mind", "mlir_expect": "skip"},
    *[{"id": f"gs_{n}{'_westmere' if w else ''}", "family": "avx2_groupsum", "direct_feed": True,
       "fixture": f"{GROUP_SUM}{n}.mind", "expect_stdout": _gs_hex(n), "runner_prefix": GS_WESTMERE if w else [],
       "objdump_required": ["vpaddq", "vpsubq", "%ymm", "cpuid", "vzeroupper"]}
      for n in ("bank3", "bank4", "acc32", "remainder", "none", "alias") for w in (0, 1)],
    {"id": "gs_bank_arity4_refused", "family": "avx2_groupsum", "direct_feed": True, "refusal": "--backend=mlir",
     "fixture": "fn main() -> i64 { let p: i64 = __mind_alloc(64); __mind_group_bank_i64(p, p, 8, 1) }"},
    {"id": "gs_acc_arity7_refused", "family": "avx2_groupsum", "direct_feed": True, "refusal": "--backend=mlir",
     "fixture": "fn main() -> i64 { let p: i64 = __mind_alloc(512); __mind_group_acc32_i64(p, p, p, 0, 0, 0, 1) }"},
]
# AVX2 ymm rows (Layer A, R-02, ISA guard) and __mind_cpu_features_v1. `mark`: a
# compiler without that capability mark must refuse the row's fixture instead (the
# self-host second fence: rc 3, no ELF), so the committed pre-reseed seed checks
# the refusal and a marked compiler checks the row. Westmere has no AVX: the ISA
# guard's failure path (exit 132, isa.avx2_missing) runs before any ymm op, which
# is how the build-only repros (they dereference 0) are run safely.
AVX2 = "testdata/native_avx2/"
WESTMERE = [QEMU, "-cpu", "Westmere"]
HASWELL = [QEMU, "-cpu", "Haswell"]
ISA_FAIL = {"runner_prefix": WESTMERE, "expect_exit": 132, "stderr_required": "isa.avx2_missing"}
MAIN_PARAMS = "488b3c24488d742408"  # mov rdi,[rsp] ; lea rsi,[rsp+8]
ARGV_CAPTURE = "488b042448b9" + (1 << 24).to_bytes(8, "little").hex() + "48894108" "488d442408" "48894110" "48c70110000000"
ISA_GUARD_HEAD = "5331c00fa2"
NATIVE_ONLY_CASES += [
    {"id": "avx2_encoder_table", "family": "avx2", "fixture": AVX2 + "encoder_table_driver.mind",
     "encoder_table": AVX2 + "native_avx2_encoder_table.tsv",
     "parts": ["nb_emit_core.mind", "nb_cpu_features.mind", "nb_x86_vex.mind", "nb_x86_ymm_epi.mind"]},
] + [
    {"id": f"avx2_repro_{name.split('_')[0]}", "family": "avx2", "fixture": f"{AVX2}{name}.mind",
     "mark": "nbymm001", "direct_feed": True, "front_door": True, **ISA_FAIL,
     "objdump_required": [op, "vmovdqu %ymm15,(%rax)", "vzeroupper", "cpuid", "xgetbv"]}
    for name, op in (("vpaddd_i32x8", "vpaddd (%rcx),%ymm15,%ymm15"), ("vpsubd_i32x8", "vpsubd (%rcx),%ymm15,%ymm15"),
                     ("vpaddb_i8x32", "vpaddb (%rcx),%ymm15,%ymm15"), ("vpshufb_i8x32", "vpshufb (%rdx),%ymm15,%ymm15"),
                     ("vmovdqu_i32x8", "vmovdqu (%rcx),%ymm15"), ("vpaddq_i64x4", "vpaddq (%rcx),%ymm15,%ymm15"),
                     ("vpsubq_i64x4", "vpsubq (%rcx),%ymm15,%ymm15"))
] + [
    {"id": "avx2_semantics", "family": "avx2", "fixture": AVX2 + "avx2_semantics.mind", "mark": "nbymm001",
     "direct_feed": True, "front_door": True, "expect_exit": 0},
    {"id": "avx2_semantics_haswell", "family": "avx2", "fixture": AVX2 + "avx2_semantics.mind",
     "mark": "nbymm001", "direct_feed": True, "runner_prefix": HASWELL, "expect_exit": 0},
    {"id": "avx2_semantics_westmere", "family": "avx2", "fixture": AVX2 + "avx2_semantics.mind",
     "mark": "nbymm001", "direct_feed": True, "front_door": True, **ISA_FAIL},
    {"id": "cpu_features_probe", "family": "avx2", "fixture": AVX2 + "cpu_features_probe.mind",
     "mark": "nbcpuf01", "direct_feed": True, "front_door": True, "expect_exit": 0,
     "objdump_required": ["cpuid", "xgetbv"]},
    {"id": "cpu_features_probe_westmere", "family": "avx2", "fixture": AVX2 + "cpu_features_probe.mind",
     "mark": "nbcpuf01", "direct_feed": True, "runner_prefix": WESTMERE, "expect_exit": 3},
    {"id": "cpu_features_probe_haswell", "family": "avx2", "fixture": AVX2 + "cpu_features_probe.mind",
     "mark": "nbcpuf01", "direct_feed": True, "runner_prefix": HASWELL, "expect_exit": 0},
    {"id": "avx2_load_store_control", "family": "avx2", "fixture": AVX2 + "load_store_i64.mind",
     "direct_feed": True, "expect_exit": 42, "objdump_forbidden": ["%ymm", "xgetbv"]},
    {"id": "avx2_names_inert", "family": "avx2", "fixture": AVX2 + "ymm_names_inert.mind",
     "direct_feed": True, "expect_exit": 42, "objdump_forbidden": ["%ymm", "xgetbv", "cpuid"]},
] + [
    # fn main(argc, argv) + a ymm row: capture (34 B, when the program reads argv) and the
    # W4b binding are the entry's first bytes, the ISA guard (push rbx ; xor eax,eax ;
    # cpuid) right after them; run as `prog mind-arg`, and on Westmere the guard exits 132.
    {"id": f"avx2_main_params{tag}{host}", "family": "avx2", "fixture": AVX2 + name, "mark": "nbymm001",
     "direct_feed": True, "args": ["mind-arg"], "entry_prefix": prefix + ISA_GUARD_HEAD, **extra}
    for tag, name, prefix in (("", "main_params_vmovdqu.mind", MAIN_PARAMS),
                              ("_capture", "main_params_argv_capture.mind", ARGV_CAPTURE + MAIN_PARAMS))
    for host, extra in (("", {"expect_exit": 0}), ("_westmere", ISA_FAIL))
]
# R-01b: the bank's ISA guard on both sides, and a name one byte off every real one.
BANK = {"family": "ymm_bank", "mark": "nbbank01", "direct_feed": True}
NATIVE_ONLY_CASES += [
    {"id": "bank_acc8_westmere", **BANK, "fixture": AVX2 + "bank_acc8.mind", **ISA_FAIL},
    {"id": "bank_acc8_haswell", **BANK, "fixture": AVX2 + "bank_acc8.mind", "runner_prefix": HASWELL,
     "expect_stdout": BANK_ACC8},
    {"id": "bank_refuse_misspelled", **BANK, "refusal": "--backend=mlir", "fixture":
     "fn f(a: i64) -> i64 { __mind_ymm_vpaddb_i32x8_m(0, a, 0); 0 }\nfn main() -> i64 { f(__mind_alloc(32)) }"},
    # A5 shared-memory rows (nb_shm.mind). Bytes: mov r10/r8/r9,[rbp+d] (4c 8b 95 / 85 / 8d,
    # syscall arguments 4-6), mov eax,9/11/203, pause. The fork row's stdout is the
    # checksum computed here independently: multi-process == single-process == this value.
    # `mark`: the capability these rows need. deferred: would assert the second-fence refusal
    # (rc 3, no ELF) when --compiler lacks it, stubbed because the row glue is the parts lane's
    # — upgrade path: run_native_only reads row["mark"]; until then such a compiler fails them.
    {"id": "a5_mmap_probe", "family": "shm", "fixture": "testdata/native_shm/shm_mmap_probe.mind",
     "direct_feed": True, "front_door": True, "mark": "nbshm001",
     "objdump_required": ["4c 8b 95", "4c 8b 85", "4c 8b 8d", "b8 09 00 00 00", "b8 0b 00 00 00",
                          "b8 cb 00 00 00", "f3 90", "pause"]},
    {"id": "a5_fork_rounds", "family": "shm", "fixture": "testdata/native_shm/shm_fork_rounds.mind",
     "direct_feed": True, "front_door": True, "timeout_s": 1200, "mark": "nbshm001",
     "expect_stdout": struct.pack("<qq", 126470945504864, 126470945504864)},
    {"id": "a5_arity_front_door", "family": "shm", "front_door": True, "stderr_required": "lane.call.arity",
     "fixture": "fn main() -> i64 { __mind_sys_mmap(0, 4096, 3, 34, 0 - 1) }"},
    {"id": "a5_arity_direct_refused", "family": "shm", "direct_feed": True, "refusal": "--backend=mlir",
     "fixture": "fn main() -> i64 { __mind_sys_mmap(0, 4096, 3, 34, 0 - 1) }"},
    {"id": "a5_misspelled_refused", "family": "shm", "direct_feed": True,
     "fixture": "fn main() -> i64 { __mind_sys_mmap2(0, 4096, 3, 34, 0 - 1, 0) }", "refusal": "--backend=mlir"},
    {"id": "a5_misspelled_front_door", "family": "shm", "front_door": True,
     "fixture": "fn main() -> i64 { __mind_atomic_load_acquire_i32(0) }", "stderr_required": "lane.call.unknown"},
    # Exit contract of driver-row images: exit_group (231) at the bounds trap (status 77 = mov edi,0x4d)
    # and the entry return (48 c7 c0 e7 = mov rax,231, made only by the kind-3 entry return) while a
    # never-returning driver thread runs; driver-only images list libcuda.so.1 only (R-06).
    {"id": "r05_trap_after_driver_thread", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/trap_after_init.mind", "fake_driver": True,
     "env": {"MIND_FAKE_CUDA_THREAD": "1"}, "timeout_s": 1, "expect_exit": 77,
     "objdump_required": ["48 c7 c0 e7 00 00 00 \tmov    $0xe7,%rax", "bf 4d 00 00 00", "b8 e7 00 00 00"],
     "objdump_forbidden": ["mov    $0x3c,%eax", "mov    $0x3c,%rax"]},
    {"id": "r05_entry_return_driver_thread", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/fake_rows.mind", "fake_driver": True,
     "env": {"MIND_FAKE_CUDA_THREAD": "1"}, "timeout_s": 1, "expect_exit": 42,
     "objdump_required": ["48 c7 c0 e7 00 00 00 \tmov    $0xe7,%rax"],
     "objdump_forbidden": ["mov    $0x3c,%eax", "mov    $0x3c,%rax"],
     "readelf_required": ["Shared library: [libcuda.so.1]"],
     "readelf_forbidden": ["Shared library: [libc.so.6]", "VERNEED", "VERSYM"]},
    {"id": "r05_libc_only_keeps_exit", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/trap_libc_only.mind", "expect_exit": 77,
     "objdump_required": ["48 c7 c0 3c 00 00 00 \tmov    $0x3c,%rax", "b8 3c 00 00 00", "bf 4d 00 00 00"],
     "objdump_forbidden": ["mov    $0xe7,%rax"],
     "readelf_required": ["Shared library: [libc.so.6]", "VERNEED"], "readelf_forbidden": ["libcuda.so.1"]},
    # The exit decision follows the DECLARED rows, the NEEDED list the CALLED ones: a libc image
    # that declares an unused driver row ends through exit_group and lists libc.so.6 only.
    {"id": "r05_declared_unused_driver_row", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/trap_declared_driver.mind", "expect_exit": 77,
     "objdump_required": ["48 c7 c0 e7 00 00 00 \tmov    $0xe7,%rax", "bf 4d 00 00 00"],
     "objdump_forbidden": ["mov    $0x3c,%eax", "mov    $0x3c,%rax"],
     "readelf_required": ["Shared library: [libc.so.6]", "VERNEED"], "readelf_forbidden": ["libcuda.so.1"]},
    {"id": "r06_driver_only_needed", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/gpu_only.mind", "fake_driver": True,
     "expect_exit": 0, "readelf_required": ["Shared library: [libcuda.so.1]", "contains 13 entries"],
     "readelf_forbidden": ["Shared library: [libc.so.6]", "VERNEED", "VERSYM"]},
    {"id": "r06_mixed_keeps_both", "family": "exit", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/vec_add.mind", "fake_driver": True,
     "expect_exit": 3, "objdump_forbidden": ["mov    $0x3c,%eax", "mov    $0x3c,%rax"],
     "readelf_required": ["Shared library: [libc.so.6]", "Shared library: [libcuda.so.1]",
                          "VERNEED", "contains 17 entries"]},
    # M-A3 measurement rows (nb_measure.mind). The byte strings are llvm-mc's
    # encodings (testdata/native_measure/measure_encoder_table.tsv): rdtscp / shl rdx,32 /
    # or rax,rdx / lfence / cpuid / bt edx,27 / mov eax,298, and the five syscall
    # argument loads mov rdi/rsi/rdx/r10/r8,[rbp+d]. `mark`: the capability these rows need.
    # deferred: would assert the second-fence refusal (rc 3, no ELF) when --compiler lacks it,
    # stubbed because the row glue is the parts lane's — upgrade path: run_native_only reads
    # row["mark"]; until then such a compiler fails them.
    {"id": "ma3_rdtscp", "family": "measure", "fixture": MEASURE + "measure_rdtscp.mind",
     "direct_feed": True, "front_door": True, "mark": "nbmeas01",
     "objdump_required": ["0f 01 f9", "rdtscp", "48 c1 e2 20", "48 09 d0", "0f ae e8", "lfence",
                          "0f a2", "0f ba e2 1b"]},
    {"id": "ma3_rdtscp_qemu_haswell", "family": "measure", "fixture": MEASURE + "measure_rdtscp.mind",
     "direct_feed": True, "mark": "nbmeas01", "runner_prefix": [QEMU, "-cpu", "Haswell"]},
    {"id": "ma3_rdtscp_qemu_qemu64", "family": "measure", "fixture": MEASURE + "measure_rdtscp.mind",
     "direct_feed": True, "mark": "nbmeas01", "runner_prefix": [QEMU, "-cpu", "qemu64"],
     "expect_exit": 132, "stderr_required": "isa.rdtscp_missing"},
    {"id": "ma3_rdtscp_helper_qemu64", "family": "measure",
     "fixture": MEASURE + "measure_rdtscp_helper.mind", "direct_feed": True, "mark": "nbmeas01",
     "runner_prefix": [QEMU, "-cpu", "qemu64"], "expect_exit": 132,
     "stderr_required": "isa.rdtscp_missing"},
    {"id": "ma3_rdtscp_helper", "family": "measure", "fixture": MEASURE + "measure_rdtscp_helper.mind",
     "direct_feed": True, "front_door": True, "mark": "nbmeas01", "objdump_required": ["0f a2", "0f 01 f9"]},
    # The lexical scan skips the name in a string literal; the lane-v1 front door
    # refuses a String local (lane.source.string), so that fixture is direct-feed only.
    {"id": "ma3_lfence_no_guard", "family": "measure", "fixture": MEASURE + "measure_lfence.mind",
     "direct_feed": True, "mark": "nbmeas01", "objdump_required": ["0f ae e8", "lfence"],
     "objdump_forbidden": ["cpuid", "rdtscp"]},
    {"id": "ma3_lfence_front_door", "family": "measure", "direct_feed": True, "front_door": True,
     "mark": "nbmeas01", "expect_exit": 5, "objdump_required": ["0f ae e8", "48 31 c0"],
     "objdump_forbidden": ["cpuid", "rdtscp"],
     "fixture": "fn main() -> i64 {\n    // __mind_rdtscp_v1() named in a comment is not a call.\n"
                "    __mind_lfence_v1() + 5\n}"},
    {"id": "ma3_perf_event_open", "family": "measure",
     "fixture": MEASURE + "measure_perf_event_open.mind", "direct_feed": True, "front_door": True,
     "mark": "nbmeas01", "objdump_forbidden": ["cpuid", "rdtscp"],
     "objdump_required": ["b8 2a 01 00 00", "48 8b bd", "48 8b b5", "48 8b 95", "4c 8b 95", "4c 8b 85",
                          "%r10", "%r8"]},
    {"id": "ma3_misspelled_refused", "family": "measure", "direct_feed": True,
     "fixture": "fn main() -> i64 { __mind_rdtscp_v2() }", "refusal": "--backend=mlir"},
    # MC-6 driver rows (nbcuda02): all 33 against the fake driver, the 10-argument cooperative
    # launch dropping its four stack words right after the call, and the named byte fence.
    {"id": "mc6_rows_fake_driver", "family": "cuda2", "direct_feed": True,
     "fixture": "../../tests/fixtures/native_lane_gpu/fake_rows2.mind", "fake_driver": True,
     "expect_exit": 42, "objdump_required": ["48 81 c4 20 00 00 00 \tadd    $0x20,%rsp"],
     "readelf_required": ["Shared library: [libcuda.so.1]", "BIND_NOW"], "readelf_forbidden": ["VERNEED"]},
    {"id": "mc6_byte_fence_refusal", "family": "cuda2", "direct_feed": True,
     "fixture": 'extern "C" { fn cuMemsetD8_v2(d: i64, uc: u32, n: i64) -> i32; }\n'
                "fn main() -> i64 { let v: i64 = 300; cuMemsetD8_v2(4660, v as u32, 12) as i64 }",
     "refusal": "extern.byte_arg_unproven: v as u32"},
]
# Assert trap (id 54). FIPS 180-4 digests, checked with sha256sum, never from a
# MIND run. `mark` nbtrap54: a seed without the mark defers these rows.
FIPS_SHA256 = (
    b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
    b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    b"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1\n"
)
TRAP54 = "testdata/native_trap/"
TRAP54_ASM = ["bf 86 00 00 00", "b8 e7 00 00 00", "0f 05", "0f 0b"]
NATIVE_ONLY_CASES += [
    {"id": "trap54_sha256_fips", "family": "assert_trap", "mark": "nbtrap54",
     "fixture": TRAP54 + "sha256_fips.mind", "direct_feed": True, "front_door": True,
     "expect_exit": 0, "expect_stdout": FIPS_SHA256, "objdump_required": TRAP54_ASM},
    {"id": "trap54_sha256_fail", "family": "assert_trap", "mark": "nbtrap54",
     "fixture": TRAP54 + "sha256_fail.mind", "direct_feed": True, "front_door": True,
     "expect_exit": 134, "expect_stdout": b"before\n", "objdump_required": TRAP54_ASM},
    {"id": "trap54_assert_direct", "family": "assert_trap", "mark": "nbtrap54",
     "fixture": TRAP54 + "assert_direct.mind", "direct_feed": True, "front_door": True,
     "expect_exit": 0, "expect_stdout": b"", "objdump_required": TRAP54_ASM},
    {"id": "trap54_assert_direct_arg", "family": "assert_trap", "mark": "nbtrap54",
     "fixture": TRAP54 + "assert_direct.mind", "direct_feed": True, "front_door": True,
     "args": ["x"], "expect_exit": 134, "expect_stdout": b"", "objdump_required": TRAP54_ASM},
    {"id": "trap54_sha256_nocall", "family": "assert_trap",
     "fixture": TRAP54 + "sha256_nocall.mind", "direct_feed": True, "front_door": True,
     "expect_exit": 0, "objdump_forbidden": ["bf 86 00 00 00"]},
    {"id": "trap54_arity3_direct", "family": "assert_trap", "direct_feed": True,
     "fixture": TRAP54 + "assert_arity3.mind", "refusal": "--backend=mlir"},
    {"id": "trap54_arity3_front", "family": "assert_trap", "front_door": True,
     "fixture": TRAP54 + "assert_arity3.mind", "stderr_required": "lane.call.arity"},
    {"id": "trap54_near_name", "family": "assert_trap", "direct_feed": True,
     "fixture": TRAP54 + "assert_near_name.mind", "refusal": "--backend=mlir"},
]
FAKE_DRIVER = REPO / "tests" / "fixtures" / "native_lane_gpu" / "fake_driver.c"
ARGV_CASE = ("main_argc_argv", "fn main(argc: i64, argv: i64) -> i64 { "
             "if argv == 0 { return 99; } argc + 4 }", 7)


def _build(mindc: str, src: pathlib.Path, out: pathlib.Path, native: bool, env: dict) -> tuple[int, str]:
    cmd = [mindc, "build", str(src), "--target", "cpu", "--emit", "binary", "--no-cache",
           "--out", str(out)]
    cmd += ["--backend", "native", "--native-profile", "bounded"] if native else ["--backend", "mlir"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, cwd=src.parent)
    return r.returncode, r.stderr


def _run(exe: pathlib.Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["/usr/bin/prlimit", "--as=2147483648", "--", str(exe), *argv],
                          capture_output=True, timeout=60, env={"LC_ALL": "C"})


def _tool(argv: list[str], controls: tuple[str, ...]) -> tuple[str, list[str]]:
    """(text, problems): a failed tool, or one whose output lacks its positive controls
    (regexes), is a FAIL for every row reading it, never a vacuous `forbidden` pass."""
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    ok = r.returncode == 0 and all(re.search(c, text, re.M) for c in controls)
    return text, [] if ok else [f"{argv[0]} rc={r.returncode} lacks its positive control"]


def _fake_driver(tmp: pathlib.Path) -> dict:
    lib = tmp / "fakecuda" / "libcuda.so.1"
    if not lib.exists():
        lib.parent.mkdir()
        subprocess.run(["cc", "-shared", "-fPIC", "-o", str(lib), str(FAKE_DRIVER)], check=True,
                       capture_output=True, timeout=120)
    return {"LD_LIBRARY_PATH": str(lib.parent)}


def _check_elf(row: dict, exe: pathlib.Path, tmp: pathlib.Path) -> list[str]:
    bad = []
    asm, asm_bad = _tool(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", str(exe)],
                         ("file format", r"^\s+[0-9a-f]+:\t[0-9a-f]{2} "))
    elf, elf_bad = _tool(["readelf", "-W", "-l", "-d", str(exe)], ("Program Headers",))
    bad += asm_bad if {"objdump_required", "objdump_forbidden"} & row.keys() else []
    bad += elf_bad if {"readelf_required", "readelf_forbidden"} & row.keys() else []
    bad += [f"objdump lacks {t!r}" for t in row.get("objdump_required", []) if t not in asm]
    bad += [f"objdump has {t!r}" for t in row.get("objdump_forbidden", []) if t in asm]
    bad += [f"readelf lacks {t!r}" for t in row.get("readelf_required", []) if t not in elf]
    bad += [f"readelf has {t!r}" for t in row.get("readelf_forbidden", []) if t in elf]
    if "entry_prefix" in row:  # hex the code at e_entry must start with (the W4b bridge check reads it)
        b = exe.read_bytes()
        (entry, phoff), phnum = struct.unpack_from("<QQ", b, 24), struct.unpack_from("<H", b, 56)[0]
        segs = [struct.unpack_from("<IIQQQQ", b, phoff + 56 * i) for i in range(phnum)]
        at = [o + entry - va for ty, _, o, va, _, fs in segs if ty == 1 and va <= entry < va + fs]
        if at and b[at[0]:at[0] + 2] == b"\x49\xb9" and b[at[0] + 52:at[0] + 58] == bytes.fromhex("31c04531c9e9"):
            at = [at[0] + 62 + struct.unpack_from("<i", b, at[0] + 58)[0]]  # bounded allocator prefix: its jmp
        got = b[at[0]:at[0] + len(row["entry_prefix"]) // 2].hex() if at else "no entry"
        bad += [f"entry starts {got}"] if got != row["entry_prefix"] else []
    env = dict(os.environ, LC_ALL="C", **row.get("env", {}))
    env.update(_fake_driver(tmp) if row.get("fake_driver") else {})
    try:
        r = subprocess.run([*row.get("runner_prefix", []), str(exe), *row.get("args", [])], capture_output=True, env=env,
                           timeout=row.get("timeout_s", 60))
    except subprocess.TimeoutExpired:  # a hang is this row's failure, never a harness crash
        return bad + [f"timeout >{row.get('timeout_s', 60)}s (the process outlived its exit)"]
    if r.returncode != row.get("expect_exit", 0):
        bad.append(f"exit={r.returncode}")
    if "expect_stdout" in row and r.stdout != row["expect_stdout"]:
        bad.append("stdout differs")
    if row.get("stderr_required", "").encode() not in r.stderr:
        bad.append(f"stderr lacks {row['stderr_required']!r}")
    return bad


def _encoder_table(row: dict, compiler: str, case_dir: pathlib.Path) -> list[str]:
    """Row kind `encoder_table`. Stage A: the driver, composed after the emitter parts
    and compiled by `compiler`, writes [len][bytes] per table row, in file order, and
    each record must equal the row. Stage B: llvm-mc assembles every llvm-mc row to its
    bytes, and objdump decodes every encoder/objdump row back to its instruction."""
    rows = [ln.split("\t") for ln in (HERE / row["encoder_table"]).read_text().splitlines()
            if ln and not ln.startswith("#")]
    prog = case_dir / "table.mind"
    prog.write_bytes(b"\n".join((HERE / p).read_bytes() for p in (*row["parts"], row["fixture"])))
    exe = case_dir / "table.elf"
    r = direct_compile(prog, pathlib.Path(compiler), exe)
    if not exe.exists():
        return [f"table driver: no ELF rc={r.returncode}"]
    out, recs = subprocess.run([str(exe)], capture_output=True, timeout=120).stdout, []
    while out:
        recs.append(out[1:1 + out[0]].hex())
        out = out[1 + out[0]:]
    bad = [f"A {t[0]}: emitted {got} != {t[2]}" for t, got in zip(rows, recs) if got != t[2]]
    bad += [f"A: {len(recs)} records for {len(rows)} rows"] if len(recs) != len(rows) else []
    work = case_dir / "row"
    if not all((llvm := {t: shutil.which(t, path=LLVM_PATH) for t in ("llvm-mc", "llvm-objcopy")}).values()):
        return bad + [f"B: no stage B, {llvm}: not found in {LLVM_PATH}"]
    for t in rows:
        if t[3].startswith("llvm-mc"):
            work.with_suffix(".s").write_text(".intel_syntax noprefix\n" + t[1].replace(" ; ", "\n") + "\n")
            subprocess.run([llvm["llvm-mc"], "--triple=x86_64", "-filetype=obj", str(work.with_suffix(".s")), "-o", str(work.with_suffix(".o"))], check=True, capture_output=True)
            subprocess.run([llvm["llvm-objcopy"], "-O", "binary", "--only-section=.text", str(work.with_suffix(".o")), str(work.with_suffix(".bin"))], check=True, capture_output=True)
            got = work.with_suffix(".bin").read_bytes().hex()
        else:
            work.with_suffix(".bin").write_bytes(bytes.fromhex(t[2]))
            out, tbad = _tool(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel",
                               str(work.with_suffix(".bin"))], (r"^\s+0:\t[0-9a-f]{2} ",))
            text = out.rstrip().splitlines()[-1]
            got = t[2] if not tbad and text.split("\t")[-1].strip() == t[1] else f"decodes as {text.split(chr(9))[-1]!r}"
        bad += [f"B {t[0]}: {got}"] if got != t[2] else []
    print(f"  INFO  {row['id']}: {len(recs)} records, {len(rows)} rows, {len(bad)} mismatches")
    return bad[:12] + ([f"... {len(bad) - 12} more"] if len(bad) > 12 else [])


def run_native_only(rows: list[dict], compiler: str, mindc: str, env: dict, tmp: pathlib.Path) -> tuple[int, int] | None:
    """(failed rows, rows not run), or None when a row's runner is absent (BLOCKED: never
    a skip). A row whose `mark` the compiler lacks checks only the refusal: FAIL when
    MIND_BENCH_REQUIRE=1 (lane gates), else DEFERRED and not counted as run."""
    fail = skipped = 0
    require = os.environ.get("MIND_BENCH_REQUIRE") == "1"
    marks = pathlib.Path(compiler).read_bytes()
    for row in rows:
        runner = (row.get("runner_prefix") or [""])[0]
        if row.get("mark") and b"\x48\xb8" + row["mark"].encode() not in marks:
            print(f"  {'FAIL' if require else 'DEFERRED'}  {row['id']}: compiler lacks mark {row['mark']}")
            fail, skipped, runner = fail + require, skipped + 1, ""
            row = {"id": row["id"], "family": row["family"], "fixture": row["fixture"],
                   "direct_feed": True, "refusal": "--backend=mlir"}  # pre-mark compiler: refused
        if runner and not shutil.which(runner):
            print(f"BLOCKED: {row['id']} needs runner {runner!r}, not on PATH")
            return None
        case_dir = tmp / row["id"]
        case_dir.mkdir()
        src = row["fixture"]
        text = (HERE / src).read_text() if src.endswith(".mind") else src + "\n"
        source = case_dir / "main.mind"
        source.write_text(text)
        seen = []
        ran = 0  # legs run: a passing leg adds nothing to `seen`
        if row.get("encoder_table"):
            ran += 1
            seen += [f"table: {b}" for b in _encoder_table(row, compiler, case_dir)]
        if row.get("direct_feed"):
            ran += 1
            exe = case_dir / "direct"
            r = direct_compile(source, pathlib.Path(compiler), exe)
            if "refusal" in row:
                ok = r.returncode == 3 and not exe.exists() and row["refusal"].encode() in r.stderr
                seen.append("refused" if ok else f"rc={r.returncode} elf={exe.exists()}")
            else:
                seen += [f"direct: {b}" for b in (_check_elf(row, exe, tmp) if exe.exists()
                                                   else [f"no ELF rc={r.returncode}"])]
        if row.get("front_door"):
            ran += 1
            exe = case_dir / "front"
            rc, err = _build(mindc, source, exe, True, env)
            if "stderr_required" in row and rc != 0:
                ok = row["stderr_required"] in err and not exe.exists()
                seen.append("refused" if ok else f"front rc={rc}")
            else:
                seen += [f"front: {b}" for b in (_check_elf(row, exe, tmp) if rc == 0 else [f"build={rc}"])]
        if row.get("compiler_driver"):
            ran += 1
            exe = case_dir / "composed"
            r = driver_compile("examples/mindc_mind/" + src, pathlib.Path(compiler), exe)
            seen += [f"composed: {b}" for b in (_check_elf(row, exe, tmp) if exe.exists()
                                                 else [f"no ELF rc={r.returncode} {r.stderr[-200:]!r}"])]
        bad = [x for x in seen if x != "refused"] if ran else ["no leg ran"]
        fail += bool(bad)
        print(f"  {'FAIL' if bad else 'PASS'}  {row['id']} [{row['family']}]: {'; '.join(bad) or 'ok'}")
    return fail, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compiler", default=os.environ.get("MINDC_NATIVE_LANE_ELF"))
    ap.add_argument("--mint-stage2", metavar="OUT")
    ap.add_argument("--assert-fixed-point", action="store_true")
    ap.add_argument("--direct-compile", metavar="PROG")
    ap.add_argument("--out")
    ap.add_argument("--only", metavar="FAMILY")
    args = ap.parse_args()
    if args.mint_stage2:
        digest = mint_stage2(pathlib.Path(args.mint_stage2), assert_fixed_point=args.assert_fixed_point)
        print(f"MINT stage2 {args.mint_stage2} sha256={digest}"
              + (" (stage2 == stage3)" if args.assert_fixed_point else ""))
        return 0
    if args.direct_compile:
        if not (args.compiler and args.out):
            print("BLOCKED: --direct-compile needs --compiler ELF and --out ELF")
            return 2
        r = direct_compile(pathlib.Path(args.direct_compile), pathlib.Path(args.compiler), pathlib.Path(args.out))
        sys.stderr.write(r.stderr.decode(errors="replace"))
        print(f"DIRECT rc={r.returncode} out={'written' if pathlib.Path(args.out).exists() else 'none'}")
        return r.returncode
    # Absolute before use: `_build` runs mindc with the case directory as cwd, so
    # a relative MINDC_BIN or --compiler (the bridge reads MINDC_NATIVE_ELF from
    # there) would name nothing and the native leg would be refused, not tested.
    mindc = os.environ.get("MINDC_BIN", str(REPO / "target" / "release" / "mindc"))
    mindc = str(pathlib.Path(mindc).resolve())
    if args.compiler:
        args.compiler = str(pathlib.Path(args.compiler).resolve())
    if not pathlib.Path(mindc).exists():
        print(f"BLOCKED: mindc not found at {mindc}")
        return 2
    if not args.compiler or not pathlib.Path(args.compiler).exists():
        print("BLOCKED: pass --compiler <marked native compiler> (CI passes the committed "
              "bounded seed, testdata/selfhost_loop_bounded/stage1.elf)")
        return 2
    if MARK not in pathlib.Path(args.compiler).read_bytes():
        print(f"BLOCKED: {args.compiler} does not carry the native-lane-v1 mark, so the bridge "
              "would apply the historical profile instead of the one under test")
        return 2
    tools = {k: v for k, v in TOOLS.items()
             if not os.environ.get(k) and os.path.isfile(v) and os.access(v, os.X_OK)}
    env = dict(os.environ, **tools,
               MINDC_NATIVE_ELF=args.compiler, MINDC_STD_DIR=str(REPO / "std"), LC_ALL="C")
    env.pop("MINDC_SO", None)
    cases: list[tuple[str, pathlib.Path | str, list[str], int, bytes | None]] = []
    cases += [(cid, src, [], want, None) for cid, src, want in EXIT_CASES]
    cases.append((ARGV_CASE[0], ARGV_CASE[1], ["a", "b"], ARGV_CASE[2], None))
    cases.append(("casts_edges", FIXTURES / "casts_edges.mind", [], 0, cast_words(cast_edges_oracle())))
    cases.append(("casts_exprs", FIXTURES / "casts_exprs.mind", [], 0, cast_words(cast_exprs_oracle())))
    cases.append(("bank_acc8_scalar_twin", HERE / "testdata/native_avx2/bank_acc8_scalar.mind", [], 0, BANK_ACC8))
    cases.append(("bank_forms_scalar", HERE / "testdata/native_avx2/bank_forms_scalar.mind", [], 0, BANK_FORMS))
    cases.append(("bank_canary_scalar", HERE / "testdata/native_avx2/bank_canary_scalar.mind", [], 0, BANK_CANARY))
    cases.append(("bank_epi_scalar", HERE / "testdata/native_avx2/bank_epi_scalar.mind", [], 0, EPI_FORMS))
    printing = PRINT_MARK in pathlib.Path(args.compiler).read_bytes()
    cases += [(f.stem, f, [], *ast.literal_eval(f.read_text().split("// expect: ")[1].split("\n")[0])) for f in sorted(FIXTURES.glob("print_*.mind"))]
    rows = [r for r in NATIVE_ONLY_CASES if args.only == r["family"]
            or (args.only is None and not r.get("compiler_driver"))]
    if args.only not in (None, "values"):
        cases = []
    ran = fail = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        native_only = run_native_only(rows, args.compiler, mindc, env, tmp)
        if native_only is None:
            return 2
        f, skipped = native_only
        fail, ran = fail + f, ran + len(rows) - skipped
        for cid, src, argv, want_exit, want_out in cases:
            case_dir = tmp / cid
            case_dir.mkdir()
            source = case_dir / "main.mind"
            text = src.read_text() if isinstance(src, pathlib.Path) else src + "\n"
            source.write_text(text)
            seen = []
            for native in (True, False):
                exe = case_dir / ("native" if native else "mlir")
                rc, err = _build(mindc, source, exe, native, env)
                if native and not printing and cid.startswith("print_"):  # the refusal IS the native result
                    seen.append("ok" if rc == 3 and "(lane.source.print)" in err else f"build={rc}"); continue
                if rc != 0:
                    seen.append(f"build={rc}: {err.strip().splitlines()[-1] if err.strip() else ''}")
                    continue
                r = _run(exe, argv)
                ok_out = want_out is None or r.stdout == want_out
                ok = r.returncode == want_exit and ok_out and r.stderr == b""
                seen.append("ok" if ok else f"exit={r.returncode} stdout={'=' if ok_out else '!='}")
            ran += 1
            passed = seen == ["ok", "ok"]
            fail += not passed
            print(f"  {'PASS' if passed else 'FAIL'}  {cid}: native={seen[0]} mlir={seen[1]}")
    print(f"SDLC-GATE native_lane_value_shapes ran={ran} fail={fail}")
    return 1 if fail or ran == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
