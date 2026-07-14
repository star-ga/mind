#!/usr/bin/env python3
"""RI-D2 S-C1: FLOAT call-RETURN dtype through the native-ELF general nb_expr lowering.

The connecting construct after float-let (S-A) and float-param (S-B): a CALL to a fn
that returns f64/f32 now flows through GENERAL nb_expr lowering. Three coupled parts:

  * Registry: `frt_build` re-lexes the module (mirroring srt_build) and records each
    `fn NAME (...) -> RETTYPE {` def's return dtype (FLOAT for f64/f32, else INT). A
    `;`-terminated extern with no body is NOT recorded as float.
  * Callee return: nb_finish_body loads a FLOAT-returning fn's tail result into xmm0
    (movsd) per the SysV AMD64 return-in-xmm0 convention (rax for int/pointer returns).
  * Caller spill: nb_expr's call arm spills a float-returning callee's result from xmm0
    (not rax) into the call's dst slot and TAGS that slot FLOAT, so a downstream expr
    (`g() + 1.5`) or a `let r: f64 = g();` binding sees the call result as float and
    routes the SSE2 f64 encoders. nb_node_dtype_tbl's new ast_call arm classifies a
    call by its callee's registered return dtype.

`selftest_native_elf_fp_call(src, len)` parses a >=2-fn source (a float-returning callee
plus a caller that consumes `g()` in a float expr), lowers ALL fns through nb_lower_all
with inter-fn rel32 patching, synthesizes an entry stub that calls the LAST fn (the
caller), truncates its xmm0 result to rax, and exits(trunc(value)).

No frozen float oracle exists (the deleted Rust native backend returned Unsupported(
ConstF64)), so byte-identity is impossible here by construction. The oracle is EXECUTION
CORRECTNESS: emit the ELF, run it, assert exit == trunc(value). A wrong rax-vs-xmm0 spill
(caller or callee), a missing FLOAT tag, or a mis-patched call faults or returns the
wrong integer, so a correct exit is non-fakeable.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_fp_call_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402


def mind_fp_call_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf_fp_call
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    buf = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(src.encode()))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_fp_call.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = str(resolve_so())
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_fp_call"):
        print("FAIL  selftest_native_elf_fp_call: symbol absent (S-C1 not built)")
        return 1

    # (source, expected exit = int(trunc(caller's float result))). The LAST fn is the
    # caller. All literals are exactly-representable dyadic values so the decimal->IEEE
    # path is exact.
    cases = [
        # immediate-binop: the call result is consumed directly by a float binop, so the
        # xmm0 spill + the ast_call FLOAT classification both fire on the call operand.
        ("fn g() -> f64 { 2.5 } fn f() -> f64 { g() + 1.5 }", 4),   # 2.5 + 1.5 = 4.0
        # let-bound: proves the SLOT-tag path — the call result flows through a `let r`
        # binding (tagged FLOAT) before the binop, not just an immediate operand.
        ("fn g() -> f64 { 3.0 } fn h() -> f64 { let r: f64 = g(); r + 1.5 }", 4),  # 4.5
        # extra op/value coverage: subtraction of the call result.
        ("fn g() -> f64 { 9.0 } fn f() -> f64 { g() - 2.5 }", 6),   # 9.0 - 2.5 = 6.5 -> 6
        # ---- RI-D2 S-C2: FLOAT call-ARGUMENTS marshalled in the SSE arg pool ----
        # single float arg: 3.0 flows into g's xmm0 (S-C2 arg), spilled to g's param
        # slot (S-B param), returned via xmm0 (S-C1 return). Non-fakeable end to end.
        ("fn g(x: f64) -> f64 { x * 2.0 } fn f() -> f64 { g(3.0) }", 6),  # 3.0*2.0=6.0
        # MIXED two-pool proof: n->GP rdi, x->SSE xmm0 on INDEPENDENT counters. A single-
        # pool ABI would put x in rsi (GP) or fault; only a correct two-pool split exits 4.
        ("fn g(n: i64, x: f64) -> f64 { x + 1.0 } fn f() -> f64 { g(5, 3.0) }", 4),  # 3+1=4
        # two float args: a->xmm0, b->xmm1 (sse_k advances 0->1 while gp_k stays 0).
        ("fn g(a: f64, b: f64) -> f64 { a - b } fn f() -> f64 { g(9.0, 2.5) }", 6),  # 9-2.5=6.5
        # ---- #152: SysV two-pool STACK-ARG classification (caller must not push by
        # position). A leading f64 shifts INT positions: the 7th int (i, gp_k=6) is a
        # stack arg. The buggy positional caller pushed args[6:] (h,i,j) and mis-padded,
        # so the callee's gp_k-keyed read of i saw h; the fix pushes ONLY true overflow
        # args (int gp>=6 / float sse>=8) by class counter. i==100 selects 42.0 vs 7.0,
        # so a wrong stack value flips the exit. All-int sibling (below) proves the
        # byte-identical path is intact.
        ("fn f(a: f64, b: i64, c: i64, d: i64, e: i64, g: i64, h: i64, i: i64, j: i64) -> f64 "
         "{ if i == 100 { return 42.0; } return 7.0; } "
         "fn caller() -> f64 { f(1.0, 11, 22, 33, 44, 55, 66, 100, 142) }", 42),
        # all-int control (no leading float): position == gp index, exercises the
        # byte-identical all-int stack path — must stay correct.
        ("fn f(b: i64, c: i64, d: i64, e: i64, g: i64, h: i64, i: i64, j: i64) -> f64 "
         "{ if i == 100 { return 42.0; } return 7.0; } "
         "fn caller() -> f64 { f(11, 22, 33, 44, 55, 66, 100, 142) }", 42),
        # interleaved GP+SSE overflow: 9 floats (f8 -> sse overflow, stack) + 7 ints
        # (i6 -> gp overflow, stack) share the stack region in arg order; i6 selects
        # 42.0 vs 7.0. Proves caller push order + callee shared stack-rank agree.
        ("fn q(f0: f64, f1: f64, f2: f64, f3: f64, f4: f64, f5: f64, f6: f64, f7: f64, f8: f64, "
         "i0: i64, i1: i64, i2: i64, i3: i64, i4: i64, i5: i64, i6: i64) -> f64 "
         "{ if i6 == 77 { return 42.0; } return 7.0; } "
         "fn caller() -> f64 { q(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,42.0, 10,20,30,40,50,60,77) }", 42),
        # ---- #166: explicit `return <float>` must place the result in xmm0 (SysV), not
        # rax. Pre-fix nb_stmt's ast_return arm unconditionally loaded rax; the callee then
        # left a STALE xmm0 and the caller's xmm0 spill read garbage. `return b` (not the
        # xmm0-resident `a`) discriminates: pre-fix returned trunc(stale xmm0=a=4), fix
        # returns 9. A wrong register file here flips the exit.
        ("fn g(a: f64, b: f64) -> f64 { return b; } fn f() -> f64 { g(4.0, 9.0) }", 9),
        # #166 stack-passed 9th float param: params 0..7 in xmm0..7, s (idx 8) on the stack.
        # Pre-fix `return s` via rax left xmm0 = p0 = 1.0 (stale), so exit was 1; fix returns
        # s = 42.0 in xmm0 -> exit 42.
        ("fn g(a: f64, b: f64, c: f64, d: f64, e: f64, p: f64, q: f64, r: f64, s: f64) -> f64 "
         "{ return s; } fn caller() -> f64 { g(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,42.0) }", 42),
        # ---- #167: an assign/rebind (`y = <float expr>`) must re-tag the rebound slot FLOAT,
        # and the fp-binop producer must tag its dst FLOAT. Pre-fix the rebound slot read INT,
        # so `g(y)` marshalled the float arg in the GP pool (rdi) while the callee read xmm.
        # The intervening `let z = <float>` clobbers xmm0 AFTER y is set, exposing the wrong
        # register file: pre-fix returned trunc(stale xmm0=z), fix marshals y correctly.
        ("fn g(x: f64) -> f64 { x } "
         "fn f() -> f64 { let mut y: f64 = 0.0; y = 3.0 * 2.0; let z: f64 = 1.0 + 1.0; g(y) }", 6),
        # #167 accumulator variant (`acc = acc + 5.0`): pre-fix returned trunc(stale xmm0 = z
        # = 3+4 = 7); fix marshals acc = 6.0 in the SSE pool -> exit 6.
        ("fn g(x: f64) -> f64 { x } "
         "fn f() -> f64 { let mut acc: f64 = 1.0; acc = acc + 5.0; let z: f64 = 3.0 + 4.0; g(acc) }", 6),
        # ---- #168: a FLOAT-operand COMPARISON must lower via ucomisd + an UNSIGNED setcc, not
        # a signed-integer compare of the raw IEEE-754 bits (which is INVERTED for two negative
        # operands — IEEE-754 is sign-magnitude). Negatives are formed by `0.0 - x` (a plain
        # negative float literal is a separate, unrelated emitter bug). a=-1.0, b=-2.0:
        #   a >  b is TRUE (1)  — pre-fix signed-bit compare gave 0 (inverted).
        ("fn f() -> f64 { let a: f64 = 0.0 - 1.0; let b: f64 = 0.0 - 2.0; "
         "if a > b { return 1.0; } return 0.0; }", 1),
        #   a <  b is FALSE (0) — pre-fix gave 1 (inverted).
        ("fn f() -> f64 { let a: f64 = 0.0 - 1.0; let b: f64 = 0.0 - 2.0; "
         "if a < b { return 1.0; } return 0.0; }", 0),
        #   a >= b is TRUE (1)  — pre-fix gave 0 (inverted).
        ("fn f() -> f64 { let a: f64 = 0.0 - 1.0; let b: f64 = 0.0 - 2.0; "
         "if a >= b { return 1.0; } return 0.0; }", 1),
        #   a <= b at equality (-2.0 <= -2.0) is TRUE (1) — boundary/ZF predicate.
        ("fn f() -> f64 { let a: f64 = 0.0 - 2.0; let b: f64 = 0.0 - 2.0; "
         "if a <= b { return 1.0; } return 0.0; }", 1),
        #   a != b is TRUE (1) — inequality over negatives.
        ("fn f() -> f64 { let a: f64 = 0.0 - 1.0; let b: f64 = 0.0 - 2.0; "
         "if a != b { return 1.0; } return 0.0; }", 1),
        #   positive-operand control: a=2.0 > b=1.0 is TRUE (1) — the case that was already
        #   correct as an integer-bit compare, must stay correct.
        ("fn f() -> f64 { let a: f64 = 2.0; let b: f64 = 1.0; "
         "if a > b { return 1.0; } return 0.0; }", 1),
        #   equality of equal positives (2.0 == 2.0) is TRUE (1).
        ("fn f() -> f64 { let a: f64 = 2.0; let b: f64 = 2.0; "
         "if a == b { return 1.0; } return 0.0; }", 1),
        # ---- #170: a NEGATIVE float LITERAL (unary minus over a float literal) must
        # lower in the native-ELF emitter, not SIGSEGV. Pre-fix nb_expr / nb_count_expr /
        # nb_ccount_expr / nb_reach_expr had no ast_neg arm, so `-1.0` fell to the binop
        # tail and null-deref'd on ast_child1 == 0. nb_expr now negates a FLOAT operand via
        # `0.0 - operand` (subsd) and tags the result FLOAT; nb_node_dtype[_tbl] classify a
        # neg by its operand dtype so a later float op/compare over `-x` routes the SSE2
        # path (a signed compare of negated IEEE-754 bits would invert, #168).
        ("fn f() -> f64 { let a: f64 = -1.0; a + 3.0 }", 2),            # -1.0 + 3.0 = 2.0
        ("fn f() -> f64 { let a: f64 = -2.5; a }", 254),               # trunc(-2.5) = -2 -> exit 254
        ("fn g(x: f64) -> f64 { x } fn f() -> f64 { g(-2.0) }", 254),  # -2.0 float arg -> -2
        # negative float operand in a comparison, result routed as a float (1.0/0.0):
        ("fn f() -> f64 { let a: f64 = -1.0; let b: f64 = 1.0; "
         "if a < b { return 1.0; } return 0.0; }", 1),                 # -1.0 < 1.0 TRUE
        ("fn f() -> f64 { let a: f64 = -1.0; let b: f64 = 1.0; "
         "if a > b { return 1.0; } return 0.0; }", 0),                 # -1.0 > 1.0 FALSE
        # ---- #171: a FLOAT-typed value-if (`if C { A } else { B }` with float branches)
        # must (a) not SIGSEGV and (b) tag its merged dst slot FLOAT so a downstream float
        # op routes the SSE2 path. Pre-fix `let r = if..; r + 1.0` returned a wrong value
        # (dst read INT). nb_node_dtype[_tbl]'s ast_if arm classifies a value-if by its
        # then-branch tail dtype, so nb_stmt's let arm tags r FLOAT.
        ("fn f() -> f64 { let c: i64 = 1; if c == 1 { 2.0 } else { 3.0 } }", 2),   # tail value-if
        ("fn f() -> f64 { let c: i64 = 0; if c == 1 { 2.0 } else { 3.0 } }", 3),   # else branch
        ("fn f() -> f64 { let c: i64 = 1; let r: f64 = if c == 1 { 2.0 } else { 3.0 }; "
         "r + 1.0 }", 3),                                                          # dst FLOAT-tagged
    ]
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for src, expected in cases:
            elf = mind_fp_call_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {src!r}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {src!r:>58} -> "
                f"exit={got} expected={expected} "
                f"(elf {len(elf)}B, SSE2 native, zero MLIR/LLVM)"
            )
    if all_ok:
        print(
            "ALL PASS  FLOAT call-RETURN dtype flows through general nb_expr lowering — "
            "frt_build registers each fn's declared return dtype, nb_finish_body returns a "
            "float fn's result in xmm0 (SysV), nb_expr's call arm spills a float-returning "
            "callee from xmm0 and tags the dst slot FLOAT, and nb_node_dtype_tbl's ast_call "
            "arm classifies a call by its callee's return dtype, native-ELF end to end with "
            "zero MLIR/LLVM (RI-D2 S-C1); AND FLOAT call-ARGUMENTS marshalled in the SSE arg "
            "pool (nb_call_args records each arg's dtype, nb_emit_argregs loads a float arg "
            "into xmm{sse_k} on an independent counter while int/ptr args keep the GP pool — "
            "the SysV two-pool caller ABI mirroring nb_emit_params, RI-D2 S-C2); AND the "
            "native-ELF float-ABI fixes: #166 explicit `return <float>` in xmm0 (not rax), "
            "#167 assign/rebind + fp-binop dst re-tag FLOAT (so a reassigned float arg keeps "
            "the SSE pool), #168 float comparison via ucomisd + unsigned setcc (not a signed "
            "IEEE-754-bits compare, which inverts for negative operands)"
        )
        return 0
    print("FAIL  float call-return dtype gate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
