#!/usr/bin/env python3
"""Permanent battery for the self-host range-`for` loop.

`for VAR in LO..HI { BODY }` is desugared at PARSE time (parse_block_stmts ->
parse_for) into the HYGIENIC form (Step 2): three SIBLING statements — a hidden
counter `let CTR = LO`, a hidden bound `let BND = HI` (upper bound evaluated
ONCE), and `while CTR < BND { let VAR = CTR ; BODY ; CTR = CTR + 1 }` — reusing
the proven native-ELF `while` lowering. The per-iteration `let VAR = CTR` copy is
body-scoped (nb_emit_while), so VAR cannot escape the loop or be corrupted by a
body write. The upper bound is EXCLUSIVE (half-open). `break` exits with no
increment; an own-level `continue` uses the TOP-increment form so the counter
still advances before the back-edge.

This battery compiles each program through the pure-MIND native-ELF entry
(`selftest_native_elf_h`), RUNS the resulting x86-64 ELF, and asserts the exit
code (main's i64 return) matches an independent hand-computed reference. Every
reference was also checked against the Rust MLIR build of the same program where
that build accepts it; two do not build there (`0..n + 2` -> E5005, and the f64
carry through `continue` -> an mlir-opt block-argument type error).

Bound shapes: literal, variable, expression, parameter, call, negative, reversed,
empty, nested (incl. an inner bound reading the outer var). Exits: `break` and
`continue` at the loop level, inside a nested `if`, in an `else`, in inner and
outer loops, and AFTER a write to a loop-carried var — the exit edge must move
that var's live value into its loop slot (nb_exit_edge_copies), or the next
iteration / the code after the loop reads the stale one. The byte-neutral desugar
of an own-level `continue` steps VAR inside the branch (`if c { VAR = VAR + 1;
continue; }`), so every range-for `continue` depends on that edge. Two plain
`while` shapes pin the same root cause outside `for`, and a swap through two
carried vars pins that the edge moves run as a parallel copy.

Unconditional `break`/`continue` in a NESTED range-for leave the desugar's step
as a dead tail; every walk of the loop body must drop it the same way.

Env: MINDC_SO = pure-MIND self-host cdylib. Exit 0 = ALL PASS.
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SO = os.environ.get("MINDC_SO", str(HERE / "libmindc_mind.so"))

# 32-byte PT_NOTE anchor is irrelevant to EXECUTION (a NOTE segment is not loaded
# or run); a zero hash yields a runnable ELF whose exit code is main's return.
ZERO_HASH = b"\x00" * 32


def load():
    lib = ctypes.CDLL(SO)
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    return lib


def mind_elf(lib, src: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(ZERO_HASH, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    n = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), n) if n > 0 else b""


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # Hard timeout so a mis-lowered loop that fails to advance its induction var
    # (an infinite loop) fails LOUD instead of hanging the battery. Any real case
    # here terminates in well under a second.
    try:
        return subprocess.run([str(p)], timeout=10).returncode
    except subprocess.TimeoutExpired:
        return -1  # sentinel: never matches an expected exit code -> FAIL


# (name, source, expected exit code). main()->i64 return IS the process exit code
# (low 8 bits). Reference values computed by hand — the independent oracle.
CASES = [
    # count: i = 0,1,2,3,4 -> 5 increments
    ("count5", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { c = c + 1; } return c; }", 5),
    # sum of induction var: 0+1+2+3+4 = 10 (body READS i)
    ("sum10", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { s = s + i; } return s; }", 10),
    # non-zero lower bound: i = 2,3,4,5 -> 4 increments (EXCLUSIVE upper)
    ("range2_6", b"fn main() -> i64 { let mut c: i64 = 0; for i in 2..6 { c = c + 1; } return c; }", 4),
    # empty range: LO == HI -> body never runs
    ("empty0_0", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..0 { c = c + 1; } return c; }", 0),
    # empty range: LO > HI would be handled by `i < HI` false immediately
    ("empty5_5", b"fn main() -> i64 { let mut c: i64 = 0; for i in 5..5 { c = c + 1; } return c; }", 0),
    # nested: 3 * 3 = 9
    ("nested3x3", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { for j in 0..3 { c = c + 1; } } return c; }", 9),
    # nested with induction read: sum over i of (i * 3) inner adds? inner adds i each: 3*(0+1+2)=9
    ("nested_read", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..3 { s = s + i; } } return s; }", 9),
    # body reads induction var via accumulation: 0+1+2+3 = 6
    ("reads_i", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { c = c + i; } return c; }", 6),
    # break at i==2: c increments for i=0,1 then break -> 2
    ("break_at2", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { if i == 2 { break; } c = c + 1; } return c; }", 2),
    # break never fires (cond false): full 5 increments
    ("break_never", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { if i == 99 { break; } c = c + 1; } return c; }", 5),
    # continue at i==2 (standard for-continue: increment STILL runs): count i in {0,1,3,4} -> 4
    ("continue_at2", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { if i == 2 { continue; } c = c + 1; } return c; }", 4),
    # continue + induction sum: 0+1+3+4 = 8 (skip i==2, but loop still advances)
    ("continue_sum", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i == 2 { continue; } s = s + i; } return s; }", 8),
    # continue in the else arm shape: skip even? here skip i<2 -> add i in {2,3,4} = 9
    ("continue_ge", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i < 2 { continue; } s = s + i; } return s; }", 9),
    # larger range sanity
    ("count10", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..10 { c = c + 1; } return c; }", 10),
    # for driving an accumulator that feeds the return via a helper param path
    ("call_in_body", b"fn inc(x: i64) -> i64 { return x + 1; } fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { c = inc(c); } return c; }", 4),
    # ---- bound shapes ----
    # negative lower bound: 100 + (-3 -2 -1 +0 +1) = 95
    ("neg_lo", b"fn main() -> i64 { let mut s: i64 = 100; for i in -3..2 { s = s + i; } return s; }", 95),
    # both bounds negative: i = -5,-4,-3,-2 -> 4 iterations
    ("neg_both", b"fn main() -> i64 { let mut c: i64 = 0; for i in -5..-1 { c = c + 1; } return c; }", 4),
    # reversed range (LO > HI): empty
    ("reversed", b"fn main() -> i64 { let mut c: i64 = 0; for i in 5..2 { c = c + 1; } return c; }", 0),
    # reversed negative range: empty
    ("reversed_neg", b"fn main() -> i64 { let mut c: i64 = 0; for i in -1..-3 { c = c + 1; } return c; }", 0),
    # variable bounds: 2+3+4+5 = 14
    ("var_bounds", b"fn main() -> i64 { let a: i64 = 2; let b: i64 = 6; let mut s: i64 = 0; for i in a..b { s = s + i; } return s; }", 14),
    # variable upper bound holding 0: empty
    ("var_empty", b"fn main() -> i64 { let n: i64 = 0; let mut c: i64 = 0; for i in 0..n { c = c + 1; } return c; }", 0),
    # expression bound n + 2 = 5: 0+1+2+3+4 = 10
    ("expr_bound", b"fn main() -> i64 { let n: i64 = 3; let mut s: i64 = 0; for i in 0..n + 2 { s = s + i; } return s; }", 10),
    # parameter bound: f(5) = 0+1+2+3+4 = 10
    ("param_bound", b"fn f(n: i64) -> i64 { let mut s: i64 = 0; for i in 0..n { s = s + i; } return s; } fn main() -> i64 { return f(5); }", 10),
    # inner bound reads the outer induction var: 0+1+2+3 = 6
    ("nested_dep", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { for j in 0..i { c = c + 1; } } return c; }", 6),
    # ---- break / continue, every bound shape ----
    # A loop-carried var's write on the path to a `continue`/`break` must reach the
    # next iteration / the code after the loop (the exit edge carries it).
    # write then continue: 0+1+100+3+4 = 108
    ("continue_write", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i == 2 { s = s + 100; continue; } s = s + i; } return s; }", 108),
    # write then break: 0+1+2+50 = 53
    ("break_write", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..9 { if i == 3 { s = s + 50; break; } s = s + i; } return s; }", 53),
    # continue with a variable bound: 0+2+3+4+5 = 14
    ("continue_var_bound", b"fn main() -> i64 { let n: i64 = 6; let mut s: i64 = 0; for i in 0..n { if i == 1 { continue; } s = s + i; } return s; }", 14),
    # continue over a negative range: 50 + (-3 -2 -1 +1 +2) = 47
    ("continue_neg", b"fn main() -> i64 { let mut s: i64 = 50; for i in -3..3 { if i == 0 { continue; } s = s + i; } return s; }", 47),
    # continue with a call bound (hidden-counter form): 0+2+3+4 = 9
    ("continue_call_bound", b"fn hi() -> i64 { return 5; } fn main() -> i64 { let mut s: i64 = 0; for i in 0..hi() { if i == 1 { continue; } s = s + i; } return s; }", 9),
    # continue in the else arm: i = 0,1,2 add 11 each -> 33
    ("continue_else", b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..6 { if i < 3 { s = s + 1; } else { continue; } s = s + 10; } return s; }", 33),
    # two carried vars, one written before the branch: a = 6, b = 7 + 5 = 12 -> 72
    ("continue_two_vars", b"fn main() -> i64 { let mut a: i64 = 0; let mut b: i64 = 0; for i in 0..6 { a = a + 1; if i == 2 { b = b + 7; continue; } b = b + 1; } return a * 10 + b; }", 72),
    # continue/break nested in a second if: skip i = 3,4 -> 7 ; stop at 5 -> 5
    ("continue_nested_if", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..9 { if i > 2 { if i < 5 { continue; } } c = c + 1; } return c; }", 7),
    ("break_nested_if", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..9 { if i > 2 { if i == 5 { break; } } c = c + 1; } return c; }", 5),
    # nested loops: inner continue 3*3 = 9 ; inner break 3*2 = 6 ; outer continue 3*3 = 9 ;
    # outer break 2*3 = 6 ; both levels continue: i in {0,2,3}, j in {0,1,3} -> 27
    ("nested_inner_continue", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { for j in 0..4 { if j == 1 { continue; } c = c + 1; } } return c; }", 9),
    ("nested_inner_break", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { for j in 0..4 { if j == 2 { break; } c = c + 1; } } return c; }", 6),
    ("nested_outer_continue", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { if i == 1 { continue; } for j in 0..3 { c = c + 1; } } return c; }", 9),
    ("nested_outer_break", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { if i == 2 { break; } for j in 0..3 { c = c + 1; } } return c; }", 6),
    ("nested_both_continue", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..4 { if i == 1 { continue; } for j in 0..4 { if j == 2 { continue; } c = c + i + j; } } return c; }", 27),
    # body writes VAR, continue BEFORE that write (hidden-counter form; the body-local
    # `let VAR = CTR` copy needs no exit move): i in {0,1,3,4} each + 10 -> 48
    ("body_assign_continue", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { if i == 2 { continue; } i = i + 10; c = c + i; } return c; }", 48),
    # f64 carried through a continue edge: 4 * 1.0 + 10.0 = 14.0 > 13.5 -> 1 (dropped write -> 4.0 -> 2)
    ("continue_f64", b"fn main() -> i64 { let mut a: f64 = 0.0; for i in 0..5 { if i == 2 { a = a + 10.0; continue; } a = a + 1.0; } if a > 13.5 { return 1; } return 2; }", 1),
    # The same exit edges in a plain `while` (one root cause): x is written before the
    # exit and again after it, so the loop slot differs from the live value at the exit.
    # continue: x ends 11 after 5 counted rounds -> 115 (pre-fix: HANG)
    ("while_continue_after_write", b"fn main() -> i64 { let mut x: i64 = 0; let mut n: i64 = 0; while x < 10 { x = x + 1; if x == 3 { continue; } x = x + 1; n = n + 1; } return x * 10 + n; }", 115),
    # break: x = 3 at the break (pre-fix: 2)
    ("while_break_after_write", b"fn main() -> i64 { let mut x: i64 = 0; while x < 100 { x = x + 1; if x == 3 { break; } x = x + 1; } return x; }", 3),
    # Exit-ONLY branches the old branch-exit guard admitted, but which still read a stale
    # loop slot because an INNER loop wrote the carried var first (silent wrong values on
    # the pre-fix compiler: 36 and 203). References from the Rust MLIR build.
    ("while_inner_loop_then_continue", b"fn main() -> i64 { let mut a: i64 = 0; let mut x: i64 = 0; while a < 4 { a = a + 1; let mut k: i64 = 0; while k < 2 { x = x + 1; k = k + 1; } if a == 2 { continue; } x = x + 10; } return x; }", 38),
    ("while_inner_break_outer_var", b"fn main() -> i64 { let mut a: i64 = 0; let mut b: i64 = 0; let mut x: i64 = 0; while a < 3 { a = a + 1; while b < 10 { b = b + 1; x = x + b; if b == 2 { break; } } if a == 2 { continue; } x = x + 100; } return x % 256; }", 255),
    # positive control for REFUSE_CASES below: the same loop without the shadowing
    # `let` compiles and runs (x == 5), so the refusal is about the shadow alone.
    ("shadow_control", b"fn main() -> i64 { let mut x: i64 = 0; let mut a: i64 = 0; while a < 5 { a = a + 1; x = x + 1; if a == 3 { continue; } } return x; }", 5),
    # A branch swaps two carried vars through bare-ident copies, then exits: at the
    # exit `a` is bound to b's loop slot and `b` to a's, so the edge moves are a
    # cycle and must run as a PARALLEL copy (sequential moves give a == b).
    # a,b = 1,2 -> (2,12) -> swap (12,2) -> (13,12) -> (14,22): 14*16+22 = 246 ; the
    # break leaves (12,2): 194. References from the Rust MLIR build.
    ("swap_in_branch_continue", b"fn main() -> i64 { let mut a: i64 = 1; let mut b: i64 = 2; let mut i: i64 = 0; while i < 4 { i = i + 1; if i == 2 { let t: i64 = a; a = b; b = t; continue; } a = a + 1; b = b + 10; } return a * 16 + b; }", 246),
    ("swap_in_branch_break", b"fn main() -> i64 { let mut a: i64 = 1; let mut b: i64 = 2; let mut i: i64 = 0; while i < 4 { i = i + 1; if i == 2 { let t: i64 = a; a = b; b = t; break; } a = a + 1; b = b + 10; } return a * 16 + b; }", 194),
    ("for_swap_in_branch_continue", b"fn main() -> i64 { let mut a: i64 = 1; let mut b: i64 = 2; for i in 0..4 { if i == 1 { let t: i64 = a; a = b; b = t; continue; } a = a + 1; b = b + 10; } return a * 16 + b; }", 246),
    # ---- unconditional exits in a NESTED range-for ----
    # The desugar appends `VAR = VAR + 1` after the body, so an unconditional
    # `break`/`continue` leaves a dead tail. The loop's own emit drops it
    # (nb_truncate_dead); the frame count and an enclosing loop's carry pre-walk
    # must drop it too, or the outer loop's post_ids land past the slots the emit
    # writes and the outer counter never advances (pre-fix: HANG on all four).
    ("nested_uncond_break", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { for j in 0..3 { c = c + 1; break; } } return c; }", 3),
    ("nested_uncond_continue", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { for j in 0..4 { c = c + 1; continue; } } return c; }", 12),
    ("nested_dead_tail_continue", b"fn main() -> i64 { let mut c: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; let mut j: i64 = 0; while j < 3 { j = j + 1; c = c + 1; continue; c = c + 100; } } return c; }", 9),
    ("nested_dead_tail_break", b"fn main() -> i64 { let mut c: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; let mut j: i64 = 0; while j < 3 { j = j + 1; c = c + 1; break; c = c + 100; } } return c; }", 3),
    # the same dead tail with the inner loop inside an `if` of the outer body: the
    # outer pre-walk counts the if through nb_count_stmt (its merge slots and next_id)
    ("nested_uncond_break_in_if", b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { if i >= 0 { for j in 0..3 { c = c + 1; break; } } c = c + 10; } return c; }", 33),
    ("nested_dead_tail_in_if_merge", b"fn main() -> i64 { let mut x: i64 = 0; let mut i: i64 = 0; while i < 3 { i = i + 1; if i > 1 { let mut k: i64 = 0; while k < 5 { k = k + 1; x = x + 1; break; x = x + 100; } } x = x + 10; } return x; }", 32),
]

# A `let` in the loop body that SHADOWS a carried var makes the exit edge's name
# lookup ambiguous (the let-env cannot tell the shadow from the carried binding), so
# a needed move is refused fail-closed (empty ELF) instead of copying the shadow.
# Source-level reference for this one is x == 5; a copied shadow would give 52.
REFUSE_CASES = [
    ("shadow_at_exit", b"fn main() -> i64 { let mut x: i64 = 0; let mut a: i64 = 0; while a < 5 { a = a + 1; x = x + 1; if a == 3 { let x: i64 = 50; continue; } } return x; }"),
]

# ---------------------------------------------------------------------------
# GATED hygiene cases (audit #4 / #5i) — SELF-HOST leg of the cross-substrate
# differential. The Rust compiler now branches range-`for` on a hygiene gate
# (`src/eval/lower.rs` For arm): when END re-evaluation is observable, VAR
# shadows an outer binding, or the body assigns VAR, it emits a hygienic form
# (span-unique counter, END pre-lowered ONCE, per-iteration `let mut VAR`
# copy). The interpreter-oracle and Rust-compiled-ELF legs (asserting
# interp/rust == hand reference) live in `tests/for_hygiene_run.rs`; THIS
# battery adds the third substrate — the self-host native-ELF.
#
# The self-host compiler still desugars range-`for` at PARSE time into the
# naive `let mut VAR=LO; while VAR<HI {B; VAR=VAR+1}` form with NO hygiene gate
# (that is Step 2 of the C/D redesign — self-host nb while-body let-env scoping
# then an unconditional hygienic form). So each gated case is EXPECTED to
# diverge on self-host TODAY. Those divergences are recorded as LOUD xfails:
# an xfail that unexpectedly PASSES (self-host silently gained the fix) fails
# the battery so the bookkeeping never drifts stale.
#
# (name, source, reference-exit-code, self_host_xfail, note)
#   self_host_xfail = True  -> self-host is expected to DIVERGE from reference
#                              (wrong value, fail-closed empty ELF, or a hang).
#   self_host_xfail = False -> self-host coincidentally matches (a PURE end
#                              call re-evaluated per-iter yields the same bound).
# Step 2 (this change) landed the self-host HYGIENIC range-`for` desugar: the
# native-ELF parse_for now emits a hidden counter `let CTR = LO`, a bound
# `let BND = HI` (END evaluated ONCE), and `while CTR < BND { let VAR = CTR;
# BODY; CTR = CTR + 1 }`, and nb_emit_while scopes the while body's let-env so the
# per-iteration `let VAR = CTR` copy dies at loop exit. So the self-host substrate
# now MATCHES the interpreter/hygienic-Rust reference on all three shapes that used
# to diverge — the former xfails are real PASSes (`self_host_xfail = False`). An
# xfail flag left True here would fail LOUD (XPASS), so the bookkeeping stays honest.
GATED_CASES = [
    # Loop var shadows an outer `let i = 100`; VAR must NOT escape the loop.
    # Hygienic: the per-iteration `let i = CTR` copy is body-scoped, so post-loop
    # `return i` resolves the outer binding -> 100.
    ("shadow_outer",
     b"fn main() -> i64 { let i: i64 = 100; for i in 0..3 { } return i; }",
     100, False, "VAR stays loop-local, outer binding preserved"),
    # Body assigns the loop var; the write hits the per-iteration copy, not the
    # hidden counter. 3 iterations -> c == 3.
    ("body_assign_iters",
     b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..3 { i = i + 5; c = c + 1; } return c; }",
     3, False, "body write to VAR hits per-iteration copy, counter advances"),
    # END reads a body-assigned var; END is pre-lowered ONCE into the hidden bound,
    # so the loop terminates at the frozen bound 3 -> c == 3 (no infinite loop).
    ("end_reads_body_assigned",
     b"fn main() -> i64 { let mut h: i64 = 3; let mut c: i64 = 0; for i in 0..h { h = h + 1; c = c + 1; } return c; }",
     3, False, "END evaluated once into hidden bound, loop terminates"),
    # END is a call. The hygienic form pre-lowers `hi()` once into the bound; the
    # value is 3 -> c == 3. Guards that a call in END compiles + runs.
    ("end_call",
     b"fn hi() -> i64 { return 3; } fn main() -> i64 { let mut c: i64 = 0; for i in 0..hi() { c = c + 1; } return c; }",
     3, False, "END call pre-lowered once into hidden bound"),
]

# DESUGAR_PAIRS (byte-identity of `for` vs the NAIVE explicit `let mut i; while
# i<HI { B; i=i+1 }` form) is RETIRED as of Step 2: the hygienic desugar emits a
# hidden counter + a pre-lowered bound + a body-local element copy, so it is
# deliberately NO LONGER byte-identical to the naive while form the pairs encode.
# Value-correctness across the same shapes is covered by CASES above; the
# hygiene-specific behavior is covered by GATED_CASES.
DESUGAR_PAIRS = []


def main() -> int:
    lib = load()
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, src, want in CASES:
            elf = mind_elf(lib, src)
            if not elf:
                print(f"  FAIL  {name}: native entry failed closed (empty ELF)")
                rc = 1
                continue
            got = run_elf(elf, tmp) & 0xFF
            if got != want:
                print(f"  FAIL  {name}: exit {got}, want {want}")
                rc = 1
            else:
                print(f"  PASS  {name}: exit {got} == {want}")

        for name, src in REFUSE_CASES:
            elf = mind_elf(lib, src)
            if elf:
                print(f"  FAIL  {name}: emitted a {len(elf)} B ELF (exit {run_elf(elf, tmp) & 0xFF}), "
                      f"want a fail-closed refusal (empty ELF)")
                rc = 1
            else:
                print(f"  PASS  {name}: refused fail-closed (empty ELF)")

        for name, a, b in DESUGAR_PAIRS:
            ea, eb = mind_elf(lib, a), mind_elf(lib, b)
            if ea and eb and ea == eb:
                print(f"  PASS  {name}: `for` ELF byte-identical to explicit while-desugar ({len(ea)} B)")
            else:
                print(f"  FAIL  {name}: for form ({len(ea)} B) != explicit while-desugar ({len(eb)} B)")
                rc = 1

        # ---- GATED hygiene cases: self-host substrate leg + xfail bookkeeping.
        xfail_expected = 0
        xfail_unexpected_pass = 0
        for name, src, want, xfail, note in GATED_CASES:
            elf = mind_elf(lib, src)
            got = (run_elf(elf, tmp) & 0xFF) if elf else None  # None = fail-closed
            matches = got == want
            if xfail:
                xfail_expected += 1
                if matches:
                    # An xfail that PASSES means self-host silently gained the
                    # fix — the bookkeeping is stale. Fail LOUD (never silent).
                    print(f"  XPASS {name}: self-host UNEXPECTEDLY matched ref {want} "
                          f"(exit {got}) — Step-2 landed? update GATED_CASES xfail flag")
                    xfail_unexpected_pass += 1
                    rc = 1
                else:
                    shown = "fail-closed" if got is None else f"exit {got}"
                    print(f"  XFAIL {name}: self-host diverges as expected "
                          f"({shown} != ref {want}) — {note}")
            else:
                if matches:
                    print(f"  PASS  {name}: self-host exit {got} == ref {want} — {note}")
                else:
                    shown = "fail-closed (empty ELF)" if got is None else f"exit {got}"
                    print(f"  FAIL  {name}: self-host {shown}, want {want} — {note}")
                    rc = 1
        print(f"  xfail bookkeeping: {xfail_expected} expected self-host divergences "
              f"(Step-2 gap), {xfail_unexpected_pass} unexpected passes")

    print("ALL PASS  (range-for: value-correct over every bound shape and exit edge; "
          "shadowed exit refused; gated hygiene self-host xfails accounted)" if rc == 0
          else "FAIL  (range-for battery had failures)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
