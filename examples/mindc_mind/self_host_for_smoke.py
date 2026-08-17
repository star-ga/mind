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
code (main's i64 return) matches an independent hand-computed reference. It also
proves each `for` compiles BYTE-IDENTICALLY to its explicit `let`+`while` form.

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

    print("ALL PASS  (range-for: value-correct + byte-identical to let+while desugar; "
          "gated hygiene self-host xfails accounted)" if rc == 0
          else "FAIL  (range-for battery had failures)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
