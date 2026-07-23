#!/usr/bin/env python3
"""Permanent battery for the self-host range-`for` loop.

`for VAR in LO..HI { BODY }` is desugared at PARSE time (parse_block_stmts ->
parse_for) into the two SIBLING statements `let mut VAR: i64 = LO;` and
`while VAR < HI { BODY ; VAR = VAR + 1; }` — reusing the proven native-ELF `while`
lowering with ZERO new emit code. The upper bound is EXCLUSIVE (half-open).
`break` exits with no increment; an own-level `continue` is rewritten to
`VAR = VAR + 1; continue` so standard for-continue semantics hold (the increment
runs before the back-edge).

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

# byte-identity: each `for` vs its explicit `let mut i:i64=LO; while i<HI { B; i=i+1; }` form.
DESUGAR_PAIRS = [
    ("count5_vs_while",
     b"fn main() -> i64 { let mut c: i64 = 0; for i in 0..5 { c = c + 1; } return c; }",
     b"fn main() -> i64 { let mut c: i64 = 0; let mut i: i64 = 0; while i < 5 { c = c + 1; i = i + 1; } return c; }"),
    ("sum10_vs_while",
     b"fn main() -> i64 { let mut s: i64 = 0; for i in 0..5 { s = s + i; } return s; }",
     b"fn main() -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; while i < 5 { s = s + i; i = i + 1; } return s; }"),
    ("range2_6_vs_while",
     b"fn main() -> i64 { let mut c: i64 = 0; for i in 2..6 { c = c + 1; } return c; }",
     b"fn main() -> i64 { let mut c: i64 = 0; let mut i: i64 = 2; while i < 6 { c = c + 1; i = i + 1; } return c; }"),
]


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

    print("ALL PASS  (range-for: value-correct + byte-identical to let+while desugar)" if rc == 0
          else "FAIL  (range-for battery had failures)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
