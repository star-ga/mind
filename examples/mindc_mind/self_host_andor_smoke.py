#!/usr/bin/env python3
"""Permanent battery for the self-host `&&` / `||` short-circuit operators.

`&&` / `||` are desugared at PARSE time (parse_pratt_ns) to a short-circuit
value-if — `A && B` -> `if A { B } else { 0 }`, `A || B` -> `if A { 1 } else { B }`
— reusing the existing value-if native-ELF lowering (nb_if_value). This battery
compiles each program through the pure-MIND native-ELF entry
(`selftest_native_elf_h`), RUNS the resulting x86-64 ELF, and asserts the exit
code (main's i64 return) matches an independent reference. It also proves each
`A && B` / `A || B` compiles BYTE-IDENTICALLY to its explicit `if`-desugar.

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
    return subprocess.run([str(p)]).returncode


# (name, source, expected exit code). Each program's main()->i64 return IS the
# process exit code (low 8 bits). Reference values are computed by hand from the
# boolean truth tables — the independent oracle.
CASES = [
    ("and_TT", b"fn main() -> i64 { if 1 > 0 && 1 > 0 { return 1; } else { return 0; } }", 1),
    ("and_TF", b"fn main() -> i64 { if 1 > 0 && 0 > 1 { return 1; } else { return 0; } }", 0),
    ("and_FT", b"fn main() -> i64 { if 0 > 1 && 1 > 0 { return 1; } else { return 0; } }", 0),
    ("or_FT",  b"fn main() -> i64 { if 0 > 1 || 1 > 0 { return 1; } else { return 0; } }", 1),
    ("or_FF",  b"fn main() -> i64 { if 0 > 1 || 0 > 1 { return 1; } else { return 0; } }", 0),
    ("or_TT",  b"fn main() -> i64 { if 1 > 0 || 1 > 0 { return 1; } else { return 0; } }", 1),
    # variable operands via fn params (the bug's original repro shape)
    ("f_or_00", b"fn f(a: i64, b: i64) -> i64 { if a > 0 || b > 0 { return 1; } else { return 0; } } fn main() -> i64 { return f(0, 0); }", 0),
    ("f_or_10", b"fn f(a: i64, b: i64) -> i64 { if a > 0 || b > 0 { return 1; } else { return 0; } } fn main() -> i64 { return f(1, 0); }", 1),
    ("g_and_11", b"fn g(a: i64, b: i64) -> i64 { if a > 0 && b > 0 { return 1; } else { return 0; } } fn main() -> i64 { return g(1, 1); }", 1),
    ("g_and_10", b"fn g(a: i64, b: i64) -> i64 { if a > 0 && b > 0 { return 1; } else { return 0; } } fn main() -> i64 { return g(1, 0); }", 0),
    ("g_and_01", b"fn g(a: i64, b: i64) -> i64 { if a > 0 && b > 0 { return 1; } else { return 0; } } fn main() -> i64 { return g(0, 1); }", 0),
    # 3-way AND chain (left-assoc): (a && b) && c
    ("and3_111", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 && c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(1, 1, 1); }", 1),
    ("and3_110", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 && c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(1, 1, 0); }", 0),
    ("and3_011", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 && c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(0, 1, 1); }", 0),
    # 3-way OR chain
    ("or3_000", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 || b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(0, 0, 0); }", 0),
    ("or3_001", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 || b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(0, 0, 1); }", 1),
    ("or3_100", b"fn h(a: i64, b: i64, c: i64) -> i64 { if a > 0 || b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return h(1, 0, 0); }", 1),
    # mixed precedence: a>0 && b>0 || c>0  ==  (a&&b) || c
    ("mix_000", b"fn m(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return m(0, 0, 0); }", 0),
    ("mix_001", b"fn m(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return m(0, 0, 1); }", 1),
    ("mix_110", b"fn m(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return m(1, 1, 0); }", 1),
    ("mix_100", b"fn m(a: i64, b: i64, c: i64) -> i64 { if a > 0 && b > 0 || c > 0 { return 1; } else { return 0; } } fn main() -> i64 { return m(1, 0, 0); }", 0),
    # value position: let x = if (a>0 && b>0) {5} else {6}
    ("val_and_T", b"fn v(a: i64, b: i64) -> i64 { let x: i64 = if a > 0 && b > 0 { 5 } else { 6 }; return x; } fn main() -> i64 { return v(1, 1); }", 5),
    ("val_and_F", b"fn v(a: i64, b: i64) -> i64 { let x: i64 = if a > 0 && b > 0 { 5 } else { 6 }; return x; } fn main() -> i64 { return v(1, 0); }", 6),
    ("val_or_T", b"fn v(a: i64, b: i64) -> i64 { let x: i64 = if a > 0 || b > 0 { 5 } else { 6 }; return x; } fn main() -> i64 { return v(0, 1); }", 5),
    ("val_or_F", b"fn v(a: i64, b: i64) -> i64 { let x: i64 = if a > 0 || b > 0 { 5 } else { 6 }; return x; } fn main() -> i64 { return v(0, 0); }", 6),
    # SHORT-CIRCUIT observability: B contains 100/a; when a==0, `a>0` is false and
    # `&&` must NOT evaluate B (else SIGFPE division-by-zero). exit 0 proves skip.
    ("sc_and", b"fn sc(a: i64) -> i64 { if a > 0 && (100 / a) > 0 { return 1; } else { return 0; } } fn main() -> i64 { return sc(0); }", 0),
    # dual for ||: when a>0 is true, `||` must NOT evaluate B (100/(a-1) with a=1 -> div0)
    ("sc_or", b"fn sc(a: i64) -> i64 { if a > 0 || (100 / (a - 1)) > 0 { return 1; } else { return 0; } } fn main() -> i64 { return sc(1); }", 1),
]

# byte-identity: `A && B` vs explicit `if A { B } else { 0 }`; `A || B` vs `if A { 1 } else { B }`.
DESUGAR_PAIRS = [
    ("and_vs_ifelse",
     b"fn t(a: i64, b: i64) -> i64 { return a > 0 && b > 0; }",
     b"fn t(a: i64, b: i64) -> i64 { return if a > 0 { b > 0 } else { 0 }; }"),
    ("or_vs_ifelse",
     b"fn t(a: i64, b: i64) -> i64 { return a > 0 || b > 0; }",
     b"fn t(a: i64, b: i64) -> i64 { return if a > 0 { 1 } else { b > 0 }; }"),
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
                print(f"  PASS  {name}: `&&`/`||` ELF byte-identical to explicit if-desugar ({len(ea)} B)")
            else:
                print(f"  FAIL  {name}: logical form ({len(ea)} B) != explicit desugar ({len(eb)} B)")
                rc = 1

    print("ALL PASS  (&&/|| short-circuit: value-correct + byte-identical to if-desugar)" if rc == 0
          else "FAIL  (&&/|| battery had failures)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
