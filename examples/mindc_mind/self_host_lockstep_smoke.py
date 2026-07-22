#!/usr/bin/env python3
"""SUB-STEP A lockstep smoke: the loop-carry frame COUNT and the loop-carry EMIT are
now a SINGLE traversal (nb_while_carry), not two hand-mirrored walks.

Before Sub-step A the frame-slot COUNT (nb_count_stmt's while arm -> nb_count_carried*
chain) and the EMIT carry table (nb_emit_while -> nb_while_carry / nb_carry_promote_inner)
were two separate computations that had to agree. Every time they drifted, the frame
count desynced from the emit carry table -> a SILENT miscompile: c27a766 returned x=1
not 3; 0b5f489 returned 2 not 6; the FIX #229 shape hung (the loop-carried var stored
to a slot the cond never re-reads). Sub-step A retires that entire class by CONSTRUCTION:
nb_count_carried now RETURNS the record count of the very same nb_while_carry pre-walk the
emit uses. There is no second computation to drift from.

This is the falsifiable lockstep check for the exact shapes Sub-step A handles: a
top-level `while{ x+=1 }`, a DIRECT `while{ while{} }`, and a triple-nest. A count!=emit
desync manifests as a wrong runtime value, a crash, or a hang (the exact 0b5f489 / FIX
#229 signature). Each fixture is emitted via the SELF-COMPILED native-ELF path
(selftest_native_elf_h) and RUN; the process exit is value-checked against an INDEPENDENT
Python model (no compiler involved). NB: the old Rust `src/native` backend was deleted
(commit 56935ce) and frozen as testdata/native_elf_oracle/ (5 fixed programs, no loop),
so there is no live Rust byte-oracle for arbitrary loop shapes -- the lockstep proof here
is (a) structural (one shared traversal, this file's docstring) and (b) empirical (the
runtime value, which a desync corrupts). Whole-module byte-identity vs the frozen oracle
+ mic3_flip + the RI-E1 self-reproduction loop remain the byte-level gates.

env: MINDC_SO=<self-built .so> MINDC_BIN=./target/release/mindc python3 <this>
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

MAIN_MIND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.mind")


def _carry_ref(iters: int) -> int:
    """`let mut x=0; while c<iters { x=x+1; c=c+1 }` -> x."""
    x = 0
    for _ in range(iters):
        x += 1
    return x


def _nest2_ref(o: int, i: int) -> int:
    x = 0
    for _ in range(o):
        for _ in range(i):
            x += 1
    return x


def _nest3_ref(a: int, b: int, c: int) -> int:
    x = 0
    for _ in range(a):
        for _ in range(b):
            for _ in range(c):
                x += 1
    return x


# (label, expected-low-byte, source). Each fn's `main` runs the loop and returns x.
SHAPES = [
    ("while{ x+=1 } x3 -> 3 (top-level carry)", _carry_ref(3),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x;\n}\n"),
    ("while{ x+=1 } x7 -> 7 (top-level carry, more iters)", _carry_ref(7),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut c: i64 = 0;\n    while c < 7 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x;\n}\n"),
    ("while{ while{ x+=1 } } 2x2 -> 4 (direct nested)", _nest2_ref(2, 2),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 2 {\n        let mut c: i64 = 0;\n        while c < 2 { x = x + 1; c = c + 1; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("while{ while{ x+=1 } } 3x4 -> 12 (direct nested)", _nest2_ref(3, 4),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut c: i64 = 0;\n        while c < 4 { x = x + 1; c = c + 1; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("while{ while{ while{ x+=1 } } } 2x2x2 -> 8 (triple-nest)", _nest3_ref(2, 2, 2),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 2 {\n        let mut b: i64 = 0;\n        while b < 2 {\n            let mut c: i64 = 0;\n            while c < 2 { x = x + 1; c = c + 1; }\n            b = b + 1;\n        }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("while{ while{ while{ x+=1 } } } 2x3x2 -> 12 (triple-nest, asym)", _nest3_ref(2, 3, 2),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 2 {\n        let mut b: i64 = 0;\n        while b < 3 {\n            let mut c: i64 = 0;\n            while c < 2 { x = x + 1; c = c + 1; }\n            b = b + 1;\n        }\n        a = a + 1;\n    }\n    return x;\n}\n"),
]


def build_so() -> str:
    so = os.environ.get("MINDC_SO")
    if so:
        return so
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    r = subprocess.run([mindc, MAIN_MIND, "--emit-shared", out], capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stderr[-3000:])
        sys.exit(1)
    return out


def main() -> int:
    so = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        return 1
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_h"):
        print("FAIL: selftest_native_elf_h absent")
        return 1
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]

    def emit(src: str) -> bytes:
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        h = ctypes.create_string_buffer(b"\x00" * 32, 32)
        es = lib.selftest_native_elf_h(
            ctypes.cast(b, ctypes.c_void_p).value, len(src.encode()),
            ctypes.cast(h, ctypes.c_void_p).value,
        )
        sh = rd(es, 0)
        ln = rd(sh, 8)
        return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""

    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        def run_elf(elf: bytes) -> int:
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            return subprocess.run([str(p)], timeout=10).returncode

        for label, exp, src in SHAPES:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  lockstep shape emitted 0B (count/emit desync refusal?): {label}")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label} -> exit {rc} "
                  f"(python-ref {exp} -> byte {want}, elf {len(elf)}B)")

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
