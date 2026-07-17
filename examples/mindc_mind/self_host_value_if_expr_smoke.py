"""
Self-host NATIVE-ELF smoke for FIX #173 — a value-if in DEEP EXPRESSION position
(a call argument, a binop operand) now lowers natively via nb_expr's ast_if arm
routing to nb_if_value, instead of the old fail-closed const-0 slot.

EXECUTION-CORRECTNESS ONLY (no byte-identity oracle): the Rust `src/native`
backend was deleted (#15, commit 56935ce), so the frozen native-ELF oracle in
testdata/native_elf_oracle/ covers only the 5 pre-capture fixtures. A NEW
construct (value-if in deep-expr position) has no frozen oracle to diff against.
This smoke therefore proves the pure-MIND-emitted ELF EMITS and RUNS with the
correct value — the appropriate gate for a construct with no captured oracle
(same posture as the native-ELF float path). The byte-identity of every already-
RUNS construct is proven separately by self_host_native_elf_smoke.py.

Each fixture is a self-contained program (no `use std.*`) compiled by the pure-
MIND `selftest_native_elf_h(src, len, hash)` entry; the 32-byte hash is embedded
in the PT_NOTE and does not affect execution, so we pass zeros.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_value_if_expr_smoke.py
"""

import ctypes
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# (name, source, expected_exit) — each nests a value-if in DEEP EXPRESSION position.
FIXTURES = [
    # call-ARG position, then-branch: g(if c>0 {a} else {b}) + 1, c>0 -> a=4 ->
    # g(4)=40 -> 41. The `return g(...) + 1` operand is a BINOP (not a value-if),
    # so it goes through nb_expr; the call arg reaches nb_expr's ast_if arm.
    (
        "callarg_then",
        (
            "fn g(v: i64) -> i64 {\n"
            "    return v * 10;\n"
            "}\n"
            "fn f(c: i64, a: i64, b: i64) -> i64 {\n"
            "    return g(if c > 0 { a } else { b }) + 1;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(1, 4, 9);\n"
            "}\n"
        ),
        41,
    ),
    # call-ARG position, else-branch: c<=0 -> b=9 -> g(9)=90 -> 91.
    (
        "callarg_else",
        (
            "fn g(v: i64) -> i64 {\n"
            "    return v * 10;\n"
            "}\n"
            "fn f(c: i64, a: i64, b: i64) -> i64 {\n"
            "    return g(if c > 0 { a } else { b }) + 1;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(0, 4, 9);\n"
            "}\n"
        ),
        91,
    ),
    # binop-OPERAND position (no call): (if c>0 {a} else {b}) * 2 + 3, bound to a
    # let then returned. c>0 -> a=5 -> 5*2+3 = 13.
    (
        "binop_operand",
        (
            "fn f(c: i64, a: i64, b: i64) -> i64 {\n"
            "    let x: i64 = (if c > 0 { a } else { b }) * 2 + 3;\n"
            "    return x;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(1, 5, 8);\n"
            "}\n"
        ),
        13,
    ),
    # binop-OPERAND, else-branch: c<=0 -> b=8 -> 8*2+3 = 19.
    (
        "binop_operand_else",
        (
            "fn f(c: i64, a: i64, b: i64) -> i64 {\n"
            "    let x: i64 = (if c > 0 { a } else { b }) * 2 + 3;\n"
            "    return x;\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(0, 5, 8);\n"
            "}\n"
        ),
        19,
    ),
    # nested deeper: value-if as an arg to g, wrapped in another arithmetic op and
    # a second call. h(g(if c>0 {a} else {b}) + a) with h(v)=v-1.
    # c>0 -> a=6 -> g(6)=60 -> 60+6=66 -> h(66)=65.
    (
        "nested_calls",
        (
            "fn g(v: i64) -> i64 {\n"
            "    return v * 10;\n"
            "}\n"
            "fn h(v: i64) -> i64 {\n"
            "    return v - 1;\n"
            "}\n"
            "fn f(c: i64, a: i64, b: i64) -> i64 {\n"
            "    return h(g(if c > 0 { a } else { b }) + a);\n"
            "}\n"
            "fn main() -> i64 {\n"
            "    return f(1, 6, 9);\n"
            "}\n"
        ),
        65,
    ),
]


def mind_elf(lib, src: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(b"\x00" * 32, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # String handle: addr/len/cap
    ln = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    if not SO.exists():
        import os

        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    print("[value-if in deep-expression position: emit + run (execution-correctness gate)]")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, src, expected in FIXTURES:
            elf = mind_elf(lib, src.encode())
            if len(elf) == 0:
                print(f"  FAIL  {name}: pure-MIND native emit failed closed (empty ELF)")
                return 1
            code = run_elf(elf, tmp)
            ok = code == expected
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name}: pure-MIND ELF ({len(elf)} B) "
                f"runs + exits {code} (expected {expected})"
            )
            if not ok:
                return 1

    print("\nALL PASS  (value-if in deep-expression position emits + runs correct; #173)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
