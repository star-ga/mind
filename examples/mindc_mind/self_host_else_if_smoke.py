"""
Self-host NATIVE-ELF smoke for the `else if` chain parser fix.

Pre-existing PARSER miscompile: `parse_if`'s else branch unconditionally called
`parse_block`, which expects `else { block }` — so `else if COND { … }` (the token
after `else` is `if`, not `{`) was mis-parsed into a malformed AST and the chain
ALWAYS yielded the FINAL else arm regardless of which condition was true. The fix
desugars `else if …` to the working explicit `else { if … }` form (parse the
trailing `if` recursively, wrap the single if-node in a one-statement block), so
BOTH forms produce a byte-identical AST shape and lower through the SAME if-emit
path with zero new emit code.

Two gates here:
  (1) EXECUTION-CORRECTNESS — each `else if` chain runs and exits with the arm
      selected by the TRUE condition (independent Python reference), for a 2-level
      and a 3-level chain, each arm taken.
  (2) BYTE-IDENTITY — the `else if` sugar and the explicit `else { if … }` form
      compile to a byte-for-byte identical native ELF (the whole point of the fix:
      reuse the proven path, add no new emit code).

Execution-correctness is the appropriate value gate for `let mut` + assign + merge
programs with no frozen native oracle (the Rust `src/native` backend was deleted,
#15). Each fixture is a self-contained program (no `use std.*`) compiled by the
pure-MIND `selftest_native_elf_h(src, len, hash)` entry; the 32-byte hash is
embedded in the PT_NOTE and does not affect execution, so we pass zeros.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_else_if_smoke.py
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

SO = resolve_so()


def _f2_elseif(a: int, b: int) -> str:
    return (
        "fn f(a: i64, b: i64) -> i64 {\n"
        "    let mut r: i64 = 0;\n"
        "    if a > 0 {\n"
        "        r = 1;\n"
        "    } else if b > 0 {\n"
        "        r = 2;\n"
        "    } else {\n"
        "        r = 3;\n"
        "    }\n"
        "    return r;\n"
        "}\n"
        f"fn main() -> i64 {{\n    return f({a}, {b});\n}}\n"
    )


def _f2_explicit(a: int, b: int) -> str:
    return (
        "fn f(a: i64, b: i64) -> i64 {\n"
        "    let mut r: i64 = 0;\n"
        "    if a > 0 {\n"
        "        r = 1;\n"
        "    } else {\n"
        "        if b > 0 {\n"
        "            r = 2;\n"
        "        } else {\n"
        "            r = 3;\n"
        "        }\n"
        "    }\n"
        "    return r;\n"
        "}\n"
        f"fn main() -> i64 {{\n    return f({a}, {b});\n}}\n"
    )


def _f3_elseif(a: int, b: int, c: int) -> str:
    return (
        "fn f(a: i64, b: i64, c: i64) -> i64 {\n"
        "    let mut r: i64 = 0;\n"
        "    if a > 0 {\n"
        "        r = 1;\n"
        "    } else if b > 0 {\n"
        "        r = 2;\n"
        "    } else if c > 0 {\n"
        "        r = 3;\n"
        "    } else {\n"
        "        r = 4;\n"
        "    }\n"
        "    return r;\n"
        "}\n"
        f"fn main() -> i64 {{\n    return f({a}, {b}, {c});\n}}\n"
    )


def _ref2(a: int, b: int) -> int:
    if a > 0:
        return 1
    elif b > 0:
        return 2
    else:
        return 3


def _ref3(a: int, b: int, c: int) -> int:
    if a > 0:
        return 1
    elif b > 0:
        return 2
    elif c > 0:
        return 3
    else:
        return 4


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
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    fail = False
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        print("[2-level else-if chain: emit + run (each arm taken)]")
        for a, b in [(1, 1), (0, 1), (0, 0)]:
            elf = mind_elf(lib, _f2_elseif(a, b).encode())
            if len(elf) == 0:
                print(f"  FAIL  f({a},{b}): empty ELF (emit failed closed)")
                fail = True
                continue
            got = run_elf(elf, tmp)
            want = _ref2(a, b)
            ok = got == want
            print(f"  {'PASS' if ok else 'FAIL'}  f({a},{b}) got={got} want={want}")
            fail = fail or not ok

        print("[3-level else-if chain: emit + run (each arm taken)]")
        for a, b, c in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]:
            elf = mind_elf(lib, _f3_elseif(a, b, c).encode())
            if len(elf) == 0:
                print(f"  FAIL  f({a},{b},{c}): empty ELF (emit failed closed)")
                fail = True
                continue
            got = run_elf(elf, tmp)
            want = _ref3(a, b, c)
            ok = got == want
            print(f"  {'PASS' if ok else 'FAIL'}  f({a},{b},{c}) got={got} want={want}")
            fail = fail or not ok

        print("[byte-identity: `else if` sugar vs explicit `else { if … }`]")
        for a, b in [(1, 1), (0, 1), (0, 0)]:
            e_sugar = mind_elf(lib, _f2_elseif(a, b).encode())
            e_explicit = mind_elf(lib, _f2_explicit(a, b).encode())
            ok = len(e_sugar) > 0 and e_sugar == e_explicit
            print(
                f"  {'PASS' if ok else 'FAIL'}  f({a},{b}) "
                f"sugar={len(e_sugar)}B explicit={len(e_explicit)}B identical={e_sugar == e_explicit}"
            )
            fail = fail or not ok

    if fail:
        print("\nFAIL  (else-if chain regression — see failing case above)")
        return 1
    print(
        "\nALL PASS  (else-if chains select the TRUE arm + emit byte-identical to the "
        "explicit else { if … } form)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
