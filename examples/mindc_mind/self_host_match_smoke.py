"""
Self-host NATIVE-ELF smoke for the N-arm `match` desugar.

Pre-existing PARSER miscompile (fixed here): parse_match handled only the minimal
2-arm shape (one int-literal arm + a `_` wildcard) AND only for a bare-expression
arm value with a `,` separator. Consequences:
  * Statement-form `match x { 1 => { return 1; } … _ => { return 9; } }` mis-parsed
    the `{ … }` block arms into garbage and always executed a fixed arm.
  * Expression-form with 3+ arms always yielded the LAST (wildcard) arm.
  * A `;` arm separator was not recognized at all.

The fix desugars an N-arm match to the PROVEN-correct else-if chain
  if x == P0 { V0 } else if x == P1 { V1 } … else { Vwild }
reusing parse_if / parse_if_else_tail's AST shape, so it lowers through the SAME
if-emit path with zero new emit code. Each arm value is a `{ … }` block (statement
form, verbatim) or a bare expression wrapped in a one-statement block (expression
form); separators may be `,`, `;`, or absent.

Two gates:
  (1) EXECUTION-CORRECTNESS — each match runs and exits with the arm the scrutinee
      selects (independent Python reference), 2/3/4-arm, statement AND expression
      form, each arm taken including the wildcard.
  (2) BYTE-IDENTITY — the match and the equivalent explicit `if/else if/else`
      chain compile to a byte-for-byte identical native ELF (the whole point:
      reuse the proven path, add no new emit code).

Execution-correctness is the value gate for these programs (no frozen native
oracle: the Rust src/native backend was deleted, #15). Each fixture is a
self-contained program (no `use std.*`) compiled by the pure-MIND
`selftest_native_elf_h(src, len, hash)` entry; the 32-byte hash is embedded in the
PT_NOTE and does not affect execution, so we pass zeros.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_match_smoke.py
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


def _match_stmt(arms, x):
    body = " ".join(f"{p} => {{ return {v}; }}" for p, v in arms) + " _ => { return 99; }"
    return (
        f"fn f(x: i64) -> i64 {{\n    match x {{ {body} }}\n}}\n"
        f"fn main() -> i64 {{ return f({x}); }}\n"
    )


def _match_expr(arms, sep, x):
    body = (sep + " ").join(f"{p} => {v}" for p, v in arms) + f"{sep} _ => 99"
    return (
        f"fn f(x: i64) -> i64 {{\n    let r: i64 = match x {{ {body} }};\n    return r;\n}}\n"
        f"fn main() -> i64 {{ return f({x}); }}\n"
    )


def _expl_stmt(arms, x):
    head = arms[0]
    s = f"    if x == {head[0]} {{ return {head[1]}; }}"
    for p, v in arms[1:]:
        s += f" else if x == {p} {{ return {v}; }}"
    s += " else { return 99; }"
    return (
        f"fn f(x: i64) -> i64 {{\n{s}\n}}\n"
        f"fn main() -> i64 {{ return f({x}); }}\n"
    )


def _expl_expr(arms, x):
    head = arms[0]
    s = f"if x == {head[0]} {{ {head[1]} }}"
    for p, v in arms[1:]:
        s += f" else if x == {p} {{ {v} }}"
    s += " else { 99 }"
    return (
        f"fn f(x: i64) -> i64 {{\n    let r: i64 = {s};\n    return r;\n}}\n"
        f"fn main() -> i64 {{ return f({x}); }}\n"
    )


def _ref(arms, x):
    for p, v in arms:
        if x == p:
            return v
    return 99


def mind_elf(lib, src: bytes) -> bytes:
    sb = ctypes.create_string_buffer(src, len(src))
    hb = ctypes.create_string_buffer(b"\x00" * 32, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(sb, ctypes.c_void_p).value, len(src),
        ctypes.cast(hb, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    ln = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


# arm sets: 2-arm, 3-arm, 4-arm (pattern -> value); wildcard yields 99.
ARMSETS = [
    [(1, 10)],                        # 2-arm (1 pattern + wildcard)
    [(1, 10), (2, 20)],               # 3-arm
    [(1, 10), (2, 20), (3, 30)],      # 4-arm
]


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

        print("[statement-form match: emit + run (each arm + wildcard taken)]")
        for arms in ARMSETS:
            xs = [p for p, _ in arms] + [50]
            for x in xs:
                elf = mind_elf(lib, _match_stmt(arms, x).encode())
                if len(elf) == 0:
                    print(f"  FAIL  {len(arms)+1}-arm x={x}: empty ELF")
                    fail = True
                    continue
                got = run_elf(elf, tmp)
                want = _ref(arms, x)
                ok = got == want
                print(f"  {'PASS' if ok else 'FAIL'}  {len(arms)+1}-arm stmt x={x} got={got} want={want}")
                fail = fail or not ok

        print("[expression-form match (`;` and `,` separators): emit + run]")
        for sep in (";", ","):
            for arms in ARMSETS:
                xs = [p for p, _ in arms] + [7]
                for x in xs:
                    elf = mind_elf(lib, _match_expr(arms, sep, x).encode())
                    if len(elf) == 0:
                        print(f"  FAIL  {len(arms)+1}-arm '{sep}' x={x}: empty ELF")
                        fail = True
                        continue
                    got = run_elf(elf, tmp)
                    want = _ref(arms, x)
                    ok = got == want
                    print(f"  {'PASS' if ok else 'FAIL'}  {len(arms)+1}-arm expr'{sep}' x={x} got={got} want={want}")
                    fail = fail or not ok

        print("[byte-identity: match vs explicit if/else-if chain]")
        for arms in ARMSETS:
            x = arms[0][0]
            e_m = mind_elf(lib, _match_stmt(arms, x).encode())
            e_e = mind_elf(lib, _expl_stmt(arms, x).encode())
            ok = len(e_m) > 0 and e_m == e_e
            print(f"  {'PASS' if ok else 'FAIL'}  {len(arms)+1}-arm stmt match={len(e_m)}B explicit={len(e_e)}B identical={e_m == e_e}")
            fail = fail or not ok
        for arms in ARMSETS:
            x = arms[0][0]
            e_m = mind_elf(lib, _match_expr(arms, ";", x).encode())
            e_e = mind_elf(lib, _expl_expr(arms, x).encode())
            ok = len(e_m) > 0 and e_m == e_e
            print(f"  {'PASS' if ok else 'FAIL'}  {len(arms)+1}-arm expr match={len(e_m)}B explicit={len(e_e)}B identical={e_m == e_e}")
            fail = fail or not ok

    if fail:
        print("\nFAIL  (N-arm match regression — see failing case above)")
        return 1
    print(
        "\nALL PASS  (N-arm match selects the scrutinee's arm, statement + expression "
        "form, `,`/`;` separators, and emits byte-identical to the explicit if/else-if chain)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
