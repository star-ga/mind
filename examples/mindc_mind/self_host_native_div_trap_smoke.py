"""
Self-host NATIVE integer division / remainder / shift semantics smoke (W2-trap).

Language rule (docs/errors.md, "Integer division trap"):
  * `x / 0`, `x % 0`, `INT_MIN / -1` and `INT_MIN % -1` on i64 are a
    DETERMINISTIC TRAP: the process performs no further action (in particular
    it writes nothing more) and exits with status 79 (MIND_DIV_TRAP_EXIT_CODE).
  * every other i64 `/` truncates toward zero and every other `%` takes the sign
    of the dividend (x86 idiv / AArch64 sdiv+msub / C99 semantics).
  * `<<` and `>>` (arithmetic for i64) use the count modulo 64 (`count & 63`),
    for every count including >= 64 and negative counts.

Both backends must implement exactly that, so this gate runs every program
through the pure-MIND native emitter (seeded `selftest_native_elf_u`, the
production native path) AND through the MLIR build of the same source, and
compares both against a Python model computed independently of either emitter.
Exit code AND stdout are compared.

  * the MATRIX program computes the full sign matrix of `/` and `%` over
    {INT_MIN, INT_MIN+1, -7, -2, -1, 0, 1, 2, 7, INT_MAX} (all non-trapping
    pairs) and the shift table over counts {0,1,7,63,64,65,127,128,-1,-63,-64,
    -65}, writing every result as 8 little-endian bytes to stdout;
  * each TRAP program writes "A", performs one trapping operation, then would
    write "B" and exit 5 -- the expected outcome is stdout "A", exit 79.

Operands reach the operators through helper-function parameters so neither
backend can fold them away; one case also divides two literals directly.

Run:
  MINDC_SO=<self-host .so> MINDC_BIN=<mindc> \
    python3 examples/mindc_mind/self_host_native_div_trap_smoke.py
"""

import pathlib
import struct
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "scripts"))
from gate_assert import check  # noqa: E402  (one [PASS]/[FAIL] line per case)
from _selfhost_so import resolve_so  # noqa: E402
from self_host_native_main_args_smoke import (  # noqa: E402
    SelfHost, mindc_bin, mlir_binary, run, write_exec,
)

TRAP_EXIT = 79
INT_MIN = -(1 << 63)
INT_MAX = (1 << 63) - 1
VALUES = [INT_MIN, INT_MIN + 1, -7, -2, -1, 0, 1, 2, 7, INT_MAX]
SHIFT_VALUES = [0, 1, -1, INT_MIN, INT_MAX, 0x123456789ABCDEF, -0x123456789ABCDEF]
SHIFT_COUNTS = [0, 1, 7, 63, 64, 65, 127, 128, -1, -63, -64, -65]
GROUP = 32  # stores per helper fn (keeps every fn small)


def wrap(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


def lit(v: int) -> str:
    """A MIND expression for v that never needs a negative or out-of-range literal."""
    if v == INT_MIN:
        return "(0 - 9223372036854775807 - 1)"
    if v < 0:
        return f"(0 - {-v})"
    return str(v)


def model(op: str, a: int, b: int):
    """Independent reference: None means "traps"."""
    if op in "/%":
        if b == 0 or (a == INT_MIN and b == -1):
            return None
        q = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            q = -q
        return wrap(q) if op == "/" else wrap(a - b * q)
    c = b & 63
    if op == "<<":
        return wrap(a << c)
    return a >> c  # arithmetic shift of a signed value


def cases():
    out = []
    for op in ("/", "%"):
        for a in VALUES:
            for b in VALUES:
                if model(op, a, b) is not None:
                    out.append((op, a, b))
    for op in ("<<", ">>"):
        for a in SHIFT_VALUES:
            for c in SHIFT_COUNTS:
                out.append((op, a, c))
    return out


_HELPERS = (
    "fn dv(a: i64, b: i64) -> i64 {\n    return a / b;\n}\n"
    "fn md(a: i64, b: i64) -> i64 {\n    return a % b;\n}\n"
    "fn sl(a: i64, b: i64) -> i64 {\n    return a << b;\n}\n"
    "fn sr(a: i64, b: i64) -> i64 {\n    return a >> b;\n}\n"
)
_FN = {"/": "dv", "%": "md", "<<": "sl", ">>": "sr"}


def matrix_program(cs) -> str:
    groups = [cs[i:i + GROUP] for i in range(0, len(cs), GROUP)]
    parts = [_HELPERS]
    for g, members in enumerate(groups):
        parts.append(f"fn grp{g}(buf: i64) -> i64 {{\n")
        for k, (op, a, b) in enumerate(members):
            off = (g * GROUP + k) * 8
            parts.append(f"    __mind_store_i64(buf + {off}, {_FN[op]}({lit(a)}, {lit(b)}));\n")
        parts.append("    return 0;\n}\n")
    n = len(cs) * 8
    parts.append("fn main() -> i64 {\n")
    parts.append(f"    let buf: i64 = __mind_alloc({n});\n")
    for g in range(len(groups)):
        parts.append(f"    let r{g}: i64 = grp{g}(buf);\n")
    parts.append(f"    let w: i64 = __mind_write(1, buf, {n}, 0 - 1);\n")
    parts.append("    return 0;\n}\n")
    return "".join(parts)


def trap_program(op: str, a: int, b: int, direct: bool) -> str:
    expr = f"{lit(a)} {op} {lit(b)}" if direct else f"{_FN[op]}({lit(a)}, {lit(b)})"
    return (
        _HELPERS
        + "fn main() -> i64 {\n"
        "    let buf: i64 = __mind_alloc(8);\n"
        "    __mind_store_i8(buf, 65);\n"
        "    let w1: i64 = __mind_write(1, buf, 1, 0 - 1);\n"
        f"    let q: i64 = {expr};\n"
        "    __mind_store_i8(buf, 66);\n"
        "    let w2: i64 = __mind_write(1, buf, 1, 0 - 1);\n"
        "    return 5 + q - q;\n"
        "}\n"
    )


TRAPS = [
    ("/", 7, 0, False), ("%", 7, 0, False), ("/", 0, 0, False), ("%", -5, 0, False),
    ("/", INT_MIN, -1, False), ("%", INT_MIN, -1, False), ("/", INT_MIN, 0, False),
    ("%", INT_MAX, 0, False), ("/", 7, 0, True), ("%", 9, 0, True),
    ("/", INT_MIN, -1, True), ("%", INT_MIN, -1, True),
]


# i32 operands: MIN/-1 wraps (quotient MIN, remainder 0) on both backends -- the
# result of dividing the sign-extended values in i64 -- and a zero divisor traps.
# Exits 1/2/3 name a wrong value; reaching the trap is the expected outcome.
NARROW_PROGRAM = (
    "fn dv(a: i32, b: i32) -> i32 {\n    return a / b;\n}\n"
    "fn md(a: i32, b: i32) -> i32 {\n    return a % b;\n}\n"
    "fn main() -> i64 {\n"
    "    let q: i32 = dv(0 - 2147483647 - 1, 0 - 1);\n"
    "    let r: i32 = md(0 - 2147483647 - 1, 0 - 1);\n"
    "    let x: i32 = dv(0 - 7, 2);\n"
    "    if q != 0 - 2147483647 - 1 {\n        return 1;\n    }\n"
    "    if r != 0 {\n        return 2;\n    }\n"
    "    if x != 0 - 3 {\n        return 3;\n    }\n"
    "    let z: i32 = dv(5, 0);\n"
    "    return 4 + z - z;\n"
    "}\n"
)


# A program that declares `_exit` itself (as MIND-Lab's campaign does) must still
# build on MLIR and trap the same way: the trap trampoline reuses the program's
# declaration (its i32 parameter is widened to i64 there) instead of adding a
# second, conflicting one. MLIR-only: the native emitter refuses extern calls.
USER_EXIT_PROGRAM = (
    "extern \"C\" {\n    safe fn _exit(code: i32);\n}\n"
    "fn dv(a: i64, b: i64) -> i64 {\n    return a / b;\n}\n"
    "fn main() -> i64 {\n"
    "    let q: i64 = dv(7, 0);\n"
    "    if q == 12345 {\n        _exit(3);\n    }\n"
    "    return 5;\n"
    "}\n"
)


# 64-bit unsigned division (W2-trap unsigned fence). The native emitter's only
# division is the SIGNED trapping idiv, and its width lattice cannot tell u64 from
# i64, so a `/` or `%` in a fn that can hold a u64 value -- the fn spells `u64` /
# `usize`, or names a u64-returning fn, a struct whose declaration spells the
# type (or one of its fields), an alias or a const of it -- must REFUSE natively
# (empty image) instead of computing a signed result. Each program's MLIR build
# is the value oracle: (name, program, MLIR exit).
U64_REFUSED = [
    ("u64 2^63 / (2^64-1)",
     "fn dv(a: u64, b: u64) -> u64 {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    let a: u64 = 9223372036854775808;\n"
     "    let b: u64 = 18446744073709551615;\n    let q: u64 = dv(a, b);\n"
     "    return (q as i64) + 3;\n}\n", 3),
    ("u64 2^63 % (2^64-1)",
     "fn md(a: u64, b: u64) -> u64 {\n    return a % b;\n}\n"
     "fn main() -> i64 {\n    let a: u64 = 9223372036854775808;\n"
     "    let b: u64 = 18446744073709551615;\n    let r: u64 = md(a, b);\n"
     "    if r == a {\n        return 5;\n    }\n    return 6;\n}\n", 5),
    ("u64 MAX / 2 / 2^62",
     "fn main() -> i64 {\n    let m: u64 = 18446744073709551615;\n"
     "    let h: u64 = m / 2;\n    let q: u64 = h / 4611686018427387904;\n"
     "    return q as i64;\n}\n", 1),
    ("usize 10 / 3",
     "fn main() -> i64 {\n    let a: usize = 10;\n    let b: usize = 3;\n"
     "    return (a / b) as i64;\n}\n", 3),
    ("i64 division in a module that also spells u64",
     "fn main() -> i64 {\n    let a: i64 = 10;\n    let b: i64 = 3;\n"
     "    let c: u64 = 5 as u64;\n    return a / b + (c as i64) - 5;\n}\n", 3),
    ("u64 from a call, divided in a fn that never spells u64",
     "fn g() -> u64 {\n    return 18446744073709551615;\n}\n"
     "fn main() -> i64 {\n    let q = g() / 2;\n"
     "    if (q as i64) == 9223372036854775807 {\n        return 4;\n    }\n    return 6;\n}\n", 4),
    ("u64 struct field, divided in a fn that never spells u64",
     "struct S {\n    v: u64,\n}\n"
     "fn mk() -> S {\n    return S { v: 18446744073709551615 };\n}\n"
     "fn main() -> i64 {\n    let s = mk();\n    let q = s.v / 2;\n"
     "    if (q as i64) == 9223372036854775807 {\n        return 4;\n    }\n    return 6;\n}\n",
     # MLIR is not a value oracle here: it divides the u64 field as SIGNED and
     # exits 6 (a separate MLIR typing bug, reported). Only the native refusal
     # is asserted for this shape.
     None),
    ("u64 alias",
     "type W = u64;\n"
     "fn dv(a: W, b: W) -> W {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    return dv(18446744073709551615, 3) as i64;\n}\n", 0x5555555555555555 & 255),
]

# Controls for the fence: `u64` only inside a comment or a string literal is not
# a type, so the division still builds natively and divides.
U64_NOT_FENCED = [
    ("u64 only in a comment",
     "// a u64 or usize mention in a comment is not a type\n"
     "fn main() -> i64 {\n    let a: i64 = 17;\n    let b: i64 = 5;\n"
     "    return a / b + a % b;\n}\n", 5),
    ("u64 confined to another fn returning i64",
     "fn w() -> i64 {\n    let z: u64 = 3;\n    return z as i64;\n}\n"
     "fn main() -> i64 {\n    let a: i64 = 17;\n    let b: i64 = 5;\n"
     "    return a / b + w() - 3;\n}\n", 3),
]

# Narrow unsigned operands are held zero-extended in 64 bits, so the signed idiv
# equals the unsigned division and only a zero divisor traps. (name, program,
# expected exit) — the expected value is the unsigned result, checked on both.
NARROW_UNSIGNED = [
    ("u32 MAX / 2", "fn dv(a: u32, b: u32) -> u32 {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    let q: u32 = dv(4294967295, 2);\n"
     "    if q == 2147483647 {\n        return 7;\n    }\n    return 1;\n}\n", 7),
    ("u32 MAX % 7", "fn md(a: u32, b: u32) -> u32 {\n    return a % b;\n}\n"
     "fn main() -> i64 {\n    let r: u32 = md(4294967295, 7);\n"
     "    return r as i64;\n}\n", 4294967295 % 7),
    ("u16 65535 / 255", "fn dv(a: u16, b: u16) -> u16 {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    let q: u16 = dv(65535, 255);\n"
     "    if q == 257 {\n        return 7;\n    }\n    return 1;\n}\n", 7),
    ("u8 255 % 16", "fn md(a: u8, b: u8) -> u8 {\n    return a % b;\n}\n"
     "fn main() -> i64 {\n    let r: u8 = md(255, 16);\n"
     "    return r as i64;\n}\n", 255 % 16),
    ("u32 x / 0 traps", "fn dv(a: u32, b: u32) -> u32 {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    let q: u32 = dv(4294967295, 0);\n"
     "    return (q as i64) + 1;\n}\n", TRAP_EXIT),
]


def main() -> int:
    so = resolve_so()
    sh = SelfHost(so)
    print(f"[div-trap smoke] so={so} mindc={mindc_bin()}")
    ok = True
    with tempfile.TemporaryDirectory(prefix="w2trap-") as td:
        d = pathlib.Path(td)

        cs = cases()
        prog = matrix_program(cs)
        want = b"".join(struct.pack("<q", model(op, a, b)) for op, a, b in cs)
        elf = sh.elf(prog)
        mbin, diag = mlir_binary(prog, d, "matrix")
        if not elf:
            ok = False
            print("  FAIL  matrix: self-host refused (empty image)")
        if mbin is None:
            ok = False
            print(f"  FAIL  matrix: MLIR oracle build failed: {diag}")
        if elf and mbin is not None:
            gn = run(write_exec(d / "matrix.native.elf", elf), [])
            gm = run(mbin, [])
            for label, got in (("native", gn), ("mlir", gm)):
                bad = [(cs[i], struct.unpack("<q", got[1][i * 8:i * 8 + 8])[0],
                        model(*cs[i]))
                       for i in range(len(cs))
                       if got[1][i * 8:i * 8 + 8] != want[i * 8:i * 8 + 8]]
                good = got[0] == 0 and got[1] == want
                ok = ok and good
                check(good, f"matrix[{label}]: {len(cs)} results "
                      f"(div/mod sign matrix + shifts), exit {got[0]}, "
                      f"{len(got[1])}B stdout, {len(bad)} wrong")
                for (op, a, b), g, w in bad[:12]:
                    print(f"        {a} {op} {b}: got {g} expected {w}")

        for op, a, b, direct in TRAPS:
            name = f"trap {a} {op} {b}{' (literal operands)' if direct else ''}"
            prog = trap_program(op, a, b, direct)
            elf = sh.elf(prog)
            mbin, diag = mlir_binary(prog, d, "trap")
            want = (TRAP_EXIT, b"A")
            gn = run(write_exec(d / "trap.native.elf", elf), []) if elf else ("refused", b"")
            gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
            good = gn == want and gm == want
            ok = ok and good
            check(good, f"{name}: native=(exit {gn[0]}, stdout {gn[1]!r}) "
                  f"mlir=(exit {gm[0]}, stdout {gm[1]!r}) expected=(exit {want[0]}, stdout {want[1]!r})")
        elf = sh.elf(NARROW_PROGRAM)
        mbin, diag = mlir_binary(NARROW_PROGRAM, d, "narrow")
        want = (TRAP_EXIT, b"")
        gn = run(write_exec(d / "narrow.native.elf", elf), []) if elf else ("refused", b"")
        gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
        good = gn == want and gm == want
        ok = ok and good
        check(good, f"i32 MIN/-1 wraps, i32 x/0 traps: native=(exit {gn[0]}) "
              f"mlir=(exit {gm[0]}) expected=(exit {want[0]})")
        mbin, diag = mlir_binary(USER_EXIT_PROGRAM, d, "user_exit")
        gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-200:], b"")
        good = gm == (TRAP_EXIT, b"")
        ok = ok and good
        check(good, f"program declaring extern _exit, 7 / 0: "
              f"mlir=(exit {gm[0]}) expected=(exit {TRAP_EXIT}) [native: extern calls refused]")
        for name, prog, mlir_want in U64_REFUSED:
            elf = sh.elf(prog)
            mbin, diag = mlir_binary(prog, d, "u64")
            gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
            good = len(elf) == 0 and (mlir_want is None or gm[0] == mlir_want)
            ok = ok and good
            check(good, f"{name}: native image {len(elf)}B (expected 0B refusal), "
                  f"mlir=(exit {gm[0]}) expected mlir=(exit {mlir_want})")
        for name, prog, want_exit in U64_NOT_FENCED + NARROW_UNSIGNED:
            elf = sh.elf(prog)
            mbin, diag = mlir_binary(prog, d, "unsigned")
            gn = run(write_exec(d / "unsigned.native.elf", elf), []) if elf else ("refused", b"")
            gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
            good = gn[0] == want_exit and gm[0] == want_exit
            ok = ok and good
            check(good, f"{name}: native=(exit {gn[0]}) mlir=(exit {gm[0]}) expected=(exit {want_exit})")
    if ok:
        print(f"\nALL PASS  (i64 / % trap with exit {TRAP_EXIT} on both backends; "
              "sign matrix and shift rule agree with the model; u64/usize division refused natively)")
        return 0
    print("\nFAIL  div-trap smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
