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

# Narrow unsigned operands are kept zero-extended to their declared width (every
# binding and every narrow arithmetic result is truncated), so the signed idiv
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


# Wrap-then-divide (verifier round 2): a narrow value produced by `+ - * <<`
# that wraps must be divided AFTER wrapping, in every way it can reach `/`/`%`:
# a let, an assignment, a parameter, a return value, a struct field, the bare
# inline expression and a widening cast of it. The expected values come from a
# Python model of wrapping arithmetic, embedded in the program as constants;
# each program returns 100 when every check holds, else the 1-based index of the
# first wrong result. Before the fix the native image computed `(200+100)/3` on
# u8 as 100 (the wrap never applied) where MLIR and the model give 14.
_NARROW_TYPES = [("u8", 8, False), ("u16", 16, False), ("u32", 32, False),
                 ("i8", 8, True), ("i16", 16, True), ("i32", 32, True)]


def _wrapw(v: int, w: int, signed: bool) -> int:
    v &= (1 << w) - 1
    if signed and v >= 1 << (w - 1):
        v -= 1 << w
    return v


def _tdiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _narrow_operands(w: int, signed: bool):
    """(op, a, b) per wrapping operator; every pair overflows width w."""
    if signed:
        hi, lo = (1 << (w - 1)) - 1, -(1 << (w - 1))
        return [("+", hi - 27, 100), ("-", lo + 5, 9), ("*", hi, 3), ("<<", hi - 27, 1)]
    m = (1 << w) - 1
    return [("+", m - 55, 100), ("-", 0, 1), ("*", m, 3), ("<<", m - 55, 1)]


def narrow_wrap_program(ty: str, w: int, signed: bool):
    """One program per width; every check binds its result, then compares it."""
    ops = _narrow_operands(w, signed)
    fns = [f"struct S {{\n    v: {ty},\n}}\n",
           f"fn dv(x: {ty}, y: {ty}) -> {ty} {{\n    return x / y;\n}}\n",
           f"fn md(x: {ty}, y: {ty}) -> {ty} {{\n    return x % y;\n}}\n"]
    for j, (op, _a, _b) in enumerate(ops):
        fns.append(f"fn r{j}(a: {ty}, b: {ty}) -> {ty} {{\n    return a {op} b;\n}}\n")
    checks = []
    calls = []
    for j, (op, a, b) in enumerate(ops):
        full = a << b if op == "<<" else eval(f"{a} {op} {b}")
        r = _wrapw(full, w, signed)
        body = []
        for k in (3, 7):
            q, m = _tdiv(r, k), r - _tdiv(r, k) * k
            x = f"{j}_{k}"
            la = f"0 - {-a}" if a < 0 else str(a)
            body += [f"    let a{x}: {ty} = {la};", f"    let b{x}: {ty} = {b};",
                     f"    let d{x}: {ty} = a{x} {op} b{x};",
                     f"    let mut e{x}: {ty} = 0;", f"    e{x} = a{x} {op} b{x};",
                     f"    let s{x} = S {{ v: a{x} {op} b{x} }};"]
            forms = [
                (ty, f"d{x} / {k}", q), (ty, f"d{x} % {k}", m),
                (ty, f"dv(a{x} {op} b{x}, {k})", q), (ty, f"md(r{j}(a{x}, b{x}), {k})", m),
                (ty, f"e{x} / {k}", q), (ty, f"s{x}.v % {k}", m),
            ]
            # The bare inline forms divide the unbound result. At 8/16 bits it
            # is wrapped first (both backends); an unbound i32/u32 result is
            # refused natively (NARROW32_INLINE_REFUSED below).
            if w < 32:
                forms += [(ty, f"(a{x} {op} b{x}) / {k}", q), (ty, f"(a{x} {op} b{x}) % {k}", m),
                          ("i64", f"((a{x} {op} b{x}) as i64) / {k}", q)]
            for t, expr, want in forms:
                checks.append(len(checks) + 1)
                n = checks[-1]
                lit = f"(0 - {-want})" if want < 0 else str(want)
                cmp = f"x{n}" if t == "i64" else f"(x{n} as i64)"
                body += [f"    let x{n}: {t} = {expr};", f"    if {cmp} != {lit} {{",
                         f"        return {n};", "    }"]
        # One helper per operator keeps every fn small for the native emitter.
        fns.append(f"fn c{j}() -> i64 {{\n" + "\n".join(body) + "\n    return 0;\n}\n")
        calls += [f"    let k{j}: i64 = c{j}();", f"    if k{j} != 0 {{", f"        return k{j};", "    }"]
    prog = "".join(fns) + "fn main() -> i64 {\n" + "\n".join(calls) + "\n    return 100;\n}\n"
    return (f"{ty} wrap-then-divide ({len(checks)} checks: let/param/return/assign/"
            f"field{'/inline/cast' if w < 32 else ''} x + - * <<)", prog, 100)


# A `/` or `%` whose operand is an UNBOUND i32/u32 arithmetic result is refused
# natively: MLIR evaluates that intermediate at 32 bits when its operands are
# parameters (`fn f(a: u32, b: u32) -> u32 { return (a - b) / 2; }` gives
# 2147483647) and at 64 bits when they are locals (the same body over two `let`s
# gives 0), so there is no single reference value. (name, program, MLIR exit or
# None) -- only the native refusal is asserted.
NARROW32_INLINE_REFUSED = [
    ("u32 (a-b)/2 over params", "fn f(a: u32, b: u32) -> u32 {\n    return (a - b) / 2;\n}\n"
     "fn main() -> i64 {\n    return (f(0, 1) as i64) & 255;\n}\n", None),
    ("u32 (a-b)/2 over locals", "fn main() -> i64 {\n    let a: u32 = 0;\n    let b: u32 = 1;\n"
     "    let q: u32 = (a - b) / 2;\n    return (q as i64) & 255;\n}\n", None),
    ("i32 (a+b)%7 over locals", "fn main() -> i64 {\n    let a: i32 = 2147483620;\n    let b: i32 = 100;\n"
     "    let r: i32 = (a + b) % 7;\n    return (r as i64) & 255;\n}\n", None),
    ("u32 9 / (a << b)", "fn main() -> i64 {\n    let a: u32 = 2147483648;\n    let b: u32 = 1;\n"
     "    let q: u32 = 9 / (a << b);\n    return q as i64;\n}\n", None),
]

NARROW_WRAP = [narrow_wrap_program(*t) for t in _NARROW_TYPES] + [
    ("u8 (200+100)/3 via let", "fn main() -> i64 {\n    let a: u8 = 200;\n    let b: u8 = 100;\n"
     "    let d: u8 = a + b;\n    let q: u8 = d / 3;\n    return q as i64;\n}\n", 14),
    ("u32 (0-1)/2 via let", "fn main() -> i64 {\n    let a: u32 = 0;\n    let b: u32 = 1;\n"
     "    let d: u32 = a - b;\n    let q: u32 = d / 2;\n    return (q as i64) & 255;\n}\n", 255),
    ("u32 0-1 compared after wrap", "fn main() -> i64 {\n    let a: u32 = 0;\n    let b: u32 = 1;\n"
     "    let d: u32 = a - b;\n    if d > 5 {\n        return 1;\n    }\n    return 2;\n}\n", 1),
    ("u8 literal 0-1 bound to a let", "fn main() -> i64 {\n    let x: u8 = 0 - 1;\n"
     "    return (x / 2) as i64;\n}\n", 127),
    ("u16 param receives 0-1", "fn dv(a: u16, c: u16) -> i64 {\n    return (a % c) as i64;\n}\n"
     "fn main() -> i64 {\n    return dv(0 - 1, 1000);\n}\n", (65535 % 1000) & 255),
    ("u8 alias let wraps", "type B = u8;\nfn main() -> i64 {\n    let a: B = 200;\n    let b: B = 100;\n"
     "    let d: B = a + b;\n    return (d as i64) / 3;\n}\n", 14),
    ("u8 wrap to a zero divisor traps", "fn main() -> i64 {\n    let a: u8 = 128;\n    let b: u8 = 2;\n"
     "    let z: u8 = a * b;\n    let q: u8 = 7 / z;\n    return (q as i64) + 1;\n}\n", TRAP_EXIT),
    ("u16 inline wrap to a zero divisor traps", "fn main() -> i64 {\n    let a: u16 = 256;\n"
     "    let b: u16 = 256;\n    return (9 % (a * b)) as i64;\n}\n", TRAP_EXIT),
]


def _leaf_call_program(ty: str, w: int, signed: bool):
    """Two calls returning `ty`, summed then divided inline (verifier round 3):
    the sum must be wrapped to `ty` before `/` and `%`, as MLIR re-masks a narrow
    call result. Returns (name, program, expected exit)."""
    a = (1 << (w - 1)) - 5 if signed else (1 << w) - 4
    b = 100
    s = _wrapw(a + b, w, signed)
    q, m = _tdiv(s, 3), s - _tdiv(s, 3) * 3
    want = ((q * 7 + m) & 0xff)
    prog = (f"fn fa() -> {ty} {{\n    return {a};\n}}\n"
            f"fn fb() -> {ty} {{\n    return {b};\n}}\n"
            "fn main() -> i64 {\n"
            "    let q: i64 = ((fa() + fb()) / 3) as i64;\n"
            "    let m: i64 = ((fa() + fb()) % 3) as i64;\n"
            "    return (q * 7 + m) & 255;\n}\n")
    return (f"{ty} call results summed then divided inline", prog, want)


def _leaf_field_program(ty: str, w: int, signed: bool):
    """A struct field of type `ty` stored (in a struct literal) from arithmetic
    over other fields, then divided: the store wraps to the field's width, so the
    division sees what MLIR stored. (A field assignment `q.x = ...` wraps the
    same way, but the seeded native entry refuses field assignment today.)"""
    a = (1 << (w - 1)) - 5 if signed else (1 << w) - 4
    b = 100
    x = _wrapw(a + b, w, signed)
    y = _wrapw(a * b, w, signed)
    q, m = _tdiv(x, 3), y - _tdiv(y, 7) * 7
    want = ((q * 7 + m) & 0xff)
    store = "    let q: P = P { x: p.x + p.y, y: p.x * p.y };\n"
    prog = (f"struct P {{\n    x: {ty},\n    y: {ty},\n}}\n"
            "fn main() -> i64 {\n"
            f"    let p: P = P {{ x: {a}, y: {b} }};\n" + store +
            f"    let d1: {ty} = q.x / 3;\n    let d2: {ty} = q.y % 7;\n"
            "    return ((d1 as i64) * 7 + (d2 as i64)) & 255;\n}\n")
    return (f"{ty} field stored from field arithmetic then divided", prog, want)


# Verifier round 3: narrow values that reach `/` or `%` through a CALL result or
# a STRUCT FIELD. Each program's expected exit comes from the wrapping model
# above; both backends must give it.
NARROW_LEAVES = (
    [_leaf_call_program(*t) for t in _NARROW_TYPES if t[1] < 32]
    + [_leaf_field_program(*t) for t in _NARROW_TYPES]
    + [
        ("i32 MIN / -1 bound to an i32 let over params",
         "fn q(a: i32, b: i32) -> i64 {\n    let x: i32 = a / b;\n    return (x as i64) & 255;\n}\n"
         "fn main() -> i64 {\n    return q(0 - 2147483647 - 1, 0 - 1);\n}\n", 0),
        ("i32 MIN / -1 assigned to an i32 name over locals",
         "fn main() -> i64 {\n    let a: i32 = 0 - 2147483647 - 1;\n    let b: i32 = 0 - 1;\n"
         "    let mut x: i32 = 5;\n    x = a / b;\n    if x == a {\n        return 9;\n    }\n"
         "    return 1;\n}\n", 9),
        ("i32 MIN / -1 stored to an i32 field",
         "struct S {\n    v: i32,\n}\n"
         "fn q(a: i32, b: i32) -> i64 {\n    let s: S = S { v: a / b };\n"
         "    if s.v == a {\n        return 9;\n    }\n    return 1;\n}\n"
         "fn main() -> i64 {\n    return q(0 - 2147483647 - 1, 0 - 1);\n}\n", 9),
        ("u8 field from an out-of-range literal, then divided",
         "struct S {\n    v: u8,\n}\n"
         "fn main() -> i64 {\n    let s: S = S { v: 300 };\n    let d: u8 = s.v / 3;\n    return d as i64;\n}\n",
         _wrapw(300, 8, False) // 3),
        ("i8 field from an out-of-range literal, then divided",
         "struct S {\n    v: i8,\n}\n"
         "fn main() -> i64 {\n    let s: S = S { v: 200 };\n    let d: i8 = s.v / 3;\n    return (d as i64) & 255;\n}\n",
         _tdiv(_wrapw(200, 8, True), 3) & 255),
        ("i32 field divisor, quotient bound to an i32 let",
         "struct S {\n    v: i32,\n}\n"
         "fn main() -> i64 {\n    let s: S = S { v: 0 - 1 };\n    let q: i32 = 9 / s.v;\n"
         "    return (q as i64) & 255;\n}\n", -9 & 255),
        ("i32 call results, MIN / -1 bound to an i32 let",
         "fn fa() -> i32 {\n    return 0 - 2147483647 - 1;\n}\nfn fb() -> i32 {\n    return 0 - 1;\n}\n"
         "fn main() -> i64 {\n    let q: i32 = fa() / fb();\n    if q == fa() {\n        return 9;\n    }\n"
         "    return 1;\n}\n", 9),
        ("small literal / i32 stays native",
         "fn q(a: i32) -> i64 {\n    return (7 / a) as i64;\n}\n"
         "fn main() -> i64 {\n    return q(0 - 1) & 255;\n}\n", -7 & 255),
        ("i32 / small literal stays native",
         "fn q(a: i32) -> i64 {\n    return (a / 7) as i64;\n}\n"
         "fn main() -> i64 {\n    return q(0 - 2147483647 - 1) & 255;\n}\n", _tdiv(-(1 << 31), 7) & 255),
    ]
)

# Shapes where the native i64 slots cannot reproduce MLIR's value: the native
# emitter must REFUSE (empty image). (name, program, MLIR exit or None).
NARROW_LEAVES_REFUSED = [
    ("i32 MIN / -1 compared before it is bound (params)",
     "fn q(a: i32, b: i32) -> i64 {\n    if a / b == a {\n        return 1;\n    }\n    return 0;\n}\n"
     "fn main() -> i64 {\n    return q(0 - 2147483647 - 1, 0 - 1);\n}\n", 1),
    ("i32 MIN / -1 widened by a cast (params)",
     "fn q(a: i32, b: i32) -> i64 {\n    return ((a / b) as i64) & 255;\n}\n"
     "fn main() -> i64 {\n    return q(0 - 2147483647 - 1, 0 - 1);\n}\n", 0),
    ("i32 MIN / -1 widened by a cast (locals)",
     "fn main() -> i64 {\n    let a: i32 = 0 - 2147483647 - 1;\n    let b: i32 = 0 - 1;\n"
     "    return ((a / b) as i64) & 255;\n}\n", None),
    ("i32 MIN / -1 returned from an i64 fn",
     "fn q(a: i32, b: i32) -> i64 {\n    let r: i64 = (a / b) as i64;\n    return r & 255;\n}\n"
     "fn main() -> i64 {\n    return q(0 - 2147483647 - 1, 0 - 1);\n}\n", 0),
    ("i32 call results summed then divided",
     "fn fa() -> i32 {\n    return 2147483647;\n}\nfn fb() -> i32 {\n    return 3;\n}\n"
     "fn main() -> i64 {\n    let r: i64 = ((fa() + fb()) / 3) as i64;\n    return r & 255;\n}\n", None),
    ("u8 fields summed then divided inline",
     "struct P {\n    x: u8,\n    y: u8,\n}\n"
     "fn main() -> i64 {\n    let p: P = P { x: 252, y: 100 };\n"
     "    return ((p.x + p.y) / 3) as i64;\n}\n", 32),
    ("u8 divided by an i64",
     "fn q(a: u8, b: i64) -> i64 {\n    return a / b;\n}\n"
     "fn main() -> i64 {\n    return q(200, 3);\n}\n", None),
    # MLIR converts the literal to the 32-bit operand type: 4294967295 is -1 (so
    # MIN / it is the narrow MIN / -1, giving MIN) and 4294967296 is 0 (a trap).
    ("i32 MIN divided by a literal wider than 32 bits",
     "fn q(a: i32) -> i64 {\n    return ((a / 4294967295) as i64) & 255;\n}\n"
     "fn main() -> i64 {\n    return q(0 - 2147483647 - 1);\n}\n", 0),
    ("u32 remainder by 2^32",
     "fn q(a: u32) -> i64 {\n    let r: u32 = a % 4294967296;\n    return (r as i64) & 255;\n}\n"
     "fn main() -> i64 {\n    return q(7);\n}\n", TRAP_EXIT),
    ("u32 call result divided by 2^32",
     "fn compute() -> u32 {\n    return 4294967297;\n}\n"
     "fn main() -> i64 {\n    return compute() / 4294967296;\n}\n", TRAP_EXIT),
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
        for name, prog, mlir_want in U64_REFUSED + NARROW32_INLINE_REFUSED + NARROW_LEAVES_REFUSED:
            elf = sh.elf(prog)
            mbin, diag = mlir_binary(prog, d, "u64")
            gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
            good = len(elf) == 0 and (mlir_want is None or gm[0] == mlir_want)
            ok = ok and good
            check(good, f"{name}: native image {len(elf)}B (expected 0B refusal), "
                  f"mlir=(exit {gm[0]}) expected mlir=(exit {mlir_want})")
        for name, prog, want_exit in U64_NOT_FENCED + NARROW_UNSIGNED + NARROW_WRAP + NARROW_LEAVES:
            elf = sh.elf(prog)
            mbin, diag = mlir_binary(prog, d, "unsigned")
            gn = run(write_exec(d / "unsigned.native.elf", elf), []) if elf else ("refused", b"")
            gm = run(mbin, []) if mbin is not None else ("build-failed: " + diag[-160:], b"")
            good = gn[0] == want_exit and gm[0] == want_exit
            ok = ok and good
            check(good, f"{name}: native=(exit {gn[0]}) mlir=(exit {gm[0]}) expected=(exit {want_exit})")
    if ok:
        print(f"\nALL PASS  (i64 / % trap with exit {TRAP_EXIT} on both backends; "
              "sign matrix and shift rule agree with the model; narrow wrap-then-divide agrees, "
              "including call-result and struct-field operands; u64/usize, unbound i32/u32 "
              "and mixed-type division refused natively)")
        return 0
    print("\nFAIL  div-trap smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
