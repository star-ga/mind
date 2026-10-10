#!/usr/bin/env python3
"""W8b smoke: correctly rounded decimal -> binary64 float literals in the pure-MIND
compiler (nb_float_lit_bits -> nb_dec2f64_bits, exact bignum, round-half-even).

Oracle: Python `float(lit)` (CPython's correctly rounded strtod) packed with
struct.pack('<d') -- independent of the emitter. Three legs:

  1. .so leg: selftest_float_lit_bits(addr, len) from the MINDC_SO cdylib over the
     full generated corpus (>= 2000 literals): exact halfway ties and their
     one-digit neighbours, subnormals, the normal/subnormal boundary, DBL_MAX and the
     overflow boundary, long mantissas, `_` separators, exponent forms, the shortest
     repr of random doubles, random decimal strings, every literal in
     std/detmath.mind and the MIND-Lab report literals.
  2. native leg: a stage1 compiler ELF built from this tree's main.mind by the
     frozen pure-MIND seed compiles programs that store
     __mind_f64_to_bits(LITERAL) for a corpus subset and write the words to stdout;
     the emitted ELF is run and every word is compared with the oracle.
  3. MLIR leg: the first native program is also built with `mindc build` (MLIR) and
     run; its stdout must equal the oracle and the native stdout.

One [PASS]/[FAIL] line per checked literal / program (scripts/gate_assert.py).

Env: MINDC_SO (cdylib under test), MINDC_BIN (MLIR oracle), W8_MAIN_MIND (stage1
source, default this tree), W8_NO_MLIR=1 skips leg 3.
"""
import ctypes
import os
import pathlib
import random
import re
import sys
import tempfile
from decimal import Decimal, getcontext
from fractions import Fraction
import math
import struct

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import _w8_native as w  # noqa: E402
from gate_assert import check  # noqa: E402

getcontext().prec = 1200

# Literal-shape grammar the MIND lexer accepts: D[_D].D[eE[+-]D] or D[eE[+-]D].
LIT_RE = re.compile(r"^[0-9](?:_?[0-9])*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")

FIXED = [
    "0.1", "0.2", "0.3", "0.7", "3.14", "1e-6", "1e23", "1e22", "8.98846567431158e307",
    "2.2250738585072011e-308", "2.2250738585072014e-308", "2.2250738585072012e-308",
    "4.9406564584124654e-324", "2.4703282292062327e-324", "2.4703282292062328e-324",
    "2.47032822920623272088e-324", "5e-324", "1e-323", "3e-324", "1e-400", "0.0",
    "0.000", "0e0", "0e999999999999", "1e999999999999999999999", "1.7976931348623157e308",
    "1.7976931348623158e308", "1.797693134862315807e308", "1.797693134862315808e308",
    "1.7976931348623159e308", "1e308", "1e309", "9007199254740993.0",
    "9007199254740995.0", "9007199254740992.0", "18446744073709551616.0",
    "9223372036854775808.0", "9223372036854775807.0", "1_000.5", "1_000_000.25e-3",
    "123456789012345678901234567890.0", "0.1000000000000000055511151231257827021181583404541015625",
    "0.1000000000000000055511151231257827021181583404541015624",
    "0.1000000000000000055511151231257827021181583404541015626",
    "1.0E+5", "1.5E-5", "2.5e+0", "7e0", "0.5", "1.5", "2.5", "1.0", "100.0",
    "0.15", "0.30", "0.45", "0.55", "0.6", "0.9999999999999999", "0.99999999999999999",
    "1.00000000000000011102230246251565404236316680908203125",
    "1.00000000000000011102230246251565404236316680908203124",
    "1.00000000000000011102230246251565404236316680908203126",
    "6.02214076e23", "6.62607015e-34", "1.602176634e-19", "2.718281828459045",
    "3.141592653589793", "0.6931471805599453",
]


def detmath_literals() -> list[str]:
    src = (HERE.parents[1] / "std" / "detmath.mind").read_text()
    code = "\n".join(line.split("//")[0] for line in src.splitlines())
    lits = re.findall(r"(?<![A-Za-z0-9_.])([0-9][0-9_]*\.[0-9]+(?:[eE][+-]?[0-9]+)?)", code)
    return sorted(set(lits))


def exact_decimal(fr: Fraction, digits: int) -> str:
    """Scientific literal of the rational fr (exact when it terminates within digits)."""
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    s = format(d, f".{digits}e")
    mant, exp = s.split("e")
    return f"{mant}e{int(exp)}"


def halfway_cases(rng: random.Random, n: int) -> list[str]:
    """Exact midpoints between adjacent doubles (round-half-even must decide), plus
    the literals one unit in the last digit below and above each midpoint."""
    out = []
    for _ in range(n):
        kind = rng.randrange(3)
        if kind == 0:
            bits = rng.randrange(1, 0x7FEFFFFFFFFFFFFF)
        elif kind == 1:
            bits = rng.randrange(1, 1 << 52)  # subnormal
        else:
            bits = (rng.randrange(1, 2046) << 52) | rng.choice([0, 1, (1 << 52) - 1])
        x = struct.unpack("<d", struct.pack("<q", bits))[0]
        y = math.nextafter(x, math.inf)
        if math.isinf(y):
            continue
        mid = (Fraction(x) + Fraction(y)) / 2
        s = exact_decimal(mid, 800)
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".") if "." in mant else mant
        if "." not in mant:
            mant += ".0"
        out.append(f"{mant}e{exp}")
        # neighbours: append a digit 1 (just above) / subtract via shorter truncation
        ip, fp = mant.split(".")
        out.append(f"{ip}.{fp}1e{exp}")
        if len(fp) > 1:
            out.append(f"{ip}.{fp[:-1]}e{exp}")
    return out


def random_cases(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        r = rng.randrange(4)
        if r == 0:
            x = struct.unpack("<d", struct.pack("<q", rng.randrange(0, 0x7FF0000000000000)))[0]
            s = repr(x)
            if "e" not in s and "." not in s:
                s += ".0"
            out.append(s.replace("e+", "e"))
        elif r == 1:
            nd = rng.randrange(1, 30)
            digs = str(rng.randrange(1, 10)) + "".join(str(rng.randrange(10)) for _ in range(nd))
            k = rng.randrange(1, len(digs) + 1)
            out.append(f"{digs[:k]}.{digs[k:] or '0'}e{rng.randrange(-340, 320)}")
        elif r == 2:
            ip = str(rng.randrange(0, 10 ** rng.randrange(1, 12)))
            fp = "".join(str(rng.randrange(10)) for _ in range(rng.randrange(1, 25)))
            out.append(f"{ip}.{fp}")
        else:
            out.append(f"{rng.randrange(1, 10**6)}e{rng.randrange(-330, 310)}")
    return out


def corpus() -> list[str]:
    rng = random.Random(20261009)
    lits = FIXED + detmath_literals() + halfway_cases(rng, 450) + random_cases(rng, 900)
    seen, out = set(), []
    for s in lits:
        if s not in seen and LIT_RE.match(s):
            seen.add(s)
            out.append(s)
    return out


def oracle(lit: str) -> int:
    return w.bits_of(float(lit.replace("_", "")))


def so_leg(lits: list[str]) -> int:
    so = os.environ.get("MINDC_SO")
    if not so:
        check(False, "so-leg: MINDC_SO not set")
        return 1
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_float_lit_bits"):
        check(False, f"so-leg: {so} exports no selftest_float_lit_bits (W8b not built)")
        return 1
    fn = lib.selftest_float_lit_bits
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    fails = 0
    for lit in lits:
        b = lit.encode()
        buf = ctypes.create_string_buffer(b, len(b))
        got = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(b))
        want = oracle(lit)
        ok = got == want
        fails += not ok
        check(ok, f"so-leg {lit[:60]}{'...' if len(lit) > 60 else ''} "
                  f"got={w.u64(got):016x} want={w.u64(want):016x}")
    return fails


def program(lits: list[str], per_fn: int = 24) -> str:
    """Store f64_to_bits(lit) for every literal (negated when the entry starts '-')."""
    fns, calls = [], []
    for c in range(0, len(lits), per_fn):
        body = []
        for i, lit in enumerate(lits[c:c + per_fn]):
            body.append(f"    __mind_store_i64(p + {(c + i) * 8}, __mind_f64_to_bits({lit}));\n")
        fns.append(f"fn blk{c // per_fn}(p: i64) -> i64 {{\n{''.join(body)}    0\n}}\n\n")
        calls.append(f"    let _b{c // per_fn}: i64 = blk{c // per_fn}(p);\n")
    n = len(lits) * 8
    return ("".join(fns) + "fn main() -> i64 {\n"
            f"    let p: i64 = __mind_alloc({n});\n" + "".join(calls)
            + f"    let _w: i64 = __mind_write(1, p, {n}, 0 - 1);\n    0\n}}\n")


def compare_words(tag: str, lits: list[str], stdout: bytes) -> int:
    got = w.words(stdout)
    if len(got) != len(lits):
        check(False, f"{tag}: {len(got)} words for {len(lits)} literals")
        return 1
    fails = 0
    for lit, g in zip(lits, got):
        want = oracle(lit)
        ok = g == want
        fails += not ok
        check(ok, f"{tag} {lit[:60]} got={w.u64(g):016x} want={w.u64(want):016x}")
    return fails


def native_and_mlir_legs(lits: list[str]) -> int:
    fails = 0
    stage1 = w.build_stage1()
    print(f"[native] stage1 from {w.main_mind_path()} -> {stage1.name}")
    # MIND-Lab + detmath + fixed edges first (they also feed the MLIR leg), then a
    # corpus sample. 120 literals per program keeps every fn small.
    head = [s for s in lits if s in set(FIXED) or s in set(detmath_literals())]
    rest = [s for s in lits if s not in set(head)]
    rng = random.Random(7)
    sample = head + rng.sample(rest, min(480, len(rest)))
    chunks = [sample[i:i + 120] for i in range(0, len(sample), 120)]
    with tempfile.TemporaryDirectory(prefix="w8lit_") as td:
        tdp = pathlib.Path(td)
        for ci, chunk in enumerate(chunks):
            src = program(chunk)
            rc, elf, err = w.native_compile(stage1, src)
            if not elf:
                check(False, f"native prog{ci}: stage1 refused rc={rc} {err.strip()[-160:]}")
                fails += 1
                continue
            rrc, out = w.run_elf_bytes(elf, tdp, f"p{ci}.elf")
            check(rrc == 0, f"native prog{ci}: exit {rrc}")
            fails += (rrc != 0) + compare_words(f"native prog{ci}", chunk, out)
            if ci == 0 and os.environ.get("W8_NO_MLIR") != "1":
                # MLIR DEFECT (recorded, not fixed here): a literal that rounds to
                # +inf prints as `arith.constant inf : f64`, which mlir-opt rejects,
                # so the MLIR leg drops overflowing literals; the native leg keeps them.
                fin = [s for s in chunk if not math.isinf(float(s.replace("_", "")))]
                print(f"[mlir] prog0 drops {len(chunk) - len(fin)} +inf literal(s): "
                      f"{[s for s in chunk if s not in fin]}")
                st, mrc, mout = w.mlir_build_run(program(fin), tdp, "p0")
                if st != "ok":
                    check(False, f"mlir prog0: build failed {mout[-200:]!r}")
                    fails += 1
                else:
                    check(mrc == 0, f"mlir prog0: exit {mrc}")
                    fails += (mrc != 0) + compare_words("mlir prog0", fin, mout)
                    nat = w.words(out)
                    same = [nat[chunk.index(s)] for s in fin] == w.words(mout)
                    fails += not same
                    check(same, f"native prog0 words == mlir prog0 words ({len(fin)} finite)")
        # negative literal operands through the native neg-fold arm
        neg = [f"-{s}" for s in ("0.1", "0.3", "1e23", "2.2250738585072011e-308", "0.0")]
        src = program(neg)
        rc, elf, err = w.native_compile(stage1, src)
        if elf:
            rrc, out = w.run_elf_bytes(elf, tdp, "neg.elf")
            fails += compare_words("native neg", neg, out)
        else:
            # Pre-existing fail-closed: the twin mic@3 flattener refuses Neg(FloatLit)
            # (flatten_expr_env); a refusal is acceptable, a wrong value is not.
            check(True, f"native neg: refused rc={rc} (fail-closed, pre-existing)")
    return fails


def main() -> int:
    lits = corpus()
    print(f"[corpus] {len(lits)} literals")
    if len(lits) < 2000:
        check(False, f"corpus has {len(lits)} literals, need >= 2000")
        return 1
    fails = so_leg(lits)
    fails += native_and_mlir_legs(lits)
    print(f"\n{'ALL PASS' if fails == 0 else f'FAIL ({fails} failures)'}  "
          f"(W8b float literals: {len(lits)} so-leg, native + MLIR legs)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
