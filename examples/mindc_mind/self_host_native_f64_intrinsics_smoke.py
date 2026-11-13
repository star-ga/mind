#!/usr/bin/env python3
"""W8c smoke: float builtins/intrinsics in the pure-MIND native compiler.

  bits   __mind_bits_to_f64 / __mind_f64_to_bits (same-width reinterpret, bit-exact
         incl. NaN payloads, signalling NaN, -0.0, subnormals)
  sqrt   the bare `sqrt` builtin on an f64 -> inline sqrtsd (IEEE correctly
         rounded); a source-defined `fn sqrt` keeps the ordinary call path
  conv   `as` casts behind __mind_conv_f64 / __mind_conv_i64 / __mind_conv_i32:
         f64 -> i64 saturating (NaN -> 0, >= 2^63 -> INT64_MAX, < -2^63 ->
         INT64_MIN, else truncate), i64 -> f64 round-to-nearest-even,
         i64 -> i32 two's-complement wrap
  detmath exp / log: the std/detmath.mind bodies (no libm) compiled natively

Every program is compiled by a stage1 compiler ELF built from this tree's
main.mind by the frozen pure-MIND seed, run, and its stdout words are compared
with (a) a Python oracle that does not share code with the emitter (struct /
math.sqrt / an IEEE re-execution of the detmath algorithm with coefficients read
from std/detmath.mind) and (b) the same program built with `mindc build` (MLIR).
NaN-producing sqrt cases (negative / NaN input) have no portable bit oracle: they
are checked for NaN-ness in Python and for bit equality against MLIR.

One [PASS]/[FAIL] line per word (scripts/gate_assert.py). Env: MINDC_BIN,
W8_MAIN_MIND (stage1 source), W8_NO_MLIR=1 skips the MLIR legs.
"""
import math
import os
import pathlib
import random
import re
import struct
import sys
import tempfile

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import _w8_native as w  # noqa: E402
from gate_assert import check  # noqa: E402

DETMATH = HERE.parents[1] / "std" / "detmath.mind"
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def f_of(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<q", bits))[0]


def B(x: float) -> int:
    return w.bits_of(x)


def bits_from_u(u: int) -> int:
    return u - (1 << 64) if u >= (1 << 63) else u


# ---- program builder: every case stores one word at p + 8*i --------------------
def program(exprs: list[str], prelude: str = "", per_fn: int = 20) -> str:
    fns, calls = [], []
    for c in range(0, len(exprs), per_fn):
        body = "".join(f"    __mind_store_i64(p + {(c + i) * 8}, {e});\n"
                       for i, e in enumerate(exprs[c:c + per_fn]))
        fns.append(f"fn blk{c // per_fn}(p: i64) -> i64 {{\n{body}    0\n}}\n\n")
        calls.append(f"    let _b{c // per_fn}: i64 = blk{c // per_fn}(p);\n")
    n = len(exprs) * 8
    return (prelude + "".join(fns) + "fn main() -> i64 {\n"
            f"    let p: i64 = __mind_alloc({n});\n" + "".join(calls)
            + f"    let _w: i64 = __mind_write(1, p, {n}, 0 - 1);\n    0\n}}\n")


def fx(bits: int) -> str:
    """An f64 operand built from its exact bit pattern (independent of literals)."""
    return f"__mind_bits_to_f64({w.i64_lit(bits)})"


# ---- case families ---------------------------------------------------------------
SPECIAL_U = [0x7FF8000000000000, 0xFFF8000000000000, 0x7FF0000000000001,
             0x7FF4000000000123, 0x7FF0000000000000, 0xFFF0000000000000,
             0x8000000000000000, 0x0000000000000000, 0x0000000000000001,
             0x000FFFFFFFFFFFFF, 0x0010000000000000, 0x7FEFFFFFFFFFFFFF,
             0x3FF0000000000000, 0xBFF8000000000000]


def bits_cases(rng: random.Random):
    pats = [bits_from_u(u) for u in SPECIAL_U] + [bits_from_u(rng.getrandbits(64)) for _ in range(30)]
    cases = []
    for b in pats:
        cases.append((f"roundtrip {w.u64(b):016x}",
                      f"__mind_f64_to_bits({fx(b)})", b, "exact"))
    for b in pats:
        x = f_of(b)
        if math.isfinite(x) and abs(x) < 1e300:
            cases.append((f"use {w.u64(b):016x} * 2.0 + 0.5",
                          f"__mind_f64_to_bits({fx(b)} * 2.0 + 0.5)", B(x * 2.0 + 0.5), "exact"))
    # int view of a float-typed let and back
    cases.append(("let f64 -> bits", "__mind_f64_to_bits(__mind_bits_to_f64(4614253070214989087))",
                  4614253070214989087, "exact"))
    return cases


def sqrt_cases(rng: random.Random):
    xs = [0.0, -0.0, 1.0, 2.0, 4.0, 0.25, 1e-310, 5e-324, 2.2250738585072014e-308,
          1.7976931348623157e308, math.inf, 3.0, 10.0, 1e-300, 123456789.0]
    xs += [f_of(rng.randrange(0, 0x7FF0000000000000)) for _ in range(30)]
    cases = [(f"sqrt({x!r})", f"__mind_f64_to_bits(sqrt({fx(B(x))}))", B(math.sqrt(x)), "exact")
             for x in xs]
    # dtype propagation: sqrt result inside a float expression
    cases.append(("sqrt(9.0) + 0.5", f"__mind_f64_to_bits(sqrt({fx(B(9.0))}) + 0.5)", B(3.5), "exact"))
    cases.append(("sqrt(sqrt(16.0))", f"__mind_f64_to_bits(sqrt(sqrt({fx(B(16.0))})))", B(2.0), "exact"))
    # NaN results: no portable bit oracle, check NaN-ness here and bits vs MLIR.
    for u in (B(-1.0), B(-math.inf), bits_from_u(0x7FF8000000000000),
              bits_from_u(0xFFF8000000000000), bits_from_u(0x7FF0000000000001), B(-1e-310)):
        cases.append((f"sqrt(bits {w.u64(u):016x}) is NaN",
                      f"__mind_f64_to_bits(sqrt({fx(u)}))", None, "nan"))
    return cases


def sat_f2i(x: float) -> int:
    if math.isnan(x):
        return 0
    if x >= 9223372036854775808.0:
        return I64_MAX
    if x < -9223372036854775808.0:
        return I64_MIN
    return int(x)


def wrap_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def conv_cases(rng: random.Random):
    fs = [math.nan, -math.nan, math.inf, -math.inf, 9223372036854775808.0,
          -9223372036854775808.0, -9223372036854777856.0, 9223372036854774784.0,
          1e30, -1e30, -0.0, 0.0, 2.5, -2.5, 0.9999999999999999, -0.9999999999999999,
          2147483648.0, -2147483649.0, 4503599627370497.5, 5e-324]
    fs += [f_of(rng.randrange(0x3C00000000000000, 0x43F0000000000000)) *
           rng.choice([1, -1]) for _ in range(20)]
    cases = [(f"({x!r}) as i64", f"{fx(B(x))} as i64", sat_f2i(x), "exact") for x in fs]
    ints = [0, 1, -1, (1 << 53) + 1, (1 << 53) + 3, I64_MAX, I64_MIN, (1 << 62) + 1,
            3000000000, -3000000000, 2147483647, -2147483648]
    ints += [rng.randrange(I64_MIN, I64_MAX) for _ in range(20)]
    for v in ints:
        cases.append((f"{v} as f64", f"__mind_f64_to_bits({w.i64_lit(v)} as f64)", B(float(v)), "exact"))
        cases.append((f"{v} as i32", f"({w.i64_lit(v)} as i32) as i64", wrap_i32(v), "exact"))
    return cases


# ---- detmath: extract exp/log (+ deps) from std/detmath.mind, IEEE re-execution ----
def detmath_fns() -> dict[str, str]:
    src = DETMATH.read_text()
    return {m.group(1): m.group(0) for m in
            re.finditer(r"^(?:pub )?fn (\w+)\(.*?^}\n", src, re.S | re.M)}


def det_coeffs() -> list[float]:
    body = detmath_fns()["det_exp"]
    first = float(re.search(r"let mut t: f64 = ([0-9.eE+-]+);", body).group(1))
    rest = [float(c) for c in re.findall(r"t = ([0-9.eE+-]+) \+ r \* t;", body)]
    return [first] + rest


def py_rnd(x: float) -> int:
    return sat_f2i(x + 0.5) if x >= 0.0 else -sat_f2i(0.5 - x)


def py_det_exp(x: float) -> float:
    ln2_hi, ln2_lo, inv_ln2 = 0.6931467056274414, 0.00000047493250390941725, 1.4426950408889634
    k = py_rnd(x * inv_ln2)
    kf = float(k)
    r = (x - kf * ln2_hi) - kf * ln2_lo
    c = det_coeffs()
    t = c[0]
    for ci in c[1:]:
        t = ci + r * t
    y = t * 1.0
    for _ in range(max(k, 0)):
        y = y * 2.0
    for _ in range(max(-k, 0)):
        y = y / 2.0
    return y


def py_exp(x: float) -> float:
    if x != x:
        return x
    if x >= 709.7827128933841:
        return math.inf
    if x <= 0.0 - 745.1332191019412:
        return 0.0
    return py_det_exp(x)


def py_log(x: float) -> float:
    if x != x:
        return x
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return -math.inf
    if x >= math.inf:
        return math.inf
    m, k = x * 1.0, 0
    while m >= 1.4142135623730951:
        m, k = m / 2.0, k + 1
    while m < 0.7071067811865476:
        m, k = m * 2.0, k - 1
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    acc = 0.0
    for n in range(29, 0, -2):
        acc = 1.0 / float(n) + s2 * acc
    return float(k) * 0.6931471805599453 + 2.0 * s * acc


def detmath_prelude() -> str:
    f = detmath_fns()
    return "".join(f[n] + "\n" for n in ("inf", "rnd_to_i64", "det_exp", "exp", "det_log_core", "log"))


def detmath_cases(rng: random.Random):
    ex = [0.0, -0.0, 1.0, -1.0, 0.5, 10.0, -10.0, 700.0, 709.78, 709.7827128933841, -745.2,
          -740.0, -708.5, 1e-10, -1e-10, math.inf, -math.inf]
    ex += [rng.uniform(-700.0, 700.0) for _ in range(25)]
    lg = [1.0, 2.0, 0.5, 1e-300, 5e-324, 1e308, 0.0, -1.0, math.inf, 10.0, 0.7071067811865476,
          1.4142135623730951, 1.4142135623730950, 3.0]
    lg += [f_of(rng.randrange(1, 0x7FF0000000000000)) for _ in range(25)]
    cases = [(f"exp({x!r})", f"__mind_f64_to_bits(exp({fx(B(x))}))", B(py_exp(x)), "exact") for x in ex]
    cases += [(f"log({x!r})", f"__mind_f64_to_bits(log({fx(B(x))}))", B(py_log(x)), "exact") for x in lg]
    nan = bits_from_u(0x7FF8000000000123)
    cases.append(("exp(NaN payload)", f"__mind_f64_to_bits(exp({fx(nan)}))", nan, "exact"))
    cases.append(("log(NaN payload)", f"__mind_f64_to_bits(log({fx(nan)}))", nan, "exact"))
    return cases


SHADOW_PRELUDE = "fn sqrt(x: f64) -> f64 {\n    x + 1.0\n}\n\n"


def shadow_case(stage1, tdp: pathlib.Path) -> int:
    """A source `fn sqrt` must never be silently replaced by the builtin. The Rust
    front end rejects the name outright (reserved builtin), so MLIR has no value
    oracle here: assert that rejection, and that native calls the user body."""
    src = program([f"__mind_f64_to_bits(sqrt({fx(B(2.0))}))"], SHADOW_PRELUDE)
    fails = 0
    rc, elf, err = w.native_compile(stage1, src)
    if elf:
        rrc, out = w.run_elf_bytes(elf, tdp, "shadow.elf")
        ok = w.words(out) == [B(3.0)]
        check(ok, f"sqrt_shadow native calls the user fn (want 3.0) got={out.hex()}")
    else:
        ok = True
        check(ok, f"sqrt_shadow native refused rc={rc} (fail-closed)")
    fails += not ok
    if os.environ.get("W8_NO_MLIR") != "1":
        st, mrc, mout = w.mlir_build_run(src, tdp, "shadow")
        ok = st != "ok" and b"may not be named `sqrt`" in mout
        fails += not ok
        check(ok, f"sqrt_shadow MLIR front end rejects a user `fn sqrt` ({st} rc={mrc})")
    return fails


# ---- refuse: arity / operand-dtype misuse of the bit intrinsics (fail-closed) ----
# __mind_bits_to_f64 takes exactly ONE i64, __mind_f64_to_bits exactly ONE f64. Any
# other arity or operand dtype must refuse the module (0 bytes, non-zero rc), never
# emit an ELF that reads a stale slot or reinterprets the wrong register file.
def one_word(expr: str, pre: str = "") -> str:
    return ("fn main() -> i64 {\n    let p: i64 = __mind_alloc(8);\n" + pre
            + f"    __mind_store_i64(p, {expr});\n"
            "    let _w: i64 = __mind_write(1, p, 8, 0 - 1);\n    0\n}\n")


REFUSE_CASES = [
    ("f2b_noarg", one_word("__mind_f64_to_bits()")),
    ("b2f_noarg", one_word("__mind_f64_to_bits(__mind_bits_to_f64())")),
    ("b2f_2arg", one_word("__mind_f64_to_bits(__mind_bits_to_f64(4607182418800017408, 7))")),
    ("f2b_2arg", one_word("__mind_f64_to_bits(1.5, 2.5)")),
    ("f2b_intarg", one_word("__mind_f64_to_bits(5)")),
    ("b2f_floatarg", one_word("__mind_f64_to_bits(__mind_bits_to_f64(1.5))")),
    ("f2b_intvar", one_word("__mind_f64_to_bits(a)", "    let a: i64 = 5;\n")),
    ("b2f_floatvar", one_word("__mind_f64_to_bits(__mind_bits_to_f64(a))", "    let a: f64 = 1.5;\n")),
    ("b2f_noarg_let", one_word("__mind_f64_to_bits(x)", "    let x: f64 = __mind_bits_to_f64();\n")),
    ("f2b_intcall", "fn k() -> i64 {\n    3\n}\n\n" + one_word("__mind_f64_to_bits(k())")),
    ("b2f_helper_2arg", "fn h(a: i64) -> f64 {\n    __mind_bits_to_f64(a, a)\n}\n\n"
     + one_word("__mind_f64_to_bits(h(4607182418800017408))")),
]
# Positive controls: the same shapes with a well-formed call build and are exact.
ACCEPT_CASES = [
    ("b2f_f2b_ok", one_word("__mind_f64_to_bits(__mind_bits_to_f64(4609434218613702656))"), B(1.5)),
    ("f2b_floatvar_ok", one_word("__mind_f64_to_bits(a)", "    let a: f64 = 1.5;\n"), B(1.5)),
    ("b2f_intvar_ok", one_word("__mind_f64_to_bits(__mind_bits_to_f64(a))",
                               "    let a: i64 = 4609434218613702656;\n"), B(1.5)),
    ("f2b_floatcall_ok", "fn k() -> f64 {\n    2.5\n}\n\n" + one_word("__mind_f64_to_bits(k())"), B(2.5)),
]


def refuse_family(stage1, tdp: pathlib.Path) -> int:
    fails = 0
    for name, src in REFUSE_CASES:
        rc, elf, err = w.native_compile(stage1, src)
        ok = rc != 0 and not elf
        fails += not ok
        check(ok, f"refuse {name}: native rc={rc} {len(elf)}B (want fail-closed, 0B)")
        if os.environ.get("W8_NO_MLIR") != "1":
            st, mrc, mout = w.mlir_build_run(src, tdp, f"ref_{name}")
            print(f"[record] refuse {name}: MLIR {st} rc={mrc}")
    for name, src, want in ACCEPT_CASES:
        rc, elf, err = w.native_compile(stage1, src)
        got = None
        if elf:
            rrc, out = w.run_elf_bytes(elf, tdp, f"{name}.elf")
            got = w.words(out)
        ok = got == [want]
        fails += not ok
        check(ok, f"accept {name}: native rc={rc} got={got} want={[want]}")
    return fails


# ---- detmath routed: BARE exp/log resolved to std/detmath.mind (no libm) -------
# The user program carries NO detmath body. The image composes std/detmath.mind into
# the std blob at its manifest position (the blob a `detmath -> seed` manifest row
# produces at the integration reseed); stage1 resolves the bare call to the detmath
# body, prunes every other detmath fn, and the result is bit-exact vs an independent
# Python re-execution. With today's pinned blob the bare call must refuse.
def routed_blob() -> bytes:
    from _stdlib_manifest import _rows, STD_DIR
    mods = [m for m, seed, _ in _rows() if seed == "seed" or m == "detmath"]
    return b"".join((STD_DIR / f"{m}.mind").read_bytes() + b"\n" for m in mods)


def routed_family(stage1, tdp: pathlib.Path, rng: random.Random) -> int:
    fails = 0
    cases = detmath_cases(rng)
    blob = routed_blob()
    variants = [("bare", ""), ("import", "import std.detmath;\n\n")]
    for vname, head in variants:
        src = head + program([c[1] for c in cases])
        rc, elf, err = w.native_compile(stage1, src, blob)
        if not elf:
            check(False, f"routed[{vname}]: stage1 refused rc={rc} {err.strip()[-160:]}")
            fails += 1
            continue
        if b"libm" in elf or b"GLIBC" in elf:
            check(False, f"routed[{vname}]: ELF references libm/glibc")
            fails += 1
        rrc, out = w.run_elf_bytes(elf, tdp, f"routed_{vname}.elf")
        got = w.words(out)
        ok = rrc == 0 and len(got) == len(cases)
        fails += not ok
        check(ok, f"routed[{vname}]: native exit {rrc}, {len(got)} words")
        for i, (label, _e, want, _m) in enumerate(cases):
            if i < len(got):
                ok = got[i] == want
                fails += not ok
                check(ok, f"routed[{vname}] {label} got={w.u64(got[i]):016x} want={w.u64(want):016x}")
    # a user-defined exp shadows the std body (last definition wins), never the reverse
    shadow = "fn exp(x: f64) -> f64 {\n    x + 1.0\n}\n\n" + program([f"__mind_f64_to_bits(exp({fx(B(2.0))}))"])
    rc, elf, err = w.native_compile(stage1, shadow, blob)
    got = None
    if elf:
        _r, out = w.run_elf_bytes(elf, tdp, "routed_shadow.elf")
        got = w.words(out)
    ok = got == [B(3.0)]
    fails += not ok
    check(ok, f"routed shadow: user fn exp wins over std/detmath got={got}")
    # fail-closed with the CURRENT pinned blob (detmath not seeded yet)
    rc, elf, err = w.native_compile(stage1, program(["__mind_f64_to_bits(exp(1.0))"]))
    ok = rc != 0 and not elf
    fails += not ok
    check(ok, f"routed: pinned blob without detmath refuses bare exp rc={rc} {len(elf)}B")
    return fails


def run_family(name: str, cases, stage1, tdp: pathlib.Path, prelude: str = "") -> int:
    src = program([c[1] for c in cases], prelude)
    (tdp / f"{name}.mind").write_text(src)
    fails = 0
    rc, elf, err = w.native_compile(stage1, src)
    nat = None
    if not elf:
        check(False, f"{name}: stage1 refused rc={rc} {err.strip()[-160:]}")
        fails += 1
    else:
        rrc, out = w.run_elf_bytes(elf, tdp, f"{name}.elf")
        check(rrc == 0 and len(out) == 8 * len(cases), f"{name}: native exit {rrc}, {len(out)}B")
        nat = w.words(out)
    mlir = None
    if os.environ.get("W8_NO_MLIR") != "1":
        st, mrc, mout = w.mlir_build_run(src, tdp, name)
        if st != "ok" or mrc != 0:
            check(False, f"{name}: MLIR oracle {st} rc={mrc} {mout[-200:]!r}")
            fails += 1
        else:
            mlir = w.words(mout)
    for i, (label, _e, want, mode) in enumerate(cases):
        for leg, got in (("native", nat), ("mlir", mlir)):
            if got is None or i >= len(got):
                continue
            g = got[i]
            if mode == "nan":
                ok = math.isnan(f_of(g))
                msg = f"{name} {leg} {label} got={w.u64(g):016x} (NaN)"
            else:
                ok = g == want
                msg = f"{name} {leg} {label} got={w.u64(g):016x} want={w.u64(want):016x}"
            fails += not ok
            check(ok, msg)
        if nat is not None and mlir is not None and i < len(nat) and i < len(mlir):
            ok = nat[i] == mlir[i]
            fails += not ok
            check(ok, f"{name} native==mlir {label} {w.u64(nat[i]):016x}")
    return fails


def libm_disagreement(tdp: pathlib.Path) -> None:
    """Record (not gate) how MLIR's bare `exp`/`log` (libm) compares with detmath."""
    if os.environ.get("W8_NO_MLIR") == "1":
        return
    rng = random.Random(99)
    xs = [rng.uniform(-50.0, 50.0) for _ in range(40)]
    ls = [rng.uniform(1e-3, 1e3) for _ in range(40)]
    exprs = [f"__mind_f64_to_bits(exp({fx(B(x))}))" for x in xs]
    exprs += [f"__mind_f64_to_bits(log({fx(B(x))}))" for x in ls]
    st, mrc, mout = w.mlir_build_run(program(exprs), tdp, "libm")
    if st != "ok":
        print(f"[record] MLIR bare exp/log build: {st} rc={mrc} {mout[-300:]!r}")
        return
    got = w.words(mout)
    want = [B(py_exp(x)) for x in xs] + [B(py_log(x)) for x in ls]
    diff = sum(1 for g, v in zip(got, want) if g != v)
    print(f"[record] MLIR bare exp/log (libm) vs detmath: {diff}/{len(want)} words differ")


def main() -> int:
    rng = random.Random(20261009)
    stage1 = w.build_stage1()
    print(f"[native] stage1 from {w.main_mind_path()} -> {stage1.name}")
    fails = 0
    with tempfile.TemporaryDirectory(prefix="w8intr_") as td:
        tdp = pathlib.Path(td)
        fails += run_family("bits", bits_cases(rng), stage1, tdp)
        fails += run_family("sqrt", sqrt_cases(rng), stage1, tdp)
        fails += shadow_case(stage1, tdp)
        fails += run_family("conv", conv_cases(rng), stage1, tdp)
        fails += run_family("detmath", detmath_cases(rng), stage1, tdp, detmath_prelude())
        fails += refuse_family(stage1, tdp)
        fails += routed_family(stage1, tdp, random.Random(7))
        libm_disagreement(tdp)
    print(f"\n{'ALL PASS' if fails == 0 else f'FAIL ({fails} failures)'}  "
          "(W8c bits/sqrt/conv/detmath exp+log: native vs Python vs MLIR)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
