#!/usr/bin/env python3
"""RI-D / #10 native-float SLICE 1a — native-ELF SATURATING f64 -> signed NARROW
(i8/i16/i32) cast, byte-VALUE-identical to the MLIR backend.

`selftest_native_elf_scalar_narrow(nmin, nmax, inr, povf, novf, nan)` emits a
runnable x86-64 ET_EXEC that sat+clamps four emit-time f64 edge bit patterns to
the signed narrow range [nmin,nmax] and folds them (acc = acc*1000003 + term,
wrapping i64), then writes the folded i64 as 8 LE bytes. The native primitive is
the proven branchless SSE2 saturating f64->i64 base (nb_sat_cast_i64) THEN an
integer maxsi/minsi clamp (nb_sat_narrow_clamp) — exactly the composition the
MLIR oracle emit_saturating_fp_to_narrow applies, so `300.0 as i8` = 127 (clamp),
NOT 44 (wrap). Zero MLIR/LLVM in the emit chain.

Two independent oracles, no self-referential canary:
  1. a numpy/Python reference (sat-toward-zero, +ovf->i64max, -ovf->i64min,
     NaN->0, then clamp to [nmin,nmax]) folded the same way; and
  2. the LIVE `mindc build` cpu (MLIR) backend on `f as iW` for the same values,
     which grounds oracle #1 against the real MLIR narrow cast — so native ==
     oracle == MLIR transitively.

Usage:
  MINDC_SO=/path/to.so [MINDC_BIN=./target/release/mindc] \
      python3 self_host_native_scalar_narrow_smoke.py
"""
import ctypes
import math
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parents[1]
_DEFAULT_SO = _HERE / "libmindc_mind.so"
MINDC = os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc"))

_MASK64 = (1 << 64) - 1

# (name, width, nmin, nmax).
WIDTHS = [
    ("i8", 8, -128, 127),
    ("i16", 16, -32768, 32767),
    ("i32", 32, -2147483648, 2147483647),
]


def _f64_bits(x: float) -> int:
    u = int.from_bytes(struct.pack("<d", x), "little")
    return u - (1 << 64) if u >= (1 << 63) else u


def _sat_f2i64(x: float) -> int:
    if math.isnan(x):
        return 0
    if x >= 9223372036854775808.0:
        return 9223372036854775807
    if x < -9223372036854775808.0:
        return -9223372036854775808
    return int(x)  # trunc toward zero


def _sat_narrow(x: float, nmin: int, nmax: int) -> int:
    v = _sat_f2i64(x)
    if v < nmin:
        return nmin
    if v > nmax:
        return nmax
    return v


def _fold(vals) -> int:
    k = 1000003
    acc = vals[0]
    for t in vals[1:]:
        acc = (acc * k + t) & _MASK64
        if acc >= (1 << 63):
            acc -= 1 << 64
    return acc


def _emit_narrow(lib, nmin, nmax, edges) -> bytes:
    fn = lib.selftest_native_elf_scalar_narrow
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 6
    es = fn(nmin, nmax, *[_f64_bits(e) for e in edges])
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def _run_elf(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "narrow.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], capture_output=True).stdout


def _live_exit(value_src: str, ity: str) -> int:
    """LIVE Rust mindc cpu(MLIR) backend exit for `value_src as ity` (& 0xFF)."""
    src = (f"fn main() -> i64 {{\n    let f: f64 = {value_src};\n"
           f"    return f as {ity};\n}}\n").encode()
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td) / "c.mind"
        b = pathlib.Path(td) / "c.bin"
        s.write_bytes(src)
        r = subprocess.run(
            [MINDC, "build", str(s), "--release", "--emit=binary", "--out", str(b)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        if r.returncode != 0 or not b.exists():
            return -1
        b.chmod(b.stat().st_mode | stat.S_IEXEC)
        return subprocess.run([str(b)], timeout=30).returncode


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_scalar_narrow"):
        print("FAIL  selftest_native_elf_scalar_narrow: symbol absent (narrow rung not built)")
        return 1

    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, w, nmin, nmax in WIDTHS:
            # Two edge groups: finite clamp (in-range / >nmax / <nmin / NaN) and
            # sat-compose (i64-overflow / -overflow / +inf / -inf all clamp to a
            # bound). Both must byte-VALUE-match the numpy oracle.
            groups = [
                ("finite", [7.5, float(nmax) + 50.9, float(nmin) - 50.9, float("nan")]),
                ("sat", [1e30, -1e30, float("inf"), float("-inf")]),
            ]
            for gname, edges in groups:
                elf = _emit_narrow(lib, nmin, nmax, edges)
                if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                    print(f"  FAIL  {name}/{gname}: not a runnable ELF (len={len(elf)})")
                    all_ok = False
                    continue
                out = _run_elf(elf, tmp)
                if len(out) != 8:
                    print(f"  FAIL  {name}/{gname}: expected 8 stdout bytes, got {len(out)}: {out.hex()}")
                    all_ok = False
                    continue
                got = struct.unpack("<q", out)[0]
                want = _fold([_sat_narrow(e, nmin, nmax) for e in edges])
                ok = got == want
                all_ok = all_ok and ok
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}/{gname:6s} native={got} oracle={want}")

            # LIVE MLIR cross-check: grounds the oracle against the real narrow cast.
            live_cases = [
                ("7.5", 7),                    # in range
                (f"{nmax}.0 + 200.0", nmax),   # finite > nmax -> clamp
                (f"0.0 - {-nmin}.0 - 200.0", nmin),  # finite < nmin -> clamp
                ("1e30", nmax),                # i64-sat -> clamp high
                ("0.0 - 1e30", nmin),          # -sat -> clamp low
            ]
            for vsrc, want_val in live_cases:
                live = _live_exit(vsrc, name)
                want_exit = want_val & 0xFF
                ok = live == want_exit
                all_ok = all_ok and ok
                print(f"  [{'PASS' if ok else 'FAIL'}] {name}/live  `{vsrc} as {name}` "
                      f"live_exit={live} want={want_exit} (val {want_val})")

    if all_ok:
        print("ALL PASS  native-ELF f64->signed-narrow is byte-VALUE-identical to the "
              "numpy oracle AND the live MLIR cpu backend — saturate-then-clamp "
              "(nb_sat_cast_i64 + branchless maxsi/minsi), zero MLIR/LLVM (RI-D #10 slice 1a)")
        return 0
    print("FAIL  native-ELF narrow cast disagrees with an oracle above — do NOT guess; "
          "report the native/oracle values.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
