#!/usr/bin/env python3
"""Runnable verification for examples/columnar/tiled_fold.mind.

Phase 0.5 of the deterministic multi-format columnar front-end: the RADIX-4096
TILED FOLD (RFC DRAFT-deterministic-format-frontend §2e). This harness compiles
the .mind file with `mindc --emit-shared` and drives the entry points through
ctypes over hand-verified buffers, asserting the f64 result BIT-FOR-BIT against
an independent Python reference that computes the IDENTICAL radix-4096 tiled
association (per-tile L->R fold, then L->R fold of the tile-partials) — NOT a
flat numpy sum.

Proofs:
  * BIT-IDENTICAL: MIND tiled_fold_sum == Python tiled reference, compared via
    struct.pack('<d', ...) raw bit patterns, on N in {1, 4096, 10000}.
  * DETERMINISM: identical bytes on a second run.
  * PINNED (not accidental): for a fractional input the tiled association differs
    OBSERVABLY (in bits) from a flat left-to-right sum — proven on the compiled
    artifact (MIND flat_fold_sum) AND against the Python flat reference.

Inputs use exactly-representable integer-valued f64 (expected value hand-checked
via N(N+1)/2) PLUS one fractional input where tiled != flat.

Usage:
    python3 examples/columnar/tiled_fold_test.py [path/to/mindc]
Deterministic: no clock, no randomness; output is a pure function of the bytes.
"""
import ctypes
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "tiled_fold.mind"
REPO = HERE.parent.parent  # examples/columnar -> repo root
DEFAULT_MINDC = REPO / "target" / "debug" / "mindc"

TILE = 4096  # must match tiled_fold.mind's radix (the PINNED_FOLD_ELEM_CAP)


def bits(x: float) -> str:
    """Raw IEEE-754 double bit pattern as hex, for byte-identity comparison."""
    return struct.pack("<d", x).hex()


def tiled_reference(vals: list[float]) -> float:
    """Independent reference: the SAME radix-4096 tiled association as the .mind.

    Per-tile L->R fold (ragged last tile, no zero-pad), then L->R fold of the
    tile-partials. Python float is IEEE-754 double with RNE rounding, matching
    MIND f64 arith.addf exactly.
    """
    n = len(vals)
    partial_acc = 0.0
    base = 0
    while base < n:
        end = min(base + TILE, n)
        tile_sum = 0.0
        i = base
        while i < end:
            tile_sum = tile_sum + vals[i]
            i += 1
        partial_acc = partial_acc + tile_sum
        base += TILE
    return partial_acc


def flat_reference(vals: list[float]) -> float:
    """Flat left-to-right fold over all elements (the contrast association)."""
    acc = 0.0
    for v in vals:
        acc = acc + v
    return acc


def make_buf(vals: list[float]):
    """Pack f64 values into a C buffer; return (buffer, address)."""
    if not vals:
        return None, 0
    buf = (ctypes.c_double * len(vals))(*vals)
    return buf, ctypes.addressof(buf)


# ---- Corpus -----------------------------------------------------------------
# Exactly-representable integer-valued cases: vals[i] = i+1, so the expected sum
# is N(N+1)/2 (hand-checkable, exact for N(N+1)/2 < 2^53). tiled == flat here
# because every partial is an exact integer (no rounding).
def ints_1_to_n(n: int) -> list[float]:
    return [float(k) for k in range(1, n + 1)]


CASES = [
    ("N=1        (1 tile)",            ints_1_to_n(1)),
    ("N=4096     (1 full tile)",       ints_1_to_n(4096)),
    ("N=4097     (2 tiles 4096+1, ragged tail 1)", ints_1_to_n(4097)),
    ("N=8192     (2 exact full tiles)", ints_1_to_n(8192)),
    ("N=10000    (3 tiles 4096+4096+1808)", ints_1_to_n(10000)),
]

# Supported domain of tiled_fold_sum: n <= 4096*4096 (see tiled_fold.mind). This
# harness MUST NOT exercise beyond it — above the domain the single-pass partial
# fold is a different association than the (deferred) level-3 radix-4096
# recursion, so any such result would be out-of-contract, not ground truth.
MAX_N = TILE * TILE  # 16_777_216

# Fractional case where the radix-4096 tiled association differs OBSERVABLY from
# a flat sum: 10000 copies of 0.1 (not exactly representable), so the grouping
# imposed by the tile boundaries changes the accumulated rounding.
FRAC = ("N=10000    frac 0.1x  (tiled != flat)", [0.1] * 10000)

# All-`-0.0` case: the `0.0` accumulator seed is POSITIVE zero, so
# `0.0 + -0.0 = +0.0` under IEEE-754 RNE for every add — an all-negative-zero
# column folds to +0.0 (bit pattern 0x0000000000000000), matching the shipped
# pinned fold's seed. Two full tiles ensure the tile-partial fold also seeds at
# +0.0. Pins that the seed is +0.0, not a `-0.0 + 0.0` padding edge.
NEG_ZERO = ("N=8192     all -0.0  (seed pins +0.0)", [-0.0] * 8192)
POS_ZERO_BITS = "0000000000000000"


def main() -> int:
    mindc = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MINDC
    if not mindc.exists():
        print(f"mindc not found: {mindc}", file=sys.stderr)
        return 2

    # F-0.5-2 domain fence: tiled_fold_sum is only valid for n <= 4096*4096.
    # Never exercise the oracle beyond its supported domain (the level-3 tier is
    # deferred), or the compared reference would not be the shipped association.
    all_inputs = [vals for _, vals in CASES] + [FRAC[1], NEG_ZERO[1]]
    for vals in all_inputs:
        assert len(vals) <= MAX_N, (
            f"test case N={len(vals)} exceeds supported domain MAX_N={MAX_N} "
            f"(4096*4096); level-3 radix-4096 recursion is deferred"
        )

    with tempfile.TemporaryDirectory() as td:
        so = Path(td) / "tiled_fold.so"
        cp = subprocess.run(
            [str(mindc), str(SRC), "--emit-shared", str(so)],
            capture_output=True, text=True,
        )
        if cp.returncode != 0 or not so.exists():
            print(cp.stdout)
            print(cp.stderr, file=sys.stderr)
            print(f"compile failed (exit {cp.returncode})", file=sys.stderr)
            return 1
        print(f"compiled: {so.name} ({so.stat().st_size} bytes)\n")

        lib = ctypes.CDLL(str(so))
        for name in ("tiled_fold_sum", "flat_fold_sum"):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_double
            fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
        lib.tile_count.restype = ctypes.c_int64
        lib.tile_count.argtypes = [ctypes.c_int64]

        ok = True

        # ---- Exact integer-valued cases: bit-identity + hand-check + determinism
        print("== radix-4096 tiled fold: exact integer-valued inputs ==")
        print(f"{'case':40} {'N':>6} {'tiles':>5} {'MIND tiled bits':>18} "
              f"{'ref bits':>18} {'expect':>12}  match")
        for label, vals in CASES:
            n = len(vals)
            buf, addr = make_buf(vals)
            r_mind = lib.tiled_fold_sum(addr, n)
            r_mind2 = lib.tiled_fold_sum(addr, n)  # determinism: second run
            r_ref = tiled_reference(vals)
            expect = n * (n + 1) // 2  # N(N+1)/2, exact
            tiles = lib.tile_count(n)
            exp_tiles = (n + TILE - 1) // TILE
            bit_match = bits(r_mind) == bits(r_ref)
            det = bits(r_mind) == bits(r_mind2)
            hand = (r_mind == float(expect))
            tile_ok = (tiles == exp_tiles)
            good = bit_match and det and hand and tile_ok
            ok = ok and good
            print(f"{label:40} {n:>6} {tiles:>5} {bits(r_mind):>18} "
                  f"{bits(r_ref):>18} {expect:>12}  "
                  f"{'OK' if good else 'FAIL'}"
                  f"{'' if det else ' [NONDET]'}"
                  f"{'' if hand else ' [HANDCHK]'}"
                  f"{'' if tile_ok else ' [TILES]'}")
            del buf

        # ---- Fractional case: tiled == ref (bit-identical) AND tiled != flat
        print("\n== pinned-association proof: fractional input ==")
        label, vals = FRAC
        n = len(vals)
        buf, addr = make_buf(vals)
        t_mind = lib.tiled_fold_sum(addr, n)
        t_mind2 = lib.tiled_fold_sum(addr, n)
        f_mind = lib.flat_fold_sum(addr, n)
        t_ref = tiled_reference(vals)
        f_ref = flat_reference(vals)
        del buf

        tiled_bit_match = bits(t_mind) == bits(t_ref)
        tiled_det = bits(t_mind) == bits(t_mind2)
        flat_bit_match = bits(f_mind) == bits(f_ref)
        # The association is PINNED (distinct from flat) iff the bits differ.
        assoc_distinct = bits(t_mind) != bits(f_mind)
        good = tiled_bit_match and tiled_det and flat_bit_match and assoc_distinct
        ok = ok and good

        print(f"{label}")
        print(f"  MIND tiled_fold_sum  = {t_mind!r}  bits={bits(t_mind)}")
        print(f"  ref  tiled           = {t_ref!r}  bits={bits(t_ref)}  "
              f"match={tiled_bit_match}")
        print(f"  MIND flat_fold_sum   = {f_mind!r}  bits={bits(f_mind)}")
        print(f"  ref  flat            = {f_ref!r}  bits={bits(f_ref)}  "
              f"match={flat_bit_match}")
        print(f"  determinism (2nd run): {tiled_det}")
        print(f"  tiled != flat (association is PINNED, not accidental): "
              f"{assoc_distinct}")
        print(f"  -> {'OK' if good else 'FAIL'}")

        # ---- Seed pinning: all-(-0.0) tile folds to +0.0 (positive-zero seed)
        print("\n== seed pinning: all -0.0 input folds to +0.0 ==")
        label, vals = NEG_ZERO
        n = len(vals)
        buf, addr = make_buf(vals)
        z_mind = lib.tiled_fold_sum(addr, n)
        z_ref = tiled_reference(vals)
        del buf
        z_bits = bits(z_mind)
        z_ref_bits = bits(z_ref)
        z_match = z_bits == z_ref_bits
        z_pos = z_bits == POS_ZERO_BITS       # exactly +0.0, not -0.0
        ref_pos = z_ref_bits == POS_ZERO_BITS
        good = z_match and z_pos and ref_pos
        ok = ok and good
        print(f"{label}")
        print(f"  MIND tiled_fold_sum  = {z_mind!r}  bits={z_bits}")
        print(f"  ref  tiled           = {z_ref!r}  bits={z_ref_bits}  "
              f"match={z_match}")
        print(f"  +0.0 bit pattern ({POS_ZERO_BITS}): MIND={z_pos} ref={ref_pos}")
        print(f"  -> {'OK' if good else 'FAIL'}")

    print("\nALL_MATCH" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
