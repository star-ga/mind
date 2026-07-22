#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND shape annotation-compat rule.

Ports `check_tensor_shape_compat` (src/type_checker/mod.rs ~4952-5014): at a
`let x: Tensor<...> = rhs` site the declared tensor annotation is compared to
the inferred tensor type in a fixed guard order, each arm pushing a precise
`shape::*` diagnostic:

  1  shape::dtype_mismatch  — `ann.dtype != inferred.dtype` (checked first;
     shape comparison is meaningless across dtypes).
  2  shape::rank_mismatch   — `ann.shape.len() != inferred.shape.len()`
     (checked before individual dims).
  3  shape::dim_mismatch    — for each right-aligned/positional axis, only when
     BOTH sides are `ShapeDim::Known`, `an != inf`. The Rust returns a single
     `pushed` bool (true if any concrete axis differed), which verdict 3 models.
  0  compatible / handled by the generic path.

Symbolic (non-`Known`) dims are encoded here as a negative value (< 0); the
pattern gate `(ShapeDim::Known, ShapeDim::Known)` skips them, so they are always
compatible at a single binding site. An axis absent at the current rank is
passed symbolic so it can never manufacture a mismatch.

The export `selftest_tc_shape_annot_compat` is additive and reached ONLY through
this selftest, never through mindc_compile, so the self-host module stays
byte-identical. Each verdict is guarded on >=1 positive AND >=1 negative so a
check that cannot fail cannot pass.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_shape_rules_smoke.py.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")


def rust_shape_annot_compat(ann_dtype, inf_dtype, ann_rank, inf_rank, ann_dims, inf_dims):
    """Recompute check_tensor_shape_compat's verdict in Rust guard order.

    ann_dims/inf_dims are per-axis dim values with negative == symbolic (skipped).
    """
    if ann_dtype != inf_dtype:
        return 1
    if ann_rank != inf_rank:
        return 2
    for a, i in zip(ann_dims, inf_dims):
        if a >= 0 and i >= 0 and a != i:
            return 3
    return 0


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, MAIN_MIND, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit(1)
    return out, True


def main():
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)

    fn = lib.selftest_tc_shape_annot_compat
    fn.argtypes = [ctypes.c_int64] * 8
    fn.restype = ctypes.c_int64

    # Case sweep. Each case: (ann_dtype, inf_dtype, ann_rank, inf_rank,
    #   ann_dims, inf_dims) where dims are the up-to-2 axes, negative=symbolic.
    # The export takes (a0,i0,a1,i1); an absent axis is passed -1 (symbolic).
    # dtype selectors: 0=f32, 1=f64, 2=i32 (only equality matters, as in Rust).
    cases = [
        # --- dtype mismatch (verdict 1): dominates even with equal shapes ---
        (0, 1, 2, 2, [3, 4], [3, 4]),          # f32 vs f64, same shape
        (2, 0, 1, 1, [5], [5]),                # i32 vs f32, same shape
        (0, 1, 2, 1, [3, 4], [3]),             # dtype beats rank mismatch
        # --- rank mismatch (verdict 2): same dtype, differing rank ---
        (0, 0, 2, 1, [3, 4], [3]),             # rank 2 vs 1
        (0, 0, 1, 2, [3], [3, 4]),             # rank 1 vs 2
        (2, 2, 2, 1, [7, 7], [7]),             # rank beats dim mismatch below
        # --- dim mismatch (verdict 3): same dtype+rank, concrete dims differ --
        (0, 0, 2, 2, [3, 4], [3, 5]),          # axis1 differs
        (0, 0, 2, 2, [3, 4], [9, 4]),          # axis0 differs
        (0, 0, 1, 1, [8], [6]),                # rank1 dim differs
        (0, 0, 2, 2, [2, 2], [3, 3]),          # both axes differ (still one verdict)
        # --- symbolic dims never mismatch (verdict 0) ---
        (0, 0, 2, 2, [-1, 4], [3, 4]),         # axis0 ann symbolic
        (0, 0, 2, 2, [3, 4], [3, -1]),         # axis1 inf symbolic
        (0, 0, 2, 2, [-1, -1], [3, 4]),        # both axes ann symbolic
        (0, 0, 1, 1, [-1], [5]),               # rank1 symbolic
        # --- fully compatible (verdict 0) ---
        (0, 0, 2, 2, [3, 4], [3, 4]),          # identical concrete shape
        (1, 1, 1, 1, [5], [5]),                # identical rank1
        (2, 2, 2, 2, [3, -1], [3, -1]),        # concrete-then-symbolic match
    ]

    total = fails = 0
    verdict_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for ann_dt, inf_dt, ann_rk, inf_rk, ad, idd in cases:
        # Pad dims to 2 axes with symbolic (-1) for the fixed-arity export.
        a0 = ad[0] if len(ad) > 0 else -1
        a1 = ad[1] if len(ad) > 1 else -1
        i0 = idd[0] if len(idd) > 0 else -1
        i1 = idd[1] if len(idd) > 1 else -1
        got = fn(ann_dt, inf_dt, ann_rk, inf_rk, a0, i0, a1, i1)
        exp = rust_shape_annot_compat(ann_dt, inf_dt, ann_rk, inf_rk, ad, idd)
        total += 1
        verdict_counts[exp] = verdict_counts.get(exp, 0) + 1
        mark = "ok " if got == exp else "DIFF"
        if got != exp:
            fails += 1
        print(f"  {mark} dt({ann_dt}vs{inf_dt}) rk({ann_rk}vs{inf_rk}) "
              f"dims({ad}vs{idd}) got={got} exp={exp}")

    print(f"shape_annot_compat: cases={total} fails={fails} verdicts={verdict_counts}")
    # Non-vacuous: at least one of each verdict (dtype/rank/dim positives + the
    # compatible negative control).
    for v, name in [(1, "dtype"), (2, "rank"), (3, "dim"), (0, "compatible")]:
        if verdict_counts.get(v, 0) < 1:
            print(f"FAIL: vacuous — no {name} (verdict {v}) case")
            sys.exit(1)
    if fails:
        print("FAIL: pure-MIND shape annotation-compat rule diverges from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()