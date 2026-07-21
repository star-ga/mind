#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2005/E2101/E2102/E2103 shape rules (E2023 deferred — blocked native-emitter/arena work).

Builds main.mind -> .so via `mindc --emit-shared`, ctypes-calls the additive
selftest exports, and asserts each verdict byte-for-byte equals the Rust
oracle rule recomputed here in Python:

  E2005 selftest_tc_arity_mismatch  — src/type_checker/mod.rs
        check_intra_fn_call: `args.len() != sig.param_count` alone decides.
  E2101 selftest_tc_broadcast_2x2 / _2x1 — src/shapes/engine.rs
        broadcast_shapes: right-aligned, missing dim = 1, each pair must be
        equal or 1 (`a == b || a == 1` -> b, `b == 1` -> a, else error).
  E2102 selftest_tc_matmul_shape (verdict 1) — engine MatMul2D rank guard:
        `lhs.len() != 2 || rhs.len() != 2`.
  E2103 selftest_tc_matmul_shape (verdict 2) — engine MatMul2D inner-dim
        guard: `lhs[1] != rhs[0]` (the `@` guard's lhs_k vs rhs_k).
  E2023 selftest_tc_reserved_prefix — check_module_types_in_file_impl:
        `name.starts_with("__mind_")` alone decides.

Each rule is guarded on >=1 positive so a check that cannot fail cannot pass.
Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_class_rules_smoke.py.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

# E2023 selector -> fn name (exactly the names the .mind byte tables encode)
RP_NAME = {0: "__mind_alloc", 1: "main", 2: "__mind_", 3: "__min", 4: "__mindx"}


def rust_e2005(expected, got):
    """CALL_ARITY: args.len() != sig.param_count."""
    return 1 if expected != got else 0


def rust_broadcast_dim(a, b):
    """engine broadcast_shapes loop body: dim or None on BroadcastError."""
    if a == b or a == 1:
        return b
    if b == 1:
        return a
    return None


def rust_e2101(lhs, rhs):
    """SHAPE_BROADCAST: right-aligned numpy-style; 1 iff any pair fails."""
    max_rank = max(len(lhs), len(rhs))
    for i in range(max_rank):
        a = lhs[len(lhs) - 1 - i] if i < len(lhs) else 1
        b = rhs[len(rhs) - 1 - i] if i < len(rhs) else 1
        if rust_broadcast_dim(a, b) is None:
            return 1
    return 0


def rust_matmul_shape(lhs_rank, lhs_k, rhs_rank, rhs_k):
    """engine MatMul2D, in guard order: 1=E2102 rank, 2=E2103 inner-dim, 0=ok."""
    if lhs_rank != 2 or rhs_rank != 2:
        return 1
    if lhs_k != rhs_k:
        return 2
    return 0


def rust_e2023(name):
    """Reserved intrinsic prefix: name.starts_with("__mind_")."""
    return 1 if name.startswith("__mind_") else 0


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


def run_rule(label, fn, cases, oracle, render):
    """Run one rule over its case list; return (total, positives, fails)."""
    total = positives = fails = 0
    for args in cases:
        got = fn(*args)
        exp = oracle(*args)
        total += 1
        if exp != 0:
            positives += 1
        mark = "ok " if got == exp else "DIFF"
        if got != exp:
            fails += 1
        print(f"  {mark} {label} {render(args)} got={got} exp={exp}")
    return total, positives, fails


def main():
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)

    def export(name, nargs):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_int64] * nargs
        fn.restype = ctypes.c_int64
        return fn

    dims = (1, 2, 3)
    rules = [
        ("E2005", export("selftest_tc_arity_mismatch", 2),
         [(e, g) for e in range(4) for g in range(4)], rust_e2005,
         lambda a: f"fn/{a[0]} called with {a[1]} arg(s)"),
        ("E2101(2x2)", export("selftest_tc_broadcast_2x2", 4),
         [(a0, a1, b0, b1) for a0 in dims for a1 in dims
          for b0 in dims for b1 in dims],
         lambda a0, a1, b0, b1: rust_e2101([a0, a1], [b0, b1]),
         lambda a: f"[{a[0]},{a[1]}] .op [{a[2]},{a[3]}]"),
        ("E2101(2x1)", export("selftest_tc_broadcast_2x1", 3),
         [(a0, a1, b0) for a0 in dims for a1 in dims for b0 in dims],
         lambda a0, a1, b0: rust_e2101([a0, a1], [b0]),
         lambda a: f"[{a[0]},{a[1]}] .op [{a[2]}]"),
        ("E2102/E2103", export("selftest_tc_matmul_shape", 4),
         [(lr, lk, rr, rk) for lr in dims for lk in (2, 3)
          for rr in dims for rk in (2, 3)],
         rust_matmul_shape,
         lambda a: f"matmul rank{a[0]}[.,k={a[1]}] @ rank{a[2]}[k={a[3]},.]"),
    ]

    grand_fails = 0
    for label, fn, cases, oracle, render in rules:
        total, positives, fails = run_rule(label, fn, cases, oracle, render)
        print(f"{label}: pairs={total} positives={positives} fails={fails}")
        if positives < 1:
            print(f"FAIL: {label} vacuous (no positive case)")
            sys.exit(1)
        if total - positives < 1:
            print(f"FAIL: {label} vacuous (no negative control)")
            sys.exit(1)
        grand_fails += fails

    if grand_fails:
        print("FAIL: pure-MIND arity/shape/prefix rules diverge from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
