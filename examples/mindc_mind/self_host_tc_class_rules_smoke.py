#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2010/E2011/E2013/E2016 class rules.

Builds main.mind -> .so via `mindc --emit-shared`, ctypes-calls the additive
selftest exports over the type-name pairs drawn from {i32, i64, f64, bool},
and asserts each verdict byte-for-byte equals the Rust oracle rule recomputed
here in Python:

  E2010 selftest_tc_return_mismatch  — src/type_checker/mod.rs
        check_return_and_cond_node Return arm (is_int_scalar ret +
        is_float_scalar val) PLUS the confidence-gated NEW direction in
        check_scalar_class_stmt (ret class Float + value class Int).
  E2011 selftest_tc_cond_mismatch    — check_cond_type: boolean-intent exempt,
        else fires iff the condition is a float scalar.
  E2013 selftest_tc_mixed_binop      — walk_expr_class_checks Node::Binary arm:
        fires iff operand classes are (Int,Float) | (Float,Int).
  E2016 selftest_tc_as_bool          — walk_expr_class_checks Node::As arm:
        is_bool_ann(target) alone decides; source-agnostic.

Each rule is guarded on >=1 positive so a check that cannot fail cannot pass.
Selectors: 0=i32, 1=i64, 2=f64, 3=bool.
Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_class_mismatch_smoke.py.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

# selector -> type name (exactly the names resolve_type_ident recognises)
SEL_NAME = {0: "i32", 1: "i64", 2: "f64", 3: "bool"}

# Rust is_int_scalar: ScalarI32 | ScalarI64 | ScalarBool
INT_SCALARS = ("i32", "i64", "bool")
# Rust is_float_scalar: ScalarF32 | ScalarF64 (only f64 in the tag universe)
FLOAT_SCALARS = ("f64",)


def rust_scalar_class(name):
    """Mirror of scalar_class_of_ann for the names in play: 'int'/'float'/None."""
    if name in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                "isize", "usize", "bool"):
        return "int"
    if name in ("f32", "f64"):
        return "float"
    return None


def rust_e2010(ret_name, val_name):
    """RETURN_TYPE_MISMATCH, both Rust directions."""
    # Direction 1 (infer pass): int-declared return + float value.
    if ret_name in INT_SCALARS and val_name in FLOAT_SCALARS:
        return 1
    # Direction 2 (class pass): float-declared return + Int-class value.
    if rust_scalar_class(ret_name) == "float" and \
            rust_scalar_class(val_name) == "int":
        return 1
    return 0


def rust_e2011(cond_name, bool_intent):
    """COND_TYPE_MISMATCH: boolean-intent exempt, else float condition fires."""
    if bool_intent:
        return 0
    return 1 if cond_name in FLOAT_SCALARS else 0


def rust_e2013(lhs_name, rhs_name):
    """MIXED_CLASS_BINOP: exactly (Int,Float) | (Float,Int)."""
    lc = rust_scalar_class(lhs_name)
    rc = rust_scalar_class(rhs_name)
    if (lc, rc) in (("int", "float"), ("float", "int")):
        return 1
    return 0


def rust_e2016(target_name):
    """AS_BOOL: is_bool_ann(target) alone decides; source never consulted."""
    return 1 if target_name == "bool" else 0


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
        exp = oracle(*args_to_names(args, oracle))
        total += 1
        if exp == 1:
            positives += 1
        mark = "ok " if got == exp else "DIFF"
        if got != exp:
            fails += 1
        print(f"  {mark} {label} {render(args)} got={got} exp={exp}")
    return total, positives, fails


def args_to_names(args, oracle):
    """Map ctypes selector args to the oracle's name/flag args."""
    if oracle is rust_e2011:
        return (SEL_NAME[args[0]], args[1])
    return tuple(SEL_NAME[a] for a in args)


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

    rules = [
        ("E2010", export("selftest_tc_return_mismatch", 2),
         [(r, v) for r in SEL_NAME for v in SEL_NAME], rust_e2010,
         lambda a: f"fn -> {SEL_NAME[a[0]]:<4} return <{SEL_NAME[a[1]]:<4}>"),
        ("E2011", export("selftest_tc_cond_mismatch", 2),
         [(c, i) for c in SEL_NAME for i in (0, 1)], rust_e2011,
         lambda a: f"if <{SEL_NAME[a[0]]:<4}> intent={a[1]}"),
        ("E2013", export("selftest_tc_mixed_binop", 2),
         [(l, r) for l in SEL_NAME for r in SEL_NAME], rust_e2013,
         lambda a: f"<{SEL_NAME[a[0]]:<4}> op <{SEL_NAME[a[1]]:<4}>"),
        ("E2016", export("selftest_tc_as_bool", 1),
         [(t,) for t in SEL_NAME], rust_e2016,
         lambda a: f"<expr> as {SEL_NAME[a[0]]:<4}"),
    ]

    grand_fails = 0
    for label, fn, cases, oracle, render in rules:
        total, positives, fails = run_rule(label, fn, cases, oracle, render)
        print(f"{label}: pairs={total} positives={positives} fails={fails}")
        if positives < 1:
            print(f"FAIL: {label} vacuous (no positive case)")
            sys.exit(1)
        grand_fails += fails

    if grand_fails:
        print("FAIL: pure-MIND class rules diverge from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
