#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2015 LET_CLASS_MISMATCH let/assign rule.

Builds main.mind -> .so via `mindc --emit-shared`, ctypes-calls the additive
`selftest_tc_let_class_mismatch(ann_sel, val_sel)` export over the type-name
pairs drawn from {i32, i64, f64, bool}, and asserts each verdict byte-for-byte
equals the Rust oracle recomputed here in Python — the Node::Let / Node::Assign
`ann_class != val_class` guard in check_scalar_class_stmt
(src/type_checker/mod.rs, LET_CLASS_MISMATCH_CODE = "E2015"), driven by
scalar_class_of_ann.

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


def rust_scalar_class(name):
    """Mirror of scalar_class_of_ann for the names in play: 'int'/'float'/None.

    In this tag universe bool is Int-class (matches tc_scalar_class), unlike the
    width rule E2004.
    """
    if name in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                "isize", "usize", "bool"):
        return "int"
    if name in ("f32", "f64"):
        return "float"
    return None


def rust_e2015_let(ann_name, val_name):
    """LET_CLASS_MISMATCH: 1 iff both classes are known scalar and differ, else 0.

    Faithful to the Node::Let / Node::Assign arm: no diagnostic unless both the
    annotated/declared class and the value's confident class are Some and unequal.
    """
    ac = rust_scalar_class(ann_name)
    vc = rust_scalar_class(val_name)
    if ac is None or vc is None:
        return 0
    return 1 if ac != vc else 0


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
    fn = lib.selftest_tc_let_class_mismatch
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    fn.restype = ctypes.c_int64

    total = 0
    positives = 0
    negatives = 0
    fails = 0
    for ann_sel, ann_name in SEL_NAME.items():
        for val_sel, val_name in SEL_NAME.items():
            got = fn(ann_sel, val_sel)
            exp = rust_e2015_let(ann_name, val_name)
            total += 1
            if exp == 1:
                positives += 1
            else:
                negatives += 1
            mark = "ok " if got == exp else "DIFF"
            if got != exp:
                fails += 1
            print(f"  {mark} let dst: {ann_name:<4} = <{val_name:<4}> "
                  f"got={got} exp={exp}")

    print(f"pairs={total} positives={positives} negatives={negatives} fails={fails}")
    if positives < 1:
        print("FAIL: vacuous (no positive mismatch case)")
        sys.exit(1)
    if negatives < 1:
        print("FAIL: vacuous (no negative match case)")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2015 let/assign diverges from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()