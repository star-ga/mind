#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2015 int<->float class-mismatch rule.

Builds main.mind -> .so via `mindc --emit-shared`, ctypes-calls the additive
`selftest_tc_class_mismatch(to_sel, from_sel)` export over the type-name pairs
drawn from {i32, i64, f64, bool}, and asserts each verdict byte-for-byte equals
the Rust oracle rule (src/type_checker/mod.rs {scalar_class_of_ann, the E2015
LET_CLASS_MISMATCH arm}) recomputed here in Python.

Selectors: 0=i32, 1=i64, 2=f64, 3=bool.
Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_narrowing_smoke.py.
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
    """Mirror of scalar_class_of_ann for the names in play: 'int'/'float'/None."""
    if name in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                "isize", "usize", "bool"):
        return "int"
    if name in ("f32", "f64"):
        return "float"
    return None


def rust_e2015(to_name, from_name):
    """LET_CLASS_MISMATCH: 1 iff both classes known and differ, else 0."""
    tc = rust_scalar_class(to_name)
    fc = rust_scalar_class(from_name)
    if tc is None or fc is None:
        return 0
    return 1 if tc != fc else 0


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
    fn = lib.selftest_tc_class_mismatch
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    fn.restype = ctypes.c_int64

    total = 0
    positives = 0
    fails = 0
    for to_sel, to_name in SEL_NAME.items():
        for from_sel, from_name in SEL_NAME.items():
            got = fn(to_sel, from_sel)
            exp = rust_e2015(to_name, from_name)
            total += 1
            if exp == 1:
                positives += 1
            mark = "ok " if got == exp else "DIFF"
            if got != exp:
                fails += 1
            print(f"  {mark} let dst: {to_name:<4} = <{from_name:<4}> "
                  f"got={got} exp={exp}")

    print(f"pairs={total} positives={positives} fails={fails}")
    if positives < 1:
        print("FAIL: vacuous (no positive mismatch case)")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2015 diverges from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
