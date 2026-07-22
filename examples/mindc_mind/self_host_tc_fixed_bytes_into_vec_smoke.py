#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2006 FIXED_BYTES_INTO_VEC rule (Bug #38).

Builds main.mind -> .so via `mindc --emit-shared`, ctypes-calls the additive
`selftest_tc_fixed_bytes_into_vec` export, and asserts each verdict byte-for-byte
equals the Rust oracle recomputed here in Python.

Oracle — the scalar-decidable type-NAME compatibility core of
`fixed_bytes_into_vec_violation` in src/type_checker/mod.rs. Per (param, arg) pair
the guard fires iff:
  1. the parameter TypeAnn is the growable bytes vec:  `n == "bytes"`, AND
  2. the argument resolves to a fixed-size `bytes[N]` buffer, whose static type
     name satisfies `typeann_is_fixed_bytes`:  `n.starts_with("bytes[")`.
(The arg's EXPRESSION-shape recognition — `bytes[N].zero()`, a `bytes[N]` local
alias, or a call returning `bytes[N]` — needs full AST/local context and is not
ported; the arg is modelled by its resolved type-name, the value the Rust rule's
`typeann_is_fixed_bytes` branch actually inspects.)

Non-vacuous by construction: the case sweep contains >=1 positive (violation) and
>=1 negative (accepted) pairing.
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

# Type-name universe. Mixes the growable vec, several fixed-buffer forms, and
# unrelated / near-miss names (a name that merely *contains* "bytes" but isn't the
# growable vec, the bare "byte", etc.) so both discriminators are exercised on
# hits AND misses.
NAMES = ["bytes", "bytes[32]", "bytes[8]", "bytes[", "byte", "i64", "f64", "bool", "bytez", "bytes2"]


def name_to_args(name):
    """Encode a type name as (b0..b6, len): first 7 bytes zero-padded + true length.
    Only the leading <=6 bytes and the length matter to either discriminator."""
    b = [ord(c) for c in name[:7]] + [0] * 7
    return tuple(b[:7]) + (len(name),)


def rust_typeann_is_fixed_bytes(name):
    """typeann_is_fixed_bytes: n.starts_with("bytes[")."""
    return name.startswith("bytes[")


def rust_e2006(param_name, arg_name):
    """fixed_bytes_into_vec_violation core: param is exactly the growable "bytes"
    vec AND the arg's resolved type is a fixed `bytes[N]` buffer."""
    if param_name == "bytes" and rust_typeann_is_fixed_bytes(arg_name):
        return 1
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

    fn = lib.selftest_tc_fixed_bytes_into_vec
    fn.argtypes = [ctypes.c_int64] * 16
    fn.restype = ctypes.c_int64

    total = positives = fails = 0
    for param in NAMES:
        for arg in NAMES:
            args = name_to_args(param) + name_to_args(arg)
            got = fn(*args)
            exp = rust_e2006(param, arg)
            total += 1
            if exp != 0:
                positives += 1
            mark = "ok " if got == exp else "DIFF"
            if got != exp:
                fails += 1
            print(f"  {mark} E2006 param={param!r:12} arg={arg!r:12} got={got} exp={exp}")

    print(f"E2006: pairs={total} positives={positives} negatives={total - positives} fails={fails}")
    if positives < 1:
        print("FAIL: E2006 vacuous (no positive/violation case)")
        sys.exit(1)
    if total - positives < 1:
        print("FAIL: E2006 vacuous (no negative control)")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2006 rule diverges from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()