"""
Self-host REAL-MLIR smoke (Rust-independence #14, PHASE 1.1) — proves the
pure-MIND front-end can emit REAL MLIR text (func.func / arith.*) byte-identical
to the Rust backend's `mindc <f> --emit-mlir`, for the scalar i64 subset.

This is the FIRST increment of porting src/mlir/lowering.rs into main.mind. It
exercises the additive `selftest_emit_mlir` export (SECTION 4b in main.mind),
which is ISOLATED from the mic@1 canary / mic@3 flip — so the keystone
(phase_g_keystone_bootstrap) stays 7/7 byte-identical.

The oracle is regenerated LIVE from the freshly-built `mindc` binary (guards
golden staleness) AND cross-checked against a hard-coded golden so a silent
mindc change is caught too. NO fake wins — the pass requires a byte-for-byte
match of the pure-MIND emit vs the real `mindc --emit-mlir` stdout.

Run:  python3 examples/mindc_mind/self_host_mlir_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# The mindc oracle binary. CI / local both point MINDC_BIN at the freshly-built
# release binary; default to the in-repo target.
_REPO = _HERE.parent.parent
_DEFAULT_MINDC = _REPO / "target" / "release" / "mindc"
MINDC = pathlib.Path(os.environ.get("MINDC_BIN", str(_DEFAULT_MINDC)))

# Scalar i64 subset fixtures. Each is plain MIND source; the oracle is the
# byte-exact `mindc <f> --emit-mlir` stdout (regenerated live below).
CASES = [
    ("add", "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"),
    ("sub", "fn sub(a: i64, b: i64) -> i64 {\n    return a - b;\n}\n"),
    ("mul", "fn mul(a: i64, b: i64) -> i64 {\n    return a * b;\n}\n"),
    ("c", "fn c() -> i64 {\n    return 5;\n}\n"),
    ("two", "fn two(a: i64, b: i64) -> i64 {\n    let x: i64 = a + b;\n    return x * b;\n}\n"),
    ("call", "fn g(x: i64) -> i64 { return x; }\nfn h() -> i64 { return g(5); }\n"),
    ("if_ret", "fn f(c: i64, a: i64, b: i64) -> i64 {\n    if c == 0 {\n        return a;\n    }\n    return b;\n}\n"),
    ("if_val", "fn f(x: i64, a: i64, b: i64) -> i64 { if x { return a; } return b; }\n"),
    ("if_phi", "fn f(c: i64, a: i64, b: i64) -> i64 {\n    let r: i64 = if c == 0 { a } else { b };\n    return r;\n}\n"),
    ("if_phi_ret", "fn f(c: i64, a: i64, b: i64) -> i64 {\n    return if c == 0 { a } else { b };\n}\n"),
]

# Hard-coded golden (verified vs `mindc --emit-mlir` 2026-06-24) for `add` — a
# staleness tripwire on the live oracle. If mindc's MLIR shape changes, this
# mismatches and the run fails loud instead of silently re-blessing.
GOLDEN_ADD = (
    "module {\n"
    "  func.func @add(%0: i64, %1: i64) -> i64 {\n"
    "    %2 = arith.addi %0, %1 : i64\n"
    "    return %2 : i64\n"
    "  }\n"
    "  func.func @main() -> (i64) {\n"
    "    %0 = arith.constant 0 : i64\n"
    "    return %0 : i64\n"
    "  }\n"
    "}\n\n"
).encode()

# Hard-coded golden for `call` (func.call + the multi-fn synthetic @main with
# one i64 result per top-level fn). Verified vs `mindc --emit-mlir` 2026-06-24.
GOLDEN_CALL = (
    "module {\n"
    "  func.func @g(%0: i64) -> i64 {\n"
    "    return %0 : i64\n"
    "  }\n"
    "  func.func @h() -> i64 {\n"
    "    %0 = arith.constant 5 : i64\n"
    "    %1 = func.call @g(%0) : (i64) -> i64\n"
    "    return %1 : i64\n"
    "  }\n"
    "  func.func @main() -> (i64, i64) {\n"
    "    %0 = arith.constant 0 : i64\n"
    "    %1 = arith.constant 0 : i64\n"
    "    return %0, %1 : i64, i64\n"
    "  }\n"
    "}\n\n"
).encode()


def oracle_mlir(src: str) -> bytes:
    with tempfile.NamedTemporaryFile("w", suffix=".mind", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run(
            [str(MINDC), path, "--emit-mlir"],
            capture_output=True,
            check=True,
        )
        return out.stdout
    finally:
        os.unlink(path)


def emit_mlir(lib, src: bytes) -> bytes:
    buf = ctypes.create_string_buffer(src, len(src))
    es = lib.selftest_emit_mlir(ctypes.cast(buf, ctypes.c_void_p).value, len(src))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def show_diff(want: bytes, got: bytes) -> None:
    wl = want.split(b"\n")
    gl = got.split(b"\n")
    for i in range(max(len(wl), len(gl))):
        w = wl[i] if i < len(wl) else b"<none>"
        g = gl[i] if i < len(gl) else b"<none>"
        mark = "  " if w == g else ">>"
        print(f"    {mark} oracle={w!r:40}  mind={g!r}")


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0
    if not MINDC.exists():
        print(f"ERROR: mindc oracle binary not found at {MINDC}")
        return 1

    # Staleness tripwire: live `add` oracle must equal the hard-coded golden.
    live_add = oracle_mlir(CASES[0][1])
    if live_add != GOLDEN_ADD:
        print("ERROR: live mindc --emit-mlir for `add` drifted from the hard-coded golden:")
        show_diff(GOLDEN_ADD, live_add)
        return 1
    call_src = dict(CASES)["call"]
    live_call = oracle_mlir(call_src)
    if live_call != GOLDEN_CALL:
        print("ERROR: live mindc --emit-mlir for `call` drifted from the hard-coded golden:")
        show_diff(GOLDEN_CALL, live_call)
        return 1

    lib = ctypes.CDLL(str(SO))
    lib.selftest_emit_mlir.restype = ctypes.c_int64
    lib.selftest_emit_mlir.argtypes = [ctypes.c_int64, ctypes.c_int64]

    failures = 0
    for name, src in CASES:
        want = oracle_mlir(src)
        got = emit_mlir(lib, src.encode())
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<6} ({len(want)} oracle bytes / {len(got)} mind bytes)")
        if not ok:
            show_diff(want, got)
            failures += 1
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}  "
          f"({len(CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
