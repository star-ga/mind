"""
Whole-module mic@3 self-host FLIP gate.

Proves the pure-MIND bootstrap compiler reproduces the CANONICAL BINARY IR exactly,
not just the mic@1 text fixed point: feed examples/mindc_mind/main.mind to the
self-hosted driver `selftest_mic3_module_nfn(...)` in libmindc_mind.so and compare its
emitted mic@3 module byte-for-byte against the Rust oracle `mindc --emit-mic3 main.mind`.

This is the load-bearing self-host invariant — the artifact the evidence chain anchors
on (trace_hash = mini_sha256(emit_mic3(ir))) is identical whether produced by Rust or by
MIND. Once green, the Rust front-end is decorative at the binary-IR layer too.

Verdicts:
  PASS  — nfn(main.mind) == mindc --emit-mic3 main.mind, byte-identical
  FAIL  — first differing byte + the surrounding window from each side
  BLOCKED — .so / mindc missing, or the driver returned no bytes

CI: point MINDC_SO at the freshly built self-host .so; a missing .so then HARD-FAILS
(refuses to skip, no false green). Local runs default to the .so next to this script.
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
# MINDC_SO (CI) verbatim; else build the self-host .so FRESH — never trust a
# stale in-tree libmindc_mind.so (a cargo build does not regenerate it).
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()
MINDC = pathlib.Path(
    os.environ.get("MINDC", str(_HERE.parents[1] / "target" / "release" / "mindc"))
)
MAIN = _HERE / "main.mind"

_P = ctypes.POINTER(ctypes.c_int64)


def _i64(addr, off=0):
    return int(ctypes.cast(addr + off, _P)[0])


def _read_string_record(handle):
    """EmitState.buf -> String record {addr, len, cap}; return its bytes."""
    if not handle:
        return b""
    addr, length = _i64(handle, 0), _i64(handle, 8)
    if not addr or not length:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def oracle_mic3(src: str):
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "m.mind"
        op = pathlib.Path(td) / "m.mic3"
        sp.write_text(src)
        subprocess.run(
            [str(MINDC), "--emit-mic3", str(op), str(sp)],
            capture_output=True,
        )
        return op.read_bytes() if op.exists() else None


def nfn_mic3(src: str):
    lib = ctypes.CDLL(str(SO))
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5
    sb = src.encode()
    sc = ctypes.create_string_buffer(sb, len(sb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
    cc = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(sc, ctypes.c_void_p).value,
        len(sb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(cc, ctypes.c_void_p).value,
    )
    return _read_string_record(_i64(es, 0)) if es else b""


def main():
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not found (build libmindc_mind.so first)")
        return 0
    if not MINDC.exists():
        print(f"BLOCKED: oracle mindc not found at {MINDC}")
        return 1

    src = MAIN.read_text()
    oracle = oracle_mic3(src)
    if oracle is None:
        print("BLOCKED: oracle `mindc --emit-mic3 main.mind` produced no output")
        return 1
    try:
        nfn = nfn_mic3(src)
    except Exception as exc:  # noqa: BLE001 — surface any FFI/driver failure
        print(f"BLOCKED: nfn driver raised {exc!r}")
        return 1
    if not nfn:
        print("BLOCKED: nfn driver returned empty output (fail-closed)")
        return 1

    if nfn == oracle:
        print(
            f"PASS: whole-module mic@3 FLIP byte-identical "
            f"({len(nfn)} B) — nfn(main.mind) == mindc --emit-mic3"
        )
        return 0

    n = min(len(nfn), len(oracle))
    di = next((i for i in range(n) if nfn[i] != oracle[i]), n)
    print(
        f"FAIL: mic@3 flip diverges — nfn={len(nfn)} B oracle={len(oracle)} B, "
        f"first diff @ byte {di}"
    )
    lo = max(0, di - 8)
    print(f"   nfn   : {list(nfn[lo:di + 8])}")
    print(f"   oracle: {list(oracle[lo:di + 8])}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
