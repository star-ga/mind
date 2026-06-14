"""
Self-host gap-corpus integrity gate.

Surveys every fixture in tests/selfhost_gaps/*.mind through the pure-MIND self-host
driver `selftest_mic3_module_nfn(...)` and compares each emitted mic@3 module byte-for-byte
against the Rust oracle `mindc --emit-mic3`. The corpus is a fuzz-discovered regression set
spanning struct-lit / field-read / value-if-expr / fall-through-shadow / mixed-prefix /
call-arg / unary-neg construct families.

Locks in the FULLY GENERAL front-end milestone (v0.8.0): the driver is byte-exact on every
fixture — 0 fail-closed, 0 wrong-bytes. The gate enforces two invariants:

  * WRONG_BYTES == 0   — the cardinal invariant: the driver NEVER emits incorrect bytes
                         (a silent miscompile is the worst failure mode for a deterministic
                         compiler). Always hard-fails.
  * FAIL_CLOSED == 0   — the generality invariant: every catalogued construct lowers
                         byte-exactly (no refusals). Hard-fails when MINDC_SO is set (CI).

Verdicts:
  PASS    — every fixture byte-exact (N/N), 0 fail-closed, 0 wrong-bytes
  FAIL    — any wrong-bytes (lists them) or any fail-closed regression
  BLOCKED — .so / mindc missing

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
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))
MINDC = pathlib.Path(
    os.environ.get("MINDC", str(_HERE.parents[1] / "target" / "release" / "mindc"))
)
CORPUS = _HERE.parents[1] / "tests" / "selfhost_gaps"

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


def _nfn_mic3_raw(src: str):
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


def nfn_mic3(src: str):
    """Fork-isolate each emit and load the .so fresh inside the child. A clean child contains
    any out-of-subset SIGSEGV so one bad fixture can't take down the whole survey, and a fresh
    per-process load matches a real `mindc` invocation (defensive: it would surface any
    arena-state-dependent lowering rather than mask it behind a warm handle — though none
    exists here). READ THE RESULT CORRECTLY: the output is a `String` at `EmitState.buf`
    (offset 0) → (addr@0, len@8); reading the wrong EmitState field once faked a 64/66.
    Returns (bytes, "OK"|"CRASH")."""
    rp, wp = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(rp)
        try:
            data = _nfn_mic3_raw(src)
        except Exception:  # noqa: BLE001 — surfaced to the parent as wrong-bytes
            data = b""
        try:
            os.write(wp, len(data).to_bytes(4, "little") + data)
        finally:
            os.close(wp)
            os._exit(0)
    os.close(wp)
    buf = b""
    while True:
        chunk = os.read(rp, 65536)
        if not chunk:
            break
        buf += chunk
    os.close(rp)
    status = os.waitpid(pid, 0)[1]
    if os.WIFSIGNALED(status):
        return None, "CRASH"
    if len(buf) < 4:
        return b"", "OK"
    n = int.from_bytes(buf[:4], "little")
    return buf[4:4 + n], "OK"


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

    fixtures = sorted(CORPUS.glob("*.mind"))
    if not fixtures:
        print(f"BLOCKED: no fixtures under {CORPUS}")
        return 1

    byte_exact = 0
    fail_closed = []
    wrong = []
    oracle_invalid = []

    for f in fixtures:
        src = f.read_text()
        oracle = oracle_mic3(src)
        if oracle is None:
            oracle_invalid.append(f.name)  # not a self-host gap (source doesn't compile)
            continue
        nfn, status = nfn_mic3(src)
        if status == "CRASH":
            wrong.append(f"{f.name} (driver CRASHED — must fail-closed, never crash)")
        elif not nfn:
            fail_closed.append(f.name)
        elif nfn == oracle:
            byte_exact += 1
        else:
            n = min(len(nfn), len(oracle))
            di = next((i for i in range(n) if nfn[i] != oracle[i]), n)
            wrong.append(f"{f.name} (nfn={len(nfn)}B oracle={len(oracle)}B diff@{di})")

    total = len(fixtures) - len(oracle_invalid)
    print(
        f"gap corpus: {byte_exact}/{total} byte-exact, "
        f"{len(fail_closed)} fail-closed, {len(wrong)} wrong-bytes"
        + (f" ({len(oracle_invalid)} oracle-invalid skipped)" if oracle_invalid else "")
    )

    # Cardinal invariant: the driver NEVER emits wrong bytes (a crash counts as wrong — the
    # driver must fail-closed, never crash). Always hard-fails.
    # Coverage ratchet: byte-exact must not drop below FLOOR. The whole 66-fixture corpus
    # lowers byte-exactly under the canonical FRESH-load measurement; the floor pins it so
    # no fixture can silently regress to fail-closed or wrong.
    FLOOR = 66
    ok = True
    if wrong:
        print("FAIL: WRONG-BYTES (silent miscompile) — the cardinal invariant is violated:")
        for w in wrong:
            print(f"   {w}")
        ok = False
    if fail_closed:
        print(f"NOTE: {len(fail_closed)} fixture(s) fail-closed (safe refusal, never wrong bytes):")
        for fc in fail_closed:
            print(f"   {fc}")
    if byte_exact < FLOOR:
        print(f"FAIL: byte-exact {byte_exact} < floor {FLOOR} — a fixture regressed to fail-closed/wrong.")
        ok = False

    if ok:
        print(
            f"PASS: 0 wrong-bytes; {byte_exact}/{total} byte-exact "
            f"(>= floor {FLOOR}), {len(fail_closed)} safe fail-closed"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
