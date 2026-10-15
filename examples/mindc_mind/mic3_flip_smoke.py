"""
Whole-module mic@3 self-host FLIP gate.

Compares the current self-host MIND compiler's canonical binary IR with the
reference compiler on the exact current main.mind source. The byte comparison
binds this source and these built instruments; it is not a general compiler
correctness proof, a runtime/type-soundness result, or pin qualification.

Feed main.mind to selftest_mic3_module_nfn in the freshly built library and
compare the emitted mic@3 bytes against mindc --emit-mic3 on the same input.
Eight additive record-binding regressions retain this original whole-module
comparison and exercise descriptor-slot versus resolved-SSA-ID references.

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
        result = subprocess.run(
            [str(MINDC), "--emit-mic3", str(op), str(sp)],
            capture_output=True,
        )
        return op.read_bytes() if result.returncode == 0 and op.exists() else None


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


# Additive regression cases for the descriptor-slot versus resolved-SSA-ID boundary.
# Local let-call bindings exposed wrong MIC references even when module lengths
# matched. Preserve exact source bytes; these checks do not assert runtime typing.
RECORD_BINDING_CASES = [
    ('parameter-control',
     'struct Other { v:i64 }\npub fn probe(a:i64,b:i64)->Ghost{Ghost{x:a,y:b}}\n'),
    ('let-call-record',
     'struct Other { v:i64 }\npub fn probe(p:i64)->Ghost{let a:i64=__mind_load_i64(p);let b:i64=__mind_load_i64(p+8);let c:i64=__mind_load_i64(p+16);Ghost{x:a,y:b,z:c}}\n'),
    ('let-call-no-schema',
     'pub fn probe(p:i64)->Ghost{let a:i64=__mind_load_i64(p);let b:i64=__mind_load_i64(p+8);Ghost{x:a,y:b}}\n'),
    ('let-arithmetic-record',
     'struct Other { v:i64 }\npub fn probe(p:i64)->Ghost{let a:i64=p+2;let b:i64=a*3;Ghost{x:a,y:b}}\n'),
    ('let-call-known',
     'struct Ghost{x:i64,y:i64}\npub fn probe(p:i64)->Ghost{let a:i64=__mind_load_i64(p);let b:i64=__mind_load_i64(p+8);Ghost{x:a,y:b}}\n'),
    ('local-shadow-record',
     'struct Other { v:i64 }\npub fn probe(a:i64)->Ghost{let a:i64=a+2;let b:i64=a*3;Ghost{x:a,y:b}}\n'),
    ('nested-let-record',
     'struct Other { v:i64 }\npub fn probe(p:i64)->Ghost{let a:i64=p+2;Ghost{x:Inner{v:a},y:a}}\n'),
    ('array-let-record',
     'struct Other { v:i64 }\npub fn probe(p:i64)->i64{let a:i64=p+2;let xs=[Ghost{x:a}];0}\n'),
]


def record_binding_regressions():
    failures = 0
    for label, source in RECORD_BINDING_CASES:
        oracle = oracle_mic3(source)
        try:
            emitted = nfn_mic3(source)
        except Exception as exc:  # noqa: BLE001 — refusal/crash is not success
            print(f"FAIL: record binding {label}: driver raised {exc!r}")
            failures += 1
            continue
        if not oracle or not emitted or emitted != oracle:
            print(
                f"FAIL: record binding {label}: "
                f"nfn={len(emitted)} B oracle={len(oracle or b'')} B"
            )
            failures += 1
    if failures:
        print(f"FAIL: record binding regressions {len(RECORD_BINDING_CASES) - failures}/{len(RECORD_BINDING_CASES)}")
        return 1
    print(f"PASS: record binding regressions {len(RECORD_BINDING_CASES)}/{len(RECORD_BINDING_CASES)} byte-identical")
    return 0


def main():
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"BLOCKED: {SO} not found (build libmindc_mind.so first)")
        return 1
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
        return record_binding_regressions()

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
