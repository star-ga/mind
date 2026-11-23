"""
Whole-module mic@3 self-host FLIP gate.

Compares the current self-host MIND compiler's canonical binary IR with the
reference compiler on the exact current compiler source: main.mind + the
COMPILER_COMPONENTS parts as ONE translation unit (`single_unit`: the intra-unit
`import <part>;` lines dropped, as the parts are inline). The byte comparison
binds this source and these built instruments; it is not a general compiler
correctness proof, a runtime/type-soundness result, or pin qualification.

Feed that unit to selftest_mic3_module_nfn in the freshly built library and
compare the emitted mic@3 bytes against mindc --emit-mic3 on the same input.
Both legs get the same bytes. The Rust leg cannot take the unit with its part
imports: standalone it refuses a local import (E2003), and as a Mind.toml project
it adds the project struct registry (qualified std/crate structs) to the module.
Forty-five additive record-binding regressions retain this original whole-module
comparison and exercise descriptor-slot versus resolved-SSA-ID references.
Seven refusal pins hold else forms that the reference accepts and this emitter
refuses; a pin is red when the emitter admits its form.

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
from _selfhost_so import resolve_mindc, resolve_so  # noqa: E402
from _selfhost_sources import compiler_library, single_unit  # noqa: E402

SO = resolve_so()
MINDC = pathlib.Path(resolve_mindc())
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
    # An inner if assigns a let-mut of a block that returns: the later read is the merge id.
    ('returning-block-merge-read',
     'pub fn probe(w:i64,a:i64)->i64{if w!=0{let mut site:i64=a;if w==5{site=7;}return site+1;}a}\n'),
    # A returning block rebinds an outer name: the merge after it keeps the pre-block id.
    ('outer-name-after-returning-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}return s+1;}s}\n'),
    ('outer-name-after-returning-assign',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=7;return s+1;}s}\n'),
    ('outer-name-after-returning-while',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{while s<9{s=s+1;}return s+1;}s}\n'),
    ('outer-names-after-returning-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{if w==5{s=7;t=8;}return s+t;}s+t}\n'),
    # A returning block writes an outer name twice (a prefix assign, a while or a segment, then a later
    # segment): the reference merges by name, so the block emits ONE binding + merge for it.
    ('returning-assign-then-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}return s;}s}\n'),
    ('returning-assign-then-if-noread',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}return s;}0}\n'),
    ('returning-assign-then-if-ret0',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}return 0;}s}\n'),
    ('returning-mul-assign-then-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s*2;if w==5{s=7;}return s;}s*3}\n'),
    ('returning-two-ifs',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}if w==6{s=8;}return s;}s}\n'),
    ('returning-assign-two-ifs',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}if w==6{s=8;}return s;}s}\n'),
    ('returning-if-assign-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}s=s+1;if w==6{s=8;}return s;}s}\n'),
    ('returning-two-names-then-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{s=s+1;t=t+2;if w==5{s=7;t=9;}return s+t;}s+t}\n'),
    ('returning-two-names-order',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{t=t+2;s=s+1;if w==5{s=7;}return s+t;}s+t}\n'),
    ('returning-two-names-cross',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{s=s+1;if w==5{t=9;}if w==6{s=7;}return s+t;}s+t}\n'),
    ('returning-while-then-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{while s<9{s=s+1;}if w==5{s=7;}return s;}s}\n'),
    ('returning-assign-if-while',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}while s<9{s=s+1;}return s;}s}\n'),
    ('returning-assign-while',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;while s<9{s=s+1;}return s;}s}\n'),
    ('returning-if-then-while',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}while s<9{s=s+1;}return s;}s}\n'),
    ('returning-nested-returning-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;return s;}return s;}s}\n'),
    ('returning-inner-returning-block',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=s+1;if w==6{s=7;}return s;}return s+1;}s}\n'),
    ('returning-local-let-then-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{let mut q:i64=1;q=q+1;if w==5{q=7;}return q;}s}\n'),
    # An outer write made BEFORE a later shadowing `let` of the same name in a returning block still
    # escapes (one binding + merge); writes after the shadow are local. The last three are controls.
    ('returning-assign-then-shadow',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let s:i64=99;return s;}s}\n'),
    ('returning-assign-seg-then-shadow',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}let s:i64=99;return s;}s}\n'),
    ('returning-seg-then-shadow',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}let s:i64=99;return s;}s}\n'),
    ('returning-while-then-shadow',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{while s<9{s=s+1;}let s:i64=99;return s;}s}\n'),
    ('returning-assign-shadow-assign',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let mut s:i64=3;s=s+2;return s;}s}\n'),
    ('returning-assign-shadow-seg',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let mut s:i64=3;if w==5{s=7;}return s;}s}\n'),
    ('returning-assign-shadow-reads-outer',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let s:i64=s+5;return s;}s}\n'),
    ('returning-two-names-shadow-s',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{s=s+1;t=t+2;let s:i64=99;return s+t;}s+t}\n'),
    ('returning-two-names-shadow-t-seg',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{s=s+1;t=t+2;let t:i64=99;if w==5{s=7;}return s+t;}s+t}\n'),
    ('returning-assign-shadow-while',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let mut s:i64=3;while s<9{s=s+1;}return s;}s}\n'),
    ('returning-assign-shadow-twice',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let s:i64=99;let s:i64=s+1;return s;}s}\n'),
    ('returning-assign-shadow-ret0',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;let s:i64=99;return 0;}s}\n'),
    ('returning-shadow-only',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{let s:i64=99;return s;}s}\n'),
    ('returning-shadow-then-assign',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{let mut s:i64=99;s=s+1;return s;}s}\n'),
    ('returning-shadow-then-seg',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{let mut s:i64=99;if w==5{s=7;}return s;}s}\n'),
]

# Else forms in or around a returning block: the reference accepts each one and the self-host emitter
# REFUSES it (no bytes, fail closed). Each pin is red when the emitter admits the form: an admitted
# form must then prove byte equality as a row of RECORD_BINDING_CASES.
RECORD_BINDING_REFUSED_CASES = [
    ('returning-inner-if-else',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}else{s=8;}return s;}s}\n'),
    ('returning-if-else-assign-if',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}else{s=8;}s=s+1;if w==6{s=9;}return s;}s}\n'),
    ('returning-then-else-falls',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}return s;}else{s=s+2;}s}\n'),
    ('falling-then-returning-else',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w==0{s=s+2;}else{s=s+1;if w==5{s=7;}return s;}s}\n'),
    ('returning-inner-if-else-twice',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{if w==5{s=7;}else{s=8;}if w==6{s=1;}else{s=2;}return s;}s}\n'),
    ('returning-two-names-inner-else',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;let mut t:i64=w;if w!=0{s=s+1;t=t+1;if w==5{t=7;}else{s=8;}return s+t;}s+t}\n'),
    ('falling-assign-if-then-else',
     'pub fn probe(w:i64,a:i64)->i64{let mut s:i64=a;if w!=0{s=s+1;if w==5{s=7;}}else{s=s+2;}s}\n'),
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


def record_binding_refusal_pins():
    failures = 0
    for label, source in RECORD_BINDING_REFUSED_CASES:
        oracle = oracle_mic3(source)
        try:
            emitted = nfn_mic3(source)
        except Exception as exc:  # noqa: BLE001 — a crash is not a refusal
            print(f"FAIL: refusal pin {label}: driver raised {exc!r}")
            failures += 1
            continue
        if not oracle or emitted:
            verdict = "equal: move it to RECORD_BINDING_CASES" if emitted and emitted == oracle else "bytes differ"
            print(f"FAIL: refusal pin {label}: nfn={len(emitted)} B oracle={len(oracle or b'')} B ({verdict})")
            failures += 1
    if failures:
        print(f"FAIL: refusal pins {len(RECORD_BINDING_REFUSED_CASES) - failures}/{len(RECORD_BINDING_REFUSED_CASES)}")
        return 1
    print(f"PASS: refusal pins {len(RECORD_BINDING_REFUSED_CASES)}/{len(RECORD_BINDING_REFUSED_CASES)} refused, reference accepts")
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

    library = compiler_library()
    src = single_unit(library).decode()
    dropped = library.count(b"\n") - src.count("\n")
    print(f"single unit: main.mind + parts, {dropped} intra-unit part imports dropped")
    if dropped < 1:
        print("FAIL: no part import dropped (the composition rule moved)")
        return 1
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
        rows_rc = record_binding_regressions()
        return record_binding_refusal_pins() or rows_rc

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
