"""
Self-host MULTI-STATEMENT LET CHAIN smoke — the recursive AST mic@3 emit lowers a
body of the form `{ let n0 = E0; let n1 = E1; ... ; FINAL }` byte-exact vs the real
`mindc --emit-mic3` oracle.

main.mind's own functions use multi-let constantly, so this is the highest-impact
cutover construct after single-let. The Rust front-end treats sequential lets as
pure SSA aliasing — each let name takes the vid of its init expression's root, and a
later reference reuses that vid with no extra instruction or string. The pure-MIND
emitter (selftest_mic3_module_nfn -> emit_mic3_module_fndef -> the (f) multi-let
branch -> emit_mic3_module_fndef_letchain) reproduces this by flattening every let
init + the trailing expr into ONE shared post-order descriptor with a let-environment
(name-span -> root slot), then feeding the proven tree_resolve_vids + tree_body pair.

Each case carries a HARD-CODED golden (captured from a real `mindc --emit-mic3`) AND
is re-checked against a LIVE regeneration of the oracle — the live check guards golden
staleness, the hard-coded golden guards against the oracle drifting silently with the
emitter (Bound by MIND-CONSTITUTION §I — a self-golden never cross-checked against real
mindc is not acceptable).

Does NOT touch the canary (mindc_compile -> emit_fn_def stays stubbed), so
fixed_point_smoke.py stays byte-identical.

Run:  python3 examples/mindc_mind/multi_let_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
_MINDC = _HERE.parents[1] / "target" / "release" / "mindc"
_Int64Ptr = ctypes.POINTER(ctypes.c_int64)


def read_i64_at(addr: int, off: int = 0) -> int:
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def read_string_handle(handle: int) -> bytes:
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length == 0:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def live_oracle(src_text: str):
    if not _MINDC.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "case.mind"
        outp = pathlib.Path(td) / "case.mic3"
        srcp.write_text(src_text)
        r = subprocess.run(
            [str(_MINDC), "--emit-mic3", str(outp), str(srcp)],
            capture_output=True,
        )
        if r.returncode != 0 or not outp.exists():
            return None
        return outp.read_bytes()


# (label, source, hard-coded golden hex from a real `mindc --emit-mic3`).
CASES = [
    ("2-let, second uses first",
     "pub fn f(x: i64) -> i64 { let a: i64 = x + 1; let b: i64 = a * 2; b }",
     "4d49433302020166017801000315000101000104000518000100010102040200"
     "000101030404040202030100001300000000"),
    ("2-let, final uses both",
     "pub fn g(x: i64, y: i64) -> i64 "
     "{ let s: i64 = x + y; let d: i64 = x - y; s * d }",
     "4d4943330203016701780179010003150002010002010104000518000100180102"
     "010402000001040301000104040202030100001300000000"),
    ("3-let chain, final ref",
     "pub fn h(x: i64) -> i64 "
     "{ let a: i64 = x + 1; let b: i64 = a + 2; let c: i64 = b + 3; c }",
     "4d494333020201680178010003150001010001060007180001000101020402000001"
     "010304040400020301050604060004050100001300000000"),
    ("const-only init then use",
     "pub fn f(x: i64) -> i64 { let a: i64 = 5; let b: i64 = x + a; b }",
     "4d4943330202016601780100031500010100010200031800010001010a04020000"
     "010100001300000000"),
]


def emit(fn, src: str) -> bytes:
    srcb = src.encode()
    srcc = ctypes.create_string_buffer(srcb, len(srcb))
    strbuf = ctypes.create_string_buffer(32768)
    offs = (ctypes.c_int64 * 1024)()
    ccell = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(ccell, ctypes.c_void_p).value)
    return read_string_handle(read_i64_at(es, 0))


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not pathlib.Path(so).exists():
        if "MINDC_SO" in os.environ:
            raise SystemExit(f"FAIL: MINDC_SO set but missing: {so}")
        print(f"SKIP: {so} not found (opt-in local build artifact)")
        return 0

    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5

    print("self-host MULTI-LET — recursive emit vs hard-coded + live --emit-mic3")
    print(f"  .so: {so}")
    print(f"  oracle: {_MINDC} (--emit-mic3)\n")

    failures = 0
    for label, src, golden_hex in CASES:
        golden = bytes.fromhex(golden_hex)
        got = emit(fn, src)
        # Staleness guard: the hard-coded golden must still equal the live oracle.
        live = live_oracle(src)
        stale = live is not None and live != golden
        ok_golden = got == golden
        ok = ok_golden and not stale
        failures += not ok
        if stale:
            tag = (f"GOLDEN STALE vs live oracle "
                   f"(golden {len(golden)}B != live {len(live)}B)")
        elif ok_golden:
            tag = f"BYTE-EXACT ({len(got)}B == golden == oracle)"
        else:
            tag = f"MISMATCH got {len(got)}B vs golden {len(golden)}B"
        print(f"  [{'OK' if ok else 'XX'}] {label:<28} {tag}")
        if not ok:
            print(f"       got:    {got.hex()}")
            print(f"       golden: {golden.hex()}")
            if live is not None:
                print(f"       live:   {live.hex()}")

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print("\nALL multi-let cases byte-exact (hard-coded golden == live oracle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
