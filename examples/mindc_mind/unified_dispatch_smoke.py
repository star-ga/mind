"""
Self-host UNIFIED DISPATCHER smoke — proves ONE pure-MIND entry
(`selftest_mic3_module_unified_from_ast`) detects the construct class of a
`.mind` source and routes it to the correct already-proven mic@3 driver, with
the routed output byte-identical to the real `mindc --emit-mic3` oracle.

The unified entry dispatches over the three byte-verified drivers:
  - no struct present                -> the AST-flatten N-function driver
                                        (const / binop / let / call / if).
  - struct + `recv.field`            -> the field-read driver.
  - struct + `recv.method(args)`     -> the UFCS method-call driver.

Detection is token + struct-table driven (build the struct field table; if a
struct exists, find the first field-access dot and check whether the ident after
it is followed by `(`). See selftest_mic3_module_unified_from_ast in main.mind.

Each case is checked BYTE-EXACT against a hard-coded golden AND a LIVE
`--emit-mic3` regeneration (golden-staleness guard) — a self-golden never
cross-checked against real `mindc` is not acceptable. This does NOT touch the
canary (mindc_compile -> lower_program -> emit_fn_def stays stubbed), so
fixed_point_smoke.py stays byte-identical (Bound by MIND-CONSTITUTION §I).

Run:  python3 examples/mindc_mind/unified_dispatch_smoke.py
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
    """MIND String record: [+0] addr, [+8] len, [+16] cap."""
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length == 0:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def live_oracle(src_text: str):
    """Regenerate the real mic@3 oracle via `mindc --emit-mic3`, or None."""
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


# Each case: (route, src, golden_hex). One source per dispatched construct class.
# The golden_hex are the captured real `mindc --emit-mic3` bytes; the live oracle
# regenerates them every run so a stale golden is a hard FAIL.
CASES = [
    (
        "from_ast (arith — no struct)",
        "pub fn one() -> i64 { 1 }  pub fn add(a: i64, b: i64) -> i64 { a + b }",
        "4d4943330204036f6e6503616464016101620200061500000100000101000201"
        "000013001501020200030101020003180002001801030104020000010101001301"
        "000000",
    ),
    (
        "field (recv.field)",
        "struct S { a: i64 }  pub fn get_a(s: S) -> i64 { s.a }",
        "4d4943330205056765745f6101730f5f5f6d696e645f6c6f61645f6936340153"
        "016102000501000013001500010100010100021800010016010201000101001301"
        "0103010400" + "00",
    ),
    (
        "method (recv.method(args) — UFCS)",
        "struct Point { x: i64, y: i64 }\n"
        "pub fn point_dist(p: Point, k: i64) -> i64 { k }\n"
        "pub fn use_it(p: Point) -> i64 { p.dist(7) }\n",
        "4d49433302070a706f696e745f646973740170016b067573655f697405506f696e"
        "740178017903000801000013001500020100020101010002180001001801020101"
        "010013011503010100010200031800010001010e16020002000101020013020104"
        "0205060000",
    ),
]


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not pathlib.Path(so).exists():
        if "MINDC_SO" in os.environ:
            raise SystemExit(f"FAIL: MINDC_SO set but missing: {so}")
        print(f"SKIP: {so} not found (opt-in local build artifact)")
        return 0

    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_module_unified_from_ast
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 8

    print("self-host unified dispatcher — one entry routes to the proven driver")
    print(f"  .so: {so}")

    failures = 0
    for route, src, golden_hex in CASES:
        srcb = src.encode()
        srcc = ctypes.create_string_buffer(srcb, len(srcb))
        strbuf = ctypes.create_string_buffer(8192)
        offs = (ctypes.c_int64 * 256)()
        ccell = (ctypes.c_int64 * 1)()
        cbuf = ctypes.create_string_buffer(256)
        argbuf = (ctypes.c_int64 * 64)()
        valbuf = (ctypes.c_int64 * 64)()
        es = fn(
            ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
            ctypes.cast(strbuf, ctypes.c_void_p).value,
            ctypes.cast(offs, ctypes.c_void_p).value,
            ctypes.cast(ccell, ctypes.c_void_p).value,
            ctypes.cast(cbuf, ctypes.c_void_p).value,
            ctypes.cast(argbuf, ctypes.c_void_p).value,
            ctypes.cast(valbuf, ctypes.c_void_p).value)
        got = read_string_handle(read_i64_at(es, 0))
        golden = bytes.fromhex(golden_hex)
        live = live_oracle(src)
        if live is not None and live != golden:
            raise SystemExit(
                f"FAIL: unified golden stale vs live ({route})\n"
                f"  golden {golden.hex()}\n  live   {live.hex()}")
        want = live if live is not None else golden
        ok = got == want
        failures += not ok
        tag = (f"OK (routed -> {route}, byte-exact vs --emit-mic3, {len(got)}B)"
               if ok else f"FAIL want {want.hex()}")
        print(f"  [{len(got):>3}B] {got.hex()}  {tag}")

    if failures:
        print(f"\n{failures} FAILED ({len(CASES)} cases)")
        return 1
    print(f"\nALL {len(CASES)} dispatch cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
