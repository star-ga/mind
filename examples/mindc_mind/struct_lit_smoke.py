"""
Self-host STRUCT-LITERAL CONSTRUCTION emit — byte-exact vs `mindc --emit-mic3`.

The pure-MIND mindc lowers a single-struct, single-param fn that returns ONE
struct literal `T { f: e, ... }` by desugaring it to the same shape the Rust
compiler emits: `ptr = __mind_alloc(n_fields*8)`, then one
`__mind_store_i64(ptr + idx*8, value)` per field in DECLARATION order, with the
allocated pointer as the function result. Field values may be an int-literal or
the (single) parameter ident.

Entry: `selftest_mic3_module_structlit_from_ast` (5 args: src, len, strbuf, offs,
ccell) — the AST-driven driver. Each case is compared byte-for-byte against a
live `mindc --emit-mic3` regeneration of the same source (a self-golden never
cross-checked against real mindc is not acceptable — Bound by MIND-CONSTITUTION
§I). Additive / canary-isolated: the canary mindc_compile path is untouched, so
fixed_point_smoke stays byte-identical.

Run:  python3 examples/mindc_mind/struct_lit_smoke.py
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


def _rd(addr: int, off: int = 0) -> int:
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def _read_handle(handle: int) -> bytes:
    if handle == 0:
        return b""
    addr = _rd(handle, 0)
    length = _rd(handle, 8)
    if addr == 0 or length == 0:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def _emit(fn, src: str) -> bytes:
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
    return _read_handle(_rd(es, 0))


def _oracle(src: str):
    if not _MINDC.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "case.mind"
        outp = pathlib.Path(td) / "case.mic3"
        srcp.write_text(src)
        r = subprocess.run(
            [str(_MINDC), "--emit-mic3", str(outp), str(srcp)],
            capture_output=True)
        if r.returncode != 0 or not outp.exists():
            return None
        return outp.read_bytes()


CASES = [
    ("two fields, param + literal",
     "pub struct Point { x: i64, y: i64 }\n"
     "pub fn mk(a: i64) -> Point { Point { x: a, y: 0 } }"),
    ("two fields, both literals",
     "pub struct Point { x: i64, y: i64 }\n"
     "pub fn origin(a: i64) -> Point { Point { x: 0, y: 0 } }"),
    ("single field, param value",
     "pub struct Box { v: i64 }\n"
     "pub fn wrap(a: i64) -> Box { Box { v: a } }"),
    ("three fields, mixed",
     "pub struct Tri { a: i64, b: i64, c: i64 }\n"
     "pub fn mk(p: i64) -> Tri { Tri { a: p, b: 7, c: p } }"),
    # Multi-param: each field value is an int-literal or ANY param ident, resolved
    # to that param's %SSA vid (param k binds vid k). The first self-host
    # constructor shape (pr_new-style: 2+ params, param-ident field values).
    ("pr_new, two params, both param-idents (field names alias params)",
     "pub struct ParseResult { next_pos: i64, node: i64 }\n"
     "pub fn pr_new(next_pos: i64, node: i64) -> ParseResult "
     "{ ParseResult { next_pos: next_pos, node: node } }"),
    ("report, three params, all param-idents",
     "pub struct Report { ok: i64, warns: i64, errs: i64 }\n"
     "pub fn mk_report(ok: i64, warns: i64, errs: i64) -> Report "
     "{ Report { ok: ok, warns: warns, errs: errs } }"),
    ("env, three params, mixed literal + cross-ordered param-idents",
     "pub struct Env { a: i64, b: i64, c: i64 }\n"
     "pub fn mk_env(p: i64, q: i64, r: i64) -> Env "
     "{ Env { a: q, b: 5, c: p } }"),
    # Step B: field VALUES are arbitrary ARITHMETIC expressions, lowered by the
    # general tree emitter and vid-interleaved through the alloc/store spine.
    ("arith value, param+1 in field 0",
     "pub struct Point { x: i64, y: i64 }\n"
     "pub fn mk(a: i64, b: i64) -> Point { Point { x: a + 1, y: b } }"),
    ("arith value, both fields arithmetic",
     "pub struct Vec2 { x: i64, y: i64 }\n"
     "pub fn mk(a: i64, b: i64) -> Vec2 { Vec2 { x: a * 2, y: a + b } }"),
    ("arith value, nested precedence + literal field",
     "pub struct T3 { a: i64, b: i64, c: i64 }\n"
     "pub fn mk(p: i64, q: i64) -> T3 { T3 { a: p * q + 1, b: 7, c: q - p } }"),
    ("field-access values, copy(s:E)->E{a:s.a,b:s.b}",
     "pub struct E { a: i64, b: i64 }\n"
     "pub fn copy(s: E) -> E { E { a: s.a, b: s.b } }"),
    ("field-access + literal mix",
     "pub struct P { x: i64, y: i64 }\n"
     "pub fn shift(p: P) -> P { P { x: p.x, y: 0 } }"),
    # Step B2b: MULTI-FN modules — a struct-lit ctor fn alongside helper fns, with
    # call-valued fields (B{v:id(a)}). Needs item-order multi-fn assembly + the
    # `pub struct` parse fix + the structural-call-node vid skip in _base.
    ("multi-fn, param value, B{v:a}",
     "fn id(x: i64) -> i64 { x }\n"
     "pub struct B { v: i64 }\n"
     "pub fn mk(a: i64) -> B { B { v: a } }"),
    ("multi-fn, CALL value, B{v:id(a)}",
     "fn id(x: i64) -> i64 { x }\n"
     "pub struct B { v: i64 }\n"
     "pub fn w(a: i64) -> B { B { v: id(a) } }"),
]


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not pathlib.Path(so).exists():
        if "MINDC_SO" in os.environ:
            raise SystemExit(f"FAIL: MINDC_SO set but missing: {so}")
        print(f"SKIP: {so} not found (opt-in local build artifact)")
        return 0

    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_module_structlit_from_ast
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5

    print("self-host STRUCT-LITERAL construction — emit vs --emit-mic3")
    failures = 0
    for label, src in CASES:
        got = _emit(fn, src)
        oracle = _oracle(src)
        if oracle is None:
            print(f"  [?] {label:<32} oracle unavailable (mindc absent)")
            continue
        ok = got == oracle
        failures += not ok
        tag = (f"BYTE-EXACT ({len(got)}B == oracle)" if ok
               else f"MISMATCH got {len(got)}B vs oracle {len(oracle)}B")
        print(f"  [{'OK' if ok else 'XX'}] {label:<32} {tag}")

    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print(f"\nALL PASS  ({len(CASES)} cases byte-exact vs live oracle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
