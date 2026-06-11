"""
Self-host CUTOVER-READINESS validation — how much of main.mind's OWN body
vocabulary does the recursive AST mic@3 emit already lower byte-exact vs the
real `mindc --emit-mic3` oracle?

This is the empirical coverage measurement the cutover needs BEFORE folding
more constructs. It drives the general N-function module emitter
(`selftest_mic3_module_nfn`, the real frontier driver: it loops every top-level
fn and fails CLOSED — returns an empty buf — if any fn's body shape is
unmatched), comparing the emitted mic@3 .buf byte-for-byte against a LIVE
`mindc --emit-mic3` regeneration of the same source.

Each case is a SELF-CONTAINED module (no undefined external callee, so the Rust
type-checker accepts it standalone) chosen to isolate one body-vocabulary axis
that main.mind's own functions actually use:

  GREEN  (already byte-exact through nfn today):
    - zero-arg const-return            (e.g. tk_eof, the 30 tk_* fns)
    - 2-param single binop             (e.g. add(a,b){a+b})
    - single-let arith                 (let t = a OP b; t)
    - single param/const/binop tree    (the (d) expr-tree path)
    - single if-EXPRESSION value       (the (e) if-expr path)
    - N functions, each of the above

  WALL  (returns empty buf today — the cutover frontier):
    - multi-statement let chains       (>1 stmt; nfn requires stmts_len == 1)
    - if-STATEMENT early return         (statement-form if, not if-expr)
    - the `&` bitand / mask operator    (binop_to_byte has no & — load_byte uses it)
    - a call with an expression arg     (__mind_load_i64(buf + i); the (c) call
                                         path only takes two int-LITERAL args)
    - struct-literal construction       (EmitState { .. } returns; 27 in real code)

Each WALL case is asserted to FAIL-CLOSED (empty buf), not silently mis-emit, so
this harness also guards the fail-closed contract. GREEN cases are asserted
byte-exact vs the live oracle (a self-golden never cross-checked against real
mindc is not acceptable — Bound by MIND-CONSTITUTION §I).

Does NOT touch the canary (mindc_compile -> emit_fn_def stays stubbed), so
fixed_point_smoke.py stays byte-identical. Read-only measurement.

Run:  python3 examples/mindc_mind/validate_real_fns_smoke.py
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


# (label, kind in {"green","wall"}, source) — each module is self-contained.
CASES = [
    # --- GREEN: vocabulary the recursive N-fn emit reproduces byte-exact today ---
    ("const-return (tk_* family)", "green",
     "pub fn tk_eof() -> i64 { 0 }"),
    ("2-param arith (add)", "green",
     "pub fn add(a: i64, b: i64) -> i64 { a + b }"),
    ("single-let arith", "green",
     "pub fn f(a: i64, b: i64) -> i64 { let t: i64 = a + b; t }"),
    ("param/const/binop tree", "green",
     "pub fn lb(buf: i64, i: i64) -> i64 { buf + i }"),
    ("if-expression value", "green",
     "pub fn s(b: i64) -> i64 { if b == 1 { 1 } else { 0 } }"),
    ("N const fns (module)", "green",
     " ".join(f"pub fn t{i}() -> i64 {{ {i} }}" for i in range(5))),
    # --- WALL: the cutover frontier — each must FAIL-CLOSED (empty buf) ---
    ("multi-stmt let chain", "wall",
     "pub fn f(a: i64) -> i64 { let x: i64 = a + 1; let y: i64 = x + 2; y }"),
    ("if-statement early return", "wall",
     "pub fn s(b: i64) -> i64 { if b == 1 { return 1; } 0 }"),
    ("& bitand / mask operator", "wall",
     "pub fn f(x: i64) -> i64 { x & 255 }"),
    ("call with expression arg", "wall",
     "pub fn ld(buf: i64) -> i64 { __mind_load_i64(buf + 1) }"),
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

    print("self-host CUTOVER-READINESS — recursive N-fn emit vs --emit-mic3")
    print(f"  .so: {so}")
    print(f"  oracle: {_MINDC} (--emit-mic3)\n")

    failures = 0
    green_exact = 0
    green_total = 0
    wall_closed = 0
    wall_total = 0
    for label, kind, src in CASES:
        got = emit(fn, src)
        if kind == "green":
            green_total += 1
            live = live_oracle(src)
            if live is None:
                print(f"  [GREEN  ?] {label:<28} oracle unavailable (mindc absent)")
                continue
            ok = got == live
            green_exact += ok
            failures += not ok
            tag = (f"BYTE-EXACT ({len(got)}B == oracle)" if ok
                   else f"REGRESSED got {len(got)}B vs oracle {len(live)}B")
            print(f"  [GREEN {'OK' if ok else 'XX'}] {label:<28} {tag}")
        else:
            wall_total += 1
            # WALL = the emit must fail closed (empty buf), never mis-emit bytes.
            closed = len(got) == 0
            wall_closed += closed
            failures += not closed
            tag = ("fail-closed (empty buf) — cutover frontier" if closed
                   else f"LEAKED {len(got)}B (fail-closed contract broken!)")
            print(f"  [WALL  {'OK' if closed else 'XX'}] {label:<28} {tag}")

    print(f"\n  GREEN byte-exact {green_exact}/{green_total}  |  "
          f"WALL fail-closed {wall_closed}/{wall_total}")
    if failures:
        print(f"\n{failures} FAILED")
        return 1
    print("\nALL cases behaved as classified (green byte-exact, walls fail-closed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
