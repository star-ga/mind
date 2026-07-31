#!/usr/bin/env python3
"""self_host_struct_return_smoke.py — regression lock for STRUCT-BY-VALUE
RETURNS and NON-PARAM struct receivers in the pure-MIND native-ELF emitter
(Rust-independence #40, PHASE-B).

Background — a STALE-ROADMAP correction
---------------------------------------
INDEPENDENCE_ROADMAP.md's B0 residual once read "non-param struct receivers /
struct returns (still refused 0B — future feature slice)". A VERIFY-FIRST sweep
(2026-07-25) proved that STALE: the mechanism that supports both landed with the
pure-MIND native backend on 2026-06-24 (nb_struct_decl_descriptor +
nb_let_descriptor's "Step-2 FieldAccess" arm) — a note written on 2026-07-22
under-claimed. Concretely, in the native-ELF path today:

  * A struct is an i64 HANDLE (a heap block pointer via the __mind_alloc +
    base+8*i ABI). Returning one `fn mk() -> P { P{..} }` returns the pointer in
    rax — the ordinary i64 return path, no hidden-return-slot ABI needed.
  * A NON-PARAM receiver bound to that returned struct — `let p: P = mk(); p.x`
    — resolves its field offset through nb_let_descriptor's Step-2 arm: the let's
    DECLARED type `: P` names a struct in the srt registry, so `p` binds to that
    struct's synthetic field-offset descriptor (nb_struct_decl_descriptor). This
    is the dominant shape in main.mind's OWN source
    (`let r: ParseResult = parse_...(); r.field`) — the self-host loop smoke
    reproducing main.mind byte-identically IS the proof the native path handles
    it. This smoke locks it against regression with SMALL fixtures.

Return-type inference for the UNANNOTATED receiver (LANDED)
----------------------------------------------------------
`let p = mk(); p.x` (no `: P`) now RUNS: nb_let_descriptor's ty_node==0 arm
infers p's struct descriptor from the callee's `-> P` return type via the frt
registry's new return-type-name span (nb_struct_desc_from_call), so an
unannotated struct-returning-call receiver resolves its field offset exactly
like the annotated `let p: P = mk()` case. Threaded through nb_let_descriptor at
all 4 count/emit call sites in lockstep (frt read from the emit lcell+16 /
count clcell+16). Non-struct call inits resolve to 0 (unchanged scalar-let
behavior), so main.mind's own compile stays byte-identical.

Direct field on a struct-returning call `mk().x` (LANDED)
---------------------------------------------------------
`mk().x` (a field read whose receiver is a struct-returning CALL, not an ident)
now RUNS correctly: nb_field_recv_desc_r resolves the call's struct descriptor
from the callee's `-> P` return type (the SAME nb_struct_desc_from_call path the
unannotated `let p = mk(); p.x` case uses), and nb_field_offset_r loads
base+offset*8 off the returned i64 struct handle. Read arms only — the field
descriptor is inferred identically for the ident-receiver case, so main.mind's
own `let r = parse_...(); r.field` reads emit byte-identically.

Fail-closed boundary (the genuine remaining gaps — correctly refused today)
-------------------------------------------------------------------------
Adjacent shapes the REFERENCE frontend (`mindc --emit-ir`) accepts but the
native path still refuses 0B — honest fail-closed gaps, NOT miscompiles:

  * Chained field on a call `mk().x.y` — the outer receiver `mk().x` is itself a
    field access (not a call/ident with a resolvable descriptor), so it defers.
  * Field on a NON-struct-returning call `fn mk()->i64{..}; mk().x` — no struct
    descriptor to infer, so the field read refuses.

Each refusal is CORRECT (fail-closed beats a wrong-value ELF) and is asserted
below as a control, so a future change that makes it run must do so *correctly*
or trip this smoke.

Usage: MINDC_SO=<built .so> python3 examples/mindc_mind/self_host_struct_return_smoke.py
"""
import ctypes
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()
if not SO or not pathlib.Path(str(SO)).exists():
    print(f"SKIP: self-host .so not resolved ({SO})")
    sys.exit(0)

lib = ctypes.CDLL(str(SO))
lib.selftest_native_elf_h.restype = ctypes.c_int64
lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3
_rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]


def emit(src: str) -> bytes:
    b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
    h = ctypes.create_string_buffer(bytes(32), 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(b, ctypes.c_void_p).value, len(src.encode()),
        ctypes.cast(h, ctypes.c_void_p).value)
    sh = _rd(es, 0)
    ln = _rd(sh, 8)
    return ctypes.string_at(_rd(sh, 0), ln) if ln > 0 else b""


def run(src: str):
    e = emit(src)
    if not e:
        return ("0B", None)
    p = pathlib.Path(tempfile.mktemp())
    p.write_bytes(e)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    try:
        return ("OK", subprocess.run([str(p)], timeout=10).returncode)
    finally:
        try:
            p.unlink()
        except OSError:
            pass


P = "struct P { x: i64, y: i64 }\n"
Q = "struct Q { a: i64, b: i64, c: i64 }\n"

# (label, source, want)  — want is a HAND-ENUMERATED expected exit code derived
# from the fixture's own semantics, INDEPENDENT of the port's algorithm (Rule 3a:
# the oracle must not mirror the thing under test). Each is a deterministic
# constant program.
SUPPORTED = [
    # struct RETURN + annotated non-param receiver: mk() returns the P handle,
    # `let p: P = mk()` binds the struct descriptor from the `: P` annotation,
    # p.x / p.y read offset 0 / 8 of the returned block.
    ("ret struct, recv .x",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ let p:P=mk(); return p.x; }", 7),
    ("ret struct, recv .y",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ let p:P=mk(); return p.y; }", 9),
    # struct return through a synthetic-else value-if body.
    ("ret struct from value-if fn",
     P + "fn pick(c:i64)->P{ if c==1 { return P{x:7,y:0}; } return P{x:1,y:0}; }\n"
         "fn main()->i64{ let p:P=pick(1); return p.x; }", 7),
    ("ret struct from value-if fn (else path)",
     P + "fn pick(c:i64)->P{ if c==1 { return P{x:7,y:0}; } return P{x:1,y:0}; }\n"
         "fn main()->i64{ let p:P=pick(0); return p.x; }", 1),
    # struct return fed DIRECTLY to a param-typed accessor: get(mk()).
    ("ret struct into param accessor",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn get(r:P)->i64{ return r.x; }\n"
         "fn main()->i64{ return get(mk()); }", 7),
    # chained struct returns: shift(mk()) both return structs; the receiver s is a
    # non-param annotated let bound to the SECOND struct-returning call.
    ("chained struct returns shift(mk())",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn shift(r:P)->P{ return P{x:r.x,y:0}; }\n"
         "fn main()->i64{ let s:P=shift(mk()); return s.x; }", 7),
    # 3-field struct return + non-param receiver, out-of-declared-order literal
    # (declared-order layout must hold across the call boundary).
    ("ret 3-field struct ooo, recv .a",
     Q + "fn mk()->Q{ return Q{c:3,a:1,b:2}; }\nfn main()->i64{ let q:Q=mk(); return q.a; }", 1),
    ("ret 3-field struct ooo, recv .c",
     Q + "fn mk()->Q{ return Q{c:3,a:1,b:2}; }\nfn main()->i64{ let q:Q=mk(); return q.c; }", 3),
    # non-param receiver where the struct-returning call takes an ARG.
    ("ret struct from call-with-arg, recv",
     P + "fn mk(n:i64)->P{ return P{x:n,y:0}; }\nfn main()->i64{ let p:P=mk(42); return p.x; }", 42),
    # a param struct receiver (already covered by failclosed smoke — kept here as
    # an adjacent control so the two receiver kinds sit side by side).
    ("param struct receiver (control)",
     P + "fn get(r:P)->i64{ return r.x; }\nfn main()->i64{ let r:P=P{x:7,y:9}; return get(r); }", 7),
    # UNANNOTATED call-init receiver (LANDED): descriptor inferred from mk's
    # `-> P` return type — no `: P` annotation needed.
    ("unannotated call-init receiver .x",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ let p=mk(); return p.x; }", 7),
    ("unannotated call-init receiver .y",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ let p=mk(); return p.y; }", 9),
    ("unannotated call-init receiver, 3-field .b",
     Q + "fn mk()->Q{ return Q{a:1,b:2,c:3}; }\nfn main()->i64{ let q=mk(); return q.b; }", 2),
    ("unannotated call-init receiver, ooo lit + call-with-arg",
     Q + "fn mk(n:i64)->Q{ return Q{c:3,a:n,b:2}; }\nfn main()->i64{ let q=mk(5); return q.a; }", 5),
    ("unannotated receiver, mut let",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ let mut p=mk(); return p.x; }", 7),
    # DIRECT field on a struct-returning call `mk().x` (LANDED, b9d596ac): the
    # receiver is a call expr; its struct descriptor is inferred from mk's `-> P`
    # return type, and the field loads off the returned handle at offset 0 / 8.
    ("direct field on call: mk().x",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ return mk().x; }", 7),
    ("direct field on call: mk().y",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ return mk().y; }", 9),
    # two struct-returning calls composed in arithmetic (each resolves its own
    # descriptor independently): mk().x + mk().y == 7 + 9.
    ("direct field on call, arith mk().x + mk().y",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ return mk().x + mk().y; }", 16),
]

# (label, source)  — MUST refuse 0B. Genuine fail-closed gaps: the native path
# cannot resolve the receiver's struct type, so it refuses rather than emit a
# wrong-value ELF. The reference frontend ACCEPTS these (they are valid MIND);
# implementing them correctly is the next slice — a change that makes them RUN
# must return the right value or this control trips.
REFUSED = [
    # chained field on a call `mk().x.y` — the OUTER receiver mk().x is itself a
    # field access (no resolvable descriptor), so it defers (0B), never a
    # wrong-value ELF. A control that the direct-call support does not over-fire
    # onto chained-receiver shapes.
    ("chained field on call: mk().x.y",
     P + "fn mk()->P{ return P{x:7,y:9}; }\nfn main()->i64{ return mk().x.y; }"),
    # direct field on a NON-struct-returning call `fn mk()->i64{..}; mk().x` MUST
    # refuse: the callee returns i64, so there is no struct descriptor to infer —
    # a control that the call-receiver inference does not over-fire on non-structs.
    ("direct field on scalar-return call: mk().x (no struct)",
     "fn mk()->i64{ return 5; }\nfn main()->i64{ return mk().x; }"),
    # a scalar-returning call bound unannotated then field-accessed MUST still
    # refuse (no struct descriptor to infer — the return type is i64, not a
    # struct): a control that the inference does not over-fire on non-structs.
    ("unannotated scalar-call receiver .x (no struct to infer)",
     "fn mk()->i64{ return 5; }\nfn main()->i64{ let p=mk(); return p.x; }"),
]


def main() -> int:
    fails = 0
    print("== SUPPORTED (struct return + non-param annotated receiver MUST run correct) ==")
    for lbl, src, want in SUPPORTED:
        st, rc = run(src)
        ok = st == "OK" and rc == (want & 0xFF)
        fails += 0 if ok else 1
        detail = f"exit {rc}" if st == "OK" else st
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {detail} want {want}")
    print("== REFUSED (genuine fail-closed gaps MUST refuse 0B, never a wrong-value ELF) ==")
    for lbl, src in REFUSED:
        st, rc = run(src)
        ok = st == "0B"
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {'0B refused' if ok else f'LEAK ({st},{rc})'}")
    if fails:
        print(f"FAIL: {fails} struct-return/receiver regression(s)")
        return 1
    print("ALL PASS  (struct-by-value returns + non-param annotated receivers run correct; "
          "unresolvable-type receivers fail-closed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
