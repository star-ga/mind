#!/usr/bin/env python3
"""general_float_netverify.py — GENERAL-path f64 value battery (B0 gate lift).

Rung 4 (`58df26aa`) fail-closed EVERY user float out of the general whole-
program driver (`selftest_native_elf_hb_from_tokens`) because the path leaked:
the parser silently DROPPED `as i64` (so `(f as i64)+5` became the mixed binop
`f+5`, whose GP fall-through added raw IEEE-754 bits — exit 5, not 7), and the
entry epilogue consumed rax while a float return sat in xmm0 (`return 1.5`
exited 0, the low byte of the movabs'd bits).

This harness locks the LIFT: the f64 shapes the dedicated fp tier already
proved (float lets/arith/compares/params/calls/returns) now compile through the
GENERAL driver and RUN to the correct value — `f as i64` with full SATURATING
Rust-`as` semantics (in-range trunc, +ovf->INT64_MAX, NaN->0, matching the MLIR
path's emit_saturating_fp_to_i64), and a float-returning entry exits
trunc(result) (cvttsd2si, the dedicated-tier scaffold convention). Every float
shape still outside the sound tier MUST refuse 0B — the two historical leak
values (5 and 0) must never reappear.

Usage: MINDC_SO=<built .so> python3 examples/mindc_mind/general_float_netverify.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

SO = os.environ.get("MINDC_SO")
if not SO or not pathlib.Path(SO).exists():
    print(f"SKIP: MINDC_SO not set / not found ({SO})")
    sys.exit(0)

lib = ctypes.CDLL(SO)
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


def M(body: str) -> str:
    return "fn main()->i64{ %s }" % body


# (label, source, want exit) — general-path f64 shapes MUST run VALUE-correct.
SUPPORTED = [
    # THE two historical leak probes (pre-rung-4 fail-OPEN values 5 and 0):
    ("cast+add leak probe (was 5)", M("let f:f64=2.0; return (f as i64) + 5;"), 7),
    ("cast+add trunc 2.5", M("let f:f64=2.5; return (f as i64) + 5;"), 7),
    ("float in int return (was 0)", M("return 1.5;"), 1),
    ("float-returning main", "fn main()->f64{ return 1.5; }", 1),
    ("float arith chain", M("let f:f64=1.5; let g:f64=f+2.0; return (g as i64);"), 3),
    ("float div", M("let f:f64=7.0; let g:f64=f/2.0; return (g as i64);"), 3),
    ("float param + call + cast",
     "fn add1(x:f64)->f64{ return x+1.0; }\nfn main()->i64{ let r:f64=add1(2.5); return (r as i64); }", 3),
    ("float compare cond", M("let f:f64=2.5; if f < 3.0 { return 42; } return 7;"), 42),
    ("float neg trunc", M("let f:f64=0.0-2.5; return (f as i64) + 10;"), 8),
    ("value-if float init", M("let r:f64 = if 1 < 2 { 1.5 } else { 2.5 }; return (r as i64);"), 1),
    # SATURATING `as i64` edges (Rust-`as`/trunc_sat, the MLIR-pinned semantics):
    ("sat +ovf -> INT64_MAX", M("let a:f64=65536.0; let b:f64=a*a; let c:f64=b*b; return (c as i64);"), 255),
    ("sat NaN -> 0", M("let z:f64=0.0; let n:f64=z/z; return (n as i64);"), 0),
    ("sat -ovf -> INT64_MIN", M("let a:f64=65536.0; let b:f64=a*a; let c:f64=0.0-b*b; return (c as i64);"), 0),
    # coexistence + regression of the new width-64 cast node on INT sources:
    ("float let unused", M("let f:f64=2.0; return 5;"), 5),
    ("float in sep fn", "fn h()->i64{ let f:f64=1.5; return 0; }\nfn main()->i64{ return h(); }", 0),
    ("f64 struct field decl only", "struct F { a: f64 }\nfn main()->i64{ return 0; }", 0),
    ("int as i64 identity", M("let x:i64=41; return (x as i64) + 1;"), 42),
    ("float let + int while", M("let f:f64=1.0; let mut c:i64=0; while c < 2 { c = c + 1; } return c;"), 2),
    # big-but-representable literals (<= 18 integer-content digits stay EXACT;
    # value-proving division so a wrong bit pattern cannot sneak through):
    ("2^52 literal exact (2^52/2^46=64)",
     M("let f:f64=4503599627370496.0; let g:f64=f/70368744177664.0; return (g as i64);"), 64),
    ("2^53 literal exact (2^53/2^46=128)",
     M("let f:f64=9007199254740992.0; let g:f64=f/70368744177664.0; return (g as i64);"), 128),
]

# (label, source) — float shapes OUTSIDE the sound tier MUST refuse 0B.
REFUSED = [
    ("mixed float+int binop", M("let f:f64=2.0; return f + 5;")),
    ("f32 let", M("let f:f32=1.5; return 1;")),
    ("non-dyadic float literal", M("let f:f64=0.1; return (f as i64);")),
    ("float % float", M("let f:f64=5.0; let g:f64=f % 2.0; return 0;")),
    ("narrowing cast of float", M("let f:f64=2.5; return f as i8;")),
    ("bare float if cond", M("let f:f64=1.5; if f { return 1; } return 2;")),
    ("bare float while cond", M("let f:f64=1.5; while f { return 0; } return 1;")),
    ("as f64 (int->float cast)", M("let x:i64=3; let f:f64=x as f64; return 0;")),
    # HUGE literals (>= 2^63): the i64 numerator in the literal builder WRAPS
    # (2^64 wraps num to exactly 0) and the wrapped value could pass the dyadic
    # test while the bits builder emitted ~0.0 — the blind-review fail-OPEN.
    # The 18-digit-content guard in nb_float_lit_is_dyadic refuses them 0B.
    ("2^63 literal", M("let f:f64=9223372036854775808.0; return (f as i64);")),
    ("2^64 literal (num wraps to 0)", M("let f:f64=18446744073709551616.0; return (f as i64);")),
    ("1e19 literal", M("let f:f64=1e19; return (f as i64);")),
    ("2^63 literal in compare", M("let f:f64=9223372036854775808.0; if f > 1.0 { return 7; } return 8;")),
]


def main() -> int:
    fails = 0
    print("== SUPPORTED general-path f64 shapes (MUST run VALUE-correct) ==")
    for lbl, src, want in SUPPORTED:
        st, rc = run(src)
        ok = st == "OK" and rc == (want & 0xFF)
        fails += 0 if ok else 1
        detail = f"exit {rc}" if st == "OK" else st
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {detail} want {want & 0xFF}")
    print("== REFUSED float shapes (MUST fail-closed to 0B, never wrong bytes) ==")
    for lbl, src in REFUSED:
        st, rc = run(src)
        ok = st == "0B"
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {'0B refused' if ok else f'LEAK ({st},{rc})'}")
    if fails:
        print(f"FAIL: {fails} general-path float violation(s)")
        return 1
    print("ALL PASS  (general-path f64: supported shapes value-correct, unsupported refuse 0B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
