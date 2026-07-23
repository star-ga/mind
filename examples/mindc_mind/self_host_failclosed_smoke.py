#!/usr/bin/env python3
"""self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
native-ELF emitter.

The self-host native-ELF emit (`selftest_native_elf_h` in
examples/mindc_mind/main.mind) supports a low-level scalar + control-flow
subset (i64 scalars, calls, if/else/while/let/assign/break/continue, and the
parse-time desugars match / && / || / for / else-if, plus declared-order
all-i64 structs via accessor fns — the idiom main.mind is written in — plus
fixed i64 arrays: `[e0, e1, ...]` literals, index read/write `a[i]`, composing
in arithmetic / call args / returns / loops, with proven-constant-OOB indexes,
non-i64 elements, `[]`, trailing commas and `.len()` refused — plus i64 TUPLES
as anonymous positional structs: `(e0, e1, ...)` literals (comma-disambiguated
from paren groups), `.N` slot reads, and `let (a, b) = <tuple literal>;`
destructuring, with non-i64 elements, 1-tuples, out-of-arity `.N` and
non-literal-RHS destructures refused). For ANY
construct outside that subset it must FAIL CLOSED: emit 0 bytes (refuse),
NEVER a running ELF with a wrong value. An adversarial sweep once found ~123
constructs that silently miscompiled; a poison mechanism (lexer
`tk_unsupported`, parser `ast_unsupported`, `nb_expr`/`nb_stmt` default guard,
`nb_field_offset` -1, sticky `nb_poison_merge`, top-level 0B) closed them.

This smoke locks that boundary: unsupported shapes MUST refuse (0B), supported
shapes MUST run correct, and an unsupported node nested anywhere in an
otherwise-supported program MUST make the WHOLE unit refuse (sticky poison).

Usage: MINDC_SO=<built .so> python3 examples/mindc_mind/self_host_failclosed_smoke.py
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


# (label, source, expected)  — expected "0B" means MUST refuse.
REFUSED = [
    # tuples are SUPPORTED (i64 anonymous-positional-struct subset, below) —
    # but the tuple boundary itself stays fail-closed: non-i64 elements, the
    # 1-tuple/trailing-comma spelling, `.N` out of the literal's arity, and a
    # destructure whose RHS is not a DIRECT tuple literal of matching arity
    # (a call/ident/array RHS has no statically-known arity — refusing beats
    # a possible OOB read).
    ("tuple float element", M("return (1.5, 2.5).0;"), "0B"),
    ("tuple 1-tuple (1,)", M("let t=(1,); return 0;"), "0B"),
    ("tuple .N out of arity", M("return (1,2).5;"), "0B"),
    ("tuple .N OOB via let", M("let t=(1,2); return t.2;"), "0B"),
    ("tuple destructure arity mismatch", M("let (a,b,c)=(1,2); return a;"), "0B"),
    ("tuple destructure ident RHS", M("let x=(1,2); let (a,b)=x; return a;"), "0B"),
    ("tuple destructure call RHS", "fn mk()->i64{ return (4,5); }\nfn main()->i64{ let (a,b)=mk(); return a; }", "0B"),
    ("tuple destructure array RHS", M("let (a,b)=[1,2]; return a;"), "0B"),
    ("tuple pattern mut", M("let (mut a, b)=(1,2); return a;"), "0B"),
    ("tuple pattern literal", M("let (a, 1)=(1,2); return a;"), "0B"),
    ("compound assign +=", M("let mut c:i64=0; c += 1; return c;"), "0B"),
    # arrays are SUPPORTED (i64 subset, below) — but the array boundary itself
    # stays fail-closed: non-i64 elements, empty literal, trailing comma,
    # proven-constant OOB indexes, .len(), field access over an array binding.
    ("array float element", M("let a=[1.5, 2.5]; return 0;"), "0B"),
    ("array empty []", M("let a=[]; return 0;"), "0B"),
    ("array trailing comma", M("let a=[1,2,]; return a[0];"), "0B"),
    ("array const OOB read", M("let a=[1,2,3]; return a[3];"), "0B"),
    ("array const OOB write", M("let mut a=[1,2,3]; a[5]=1; return 0;"), "0B"),
    ("array negative index", M("let a=[1,2,3]; return a[-1];"), "0B"),
    ("array direct-lit OOB", M("return [1,2,3][7];"), "0B"),
    ("array .len()", M("let a=[1,2,3]; return a.len();"), "0B"),
    ("array field access", M("let a=[1,2,3]; return a.x;"), "0B"),
    ("array missing ]", M("let a=[1,2; return 0;"), "0B"),
    # struct literals must be EXACTLY the declared field set: under the
    # declared-order layout a missing/unknown/duplicate field would leave a
    # slot unwritten or doubly written — a wrong-value ELF — so refuse.
    ("struct-lit missing field",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1 }; return p.x; }", "0B"),
    ("struct-lit extra field",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1, y: 2, z: 3 }; return p.x; }", "0B"),
    ("struct-lit duplicate field",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1, x: 2 }; return p.x; }", "0B"),
    ("struct-lit unknown field",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1, z: 2 }; return p.x; }", "0B"),
    # FLOATS: the general path has no float tier (that lives behind the
    # dedicated selftest_native_elf_fp_* entries) — every float-bearing shape
    # (float literal, f64/f32 annotation on let/param/return, as-cast) must
    # refuse 0B. Confirmed fail-OPEN before the pre-gate: "used float cast+add"
    # RAN to 5 (want 7, the f64 read back as 0) and "float in int return" RAN
    # to 0 — running wrong-value ELFs, the exact forbidden class.
    ("used float cast+add", M("let f:f64=2.0; return (f as i64) + 5;"), "0B"),
    ("float in int return", M("return 1.5;"), "0B"),
    ("float arith chain", M("let f:f64=1.5; let g:f64=f+2.0; return (g as i64);"), "0B"),
    ("float param", "fn add1(x:f64)->f64{ return x+1.0; }\nfn main()->i64{ let r:f64=add1(2.5); return 0; }", "0B"),
    ("float return type", "fn mk()->f64{ return 1.5; }\nfn main()->i64{ let r:f64=mk(); return 0; }", "0B"),
    ("as f64 cast", M("let x:i64=3; let f:f64=x as f64; return 0;"), "0B"),
    ("float let unused", M("let f:f64=2.0; return 5;"), "0B"),
    ("f32 let", M("let f:f32=2.0; return 5;"), "0B"),
    ("f64 struct field", "struct F { a: f64 }\nfn main()->i64{ return 0; }", "0B"),
    ("float range bound", M("let mut c:i64=0; for i in 0..1.5 { c=c+1; } return c;"), "0B"),
    ("float in sep fn", "fn h()->i64{ let f:f64=1.5; return 0; }\nfn main()->i64{ return h(); }", "0B"),
    # sticky poison: unsupported nested in supported -> whole unit refuses
    # (`.len()` is still out-of-subset; i64 references LANDED — see the
    # SUPPORTED ref cases below and ref_netverify.py for their own boundary)
    (".len() nested in if", M("let a=[1,2,3]; if a[0]>0 { return a.len(); } return 0;"), "0B"),
    # unsupported node in a separate function still refuses the whole unit
    ("unsupported in sep fn", "fn helper()->i64{ let a=[1,2,3]; return a.len(); }\nfn main()->i64{ return helper(); }", "0B"),
]

# (label, source, expected exit value) — the supported subset MUST run correct.
SUPPORTED = [
    ("scalar arith", M("return 2+3*4;"), 14),
    # i64 references (LANDED — read-through + deref; ref_netverify.py locks the
    # write-back battery and the remaining ref fail-closed boundary)
    ("ref + deref", M("let x:i64=5; let r=&x; return *r;"), 5),
    ("ref nested in if", M("let x:i64=5; if x>0 { let r=&x; return *r; } return 0;"), 5),
    # fixed i64 arrays (alloc + base+8*i load/store ABI)
    ("array literal index", M("return [1,2,3][1];"), 2),
    ("array let + index", M("let a=[10,20,30]; return a[2];"), 30),
    ("array variable index", M("let a=[10,20,30]; let i=2; return a[i];"), 30),
    ("array arith compose", M("let a=[10,20,30]; return a[0]+a[1];"), 30),
    ("array index write", M("let mut a=[1,2,3]; a[1]=9; return a[1];"), 9),
    ("array for-loop sum", M("let a=[10,20,30]; let mut s=0; for i in 0..3 { s=s+a[i]; } return s;"), 60),
    ("array while-loop sum", M("let a=[10,20,30]; let mut s=0; let mut i=0; while i<3 { s=s+a[i]; i=i+1; } return s;"), 60),
    ("array write in loop", M("let mut a=[0,0,0]; for i in 0..3 { a[i]=i*10; } return a[0]+a[1]+a[2];"), 30),
    ("array as call arg", "fn get(p:i64, i:i64)->i64{ return p[i]; }\nfn main()->i64{ let a=[7,8,9]; return get(a, 1); }", 8),
    ("array as return value", "fn mk()->i64{ return [4,5,6]; }\nfn main()->i64{ let a=mk(); return a[2]; }", 6),
    ("array nested literal", M("let a=[[1,2],[3,4]]; return a[1][0];"), 3),
    ("array in while body", M("let mut c:i64=0; while c<3 { let a=[1,2]; c=c+a[0]; } return c;"), 3),
    ("array in sep fn", "fn helper()->i64{ let a=[1,2,3]; return a[0]; }\nfn main()->i64{ return helper(); }", 1),
    # i64 TUPLES (anonymous positional structs — the array alloc + base+8*i ABI
    # with `.N` slot reads and literal-RHS destructuring; parse-time desugar,
    # zero new emit surface)
    ("tuple literal .1", M("return (1,2).1;"), 2),
    ("tuple 3-elem .2", M("return (10,20,30).2;"), 30),
    ("tuple destructure", M("let (a,b)=(1,2); return a+b;"), 3),
    ("tuple 3-destructure", M("let (a,b,c)=(7,8,9); return a+b*c;"), 79),
    ("tuple destructure swap", M("let x=1; let y=2; let (a,b)=(y,x); return a*10+b;"), 21),
    ("tuple let + arith", M("let t=(3,4); return t.0+t.1;"), 7),
    ("tuple as call arg", "fn snd(p:i64)->i64{ return p.1; }\nfn main()->i64{ let t=(7,8); return snd(t); }", 8),
    ("tuple as return value", "fn mk()->i64{ return (4,5); }\nfn main()->i64{ let t=mk(); return t.1; }", 5),
    ("tuple nested .1.0", M("return ((1,2),(3,4)).1.0;"), 3),
    ("tuple nested via let", M("let t=((1,2),(3,4)); return t.0.1;"), 2),
    ("tuple computed elems", M("let x=5; return (x+1, x*2).1;"), 10),
    ("tuple computed destructure", M("let x=5; let (a,b)=(x+1,x*2); return a+b;"), 16),
    ("tuple in if branch", M("let x:i64=5; if x>0 { return (7,8).0; } return 0;"), 7),
    # paren-grouping is NOT perturbed by the tuple comma-disambiguation
    ("paren grouping (1+2)*3", M("return (1+2)*3;"), 9),
    ("paren double ((1))", M("return ((1));"), 1),
    ("if/else", M("let x:i64=5; if x>3 { return 3; } else { return 8; }"), 3),
    ("while carry", M("let mut c:i64=0; let mut i:i64=0; while i<10 { c=c+1; i=i+1; } return c;"), 10),
    ("match 3-arm", "fn main()->i64{ match 3 { 1=>{return 1;} 2=>{return 2;} _=>{return 9;} } }", 9),
    ("&& short-circuit", M("if 1>0 && 1>0 { return 1; } return 0;"), 1),
    ("|| short-circuit", M("if 0>1 || 1>0 { return 1; } return 0;"), 1),
    ("range for", M("let mut c:i64=0; for i in 0..5 { c=c+1; } return c;"), 5),
    ("untyped let", M("let mut i=0; let mut c:i64=0; while i<4 { c=c+1; i=i+1; } return c;"), 4),
    ("else-if chain", M("let x:i64=2; if x==1 {return 1;} else if x==2 {return 2;} else {return 3;}"), 2),
    ("recursion", "fn fib(n:i64)->i64{ if n<2 { return n; } return fib(n-1)+fib(n-2); }\nfn main()->i64{ return fib(7); }", 13),
    # OUT-OF-DECLARED-ORDER struct literals: the physical layout is DECLARED
    # order (mirrors the Rust oracle's canonical reordering), so `P { y:2, x:1 }`
    # reads back p.x==1 / p.y==2 — including through the accessor-fn idiom,
    # where the caller's literal layout and the param-typed srt decl-descriptor
    # read MUST agree (the silent miscompile this battery locks against).
    ("struct ooo p.x",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { y: 2, x: 1 }; return p.x; }", 1),
    ("struct ooo p.y",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { y: 2, x: 1 }; return p.y; }", 2),
    ("struct in-order p.y",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1, y: 2 }; return p.y; }", 2),
    ("struct 3-field ooo q.a",
     "struct Q { a: i64, b: i64, c: i64 }\nfn main()->i64{ let q: Q = Q { c: 3, a: 1, b: 2 }; return q.a; }", 1),
    ("struct 3-field ooo q.b",
     "struct Q { a: i64, b: i64, c: i64 }\nfn main()->i64{ let q: Q = Q { c: 3, a: 1, b: 2 }; return q.b; }", 2),
    ("struct 3-field ooo q.c",
     "struct Q { a: i64, b: i64, c: i64 }\nfn main()->i64{ let q: Q = Q { c: 3, a: 1, b: 2 }; return q.c; }", 3),
    ("struct accessor-fn in-order",
     "struct R { u: i64, v: i64 }\nfn get_u(r: R)->i64{ return r.u; }\nfn main()->i64{ let r: R = R { u: 7, v: 9 }; return get_u(r); }", 7),
    ("struct accessor-fn ooo lit",
     "struct R { u: i64, v: i64 }\nfn get_u(r: R)->i64{ return r.u; }\nfn main()->i64{ let r: R = R { v: 9, u: 7 }; return get_u(r); }", 7),
]


def main() -> int:
    fails = 0
    print("== REFUSED (unsupported constructs MUST fail-closed to 0B) ==")
    for lbl, src, _ in REFUSED:
        st, rc = run(src)
        ok = st == "0B"
        fails += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {'0B refused' if ok else f'LEAK ({st},{rc})'}")
    print("== SUPPORTED (subset MUST run correct, no over-refusal) ==")
    for lbl, src, want in SUPPORTED:
        st, rc = run(src)
        ok = st == "OK" and rc == (want & 0xFF)
        fails += 0 if ok else 1
        detail = f"exit {rc}" if st == "OK" else st
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {detail} want {want}")
    if fails:
        print(f"FAIL: {fails} fail-closed boundary violation(s)")
        return 1
    print("ALL PASS  (fail-closed: unsupported refuses 0B, sticky through nesting; supported runs correct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
