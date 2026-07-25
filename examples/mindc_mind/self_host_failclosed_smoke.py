#!/usr/bin/env python3
"""self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
native-ELF emitter.

The self-host native-ELF emit (`selftest_native_elf_h` in
examples/mindc_mind/main.mind) supports a low-level scalar + control-flow
subset (i64 scalars, calls, if/else/while/let/assign/break/continue, and the
parse-time desugars match / && / || / for / else-if, plus declared-order
all-i64 structs via accessor fns — the idiom main.mind is written in — plus
fixed i64 AND f64 arrays: `[e0, e1, ...]` literals, index read/write `a[i]`
(an f64-element array read routes the SSE2 arm so `a[i]+a[j]` sums via addsd),
the zero-arg `.len()` method (compile-time-constant element count), composing in
arithmetic / call args / returns / loops, with proven-constant-OOB indexes,
`[]`, trailing commas refused — the f64 tier is ARRAY-ONLY (a float element in a
TUPLE still refuses; tuples stay i64-only) and NON-f64 float/mixed element types
(f32, mixed `[1.0, 2]`) plus narrow-int-annotated element arrays (i32/i8) and
TYPE-annotated `[bool; N]` arrays refuse (0B) — an UNTYPED `[true, false]`
folds to a plain i64 array [1, 0] and RUNS (bool literals `true`/`false` fold
to synthetic i64 int-lits 1/0), and every NON-`len` method /
`.len()` with args / `.len()` on a non-array or unbound receiver refused —
plus i64 TUPLES
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
    # compound-assign `NAME OP= RHS` is SUPPORTED for arithmetic ops on a plain
    # ident LHS (parse-time desugar `lhs OP= rhs` -> `lhs = lhs OP rhs`, see the
    # SUPPORTED cases below). The BITWISE/SHIFT compound forms (`&= |= ^= <<= >>=`)
    # stay out of subset -> fail-closed; an array/field compound target is out of
    # the ident-only scope -> fail-closed (never a double-evaluated miscompile);
    # a MALFORMED RHS poisons.
    ("compound bitand-assign &=", M("let mut c:i64=3; c &= 1; return c;"), "0B"),
    ("compound bitor-assign |=", M("let mut c:i64=1; c |= 2; return c;"), "0B"),
    ("compound shl-assign <<=", M("let mut c:i64=1; c <<= 2; return c;"), "0B"),
    ("compound array-elem += (out of scope)", M("let mut a=[1,2,3]; a[0] += 4; return a[0];"), "0B"),
    ("compound field += (out of scope)",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let mut p: P = P { x: 1, y: 2 }; p.x += 4; return p.x; }", "0B"),
    ("compound malformed empty RHS", M("let mut c:i64=0; c += ; return c;"), "0B"),
    # arrays are SUPPORTED (i64 AND f64 subset, below — incl. zero-arg `.len()`) —
    # but the array boundary itself stays fail-closed: NON-f64 float element types
    # and MIXED-dtype arrays, empty literal, trailing comma, proven-constant OOB
    # indexes, field access over an array binding.
    # The f64 element tier is ARRAY-ONLY: a homogeneous-float TUPLE shares the
    # ast_array_lit node (child0==1 marker) but tuples are i64-only, so a float
    # tuple element STILL refuses (regression guard — see `tuple float element`
    # above). MIXED arrays refuse via the homogeneity gate (element dtype !=
    # element 0's). f32 refuses via the module-wide nb_user_float_tok pre-gate
    # (the general float tier computes in scalar-double, so a declared-f32 array
    # would f64-round — a silent miscompile). Narrow-int-annotated (i32/i8)
    # element arrays are out of the supported element-type subset. An UNTYPED
    # bool-element array (`[true, false, ...]`) is NOT refused: since
    # `true`/`false` fold to synthetic i64 int-lits (1/0), it is an ordinary
    # i64 array that RUNS correct (indexing gives 1/0) — SUPPORTED below. Only a
    # TYPE-annotated `[bool; N]` array still refuses (the typed-array annotation
    # scopes to i64/f64; bool-typed stays fail-closed) — see the case just below.
    ("array f32 elements", M("let a: [f32; 3] = [1.0, 2.0, 3.0]; return a[0] as i64;"), "0B"),
    ("array i32 elements", M("let a: [i32; 3] = [1, 2, 3]; return a[0];"), "0B"),
    ("array i8 elements", M("let a: [i8; 3] = [1, 2, 3]; return a[0];"), "0B"),
    ("typed bool array [bool;N]", M("let a: [bool; 2] = [true, false]; return 0;"), "0B"),
    ("array mixed float-then-int", M("let a=[1.0, 2]; return a.len();"), "0B"),
    ("array mixed int-then-float", M("let a=[1, 2.0]; return a.len();"), "0B"),
    # NESTED-FLOAT array index — the fail-OPEN class closed by nb_index_leaf_dtype.
    # nb_index_elem_dtype classifies only a DIRECT ident/array-lit base and returns
    # INT for a nested ast_index base, so a nested-FLOAT read `a[1][0]` used to
    # GP-load the raw IEEE-754 bits and `as i64` no-op them to 0 — a RUNNING ELF
    # returning the WRONG value (want 3, got 0). The nested base is now poisoned
    # (float or unprovable leaf dtype) -> 0B. Nested-i64 (`[[1,2],[3,4]]; a[1][0]`)
    # PROVES INT and keeps running (SUPPORTED, below). Both READ and WRITE arms,
    # const AND variable index, and a nested-FLOAT cell written with an INT rhs
    # (which would corrupt the float cell) are covered. deferred: correct
    # nested-FLOAT support (movsd the leaf through the row pointer) — a feature.
    ("nested-float read a[1][0]", M("let a=[[1.0,2.0],[3.0,4.0]]; return a[1][0] as i64;"), "0B"),
    ("nested-float read var idx", M("let a=[[1.0,2.0],[3.0,4.0]]; let i=1; let j=0; return a[i][j] as i64;"), "0B"),
    ("nested-float write float rhs", M("let mut a=[[1.0,2.0],[3.0,4.0]]; a[1][0]=9.0; return a[1][0] as i64;"), "0B"),
    ("nested-float write int rhs", M("let mut a=[[1.0,2.0],[3.0,4.0]]; a[1][0]=9; return a[1][0] as i64;"), "0B"),
    ("array empty []", M("let a=[]; return 0;"), "0B"),
    ("array trailing comma", M("let a=[1,2,]; return a[0];"), "0B"),
    ("array const OOB read", M("let a=[1,2,3]; return a[3];"), "0B"),
    ("array const OOB write", M("let mut a=[1,2,3]; a[5]=1; return 0;"), "0B"),
    ("array negative index", M("let a=[1,2,3]; return a[-1];"), "0B"),
    ("array direct-lit OOB", M("return [1,2,3][7];"), "0B"),
    ("array field access", M("let a=[1,2,3]; return a.x;"), "0B"),
    ("array missing ]", M("let a=[1,2; return 0;"), "0B"),
    # TYPED-ARRAY annotation `let NAME : [T; N] = INIT` (parse-past the balanced
    # annotation, then the existing i64/f64 array-lit emit path runs). A MALFORMED
    # annotation (no matching `]`) MUST still fail-close (0B), never run — the
    # balanced scan (let_arr_ann_after) hits EOF and poisons.
    ("typed-array malformed no ]", M("let a:[i64; = [1,2,3]; return a[0];"), "0B"),
    ("typed-array malformed close paren", M("let a:[i64; 3) = [1,2,3]; return a[0];"), "0B"),
    # METHOD-CALL over-fire guard (B0): ONLY zero-arg `.len()` on a fixed-size
    # array receiver lowers. Every OTHER method-call shape MUST refuse (0B) —
    # never a wrong running ELF. These previously ALL refused via the parser's
    # non-method fall-through; now the parser builds a real ast_method node and
    # nb_expr's arm must still refuse each.
    (".len() with arg", M("let a=[1,2,3]; return a.len(1);"), "0B"),
    ("unknown method .foo() on array", M("let a=[1,2,3]; return a.foo();"), "0B"),
    (".push(4) on array", M("let mut a=[1,2,3]; return a.push(4);"), "0B"),
    (".len() on int scalar", M("let x:i64=5; return x.len();"), "0B"),
    (".len() on unbound name", M("return z.len();"), "0B"),
    (".len() on int literal", M("return 5.len();"), "0B"),
    (".len() on struct receiver",
     "struct P { x: i64, y: i64 }\nfn main()->i64{ let p: P = P { x: 1, y: 2 }; return p.len(); }", "0B"),
    (".len() on array reassigned-from-call",
     "fn mk()->i64{ return [4,5,6]; }\nfn main()->i64{ let a=mk(); return a.len(); }", "0B"),
    # a TUPLE shares the ast_array_lit node (child0==1 marker) but is NOT an
    # array — `.len()` is array-only this slice, so a tuple receiver refuses.
    (".len() on tuple literal", M("return (1,2,3).len();"), "0B"),
    (".len() on tuple via let", M("let t=(1,2,3); return t.len();"), "0B"),
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
    # DIRECT field access on a CALL receiver `mk().field` (#257 item 7): the
    # struct-returning-call case is SUPPORTED (below), but the fail-closed
    # boundary stays sharp — a call whose callee returns a NON-struct, a struct-
    # returning call to an UNKNOWN field, and a CHAINED `mk().x.y` (a non-call
    # receiver on the outer field) all refuse (0B), never a running wrong-value ELF.
    # (There is no call-receiver field-WRITE to probe: a statement opening
    # `mk() . x = 3` never enters the field-assign parser — that fires only for an
    # `IDENT .` receiver — so `=` parses as an infix comparison `mk().x == 3`, a
    # supported discarded expression, exactly as the reference frontend accepts.)
    ("call field on non-struct return",
     "fn mk()->i64{ return 5; }\nfn main()->i64{ return mk().x; }", "0B"),
    ("call field unknown field mk().z",
     "struct P { x: i64, y: i64 }\nfn mk()->P{ P{x:7,y:8} }\nfn main()->i64{ return mk().z; }", "0B"),
    ("call field chained mk().x.y (defer)",
     "struct P { x: i64, y: i64 }\nfn mk()->P{ P{x:7,y:8} }\nfn main()->i64{ return mk().x.y; }", "0B"),
    # FLOATS (B0 gate LIFTED for f64): the general path now lowers the f64
    # tier soundly (see general_float_netverify.py for the value battery —
    # float lets/arith/compares/params/calls, saturating `f as i64`, entry
    # trunc-to-exit). The REMAINING float refusals are locked here: f32 (the
    # general tier computes in scalar-double — f64-rounding a declared-f32
    # program would be a silent miscompile), `as` to a non-integer target
    # (an explicit parser refusal — the old fall-through was a coercion-shaped
    # leak), MIXED float/int binops (the pre-gate fail-OPEN leak class: raw
    # IEEE-754 bits flowed into the GP integer path), and float conditions.
    ("as f64 cast", M("let x:i64=3; let f:f64=x as f64; return 0;"), "0B"),
    ("f32 let", M("let f:f32=2.0; return 5;"), "0B"),
    ("float range bound", M("let mut c:i64=0; for i in 0..1.5 { c=c+1; } return c;"), "0B"),
    ("mixed float+int binop", M("let f:f64=2.0; return f + 5;"), "0B"),
    ("float % float", M("let f:f64=5.0; let g:f64=f % 2.0; return 0;"), "0B"),
    ("narrowing cast of float", M("let f:f64=2.5; return f as i8;"), "0B"),
    ("float condition", M("let f:f64=1.5; if f { return 1; } return 2;"), "0B"),
    ("non-dyadic float literal", M("let f:f64=0.1; return (f as i64);"), "0B"),
    # sticky poison: unsupported nested in supported -> whole unit refuses
    # (an unknown method `.foo()` is still out-of-subset; zero-arg `.len()` on an
    # array LANDED — see the SUPPORTED cases below)
    (".foo() nested in if", M("let a=[1,2,3]; if a[0]>0 { return a.foo(); } return 0;"), "0B"),
    # unsupported node in a separate function still refuses the whole unit
    ("unsupported method in sep fn", "fn helper()->i64{ let a=[1,2,3]; return a.foo(); }\nfn main()->i64{ return helper(); }", "0B"),
    # ITEM-LEVEL ATTRIBUTES: `#[...]` on a declaration is out of subset. The
    # parser's unrecognized-item-start fallback poisons (ast_unsupported) so the
    # stray `#`/`[`/ident/`]` tokens can NEVER silently drop and emit a running
    # ELF (B0 fail-OPEN). Previously `#[inline]\nfn main()->i64{7}` emitted a
    # 397B running ELF byte-identical to the un-attributed program.
    ("item attr #[inline]", "#[inline]\nfn main()->i64{7}", "0B"),
    ("item attr #[nonsense]", "#[nonsense]\nfn main()->i64{7}", "0B"),
]

# (label, source, expected exit value) — the supported subset MUST run correct.
SUPPORTED = [
    ("scalar arith", M("return 2+3*4;"), 14),
    # control: the SAME fn without the item-level attribute still emits + runs
    # (proves the poison is scoped to the unrecognized attribute token, not the
    # bare-tail-expr `fn main()->i64{7}` shape itself)
    ("attr-free control fn", "fn main()->i64{7}", 7),
    # i64 references (LANDED — read-through + deref; ref_netverify.py locks the
    # write-back battery and the remaining ref fail-closed boundary)
    ("ref + deref", M("let x:i64=5; let r=&x; return *r;"), 5),
    ("ref nested in if", M("let x:i64=5; if x>0 { let r=&x; return *r; } return 0;"), 5),
    # fixed i64 arrays (alloc + base+8*i load/store ABI)
    ("array literal index", M("return [1,2,3][1];"), 2),
    # TYPED-ARRAY annotation `let NAME : [i64; N] = INIT` — the parser now consumes
    # the balanced `[i64; N]` annotation (type ASSERTION only; the init `[..]`
    # literal carries the elements + length), so `=` is found at the right position
    # and the proven i64 array-lit emit path runs. The UN-annotated form already
    # worked; only the annotated form was refusing (0B) before this fix.
    ("typed-array i64 let index", M("let a:[i64;3]=[1,2,3]; return a[1];"), 2),
    ("typed-array i64 let sum", M("let a:[i64;2]=[10,20]; return a[0]+a[1];"), 30),
    ("typed-array i64 spaced annot", M("let a: [i64; 3] = [1, 2, 3]; return a[2];"), 3),
    ("typed-array i64 mut write", M("let mut a:[i64;3]=[1,2,3]; a[1]=9; return a[1];"), 9),
    # f64 element arrays are also value-safe under the ty=0 path (raw IEEE-754
    # cells; index read routes the SSE2 arm). f32 stays refused (above).
    ("typed-array f64 read", M("let a:[f64;2]=[1.0,2.0]; return a[0] as i64;"), 1),
    ("typed-array f64 sum", M("let a:[f64;3]=[1.0,2.0,4.0]; return (a[0]+a[1]+a[2]) as i64;"), 7),
    ("array let + index", M("let a=[10,20,30]; return a[2];"), 30),
    ("array variable index", M("let a=[10,20,30]; let i=2; return a[i];"), 30),
    # RUNTIME bounds guard: an IN-RANGE variable/computed index still returns the
    # correct value (the guard falls straight through — no perturbation of the
    # in-bounds path). The matching OUT-OF-RANGE cases are in TRAP below.
    ("array var index low edge (i=0)", M("let a=[10,20,30]; let i=0; return a[i];"), 10),
    ("array var index high edge (i=N-1)", M("let a=[10,20,30]; let i=2; return a[i];"), 30),
    ("array computed index in-range", M("let a=[10,20,30]; let i=1+1; return a[i];"), 30),
    ("array var write in-range", M("let mut a=[1,2,3]; let i=2; a[i]=9; return a[2];"), 9),
    ("array arith compose", M("let a=[10,20,30]; return a[0]+a[1];"), 30),
    ("array index write", M("let mut a=[1,2,3]; a[1]=9; return a[1];"), 9),
    ("array for-loop sum", M("let a=[10,20,30]; let mut s=0; for i in 0..3 { s=s+a[i]; } return s;"), 60),
    ("array while-loop sum", M("let a=[10,20,30]; let mut s=0; let mut i=0; while i<3 { s=s+a[i]; i=i+1; } return s;"), 60),
    ("array write in loop", M("let mut a=[0,0,0]; for i in 0..3 { a[i]=i*10; } return a[0]+a[1]+a[2];"), 30),
    ("array as call arg", "fn get(p:i64, i:i64)->i64{ return p[i]; }\nfn main()->i64{ let a=[7,8,9]; return get(a, 1); }", 8),
    ("array as return value", "fn mk()->i64{ return [4,5,6]; }\nfn main()->i64{ let a=mk(); return a[2]; }", 6),
    ("array nested literal", M("let a=[[1,2],[3,4]]; return a[1][0];"), 3),
    # nested-i64 index PROVES INT at every descent level (nb_index_leaf_dtype) so it
    # keeps running — the regression guard against over-refusing when the
    # nested-FLOAT fail-closed poison landed. WRITE and 3-deep READ both covered.
    ("array nested i64 write", M("let mut a=[[1,2],[3,4]]; a[1][0]=9; return a[1][0];"), 9),
    ("array nested i64 3-deep", M("let a=[[[1,2],[3,4]],[[5,6],[7,8]]]; return a[1][0][1];"), 6),
    ("array in while body", M("let mut c:i64=0; while c<3 { let a=[1,2]; c=c+a[0]; } return c;"), 3),
    ("array in sep fn", "fn helper()->i64{ let a=[1,2,3]; return a[0]; }\nfn main()->i64{ return helper(); }", 1),
    # zero-arg `.len()` on a fixed-size array — compile-time-constant element
    # count (arrays are fixed-size), lowered to a single i64 const.
    ("array .len() via let", M("let a=[1,2,3]; return a.len();"), 3),
    ("array .len() direct lit", M("return [10,20,30,40].len();"), 4),
    ("array .len() in arith", M("let a=[1,2,3,4]; return a.len()+1;"), 5),
    ("array .len() nested in if", M("let a=[1,2,3]; if a[0]>0 { return a.len(); } return 0;"), 3),
    ("array .len() 1-elem", M("let a=[7]; return a.len();"), 1),
    ("array .len() in sep fn", "fn helper()->i64{ let a=[5,6]; return a.len(); }\nfn main()->i64{ return helper(); }", 2),
    # bool literals `true`/`false` fold to synthetic i64 int-lits (true=1,
    # false=0) at parse time — matching bool's i64 backing, the comparison
    # path's 1/0, and mindc-Rust's `fmt` fold. They flow through the existing
    # i64 path in conditions, lets, returns, arithmetic, loop guards, and array
    # elements. NO new AST kind, NO new emit surface. (A TYPE-annotated
    # `[bool; N]` array still refuses — see REFUSED above.)
    ("bool let true cond", M("let b=true; if b {return 7;} return 0;"), 7),
    ("bool let false cond", M("let b=false; if b {return 1;} return 9;"), 9),
    ("bool return true", M("return true;"), 1),
    ("bool return false", M("return false;"), 0),
    ("bool arith true+true", M("return true + true;"), 2),
    ("bool arith false+true", M("return false + true;"), 1),
    ("bool while guard", M("let mut b=true; let mut n=0; while b { n=n+1; if n>2 { b=false; } } return n;"), 3),
    ("bool while true literal guard", M("let mut s=0; let mut i=0; while true { if i>3 { return s; } s=s+i; i=i+1; } return s;"), 6),
    # UNTYPED bool-element array folds to an ordinary i64 array [1,0,1] — RUNS
    # correct (len + index give 1/0). This is the reconciled ex-`array bool
    # elements` REFUSED fixture: running-correct 1/0 is honest, NOT a fail-OPEN.
    ("untyped bool array .len()", M("let a=[true, false, true]; return a.len();"), 3),
    ("untyped bool array index", M("let a=[true, false, true]; return a[0]+a[2];"), 2),
    # fixed f64-element arrays (the f64 element tier — #256): the array cell holds
    # the raw IEEE-754 bits (nb_expr's float-lit arm movsd-spills them; the 8-byte
    # GP element store copies them byte-exact). An index READ tags its result FLOAT
    # so a[i]+a[j] routes the SSE2 addsd arm; `.len()` is the compile-time count.
    # Correctness is execution-gated (native-ELF float has no byte-identity oracle).
    ("f64 array index read", M("let a=[10.0, 20.0, 30.0]; return a[2] as i64;"), 30),
    ("f64 array sum via addsd", M("let a=[1.0, 2.0, 4.0]; let s=a[0]+a[1]+a[2]; return s as i64;"), 7),
    ("f64 array .len()", M("let a=[1.0, 2.0, 3.0, 4.0, 5.0]; return a.len();"), 5),
    ("f64 array variable index", M("let a=[2.0, 8.0, 32.0]; let i=1; return a[i] as i64;"), 8),
    ("f64 array direct-lit index", M("return [3.0, 6.0, 9.0][1] as i64;"), 6),
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
    # compound-assign arithmetic desugar `lhs OP= rhs` -> `lhs = lhs OP rhs`
    # (parse-time; plain-ident LHS; zero new emit surface — the desugared
    # assign/binop arms run). All five arithmetic ops + chaining + loop body.
    ("compound += basic", M("let mut c:i64=0; c += 5; return c;"), 5),
    ("compound -= basic", M("let mut c:i64=10; c -= 2; return c;"), 8),
    ("compound *= basic", M("let mut c:i64=3; c *= 3; return c;"), 9),
    ("compound /= basic", M("let mut c:i64=20; c /= 4; return c;"), 5),
    ("compound %= basic", M("let mut c:i64=17; c %= 5; return c;"), 2),
    ("compound += chain", M("let mut c:i64=0; c += 5; c -= 2; c += 4; return c;"), 7),
    ("compound += rhs-expr", M("let mut c:i64=1; c += 2*3; return c;"), 7),
    ("compound += in for-loop", M("let mut s:i64=0; for i in 0..3 { s += i; } return s;"), 3),
    ("compound += in while-loop", M("let mut s:i64=0; let mut i:i64=0; while i<4 { s += i; i += 1; } return s;"), 6),
    ("compound *= in loop (factorial)", M("let mut p:i64=1; for i in 1..5 { p *= i; } return p;"), 24),
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
    # DIRECT field access on a CALL receiver `mk().field` (#257 item 7): the
    # `let p = mk(); p.field` form already worked (d0c40587 infers the struct
    # descriptor from the callee's `-> P` return type); this lands the DIRECT
    # form, resolving the field offset through the SAME decl-descriptor path and
    # loading base+offset*8 off the call's returned i64 struct handle. Field 0
    # (offset 0), a nonzero-offset field, and two calls composed in arithmetic.
    ("call field mk().x",
     "struct P { x: i64, y: i64 }\nfn mk()->P{ P{x:7,y:8} }\nfn main()->i64{ return mk().x; }", 7),
    ("call field mk().y",
     "struct P { x: i64, y: i64 }\nfn mk()->P{ P{x:7,y:8} }\nfn main()->i64{ return mk().y; }", 8),
    ("call field arith mk().x + mk().y",
     "struct P { x: i64, y: i64 }\nfn mk()->P{ P{x:7,y:8} }\nfn main()->i64{ return mk().x + mk().y; }", 15),
]

# (label, source, expected exit code) — a RUNTIME array-bounds VIOLATION on a
# DIRECT let-bound known-length array (#256 slice 5/5). The compile-time
# nb_index_oob refusal only catches CONSTANT indices; a VARIABLE/computed index
# used to do an UNCHECKED load/store past the array (a memory-safety hole). The
# runtime guard (nb_emit_bounds_guard) now halts DETERMINISTICALLY via
# _exit(nb_oob_trap_code) == _exit(77) — a RUNNING ELF that traps with a FIXED
# reproducible code, NEVER a wrong value and never an OOB access. Both READ
# (ast_index) and WRITE (ast_index_assign) arms are covered. Only the
# direct-binding known-length case; a call-boundary array (no runtime length)
# stays unchecked (deferred LARGE part) and is NOT probed here.
OOB_TRAP = 77
TRAP = [
    ("array var index OOB (i=N)", M("let a=[10,20,30]; let i=3; return a[i];")),
    ("array var index OOB (i=N+2)", M("let a=[10,20,30]; let i=5; return a[i];")),
    ("array var index OOB negative", M("let a=[10,20,30]; let i=0-1; return a[i];")),
    ("array computed index OOB (1+2==N)", M("let a=[10,20,30]; let i=1+2; return a[i];")),
    ("array var WRITE OOB (i=N+2)", M("let mut a=[1,2,3]; let i=5; a[i]=9; return 0;")),
    ("array var WRITE OOB negative", M("let mut a=[1,2,3]; let i=0-2; a[i]=9; return 0;")),
    ("array for-loop index OOB", M("let a=[10,20,30]; let mut s=0; for i in 0..4 { s=s+a[i]; } return s;")),
    ("array direct-lit var index OOB", M("let i=3; return [10,20,30][i];")),
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
    print("== RUNTIME OOB TRAP (variable/computed index MUST deterministically trap) ==")
    for lbl, src in TRAP:
        st, rc = run(src)
        # A running ELF (emit succeeded) that exits with the fixed trap code —
        # NOT a wrong value (any other rc) and NOT a compile-time refusal (0B).
        ok = st == "OK" and rc == OOB_TRAP
        fails += 0 if ok else 1
        detail = f"exit {rc}" if st == "OK" else st
        print(f"  {'PASS' if ok else 'FAIL'}  {lbl}: {detail} want trap-exit {OOB_TRAP}")
    if fails:
        print(f"FAIL: {fails} fail-closed boundary violation(s)")
        return 1
    print("ALL PASS  (fail-closed: unsupported refuses 0B, sticky through nesting; supported runs correct; runtime OOB traps deterministically)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
