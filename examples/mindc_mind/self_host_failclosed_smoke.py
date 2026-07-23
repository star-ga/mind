#!/usr/bin/env python3
"""self_host_failclosed_smoke.py — the fail-closed boundary of the pure-MIND
native-ELF emitter.

The self-host native-ELF emit (`selftest_native_elf_h` in
examples/mindc_mind/main.mind) supports a low-level scalar + control-flow
subset (i64 scalars, calls, if/else/while/let/assign/break/continue, and the
parse-time desugars match / && / || / for / else-if, plus declared-order
all-i64 structs via accessor fns — the idiom main.mind is written in). For ANY
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
    ("array literal index", M("return [1,2,3][1];"), "0B"),
    ("array let + index", M("let a=[10,20,30]; return a[2];"), "0B"),
    ("tuple field", M("return (1,2).1;"), "0B"),
    ("tuple destructure", M("let (a,b)=(1,2); return b;"), "0B"),
    ("ref + deref", M("let x:i64=5; let r=&x; return *r;"), "0B"),
    ("compound assign +=", M("let mut c:i64=0; c += 1; return c;"), "0B"),
    # sticky poison: unsupported nested in supported -> whole unit refuses
    ("array nested in while", M("let mut c:i64=0; while c<3 { let a=[1,2]; c=c+1; } return c;"), "0B"),
    ("tuple nested in if", M("let x:i64=5; if x>0 { return (7,8).0; } return 0;"), "0B"),
    # unsupported node in a separate function still refuses the whole unit
    ("unsupported in sep fn", "fn helper()->i64{ let a=[1,2,3]; return a[0]; }\nfn main()->i64{ return helper(); }", "0B"),
]

# (label, source, expected exit value) — the supported subset MUST run correct.
SUPPORTED = [
    ("scalar arith", M("return 2+3*4;"), 14),
    ("if/else", M("let x:i64=5; if x>3 { return 3; } else { return 8; }"), 3),
    ("while carry", M("let mut c:i64=0; let mut i:i64=0; while i<10 { c=c+1; i=i+1; } return c;"), 10),
    ("match 3-arm", "fn main()->i64{ match 3 { 1=>{return 1;} 2=>{return 2;} _=>{return 9;} } }", 9),
    ("&& short-circuit", M("if 1>0 && 1>0 { return 1; } return 0;"), 1),
    ("|| short-circuit", M("if 0>1 || 1>0 { return 1; } return 0;"), 1),
    ("range for", M("let mut c:i64=0; for i in 0..5 { c=c+1; } return c;"), 5),
    ("untyped let", M("let mut i=0; let mut c:i64=0; while i<4 { c=c+1; i=i+1; } return c;"), 4),
    ("else-if chain", M("let x:i64=2; if x==1 {return 1;} else if x==2 {return 2;} else {return 3;}"), 2),
    ("recursion", "fn fib(n:i64)->i64{ if n<2 { return n; } return fib(n-1)+fib(n-2); }\nfn main()->i64{ return fib(7); }", 13),
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
