#!/usr/bin/env python3
# Canonical independent net-verify harness for CLOSURES / FN-VALUES / UNRESOLVED
# CALLEES in the native-ELF backend.
# Run: MINDC_SO=/path/to/mindc_self.so python3 closure_netverify.py
#
# First-class functions do not exist in the language yet (no lambda syntax; the
# `fn(..)->..` type is an extern-"C" callback declaration only; the reference
# compiler rejects fn-value calls with E2012). Every closure/fn-value shape MUST
# therefore refuse (0B, fail-closed) — and so must every call whose callee does
# not resolve to a defined fn (pre-gate those were fail-OPEN: nb_patch_all
# patched a garbage rel32 -> SIGILL, or worse a RUNNING ELF with a silently
# wrong value). The OK cases prove the nb_calls_all_resolve gate does not
# over-refuse legitimate direct calls.
#
# Every case is (label, source, want_rc, expect): "OK" = must run and return
# want_rc; "0B" = must refuse (zero bytes). A wrong rc or an emitted ELF on a
# 0B case is a fail-OPEN miscompile and MUST be reported FAIL, never PASS.
import ctypes, pathlib, subprocess, tempfile, sys, os

so = os.environ.get("MINDC_SO")
if not so:
    print("set MINDC_SO"); sys.exit(2)
lib = ctypes.CDLL(so)
lib.selftest_native_elf_h.restype = ctypes.c_int64
lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3
rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]

def run(s):
    b = ctypes.create_string_buffer(s.encode(), len(s.encode()))
    h = ctypes.create_string_buffer(bytes(32), 32)
    es = lib.selftest_native_elf_h(ctypes.cast(b, ctypes.c_void_p).value,
                                   len(s.encode()),
                                   ctypes.cast(h, ctypes.c_void_p).value)
    sh = rd(es, 0); ln = rd(sh, 8)
    e = ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""
    if not e:
        return ("0B", None)
    p = pathlib.Path(tempfile.mktemp()); p.write_bytes(e); p.chmod(0o755)
    try:
        return ("OK", subprocess.run([str(p)], timeout=8).returncode)
    except Exception as ex:
        return ("ERR", str(ex))

CASES = [
    # --- fn-value / closure shapes: no language surface -> MUST refuse ---
    ("fn-as-value let, then called",
     "fn add1(x:i64)->i64{ return x+1; }\nfn main()->i64{ let f = add1; return f(41); }", 0, "0B"),
    ("fn-as-value let (typed i64), then called",
     "fn add1(x:i64)->i64{ return x+1; }\nfn main()->i64{ let f:i64 = add1; return f(41); }", 0, "0B"),
    ("lambda syntax |x| x+1",
     "fn main()->i64{ let f = |x:i64| x+1; return f(41); }", 0, "0B"),
    ("fn-ptr type annotation outside extern",
     "fn add1(x:i64)->i64{ return x+1; }\nfn main()->i64{ let f: fn(i64)->i64 = add1; return f(41); }", 0, "0B"),
    ("fn name returned as a value",
     "fn add1(x:i64)->i64{ return x+1; }\nfn main()->i64{ return add1; }", 0, "0B"),
    ("higher-order: fn name passed as arg, called via param",
     "fn add1(x:i64)->i64{ return x+1; }\nfn app(g:i64,v:i64)->i64{ return g(v); }\nfn main()->i64{ return app(add1, 41); }", 0, "0B"),
    ("capturing-closure shape (lambda over outer local)",
     "fn main()->i64{ let y:i64 = 5; let f = |x:i64| x+y; return f(1); }", 0, "0B"),
    # --- unresolved-callee shapes: pre-gate fail-OPEN (garbage rel32) -> MUST refuse ---
    ("call through let-bound int value",
     "fn main()->i64{ let f:i64 = 5; return f(41); }", 0, "0B"),
    ("bare undefined callee in main",
     "fn main()->i64{ return g(41); }", 0, "0B"),
    ("misspelled recursion target",
     "fn fib(n:i64)->i64{ if n < 2 { return n; } return fibb(n-1); }\nfn main()->i64{ return fib(5); }", 0, "0B"),
    ("param name called as a fn",
     "fn app(g:i64,v:i64)->i64{ return g(v); }\nfn main()->i64{ return app(3, 41); }", 0, "0B"),
    ("struct name called as a fn",
     "struct S{x:i64}\nfn main()->i64{ return S(1); }", 0, "0B"),
    ("undefined callee in a non-main fn (was a RUNNING wrong-value ELF)",
     "fn h()->i64{ return nope(1); }\nfn main()->i64{ return h(); }", 0, "0B"),
    ("undefined callee only on one branch",
     "fn main()->i64{ let x:i64 = 1; if x == 2 { return zap(3); } return 7; }", 0, "0B"),
    # --- positive regression: the resolve gate must NOT over-refuse ---
    ("direct call still works",
     "fn add1(x:i64)->i64{ return x+1; }\nfn main()->i64{ return add1(41); }", 42, "OK"),
    ("forward-reference call (callee defined after caller)",
     "fn main()->i64{ return later(6); }\nfn later(x:i64)->i64{ return x*7; }", 42, "OK"),
    ("recursion still works",
     "fn fib(n:i64)->i64{ if n < 2 { return n; } return fib(n-1) + fib(n-2); }\nfn main()->i64{ return fib(10); }", 55, "OK"),
    ("call chain a->b->c still works",
     "fn c(x:i64)->i64{ return x+2; }\nfn b(x:i64)->i64{ return c(x)+3; }\nfn a(x:i64)->i64{ return b(x)+4; }\nfn main()->i64{ return a(33); }", 42, "OK"),
    ("7-arg call (stack-args path) still works",
     "fn s7(a:i64,b:i64,c:i64,d:i64,e:i64,f:i64,g:i64)->i64{ return a+b+c+d+e+f+g; }\nfn main()->i64{ return s7(1,2,3,4,5,6,21); }", 42, "OK"),
    ("call-free module still emits",
     "fn main()->i64{ return 42; }", 42, "OK"),
]

fails = 0
for label, src, want_rc, expect in CASES:
    st, rc = run(src)
    if expect == "0B":
        ok = st == "0B"
        got = "0B" if ok else "%s rc=%s" % (st, rc)
    else:
        ok = st == "OK" and rc == want_rc
        got = "%s rc=%s" % (st, rc)
    if not ok:
        fails += 1
    print("%-4s %-62s want=%-8s got=%s" % ("PASS" if ok else "FAIL", label,
                                           expect if expect == "0B" else "rc=%d" % want_rc, got))
print("closure_netverify: %d/%d PASS" % (len(CASES) - fails, len(CASES)))
sys.exit(1 if fails else 0)
