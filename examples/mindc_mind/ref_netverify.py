#!/usr/bin/env python3
# Canonical independent net-verify harness for i64 references in the native-ELF backend.
# Run: MINDC_SO=/path/to/mindc_self.so python3 ref_netverify.py
# Every case is (label, source, want_rc, expect) where expect is "OK" (must run and
# return want_rc) or "0B" (must refuse — emit zero bytes / fail-closed).
# A store-through-ref-passed-into-a-callee that returns the stale value is a
# fail-OPEN silent miscompile and MUST be reported FAIL, never PASS.
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
    # --- pass-by-ref WRITE-BACK (the class the prior agent misreported) ---
    ("passby literal-store 1-param",
     "fn setr(r:i64){ *r=7; }\nfn main()->i64{ let mut x:i64=0; setr(&x); return x; }", 7, "OK"),
    ("passby var-store 2-param",
     "fn setr(r:i64,v:i64){ *r=v; }\nfn main()->i64{ let mut x:i64=0; setr(&x,7); return x; }", 7, "OK"),
    ("passby var-store swapped-args",
     "fn setr(v:i64,r:i64){ *r=v; }\nfn main()->i64{ let mut x:i64=0; setr(7,&x); return x; }", 7, "OK"),
    ("passby copy-through-two-refs",
     "fn cp(dst:i64,src:i64){ *dst=*src; }\nfn main()->i64{ let a:i64=9; let mut b:i64=0; cp(&b,&a); return b; }", 9, "OK"),
    # --- pass-by-ref READ (works today; must keep working) ---
    ("passby read-through 2-param",
     "fn get(r:i64,pad:i64)->i64{ return *r + pad; }\nfn main()->i64{ let x:i64=5; return get(&x,10); }", 15, "OK"),
    # --- within-function refs ---
    ("within-fn deref read",
     "let x:i64=5; let r=&x; return *r;", 5, "OK"),
    ("within-fn deref-write then read",
     "let mut x:i64=0; let r=&x; *r=9; return x;", 9, "OK"),
    ("within-fn read-before-and-after-write reloads",
     "let mut x:i64=1; let r=&x; let y:i64=x; *r=9; return x+y;", 10, "OK"),
    ("within-fn deref arith",
     "let x:i64=5; let r=&x; return *r + 1;", 6, "OK"),
    # --- binary ops MUST NOT be broken by &/* disambiguation ---
    ("bitand 6&3", "return 6 & 3;", 2, "OK"),
    ("mul 6*7", "return 6 * 7;", 42, "OK"),
    ("mixed (a&b)*c", "let a:i64=6; let b:i64=3; let c:i64=4; return (a & b) * c;", 8, "OK"),
    ("bitand-with-deref-rhs a&*r", "let x:i64=3; let r=&x; return 6 & *r;", 2, "OK"),
    # --- fail-closed: unsupported ref shapes MUST refuse 0B ---
    ("refuse &a[i]", "let a=[1,2,3]; let r=&a[1]; return *r;", None, "0B"),
    ("refuse &p.field", "struct P{x:i64}\nfn main()->i64{ let p=P{x:5}; let r=&p.x; return *r; }", None, "0B"),
    ("refuse &literal", "let r=&5; return *r;", None, "0B"),
    ("refuse ref-then-directly-assign",
     "let mut x:i64=1; let r=&x; x=5; return *r;", None, "0B"),
    ("refuse ref-of-narrow", "let x:i8=5; let r=&x; return *r;", None, "0B"),
    ("refuse &&x address-of-address-of", "let x:i64=5; let r=&&x; return **r;", None, "0B"),
    # --- CRITICAL (blind review): &local + deref-store in a recursive fn corrupts the
    #     frame (SIGSEGV / silent-wrong, frame-layout-dependent) -> MUST refuse 0B ---
    ("refuse recursive+ref+store non-tail",
     "fn f(n:i64)->i64{ if n==0 {return 0;} let mut m:i64=n; let lr=&m; *lr=n*10; let s:i64=f(n-1); return s+n; }\nfn main()->i64{ return f(1); }", None, "0B"),
    ("refuse recursive+ref+store tail",
     "fn f(n:i64)->i64{ if n==0 {return 0;} let mut m:i64=n; let lr=&m; *lr=n*10; return f(n-1); }\nfn main()->i64{ return f(3); }", None, "0B"),
    # --- CRITICAL (net-verify): mutual recursion re-enters the address-taken frame too;
    #     f->g->f with &local+store SIGSEGVs -> the recursion guard must be call-cycle-aware ---
    ("refuse mutual-recursion+ref+store 2-cycle",
     "fn g(n:i64)->i64{ if n==0 {return 0;} return f(n-1); }\nfn f(n:i64)->i64{ if n==0 {return 0;} let mut m:i64=n; let lr=&m; *lr=n*10; let s:i64=g(n-1); return s+n; }\nfn main()->i64{ return f(2); }", None, "0B"),
    ("refuse mutual-recursion+ref+store 3-cycle",
     "fn h(n:i64)->i64{ if n==0 {return 0;} return f(n-1); }\nfn g(n:i64)->i64{ if n==0 {return 0;} return h(n-1); }\nfn f(n:i64)->i64{ if n==0 {return 0;} let mut m:i64=n; let lr=&m; *lr=n; let s:i64=g(n-1); return s+n; }\nfn main()->i64{ return f(3); }", None, "0B"),
    # --- MEDIUM (blind review): returning &local escapes a dangling stack address -> refuse ---
    ("refuse return &local",
     "fn g()->i64{ let x:i64=5; return &x; }\nfn main()->i64{ return g(); }", None, "0B"),
]

def main():
    ok = True
    for lbl, s, w, exp in CASES:
        src = s if ("fn main" in s) else ("fn main()->i64{ %s }" % s)
        st, rc = run(src)
        if exp == "0B":
            good = (st == "0B")
        else:
            good = (st == "OK" and rc == w)
        if not good:
            ok = False
        tag = "PASS" if good else "FAIL"
        detail = f"({st},{rc})" + ("" if exp == "0B" else f" want {w}")
        print(f"  [{tag}] {lbl}: {detail}")
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)

main()
