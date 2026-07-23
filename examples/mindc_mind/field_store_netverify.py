#!/usr/bin/env python3
# Canonical independent value harness for struct field STORES (`p.x = v`) in the
# native-ELF backend. Run: MINDC_SO=/path/to/mindc_self.so python3 field_store_netverify.py
# Every case is (label, source, want_rc, expect) where expect is "OK" (must run and
# return want_rc) or "0B" (must refuse — emit zero bytes / fail-closed).
# A field store the next read does not observe is a fail-OPEN silent miscompile
# and MUST be reported FAIL, never PASS.
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

P1 = "struct P{x:i64}\n"
P2 = "struct P{x:i64,y:i64}\n"

CASES = [
    # --- straight-line store + read-back ---
    ("straight-line store field0",
     P1 + "fn main()->i64{ let p=P{x:1}; p.x=7; return p.x; }", 7, "OK"),
    ("straight-line store field1 (off>0)",
     P2 + "fn main()->i64{ let p=P{x:1,y:2}; p.y=9; return p.x+p.y; }", 10, "OK"),
    ("multi-field store both",
     P2 + "fn main()->i64{ let p=P{x:0,y:0}; p.x=3; p.y=4; return p.x*10+p.y; }", 34, "OK"),
    ("store rhs reads same field (p.x = p.x + 1)",
     P1 + "fn main()->i64{ let p=P{x:41}; p.x=p.x+1; return p.x; }", 42, "OK"),
    ("store rhs is a call",
     P1 + "fn f()->i64{ return 33; }\nfn main()->i64{ let p=P{x:0}; p.x=f(); return p.x; }", 33, "OK"),
    ("store rhs reads OTHER field",
     P2 + "fn main()->i64{ let p=P{x:5,y:0}; p.y=p.x*2; return p.y; }", 10, "OK"),
    # --- branches (memory store: visible across regions, no carry machinery) ---
    ("conditional store taken then read",
     P1 + "fn main()->i64{ let c:i64=1; let p=P{x:1}; if c==1 { p.x=5; } return p.x; }", 5, "OK"),
    ("conditional store NOT taken then read",
     P1 + "fn main()->i64{ let c:i64=0; let p=P{x:1}; if c==1 { p.x=5; } return p.x; }", 1, "OK"),
    ("both-branch store then read",
     P1 + "fn main()->i64{ let c:i64=0; let p=P{x:1}; if c==1 { p.x=5; } else { p.x=6; } return p.x; }", 6, "OK"),
    # --- loops (store composes with the while emit; reload each iteration) ---
    ("loop increment p.x = p.x + 1 three times",
     P1 + "fn main()->i64{ let p=P{x:10}; let mut i:i64=0; while i<3 { p.x=p.x+1; i=i+1; } return p.x; }", 13, "OK"),
    ("loop conditional store inside if inside while",
     P2 + "fn main()->i64{ let p=P{x:0,y:0}; let mut i:i64=0; while i<4 { if i>1 { p.y=p.y+i; } i=i+1; } return p.y; }", 5, "OK"),
    # --- struct passed to fn (memory-backed: callee store lands in caller's block) ---
    ("callee stores field, caller observes (write-back)",
     P1 + "fn bump(q:P){ q.x=7; }\nfn main()->i64{ let p=P{x:1}; bump(p); return p.x; }", 7, "OK"),
    ("callee stores from its own param value",
     P2 + "fn setxy(q:P,v:i64){ q.x=v; q.y=v+1; }\nfn main()->i64{ let p=P{x:0,y:0}; setxy(p,4); return p.x*10+p.y; }", 45, "OK"),
    ("caller stores, callee reads",
     P1 + "fn get(q:P)->i64{ return q.x; }\nfn main()->i64{ let p=P{x:1}; p.x=7; return get(p); }", 7, "OK"),
    ("struct-typed let alias observes store",
     P1 + "fn main()->i64{ let p=P{x:1}; let q:P=p; p.x=7; return q.x; }", 7, "OK"),
    # --- field READ / comparison statements must be UNCHANGED by the parse split ---
    ("p.x == 5 still a comparison in if-cond",
     P1 + "fn main()->i64{ let p=P{x:5}; if p.x==5 { return 1; } return 0; }", 1, "OK"),
    ("field-read-opening expression statement",
     P1 + "fn main()->i64{ let p=P{x:5}; let v:i64=p.x+1; return v; }", 6, "OK"),
    # --- fail-closed: unsupported store shapes MUST refuse 0B ---
    ("refuse nested path p.q.x = v",
     "struct Q{x:i64}\nstruct P{q:Q}\nfn main()->i64{ let q=Q{x:1}; let p=P{q:q}; p.q.x=5; return p.q.x; }", None, "0B"),
    ("refuse unbound receiver",
     P1 + "fn main()->i64{ z.x=5; return 0; }", None, "0B"),
    ("refuse unknown field",
     P1 + "fn main()->i64{ let p=P{x:1}; p.z=5; return p.x; }", None, "0B"),
    ("refuse non-struct receiver",
     P1 + "fn main()->i64{ let n:i64=3; n.x=5; return n; }", None, "0B"),
    ("refuse float field store",
     "struct F{a:f64}\nfn main()->i64{ let f=F{a:1.5}; f.a=2.5; return 0; }", None, "0B"),
    ("refuse float rhs into i64 field",
     P1 + "fn main()->i64{ let p=P{x:1}; p.x=1.5; return p.x; }", None, "0B"),
]

def main():
    ok = True
    for lbl, s, w, exp in CASES:
        st, rc = run(s)
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
