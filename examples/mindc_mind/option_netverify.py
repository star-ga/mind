#!/usr/bin/env python3
# Canonical independent net-verify harness for SINGLE-PAYLOAD ENUMS (the
# Option/Result tier) in the native-ELF backend.
# Run: MINDC_SO=/path/to/mindc_self.so python3 option_netverify.py
# Every case is (label, source, want_rc, expect) where expect is "OK" (must run
# and return want_rc) or "0B" (must refuse — emit zero bytes / fail-closed).
# A payload-enum value is a tagged 2-word block [tag, payload] built with the
# proven array-literal ABI; a wrong tag, a wrong payload bind, or a wrong
# match dispatch is a fail-OPEN silent miscompile and MUST be reported FAIL,
# never PASS.
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

OPT = "enum Opt { Some(i64), None }\n"

CASES = [
    # --- construction + match binding (both variants, no wildcard: coverage) ---
    ("Some(5) match binds x=5",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(5); return match o { Opt::Some(x) => x, Opt::None => 99 }; }", 5, "OK"),
    ("None dispatches the None arm",
     OPT + "fn main()->i64{ let o:Opt = Opt::None; return match o { Opt::Some(x) => x, Opt::None => 99 }; }", 99, "OK"),
    ("arm order flipped (None first)",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(7); return match o { Opt::None => 99, Opt::Some(x) => x }; }", 7, "OK"),
    # --- payload used in arithmetic in the arm ---
    ("payload arithmetic x*3+1",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(4); return match o { Opt::Some(x) => x * 3 + 1, Opt::None => 0 }; }", 13, "OK"),
    ("payload from an expression Some(2+3*4)",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(2 + 3 * 4); return match o { Opt::Some(x) => x, Opt::None => 0 }; }", 14, "OK"),
    # --- wildcard terminal instead of full coverage ---
    ("wildcard arm on Some",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(6); return match o { Opt::Some(x) => x + 1, _ => 50 }; }", 7, "OK"),
    ("wildcard arm taken on None",
     OPT + "fn main()->i64{ let o:Opt = Opt::None; return match o { Opt::Some(x) => x + 1, _ => 50 }; }", 50, "OK"),
    # --- block-statement arms with return ---
    ("block arms with return",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(9); return match o { Opt::Some(x) => { return x * 2; }, Opt::None => { return 1; } }; }", 18, "OK"),
    # --- fn returning Opt, matched by the caller ---
    ("fn returns Opt, caller matches Some",
     OPT + "fn mk(n:i64)->Opt{ return Opt::Some(n + 1); }\nfn main()->i64{ let o:Opt = mk(41); return match o { Opt::Some(x) => x, Opt::None => 0 }; }", 42, "OK"),
    ("fn returns Opt, caller matches None",
     OPT + "fn mk(n:i64)->Opt{ return Opt::None; }\nfn main()->i64{ let o:Opt = mk(0); return match o { Opt::Some(x) => x, Opt::None => 77 }; }", 77, "OK"),
    # --- Some/None chosen by a runtime condition ---
    ("runtime-chosen Some",
     OPT + "fn pick(c:i64)->Opt{ if c == 1 { return Opt::Some(33); } return Opt::None; }\nfn main()->i64{ let o:Opt = pick(1); return match o { Opt::Some(x) => x, Opt::None => 9 }; }", 33, "OK"),
    ("runtime-chosen None",
     OPT + "fn pick(c:i64)->Opt{ if c == 1 { return Opt::Some(33); } return Opt::None; }\nfn main()->i64{ let o:Opt = pick(0); return match o { Opt::Some(x) => x, Opt::None => 9 }; }", 9, "OK"),
    # --- nested: Option of a match result / match result feeding construction ---
    ("Some(match result) nested",
     OPT + "fn main()->i64{ let n:i64 = 2; let o:Opt = Opt::Some(match n { 1 => 10, 2 => 20, _ => 30 }); return match o { Opt::Some(x) => x + 1, Opt::None => 0 }; }", 21, "OK"),
    ("match result then re-wrapped",
     OPT + "fn main()->i64{ let a:Opt = Opt::Some(5); let m:i64 = match a { Opt::Some(x) => x, Opt::None => 0 }; let b:Opt = Opt::Some(m * 2); return match b { Opt::Some(y) => y, Opt::None => 1 }; }", 10, "OK"),
    # --- payload-free variants carry an unused payload slot: 3-variant enum ---
    ("3-variant Result-like Ok arm",
     "enum Res { Okv(i64), Errv(i64), Nil }\nfn main()->i64{ let r:Res = Res::Errv(8); return match r { Res::Okv(a) => a, Res::Errv(e) => e * 10, Res::Nil => 3 }; }", 80, "OK"),
    ("3-variant payload-free arm dispatch",
     "enum Res { Okv(i64), Errv(i64), Nil }\nfn main()->i64{ let r:Res = Res::Nil; return match r { Res::Okv(a) => a, Res::Errv(e) => e * 10, Res::Nil => 3 }; }", 3, "OK"),
    # --- C-like enums must keep working (regression, same parse path) ---
    ("C-like enum still resolves + matches",
     "enum Color { Red, Green, Blue }\nfn main()->i64{ let c:i64 = Color::Green; return match c { Color::Red => 10, Color::Green => 20, _ => 30 }; }", 20, "OK"),
    # --- int match must keep working (parse_match_chain restructure regression) ---
    ("int match still dispatches",
     "fn main()->i64{ let x:i64 = 2; return match x { 1 => 10, 2 => 20, _ => 30 }; }", 20, "OK"),
    ("int match wildcard still dispatches",
     "fn main()->i64{ let x:i64 = 9; return match x { 1 => 10, 2 => 20, _ => 30 }; }", 30, "OK"),
    # --- FAIL-CLOSED: decl shapes -> 0B ---
    ("multi-field payload refuses",
     "enum P { Two(i64, i64), None }\nfn main()->i64{ let o:P = P::Two(1); return 0; }", 0, "0B"),
    ("tuple payload refuses",
     "enum P { T((i64, i64)), None }\nfn main()->i64{ let o:P = P::T(1); return 0; }", 0, "0B"),
    ("non-i64 payload (f64) refuses",
     "enum P { F(f64), None }\nfn main()->i64{ let o:P = P::F(1); return 0; }", 0, "0B"),
    ("non-i64 payload (i8) refuses",
     "enum P { N(i8), None }\nfn main()->i64{ let o:P = P::N(1); return 0; }", 0, "0B"),
    ("generic enum decl refuses",
     "enum Box<T> { Val(T), Nil }\nfn main()->i64{ let o:Box = Box::Val(1); return 0; }", 0, "0B"),
    ("explicit discriminant in payload enum refuses",
     "enum P { A(i64), B = 5 }\nfn main()->i64{ let o:P = P::A(1); return 0; }", 0, "0B"),
    # --- FAIL-CLOSED: use-site shapes -> 0B ---
    ("payload variant without payload refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some; return 0; }", 0, "0B"),
    ("payload-free variant called with args refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::None(5); return 0; }", 0, "0B"),
    ("two-argument construction refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1, 2); return 0; }", 0, "0B"),
    ("binding on payload-free variant refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::None; return match o { Opt::None(x) => x, _ => 0 }; }", 0, "0B"),
    ("non-binding arm on payload variant refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some => 5, Opt::None => 0 }; }", 0, "0B"),
    ("underscore binding refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some(_) => 5, Opt::None => 0 }; }", 0, "0B"),
    # --- FAIL-CLOSED: exhaustiveness / mixing -> 0B ---
    ("non-exhaustive payload match refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some(x) => x }; }", 0, "0B"),
    ("mixed payload + int pattern refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some(x) => x, 1 => 5, _ => 0 }; }", 0, "0B"),
    ("mixed int + payload pattern refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Some(1); return match o { 1 => 5, Opt::Some(x) => x, _ => 0 }; }", 0, "0B"),
    ("mixed two-enum payload match refuses",
     OPT + "enum Alt { A(i64), B }\nfn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some(x) => x, Alt::B => 5, _ => 0 }; }", 0, "0B"),
    ("mixed C-like + payload pattern refuses",
     OPT + "enum Color { Red, Green }\nfn main()->i64{ let o:Opt = Opt::Some(1); return match o { Opt::Some(x) => x, Color::Red => 5, _ => 0 }; }", 0, "0B"),
    ("duplicate variant name refuses",
     "enum D { A(i64), A(i64) }\nfn main()->i64{ let o:D = D::A(1); return 0; }", 0, "0B"),
    ("unknown variant refuses",
     OPT + "fn main()->i64{ let o:Opt = Opt::Nope(1); return 0; }", 0, "0B"),
]

fails = 0
for label, src, want, expect in CASES:
    st, rc = run(src)
    if expect == "0B":
        ok = (st == "0B")
        got = st
    else:
        ok = (st == "OK" and rc == want)
        got = f"({st},{rc})"
    print(f"{'PASS' if ok else 'FAIL'}  {label:48s} expect={expect} want_rc={want} got={got}")
    if not ok: fails += 1
print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
