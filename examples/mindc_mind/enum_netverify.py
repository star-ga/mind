#!/usr/bin/env python3
# Canonical independent net-verify harness for C-LIKE ENUMS in the native-ELF backend.
# Run: MINDC_SO=/path/to/mindc_self.so python3 enum_netverify.py
# Every case is (label, source, want_rc, expect) where expect is "OK" (must run and
# return want_rc) or "0B" (must refuse — emit zero bytes / fail-closed).
# Enum paths desugar at parse time to synthetic int-lit constants; a wrong
# discriminant or a wrong match dispatch is a fail-OPEN silent miscompile and
# MUST be reported FAIL, never PASS.
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

COLOR = "enum Color { Red, Green, Blue }\n"

CASES = [
    # --- variant constants (implicit sequential discriminants) ---
    ("variant const Red=0",
     COLOR + "fn main()->i64{ return Color::Red; }", 0, "OK"),
    ("variant const Green=1",
     COLOR + "fn main()->i64{ return Color::Green; }", 1, "OK"),
    ("variant const Blue=2",
     COLOR + "fn main()->i64{ return Color::Blue; }", 2, "OK"),
    # --- explicit discriminants (increasing only) ---
    ("explicit disc A=5",
     "enum E { A = 5, B }\nfn main()->i64{ return E::A; }", 5, "OK"),
    ("implicit continues after explicit B=6",
     "enum E { A = 5, B }\nfn main()->i64{ return E::B; }", 6, "OK"),
    ("explicit mid-list C=11",
     "enum F { A, B = 10, C }\nfn main()->i64{ return F::C; }", 11, "OK"),
    # --- variant in arithmetic ---
    ("variant arithmetic Blue*10+Green",
     COLOR + "fn main()->i64{ return Color::Blue * 10 + Color::Green; }", 21, "OK"),
    ("variant equality compare",
     COLOR + "fn main()->i64{ let c:i64 = Color::Green; if c == Color::Green { return 7; } return 8; }", 7, "OK"),
    # --- match dispatch: each variant + the wildcard ---
    ("match dispatch Red arm",
     COLOR + "fn main()->i64{ let c:i64 = Color::Red; return match c { Color::Red => 10, Color::Green => 20, _ => 30 }; }", 10, "OK"),
    ("match dispatch Green arm",
     COLOR + "fn main()->i64{ let c:i64 = Color::Green; return match c { Color::Red => 10, Color::Green => 20, _ => 30 }; }", 20, "OK"),
    ("match dispatch wildcard arm (Blue unlisted)",
     COLOR + "fn main()->i64{ let c:i64 = Color::Blue; return match c { Color::Red => 10, Color::Green => 20, _ => 30 }; }", 30, "OK"),
    # --- match result used in an expression ---
    ("match result in expr",
     COLOR + "fn main()->i64{ let c:i64 = Color::Green; let m:i64 = match c { Color::Red => 10, Color::Green => 20, _ => 30 }; return m + 5; }", 25, "OK"),
    # --- enum returned from a fn (both -> Color and -> i64 spellings) ---
    ("enum returned from fn (-> Color)",
     COLOR + "fn pick(n:i64)->Color{ if n == 1 { return Color::Green; } return Color::Blue; }\nfn main()->i64{ return pick(1); }", 1, "OK"),
    ("enum returned from fn (-> i64)",
     COLOR + "fn pick(n:i64)->i64{ if n == 1 { return Color::Green; } return Color::Blue; }\nfn main()->i64{ return pick(0); }", 2, "OK"),
    # --- enum as a fn param ---
    ("enum as fn param (c: Color) matched in callee",
     COLOR + "fn score(c:Color)->i64{ return match c { Color::Red => 100, Color::Green => 50, _ => 1 }; }\nfn main()->i64{ return score(Color::Green); }", 50, "OK"),
    # --- int match must keep working (parse_match_chain restructure regression) ---
    ("int match still dispatches",
     "fn main()->i64{ let x:i64 = 2; return match x { 1 => 10, 2 => 20, _ => 30 }; }", 20, "OK"),
    ("int match wildcard still dispatches",
     "fn main()->i64{ let x:i64 = 9; return match x { 1 => 10, 2 => 20, _ => 30 }; }", 30, "OK"),
    # --- single-payload tier landed: payload decls now CONSTRUCT (tagged
    # --- 2-word block) instead of refusing; full battery in option_netverify.py ---
    ("payload enum payload-free variant constructs + matches",
     "enum E { A(i64), B }\nfn main()->i64{ let e:E = E::B; return match e { E::A(x) => x, E::B => 21 }; }", 21, "OK"),
    ("payload construction E::A(5) constructs + binds",
     "enum E { A(i64), B }\nfn main()->i64{ let e:E = E::A(5); return match e { E::A(x) => x, E::B => 0 }; }", 5, "OK"),
    # --- FAIL-CLOSED: exhaustiveness / unresolvable shapes -> 0B ---
    ("refuse payload-binding match arm on C-like enum",
     COLOR + "fn main()->i64{ let c:i64 = Color::Red; return match c { Color::Red(x) => 1, _ => 0 }; }", None, "0B"),
    ("refuse payload-binding in NON-first arm (else-position poison must be walked)",
     COLOR + "fn main()->i64{ let c:i64 = Color::Red; return match c { Color::Red => 1, Color::Green(x) => 2, _ => 0 }; }", None, "0B"),
    ("refuse non-exhaustive enum match without wildcard",
     COLOR + "fn main()->i64{ let c:i64 = Color::Red; return match c { Color::Red => 1, Color::Green => 2 }; }", None, "0B"),
    ("refuse non-exhaustive int match without wildcard",
     "fn main()->i64{ let x:i64 = 1; return match x { 1 => 10, 2 => 20 }; }", None, "0B"),
    ("refuse unknown variant Color::Purple",
     COLOR + "fn main()->i64{ return Color::Purple; }", None, "0B"),
    ("refuse undeclared enum path",
     "fn main()->i64{ return Shape::Circle; }", None, "0B"),
    ("refuse duplicate enum decl (ambiguous)",
     "enum D { A }\nenum D { A, B }\nfn main()->i64{ return D::A; }", None, "0B"),
    ("refuse non-increasing explicit discriminant",
     "enum G { A = 5, B = 2 }\nfn main()->i64{ return G::B; }", None, "0B"),
    ("refuse negative explicit discriminant",
     "enum H { A = -1, B }\nfn main()->i64{ return H::B; }", None, "0B"),
    ("refuse duplicate variant name",
     "enum J { A, A }\nfn main()->i64{ return J::A; }", None, "0B"),
    # --- pre-existing fail-open now closed: bare-ident pattern refuses ---
    ("refuse bare-ident match pattern (was garbage-constant fail-open)",
     COLOR + "fn main()->i64{ let c:i64 = Color::Red; return match c { Red => 1, _ => 0 }; }", None, "0B"),
    ("refuse bare unqualified variant reference",
     COLOR + "fn main()->i64{ return Red; }", None, "0B"),
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
