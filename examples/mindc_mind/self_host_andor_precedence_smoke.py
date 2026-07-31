#!/usr/bin/env python3
"""Front-end PARITY battery for `&&` / `||` precedence + short-circuit.

Companion to self_host_andor_smoke.py. Where that battery proves the pure-MIND
native-ELF path is value-correct and byte-identical to its explicit if-desugar,
THIS battery proves the two front-ends AGREE with each other and with the correct
tighter-binding semantics:

  1. PRECEDENCE PARITY — for each program+input, the pure-MIND native-ELF self-host
     path (`selftest_native_elf_h`) and the live Rust `mindc build --emit=binary`
     CPU oracle both run to the SAME exit code, and that code matches the correct
     (tighter-binding) truth-table value. Covers:
       - `&&` tighter than `||`   (a || b && c  ==  a || (b && c))
       - `&&` tighter than `||`   (a && b || c  ==  (a && b) || c)
       - `==` tighter than `&&`   (p == q && r == s  ==  (p==q) && (r==s))
       - `==` tighter than `||`   (p == q || r == s  ==  (p==q) || (r==s))
     A self-host != live divergence is a REPORTABLE precedence bug (like the `as`
     vs `+` precedence bug the signed-cast desugar surfaced).

  2. SHORT-CIRCUIT PARITY (RHS-would-fault) — `x != 0 && (100/x) > 5` with x=0 and
     `x == 0 || (100/x) > 5` with x=0 must return the LHS-decided value WITHOUT
     evaluating the RHS (an eager eval would SIGFPE => exit 136). Proven on BOTH
     front-ends.

Env: MINDC_SO = pure-MIND self-host cdylib; MINDC_BIN = live Rust mindc.
Exit 0 = ALL PASS; 2 = front-end divergence; 1 = wrong value.
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SO = os.environ.get("MINDC_SO", str(HERE / "libmindc_mind.so"))
MINDC = os.environ.get("MINDC_BIN", "./target/release/mindc")
ZERO_HASH = b"\x00" * 32

lib = ctypes.CDLL(SO)
lib.selftest_native_elf_h.restype = ctypes.c_int64
lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]


def mind_elf(src: bytes) -> bytes:
    sb = ctypes.create_string_buffer(src, len(src))
    hb = ctypes.create_string_buffer(ZERO_HASH, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(sb, ctypes.c_void_p).value, len(src),
        ctypes.cast(hb, ctypes.c_void_p).value)
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    n = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), n) if n > 0 else b""


def run_bytes(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "m.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def live_rc(src: bytes, tmp: pathlib.Path) -> int:
    s = tmp / "s.mind"
    s.write_bytes(src)
    o = tmp / "s.elf"
    r = subprocess.run([MINDC, "build", str(s), "--release", "--emit=binary",
                        "--out=" + str(o)], capture_output=True)
    if r.returncode != 0:
        return -999
    return subprocess.run([str(o)]).returncode


P3 = ("fn h(a: i64, b: i64, c: i64) -> i64 {{ if {e} {{ return 1; }} else {{ return 0; }} }} "
      "fn main() -> i64 {{ return h({a}, {b}, {c}); }}")
P4 = ("fn h(p: i64, q: i64, r: i64, s: i64) -> i64 {{ if {e} {{ return 1; }} else {{ return 0; }} }} "
      "fn main() -> i64 {{ return h({p}, {q}, {r}, {s}); }}")

CASES = []
# && tighter than || : a || b && c == a || (b && c)
_e = "a > 0 || b > 0 && c > 0"
CASES += [("or_of_and", _e, P3, dict(e=_e, a=a, b=b, c=c),
           1 if (a > 0 or (b > 0 and c > 0)) else 0)
          for (a, b, c) in [(1, 1, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)]]
# && tighter than || : a && b || c == (a && b) || c
_e = "a > 0 && b > 0 || c > 0"
CASES += [("and_or", _e, P3, dict(e=_e, a=a, b=b, c=c),
           1 if ((a > 0 and b > 0) or c > 0) else 0)
          for (a, b, c) in [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 0), (1, 1, 1)]]
# == tighter than && : p == q && r == s == (p==q) && (r==s)
_e = "p == q && r == s"
CASES += [("cmp_and", _e, P4, dict(e=_e, p=p, q=q, r=r, s=s),
           1 if ((p == q) and (r == s)) else 0)
          for (p, q, r, s) in [(1, 1, 2, 2), (1, 1, 1, 2), (1, 2, 2, 2), (5, 5, 7, 7), (5, 6, 7, 7)]]
# == tighter than || : p == q || r == s == (p==q) || (r==s)
_e = "p == q || r == s"
CASES += [("cmp_or", _e, P4, dict(e=_e, p=p, q=q, r=r, s=s),
           1 if ((p == q) or (r == s)) else 0)
          for (p, q, r, s) in [(1, 2, 3, 4), (1, 1, 3, 4), (1, 2, 3, 3), (0, 0, 0, 0)]]

# Short-circuit RHS-would-fault (div by zero in RHS): must not evaluate RHS.
SC = [
    ("sc_and_divzero",
     b"fn g(x: i64) -> i64 { if x != 0 && (100 / x) > 5 { return 1; } else { return 0; } } "
     b"fn main() -> i64 { return g(0); }", 0),
    ("sc_or_divzero",
     b"fn g(x: i64) -> i64 { if x == 0 || (100 / x) > 5 { return 7; } else { return 3; } } "
     b"fn main() -> i64 { return g(0); }", 7),
]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        divergences = 0
        fails = 0
        printed = set()
        for name, expr, tmpl, kw, exp in CASES:
            src = tmpl.format(**kw).encode()
            sh = run_bytes(mind_elf(src), tmp)
            lv = live_rc(src, tmp)
            parity = (sh == lv)
            ok = parity and sh == exp and lv == exp
            tag = "PASS" if ok else ("DIVERGE" if not parity else "WRONGVAL")
            if not parity:
                divergences += 1
            if not ok:
                fails += 1
            note = "" if name in printed else "  [" + expr + "]"
            printed.add(name)
            args = [v for k, v in kw.items() if k != "e"]
            print(f"  {tag:8} {name:10} args={args} self={sh} live={lv} exp={exp}{note}")
        for name, src, exp in SC:
            sh = run_bytes(mind_elf(src), tmp)
            lv = live_rc(src, tmp)
            parity = (sh == lv)
            ok = parity and sh == exp and lv == exp
            tag = "PASS" if ok else ("DIVERGE" if not parity else "WRONGVAL")
            if not parity:
                divergences += 1
            if not ok:
                fails += 1
            print(f"  {tag:8} {name:15} self={sh} live={lv} exp={exp}  (136=SIGFPE=eager-eval)")
        print()
        if divergences:
            print(f"FAIL: {divergences} FRONT-END DIVERGENCE(S) (self-host != live Rust)")
            return 2
        if fails:
            print(f"FAIL: {fails} wrong-value case(s)")
            return 1
        print(f"ALL PASS  ({len(CASES) + len(SC)} cases: self-host == live Rust == correct; "
              f"&& tighter than ||, == tighter than both; short-circuit proven both front-ends)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
