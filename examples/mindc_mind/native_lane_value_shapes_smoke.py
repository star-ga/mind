#!/usr/bin/env python3
"""native-lane-v1 value shapes: the bridge path, a marked compiler, and MLIR.

The candidate native-lane-v1 profile (src/ir/native_lane_profile.rs) admits
three shapes the historical frozen profile refuses, because the MIND-Lab
programs need them and a compiler carrying the `nblane01` mark implements them:

  * a store intrinsic's result read as a value (the `w_set` helper returns it):
    the defined 0 that MLIR materialises and the marked compiler writes;
  * full-width `i64 as f64` / `f64 as i64` (round-to-nearest-even / saturating);
  * an `if` whose other branch returns, breaks or continues (lowering forwards
    the reaching branch's value to the merge).

Every case is built twice through `mindc build` -- `--backend native
--native-profile bounded` with MINDC_NATIVE_ELF naming the marked compiler, and
`--backend mlir` -- and both executables must produce the value computed here
independently of either backend (exit status, or 8-byte little-endian stdout
words). The store cases dirty the stack first, so a compiler that leaves a store
result slot unwritten reads stale data and fails (the pre-fix frozen seed returns
19 instead of 17 on `deep_dirty_then_tail`). `main(argc, argv)` is included
because an explicit-layout image starts with the allocator runtime prefix, and
the bridge must find the argc/argv binding at the program entry behind it.

The smoke names its compiler explicitly and refuses to run without one; CI
passes the committed bounded seed, which carries the mark since the reseed:

  MINDC_BIN=target/release/mindc python3 native_lane_value_shapes_smoke.py \\
      --compiler examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf

Both paths are resolved against the invoking directory before use: every build
runs with its case directory as cwd, where a relative path names nothing.

Exit: 0 every case matches; 1 a case differs; 2 BLOCKED (no mindc, no marked
compiler, or a compiler without the mark).
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = HERE / "testdata" / "native_lane_values"
MARK = b"\x48\xb8nblane01"
TOOLS = {
    "MLIR_OPT": "/usr/lib/llvm-20/bin/mlir-opt",
    "MLIR_TRANSLATE": "/usr/lib/llvm-20/bin/mlir-translate",
    "CLANG": "/usr/lib/llvm-20/bin/clang",
}

DIRTY = (
    "fn dirty(a: i64) -> i64 { let b: i64 = a + 1; let c: i64 = b * 3; let d: i64 = c + b; "
    "let e: i64 = d - 2; let f: i64 = e * e; let g: i64 = f + d; g }\n"
)
ALLOC = "let p: i64 = __mind_alloc(8);"

# (id, source, expected exit status). A store result is the defined value 0.
EXIT_CASES = [
    ("tail_i64", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i64(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 7 + (z - z) }}", 7),
    ("tail_i8", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i8(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 9 + (z - z) }}", 9),
    ("tail_i32", DIRTY + "fn w(p: i64) -> i64 { __mind_store_i32(p, 99) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(5); w(p) + 11 + (z - z) }}", 11),
    ("let_init", DIRTY + f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(3); "
     "let x: i64 = __mind_store_i64(p, 77); x + 5 + (z - z) }", 5),
    ("assign", f"fn main() -> i64 {{ {ALLOC} let mut x: i64 = 40; x = __mind_store_i64(p, 77); "
     "x + __mind_load_i64(p) - 70 }", 7),
    ("return_value", "fn w(p: i64, v: i64) -> i64 { return __mind_store_i64(p, v); }\n"
     f"fn main() -> i64 {{ {ALLOC} w(p, 41) + __mind_load_i64(p) }}", 41),
    ("binop", f"fn main() -> i64 {{ {ALLOC} return 13 + 6 * __mind_store_i8(p, 9); }}", 13),
    ("call_arg", "fn id(v: i64) -> i64 { v + 2 }\n"
     f"fn main() -> i64 {{ {ALLOC} id(__mind_store_i64(p, 50)) }}", 2),
    ("nested_store_arg", "fn main() -> i64 { let p: i64 = __mind_alloc(16); "
     "__mind_store_i64(p + 8, __mind_store_i64(p, 33)); "
     "__mind_load_i64(p + 8) + __mind_load_i64(p) }", 33),
    ("if_cond", f"fn main() -> i64 {{ {ALLOC} if __mind_store_i64(p, 5) == 0 {{ return 21; }} "
     "return 3; }", 21),
    ("while_cond", f"fn main() -> i64 {{ {ALLOC} let mut i: i64 = 0; "
     "while __mind_store_i64(p, i) + i < 6 { i = i + 1; } i + __mind_load_i64(p) }", 12),
    ("value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 0; let y: i64 = if c == 1 {{ 5 }} "
     "else { __mind_store_i8(p, 77) }; y + 3 }", 3),
    ("nested_value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 1; let d: i64 = 1; "
     "let y: i64 = if c == 1 { if d == 1 { __mind_store_i64(p, 70) } else { 2 } } else { 3 }; "
     "y + 5 }", 5),
    ("deep_value_if", f"fn main() -> i64 {{ {ALLOC} let c: i64 = 1; "
     "let y: i64 = 40 + if c == 1 { __mind_store_i64(p, 70) } else { 2 }; y }", 40),
    ("stmt_if_both_stores", "fn f(p: i64, k: i64) -> i64 { if k == 0 { __mind_store_i64(p, 4); } "
     "else { __mind_store_i64(p, 6); } __mind_load_i64(p) }\n"
     f"fn main() -> i64 {{ {ALLOC} f(p, 0) * 10 + f(p, 1) }}", 46),
    ("store_as_f64", f"fn main() -> i64 {{ {ALLOC} let f: f64 = __mind_store_i64(p, 70) as f64; "
     "let g: f64 = f + 2.5; g as i64 }", 2),
    ("logic_and_bits", f"fn main() -> i64 {{ {ALLOC} if !__mind_store_i64(p, 1) && "
     "__mind_load_i64(p) == 1 { return __mind_store_i64(p, 255) | 19; } 2 }", 19),
    ("loop_accumulate", "fn w(p: i64) -> i64 { let mut i: i64 = 0; let mut acc: i64 = 0; "
     "while i < 5 { let x: i64 = __mind_store_i64(p, i + 100); acc = acc + x + 1; i = i + 1; } acc }\n"
     f"fn main() -> i64 {{ {ALLOC} w(p) + __mind_load_i64(p) - 100 }}", 9),
    ("deep_dirty_then_tail", DIRTY + "fn w(p: i64, a: i64, b: i64, c: i64) -> i64 { "
     "let s: i64 = a * b + c; __mind_store_i64(p, s) }\n"
     f"fn main() -> i64 {{ {ALLOC} let z: i64 = dirty(9); let r: i64 = w(p, 3, 4, 5); "
     "r + __mind_load_i64(p) + (z - z) }", 17),
    ("w_set_shape", "fn w_get(w_addr: i64, i: i64) -> i64 { __mind_load_i64(w_addr + i * 8) }\n"
     "fn w_set(w_addr: i64, i: i64, v: i64) -> i64 { __mind_store_i64(w_addr + i * 8, v) }\n"
     "fn main() -> i64 { let w: i64 = __mind_alloc(64); let mut i: i64 = 0; let mut bad: i64 = 0; "
     "while i < 8 { bad = bad + w_set(w, i, i * 3 + 1); i = i + 1; } w_get(w, 7) + bad }", 22),
    ("diverging_else", "fn f(first: i64) -> i64 { let mut width: i64 = 0; let mut cp: i64 = 0; "
     "if first < 128 { width = 1; cp = first; } else { if first >= 194 && first <= 223 { "
     "width = 2; cp = first & 31; } else { return 0 - 1; } } width * 100 + cp }\n"
     "fn main() -> i64 { f(5) + f(200) - f(250) - 309 }", 5),
    ("diverging_loop_exits", "fn f(n: i64) -> i64 { let mut i: i64 = 0; let mut s: i64 = 0; "
     "while i < n { if i == 3 { s = s + 10; } else { if i > 6 { break; } else { i = i + 1; "
     "continue; } } i = i + 1; } s + i }\nfn main() -> i64 { f(20) }", 17),
]
ARGV_CASE = ("main_argc_argv", "fn main(argc: i64, argv: i64) -> i64 { "
             "if argv == 0 { return 99; } argc + 4 }", 7)


def _bits(u: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", u & (2**64 - 1)))[0]


def _fbits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _sat(x: float) -> int:
    if math.isnan(x):
        return 0
    if x >= 2.0**63:
        return 2**63 - 1
    if x <= -(2.0**63):
        return -(2**63)
    return int(x)


def _edges_oracle() -> list[int]:
    nan, nnan = _bits(9221120237041090560), _bits(-2251799813685248)
    pinf, ninf = _bits(9218868437227405312), _bits(-4503599627370496)
    big, two63, below = _bits(5085969321953787526), _bits(4890909195324358656), _bits(4890909195324358655)
    floats = (nan, nnan, pinf, ninf, big, -big, 9.7, -9.7, 0.5, -0.5, two63, -two63, below, -below,
              _bits(-(2**63)), _bits(1))
    vals = [_sat(v) for v in floats] + [_sat(nan), _sat(big), _sat(-big)]
    vals += [_fbits(float(k)) for k in (9007199254740993, 2**63 - 1, -(2**63), -1, 0,
                                         9007199254740995, -9007199254740993)]
    vals += [_sat(float(-123456789)), _fbits(3.5), _sat(14.0)]
    return vals


def _exprs_oracle() -> list[int]:
    a, b, pv, k = 3.75, 2.5, 4617315517961601024, 1000001
    acc = 0.0
    for i in range(10):
        acc = acc + float(i) * 0.5
    return [_sat(a + b), _sat(a * b), _sat(a - b * 4.0), _sat(a / 2.0), _sat(a + b + 0.25),
            _sat(7.75), _sat(-7.75), _sat(_bits(pv)), _sat(-a), _sat(float(k) / 3.0),
            _fbits(float(k * 3 + 1)), _fbits(float(pv)), _sat(a), _sat(acc), _fbits(acc),
            _sat(float(_sat(a))), _fbits(1.0), _fbits(0.0)]


def _words(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", v) for v in values)


def _build(mindc: str, src: pathlib.Path, out: pathlib.Path, native: bool, env: dict) -> tuple[int, str]:
    cmd = [mindc, "build", str(src), "--target", "cpu", "--emit", "binary", "--no-cache",
           "--out", str(out)]
    cmd += ["--backend", "native", "--native-profile", "bounded"] if native else ["--backend", "mlir"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, cwd=src.parent)
    return r.returncode, r.stderr


def _run(exe: pathlib.Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["/usr/bin/prlimit", "--as=2147483648", "--", str(exe), *argv],
                          capture_output=True, timeout=60, env={"LC_ALL": "C"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compiler", default=os.environ.get("MINDC_NATIVE_LANE_ELF"))
    args = ap.parse_args()
    # Absolute before use: `_build` runs mindc with the case directory as cwd, so
    # a relative MINDC_BIN or --compiler (the bridge reads MINDC_NATIVE_ELF from
    # there) would name nothing and the native leg would be refused, not tested.
    mindc = os.environ.get("MINDC_BIN", str(REPO / "target" / "release" / "mindc"))
    mindc = str(pathlib.Path(mindc).resolve())
    if args.compiler:
        args.compiler = str(pathlib.Path(args.compiler).resolve())
    if not pathlib.Path(mindc).exists():
        print(f"BLOCKED: mindc not found at {mindc}")
        return 2
    if not args.compiler or not pathlib.Path(args.compiler).exists():
        print("BLOCKED: pass --compiler <marked native compiler> (CI passes the committed "
              "bounded seed, testdata/selfhost_loop_bounded/stage1.elf)")
        return 2
    if MARK not in pathlib.Path(args.compiler).read_bytes():
        print(f"BLOCKED: {args.compiler} does not carry the native-lane-v1 mark, so the bridge "
              "would apply the historical profile instead of the one under test")
        return 2
    env = dict(os.environ, **{k: os.environ.get(k, v) for k, v in TOOLS.items()},
               MINDC_NATIVE_ELF=args.compiler, MINDC_STD_DIR=str(REPO / "std"), LC_ALL="C")
    env.pop("MINDC_SO", None)
    cases: list[tuple[str, pathlib.Path | str, list[str], int, bytes | None]] = []
    cases += [(cid, src, [], want, None) for cid, src, want in EXIT_CASES]
    cases.append((ARGV_CASE[0], ARGV_CASE[1], ["a", "b"], ARGV_CASE[2], None))
    cases.append(("casts_edges", FIXTURES / "casts_edges.mind", [], 0, _words(_edges_oracle())))
    cases.append(("casts_exprs", FIXTURES / "casts_exprs.mind", [], 0, _words(_exprs_oracle())))
    ran = fail = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for cid, src, argv, want_exit, want_out in cases:
            case_dir = tmp / cid
            case_dir.mkdir()
            source = case_dir / "main.mind"
            text = src.read_text() if isinstance(src, pathlib.Path) else src + "\n"
            source.write_text(text)
            seen = []
            for native in (True, False):
                exe = case_dir / ("native" if native else "mlir")
                rc, err = _build(mindc, source, exe, native, env)
                if rc != 0:
                    seen.append(f"build={rc}: {err.strip().splitlines()[-1] if err.strip() else ''}")
                    continue
                r = _run(exe, argv)
                ok_out = want_out is None or r.stdout == want_out
                ok = r.returncode == want_exit and ok_out and r.stderr == b""
                seen.append("ok" if ok else f"exit={r.returncode} stdout={'=' if ok_out else '!='}")
            ran += 1
            passed = seen == ["ok", "ok"]
            fail += not passed
            print(f"  {'PASS' if passed else 'FAIL'}  {cid}: native={seen[0]} mlir={seen[1]}")
    print(f"SDLC-GATE native_lane_value_shapes ran={ran} fail={fail}")
    return 1 if fail or ran == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
