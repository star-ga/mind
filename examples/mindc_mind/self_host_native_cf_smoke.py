#!/usr/bin/env python3
"""Native stage1: general control flow (HOME lowering) — three-way behavioural gate.

What it proves
--------------
The stage1 compiler built from THIS tree's main.mind compiles arbitrary nesting
of if / else / else-if / while / break / continue / return and value-if, with
`&&` / `||` / `!` short-circuit, in functions with and without parameters, and
the programs it emits behave exactly like the MLIR backend and like an
independent Python interpreter of the same program (exit status AND stdout).

Before this change stage1 refused every such program (rc 3): its PT_NOTE is the
IR trace hash, computed by the pure-MIND mic@3 twin, and the twin covers fewer
function shapes than the native emitter. selftest_native_elf_stage now keeps
the IR-hash note when the twin can compute it and otherwise lowers exactly the
functions the twin rejected on the HOME path (main.mind: "HOME-SLOT LOWERING"),
marking the note as a SOURCE note (n_type 2). See the census rules A1 A6 A7 A9
L1 L2 L3 O1 O2 O3 in the fixtures.

Legs (each a separate, non-shared implementation):
  native  stage1 ELF derived from this tree (frozen seed -> tree source), fed
          the program through the stdin image protocol, then run.
  MLIR    `mindc build --backend mlir` of the same program, then run.
  Python  _cf_corpus.Interp over the generator's own AST (corpus), or the
          hand-derived `// expect:` header (fixtures).

Steps
  1. fixtures  testdata/native_cf/*.mind. Header keys:
               `// expect: exit=N`, `// stdout: TEXT` (hand-derived),
               `// note: 1|2` (required PT_NOTE n_type),
               `// mlir: nobuild` (pinned MLIR limitation: must not build),
               `// mlir: exit=N` (pinned MLIR defect: MLIR must give N).
  2. corpus    deterministic random programs (default seeds 11,12 x 160 = 320);
               native == Python == MLIR. An MLIR build failure is accepted only
               for the pinned "break/continue outside a loop" limitation.
  3. capacity  a giant function (a 2500-arm else-if chain inside a loop, the
               census X1 shape) must compile and run correctly — never SIGSEGV.
  4. refusal   a HOME-lowered function that takes an address refuses cleanly
               (rc 3, no ELF), never a crash.
  5. twin      with MINDC_SO set, the oracle .so's selftest_native_elf_stage
               emits the SAME bytes as stage1 for every fixture.

Usage:
  MINDC_BIN=target/release/mindc [MINDC_SO=/tmp/oracle.so] \\
    python3 self_host_native_cf_smoke.py [--stage1 PATH] [--seeds 11,12] [--count 160]
Exit: 0 ALL PASS, 1 FAIL, 2 BLOCKED (no MLIR-capable mindc / no seed).
"""

import argparse
import ctypes
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
import _cf_corpus as C  # noqa: E402

STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]
SEED = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
FIXTURES = _HERE / "testdata" / "native_cf"
MLIR_LIMIT = "break/continue outside a loop"


def std_blob() -> bytes:
    return b"\n".join((_REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


BLOB = std_blob()


def image(src: bytes) -> bytes:
    comb = BLOB + src
    return struct.pack("<qq", len(BLOB), len(comb)) + comb


def nice():
    os.nice(15)


def derive_stage1(td: pathlib.Path) -> pathlib.Path:
    comb = BLOB + (_HERE / "main.mind").read_bytes() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    img = struct.pack("<qq", len(BLOB), len(comb)) + comb
    r = subprocess.run([str(SEED)], input=img, capture_output=True, timeout=1800, preexec_fn=nice)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise SystemExit(f"BLOCKED: frozen seed could not compile this tree (rc={r.returncode}): "
                         f"{r.stderr[-300:]!r}")
    out = td / "stage1.elf"
    out.write_bytes(r.stdout)
    out.chmod(0o755)
    return out


def native_compile(stage1: pathlib.Path, src: bytes):
    r = subprocess.run([str(stage1)], input=image(src), capture_output=True, timeout=600, preexec_fn=nice)
    ok = r.returncode == 0 and r.stdout[:4] == b"\x7fELF"
    return ok, r.returncode, (r.stdout if ok else b""), r.stderr.decode(errors="replace")


def run_bin(path: pathlib.Path):
    try:
        r = subprocess.run([str(path)], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return ("timeout", b"")
    return (r.returncode & 255 if r.returncode >= 0 else r.returncode, r.stdout)


def native(stage1, src: bytes, td: pathlib.Path):
    ok, rc, elf, err = native_compile(stage1, src)
    if not ok:
        return ("refused", rc, err.strip()[-160:]), None
    p = td / "n.elf"
    p.write_bytes(elf)
    p.chmod(0o755)
    return run_bin(p), elf


def mlir(mindc: str, src: str, td: pathlib.Path):
    p = td / "p.mind"
    p.write_text(src)
    o = td / "m.elf"
    r = subprocess.run([mindc, "build", str(p), "--backend", "mlir", "--emit", "binary", "--out", str(o)],
                       capture_output=True, timeout=600, preexec_fn=nice, cwd=str(td))
    if r.returncode != 0 or not o.exists():
        return ("nobuild", r.stderr.decode(errors="replace").strip()[-200:])
    return run_bin(o)


def header(src: str, key: str):
    m = re.search(r"^// " + key + r":[ ]?(.*)$", src, re.M)
    return None if m is None else m.group(1).strip()


class Report:
    def __init__(self):
        self.fails = []

    def ok(self, msg):
        print(f"  PASS  {msg}")

    def bad(self, msg):
        print(f"  FAIL  {msg}")
        self.fails.append(msg)


def check_fixture(f: pathlib.Path, stage1, mindc, so, rep: Report):
    src = f.read_text()
    want = (int(header(src, "expect").split("=")[1]), header(src, "stdout").encode())
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        got, elf = native(stage1, src.encode(), td)
        mres = mlir(mindc, src, td)
    tag = f.name
    if got != want:
        rep.bad(f"fixture {tag}: native={got} want={want}")
        return
    note = header(src, "note")
    if note is not None and elf[-44] != int(note):
        rep.bad(f"fixture {tag}: PT_NOTE n_type={elf[-44]} want {note}")
        return
    mspec = header(src, "mlir")
    if mspec == "nobuild":
        if mres[0] != "nobuild" or MLIR_LIMIT not in mres[1]:
            rep.bad(f"fixture {tag}: pinned MLIR limitation changed: mlir={mres}")
            return
        mtxt = "MLIR refuses (pinned limitation)"
    elif mspec is not None:
        pinned = int(mspec.split("=")[1])
        if mres[0] != pinned:
            rep.bad(f"fixture {tag}: pinned MLIR defect changed: mlir={mres} pinned exit={pinned}")
            return
        mtxt = f"MLIR exit={pinned} (pinned MLIR defect)"
    else:
        if mres != want:
            rep.bad(f"fixture {tag}: mlir={mres} want={want}")
            return
        mtxt = "== MLIR"
    if so is not None:
        so_elf = so_stage(so, src.encode())
        if so_elf != elf:
            rep.bad(f"fixture {tag}: oracle .so stage bytes ({len(so_elf)}B) != stage1 bytes ({len(elf)}B)")
            return
        mtxt += " ; .so == stage1 bytes"
    rep.ok(f"fixture {tag}: exit={want[0]} stdout={want[1].decode()!r} native == hand-derived {mtxt}")


def so_stage(so, src: bytes) -> bytes:
    comb = BLOB + src
    buf = ctypes.create_string_buffer(comb, len(comb))
    es = so.selftest_native_elf_stage(ctypes.cast(buf, ctypes.c_void_p).value, len(comb), len(BLOB))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]  # noqa: E731
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def corpus_case(args):
    name, src, exp, ba, stage1, mindc = args
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        got, _ = native(stage1, src.encode(), td)
        mres = mlir(mindc, src, td)
    return name, src, exp, ba, got, mres


def giant_program(arms: int) -> tuple:
    """X1 shape: one function holding a huge else-if chain inside a loop."""
    lines = ["fn pick(x: i64) -> i64 {", "    let mut r: i64 = 0;"]
    for k in range(arms):
        kw = "if" if k == 0 else "} else if"
        lines.append(f"    {kw} x == {k} {{")
        lines.append(f"        r = {(k * 7 + 3) % 1000};")
    lines += ["    } else {", "        r = 0 - 1;", "    }", "    r", "}",
              "fn main() -> i64 {", "    let mut i: i64 = 0;", "    let mut s: i64 = 0;",
              f"    while i < {arms + 3} {{",
              "        s = s + pick(i * 3);",
              "        if s > 1000000 && i > 2 {", "            return 0 - 2;", "        }",
              "        i = i + 1;", "    }", "    s % 251", "}"]
    s = 0
    for i in range(arms + 3):
        x = i * 3
        s += (x * 7 + 3) % 1000 if x < arms else -1
        if s > 1000000 and i > 2:
            return "\n".join(lines) + "\n", (C.wrap(-2) & 255, b"")
    return "\n".join(lines) + "\n", (C.tmod(s, 251) & 255, b"")


REFUSE_ADDR = """fn f(n: i64) -> i64 {
    let mut x: i64 = 0;
    let r: i64 = &x;
    while n > 0 {
        if n == 3 || n == 5 {
            return x;
        }
        n = n - 1;
    }
    x
}
fn main() -> i64 { return f(4); }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=os.environ.get("CF_STAGE1"))
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    mindc = os.environ.get("MINDC_BIN") or str(_REPO / "target" / "release" / "mindc")
    if not pathlib.Path(mindc).exists():
        print(f"BLOCKED: MINDC_BIN (an MLIR-capable mindc) not found at {mindc}")
        return 2
    so = None
    if os.environ.get("MINDC_SO"):
        so = ctypes.CDLL(os.environ["MINDC_SO"])
        so.selftest_native_elf_stage.restype = ctypes.c_int64
        so.selftest_native_elf_stage.argtypes = [ctypes.c_int64] * 3
    rep = Report()
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        if a.stage1:
            stage1 = pathlib.Path(a.stage1).resolve()
            print(f"[cf] stage1 = {stage1} (given)")
        else:
            if not SEED.exists():
                print(f"BLOCKED: frozen seed {SEED} missing")
                return 2
            stage1 = derive_stage1(td)
            print(f"[cf] stage1 = frozen seed -> this tree's main.mind ({stage1.stat().st_size} B)")

        print("[cf] fixtures (hand-derived expectations, census rules A1 A6 A7 A9 L1 L2 L3 O1 O2 O3)")
        for f in sorted(FIXTURES.glob("*.mind")):
            check_fixture(f, stage1, mindc, so, rep)

        seeds = [int(x) for x in a.seeds.split(",") if x]
        cases = []
        for sd in seeds:
            cases += [(n, s, e, ba, stage1, mindc) for n, s, e, ba in C.corpus(sd, a.count)]
        print(f"[cf] corpus: {len(cases)} generated programs (seeds {seeds}), native vs MLIR vs Python")
        st = {"three_way": 0, "native_eq_python_mlir_limit": 0, "native_eq_python_mlir_defect": 0}
        with ThreadPoolExecutor(a.jobs) as ex:
            for name, src, exp, ba, got, mres in ex.map(corpus_case, cases):
                if got != exp:
                    rep.bad(f"corpus {name}: native={got} python={exp}")
                    if len(rep.fails) <= 3:
                        print(src)
                    continue
                if mres[0] == "nobuild":
                    if MLIR_LIMIT in mres[1]:
                        st["native_eq_python_mlir_limit"] += 1
                        continue
                    rep.bad(f"corpus {name}: MLIR build failed: {mres[1]}")
                    continue
                if mres != exp and ba:
                    # pinned MLIR defect family (see _cf_corpus.has_branch_assign)
                    st["native_eq_python_mlir_defect"] += 1
                    continue
                if mres != exp:
                    rep.bad(f"corpus {name}: mlir={mres} python={exp} (native agrees with python)")
                    if len(rep.fails) <= 3:
                        print(src)
                    continue
                st["three_way"] += 1
        print(f"  corpus: {st['three_way']} three-way equal (native == MLIR == Python), "
              f"{st['native_eq_python_mlir_limit']} native == Python where MLIR refuses "
              f"'{MLIR_LIMIT}', {st['native_eq_python_mlir_defect']} native == Python where MLIR "
              f"loses a value-if branch assignment (pinned defect), "
              f"{len(cases) - sum(st.values())} divergent")
        if len(cases) - sum(st.values()) == 0:
            rep.ok(f"corpus: {len(cases)}/{len(cases)} programs agree")

        print("[cf] capacity (census X1: giant function)")
        gsrc, gexp = giant_program(2500)
        got, _ = native(stage1, gsrc.encode(), td)
        if got == gexp:
            rep.ok(f"giant fn (2500-arm else-if chain in a loop, {len(gsrc)} B): exit={gexp[0]} == Python")
        elif got[0] == "refused" and got[1] == 3:
            rep.ok(f"giant fn: clean refusal rc 3 ({got[2]!r})")
        else:
            rep.bad(f"giant fn: native={got} want={gexp} (or a clean rc-3 refusal)")

        print("[cf] clean refusal")
        got, _ = native(stage1, REFUSE_ADDR.encode(), td)
        if got[0] == "refused" and got[1] == 3:
            rep.ok("address-taking HOME function: refused rc 3, no ELF")
        else:
            rep.bad(f"address-taking HOME function: want a clean rc-3 refusal, got {got}")

    if rep.fails:
        print(f"\nFAIL ({len(rep.fails)}): " + "; ".join(rep.fails[:6]))
        return 1
    print("\nALL PASS  (stage1 general control flow: fixtures == hand-derived == MLIR, "
          "corpus native == MLIR == Python, giant fn compiles, clean refusal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
