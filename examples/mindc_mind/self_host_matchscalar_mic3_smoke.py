"""
Self-host mic@3 note-emit gate for a TRAILING return-terminated diverging if-ELSE
chain after a leading let — the scalar-int `match` / fieldless-enum-discriminant
desugar (#311, RI-D native-ELF frontier slice, #110/#40).

Before this port, emit_mic3_module_fndef_seq's trailing-expr logic handled a bare
trailing `return E` and a trailing VALUE if-expr (`if c {A} else {B}`), but a
trailing return-terminated diverging if-ELSE STATEMENT (`...; if x==0 {return A;}
else {return B;}`) was neither: is_trail_ret=0, trail_is_if=0, so it fell to
flatten_expr_env(if-stmt) which returns -1 -> fail-closed -> 0-byte ELF from
selftest_native_elf_u. The fix adds a `trail_is_chain` case that emits the
else-nested OP_IF via emit_mic3_ifret_chain_instr_lv (the fndef-path chain emitter,
lv-env clone so the cond can reference the leading let). A statement-form
`match x { 0=>{return A;} .. _=>{return Z;} }` desugars to the SAME diverging
if-else chain, so ONE trailing case closes both scalar-int match and
fieldless-enum-discriminant match.

main.mind uses no such trailing chain (mic3_flip stays byte-identical), so THIS
fixture is the only regression coverage for the new trailing arm.

Two gates per fixture:
  (1) mic@3 NOTE BYTE-IDENTITY — selftest_mic3_module_nfn(<fn src>) == mindc
      --emit-mic3 <fn src> byte-for-byte.
  (2) NATIVE-ELF EXEC — selftest_native_elf_u(std ++ prog) emits a running ELF
      exiting with the arm the scrutinee selects (every arm incl. wildcard).

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3.12 examples/mindc_mind/self_host_matchscalar_mic3_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()
MINDC = pathlib.Path(
    os.environ.get("MINDC_BIN", str(_HERE.parents[1] / "target" / "release" / "mindc"))
)
_REPO = _HERE.parents[1]
_P = ctypes.POINTER(ctypes.c_int64)


def _i64(a, o=0):
    return int(ctypes.cast(a + o, _P)[0])


STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring", "json", "map",
    "net", "process", "reactor", "regex", "ring", "sha256", "string", "time", "toml",
    "tui", "vec",
]


def std_blob():
    return b"\n".join((_REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


_LIB = ctypes.CDLL(str(SO)) if SO and pathlib.Path(SO).exists() else None
if _LIB is not None:
    _LIB.selftest_native_elf_u.restype = ctypes.c_int64
    _LIB.selftest_native_elf_u.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]


def oracle_mic3(src):
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "m.mind"
        op = pathlib.Path(td) / "m.mic3"
        sp.write_text(src)
        subprocess.run([str(MINDC), "--emit-mic3", str(op), str(sp)], capture_output=True)
        return op.read_bytes() if op.exists() else None


def nfn_mic3(src):
    fn = _LIB.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5
    sb = src.encode()
    sc = ctypes.create_string_buffer(sb, len(sb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
    cc = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(sc, ctypes.c_void_p).value, len(sb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(cc, ctypes.c_void_p).value,
    )
    if not es:
        return b""
    sh = _i64(es, 0)
    n = _i64(sh, 8)
    return ctypes.string_at(_i64(sh, 0), n) if n > 0 else b""


def run_native(prog, run_timeout=8):
    comb = std_blob() + prog.encode()
    ulo = len(std_blob())
    sb = ctypes.create_string_buffer(comb, len(comb))
    es = _LIB.selftest_native_elf_u(ctypes.cast(sb, ctypes.c_void_p).value, len(comb), ulo)
    if not es:
        return None
    sh = _i64(es, 0)
    n = _i64(sh, 8)
    elf = ctypes.string_at(_i64(sh, 0), n) if n > 0 else b""
    if len(elf) < 4 or elf[:4] != b"\x7fELF":
        return None
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
        f.write(elf)
        p = f.name
    os.chmod(p, 0o755)
    try:
        return subprocess.run([p], timeout=run_timeout).returncode
    except subprocess.TimeoutExpired:
        return "HANG"
    finally:
        os.unlink(p)


def match_fn(arms, wild):
    body = " ".join(f"{p} => {{ return {v}; }}" for p, v in arms) + f" _ => {{ return {wild}; }}"
    return "fn classify(x: i64) -> i64 {\n    let x2: i64 = x;\n    match x2 { " + body + " }\n}\n"


def ifelse_fn():
    return ("fn classify(x: i64) -> i64 {\n"
            "    let x2: i64 = x;\n"
            "    if x2 == 0 { return 10; } else { return 20; }\n}\n")


# (name, fn_src, [(x, expected_exit), ...])
def build_cases():
    cases = []
    cases.append(("ifelse_let", ifelse_fn(), [(0, 10), (7, 20)]))
    cases.append(("match2_let", match_fn([(1, 10)], 30), [(1, 10), (9, 30)]))
    cases.append(("match3_let", match_fn([(0, 10), (2, 20)], 0), [(0, 10), (2, 20), (5, 0)]))
    cases.append(("match4_let", match_fn([(1, 10), (2, 20), (3, 30)], 40),
                  [(1, 10), (2, 20), (3, 30), (99, 40)]))
    return cases


def main():
    if not SO or not pathlib.Path(SO).exists():
        print(f"BLOCKED: .so not found ({SO})")
        return 1
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 1

    fails = 0
    for name, fn_src, probes in build_cases():
        o = oracle_mic3(fn_src)
        m = nfn_mic3(fn_src)
        ol = len(o) if o else -1
        ml = len(m)
        byte_ok = (m == o and o is not None)
        exec_ok = True
        got = []
        for x, want in probes:
            prog = fn_src + f"fn main() -> i64 {{ return classify({x}); }}\n"
            rc = run_native(prog)
            got.append((x, rc, want))
            if rc != (want & 0xFF):
                exec_ok = False
        ok = byte_ok and exec_ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  nfn={ml} oracle={ol} byte_id={byte_ok}"
              f"  exec={['x%d->%s(want%d)' % g for g in got]}")
        if not ok:
            fails += 1
            if not byte_ok and o and m:
                k = min(len(o), len(m))
                di = next((i for i in range(k) if o[i] != m[i]), k)
                lo = max(0, di - 6)
                print(f"       first diff @ {di}  nfn={list(m[lo:di+10])}  oracle={list(o[lo:di+10])}")

    if fails:
        print(f"FAIL: {fails} matchscalar fixture(s) diverged")
        return 1
    print("ALL PASS  (trailing scalar-int match / diverging if-else after let: mic@3 note "
          "byte-identical to --emit-mic3 AND native-ELF run-correct via selftest_native_elf_u)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
