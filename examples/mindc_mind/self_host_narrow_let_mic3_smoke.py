"""
Self-host mic@3 note-emit gate for the narrow UNSIGNED let INIT mask (#312, RI-D).

`let x: u8/u16/u32 = E` — the Rust oracle masks the init to the declared width
(E & (2^w-1)); the pure-MIND whole-module note builder omitted it, so the note
diverged from `mindc --emit-mic3` (native-ELF exec was already correct — the
native path masks via nb_width_of_ann). The fix desugars the narrow-unsigned let
init to the proven `E & mask` binop (mirror of the `as uN` parser desugar) in
flatten_stmt_seq's plain-let path and flattens the masked init, so binding + note
are byte-identical. main.mind uses narrow lets only in comments, so this is inert
on self-compile and the whole-module mic3_flip stays byte-identical — THIS fixture
is the only regression coverage.

SCOPE: the narrow-let INIT mask only. ORTHOGONAL, still-open follow-ups (excluded
here, byte-decode-proven separate): (1) narrow-typed ARITHMETIC results — `a + 1`
where a:u8 lowers to `(a+1) & mask` in the oracle and still diverges in the note;
(2) signed narrow (i8/i16/i32 sign-extend). So fixtures avoid narrow arithmetic on
the masked value.

Two gates per fixture:
  (1) mic@3 NOTE BYTE-IDENTITY — selftest_mic3_module_nfn(src) == mindc --emit-mic3 src.
  (2) NATIVE-ELF EXEC — selftest_native_elf_u(std ++ prog) runs, exits the masked value.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3.12 examples/mindc_mind/self_host_narrow_let_mic3_smoke.py
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


# (name, fn_body_returning_the_masked_value, expected_exit mod 256). No narrow arith
# on the masked value (that is the orthogonal follow-up gap).
FIXTURES = [
    ("u8_let",      "fn compute() -> i64 { let x: u8 = 200; return x; }\n", 200),
    ("u8_neg_wrap", "fn compute() -> i64 { let x: u8 = 0 - 1; return x; }\n", 255),
    ("u16_let",     "fn compute() -> i64 { let x: u16 = 40000; return x; }\n", 40000 & 0xFF),
    ("u32_let",     "fn compute() -> i64 { let x: u32 = 3000000000; return x; }\n", 3000000000 & 0xFF),
    ("two_narrow_ret_first",
     "fn compute() -> i64 { let a: u8 = 200; let b: u16 = 500; return a; }\n", 200),
    ("narrow_then_widen",
     "fn compute() -> i64 { let x: u8 = 200; let y: i64 = x; return y; }\n", 200),
]


def _prog(fn_src):
    return fn_src + "fn main() -> i64 { return compute(); }\n"


def main():
    if not SO or not pathlib.Path(SO).exists():
        print(f"BLOCKED: .so not found ({SO})")
        return 1
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 1
    fails = 0
    for name, fn_src, want in FIXTURES:
        o = oracle_mic3(fn_src)
        m = nfn_mic3(fn_src)
        ol = len(o) if o else -1
        ml = len(m)
        byte_ok = (m == o and o is not None)
        rc = run_native(_prog(fn_src))
        exec_ok = (rc == (want & 0xFF))
        ok = byte_ok and exec_ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  nfn={ml} oracle={ol} byte_id={byte_ok}"
              f"  exec rc={rc}(want {want & 0xFF})")
        if not ok:
            fails += 1
            if not byte_ok and o and m:
                k = min(len(o), len(m))
                di = next((i for i in range(k) if o[i] != m[i]), k)
                lo = max(0, di - 6)
                print(f"       first diff @ {di}  nfn={list(m[lo:di+10])}  oracle={list(o[lo:di+10])}")
    if fails:
        print(f"FAIL: {fails} narrow-let fixture(s) diverged")
        return 1
    print("ALL PASS  (narrow UNSIGNED let init mask: mic@3 note byte-identical to --emit-mic3 "
          "AND native-ELF run-correct via selftest_native_elf_u)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
