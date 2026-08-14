"""
Self-host mic@3 note-emit gate for `EXPR as i64` CAST expressions (#309, RI-D
native-ELF frontier slice, #110/#40).

Before this port, an `as i64` cast node had no mic@3 note descriptor in the
pure-MIND whole-module emitter, so a user program containing a widening or
float->int cast FAILED CLOSED in the note path. This slice adds a self-contained
`__mind_conv_i64` conv-call descriptor family (kind-11), mirroring the proven
vec_new/vec_push synthetic-name pattern: `X as i64` is emitted as
`CALL __mind_conv_i64([X])` — byte-identical to how the Rust `--emit-mic3` oracle
decodes an i64 cast. main.mind's own source uses `as` only in stripped comments
(no real cast nodes), so the new arms never fire during self-compile and the
whole-module mic3_flip stays byte-identical — THIS fixture is the only regression
coverage for them.

Scope note: `as i64` (width 64) is the slice crux. u8/i8/i16/i32 narrowing
desugars to shifts/masks in the parser and `as f64` refuses (fail-closed) — both
out of scope here. The u8-narrow-LET truncation-mask note gap (fixture #2 in the
dev report) is a SEPARATE pre-existing slice, not a cast defect.

Two gates per fixture:
  (1) EXECUTION-CORRECTNESS — compile with the live Rust `mindc build --emit=binary`
      and RUN it; exit code must be the truncated-toward-zero cast result (mod 256).
  (2) mic@3 NOTE BYTE-IDENTITY — `selftest_mic3_module_nfn(<fn src>)` must equal
      `mindc --emit-mic3 <fn src>` byte-for-byte.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3.12 examples/mindc_mind/self_host_cast_mic3_smoke.py
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

_P = ctypes.POINTER(ctypes.c_int64)


def _i64(addr, off=0):
    return int(ctypes.cast(addr + off, _P)[0])


def _read_string_record(handle):
    if not handle:
        return b""
    addr, length = _i64(handle, 0), _i64(handle, 8)
    if not addr or not length:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def oracle_mic3(src):
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "m.mind"
        op = pathlib.Path(td) / "m.mic3"
        sp.write_text(src)
        subprocess.run([str(MINDC), "--emit-mic3", str(op), str(sp)], capture_output=True)
        return op.read_bytes() if op.exists() else None


def nfn_mic3(src):
    lib = ctypes.CDLL(str(SO))
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5
    sb = src.encode()
    sc = ctypes.create_string_buffer(sb, len(sb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
    cc = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(sc, ctypes.c_void_p).value,
        len(sb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(cc, ctypes.c_void_p).value,
    )
    return _read_string_record(_i64(es, 0)) if es else b""


import stat

# stdlib blob in the FIXED self-host module order — the native-ELF entry expects
# `combined = std_blob ++ user_src` with user_lo = len(std_blob) (mind-runtime NOT
# required: selftest_native_elf_u emits a standalone ELF, unlike `mindc build`).
STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring", "json", "map",
    "net", "process", "reactor", "regex", "ring", "sha256", "string", "time", "toml",
    "tui", "vec",
]
_REPO = _HERE.parents[1]


def std_blob() -> bytes:
    return b"\n".join((_REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


_LIB = ctypes.CDLL(str(SO)) if SO.exists() else None
if _LIB is not None:
    _LIB.selftest_native_elf_u.restype = ctypes.c_int64
    _LIB.selftest_native_elf_u.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]


def run_native(prog: str, run_timeout: int = 8):
    """Emit a standalone native ELF via the pure-MIND selftest_native_elf_u (zero
    mind-runtime, zero MLIR) over std_blob++prog, run it, return the exit code.
    None on fail-closed (empty ELF)."""
    combined = std_blob() + prog.encode()
    user_lo = len(std_blob())
    sb = ctypes.create_string_buffer(combined, len(combined))
    es = _LIB.selftest_native_elf_u(
        ctypes.cast(sb, ctypes.c_void_p).value, len(combined), user_lo
    )
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


# (name, fn_src, expected_exit mod 256) — each fn returns an `EXPR as i64`.
FIXTURES = [
    (
        "int_literal_as_i64",
        "fn compute() -> i64 { return 5 as i64; }\n",
        5,
    ),
    (
        "f64_pos_trunc_as_i64",
        "fn compute() -> i64 { let a: f64 = 2.75; return a as i64; }\n",
        2,
    ),
    (
        "f64_neg_trunc_as_i64",
        "fn compute() -> i64 { let a: f64 = 0.0 - 2.75; return a as i64; }\n",
        254,  # -2 truncated toward zero, mod 256
    ),
]


def _prog(fn_src):
    return fn_src + "fn main() -> i64 { return compute(); }\n"


def main():
    if not SO.exists():
        print(f"BLOCKED: {SO} not found")
        return 1
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 1

    fails = 0
    for name, fn_src, want in FIXTURES:
        o = oracle_mic3(fn_src)
        m = nfn_mic3(fn_src)
        ol = len(o) if o else -1
        ml = len(m) if m else 0
        byte_ok = (m == o and o is not None)
        rc = run_native(_prog(fn_src))
        exec_ok = (rc == (want & 0xFF))
        ok = byte_ok and exec_ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}  {name}  mic3 nb_len={ml} oracle_len={ol} byte_id={byte_ok}"
            f"  exec rc={rc}(want {want})"
        )
        if not ok:
            fails += 1
            if not byte_ok and o and m:
                k = min(len(o), len(m))
                di = next((i for i in range(k) if o[i] != m[i]), k)
                lo = max(0, di - 6)
                print(f"       first diff @ {di}  nb={list(m[lo:di+10])}  oracle={list(o[lo:di+10])}")

    if fails:
        print(f"FAIL: {fails} as-cast fixture(s) diverged")
        return 1
    print("ALL PASS  (EXPR as i64: mic@3 note byte-identical to --emit-mic3 AND native-ELF run-correct via selftest_native_elf_u)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
