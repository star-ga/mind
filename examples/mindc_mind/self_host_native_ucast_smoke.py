"""
Self-host NATIVE-ELF UNSIGNED-CAST smoke (RI-E / #111, #40 remaining leg).

Proves the pure-MIND front-end (main.mind) lowers `EXPR as u8/u16/u32` — the
UNSIGNED narrowing cast — to a correct running native x86-64 ELF, through the
FULL self-host path (`selftest_native_elf_u`, self-computed mic@3 trace-hash
note, ZERO Rust/LLVM in the emit chain).

Gap it closes
-------------
The cast PARSER (`narrow_cast_width`) previously recognised only `i8/i16/i32`
(+ `i64`/`f64`); ANY `as u8/u16/u32` produced a fail-closed poison node -> 0-byte
refusal. This smoke exercises the parse-time DESUGAR `EXPR as uN` -> `EXPR & mask`
(zero-extended truncation; the low N bits kept, high bits cleared — exact for
every i64 input, negatives included). The AND binop is fully proven and — unlike
the signed `ast_cast` node, which the mic@3 NOTE builder (`nb_build_mic3`)
refuses — lowers through BOTH the native path AND the note builder, so an
unsigned cast self-hosts through the whole `_u` path.

Oracle model
------------
The deleted Rust `src/native` backend (#15) leaves NO frozen native-ELF oracle
for a NEW construct, so — exactly like the float smokes — the CPU is the oracle:
the pure-MIND-emitted ELF must RUN and exit with the correct value. Each fixture
is ALSO cross-checked against the LIVE Rust `mindc build --emit=binary` cpu
backend, so the unsigned-narrowing SEMANTICS (zero-extend, not sign-extend) are
oracle-confirmed by execution, not merely self-consistent.

The exit code is 8-bit, so every fixture SHIFTS the cast result down before
returning it — a signed `iN` cast would sign-extend and produce a DIFFERENT low
byte (e.g. `(200 as u8) >> 4` = 12, but `(200 as i8) >> 4` = 252), so a green
run genuinely distinguishes zero- from sign-extension.

Run:  MINDC_SO=<so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_native_ucast_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()
MINDC = os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc"))

# (source, expected_exit, description). The result is always shifted below the
# 8-bit exit window so zero- vs sign-extension is observable in the exit code.
CASES = [
    (
        b"fn main() -> i64 {\n    let x: i64 = 200;\n    return (x as u8) >> 4;\n}\n",
        12,
        "200 as u8 = 200, >>4 = 12  (signed i8 would give 252)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 0 - 1;\n    return x as u8;\n}\n",
        255,
        "-1 as u8 = 255  (zero-extend, not sign-extend)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 300;\n    return (x as u16) >> 4;\n}\n",
        18,
        "300 as u16 = 300, >>4 = 18",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 70000;\n    return (x as u32) >> 12;\n}\n",
        17,
        "70000 as u32 = 70000, >>12 = 17",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 0 - 5;\n    return (x as u32) >> 24;\n}\n",
        255,
        "-5 as u32 = 4294967291, >>24 = 255  (u32 zero-extend)",
    ),
]


def _load_so() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3
    return lib


def _emit_u(lib: ctypes.CDLL, src: bytes) -> bytes:
    """Pure-MIND native-ELF via the FULL self-host path (self-computed note)."""
    buf = ctypes.create_string_buffer(src, len(src))
    es = lib.selftest_native_elf_u(ctypes.cast(buf, ctypes.c_void_p).value, len(src), 0)
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    if not es:
        return b""
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def _run_elf(elf: bytes) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".elf") as f:
        f.write(elf)
        path = f.name
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    try:
        return subprocess.run([path], timeout=30).returncode
    finally:
        os.unlink(path)


def _run_live(src: bytes) -> int:
    """LIVE Rust mindc cpu backend — the value-semantics cross-oracle."""
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td) / "case.mind"
        b = pathlib.Path(td) / "case.bin"
        s.write_bytes(src)
        r = subprocess.run(
            [MINDC, "build", str(s), "--release", "--emit=binary", "--out", str(b)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )
        if r.returncode != 0 or not b.exists():
            return -1
        b.chmod(b.stat().st_mode | stat.S_IEXEC)
        return subprocess.run([str(b)], timeout=30).returncode


def main() -> int:
    lib = _load_so()
    all_ok = True
    for src, want, desc in CASES:
        elf = _emit_u(lib, src)
        if not elf:
            print(f"[FAIL] POISON (0B) — full-path _u refused: {desc}")
            all_ok = False
            continue
        got = _run_elf(elf)
        live = _run_live(src)
        ok = got == want and live == want
        all_ok = all_ok and ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] pure-MIND _u emit={len(elf):4d}B "
            f"run_exit={got} live_mindc={live} want={want}  # {desc}"
        )
    if all_ok:
        print("ALL PASS  (pure-MIND native-ELF unsigned cast self-hosts through _u "
              "and matches the live mindc value-oracle)")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
