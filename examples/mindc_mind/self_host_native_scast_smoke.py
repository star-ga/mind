"""
Self-host NATIVE-ELF SIGNED narrowing-cast smoke (RI-E / #111, #40 remaining leg).

Proves the pure-MIND front-end (main.mind) lowers `EXPR as i8/i16/i32` — the
SIGNED narrowing cast — to a correct running native x86-64 ELF, through the FULL
self-host path (`selftest_native_elf_u`, self-computed mic@3 trace-hash note,
ZERO Rust/LLVM in the emit chain).

Gap it closes
-------------
A signed narrowing cast previously parsed into a real `ast_cast` node lowered by
a reg-to-reg movsx. That node RUNS in the native-ELF path but the mic@3 NOTE
builder (`nb_build_mic3`) REFUSES it (0-byte poison) — so a program using
`as iN` fail-closed through the `_u` self-host path even though it ran natively.
This smoke exercises the parse-time DESUGAR `EXPR as iN` -> `(EXPR << (64-N)) >> (64-N)`.
MIND's `>>` is an ARITHMETIC (SAR) shift, so this pair sign-extends the low N
bits — exactly the movsx wrap — through two fully-proven shift binops that lower
through BOTH the native path AND the note builder, so a signed cast now self-hosts
through the whole `_u` path.

Oracle model
------------
The deleted Rust `src/native` backend (#15) leaves NO frozen native-ELF oracle
for this construct, so — exactly like the float / unsigned-cast smokes — the CPU
is the oracle: the pure-MIND-emitted ELF must RUN and exit with the correct value,
AND each fixture is cross-checked against the LIVE Rust `mindc build --emit=binary`
cpu backend, so the SIGNED-narrowing SEMANTICS (sign-extend, not zero-extend) are
oracle-confirmed by execution, not merely self-consistent.

Every fixture arithmetic-right-shifts the cast result before returning it: a
SIGNED cast sign-extends, so `(200 as i8) >> 4` = -4 (exit 252), whereas an
UNSIGNED cast would zero-extend to `(200 as u8) >> 4` = 12 — the shift makes zero-
vs sign-extension observable in the 8-bit exit code, so a green run genuinely
distinguishes the signed cast from the unsigned one.

Run:  MINDC_SO=<so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_native_scast_smoke.py
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

# (source, expected_exit, description). Each result is arithmetic-right-shifted
# so signed (sign-extend) vs unsigned (zero-extend) is observable in the exit code.
CASES = [
    (
        b"fn main() -> i64 {\n    let x: i64 = 200;\n    return (x as i8) >> 4;\n}\n",
        252,
        "(200 as i8) = -56, >>4 = -4 -> exit 252  (unsigned u8 would give 12)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 0 - 1;\n    return (x as i8) >> 1;\n}\n",
        255,
        "(-1 as i8) = -1, >>1 = -1 -> exit 255  (sign-extend, not zero-extend)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 40000;\n    return (x as i16) >> 12;\n}\n",
        249,
        "(40000 as i16) = -25536, >>12 = -7 -> exit 249  (unsigned u16 would give 9)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 0 - 3;\n    return (x as i16) >> 12;\n}\n",
        255,
        "(-3 as i16) = -3, >>12 = -1 -> exit 255  (unsigned u16 would give 15)",
    ),
    (
        b"fn main() -> i64 {\n    let x: i64 = 3000000000;\n    return (x as i32) >> 28;\n}\n",
        251,
        "(3000000000 as i32) = -1294967296, >>28 = -5 -> exit 251  (unsigned u32 would give 11)",
    ),
    # PRECEDENCE regression guard (UNPARENTHESIZED): `as` binds tighter than `+`
    # (Rust-lang order). `a + b as i8` MUST be `a + (b as i8)` on BOTH front-ends.
    # pure-MIND consumes `as` in postfix position (always tight); the live Rust mindc
    # cross-oracle catches the parser bug where `as` lbp=7 was looser than additive
    # (fixed to lbp=15 in src/parser/mod.rs). A loose parse would give 250, not 122.
    (
        b"fn main() -> i64 {\n    let a: i64 = 300;\n    let b: i64 = 200;\n    return (a + b as i8) >> 1;\n}\n",
        122,
        "a + b as i8 = a + (b as i8) = 300 + (-56) = 244, >>1 = 122  (loose-parse bug = 250)",
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
        print("ALL PASS  (pure-MIND native-ELF signed cast self-hosts through _u "
              "and matches the live mindc value-oracle)")
        return 0
    print("FAILURES PRESENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
