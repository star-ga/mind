"""
Self-host NATIVE-ELF BLAS Q16 dot smoke (RI-D2 S-E1) — proves the pure-MIND
front-end lowers a REAL `std/blas.mind` Q16.16 dot-product call
(`dot_q16_v(a, b, n)`, whose body is the `__mind_blas_dot_q16_v` BLAS extern)
to native x86-64 ELF with ZERO MLIR/LLVM and NO C-runtime link — the native
analogue of MLIR Track-B's inline dispatch.

nb_expr's ast_call arm now name-matches `__mind_blas_dot_q16_v` and INLINES the
Q16.16 MAC loop (nb_emit_blas_dot_q16) instead of emitting a `call` to an
unresolved external symbol. The inlined reduction is byte-identical to the
frozen nb_q16dot_body encoder: per product `sext32(a[i]) * sext32(b[i])`, then
arithmetic `>> 16`, i64 accumulate, final narrow ONCE to i32 (`acc as i32`).
Array element stride is 8 bytes with the value in the low i32 (movsxd-loaded),
matching the dot-l2-q16 canary layout.

Gate (BYTE-EXACT — Q16 is bit-exact): for several known Q16 arrays we build a
tiny fixture whose `main` calls `dot_q16_v`, compile it through the pure-MIND
native-ELF pipeline (selftest_native_elf_u, stdlib-seeded so blas.mind's wrapper
+ extern resolve exactly as a real `use std.blas` user), run the ELF, read the
8-byte little-endian result from stdout, and assert it equals the scalar Q16
reference computed here the same way the RI-B2 Q16 scalar oracle does. A single
wrong byte in the reduction fails the compare.

Run:  MINDC_SO=<so> python3 examples/mindc_mind/self_host_native_blas_dot_q16_smoke.py
"""

import ctypes
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# Same 21 bundled stdlib modules, same order, as the native-ELF seeded rung.
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]


def _std_blob() -> bytes:
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    return b"\n".join(parts) + b"\n"


def _sext32(x: int) -> int:
    return ((x + 2**31) % 2**32) - 2**31


def scalar_q16_dot(a: list[int], b: list[int]) -> int:
    """The RI-B2 Q16 scalar reference: acc(i64) = sum((sext32(a[i]) *
    sext32(b[i])) >> 16); result = acc narrowed ONCE to i32. Python `>>` on ints
    is arithmetic, matching x86 SAR."""
    acc = 0
    for ai, bi in zip(a, b):
        acc += (_sext32(ai) * _sext32(bi)) >> 16
    return _sext32(acc)  # movsxd(acc): (acc as i32)


def fixture_source(a: list[int], b: list[int]) -> str:
    """A user fixture whose `main` fills two Q16 arrays (i32 in the low 32 bits of
    8-byte cells, stride 8 — the canary layout) and calls the std wrapper
    `dot_q16_v`, then writes the 8-byte LE result to stdout. Uses only the
    bundled stdlib surface (dot_q16_v + the __mind_* arena/io externs)."""
    n = len(a)

    def lit(v: int) -> str:
        # main.mind's native front-end has no unary-minus / negative-literal
        # surface (it uses the `0 - N` idiom everywhere); a bare `-N` literal
        # crashes the compile, so emit negatives the same way.
        return f"{v}" if v >= 0 else f"0 - {-v}"

    lines = ["fn main() -> i64 {"]
    lines.append(f"    let a: i64 = __mind_alloc({n} * 8);")
    for i, v in enumerate(a):
        # Store the raw i32 value (as an i64); the inline reads the low i32 sx.
        lines.append(f"    __mind_store_i64(a + {i} * 8, {lit(v)});")
    lines.append(f"    let b: i64 = __mind_alloc({n} * 8);")
    for i, v in enumerate(b):
        lines.append(f"    __mind_store_i64(b + {i} * 8, {lit(v)});")
    lines.append(f"    let r: i64 = dot_q16_v(a, b, {n});")
    lines.append("    let rbuf: i64 = __mind_alloc(8);")
    lines.append("    __mind_store_i64(rbuf, r);")
    lines.append("    __mind_write(1, rbuf, 8);")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_elf(lib, source: str) -> bytes:
    std_blob = _std_blob()
    combined = std_blob + source.encode()
    user_lo = len(std_blob)
    sb = ctypes.create_string_buffer(combined, len(combined))
    es = lib.selftest_native_elf_u(
        ctypes.cast(sb, ctypes.c_void_p).value, len(combined), user_lo
    )
    rd = lambda addr, off=0: ctypes.cast(addr + off, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    n = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), n) if n > 0 else b""


def run_elf(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "blas_dot.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], capture_output=True).stdout


# Q16.16 helpers: 1.0 == 65536.
_Q = 65536
VECTORS = [
    # name, a, b — mix of positive, fractional, and NEGATIVE (SAR + sign-extend).
    ("ones_x_ramp", [_Q, 2 * _Q, 3 * _Q], [_Q, _Q, _Q]),
    ("frac_half", [_Q // 2, _Q // 2], [_Q, 2 * _Q]),
    ("neg_mix", [-2 * _Q, _Q, -_Q], [3 * _Q, _Q, -4 * _Q]),
    ("zeros", [0, 0, 0, 0], [_Q, 2 * _Q, 3 * _Q, 4 * _Q]),
    # 16-element deterministic ramp with alternating sign — larger loop trip count.
    (
        "ramp16",
        [((-1) ** i) * ((i + 1) * _Q // 4) for i in range(16)],
        [((i * 7 + 3) % 11) * _Q // 3 for i in range(16)],
    ),
]


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0
    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3

    print("[RI-D2 S-E1: native-ELF __mind_blas_dot_q16_v inline dispatch]")
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, a, b in VECTORS:
            expected = scalar_q16_dot(a, b)
            elf = build_elf(lib, fixture_source(a, b))
            if len(elf) == 0:
                print(f"  FAIL  {name}: native-ELF pipeline returned empty ELF (failed closed)")
                return 1
            out = run_elf(elf, tmp)
            if len(out) != 8:
                print(f"  FAIL  {name}: expected 8 stdout bytes, got {len(out)} ({out!r})")
                return 1
            got = struct.unpack("<q", out)[0]
            ok = got == expected
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name}: dot_q16_v(n={len(a)}) native-ELF "
                f"result {got} == scalar Q16 reference {expected}  ({len(elf)} B ELF)"
            )
            if not ok:
                print(f"        stdout bytes: {out.hex()}  expected: {struct.pack('<q', expected).hex()}")
                return 1
    if all_ok:
        print(
            "\nALL PASS  (real std/blas.mind dot_q16_v call lowers to native x86-64 ELF, "
            "byte-exact Q16 reduction, zero MLIR/LLVM, no C-runtime link)"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
