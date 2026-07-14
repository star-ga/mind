"""
Self-host NATIVE-ELF BLAS int16 dot smoke (RI-D2 S-E2) — proves the pure-MIND
front-end lowers a `__mind_blas_dot_i16_v(a, b, n)` BLAS extern call to native
x86-64 ELF with ZERO MLIR/LLVM and NO C-runtime link — the int16 sibling of
S-E1's `__mind_blas_dot_q16_v` inline dispatch (the native analogue of MLIR
Track-B's inline dispatch).

nb_expr's ast_call arm now name-matches `__mind_blas_dot_i16_v` and INLINES the
int16 MAC loop (nb_emit_blas_dot_i16) instead of emitting a `call` to an
unresolved external symbol. The inlined reduction is byte-identical to the frozen
nb_i16dot_body encoder: per element `sext16(a[i]) * sext16(b[i])` (movsx WORD
load), i64 accumulate, final narrow ONCE to i32 (`acc as i32`), NO Q16 rescale.
Array element stride is 8 bytes with the value in the low i16 (movsx-word loaded),
matching the dot-i16 canary layout.

Two gates:
  (1) BYTE-EXACT small-vector cross-check — for several signed/zero vectors we
      build a tiny fixture whose `main` fills two int16 arrays (i16 in the low
      16 bits of 8-byte cells, stride 8), calls `__mind_blas_dot_i16_v`, and
      writes the 8-byte LE result to stdout; assert it equals the scalar int16
      reference (ref_dot_i16_scalar: i64 MAC, narrow once to i32).
  (2) CANARY af0fc3cf — a fixture generates the EXACT dot-i16-4096 input at
      runtime (seed 0xDEADBEEF; state = state*1664525 + 1013904223; element =
      (state >> 32) low 16 as i16; a-then-b), calls `__mind_blas_dot_i16_v`,
      writes the 8-byte LE result; assert sha256(stdout) == the committed
      cross-substrate reference for `dot-i16-4096`. This gates the INLINE
      dispatch path against the same byte-identity oracle the standalone S2
      encoder (selftest_native_elf_intdot_i16) hits.

Run:  MINDC_SO=<so> python3 examples/mindc_mind/self_host_native_blas_dot_i16_smoke.py
"""

import ctypes
import hashlib
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

# Committed cross-substrate reference: canary `dot-i16-4096` =
# canonical_hash(result) = sha256 of the 8 LE bytes of the i64 dot result
# (tests/cross_substrate_identity.rs dot_i16_reproducibility_gate).
CANARY = "af0fc3cf1b510f8f7306a5d7250ae25a52b35281a7cefff2a0ac94b0cd80a127"
CANARY_LEN = 4096
_MASK64 = (1 << 64) - 1

# Same 21 bundled stdlib modules, same order, as the native-ELF seeded rung — so
# the fixture's `__mind_*` externs resolve exactly as a real stdlib user's would.
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]


def _std_blob() -> bytes:
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    return b"\n".join(parts) + b"\n"


def _sext(x: int, bits: int) -> int:
    m = 1 << (bits - 1)
    return ((x & ((1 << bits) - 1)) ^ m) - m


def scalar_i16_dot(a: list[int], b: list[int]) -> int:
    """The RI-B2 int16 scalar reference (== Rust ref_dot_i16_scalar): acc(i64) =
    sum(sext16(a[i]) * sext16(b[i])); result = acc narrowed ONCE to i32. No Q16
    shift."""
    acc = 0
    for ai, bi in zip(a, b):
        acc += _sext(ai, 16) * _sext(bi, 16)
    return _sext(acc, 32)  # movsxd(acc): (acc as i32)


def make_pair_i16(length: int, seed: int) -> tuple[list[int], list[int]]:
    """Reproduce dot-i16-4096's LCG (a before b), byte-for-byte the Rust
    Lcg/next_i16: state = state*1664525 + 1013904223 (wrapping u64); element =
    (state >> 32) low 16 bits as i16."""
    state = seed

    def nxt() -> int:
        nonlocal state
        state = (state * 1664525 + 1013904223) & _MASK64
        return _sext((state >> 32) & 0xFFFF, 16)

    a = [nxt() for _ in range(length)]
    b = [nxt() for _ in range(length)]
    return a, b


def _lit(v: int) -> str:
    # main.mind's native front-end has no unary-minus / negative-literal surface
    # (it uses the `0 - N` idiom everywhere); a bare `-N` literal crashes the
    # compile, so emit negatives the same way (mirrors the S-E1 smoke).
    return f"{v}" if v >= 0 else f"0 - {-v}"


def fixture_source_explicit(a: list[int], b: list[int]) -> str:
    """A user fixture whose `main` fills two int16 arrays (i16 in the low 16 bits
    of 8-byte cells, stride 8 — the canary layout) and calls the BLAS extern
    `__mind_blas_dot_i16_v` directly, then writes the 8-byte LE result to stdout."""
    n = len(a)
    lines = ["fn main() -> i64 {"]
    lines.append(f"    let a: i64 = __mind_alloc({n} * 8);")
    for i, v in enumerate(a):
        lines.append(f"    __mind_store_i64(a + {i} * 8, {_lit(v)});")
    lines.append(f"    let b: i64 = __mind_alloc({n} * 8);")
    for i, v in enumerate(b):
        lines.append(f"    __mind_store_i64(b + {i} * 8, {_lit(v)});")
    lines.append(f"    let r: i64 = __mind_blas_dot_i16_v(a, b, {n});")
    lines.append("    let rbuf: i64 = __mind_alloc(8);")
    lines.append("    __mind_store_i64(rbuf, r);")
    lines.append("    __mind_write(1, rbuf, 8);")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def fixture_source_lcg(n: int, seed: int) -> str:
    """A user fixture that GENERATES the dot-i16-4096 input at RUNTIME via the exact
    LCG (state = state*1664525 + 1013904223; element = state >> 32, whose low 16
    bits are the i16 read by the inline movsx-word load), a before b, then calls
    `__mind_blas_dot_i16_v` and writes the 8-byte LE result. Arena-light: a ~20-line
    source, no 8192-store blowup."""
    lines = ["fn main() -> i64 {"]
    lines.append(f"    let n: i64 = {n};")
    lines.append("    let a: i64 = __mind_alloc(n * 8);")
    lines.append("    let b: i64 = __mind_alloc(n * 8);")
    lines.append(f"    let state: i64 = {seed};")
    # Distinct index vars per loop (i then j): the native front-end mis-lowers a
    # single index var RESET and reused across two while loops (empty-ELF fail-closed
    # — a separate front-end while-carry limitation, orthogonal to this inline dot),
    # while distinct counters lower correctly.
    lines.append("    let i: i64 = 0;")
    lines.append("    while i < n {")
    lines.append("        state = state * 1664525 + 1013904223;")
    lines.append("        __mind_store_i64(a + i * 8, state >> 32);")
    lines.append("        i = i + 1;")
    lines.append("    }")
    lines.append("    let j: i64 = 0;")
    lines.append("    while j < n {")
    lines.append("        state = state * 1664525 + 1013904223;")
    lines.append("        __mind_store_i64(b + j * 8, state >> 32);")
    lines.append("        j = j + 1;")
    lines.append("    }")
    lines.append("    let r: i64 = __mind_blas_dot_i16_v(a, b, n);")
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
    p = tmp / "blas_dot_i16.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], capture_output=True).stdout


VECTORS = [
    # name, a, b — mix of positive, negative and zero (movsx-word sign-extend).
    ("small_pos", [1, 2, 3], [4, 5, 6]),
    ("neg_mix", [-32768, 100, -7], [3, -32768, 32767]),
    ("zeros", [0, 0, 0, 0], [1, 2, 3, 4]),
    ("full_range", [32767, -32768, 12345, -6789], [-1, 1, -32768, 32767]),
    # 16-element deterministic ramp with alternating sign — larger loop trip count.
    (
        "ramp16",
        [((-1) ** i) * ((i * 137 + 11) % 32768) for i in range(16)],
        [((i * 91 + 7) % 65536) - 32768 for i in range(16)],
    ),
]


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0
    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3

    print("[RI-D2 S-E2: native-ELF __mind_blas_dot_i16_v inline dispatch]")
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        # --- Gate (1): byte-exact small-vector cross-check ---
        for name, a, b in VECTORS:
            expected = scalar_i16_dot(a, b)
            elf = build_elf(lib, fixture_source_explicit(a, b))
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
                f"  {'PASS' if ok else 'FAIL'}  {name}: dot_i16_v(n={len(a)}) native-ELF "
                f"result {got} == scalar int16 reference {expected}  ({len(elf)} B ELF)"
            )
            if not ok:
                print(f"        stdout bytes: {out.hex()}  expected: {struct.pack('<q', expected).hex()}")
                return 1

        # --- Gate (2): canary af0fc3cf via runtime-LCG dot-i16-4096 ---
        a4, b4 = make_pair_i16(CANARY_LEN, 0xDEADBEEF)
        py_ref = scalar_i16_dot(a4, b4)
        py_sha = hashlib.sha256(struct.pack("<q", py_ref)).hexdigest()
        if py_sha != CANARY:
            print(f"  FAIL  python reference sha {py_sha} != canary {CANARY} (LCG port drift)")
            return 1
        elf = build_elf(lib, fixture_source_lcg(CANARY_LEN, 0xDEADBEEF))
        if len(elf) == 0:
            print("  FAIL  canary: native-ELF pipeline returned empty ELF (failed closed)")
            return 1
        out = run_elf(elf, tmp)
        if len(out) != 8:
            print(f"  FAIL  canary: expected 8 stdout bytes, got {len(out)}: {out.hex()}")
            return 1
        got_val = int.from_bytes(out, "little", signed=True)
        got_sha = hashlib.sha256(out).hexdigest()
        ok = got_sha == CANARY
        all_ok = all_ok and ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  canary dot-i16-4096: native-ELF inline result "
            f"{got_val}  (LE bytes {out.hex()}, {len(elf)} B ELF)"
        )
        print(f"        native sha256 = {got_sha}")
        print(f"        canary        = {CANARY}")
        if not ok:
            return 1

    if all_ok:
        print(
            "\nALL PASS  (__mind_blas_dot_i16_v call lowers to native x86-64 ELF, byte-exact "
            "int16 reduction, dot-i16-4096 == canary af0fc3cf, zero MLIR/LLVM, no C-runtime link)"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
