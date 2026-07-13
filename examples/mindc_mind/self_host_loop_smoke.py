"""
Self-host LOOP gate (Rust-independence #14, roadmap A7) — the PERMANENT proof that
MIND compiles MIND with Rust+LLVM out of the loop, for the scalar-i64 subset that
the pure-MIND compiler (main.mind) is written in.

The claim, gated fail-closed here:

  stage0 (Rust `.so`, the one-time foreign seed) emits stage1.elf — a static native
  x86-64 ELF whose entry is the pure-MIND self-host driver (selfhost_driver.mind).
  Running stage1.elf on the SAME seeded source (with ONLY read/write/exit syscalls —
  no rustc, no LLVM, no clang, no dynamic linker) emits stage2; running stage2 emits
  stage3. This gate asserts stage1 == stage2 == stage3, byte-identical.

The seeded source is  [8B user_lo LE][8B src_len LE][ 21 std/*.mind ++ main.mind ++
selfhost_driver.mind ]  on stdin (fd 0); the ELF is written to stdout (fd 1). main.mind
is NOT modified — the driver is a separate appended shim, so the mic@1 fixed-point and
mic@3-flip gates are untouched.

FAIL-CLOSED (never skips when asked to run):
  * MINDC_SO unset/missing  -> BLOCKED exit 2  (like mic3_flip_smoke.py)
  * stage0 emits an empty / non-ELF image        -> FAIL exit 1
  * stageN.elf exits non-zero or emits nothing    -> FAIL exit 1
  * stage1 != stage2 or stage2 != stage3          -> FAIL exit 1
  * fresh stage1 != frozen bootstrap fixture (A6) -> FAIL exit 1  (source drifted;
        re-freeze testdata/selfhost_loop/stage1.elf + MANIFEST.txt in the same change)

Run:
  MINDC_SO=/path/to/libmindc_mind.so python3 examples/mindc_mind/self_host_loop_smoke.py
"""

import ctypes
import hashlib
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
# MINDC_SO (CI) verbatim; else build the self-host .so FRESH — never trust a
# stale in-tree libmindc_mind.so (a cargo build does not regenerate it).
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

_FROZEN = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
_FROZEN_MANIFEST = _HERE / "testdata" / "selfhost_loop" / "MANIFEST.txt"


def build_seed() -> tuple[bytes, bytes, int]:
    """Return (combined_source, stdin_image, user_lo). combined_source is the exact
    byte stream compiled; stdin_image is [8B user_lo][8B src_len][combined_source]."""
    std_dir = _REPO / "std"
    std_blob = b"\n".join(
        (std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES
    ) + b"\n"
    main = (_HERE / "main.mind").read_bytes()
    driver = (_HERE / "selfhost_driver.mind").read_bytes()
    combined = std_blob + main + b"\n" + driver
    user_lo = len(std_blob)
    stdin_image = struct.pack("<qq", user_lo, len(combined)) + combined
    return combined, stdin_image, user_lo


def stage0_emit(combined: bytes, user_lo: int) -> bytes:
    """Rust `.so` (the one-time foreign seed) emits stage1.elf from the source."""
    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_u.restype = ctypes.c_int64
    lib.selftest_native_elf_u.argtypes = [ctypes.c_int64] * 3
    buf = ctypes.create_string_buffer(combined, len(combined))
    es = lib.selftest_native_elf_u(
        ctypes.cast(buf, ctypes.c_void_p).value, len(combined), user_lo
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf_path: pathlib.Path, stdin_image: bytes) -> bytes:
    """Run the native ELF with the seeded stdin; return its stdout (the emitted ELF).
    Raises on non-zero exit or empty output (fail-closed)."""
    r = subprocess.run(
        [str(elf_path)], input=stdin_image, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{elf_path.name} exited {r.returncode} (signal/segfault?) — "
            f"stderr={r.stderr[:200]!r}"
        )
    if not r.stdout:
        raise RuntimeError(f"{elf_path.name} emitted no bytes")
    return r.stdout


def is_static_elf(b: bytes) -> bool:
    return len(b) > 4096 and b[:4] == b"\x7fELF" and b[16:18] == b"\x02\x00"  # ET_EXEC


def main() -> int:
    if not SO.exists():
        print(f"BLOCKED: {SO} not found (build the self-host .so first). This gate "
              f"does not skip — set MINDC_SO to a driver-capable libmindc_mind.so.")
        return 2

    combined, stdin_image, user_lo = build_seed()
    print(f"[self-host loop] combined={len(combined)}B user_lo={user_lo} "
          f"seed={len(stdin_image)}B  so={SO.name}")

    # stage0 (Rust .so, one-time foreign seed) -> stage1.elf
    stage1 = stage0_emit(combined, user_lo)
    if not is_static_elf(stage1):
        print(f"  FAIL  stage0 emitted a non-ELF/empty image ({len(stage1)}B) — "
              f"nb_trace_hash may have failed closed, or the driver entry is missing.")
        return 1
    h1 = hashlib.sha256(stage1).hexdigest()
    print(f"  stage1 (Rust .so seed): {len(stage1)}B sha256={h1}")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        p1 = tmp / "stage1.elf"
        p1.write_bytes(stage1)
        p1.chmod(p1.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        try:
            # stage1 -> stage2  (Rust+LLVM OUT of the loop: pure native execution)
            stage2 = run_elf(p1, stdin_image)
            p2 = tmp / "stage2.elf"
            p2.write_bytes(stage2)
            p2.chmod(p2.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            # stage2 -> stage3
            stage3 = run_elf(p2, stdin_image)
        except RuntimeError as e:
            print(f"  FAIL  {e}")
            return 1

    h2 = hashlib.sha256(stage2).hexdigest()
    h3 = hashlib.sha256(stage3).hexdigest()
    print(f"  stage2 (stage1 run natively): {len(stage2)}B sha256={h2}")
    print(f"  stage3 (stage2 run natively): {len(stage3)}B sha256={h3}")

    if not (stage1 == stage2 == stage3):
        print("  FAIL  self-host loop NOT closed:")
        if stage1 != stage2:
            print(f"        stage1 != stage2 (sizes {len(stage1)} vs {len(stage2)})")
        if stage2 != stage3:
            print(f"        stage2 != stage3 (sizes {len(stage2)} vs {len(stage3)})")
        return 1

    print(f"  PASS  stage1 == stage2 == stage3 BYTE-IDENTICAL ({len(stage1)}B, "
          f"sha256={h1}) — MIND compiles MIND, Rust+LLVM out of the loop (scalar subset)")

    # Drift check against the frozen bootstrap fixture (A6).
    if _FROZEN.exists():
        frozen = _FROZEN.read_bytes()
        if frozen != stage1:
            print(f"  FAIL  fresh stage1 ({h1}) != frozen bootstrap "
                  f"{hashlib.sha256(frozen).hexdigest()} — source drifted; re-freeze "
                  f"testdata/selfhost_loop/stage1.elf + MANIFEST.txt in this change.")
            return 1
        print(f"  PASS  fresh stage1 == frozen bootstrap fixture (A6), "
              f"testdata/selfhost_loop/stage1.elf")
    else:
        print(f"  NOTE  no frozen bootstrap fixture at {_FROZEN} (freeze it to pin "
              f"the shipped bootstrap ELF).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
