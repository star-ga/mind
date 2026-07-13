"""
Self-host LOOP gate (Rust-independence #14, roadmap A7 + RI-E1) — the PERMANENT proof
that MIND reproduces its own compiler with Rust+LLVM out of the loop, for the scalar-i64
subset that the pure-MIND compiler (main.mind) is written in.

RI-E1 — reproduction-independence via a checked-in pure-MIND bootstrap stage0
-----------------------------------------------------------------------------
This is the standard GCC/rustc "checked-in stage0" bootstrap model: the frozen
`testdata/selfhost_loop/stage1.elf` IS the bootstrap compiler, and the Rust `.so`
is DEMOTED from seed to a re-freeze / drift oracle.

  * PRIMARY (always, no Rust in the chain): seed stage1 by RUNNING the frozen
    pure-MIND ELF on the seeded stdin — `stage1 = run_elf(FROZEN, stdin)` — then
    stage2 = run_elf(stage1), stage3 = run_elf(stage2). Assert
    stage1 == stage2 == stage3 == frozen. The ONLY syscalls in this chain are
    execve(self)/read/write/exit — zero rustc, zero LLVM, zero clang, zero .so.
  * ORACLE (drift check, when the Rust `.so` is present): assert the FRESH `.so`
    output stage0_emit(combined, user_lo) == frozen. This catches std/*.mind or
    main.mind SOURCE drift where the frozen ELF was not re-blessed. It SOFT-SKIPS
    if the `.so` is unavailable — the primary reproduction path never depends on
    the `.so` being buildable.
  * RESEED (--reseed / MIND_SELFHOST_RESEED=1): the ONLY mode that uses the `.so`
    as the seed. Emits a fresh stage1 via the Rust `.so`, confirms the loop closes,
    and re-freezes testdata/selfhost_loop/{stage1.elf,MANIFEST.txt} — the deliberate
    re-bless path for a source change.

HONEST FRAMING (no overclaim):
  This proves REPRODUCTION-independence — the seed chain is now Rust-free and the
  `.so` is only an oracle. It does NOT claim "mindc builds from scratch with zero
  Rust" nor "LLVM dropped". Two residuals remain, orthogonal and stated plainly:
    (i)  the FIRST frozen stage1.elf was originally minted by the Rust `.so`
         (chicken-and-egg; residual trusting-trust, universal to every bootstrapped
         toolchain — gcc/rustc included);
    (ii) a HARNESS-FREE standalone mindc (its own file-IO/argv/CLI, no Python
         driver) is NOT delivered here — that is the separate C8 + argv/CLI track.

The seeded source is  [8B user_lo LE][8B src_len LE][ 21 std/*.mind ++ main.mind ++
selfhost_driver.mind ]  on stdin (fd 0); the ELF is written to stdout (fd 1). main.mind
is NOT modified — the driver is a separate appended shim, so the mic@1 fixed-point and
mic@3-flip gates are untouched.

FAIL-CLOSED (never skips when asked to run):
  * frozen bootstrap fixture missing                 -> BLOCKED exit 2  (it is the seed/oracle now)
  * running the frozen ELF exits non-zero / emits nothing -> FAIL exit 1
  * stage1 != stage2 or stage2 != stage3             -> FAIL exit 1
  * stage1 (from frozen) != frozen fixture           -> FAIL exit 1  (should be impossible;
        run_elf(frozen) reproduces frozen by construction)
  * .so present AND fresh .so output != frozen        -> FAIL exit 1  (source drifted;
        re-freeze with --reseed in the same change)
  --reseed only:
  * MINDC_SO unset/missing                            -> BLOCKED exit 2  (needs the seed .so)
  * .so emits an empty / non-ELF image                -> FAIL exit 1

Run:
  python3 examples/mindc_mind/self_host_loop_smoke.py                       # PRIMARY + oracle(if .so)
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py    # + .so drift oracle
  MINDC_SO=/path/to/libmindc_mind.so python3 .../self_host_loop_smoke.py --reseed   # re-freeze
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

_RESEED = "--reseed" in sys.argv[1:] or os.environ.get("MIND_SELFHOST_RESEED") == "1"


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
    """Rust `.so` (the DRIFT ORACLE / re-freeze seed) emits an ELF from the source."""
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


def _write_exec(dir_: pathlib.Path, name: str, data: bytes) -> pathlib.Path:
    p = dir_ / name
    p.write_bytes(data)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


def run_loop_from_seed(seed_exe: pathlib.Path, stdin_image: bytes,
                       td: pathlib.Path) -> tuple[bytes, bytes, bytes]:
    """Given an executable seed ELF, produce (stage1, stage2, stage3):
    stage1 = seed(stdin); stage2 = stage1(stdin); stage3 = stage2(stdin)."""
    stage1 = run_elf(seed_exe, stdin_image)
    if not is_static_elf(stage1):
        raise RuntimeError(f"seed emitted a non-ELF/empty image ({len(stage1)}B)")
    p1 = _write_exec(td, "stage1.elf", stage1)
    stage2 = run_elf(p1, stdin_image)
    p2 = _write_exec(td, "stage2.elf", stage2)
    stage3 = run_elf(p2, stdin_image)
    return stage1, stage2, stage3


def do_reseed(combined: bytes, stdin_image: bytes, user_lo: int) -> int:
    """RESEED / re-bless: seed stage1 with the Rust `.so`, confirm the loop closes,
    and re-freeze testdata/selfhost_loop/{stage1.elf,MANIFEST.txt}. The ONLY mode
    that uses the `.so` as a seed. Needs MINDC_SO (fail-closed)."""
    if not SO.exists():
        print(f"BLOCKED: --reseed needs the Rust seed .so; {SO} not found "
              f"(set MINDC_SO to a driver-capable libmindc_mind.so).")
        return 2
    stage1 = stage0_emit(combined, user_lo)
    if not is_static_elf(stage1):
        print(f"  FAIL  .so emitted a non-ELF/empty image ({len(stage1)}B) — "
              f"nb_trace_hash may have failed closed, or the driver entry is missing.")
        return 1
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        p1 = _write_exec(tmp, "stage1.elf", stage1)
        try:
            stage2 = run_elf(p1, stdin_image)
            p2 = _write_exec(tmp, "stage2.elf", stage2)
            stage3 = run_elf(p2, stdin_image)
        except RuntimeError as e:
            print(f"  FAIL  {e}")
            return 1
    if not (stage1 == stage2 == stage3):
        print("  FAIL  --reseed: fresh .so loop did NOT close (stage1/2/3 differ) — "
              "the source is not self-reproducing; do not freeze.")
        return 1
    h1 = hashlib.sha256(stage1).hexdigest()
    _FROZEN.write_bytes(stage1)
    _FROZEN.chmod(_FROZEN.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _FROZEN_MANIFEST.write_text(
        "# Frozen self-host bootstrap ELF (A6/RI-E1): the checked-in pure-MIND stage0\n"
        "# compiler that reproduces itself byte-identically (stage1==stage2==stage3)\n"
        "# and seeds the PRIMARY reproduction-independence loop. Regenerate via\n"
        "# self_host_loop_smoke.py --reseed with MINDC_SO set. NOT skippable in CI.\n"
        "# name\tsize_bytes\tsha256\n"
        f"stage1.elf\t{len(stage1)}\t{h1}\n"
    )
    print(f"  RESEEDED  frozen bootstrap stage1.elf re-blessed: {len(stage1)}B "
          f"sha256={h1}\n            wrote {_FROZEN} and {_FROZEN_MANIFEST}")
    return 0


def main() -> int:
    combined, stdin_image, user_lo = build_seed()
    print(f"[self-host loop] combined={len(combined)}B user_lo={user_lo} "
          f"seed={len(stdin_image)}B  so={SO.name}  reseed={_RESEED}")

    if _RESEED:
        return do_reseed(combined, stdin_image, user_lo)

    # ------------------------------------------------------------------
    # PRIMARY: reproduction-independence from the checked-in pure-MIND ELF.
    # Zero Rust / LLVM in this chain — only execve(self)/read/write/exit.
    # ------------------------------------------------------------------
    if not _FROZEN.exists():
        print(f"BLOCKED: frozen bootstrap seed {_FROZEN} not found — it is the "
              f"checked-in pure-MIND stage0. Re-freeze with --reseed (MINDC_SO set).")
        return 2
    frozen = _FROZEN.read_bytes()
    hf = hashlib.sha256(frozen).hexdigest()
    print(f"  seed = frozen pure-MIND stage0 ELF: {len(frozen)}B sha256={hf}")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        try:
            stage1, stage2, stage3 = run_loop_from_seed(_FROZEN, stdin_image, tmp)
        except RuntimeError as e:
            print(f"  FAIL  {e}")
            return 1

    h1 = hashlib.sha256(stage1).hexdigest()
    h2 = hashlib.sha256(stage2).hexdigest()
    h3 = hashlib.sha256(stage3).hexdigest()
    print(f"  stage1 (frozen stage0 run natively): {len(stage1)}B sha256={h1}")
    print(f"  stage2 (stage1 run natively):        {len(stage2)}B sha256={h2}")
    print(f"  stage3 (stage2 run natively):        {len(stage3)}B sha256={h3}")

    if not (stage1 == stage2 == stage3):
        print("  FAIL  self-host loop NOT closed:")
        if stage1 != stage2:
            print(f"        stage1 != stage2 (sizes {len(stage1)} vs {len(stage2)})")
        if stage2 != stage3:
            print(f"        stage2 != stage3 (sizes {len(stage2)} vs {len(stage3)})")
        return 1
    if stage1 != frozen:
        print(f"  FAIL  stage1 ({h1}) != frozen seed ({hf}) — running the frozen "
              f"pure-MIND stage0 did NOT reproduce it; the bootstrap is not fixed.")
        return 1

    print(f"  PASS  [PRIMARY] stage1 == stage2 == stage3 == frozen stage0 "
          f"BYTE-IDENTICAL ({len(stage1)}B, sha256={h1}) — MIND reproduces its "
          f"compiler with ZERO Rust/LLVM in the chain (scalar subset, RI-E1).")

    # ------------------------------------------------------------------
    # ORACLE: Rust `.so` drift check (soft-skip if the .so is unavailable).
    # Catches std/*.mind or main.mind source drift where the frozen ELF was
    # not re-blessed. The PRIMARY path above does NOT depend on this.
    # ------------------------------------------------------------------
    if not SO.exists():
        print(f"  NOTE  [ORACLE] Rust drift .so not present ({SO}) — SKIPPED "
              f"(source-drift detection unavailable; primary loop still gated). "
              f"Set MINDC_SO or build the self-host .so for full coverage.")
        return 0
    try:
        so_stage1 = stage0_emit(combined, user_lo)
    except OSError as e:
        print(f"  NOTE  [ORACLE] could not load drift .so ({e}) — SKIPPED.")
        return 0
    if not is_static_elf(so_stage1):
        print(f"  FAIL  [ORACLE] fresh .so emitted a non-ELF/empty image "
              f"({len(so_stage1)}B) — .so seed path is broken.")
        return 1
    hso = hashlib.sha256(so_stage1).hexdigest()
    if so_stage1 != frozen:
        print(f"  FAIL  [ORACLE] fresh Rust .so output ({hso}) != frozen bootstrap "
              f"({hf}) — std/main.mind SOURCE drifted; re-freeze with "
              f"`self_host_loop_smoke.py --reseed` (MINDC_SO set) in THIS change.")
        return 1
    print(f"  PASS  [ORACLE] fresh Rust .so output == frozen bootstrap "
          f"({hso}) — no source drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
