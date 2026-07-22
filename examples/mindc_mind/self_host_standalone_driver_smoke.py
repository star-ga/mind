"""
Self-host STANDALONE-DRIVER smoke (Rust-independence #14, C8 — the "no Python
driver" headline for RI-E1).

Proves that a STANDALONE pure-MIND binary — with its OWN argv parsing + file I/O,
no Python harness in the run loop — reproduces the self-host compiler. The driver
source is examples/mindc_mind/selfhost_argv_driver.mind: an argv/file-IO twin of
selfhost_driver.mind that reads the seeded source image FROM A FILE named on argv,
compiles it with the existing pure-MIND front-end (selftest_native_elf_u), and
writes the emitted native ELF to stdout — using ONLY MIND + raw open/read/write/exit
syscalls (the __mind_* intrinsics). main.mind is UNCHANGED (the driver is appended
source only), so the frozen stage1.elf seed and every byte-identity gate are
untouched — this smoke adds a proof, it does not perturb one.

THE HEADLINE PROOF (step 2): build the standalone driver ELF, then run it as a real
process — `./driver <seed_image_file>` (stdout captured) — on the SAME seeded
stdlib+main.mind+driver source the Python self-host loop compiles, and assert its
emitted ELF is BYTE-IDENTICAL to testdata/selfhost_loop/stage1.elf (the frozen
pure-MIND stage0). That is a pure-MIND binary reproducing the compiler with ZERO
Python in the loop.

Non-vacuous (step 3): the driver is ALSO run on a SMALL std-seeded `add` program and
its emitted ELF is required to equal selftest_native_elf_u's direct output for the
same (src, len, user_lo) AND to run + exit 5 — proving the driver's read->compile->
write path is a faithful compiler for arbitrary source, not just a stage1.elf replayer.

Zero-Rust bonus (step 5): the driver ELF is ALSO minted by RUNNING the frozen stage0
(no .so, no Rust) and required to be byte-identical to the .so-built one, and to
itself reproduce stage1.elf — so the WHOLE chain (build the driver, run the driver)
can be Rust-free.

OUTPUT PATH NOTE: the emitted ELF goes to stdout (fd 1). The freestanding native
backend's only open(2) intrinsic is __mind_open (O_RDONLY); there is no open-for-write
intrinsic, and adding one would change main.mind and force a re-freeze of the frozen
stage1.elf. Writing to the inherited stdout fd (exactly as selfhost_driver.mind does)
keeps main.mind byte-identical; the caller places the bytes at the output path via
stdout redirection. See the deferred marker in selfhost_argv_driver.mind for the
argv[2]-open-for-write upgrade path.

FAIL-CLOSED:
  * .so unavailable and MINDC_SO set  -> ERROR exit 1 (refuse to skip)
  * .so unavailable and MINDC_SO unset -> SKIP  exit 0 (matches sibling smokes)
  * frozen stage1.elf missing          -> BLOCKED exit 2 (it is the seed/oracle)
  * driver ELF empty / not an ELF      -> FAIL exit 1
  * driver exits non-zero / no output  -> FAIL exit 1
  * driver out.elf != frozen stage1    -> FAIL exit 1
  * small-fixture out != direct emit / wrong exit -> FAIL exit 1

Run:
  MINDC_SO=/path/to/libmindc_mind.so python3 \\
      examples/mindc_mind/self_host_standalone_driver_smoke.py
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
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
# Reuse the loop's EXACT source-combining so the headline compiles the identical
# bytes the frozen stage1.elf was minted from.
from self_host_loop_smoke import build_seed  # noqa: E402

SO = resolve_so()

_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]
_FROZEN = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"

# A small std-seeded program for the non-vacuous rung: add(2,3) -> exit 5.
_ADD_PROG = (
    "fn add(a: i64, b: i64) -> i64 {\n"
    "    return a + b;\n"
    "}\n"
    "fn main() -> i64 {\n"
    "    return add(2, 3);\n"
    "}\n"
)


def _std_blob() -> bytes:
    std_dir = _REPO / "std"
    return b"\n".join(
        (std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES
    ) + b"\n"


def _emit_u(lib, combined: bytes, user_lo: int) -> bytes:
    """The .so's selftest_native_elf_u(src, len, user_lo) -> emitted ELF bytes."""
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


def _is_static_elf(b: bytes) -> bool:
    return len(b) > 256 and b[:4] == b"\x7fELF" and b[16:18] == b"\x02\x00"


def _write_exec(d: pathlib.Path, name: str, data: bytes) -> pathlib.Path:
    p = d / name
    p.write_bytes(data)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


def _build_driver_elf(lib) -> bytes:
    """Compile [std ++ main.mind ++ selfhost_argv_driver.mind] to the standalone
    argv-driven driver ELF (the argv driver's `main` becomes the ELF entry)."""
    std_blob = _std_blob()
    main = (_HERE / "main.mind").read_bytes()
    argv_driver = (_HERE / "selfhost_argv_driver.mind").read_bytes()
    combined = std_blob + main + b"\n" + argv_driver
    return _emit_u(lib, combined, len(std_blob))


def _run_driver(driver_path: pathlib.Path, image: bytes, td: pathlib.Path,
                extra_args: list[str] | None = None) -> tuple[int, bytes]:
    """Run `driver <image_file> [extra...]`; return (exit_code, stdout)."""
    img = td / "seed.image"
    img.write_bytes(image)
    args = [str(driver_path), str(img)] + (extra_args or [])
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=180)
    return r.returncode, r.stdout


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built (set MINDC_SO or build the self-host .so)")
        return 0
    if not _FROZEN.exists():
        print(f"BLOCKED: frozen bootstrap seed {_FROZEN} not found — it is the "
              f"headline oracle. Re-freeze via self_host_loop_smoke.py --reseed.")
        return 2

    lib = ctypes.CDLL(str(SO))

    # ------------------------------------------------------------------
    # 1. Build the STANDALONE argv/file-IO driver ELF.
    # ------------------------------------------------------------------
    driver_elf = _build_driver_elf(lib)
    if not _is_static_elf(driver_elf):
        print(f"  FAIL  standalone driver ELF not emitted "
              f"({len(driver_elf)}B, magic={driver_elf[:4]!r})")
        return 1
    print(f"  ok    built standalone argv driver ELF: {len(driver_elf)}B "
          f"sha256={hashlib.sha256(driver_elf).hexdigest()[:16]}")

    frozen = _FROZEN.read_bytes()
    hf = hashlib.sha256(frozen).hexdigest()

    with tempfile.TemporaryDirectory() as tds:
        td = pathlib.Path(tds)
        drv = _write_exec(td, "driver.elf", driver_elf)

        # --------------------------------------------------------------
        # 2. HEADLINE: run the standalone driver on the loop's seed image;
        #    its emitted ELF must be byte-identical to the frozen stage1.elf.
        # --------------------------------------------------------------
        _combined, seed_image, _user_lo = build_seed()  # the loop's exact bytes
        rc, out = _run_driver(drv, seed_image, td)
        if rc != 0:
            print(f"  FAIL  standalone driver exited {rc} on the seed image "
                  f"(0 expected) — read/compile/write path broke")
            return 1
        if not _is_static_elf(out):
            print(f"  FAIL  standalone driver emitted a non-ELF/empty image "
                  f"({len(out)}B)")
            return 1
        ho = hashlib.sha256(out).hexdigest()
        print(f"  driver out.elf : {len(out)}B sha256={ho}")
        print(f"  frozen stage1  : {len(frozen)}B sha256={hf}")
        if out != frozen:
            print("  FAIL  [HEADLINE] standalone driver out.elf != frozen stage1.elf "
                  "— the pure-MIND binary did NOT reproduce the compiler byte-identically")
            return 1
        print(f"  PASS  [HEADLINE] standalone pure-MIND driver reproduced the frozen "
              f"stage1.elf BYTE-IDENTICAL ({len(out)}B, sha256={ho}) — ZERO Python in "
              f"the compile loop (argv + open + read + write + exit, all pure MIND)")

        # --------------------------------------------------------------
        # 3. NON-VACUOUS: the driver is a faithful compiler for arbitrary
        #    source, not just a stage1 replayer. Compile a small std-seeded
        #    `add` program THROUGH the driver and require its ELF to equal
        #    selftest_native_elf_u's direct output AND to run + exit 5.
        # --------------------------------------------------------------
        std_blob = _std_blob()
        combined_s = std_blob + _ADD_PROG.encode()
        user_lo_s = len(std_blob)
        direct = _emit_u(lib, combined_s, user_lo_s)
        if not _is_static_elf(direct):
            print(f"  FAIL  small-fixture direct emit is not an ELF ({len(direct)}B) — "
                  f"cannot form a non-vacuous oracle")
            return 1
        image_s = struct.pack("<qq", user_lo_s, len(combined_s)) + combined_s
        rc2, out2 = _run_driver(drv, image_s, td)
        if rc2 != 0 or out2 != direct:
            print(f"  FAIL  small-fixture: driver ELF (exit {rc2}, {len(out2)}B) != "
                  f"direct selftest_native_elf_u ({len(direct)}B) — driver miscompiled")
            return 1
        code = subprocess.run([str(_write_exec(td, "add.elf", out2))]).returncode
        if code != 5:
            print(f"  FAIL  small-fixture ELF ran but exited {code} (expected 5)")
            return 1
        print(f"  PASS  [NON-VACUOUS] driver-compiled `add` == direct emit "
              f"({len(out2)}B) and the emitted ELF runs + exits 5")

        # --------------------------------------------------------------
        # 4. NEGATIVE: no input-path arg -> usage exit 2 (fail-closed CLI).
        # --------------------------------------------------------------
        r_noarg = subprocess.run([str(drv)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=30)
        if r_noarg.returncode != 2 or r_noarg.stdout:
            print(f"  FAIL  no-arg invocation exited {r_noarg.returncode} with "
                  f"{len(r_noarg.stdout)}B stdout (expected exit 2, no output)")
            return 1
        print("  PASS  [NEGATIVE] no input-path arg -> exit 2, no output")

        # --------------------------------------------------------------
        # 5. ZERO-RUST bonus: mint the driver ELF by RUNNING the frozen
        #    stage0 (no .so, no Rust), require it byte-identical to the
        #    .so-built driver, and require IT to reproduce stage1.elf too.
        # --------------------------------------------------------------
        combined_a = std_blob + (_HERE / "main.mind").read_bytes() + b"\n" + \
            (_HERE / "selfhost_argv_driver.mind").read_bytes()
        image_a = struct.pack("<qq", len(std_blob), len(combined_a)) + combined_a
        r_pure = subprocess.run([str(_FROZEN)], input=image_a,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=180)
        driver_pure = r_pure.stdout
        if r_pure.returncode != 0 or driver_pure != driver_elf:
            print(f"  FAIL  [ZERO-RUST] frozen-stage0-built driver (exit "
                  f"{r_pure.returncode}, {len(driver_pure)}B) != .so-built driver "
                  f"({len(driver_elf)}B)")
            return 1
        drv2 = _write_exec(td, "driver_pure.elf", driver_pure)
        rc3, out3 = _run_driver(drv2, seed_image, td)
        if rc3 != 0 or out3 != frozen:
            print(f"  FAIL  [ZERO-RUST] pure-MIND-built driver did not reproduce "
                  f"stage1.elf (exit {rc3}, {len(out3)}B)")
            return 1
        print("  PASS  [ZERO-RUST] driver ELF minted by the frozen stage0 (no Rust) "
              "is byte-identical to the .so build AND reproduces stage1.elf — the "
              "whole build+run chain is Rust-free")

    print("\nALL PASS  (a standalone pure-MIND binary reproduces the self-host "
          "compiler byte-identically with no Python driver — RI-E1 C8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
