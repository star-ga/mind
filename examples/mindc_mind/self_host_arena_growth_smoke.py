"""
Self-host ARENA-GROWTH smoke (Rule-2c regression gate) — proves the frozen
pure-MIND stage0 can self-compile a GROWN source without tripping the fixed
1 GiB nb_arena capacity guard, and that the grown compile still CONVERGES
(stage1 == stage2 byte-identical).

WHY THIS EXISTS (2026-07-21 root-cause): the native self-host compiler runs in
a fixed 1 GiB bump arena (nb_arena_bytes(), main.mind ~19874) with NO free;
every allocation is permanently resident. The u -> trace_hash -> hb entry chain
lexed the full ~1.9 MB combined source THREE times (each full-source lex()
permanently occupies ~10s of MB via Vec doubling + leaked realloc copies), which
left the whole-compiler self-compile ~150 KB under the cap. Adding as little as
+1.3 KB of source (e.g. 7 one-if fns — the E2023 / E2005-E2103 type-checker-rule
ports) pushed a 2 MiB+8 allocation past the cap and fired the emitted
`cmp new_top, CAP; jae ok; ud2` guard => SIGILL (`stage1.elf exited -4` in
self_host_loop_smoke.py) — a capacity cliff, NOT a codegen bug. The fix
(selftest_native_elf_hb_from_tokens) threads the tokens already lexed by
selftest_native_elf_u into the _hb body instead of re-lexing, reclaiming enough
arena that the self-compile survives ~150+ extra fns of source growth.

THIS GATE: appends GROWTH_FNS (24) synthetic one-if fns — ~3x the size of the
type-checker-rule cluster that used to crash, and >3x the old 7-fn crash
threshold — to the combined self-host source, then asserts:
  1. the FROZEN stage0 compiles the grown source (exit 0, emits a static ELF)
     — this is the exact step that used to die with SIGILL (-4);
  2. stage1 (the grown compiler) itself runs the same grown source and emits
     stage2 == stage1 BYTE-IDENTICAL (full-width compare, convergence — the
     grown compiler is a working compiler, not just a survivor).
Fail-closed: any non-zero exit, empty/non-ELF emit, or stage1 != stage2 is a
FAIL (exit 1); a missing frozen fixture is BLOCKED (exit 2). No .so, no Rust,
no LLVM anywhere in the chain.

Run:  python3 examples/mindc_mind/self_host_arena_growth_smoke.py
"""

import hashlib
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
_FROZEN = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"

_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

# 24 one-if fns ~= 3x the E2005/E2101/E2102/E2103 type-checker cluster; the
# pre-fix arena cliff fired at just 7 of these.
GROWTH_FNS = 24


def build_grown_stdin() -> bytes:
    std_dir = _REPO / "std"
    std_blob = b"\n".join(
        (std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES
    ) + b"\n"
    main = (_HERE / "main.mind").read_bytes()
    growth = b"".join(
        (
            f"pub fn zz_arena_growth_probe_{i}(x: i64) -> i64 {{\n"
            f"    if x != 0 {{\n        return 1;\n    }}\n    0\n}}\n"
        ).encode()
        for i in range(GROWTH_FNS)
    )
    driver = (_HERE / "selfhost_driver.mind").read_bytes()
    combined = std_blob + main + b"\n" + growth + b"\n" + driver
    return struct.pack("<qq", len(std_blob), len(combined)) + combined


def run_elf(elf_path: pathlib.Path, stdin_image: bytes, what: str) -> bytes:
    r = subprocess.run(
        [str(elf_path)], input=stdin_image, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{what} exited {r.returncode} (rc -4 == SIGILL == the nb_arena "
            f"capacity guard fired: the self-compile no longer fits the fixed "
            f"1 GiB arena — re-check for a new full-source re-lex / permanent "
            f"allocation on the selftest_native_elf_u path; NEVER grow the arena)"
        )
    if len(r.stdout) < 4096 or r.stdout[:4] != b"\x7fELF":
        raise RuntimeError(f"{what} emitted a non-ELF/empty image ({len(r.stdout)}B)")
    return r.stdout


def main() -> int:
    if not _FROZEN.exists():
        print(f"BLOCKED: frozen bootstrap seed {_FROZEN} not found.")
        return 2
    stdin_image = build_grown_stdin()
    print(f"[arena-growth smoke] grown seed={len(stdin_image)}B "
          f"(+{GROWTH_FNS} synthetic fns; pre-fix cliff was 7)")
    try:
        stage1 = run_elf(_FROZEN, stdin_image, "frozen stage0 (grown-source compile)")
        with tempfile.TemporaryDirectory() as td:
            p1 = pathlib.Path(td) / "stage1.elf"
            p1.write_bytes(stage1)
            p1.chmod(p1.stat().st_mode | stat.S_IEXEC)
            stage2 = run_elf(p1, stdin_image, "grown stage1 (self-recompile)")
    except RuntimeError as e:
        print(f"  FAIL  {e}")
        return 1
    h1 = hashlib.sha256(stage1).hexdigest()
    h2 = hashlib.sha256(stage2).hexdigest()
    print(f"  stage1 (frozen stage0 on grown source): {len(stage1)}B sha256={h1}")
    print(f"  stage2 (grown stage1 self-recompile):   {len(stage2)}B sha256={h2}")
    if stage1 != stage2:
        print("  FAIL  grown-source loop did NOT converge (stage1 != stage2).")
        return 1
    print(f"  PASS  frozen stage0 self-compiles +{GROWTH_FNS}-fn grown source with "
          f"NO arena-guard SIGILL, and the grown compiler converges "
          f"(stage1 == stage2 byte-identical, {len(stage1)}B).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
