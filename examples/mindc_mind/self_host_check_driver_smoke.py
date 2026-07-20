"""
Self-hosted `check` driver smoke (Rust-independence RI-C, task #109 slice B).

Compiles the pure-MIND `check` driver (examples/mindc_mind/check_driver.mind,
appended after the seeded stdlib + main.mind) to a native x86-64 ELF via the
self-hosted native-ELF path, then runs that ELF as a real subprocess feeding a
.mind source on STDIN. Asserts:
  * a CLEAN source  -> exit 0 (typecheck report has no "ERROR:")
  * an ERROR source -> exit 1 (return-type mismatch flagged)
plus the flag-recognition behavior (--no-typecheck gates the pass; --reporter
json and --fix are recognized + surfaced as stubs on stdout).

This is the load-bearing proof that lex -> parse -> TYPECHECK (which uses the
pure-MIND Map/env) EXECUTE correctly through the native-ELF runtime — not just
the self-compile emitter path the loop gate exercises. Execution-correctness
gated (no frozen byte-oracle; a dummy PT_NOTE is fed, loader-ignored).

Source is read from stdin (not the path args) this slice — reading by path needs
__mind_open (task #228). The path/flag args are still parsed by the driver.

Run:  MINDC_SO=<self-host .so> python3 examples/mindc_mind/self_host_check_driver_smoke.py
"""

import ctypes
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]
_DUMMY_HASH = bytes(range(32))


def build_check_elf(lib) -> bytes:
    """Compile std ++ main.mind ++ check_driver.mind to a native ELF (the check tool)."""
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    std_blob = b"\n".join(parts) + b"\n"
    main = (_HERE / "main.mind").read_bytes()
    driver = (_HERE / "check_driver.mind").read_bytes()
    combined = std_blob + main + b"\n" + driver
    user_lo = len(std_blob)
    sb = ctypes.create_string_buffer(combined, len(combined))
    hb = ctypes.create_string_buffer(_DUMMY_HASH, 32)
    es = lib.selftest_native_elf_hb(
        ctypes.cast(sb, ctypes.c_void_p).value,
        len(combined),
        ctypes.cast(hb, ctypes.c_void_p).value,
        user_lo,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8)) if rd(sh, 8) > 0 else b""


def run_check(elf_path: pathlib.Path, source: str, extra_args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [str(elf_path), *extra_args],
        input=source.encode(),
        capture_output=True,
    )
    return proc.returncode, proc.stdout.decode(errors="replace")


# Clean: `return 42;` is i64, matches the default i64 return type -> no ERROR.
CLEAN_SRC = "fn f() -> i64 {\n    let x: i64 = 5;\n    return 42;\n}\n"
# Error: `return 1 < 2;` is bool, mismatches the expected i64 -> "ERROR:".
ERROR_SRC = "fn g() -> i64 {\n    return 1 < 2;\n}\n"


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_hb.restype = ctypes.c_int64
    lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4

    elf = build_check_elf(lib)
    if len(elf) == 0:
        print("  FAIL  check_driver compiled to an empty ELF (native emit failed closed)")
        return 1
    print(f"  ok    check driver -> native ELF ({len(elf)} bytes)")

    ok = True
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "check.elf"
        p.write_bytes(elf)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # (1) clean source on stdin -> exit 0
        code, out = run_check(p, CLEAN_SRC, ["clean.mind"])
        good = code == 0
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  clean source -> exit {code} (expected 0)")
        print(f"          stdout: {out.strip()!r}")

        # (2) type-error source on stdin -> exit 1, and stdout contains ERROR:
        code, out = run_check(p, ERROR_SRC, ["bad.mind"])
        good = code == 1 and "ERROR:" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  error source -> exit {code} (expected 1), "
              f"'ERROR:' in stdout: {'ERROR:' in out}")
        print(f"          stdout: {out.strip()!r}")

        # (3) --no-typecheck gates the pass: even the error source is 'clean' -> exit 0
        code, out = run_check(p, ERROR_SRC, ["--no-typecheck", "bad.mind"])
        good = code == 0
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  --no-typecheck on error source -> exit {code} (expected 0)")

        # (4) --fix + --reporter json are recognized and surfaced as stubs on stdout
        code, out = run_check(p, CLEAN_SRC, ["--fix", "--reporter", "json", "x.mind"])
        good = code == 0 and "note:" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  --fix/--reporter json stubs surfaced "
              f"(exit {code}, 'note:' in stdout: {'note:' in out})")
        print(f"          stdout: {out.strip()!r}")

    if ok:
        print("\nALL PASS  (self-hosted lex+parse+TYPECHECK run in native ELF; exit-code contract holds)")
        return 0
    print("\nFAIL  check driver smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
