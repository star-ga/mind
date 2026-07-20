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
    """stdin-fallback mode: no positional path, source piped on stdin."""
    proc = subprocess.run(
        [str(elf_path), *extra_args],
        input=source.encode(),
        capture_output=True,
    )
    return proc.returncode, proc.stdout.decode(errors="replace")


def run_check_path(elf_path: pathlib.Path, src_file, extra_args: list[str]) -> tuple[int, str]:
    """BY-PATH mode (RI-C #228): pass a real file path as a positional arg, no stdin."""
    proc = subprocess.run(
        [str(elf_path), *extra_args, str(src_file)],
        input=b"",
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

        clean_f = pathlib.Path(td) / "clean.mind"
        clean_f.write_text(CLEAN_SRC)
        bad_f = pathlib.Path(td) / "bad.mind"
        bad_f.write_text(ERROR_SRC)

        # (1) BY-PATH clean file -> exit 0 (RI-C #228: reads the real file via __mind_open)
        code, out = run_check_path(p, clean_f, [])
        good = code == 0
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  BY-PATH clean file -> exit {code} (expected 0)")
        print(f"          stdout: {out.strip()!r}")

        # (2) BY-PATH type-error file -> exit 1, stdout contains ERROR:
        code, out = run_check_path(p, bad_f, [])
        good = code == 1 and "ERROR:" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  BY-PATH error file -> exit {code} (expected 1), "
              f"'ERROR:' in stdout: {'ERROR:' in out}")
        print(f"          stdout: {out.strip()!r}")

        # (3) --no-typecheck gates the pass: error file reported clean -> exit 0
        code, out = run_check_path(p, bad_f, ["--no-typecheck"])
        good = code == 0
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  --no-typecheck BY-PATH error file -> exit {code} (expected 0)")

        # (4) --fix + --reporter json recognized + stubbed (path arg not swallowed as the value)
        code, out = run_check_path(p, clean_f, ["--fix", "--reporter", "json"])
        good = code == 0 and "note:" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  --fix/--reporter json stubs surfaced BY-PATH "
              f"(exit {code}, 'note:' in stdout: {'note:' in out})")
        print(f"          stdout: {out.strip()!r}")

        # (5) nonexistent path -> "cannot open path" + exit 1, no crash
        code, out = run_check_path(p, pathlib.Path(td) / "does_not_exist.mind", [])
        good = code == 1 and "open" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  nonexistent path -> exit {code} (expected 1), "
              f"'open' note: {'open' in out}")
        print(f"          stdout: {out.strip()!r}")

        # (6) STDIN fallback (no positional path): clean source on stdin -> exit 0
        code, out = run_check(p, CLEAN_SRC, [])
        good = code == 0
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  STDIN-fallback clean -> exit {code} (expected 0)")

        # (7) STDIN fallback: error source on stdin -> exit 1
        code, out = run_check(p, ERROR_SRC, [])
        good = code == 1 and "ERROR:" in out
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  STDIN-fallback error -> exit {code} (expected 1)")

    if ok:
        print("\nALL PASS  (by-path __mind_open read + stdin fallback; self-hosted lex+parse+TYPECHECK in native ELF)")
        return 0
    print("\nFAIL  check driver smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
