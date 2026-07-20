"""
Self-host args_from_os round-trip smoke (Rust-independence RI-C) — proves
std/cli.mind's `args_from_os()` populates an `Args` from the REAL process argv,
end to end through the pure-MIND native-ELF emitter.

Builds on task #222 (the `__mind_argc()` / `__mind_argv(i)` intrinsics): a tiny
program that calls `args_from_os()` then reads back a specific arg via
`args_len` / `args_get` is compiled — WITH the bundled stdlib seeded so
std.cli/std.vec/std.string resolve — to a native x86-64 ELF, run with a
controlled argv, and its exit code asserted against the real argument vector.

Execution-correctness gated (there is no frozen byte-oracle for this new
runtime path). A dummy PT_NOTE hash is fed (the note is loader-ignored metadata
and does not affect the exit code we assert on).

Run:  MINDC_SO=<self-host .so> python3 examples/mindc_mind/self_host_args_from_os_smoke.py
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

# The 21 bundled stdlib modules, in STDLIB_MIND_SOURCES order (must match the
# native emitter's seeding order — same list self_host_native_elf_smoke.py uses).
_STDLIB_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]

# Loader-ignored 32-byte PT_NOTE; execution (the exit code) is what we assert on.
_DUMMY_HASH = bytes(range(32))


def seeded_buffer(user_src: bytes) -> tuple[bytes, int]:
    """std/*.mind (seeding order) ++ the user program. Returns (combined, user_lo)
    where user_lo is the seam the native emitter treats as the prune root."""
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    std_blob = b"\n".join(parts) + b"\n"
    return std_blob + user_src, len(std_blob)


def mind_elf(lib, user_src: str) -> bytes:
    combined, user_lo = seeded_buffer(user_src.encode())
    sb = ctypes.create_string_buffer(combined, len(combined))
    hb = ctypes.create_string_buffer(_DUMMY_HASH, 32)
    es = lib.selftest_native_elf_hb(
        ctypes.cast(sb, ctypes.c_void_p).value,
        len(combined),
        ctypes.cast(hb, ctypes.c_void_p).value,
        user_lo,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # String handle: addr/len/cap
    return ctypes.string_at(rd(sh, 0), rd(sh, 8)) if rd(sh, 8) > 0 else b""


def run_elf(elf: bytes, tmp: pathlib.Path, argv_tail: list[str]) -> int:
    p = tmp / "args.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p), *argv_tail]).returncode


# args_from_os() -> Args, then return the entry count (== argc).
LEN_PROG = (
    "fn main() -> i64 {\n"
    "    let a: Args = args_from_os();\n"
    "    return args_len(a);\n"
    "}\n"
)
# args_from_os() -> Args, then return the first byte of argv[1] (round-trips the
# borrowed String view through args_get).
GET_PROG = (
    "fn main() -> i64 {\n"
    "    let a: Args = args_from_os();\n"
    "    let s: String = args_get(a, 1);\n"
    "    return string_get_byte(s, 0);\n"
    "}\n"
)


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_hb.restype = ctypes.c_int64
    lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4

    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        len_elf = mind_elf(lib, LEN_PROG)
        if len(len_elf) == 0:
            print("  FAIL  args_from_os()/args_len program emitted an empty ELF (native emit failed closed)")
            return 1
        print(f"  ok    args_from_os()/args_len program -> native ELF ({len(len_elf)} bytes)")
        for tail in ([], ["x"], ["a", "b", "c"], ["1", "2", "3", "4", "5"]):
            expected = 1 + len(tail)  # argv[0] + the tail
            got = run_elf(len_elf, tmp, tail)
            good = got == expected
            ok = ok and good
            print(
                f"  {'PASS' if good else 'FAIL'}  args_len(args_from_os()) with {len(tail)} "
                f"extra arg(s): exit {got} (expected {expected})"
            )

        get_elf = mind_elf(lib, GET_PROG)
        if len(get_elf) == 0:
            print("  FAIL  args_from_os()/args_get program emitted an empty ELF (native emit failed closed)")
            return 1
        print(f"  ok    args_from_os()/args_get program -> native ELF ({len(get_elf)} bytes)")
        for ch in ("A", "Z", "m", "7"):
            expected = ord(ch)
            got = run_elf(get_elf, tmp, [ch + "tail"])
            good = got == expected
            ok = ok and good
            print(
                f"  {'PASS' if good else 'FAIL'}  string_get_byte(args_get(args_from_os(),1),0) "
                f"with argv[1]={ch!r}: exit {got} (expected {expected} = ord({ch!r}))"
            )

    if ok:
        print("\nALL PASS  (args_from_os round-trips real OS argv through the pure-MIND native ELF)")
        return 0
    print("\nFAIL  args_from_os smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
