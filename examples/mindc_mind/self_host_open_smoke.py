"""
Self-host __mind_open smoke (Rust-independence RI-C, task #228).

Proves the new `__mind_open` intrinsic (open(2), O_RDONLY, by path) works through
the pure-MIND native-ELF emitter: a tiny program reads a real file BY PATH (given
as argv[1]) via std.fs `read_file_native` (which uses __mind_open + __mind_read,
zero libc) and returns its first byte; a nonexistent path yields a sane error
(negative fd -> -1) with no crash.

This is the prerequisite that unblocks a by-path `check <paths>` driver — the last
missing piece was turning a path into source bytes without libc (there was no
open() syscall intrinsic; std.fs's open/lseek/close are libc extern, unusable in a
freestanding static ELF).

Execution-correctness gated (no frozen byte-oracle for this new intrinsic; a dummy
PT_NOTE is fed, loader-ignored).

Run:  MINDC_SO=<self-host .so> python3 examples/mindc_mind/self_host_open_smoke.py
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

# argv[1] = path; read the file via std.fs read_file_native (open+read, no libc);
# return first byte, or 200 if open failed, or 201 if the file is empty.
OPEN_PROG = (
    "fn main() -> i64 {\n"
    "    let path: i64 = __mind_argv(1);\n"
    "    let buf: i64 = __mind_alloc(65536);\n"
    "    let n: i64 = read_file_native(path, buf, 65536);\n"
    "    if n < 0 {\n"
    "        return 200;\n"
    "    }\n"
    "    if n == 0 {\n"
    "        return 201;\n"
    "    }\n"
    "    return __mind_load_i8(buf);\n"
    "}\n"
)


def build_elf(lib, user_src: str) -> bytes:
    std_dir = _REPO / "std"
    parts = [(std_dir / f"{m}.mind").read_bytes() for m in _STDLIB_MODULES]
    std_blob = b"\n".join(parts) + b"\n"
    main = (_HERE / "main.mind").read_bytes()
    combined = std_blob + main + b"\n" + user_src.encode()
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


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_hb.restype = ctypes.c_int64
    lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4

    elf = build_elf(lib, OPEN_PROG)
    if len(elf) == 0:
        print("  FAIL  __mind_open program emitted an empty ELF (native emit failed closed)")
        return 1
    print(f"  ok    __mind_open program -> native ELF ({len(elf)} bytes)")

    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        p = tmp / "open.elf"
        p.write_bytes(elf)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # (1) existing files with known first byte
        for content, first in [("Zhello", ord("Z")), ("apple", ord("a")), ("7up", ord("7"))]:
            f = tmp / "data.txt"
            f.write_text(content)
            code = subprocess.run([str(p), str(f)]).returncode
            good = code == first
            ok = ok and good
            print(f"  {'PASS' if good else 'FAIL'}  read {content!r} by path -> exit {code} "
                  f"(expected {first} = first byte)")

        # (2) multi-byte content read correctly (verify it's the file's real first byte)
        big = tmp / "big.txt"
        big.write_text("X" + "y" * 5000)
        code = subprocess.run([str(p), str(big)]).returncode
        good = code == ord("X")
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  read 5001-byte file -> exit {code} (expected {ord('X')})")

        # (3) nonexistent path -> sane error (read_file_native returns -1 -> exit 200), no crash
        code = subprocess.run([str(p), str(tmp / "does_not_exist.txt")]).returncode
        good = code == 200
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  nonexistent path -> exit {code} "
              f"(expected 200 = open failed, no crash)")

        # (4) empty file -> exit 201
        empty = tmp / "empty.txt"
        empty.write_text("")
        code = subprocess.run([str(p), str(empty)]).returncode
        good = code == 201
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  empty file -> exit {code} (expected 201)")

    if ok:
        print("\nALL PASS  (__mind_open reads real files by path through the pure-MIND native ELF)")
        return 0
    print("\nFAIL  __mind_open smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
