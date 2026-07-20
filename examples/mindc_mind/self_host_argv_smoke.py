"""
Self-host ARGV smoke (Rust-independence RI-C) — proves the pure-MIND native-ELF
emitter (main.mind SECTION 4c) gives compiled MIND programs real OS argc/argv
access via two new inlined intrinsics:

  __mind_argc()  -> i64  the process argument count
  __mind_argv(i) -> i64  raw pointer to argv[i] (a NUL-terminated C string)

The argc/argv are captured from the initial SysV stack as the literal FIRST
instructions of an argv-using entry, BEFORE the `movabs rsp,STACK_TOP` switch
abandons the OS stack, and stashed in two reserved header slots of the fixed BSS
arena (ARENA_ADDR+8 = argc, ARENA_ADDR+16 = &argv[0]) with the bump cursor
pre-seeded to 16 so those bytes are never allocated. The capture is emitted ONLY
for a module that references argv — every argv-free program stays byte-identical
to before this feature (verified separately by self_host_native_elf_smoke.py).

This gate compiles tiny argv programs via the pure-MIND native path
(selftest_native_elf_h) and RUNS the emitted ELF with a controlled argv,
asserting the exit code encodes the captured argc / argv[i] byte. Execution-
correctness gated (there is no frozen byte-oracle for this brand-new construct).

Run:  MINDC_SO=<self-host .so> python3 examples/mindc_mind/self_host_argv_smoke.py
"""

import ctypes
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# A dummy 32-byte PT_NOTE hash: the note is metadata the loader ignores, so it does
# not affect execution — this test proves runtime argc/argv behavior, not byte
# identity (there is no frozen oracle for this new construct).
_DUMMY_HASH = bytes(range(32))


def mind_elf(lib, src: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(_DUMMY_HASH, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # String handle: addr/len/cap
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path, argv_tail: list[str]) -> int:
    p = tmp / "argv.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    # argv[0] is the path; argv_tail are the extra args the child sees.
    return subprocess.run([str(p), *argv_tail]).returncode


# `main` returns argc directly -> exit code == argc.
ARGC_PROG = (
    "fn main() -> i64 {\n"
    "    return __mind_argc();\n"
    "}\n"
)
# `main` returns the first byte of argv[1] -> exit code == ord(argv[1][0]).
ARGV_BYTE_PROG = (
    "fn main() -> i64 {\n"
    "    return __mind_load_i8(__mind_argv(1));\n"
    "}\n"
)


def main() -> int:
    if not SO.exists():
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3

    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        # (1) argc: exit code must equal the real process argument count.
        argc_elf = mind_elf(lib, ARGC_PROG.encode())
        for tail in ([], ["x"], ["a", "b", "c"], ["1", "2", "3", "4", "5"]):
            expected = 1 + len(tail)  # argv[0] (the path) + the tail args
            got = run_elf(argc_elf, tmp, tail)
            good = got == expected
            ok = ok and good
            print(
                f"  {'PASS' if good else 'FAIL'}  __mind_argc with {len(tail)} extra "
                f"arg(s): exit {got} (expected {expected})"
            )

        # (2) argv[i]: exit code must equal the first byte of argv[1].
        argv_elf = mind_elf(lib, ARGV_BYTE_PROG.encode())
        for ch in ("A", "Z", "m", "7"):
            expected = ord(ch)
            got = run_elf(argv_elf, tmp, [ch + "tail"])
            good = got == expected
            ok = ok and good
            print(
                f"  {'PASS' if good else 'FAIL'}  __mind_argv(1)[0] with argv[1]={ch!r}: "
                f"exit {got} (expected {expected} = ord({ch!r}))"
            )

    if ok:
        print("\nALL PASS  (pure-MIND native ELF reads real OS argc/argv)")
        return 0
    print("\nFAIL  argv smoke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
