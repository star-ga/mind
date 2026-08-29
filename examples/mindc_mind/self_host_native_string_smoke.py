"""
Self-host NATIVE-ELF STRING-LITERAL smoke (Rust-independence #14) — proves the
pure-MIND front-end (examples/mindc_mind/main.mind) lowers a source string literal
to a read-only rodata PT_LOAD segment and materializes a borrowed String handle
{ addr = 0x800000 + off, len, cap = 0 }, entirely in pure MIND with ZERO Rust
codegen in the emit path.

There is NO frozen byte-oracle for this construct (it is NEW — the deleted Rust
`src/native` backend never emitted rodata/strings), so the gate is
EXECUTION-CORRECTNESS + BYTE-DETERMINISM, not byte-identity vs a Rust reference:

  * runs + exits with the expected value  — the emitted ELF actually maps the
    rodata segment at 0x800000 and the borrowed handle's fields are correct.
  * two independent compiles are byte-identical  — the emit is deterministic
    (the rodata vaddr is a compile-time constant, not a runtime alloc address).

The `strbyte` fixture double-dereferences the handle (addr field -> first qword of
the literal bytes) so a PASS proves the rodata PT_LOAD segment is genuinely mapped
R and contains the copied literal bytes at runtime — not merely that the len field
was set.

Run:  MINDC_SO=<fresh .so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_native_string_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# A dummy 32-byte PT_NOTE trace hash. This smoke gates runtime behavior +
# determinism of the CODE + rodata regions, not the note (there is no oracle to
# match), so any fixed 32 bytes is fine — both compiles feed the SAME hash, so the
# note stays constant and cannot mask a real code-region divergence.
_DUMMY_HASH = bytes(range(32))

# (name, source, expected_exit, expected_phdrs). Each binds a string literal and
# returns an observable i64 (exit code = low 8 bits of the returned value).
# expected_phdrs pins the CONDITIONAL-EMIT regime (a design decision): a program
# with a non-empty literal is 5-phdr; an all-empty-literal program stays 4-phdr and
# byte-identical to the stringless layout.
FIXTURES = [
    # len field of the borrowed handle: s.len via __mind_load_i64(s + 8). "hello"
    # -> 5. Proves the ast_str_lit arm ran, the accumulator bumped, and the handle
    # carries the correct length.
    (
        "strlen",
        (
            "fn main() -> i64 {\n"
            '    let s = "hello";\n'
            "    return __mind_load_i64(s + 8);\n"
            "}\n"
        ),
        5,
        5,
    ),
    # first literal byte via a DOUBLE dereference: addr field (s+0) -> rodata
    # vaddr, then the first 8 bytes at that vaddr. "hi" -> 0x6968; exit = 0x68 =
    # 'h' = 104. Proves the rodata PT_LOAD segment maps at 0x800000 and holds the
    # copied bytes at runtime (the whole point of this slice).
    (
        "strbyte",
        (
            "fn main() -> i64 {\n"
            '    let s = "hi";\n'
            "    return __mind_load_i64(__mind_load_i64(s));\n"
            "}\n"
        ),
        104,
        5,
    ),
    # EDGE (an audit finding): an all-EMPTY-literal program appends 0 rodata bytes, so
    # rod_len stays 0 and the conditional 5th phdr is NOT emitted — the binary stays
    # 4-phdr, byte-identical to the stringless layout. The handle is {0x800000, 0, 0};
    # s.len (s + 8) == 0, so exit 0, and nothing dereferences the (unmapped) addr.
    (
        "empty_only",
        (
            "fn main() -> i64 {\n"
            '    let s = "";\n'
            "    return __mind_load_i64(s + 8);\n"
            "}\n"
        ),
        0,
        4,
    ),
    # EDGE (an audit finding): a "" beside a non-empty literal. The "" gets a zero-length
    # handle at the rodata base; "hi" drives rod_len>0 so the 5th phdr IS emitted.
    # exit = a.len + b.len = 0 + 2 = 2.
    (
        "mixed_empty",
        (
            "fn main() -> i64 {\n"
            '    let a = "";\n'
            '    let b = "hi";\n'
            "    return __mind_load_i64(a + 8) + __mind_load_i64(b + 8);\n"
            "}\n"
        ),
        2,
        5,
    ),
]


def _phdrs(elf: bytes):
    """Parse the ELF64 program headers as
    [(p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align), ...]."""
    e_phoff = struct.unpack_from("<Q", elf, 0x20)[0]
    e_phnum = struct.unpack_from("<H", elf, 0x38)[0]
    return [struct.unpack_from("<IIQQQQQQ", elf, e_phoff + i * 56) for i in range(e_phnum)]


def mind_string_elf(lib, src: bytes) -> bytes:
    """Lower `src` through the pure-MIND native-ELF front-end (selftest_native_elf_h,
    dummy note) and return the emitted ELF bytes (empty on a fail-closed refusal)."""
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(_DUMMY_HASH, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    length = rd(sh, 8)
    if length <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), length)


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = ctypes.CDLL(str(SO))
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64] * 3

    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for name, src, expected_exit, expected_phdrs in FIXTURES:
            elf = mind_string_elf(lib, src.encode())
            if len(elf) == 0:
                print(f"  FAIL  {name}: pure-MIND emit failed closed (empty ELF)")
                all_ok = False
                continue
            # ELF sanity: real ELF64 magic.
            if elf[:4] != b"\x7fELF":
                print(f"  FAIL  {name}: emitted bytes are not an ELF ({elf[:4]!r})")
                all_ok = False
                continue

            # (0) STRUCTURAL — conditional-emit regime + the rodata READ-ONLY flag.
            # A wrong p_flags (RW/RX) still runs, still exits correctly, still hashes
            # byte-deterministic — so it is INVISIBLE to the run + determinism checks
            # below. This is the one gate that carries that information (an audit finding).
            ph = _phdrs(elf)
            struct_ok = len(ph) == expected_phdrs
            detail = ""
            if len(ph) == 5:
                p_type, p_flags, p_off, p_vaddr, _pp, _fsz, _msz, p_align = ph[4]
                rod_ok = (
                    p_type == 1  # PT_LOAD
                    and p_flags == 4  # R only (never RW/RX)
                    and p_vaddr == 0x800000
                    and p_off % p_align == p_vaddr % p_align  # mmap congruence
                )
                struct_ok = struct_ok and rod_ok
                detail = (
                    "; rodata PT_LOAD R-only @0x800000, congruent"
                    if rod_ok
                    else f"; rodata phdr WRONG (type={p_type} flags={p_flags} vaddr={p_vaddr:#x})"
                )
            print(
                f"  {'PASS' if struct_ok else 'FAIL'}  {name} phdrs={len(ph)} "
                f"(expected {expected_phdrs}){detail}"
            )
            all_ok = all_ok and struct_ok

            # (1) EXECUTION-CORRECTNESS.
            code = run_elf(elf, tmp)
            run_ok = code == expected_exit
            print(
                f"  {'PASS' if run_ok else 'FAIL'}  {name} pure-MIND string ELF runs + "
                f"exits {code} (expected {expected_exit}) [{len(elf)} bytes]"
            )
            all_ok = all_ok and run_ok

            # (2) BYTE-DETERMINISM: a second independent compile must be identical.
            elf2 = mind_string_elf(lib, src.encode())
            det_ok = elf2 == elf
            print(
                f"  {'PASS' if det_ok else 'FAIL'}  {name} byte-deterministic "
                f"(recompile {'identical' if det_ok else 'DIVERGED'}, "
                f"{len(elf)} vs {len(elf2)} bytes)"
            )
            all_ok = all_ok and det_ok

    if not all_ok:
        print("\nFAIL  native-ELF string-literal smoke")
        return 1
    print(
        "\nALL PASS  (pure-MIND native-ELF string literals: rodata PT_LOAD segment "
        "maps + holds the copied bytes, borrowed handle fields correct, emit "
        "byte-deterministic)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
