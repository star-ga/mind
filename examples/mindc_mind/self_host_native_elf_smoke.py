"""
Self-host NATIVE-ELF smoke (Rust-independence #14, PHASE 1.3) — proves the
pure-MIND front-end can emit a static x86-64 ELF64 byte-identical to the Rust
`mind-native` backend (src/native/mod.rs) for the SCALAR i64 subset.

This is the FIRST increment of porting src/native into main.mind. It exercises
the additive `selftest_native_elf_h` export (SECTION 4c in main.mind), which is
ISOLATED from the mic@1 canary / mic@3 flip — so the keystone stays consistent.

The oracle is regenerated LIVE from a freshly-built `mind-native` binary (guards
golden staleness). NO fake wins — the pass requires a byte-for-byte match of the
pure-MIND-emitted ELF vs the Rust `mind-native` ELF, AND the pure-MIND ELF must
run and exit with the fixture's value (add(2,3) = 5).

The 32-byte ir_trace_hash that anchors the ELF's PT_NOTE is not yet ported to
pure MIND (FOLLOW-ON increment); the harness reads it from the oracle ELF's note
and passes it in, so the note is byte-identical. The instruction stream, the four
phdrs, and the ELF skeleton are ALL emitted in pure MIND.

Run:  python3 examples/mindc_mind/self_host_native_elf_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import subprocess
import sys

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

# The mind-native oracle binary (built with --features native-backend). CI / local
# point MIND_NATIVE_BIN at the freshly-built debug/release binary.
_REPO = _HERE.parent.parent
_DEFAULT_NATIVE = pathlib.Path("/tmp/mind-native-target/debug/mind-native")
MIND_NATIVE = pathlib.Path(os.environ.get("MIND_NATIVE_BIN", str(_DEFAULT_NATIVE)))

# The scalar i64 add/main fixture. Typed params, intra-module call, exit value 5.
FIXTURE = (
    "fn add(a: i64, b: i64) -> i64 {\n"
    "    return a + b;\n"
    "}\n"
    "fn main() -> i64 {\n"
    "    return add(2, 3);\n"
    "}\n"
)
EXPECTED_EXIT = 5


def oracle_elf(src: str, tmp: pathlib.Path) -> bytes:
    src_path = tmp / "fixture_add.mind"
    elf_path = tmp / "rust.elf"
    src_path.write_text(src)
    subprocess.run(
        [str(MIND_NATIVE), str(src_path), str(elf_path)],
        capture_output=True,
        check=True,
    )
    return elf_path.read_bytes()


def mind_elf(lib, src: bytes, trace_hash: bytes) -> bytes:
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(trace_hash, 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def first_diverge(a: bytes, b: bytes) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            lo, hi = max(0, i - 8), min(n, i + 16)
            return (
                f"FIRST DIVERGE at offset {i} (0x{i:x}): "
                f"mind={a[i]:#04x} rust={b[i]:#04x}\n"
                f"  mind: {a[lo:hi].hex()}\n  rust: {b[lo:hi].hex()}"
            )
    return f"length mismatch: mind={len(a)} rust={len(b)} (common prefix identical)"


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
    if not MIND_NATIVE.exists():
        print(
            f"SKIP: mind-native oracle binary not found at {MIND_NATIVE} "
            "(build with: CARGO_TARGET_DIR=/tmp/mind-native-target "
            "cargo build --features native-backend --bin mind-native)"
        )
        return 0

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        rust = oracle_elf(FIXTURE, tmp)
        # The ir_trace_hash anchoring the note: last 52 bytes = 12-byte nhdr +
        # "MIND\0\0\0\0" (8) + 32-byte hash.
        note = rust[-52:]
        if note[12:16] != b"MIND":
            print(f"ERROR: oracle ELF note missing MIND name: {note[12:20]!r}")
            return 1
        trace_hash = note[20:52]

        lib = ctypes.CDLL(str(SO))
        lib.selftest_native_elf_h.restype = ctypes.c_int64
        lib.selftest_native_elf_h.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        got = mind_elf(lib, FIXTURE.encode(), trace_hash)

        ok = got == rust
        print(
            f"  {'PASS' if ok else 'FAIL'}  add/main native ELF "
            f"({len(rust)} oracle bytes / {len(got)} mind bytes)"
        )
        if not ok:
            print(first_diverge(got, rust))
            return 1

        # Run the pure-MIND-emitted ELF; it must exit with the fixture's value.
        code = run_elf(got, tmp)
        run_ok = code == EXPECTED_EXIT
        print(
            f"  {'PASS' if run_ok else 'FAIL'}  pure-MIND ELF runs + exits "
            f"{code} (expected {EXPECTED_EXIT})"
        )
        if not run_ok:
            return 1

    print("\nALL PASS  (byte-identical to Rust mind-native + runs + exits 5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
