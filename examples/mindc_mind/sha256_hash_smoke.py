#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0.
# Part of the MIND project (Machine Intelligence Native Design).
#
# Project-mode smoke for std.sha256's `hash(bytes) -> bytes[32]` wrapper.
#
# The substrate-object link path (project/mod.rs) only fires under `mindc build`
# (project mode), NOT single-file `--emit-shared`, so this must build a real
# Mind.toml project that imports std.sha256. It hashes b"abc" and checks the
# digest byte-for-byte against the known SHA-256("abc"), confirming both the
# bytes[32] (i64-handle) return ABI and the growable-`bytes` input decode.
#
# Run: python3 examples/mindc_mind/sha256_hash_smoke.py

import ctypes
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

MINDC = Path(__file__).resolve().parents[2] / "target" / "release" / "mindc"
if not MINDC.exists():
    MINDC = Path(__file__).resolve().parents[2] / "target" / "debug" / "mindc"

MAIN_MIND = """import std.sha256

pub fn digest_byte(i: i64) -> i64 {
    let mut b: bytes = []
    b.push(97)
    b.push(98)
    b.push(99)
    let d = sha256.hash(b)
    return __mind_load_i8(d + i) & 255
}
"""

MIND_TOML = """[package]
name = "sha256_hash_smoke"
version = "0.1.0"

[build]
entry = "src/main.mind"
output = "sha256_hash_smoke"
emit = "cdylib"

[targets.cpu]
backend = "cpu"
sources = ["src/main.mind"]
"""


def main() -> int:
    if not MINDC.exists():
        print("sha256-hash-smoke: mindc not found; skipping")
        return 0
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        (proj / "src").mkdir()
        (proj / "Mind.toml").write_text(MIND_TOML)
        (proj / "src" / "main.mind").write_text(MAIN_MIND)

        out = subprocess.run(
            [str(MINDC), "build", "--emit", "cdylib"],
            cwd=proj,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            stderr = out.stderr
            if "mlir-build" in stderr and "requires" in stderr:
                print("sha256-hash-smoke: needs mlir-build; skipping")
                return 0
            print("sha256-hash-smoke: mindc build failed:\n" + stderr)
            return 1

        so = next(proj.glob("target/**/libsha256_hash_smoke.so"), None)
        if so is None:
            print("sha256-hash-smoke: output .so not found")
            return 1

        lib = ctypes.CDLL(str(so))
        lib.digest_byte.restype = ctypes.c_int64
        lib.digest_byte.argtypes = [ctypes.c_int64]
        got = bytes(lib.digest_byte(i) & 0xFF for i in range(32))
        want = hashlib.sha256(b"abc").digest()
        if got != want:
            print(f"sha256-hash-smoke: MISMATCH\n got: {got.hex()}\nwant: {want.hex()}")
            return 1

    print(f"sha256-hash-smoke: PASS — sha256.hash(b'abc') == {want.hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
