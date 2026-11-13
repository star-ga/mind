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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Honour the corpus-wide MINDC_BIN / MINDC handles before the in-tree path.
# Reading only target/release/mindc made this gate impossible to aim at a
# specific compiler, and made it report success against whatever binary
# happened to be lying in the tree — including when the caller had said,
# explicitly, that no compiler was available.
_ENV_MINDC = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
MINDC = Path(_ENV_MINDC) if _ENV_MINDC else (
    Path(__file__).resolve().parents[2] / "target" / "release" / "mindc")
if not MINDC.exists() and not _ENV_MINDC:
    MINDC = Path(__file__).resolve().parents[2] / "target" / "debug" / "mindc"

ENV_SKIP_OPT_IN = "MIND_SMOKE_ALLOW_ENV_SKIP"


def _env_skip_allowed() -> bool:
    """True only for a bare local run that explicitly opted in to skipping on a
    missing toolchain. A harness run (CI or fast_keystone.sh) always sets
    MINDC_SO/MINDC_BIN, and that DOMINATES the opt-in, so a stray
    MIND_SMOKE_ALLOW_ENV_SKIP leaking into a CI environment still cannot re-open
    the hole. Same idiom as mod_operator_smoke.py."""
    if os.environ.get("MINDC_SO") or os.environ.get("MINDC_BIN"):
        return False
    return os.environ.get(ENV_SKIP_OPT_IN) == "1"


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
    # Both skip paths below used to `return 0` unconditionally. This smoke runs
    # inside a ci.yml batch loop whose only vacuous-green backstop is
    # `grep -q '^SKIP'` on the output -- and neither message starts with "SKIP",
    # so a missing mindc or a misread build failure graded as a passing gate with
    # nothing to catch it. Fail closed unless a bare local run opted in.
    if not MINDC.exists():
        if _env_skip_allowed():
            print(f"SKIP: sha256-hash-smoke: mindc not found ({ENV_SKIP_OPT_IN}=1)")
            return 0
        print(f"sha256-hash-smoke: FAIL - mindc not found at {MINDC} - refusing to "
              f"skip; the sha256.hash ABI assertion did not run. Build it, or set "
              f"{ENV_SKIP_OPT_IN}=1 for a bare local run with neither MINDC_SO nor "
              f"MINDC_BIN set.")
        return 1
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td)
        (proj / "src").mkdir()
        (proj / "Mind.toml").write_text(MIND_TOML)
        (proj / "src" / "main.mind").write_text(MAIN_MIND)

        out = subprocess.run(
            [str(MINDC), "build", "--backend", "mlir", "--emit", "cdylib"],
            cwd=proj,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            stderr = out.stderr
            if "mlir-build" in stderr and "requires" in stderr:
                if _env_skip_allowed():
                    print(f"SKIP: sha256-hash-smoke: needs mlir-build "
                          f"({ENV_SKIP_OPT_IN}=1)")
                    return 0
                print("sha256-hash-smoke: FAIL - mindc lacks the mlir-build feature "
                      "- refusing to skip; the sha256.hash ABI assertion did not "
                      f"run. Set {ENV_SKIP_OPT_IN}=1 for a bare local run.\n"
                      + stderr)
                return 1
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
