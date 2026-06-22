#!/usr/bin/env python3
# Copyright 2025 STARGA Inc.
# Licensed under the Apache License, Version 2.0.
# Part of the MIND project (Machine Intelligence Native Design).
#
# Project-mode smoke for std.time's `now_ns() -> i64` (the __mind_now_ns runtime
# clock intrinsic). The substrate-object link path only fires under `mindc build`
# (project mode), so this builds a real Mind.toml project that imports std.time.
# now_ns is explicitly non-deterministic (wall-clock evidence timestamps); the
# check only asserts it links, runs, and returns a value plausibly near the host
# wall clock — NOT a fixed value.
#
# Run: python3 examples/mindc_mind/now_ns_smoke.py

import ctypes
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MINDC = Path(__file__).resolve().parents[2] / "target" / "release" / "mindc"
if not MINDC.exists():
    MINDC = Path(__file__).resolve().parents[2] / "target" / "debug" / "mindc"

MAIN_MIND = """import std.time

pub fn get_ns() -> i64 {
    return time.now_ns()
}
"""

MIND_TOML = """[package]
name = "now_ns_smoke"
version = "0.1.0"

[build]
entry = "src/main.mind"
output = "now_ns_smoke"
emit = "cdylib"

[targets.cpu]
backend = "cpu"
sources = ["src/main.mind"]
"""


def main() -> int:
    if not MINDC.exists():
        print("now-ns-smoke: mindc not found; skipping")
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
                print("now-ns-smoke: needs mlir-build; skipping")
                return 0
            print("now-ns-smoke: mindc build failed:\n" + stderr)
            return 1

        so = next(proj.glob("target/**/libnow_ns_smoke.so"), None)
        if so is None:
            print("now-ns-smoke: output .so not found")
            return 1

        lib = ctypes.CDLL(str(so))
        lib.get_ns.restype = ctypes.c_int64
        got = lib.get_ns()
        wall = int(time.time() * 1e9)
        # Plausible: positive and within 10s of the host wall clock.
        if got <= 0 or abs(got - wall) > 10_000_000_000:
            print(f"now-ns-smoke: implausible now_ns={got} (wall={wall})")
            return 1

    print(f"now-ns-smoke: PASS — now_ns() = {got} (~wall clock)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
