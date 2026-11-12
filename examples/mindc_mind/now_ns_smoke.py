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
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# MINDC_BIN first: every other gate in this corpus honours it, and CI points it
# at the freshly built binary. Reading only the in-tree target/ path meant this
# gate could not be aimed at a real compiler at all — and then took the skip
# branch below and reported success.
_ENV_MINDC = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
MINDC = Path(_ENV_MINDC) if _ENV_MINDC else (
    Path(__file__).resolve().parents[2] / "target" / "release" / "mindc")
if not MINDC.exists() and not _ENV_MINDC:
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
        # FAIL CLOSED. This used to `print(... skipping); return 0`, so the gate
        # reported success on a tree with no compiler at all — it asserted
        # nothing and said so in a line that reads like a pass. A gate that
        # cannot run its subject must go red and name what is missing.
        print(f"now-ns-smoke: FAILED — no mindc at {MINDC}; point MINDC_BIN at a "
              "built binary. A missing compiler is a red gate, never a green one.")
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
                # FAIL CLOSED, same reason as the missing-binary branch above: a
                # mindc built without `mlir-build` cannot lower this program, so
                # nothing was checked. Returning 0 here made a toolchain gap
                # indistinguishable from a verified clock intrinsic.
                print("now-ns-smoke: FAILED — mindc was built without the "
                      "`mlir-build` feature, so std.time could not be lowered and "
                      "nothing was verified. Rebuild with --features mlir-build.")
                return 1
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
