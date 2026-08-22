#!/usr/bin/env python3
"""RH f64-aggregate surface canary (RH_REQUIRED_F64_AGGREGATE_SURFACE).

Compiles `testdata/rh_f64_aggregate_canary.mind` — a deterministic LDL^T-style
elimination on a 4x4 rational SPD matrix whose entries AND every factor are
exactly representable in IEEE-754 binary64 — and asserts the process exit code
is the bit-exact expected result, 37. The canary exercises the full fixed
`[f64; N]` aggregate surface: local arrays, runtime-indexed load + store, store
inside loops visible after, flat `m[i*n+j]` matrix with nested scalar loops, a
read-only fixed `[f64; 4]` parameter helper, and f64 loop-carried accumulators
(add/sub/mul/div, no fast-math / reassociation).

Two authorities, per the RH directive:
  * MLIR  — the normal Rust-built `mindc` (LLVM/MLIR path) via `--emit=cdylib`,
    then the compiled `main` is called through ctypes and its bit-exact result
    asserted. cdylib (not `--emit=binary`) because it is RH's actual consumer
    path (`--emit=shared`, cross-module) AND the self-host CI job builds a
    cdylib-capable mindc with no linked `cpu` runtime for a full executable.
    This leg is REQUIRED: a wrong value or a compile failure fails the smoke.
  * NATIVE — the pure-MIND self-host / native-ELF backend (`--backend native`,
    the frozen stage1.elf). The fixed-array *store* (`a[i] = v`) and by-value
    fixed-array *parameter* are not yet ported to the native emitter / mic@3
    wire format (no OP_ARRAY_STORE opcode), so this leg is EXPECTED to fail
    CLOSED today: the smoke asserts it either produces the exact value 37 (once
    the native store + array-param + reseed land) OR refuses cleanly with a
    non-zero exit and NO artifact. A silent wrong-value native artifact fails
    the smoke. When the native leg starts producing 37, flip
    NATIVE_REQUIRED = True to make it a hard gate.

Usage:
  MINDC_BIN=./target/release/mindc \
  [MINDC_NATIVE_ELF=examples/mindc_mind/testdata/selfhost_loop/stage1.elf] \
  [MINDC_STD_DIR=std] \
      python3 examples/mindc_mind/rh_f64_aggregate_canary_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parents[1]
_CANARY = _HERE / "testdata" / "rh_f64_aggregate_canary.mind"
MINDC = os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc"))
EXPECTED = 37
NATIVE_REQUIRED = False  # flip True once the native store + array-param + reseed land


def _is_elf(p: pathlib.Path) -> bool:
    return p.exists() and p.stat().st_size >= 256 and p.read_bytes()[:4] == b"\x7fELF"


def _run(p: pathlib.Path) -> int:
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], timeout=30).returncode


def _mlir_leg(tmp: pathlib.Path) -> bool:
    so = tmp / "canary_mlir.so"
    r = subprocess.run(
        [MINDC, "build", str(_CANARY), "--release", "--emit=cdylib", "--out", str(so)],
        capture_output=True, text=True, timeout=180,
    )
    if not so.exists() or so.stat().st_size < 256:
        print(f"FAIL  MLIR(cdylib): no artifact (rc={r.returncode})\n{r.stderr[-400:]}")
        return False
    try:
        lib = ctypes.CDLL(str(so))
    except OSError as e:
        print(f"FAIL  MLIR(cdylib): could not load artifact: {e}")
        return False
    if not hasattr(lib, "main"):
        print("FAIL  MLIR(cdylib): exported `main` symbol absent")
        return False
    lib.main.restype = ctypes.c_int64
    lib.main.argtypes = []
    got = lib.main()
    ok = got == EXPECTED
    print(f"[{'PASS' if ok else 'FAIL'}] MLIR  `mindc build --emit=cdylib` main()={got} "
          f"want={EXPECTED}")
    return ok


def _native_leg(tmp: pathlib.Path) -> bool:
    elf = os.environ.get(
        "MINDC_NATIVE_ELF",
        str(_REPO / "examples" / "mindc_mind" / "testdata" / "selfhost_loop" / "stage1.elf"),
    )
    if not os.path.exists(elf):
        print(f"SKIP  NATIVE: stage1.elf not found ({elf})")
        return not NATIVE_REQUIRED
    env = dict(os.environ)
    env["MINDC_NATIVE_ELF"] = elf
    env.setdefault("MINDC_STD_DIR", str(_REPO / "std"))
    out = tmp / "canary_native.elf"
    r = subprocess.run(
        [MINDC, "build", str(_CANARY), "--backend", "native", "--out", str(out)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if _is_elf(out):
        got = _run(out)
        ok = got == EXPECTED
        print(f"[{'PASS' if ok else 'FAIL'}] NATIVE `--backend native` exit={got} "
              f"want={EXPECTED}")
        return ok
    # No artifact: the native backend must fail CLOSED (non-zero, no ELF), never
    # emit a silent wrong-value artifact. That is correct behaviour TODAY.
    closed = r.returncode != 0 and not out.exists()
    tag = "PASS" if (closed and not NATIVE_REQUIRED) else "FAIL"
    note = "fail-closed (store/array-param not yet ported to native)" if closed \
        else "did NOT fail closed"
    print(f"[{tag}] NATIVE `--backend native` refused — {note}"
          f"{' [REQUIRED]' if NATIVE_REQUIRED else ' [pending leg]'}")
    return closed and not NATIVE_REQUIRED


def main() -> int:
    if not os.path.exists(MINDC):
        print(f"SKIP  mindc not built: {MINDC}")
        return 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        mlir_ok = _mlir_leg(tmp)
        native_ok = _native_leg(tmp)
    if mlir_ok and native_ok:
        print("ALL PASS  RH f64-aggregate surface: MLIR path bit-exact (37); native "
              "leg accounted for (fail-closed until store+array-param+reseed land).")
        return 0
    print("FAIL  RH f64-aggregate canary — see legs above; do NOT guess.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
