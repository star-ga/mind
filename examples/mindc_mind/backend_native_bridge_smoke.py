#!/usr/bin/env python3
"""RI-D slice 1 gate (task #110): `mindc build --backend native` is a byte-faithful
pass-through to the frozen pure-MIND native-ELF compiler (stage1.elf) — it adds ZERO
bytes, the emitted ELF runs correctly, output is run-to-run deterministic, it matches a
frozen pure-MIND byte anchor, and it is FAIL-CLOSED (no MLIR fallback).

This is NOT gated against native_elf_oracle/ — that is the retired old-Rust src/native
backend's snapshot (deleted in #15). The canonical RI-D reference is the pure-MIND
stage1.elf output; see testdata/backend_native_bridge/MANIFEST.txt for provenance and
the characterized old-Rust divergence.

Exit: 0 all pass; 1 a gate failed; 2 BLOCKED (missing mindc / stage1.elf).
"""
import os
import struct
import subprocess
import hashlib
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
MINDC = REPO / "target" / "release" / "mindc"
STAGE1 = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
ORACLE = HERE / "testdata" / "backend_native_bridge" / "add.elf"
# Twin of the Rust bridge's STD_MODULES and self_host_standalone_driver_smoke.py's
# _STDLIB_MODULES. A drift here vs the bridge surfaces as a pass-through FAIL.
STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring", "json", "map",
    "net", "process", "reactor", "regex", "ring", "sha256", "string", "time", "toml",
    "tui", "vec",
]
ADD_PROG = (
    "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"
    "fn main() -> i64 {\n    return add(2, 3);\n}\n"
)
ADD_EXPECT_EXIT = 5


def std_blob() -> bytes:
    return b"\n".join((REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


def stage1_direct(user_src: bytes):
    std = std_blob()
    comb = std + user_src
    img = struct.pack("<qq", len(std), len(comb)) + comb
    r = subprocess.run([str(STAGE1)], input=img, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode, r.stdout


def bridge_build(user_src: bytes, out: pathlib.Path):
    src = out.parent / (out.stem + "_prog.mind")
    src.write_bytes(user_src)
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"), MINDC_NATIVE_ELF=str(STAGE1))
    r = subprocess.run(
        [str(MINDC), "build", "--backend", "native", str(src), "--out", str(out)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    data = out.read_bytes() if out.exists() else b""
    return r.returncode, data, r.stderr


def main() -> int:
    if not MINDC.exists():
        print(f"BLOCKED: mindc not built at {MINDC}")
        return 2
    if not STAGE1.exists():
        print(f"BLOCKED: stage1.elf missing at {STAGE1}")
        return 2

    add = ADD_PROG.encode()
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        # 1. pass-through: bridge output == stage1.elf-direct output (zero bytes added).
        de, do = stage1_direct(add)
        be, bd, berr = bridge_build(add, td / "add.elf")
        dh, bh = hashlib.sha256(do).hexdigest(), hashlib.sha256(bd).hexdigest()
        if be != 0 or de != 0 or not bd or bd != do:
            fails.append(
                f"pass-through: bridge_exit={be} direct_exit={de} bridge_sha={bh} "
                f"direct_sha={dh} stderr={berr[:200]!r}"
            )
        else:
            print(f"  ok    pass-through: bridge == stage1-direct ({len(bd)}B sha={bh})")

        # 2. functional: the emitted ELF runs with the expected exit code.
        if bd:
            p = td / "run.elf"
            p.write_bytes(bd)
            p.chmod(0o755)
            rc = subprocess.run([str(p)]).returncode
            if rc != ADD_EXPECT_EXIT:
                fails.append(f"functional: emitted ELF exit={rc} expected {ADD_EXPECT_EXIT}")
            else:
                print(f"  ok    functional: emitted ELF exit={rc}")

        # 3. frozen pure-MIND byte anchor (catches stage1.elf drift).
        if ORACLE.exists():
            oh = hashlib.sha256(ORACLE.read_bytes()).hexdigest()
            if oh != bh:
                fails.append(f"oracle: bridge sha {bh} != frozen pure-MIND oracle {oh}")
            else:
                print(f"  ok    oracle: bridge == frozen pure-MIND anchor ({oh})")
        else:
            fails.append(f"oracle: missing {ORACLE}")

        # 4. run-to-run determinism of the pure-MIND compiler.
        _, do2 = stage1_direct(add)
        if do2 != do:
            fails.append("determinism: stage1.elf output differs run-to-run")
        else:
            print("  ok    determinism: stage1.elf run-to-run identical")

        # 5. fail-closed: a program the pure-MIND compiler cannot emit must yield a
        #    non-zero exit and NO artifact (never a silent MLIR fallback).
        bad = b"fn main( { not valid mind @@@ "
        fe, fd, ferr = bridge_build(bad, td / "bad.elf")
        if fe == 0 or fd:
            fails.append(
                f"fail-closed: garbage produced exit={fe} bytes={len(fd)} "
                f"(must be non-zero + no artifact)"
            )
        else:
            print(f"  ok    fail-closed: garbage -> exit={fe}, no artifact")

    if fails:
        print("FAIL  backend-native bridge gate:")
        for f in fails:
            print("   -", f)
        return 1
    print(
        "PASS  backend-native bridge: pass-through + functional + oracle + "
        "determinism + fail-closed (zero MLIR/LLVM/clang)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
