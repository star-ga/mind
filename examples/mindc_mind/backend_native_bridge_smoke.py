#!/usr/bin/env python3
"""RI-D slice 1 gate (task #110): `mindc build --backend native` is a byte-faithful
pass-through to the frozen pure-MIND native-ELF compiler (stage1.elf) — it adds ZERO
bytes, the emitted ELF runs correctly, output is run-to-run deterministic, it matches a
frozen pure-MIND byte anchor, it is FAIL-CLOSED (no MLIR fallback), and it invokes
NO MLIR/LLVM/clang binary at all — PROVEN, not asserted in a print (gate 6).

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
import shutil
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


def bridge_build(user_src: bytes, out: pathlib.Path, env_override=None, wrap=()):
    src = out.parent / (out.stem + "_prog.mind")
    src.write_bytes(user_src)
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"), MINDC_NATIVE_ELF=str(STAGE1))
    env.update(env_override or {})
    r = subprocess.run(
        [*wrap, str(MINDC), "build", "--backend", "native", str(src), "--out", str(out)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    data = out.read_bytes() if out.exists() else b""
    return r.returncode, data, r.stderr


# Binaries whose invocation would falsify "zero MLIR/LLVM/clang". mindc resolves
# every one of these through `which::which` (src/eval/mlir_build.rs resolve_tools),
# i.e. through PATH, which is exactly what the tripwire below intercepts.
TOOLCHAIN = (
    "mlir-opt", "mlir-translate", "mlir-cpu-runner", "clang", "clang-20",
    "clang++", "cc", "c++", "gcc", "g++", "ld", "ld.lld", "lld", "llc",
    "llvm-as", "llvm-link", "opt",
)


def toolchain_absence_proof(td: pathlib.Path, user_src: bytes, expect_bytes: bytes):
    """PROVE the native backend invoked no MLIR/LLVM/clang binary.

    This gate exists because the line it now backs used to be a bare string: the
    smoke printed "(zero MLIR/LLVM/clang)" while testing nothing of the kind, so a
    silent fallback to the MLIR path would have printed the same PASS. That is the
    same defect class as a `--test` selector naming a cfg-erased file — a claim in
    the output that no assertion stands behind.

    Method, in the order that makes it non-vacuous:
      1. A tripwire directory of executables named after every toolchain binary,
         each recording its own name and exiting non-zero.
      2. A CONTROL: resolve `clang` through that PATH via /bin/sh and require the
         tripwire to fire. Without this the whole proof could pass because the
         mechanism was broken (wrong PATH, non-executable shim, unwritable
         witness) rather than because the toolchain was never called — "no
         evidence of a call" would be indistinguishable from "no evidence at all".
      3. The real build, through the same PATH, with MLIR_OPT/MLIR_TRANSLATE/CLANG
         cleared so an env override cannot route around the tripwire. It must
         succeed, emit the SAME bytes as the unjailed build (so this is a proof
         about the real artifact, not a degraded path), and leave the witness empty.
      4. If `strace` works here, additionally require a POSITIVE execve count and
         zero toolchain execs. This is strictly stronger than the PATH proof — it
         also covers an absolute-path invocation that never consults PATH — but it
         needs ptrace, which many containers deny, so it upgrades the evidence
         when available and never weakens it when absent.

    Returns (failures, notes).
    """
    fails, notes = [], []
    jail = td / "toolchain_tripwire"
    jail.mkdir()
    witness = td / "toolchain_witness.txt"
    for name in TOOLCHAIN:
        shim = jail / name
        # The witness path is BAKED IN, not passed via the environment: mindc
        # scrubs parts of the environment it hands to build subprocesses, and a
        # tripwire that silently lost its witness path would record nothing and
        # read as proof of absence.
        shim.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$(basename "$0")" >> {witness}\n'
            "exit 97\n"
        )
        shim.chmod(0o755)

    jailed_path = f"{jail}:{os.environ.get('PATH', '')}"
    env_jail = {"PATH": jailed_path, "MLIR_OPT": "", "MLIR_TRANSLATE": "", "CLANG": ""}

    # (2) control — the tripwire must be able to fire at all.
    subprocess.run(
        ["/bin/sh", "-c", "clang --version >/dev/null 2>&1"],
        env=dict(os.environ, PATH=jailed_path),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    fired = witness.read_text().split() if witness.exists() else []
    if "clang" not in fired:
        fails.append(
            "toolchain-absence CONTROL did not fire: a `clang` call through the "
            f"tripwire PATH recorded {fired!r}. The absence proof below would be "
            "vacuous, so it is reported as a FAILURE rather than a pass."
        )
        return fails, notes
    notes.append("  ok    toolchain-absence control: tripwire fires on a real `clang` call")
    witness.write_text("")

    # (3) the real build, jailed.
    je, jd, jerr = bridge_build(user_src, td / "jailed.elf", env_override=env_jail)
    called = sorted(set((witness.read_text().split() if witness.exists() else [])))
    if je != 0 or not jd:
        fails.append(
            f"toolchain-absence: the jailed build FAILED (exit={je}, {len(jd)}B, "
            f"stderr={jerr[:200]!r}) — cannot claim 'zero MLIR/LLVM/clang' from a "
            f"build that did not happen."
        )
    elif jd != expect_bytes:
        fails.append(
            "toolchain-absence: the jailed build emitted DIFFERENT bytes than the "
            "normal build, so the proof would be about a degraded path, not the "
            "artifact this gate ships."
        )
    elif called:
        fails.append(
            f"toolchain-absence: the native backend EXECUTED {called} — the "
            f"'zero MLIR/LLVM/clang' claim is FALSE."
        )
    else:
        notes.append(f"  ok    toolchain-absence: {len(TOOLCHAIN)} tripwires armed, "
                     f"none fired, bytes identical to the unjailed build")

    # (4) optional, strictly stronger: syscall-level absence.
    if shutil.which("strace"):
        trace = td / "execve.log"
        se, sd, _ = bridge_build(
            user_src, td / "straced.elf",
            wrap=("strace", "-f", "-qq", "-e", "trace=execve", "-o", str(trace)),
        )
        lines = trace.read_text(errors="replace").splitlines() if trace.exists() else []
        execs = [l for l in lines if "execve(" in l]
        if se != 0 or not execs:
            notes.append(f"  note  strace leg unusable here (exit={se}, "
                         f"{len(execs)} execve records) — PATH proof stands alone")
        else:
            # strace renders the call as: execve("/usr/bin/clang", ["clang", ...
            # so the binary's own name is the tail of the first quoted path.
            hit = sorted({t for t in TOOLCHAIN for l in execs if f'/{t}"' in l})
            if hit:
                fails.append(f"toolchain-absence (strace): execve of {hit}")
            else:
                notes.append(f"  ok    toolchain-absence (strace): {len(execs)} execve "
                             f"call(s) traced, zero of them MLIR/LLVM/clang")
    else:
        notes.append("  note  strace not installed — absence proven via the PATH "
                     "tripwire only (an absolute-path call would escape it)")
    return fails, notes


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

        # 6. zero MLIR/LLVM/clang — PROVEN, not printed. Needs the good build's
        #    bytes from gate 1, so it runs last.
        if bd:
            tf, tn = toolchain_absence_proof(td, add, bd)
            for line in tn:
                print(line)
            fails.extend(tf)
        else:
            fails.append("toolchain-absence: skipped, gate 1 produced no bytes to "
                         "compare against — never report the claim as proven")

    if fails:
        print("FAIL  backend-native bridge gate:")
        for f in fails:
            print("   -", f)
        return 1
    print(
        "PASS  backend-native bridge: pass-through + functional + oracle + "
        "determinism + fail-closed + toolchain-absence (zero MLIR/LLVM/clang, "
        "proven by a control-verified tripwire, not asserted)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
