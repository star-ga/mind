#!/usr/bin/env python3
"""RI-D1 readiness gate (task #313): prove `mindc build --backend native` is READY to be
the default backend FOR A FROZEN PRODUCTION PROFILE — WITHOUT flipping the global default.

This is the evidence the RI dependency-cut matrix (docs/RI_DEPENDENCY_MATRIX.md rows 3-7)
requires before RI-D1 can flip the default for the supported subset. The existing
backend_native_bridge_smoke.py proves ONE program is a faithful pass-through and argues
zero-toolchain from the code path; this gate proves it PER RUN over a CORPUS via strace,
and proves the fail-closed boundary spawns no toolchain either (no silent MLIR fallback).

The frozen profile = exactly what the FROZEN pure-MIND compiler ELF (stage1.elf) actually
emits, NOT what current main.mind source can emit (stage1.elf is re-frozen only on RI-E1
reseed, so it lags newly-landed constructs — that lag is WHY the profile is defined by the
shipped ELF, which is what `--backend native` really runs).

Three claims, each asserted per program via strace -f -e trace=execve:
  IN-PROFILE  -> rc=0, ELF artifact, runs with expected exit, and the ONLY binaries
                 execve'd are {mindc, stage1.elf} (ZERO mlir-opt/mlir-translate/clang/ld).
  OUT-PROFILE -> rc!=0, NO artifact, error[backend-native] on stderr, and STILL zero
                 toolchain execve — proving refusal is fail-closed, never a silent MLIR
                 fallback (dependency removal must not be faked by capability regression).

Exit: 0 all pass; 1 a gate failed; 2 BLOCKED (missing mindc / stage1.elf / strace).
"""
import os
import re
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
MINDC = REPO / "target" / "release" / "mindc"
STAGE1 = HERE / "testdata" / "selfhost_loop" / "stage1.elf"

# Binaries the toolchain-free native path must NEVER spawn. A hit here means the build
# fell back to the MLIR pipeline — the exact regression RI-D1 must rule out.
TOOLCHAIN_BINS = ("mlir-opt", "mlir-translate", "clang", "ld.lld", "lld", "cc", "gcc")
# Only these two binaries are legitimate on the native path: the Rust driver that routes
# to the bridge, and the frozen pure-MIND compiler ELF it streams the source image to.
ALLOWED_BINS = ("mindc", "stage1.elf")

# Frozen-profile corpus: constructs the frozen stage1.elf provably emits (verified
# empirically 2026-08-21). Each returns its value as the process exit code (main()->i64).
IN_PROFILE = [
    ("int_arith", "fn main()->i64{return 7+35;}", 42),
    ("float_as_i64", "fn main()->i64{return (2.5+4.0) as i64;}", 6),
    ("struct_return",
     "struct P{a:i64,b:i64} fn mk()->P{return P{a:3,b:4};} "
     "fn main()->i64{let p=mk(); return p.a+p.b;}", 7),
    # u8 200+100 wraps to 44 (mod 256); `as i64` widens the already-wrapped value.
    ("narrow_u8_wrap", "fn main()->i64{let x:u8=200; let y:u8=100; return (x+y) as i64;}", 44),
    ("array_idx_loop",
     "fn main()->i64{let a=[1,2,3,4]; let mut s:i64=0; for i in 0..4 {s=s+a[i];} return s;}", 10),
]

# Out-of-profile: matrix-declared NOT in the native subset (row 11 tensor = NO / RI-E,
# row 12 trait dispatch = PARTIAL / RI-F). Must fail-closed with zero toolchain spawn.
OUT_PROFILE = [
    ("tensor", "fn main()->i64{let t=zeros([4]); return 0;}"),
    ("trait",
     "trait T{fn f(self)->i64;} struct S{} impl T for S{fn f(self)->i64{return 1;}} "
     "fn main()->i64{let s=S{}; return s.f();}"),
]

EXECVE_PATH = re.compile(r'execve\("([^"]+)"')


def strace_native_build(name, src, td):
    """Run `mindc build --backend native` under strace; return
    (rc, artifact_bytes, stderr_text, execve_basenames)."""
    srcf = td / f"{name}.mind"
    outf = td / f"{name}.elf"
    stf = td / f"{name}.strace"
    srcf.write_text(src)
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"), MINDC_NATIVE_ELF=str(STAGE1))
    with open(stf, "wb") as se:
        r = subprocess.run(
            ["strace", "-f", "-e", "trace=execve", str(MINDC), "build",
             "--backend", "native", str(srcf), "--out", str(outf)],
            env=env, stdout=subprocess.PIPE, stderr=se,
        )
    # Parse ONLY execve("<path>") tokens — never arbitrary text (mindc's own status line
    # says "zero MLIR/LLVM/clang", which a naive grep would false-positive on).
    strace_txt = stf.read_text(errors="replace")
    execs = sorted({pathlib.Path(p).name for p in EXECVE_PATH.findall(strace_txt)})
    artifact = outf.read_bytes() if outf.exists() else b""
    return r.returncode, artifact, strace_txt, execs


def toolchain_hits(execs):
    return [b for b in execs if any(t in b for t in TOOLCHAIN_BINS)]


def unexpected_bins(execs):
    return [b for b in execs if not any(b == a or a in b for a in ALLOWED_BINS)]


def main() -> int:
    for label, p in (("mindc", MINDC), ("stage1.elf", STAGE1)):
        if not p.exists():
            print(f"BLOCKED: {label} missing at {p}")
            return 2
    if subprocess.run(["sh", "-c", "command -v strace"], stdout=subprocess.DEVNULL).returncode != 0:
        print("BLOCKED: strace not installed (required to prove per-run zero-toolchain)")
        return 2

    # Positive-count floor (anti-false-green): a job that builds NOTHING also shows zero
    # toolchain execve. Pin the corpus size so a silently-deleted fixture is a FAILURE,
    # not a vacuous pass. Bump deliberately when the allowlist+corpus grow together.
    PINNED_IN_PROFILE = 5
    PINNED_OUT_PROFILE = 2
    if len(IN_PROFILE) < PINNED_IN_PROFILE or len(OUT_PROFILE) < PINNED_OUT_PROFILE:
        print(
            f"FAIL  corpus shrank below pinned floor: in={len(IN_PROFILE)}<{PINNED_IN_PROFILE} "
            f"or out={len(OUT_PROFILE)}<{PINNED_OUT_PROFILE} — a deleted fixture is a failure."
        )
        return 1

    fails = []
    built = 0
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        for name, src, expect_exit in IN_PROFILE:
            rc, art, _serr, execs = strace_native_build(name, src, td)
            th, ub = toolchain_hits(execs), unexpected_bins(execs)
            if rc != 0 or not art:
                fails.append(f"in/{name}: build rc={rc} artifact={len(art)}B (want rc=0, ELF)")
                continue
            if th:
                fails.append(f"in/{name}: SPAWNED TOOLCHAIN {th} — not toolchain-free")
                continue
            if ub:
                fails.append(f"in/{name}: unexpected execve {ub} (allowed: {ALLOWED_BINS})")
                continue
            runf = td / f"{name}.run.elf"
            runf.write_bytes(art)
            runf.chmod(0o755)
            got = subprocess.run([str(runf)]).returncode
            if got != expect_exit:
                fails.append(f"in/{name}: emitted ELF exit={got} expected {expect_exit}")
            else:
                print(f"  ok   in-profile  {name:16} rc=0 exit={got} execve={execs} 0-toolchain")

        for name, src in OUT_PROFILE:
            rc, art, serr, execs = strace_native_build(name, src, td)
            th = toolchain_hits(execs)
            problems = []
            if rc == 0:
                problems.append(f"rc=0 (want non-zero refusal)")
            if art:
                problems.append(f"produced {len(art)}B artifact (want none)")
            if th:
                problems.append(f"SPAWNED TOOLCHAIN {th} — SILENT MLIR FALLBACK")
            if "error[backend-native]" not in serr and "backend-native" not in serr:
                problems.append("no error[backend-native] diagnostic")
            if problems:
                fails.append(f"out/{name}: " + "; ".join(problems))
            else:
                print(f"  ok   out-profile {name:16} rc={rc} fail-closed, no artifact, 0-toolchain")

    if fails:
        print("\nFAIL  RI-D1 frozen-profile readiness gate:")
        for f in fails:
            print("   -", f)
        return 1
    print(
        f"\nPASS  RI-D1 readiness: {len(IN_PROFILE)}/{len(IN_PROFILE)} in-profile programs "
        f"build+run toolchain-free (execve ⊆ {{mindc, stage1.elf}}), "
        f"{len(OUT_PROFILE)}/{len(OUT_PROFILE)} out-of-profile fail-closed with no MLIR fallback.\n"
        "Native backend is READY to be default FOR THIS FROZEN PROFILE (default NOT flipped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
