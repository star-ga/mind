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
    # Row 10 FLOAT_LANGUAGE_COVERAGE — the float COMPARISON, formerly OUT_PROFILE. Native
    # used to lower it to `ucomisd` + a bare UNSIGNED setcc; an unordered compare sets
    # CF=ZF=PF=1, so `<`/`<=`/`==` read TRUE for a NaN and `!=` FALSE (measured: native
    # exit 3 vs MLIR exit 0 on this exact source). main.mind::nb_fp_setcc_opcode now
    # emits the IEEE-ordered forms (swapped-operand seta/setae for `<`/`<=`, sete AND
    # setnp for `==`, setne OR setp for `!=`). NaN comes from profile-only constructs:
    # six squarings of 65536.0 overflow to +inf and `+inf - +inf` is NaN.
    ("float_nan_compare",
     "fn main()->i64{let a:f64=65536.0; let b:f64=a*a; let c:f64=b*b; let d:f64=c*c; "
     "let e:f64=d*d; let g:f64=e*e; let h:f64=g*g; let n:f64=h-h; "
     "if n == 0.0 { if n < 0.0 { return 3; } return 1; } if n < 0.0 { return 2; } return 0;}",
     0),
]

# Float-comparison truth-table corpus. Each program asks ALL SIX predicates of one
# operand pair and returns them as a bitmask (the exit code):
#   lt=1  le=2  gt=4  ge=8  eq=16  ne=32
# The expected mask is computed from PYTHON's float comparisons (IEEE-754 binary64,
# NaN-ordered exactly like MLIR's arith.cmpf olt/ole/ogt/oge/oeq/une) — an oracle
# derived independently of both compiler backends. Each program is additionally built
# with the MLIR backend and must return the SAME exit (see `mlir_exit`): native == IEEE
# == MLIR, three ways. Special values come from profile-only constructs: `h` = +inf
# (six squarings of 65536.0), `n` = NaN (`h - h`), `ni` = -inf, `nz` = -0.0.
_FCMP_VALUES = {
    "n": float("nan"), "h": float("inf"), "ni": float("-inf"), "nz": -0.0,
    "1.0": 1.0, "1.5": 1.5, "3.0": 3.0, "0.0": 0.0,
    "m1": -1.0, "m2": -2.0, "m3": -3.0,
}
_FCMP_PRELUDE = (
    "let a:f64=65536.0; let b:f64=a*a; let c:f64=b*b; let d:f64=c*c; let e:f64=d*d; "
    "let g:f64=e*e; let h:f64=g*g; let n:f64=h-h; let ni:f64=0.0-h; "
    "let m1:f64=0.0-1.0; let m2:f64=0.0-2.0; let m3:f64=0.0-3.0; let nz:f64=0.0*m1; "
)
_FCMP_FN = (
    "fn k(x:f64, y:f64)->i64{let mut r:i64=0; if x<y {r=r+1;} if x<=y {r=r+2;} "
    "if x>y {r=r+4;} if x>=y {r=r+8;} if x==y {r=r+16;} if x!=y {r=r+32;} return r;} "
)
# (fixture name, lhs, rhs): NaN on each side and both, NaN vs inf, ordinary finite
# (incl. two negatives — the #168 sign-magnitude case), +/-0.0, and +/-inf.
_FCMP_PAIRS = [
    ("fcmp_nan_lhs", "n", "1.0"),
    ("fcmp_nan_rhs", "1.0", "n"),
    ("fcmp_nan_both", "n", "n"),
    ("fcmp_nan_vs_inf", "n", "h"),
    ("fcmp_inf_vs_nan", "ni", "n"),
    ("fcmp_finite_lt", "1.0", "1.5"),
    ("fcmp_finite_gt", "3.0", "m3"),
    ("fcmp_finite_eq", "1.5", "1.5"),
    ("fcmp_two_negs", "m2", "m1"),
    ("fcmp_signed_zero", "nz", "0.0"),
    ("fcmp_zero_signed", "0.0", "nz"),
    ("fcmp_ninf_pinf", "ni", "h"),
    ("fcmp_inf_eq", "h", "h"),
    ("fcmp_inf_vs_fin", "h", "3.0"),
]


def _fcmp_mask(x: float, y: float) -> int:
    return ((x < y) * 1 + (x <= y) * 2 + (x > y) * 4 + (x >= y) * 8
            + (x == y) * 16 + (x != y) * 32)


FCMP_PROGRAMS = [
    (name, _FCMP_FN + "fn main()->i64{" + _FCMP_PRELUDE + f"return k({lhs}, {rhs});}}",
     _fcmp_mask(_FCMP_VALUES[lhs], _FCMP_VALUES[rhs]))
    for name, lhs, rhs in _FCMP_PAIRS
]
IN_PROFILE += FCMP_PROGRAMS
# Every float-comparison program (the truth table + the historical NaN program) must
# ALSO agree with the MLIR backend — this is the cross-backend half of the claim.
MLIR_CROSSCHECK = {name for name, _s, _e in FCMP_PROGRAMS} | {"float_nan_compare"}

# Out-of-profile: matrix-declared NOT in the native subset (row 11 tensor = NO / RI-E,
# row 12 trait dispatch = PARTIAL / RI-F). Must fail-closed with zero toolchain spawn.
OUT_PROFILE = [
    # Each fixture declares WHICH fence must refuse it. Without this the gate only
    # checked `rc != 0` and the substring "backend-native" — which BOTH fences print —
    # so the Rust-side first fence had ZERO negative coverage distinguishable from the
    # frozen ELF's own refusal. A first fence that defers to the second is not a fence.
    #   "first"  -> the Rust frozen-profile predicate must name the construct
    #   "second" -> the pure-MIND compiler must be the one that rejects
    # enforces: RI-D1-PROFILE
    # These two fixtures are the TEST for the Rust-side frozen-profile fence. Both are
    # accepted by the pure-MIND compiler and refused ONLY by the predicate, so deleting
    # the fence makes them build instead of exiting 3 — which is exactly what the
    # enforcement/test pairing lint requires: a rule whose removal something notices.
    ("div_first_fence", "fn main()->i64{let a:i64=7; let b:i64=2; return a / b;}", "first"),
    ("shr_first_fence", "fn main()->i64{let x:i64=256; return x >> 2;}", "first"),
    # A declared f32 lowers to the same ConstF64 + Lt IR as f64, so only the source-level
    # half of the fence (profile_frozen_admits_source) can name it. The pure-MIND compiler
    # refuses it too, without a name; this row fails if the first fence stops naming it.
    ("f32_compare_first_fence",
     "fn main()->i64{let x:f32=1.0; let y:f32=2.0; if x < y {return 1;} return 0;}", "first"),
    ("tensor", "fn main()->i64{let t=zeros([4]); return 0;}", "first"),
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


def mlir_exit(name, src, td):
    """Build `src` with the DEFAULT (MLIR) backend and run it; return (exit, error).
    The oracle half of the float-comparison claim: the native artifact must answer
    exactly what `arith.cmpf` answers. A failed MLIR build is reported, never skipped —
    a cross-check that did not run is not a cross-check that passed."""
    srcf = td / f"{name}_mlir.mind"
    outf = td / f"{name}_mlir.elf"
    srcf.write_text(src)
    env = dict(os.environ, MINDC_STD_DIR=str(REPO / "std"))
    env.pop("MINDC_NATIVE_ELF", None)
    r = subprocess.run([str(MINDC), "build", str(srcf), "--out", str(outf)],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not outf.exists():
        tail = r.stderr.decode(errors="replace").strip().splitlines()[-1:]
        return None, f"MLIR build rc={r.returncode} {tail}"
    outf.chmod(0o755)
    return subprocess.run([str(outf)]).returncode, None


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
    PINNED_IN_PROFILE = 20
    PINNED_OUT_PROFILE = 3
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
                continue
            if name in MLIR_CROSSCHECK:
                mgot, merr = mlir_exit(name, src, td)
                if merr is not None:
                    fails.append(f"in/{name}: MLIR cross-check did not run: {merr}")
                    continue
                if mgot != got:
                    fails.append(f"in/{name}: native exit={got} != MLIR exit={mgot}")
                    continue
                print(f"  ok   in-profile  {name:16} rc=0 exit={got} == MLIR {mgot} "
                      f"execve={execs} 0-toolchain")
                continue
            print(f"  ok   in-profile  {name:16} rc=0 exit={got} execve={execs} 0-toolchain")

        for entry in OUT_PROFILE:
            # A fixture may declare WHICH fence must refuse it. Optional, so existing
            # two-field fixtures keep working; when present it is asserted.
            if len(entry) == 3:
                name, src, want_fence = entry
            else:
                name, src = entry
                want_fence = None
            rc, art, serr, execs = strace_native_build(name, src, td)
            th = toolchain_hits(execs)
            problems = []
            if rc == 0:
                problems.append(f"rc=0 (want non-zero refusal)")
            # `rc != 0` alone cannot tell a REFUSAL from a DEATH: a process killed by
            # a signal reports a negative returncode, which reads as "non-zero, so it
            # fail-closed". Refusing a construct is the behaviour this gate certifies;
            # segfaulting on it is a defect that happens to look identical from here
            # (mindc can emit the diagnostic and then die on the way out, satisfying
            # the stderr check below).
            elif rc < 0:
                problems.append(
                    f"KILLED BY SIGNAL {-rc} — a crash is not a fail-closed refusal"
                )
            if art:
                problems.append(f"produced {len(art)}B artifact (want none)")
            if th:
                problems.append(f"SPAWNED TOOLCHAIN {th} — SILENT MLIR FALLBACK")
            if "error[backend-native]" not in serr and "backend-native" not in serr:
                problems.append("no error[backend-native] diagnostic")
            # WHICH fence refused. Both print "backend-native", so without this the
            # first fence's coverage is indistinguishable from the second's — and a
            # first fence whose every case is also caught downstream has proven
            # nothing. The Rust predicate NAMES the construct; the pure-MIND compiler
            # says it rejected the program.
            if want_fence == "first":
                if "out-of-profile construct:" not in serr:
                    problems.append(
                        "expected the FIRST fence (the Rust frozen-profile predicate) to "
                        "refuse and name the construct, but it did not — this fixture "
                        "gives the first fence no coverage"
                    )
            elif want_fence == "second":
                if "out-of-profile construct:" in serr:
                    problems.append(
                        "expected the SECOND fence (the pure-MIND compiler) to refuse, "
                        "but the Rust predicate caught it first — the fixture no longer "
                        "exercises the compiler's own fail-closed path"
                    )
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
        f"{len(OUT_PROFILE)}/{len(OUT_PROFILE)} out-of-profile fail-closed with no MLIR fallback; "
        f"{len(MLIR_CROSSCHECK)} float-comparison programs native == IEEE == MLIR.\n"
        "Native backend is READY to be default FOR THIS FROZEN PROFILE (default NOT flipped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
