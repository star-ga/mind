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

FIVE claims, each asserted per program via strace -f -e trace=execve:
  IN-PROFILE  -> rc=0, ELF artifact, runs with expected exit, and the ONLY binaries
                 execve'd are {mindc, stage1.elf} (ZERO mlir-opt/mlir-translate/clang/ld).
  OUT-PROFILE -> rc!=0, NO artifact, error[backend-native] on stderr, the DECLARED fence
                 refuses it NAMING the declared construct, and STILL zero toolchain execve
                 — proving refusal is fail-closed, never a silent MLIR fallback
                 (dependency removal must not be faked by capability regression).
  DEFERRED    -> a program the frozen ELF provably COMPILES but fence 1 currently
                 OVER-rejects. Asserted refused-with-the-expected-construct and pinned by
                 EXACT COUNT, so the known gap can neither grow silently nor shrink
                 silently. See "THE DEFERRED LIST" below.
  ELF-LAG     -> the OPPOSITE porosity: a program fence 1 ADMITS that the frozen
                 stage1.elf cannot compile. Asserted admitted-by-fence-1 and refused by
                 the ELF, also pinned by exact count — this is the gap between what the
                 allowlist promises and what the shipped artifact delivers, so it is the
                 one a readiness claim must not lose track of.
  CONTROL     -> every unsigned OUT-PROFILE negative is paired with an i64 twin in
                 IN_PROFILE that differs ONLY in the type annotation, so a refusal is
                 attributable to the unsigned taint rather than to the program's shape.

THE DEFERRED LIST (2026-09-16). Three programs that shipped in IN_PROFILE — float_as_i64,
struct_return, narrow_u8_wrap — are refused by fence 1 as `call.undefined_or_builtin`:
they lower to compiler-synthesised intrinsic calls (`__mind_conv_i64`, `__mind_alloc` /
`__mind_store_i64` / `__mind_load_i64`) and `admit_call` admits only callees DEFINED in the
module. The frozen ELF compiles all three; the Rust predicate is simply stricter than the
corpus. That is OVER-rejection: fail-closed, loud, recoverable — the safe direction — but
it means the allowlist is not in bijection with the corpus, which is what kept this gate
RED. Deleting the three fixtures would have turned the gate green by forgetting the gap.
Instead they move here, where each one is a POSITIVE assertion about the current fence.

Relaxing `admit_call` to admit those intrinsics by NAME is deliberately NOT the fix: the
corpus proves `__mind_store_i64` for an i64 struct field only, and an f64 field is a
different native store path — a name-keyed admission repeats the exact constructor-vs-
denotation defect this file's history is made of. The sound form needs (callee,
operand_type) keying, i.e. `FnDef.value_types` threaded into the predicate (task #313).
When that lands, each fixture moves BACK to IN_PROFILE with a real expected exit — and
until then this gate goes RED if anyone quietly admits one, because PINNED_DEFERRED is an
EXACT count, not a floor.

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


CORPUS_TSV = HERE / "testdata" / "signed_div_mod_edge_corpus.tsv"


def load_shared_corpus():
    """The signed i64 div/mod edge corpus, shared with the MLIR-backend parity test
    (tests/signed_div_mod_cross_backend_parity_run.rs) and with the interpreter unit
    tests. Returns [(name, src, expected_exit)].

    Reading the corpus instead of restating it is the point: "native == MLIR" asserted
    over two hand-maintained lists is a claim about two lists, and this repo's most
    expensive recurring defect is exactly a second list nothing forces to agree with the
    first. A fixture here can only be edited for every tier at once.
    """
    out = []
    for lineno, raw in enumerate(CORPUS_TSV.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            raise SystemExit(
                f"{CORPUS_TSV}:{lineno}: expected 4 TAB-separated fields, got {len(parts)}"
            )
        name, expect, kind, payload = (p.strip() for p in parts)
        if kind == "file":
            payload = (HERE / "testdata" / payload).read_text()
        elif kind != "src":
            raise SystemExit(f"{CORPUS_TSV}:{lineno}: unknown kind '{kind}' (want src|file)")
        out.append((name, payload, int(expect)))
    return out


# Frozen-profile corpus: constructs the frozen stage1.elf provably emits (verified
# empirically 2026-08-21, extended 2026-09-16). Each returns its value as the process
# exit code (main()->i64), so every expected value is in 0..255 — a signal death reports
# a NEGATIVE returncode and is therefore always distinguishable from a legitimate answer.
IN_PROFILE = [
    ("int_arith", "fn main()->i64{return 7+35;}", 42),
    ("array_idx_loop",
     "fn main()->i64{let a=[1,2,3,4]; let mut s:i64=0; for i in 0..4 {s=s+a[i];} return s;}", 10),

    # ---- SIGNEDNESS CONTROLS for the OUT_PROFILE unsigned negatives ----
    # Each is the i64 TWIN of an unsigned negative below, differing ONLY in the type
    # annotation. Without them, "the u64 program was refused" is not evidence that the
    # UNSIGNED TAINT refused it — a typo, an unsupported shape or an unrelated fence
    # would look identical. With them, the pair isolates the one variable.
    ("ctl_i64_param_div",
     "fn d(a:i64,b:i64)->i64{return a / b;} fn main()->i64{return d(84,2);}", 42),
    ("ctl_i64_ret_mod",
     "fn m(a:i64,b:i64)->i64{return a % b;} fn main()->i64{return m(47,5);}", 2),
    ("ctl_i64_param_lt",
     "fn lt(a:i64,b:i64)->i64{return a < b;} fn main()->i64{return lt(3,5);}", 1),
]

# ---- signed i64 div/mod edge corpus + the Tier-0 RH totality drivers ----
# Admitted 2026-09-16 (arch-reviewed). `div_first_fence` used to sit in OUT_PROFILE as
# the first fence's negative coverage; signed i64 Div is now ADMITTED, so it is promoted
# here with its value (as `div_basic`) and the first fence keeps `shr_first_fence` plus
# the four unsigned negatives as its coverage.
#
# The Tier-0 drivers come in as three fixtures, not one: a verifier that only ever
# answers "all clear" proves neither that its loop ran nor that a FAILING verdict is
# reachable. The digest is the miscompile canary; the budget fixture forces the
# counterexample arm. See testdata/rh_totality_tier0_verdict.mind for the contract.
IN_PROFILE += load_shared_corpus()

# Programs the frozen stage1.elf COMPILES but fence 1 currently OVER-rejects. Not a
# silent deletion and not a pass — each is asserted to be refused by the FIRST fence
# naming the EXPECTED construct, and the list is pinned by EXACT count. See the module
# docstring, "THE DEFERRED LIST".
DEFERRED_OVER_REJECTED = [
    # `(2.5+4.0) as i64` -> `__mind_conv_i64`, a callee not defined in the module.
    ("float_as_i64", "fn main()->i64{return (2.5+4.0) as i64;}", "call.undefined_or_builtin"),
    # Aggregate ABI: `__mind_alloc` / `__mind_store_i64` / `__mind_load_i64`.
    ("struct_return",
     "struct P{a:i64,b:i64} fn mk()->P{return P{a:3,b:4};} "
     "fn main()->i64{let p=mk(); return p.a+p.b;}", "call.undefined_or_builtin"),
    # u8 200+100 wraps to 44 (mod 256); the `as i64` widening is the synthesised call.
    ("narrow_u8_wrap",
     "fn main()->i64{let x:u8=200; let y:u8=100; return (x+y) as i64;}",
     "call.undefined_or_builtin"),
]

# Out-of-profile: matrix-declared NOT in the native subset (row 11 tensor = NO / RI-E,
# row 12 trait dispatch = PARTIAL / RI-F). Must fail-closed with zero toolchain spawn.
#
# Each fixture declares WHICH fence must refuse it, and — for a first-fence refusal —
# WHICH construct the predicate must name. Without the fence field the gate only checked
# `rc != 0` and the substring "backend-native", which BOTH fences print, so the Rust-side
# first fence had ZERO negative coverage distinguishable from the frozen ELF's own
# refusal. A first fence that defers to the second is not a fence. Without the CONSTRUCT
# field, any first-fence refusal satisfied any first-fence fixture — so a predicate that
# rejected a u64 division for the wrong reason (or rejected every program) still passed.
#   "first"  -> the Rust frozen-profile predicate must refuse and name `construct`
#   "second" -> the pure-MIND compiler must be the one that rejects (`construct` is None)
# enforces: RI-D1-PROFILE
OUT_PROFILE = [
    # The TEST for the Rust-side frozen-profile fence: accepted by the pure-MIND
    # compiler and refused ONLY by the predicate, so deleting the fence makes these
    # build instead of exiting 3 — exactly what the enforcement/test pairing lint
    # requires: a rule whose removal something notices.
    ("shr_first_fence", "fn main()->i64{let x:i64=256; return x >> 2;}", "first", "binop.shr"),
    # `zeros([4])` LOWERS TO A CALL, so it is refused by `admit_call` and never reaches
    # the `Instr::ConstTensor` arm — measured, not assumed. That means this fixture on
    # its own gives the tensor.* rejection arms NO coverage, which is why the annotated
    # form below is also pinned: `let t: Tensor[f32,(2,3)] = 0` does reach `const-tensor`.
    ("tensor_ctor_call", "fn main()->i64{let t=zeros([4]); return 0;}",
     "first", "call.undefined_or_builtin"),
    ("tensor_const", "fn main()->i64{let t: Tensor[f32,(2,3)] = 0; return 0;}",
     "first", "const-tensor"),

    # ---- UNSIGNED negatives (issue #99 family; admitted-Div/Mod counterweight) ----
    # The native emitter has ONLY signed `idiv` and signed `setcc`. On a full-width
    # unsigned operand that silently disagrees with MLIR's `divui` / `setb` — measured
    # live for `u64 <` (native 1 vs MLIR 0). Signed i64 Div/Mod became admitted on
    # 2026-09-16; these four are what keeps that admission SIGNED-ONLY. Each has an
    # i64 twin in IN_PROFILE (`ctl_*`) proving the shape itself is fine.
    ("u64_param_div",
     "fn d(a:u64,b:u64)->u64{return a / b;} fn main()->i64{return 0;}",
     "first", "binop.div_mod_unsigned"),
    ("u64_ret_mod",
     "fn m(a:i64,b:i64)->u64{return a % b;} fn main()->i64{return 0;}",
     "first", "binop.div_mod_unsigned"),
    ("u64_param_lt",
     "fn lt(a:u64,b:u64)->i64{return a < b;} fn main()->i64{return 0;}",
     "first", "binop.ordered_compare_unsigned"),
    # The `let x: u64 = y as u64` door: no u64 appears in any fn signature, so the
    # signature taint cannot see it. It is closed instead by `admit_call` refusing the
    # synthesised `__mind_conv_u64` — the LOAD-BEARING coupling documented in
    # frozen_profile.rs. This fixture is what notices if that refusal is ever relaxed
    # without extending the taint, which would silently re-open the unsigned-div hole.
    ("u64_cast_then_div",
     "fn main()->i64{let y:i64=84; let x:u64=y as u64; let z:u64=2 as u64; "
     "let q:u64=x / z; return 0;}",
     "first", "call.undefined_or_builtin"),

    ("trait",
     "trait T{fn f(self)->i64;} struct S{} impl T for S{fn f(self)->i64{return 1;}} "
     "fn main()->i64{let s=S{}; return s.f();}", "second", None),
    # Row 10 FLOAT_LANGUAGE_COVERAGE — the float COMPARISON, which the two backends
    # answer DIFFERENTLY. Native lowers it to `ucomisd` + an UNSIGNED setcc
    # (main.mind::nb_fp_setcc_opcode), and an unordered compare sets CF=ZF=PF=1, so
    # `<` / `<=` / `==` read TRUE for a NaN operand and `!=` reads FALSE. MLIR lowers
    # it to `arith.cmpf "olt"/"ole"/"oeq"` + `"une"` — the ORDERED IEEE predicates,
    # which are false for NaN (true for `!=`).
    #
    # This source reaches NaN using ONLY profile constructs (f64 literal, `*`, `-`):
    # six squarings of 65536.0 overflow to +inf, and `+inf - +inf` is NaN. It then asks
    # two questions no real number can both answer yes: `n == 0.0` AND `n < 0.0`.
    # Measured before the rejection landed: native exit 3 (BOTH true — the
    # ucomisd-unordered signature), MLIR exit 0. Both fences admitted it, so the native
    # artifact carried a silently wrong value. Per-operator: Lt/Le/Eq/Ne diverge,
    # Gt/Ge agree.
    #
    # Gate meaning: this MUST stay fail-closed until `nb_fp_setcc_opcode` grows a
    # NaN-aware predicate (parity masking for ==/!=, swapped-operand seta/setae for
    # </<=). Only then may it move to IN_PROFILE, with an expected exit of 0 matching
    # the MLIR path.
    ("float_nan_compare",
     "fn main()->i64{let a:f64=65536.0; let b:f64=a*a; let c:f64=b*b; let d:f64=c*c; "
     "let e:f64=d*d; let g:f64=e*e; let h:f64=g*g; let n:f64=h-h; "
     "if n == 0.0 { if n < 0.0 { return 3; } return 1; } if n < 0.0 { return 2; } return 0;}",
     "first", "binop.compare_in_float_module"),
]

# Programs fence 1 ADMITS but the FROZEN stage1.elf cannot compile ("unsupported
# construct") — the opposite porosity from DEFERRED_OVER_REJECTED, and the one that
# actually matters for RI-D1 readiness: it is the gap between what the allowlist
# promises and what the shipped ELF delivers. Fence 2 catches all three today, so
# nothing is unsafe; what would be unsafe is FORGETTING them, because the allowlist
# would then read as a capability claim the artifact cannot honour.
#
# MEASURED against the shipped stage1.elf, 2026-09-16. Each of these is a construct
# root's L70-MACH totality driver would naturally reach for, which is why they are
# pinned here rather than left as folklore:
#   * `return` inside a `while` — the "found a counterexample, stop now" shape;
#   * `&&` / `||` in a condition — the obvious way to write a bounded sweep;
#   * an `else` whose branches do not both `return` — the obvious parity split.
# `testdata/rh_totality_tier0_verdict.mind` documents the workaround it uses for each.
#
# Pinned by EXACT count: a stage1.elf reseed that LIFTS one must promote it to
# IN_PROFILE with a real expected exit rather than quietly deleting the fixture, and a
# reseed that BREAKS something new must add it here deliberately.
DEFERRED_FROZEN_ELF_LAG = [
    ("elf_lag_loop_early_return",
     "fn main()->i64{let mut i:i64=0; while i < 5 { if i > 2 { return 7; } i = i + 1; } return 0;}"),
    ("elf_lag_short_circuit_and",
     "fn main()->i64{let a:i64=1; let b:i64=2; if a == 1 && b == 2 { return 5; } return 0;}"),
    ("elf_lag_else_join",
     "fn main()->i64{let x:i64=3; let mut y:i64=0; if x > 2 { y = 1; } else { y = 2; } return y;}"),
]

EXECVE_PATH = re.compile(r'execve\("([^"]+)"')
# The first fence's diagnostic shape (src/bin/mindc.rs): "... (out-of-profile construct: X)".
FIRST_FENCE_CONSTRUCT = re.compile(r"out-of-profile construct: ([A-Za-z0-9_.\-]+)")


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


def check_refusal(name, src, td, want_fence, want_construct):
    """Assert one fail-closed refusal. Returns a list of problem strings (empty = ok).

    Shared by OUT_PROFILE and DEFERRED_OVER_REJECTED: a deferred over-rejection and a
    declared out-of-profile rejection are the SAME observable event, and the difference
    between them is bookkeeping about intent, not about behaviour. Sharing the checker
    means a deferred fixture is held to exactly the standard a real negative is.
    """
    rc, art, serr, execs = strace_native_build(name, src, td)
    problems = []
    if rc == 0:
        problems.append("rc=0 (want non-zero refusal)")
    # `rc != 0` alone cannot tell a REFUSAL from a DEATH: a process killed by a signal
    # reports a negative returncode, which reads as "non-zero, so it fail-closed".
    # Refusing a construct is the behaviour this gate certifies; segfaulting on it is a
    # defect that happens to look identical from here (mindc can emit the diagnostic and
    # then die on the way out, satisfying the stderr check below).
    elif rc < 0:
        problems.append(f"KILLED BY SIGNAL {-rc} — a crash is not a fail-closed refusal")
    if art:
        problems.append(f"produced {len(art)}B artifact (want none)")
    th = toolchain_hits(execs)
    if th:
        problems.append(f"SPAWNED TOOLCHAIN {th} — SILENT MLIR FALLBACK")
    if "error[backend-native]" not in serr and "backend-native" not in serr:
        problems.append("no error[backend-native] diagnostic")

    # WHICH fence refused, and — for the first fence — WHICH construct it named. Both
    # fences print "backend-native", so without the fence check the first fence's
    # coverage is indistinguishable from the second's; and without the construct check
    # ANY first-fence refusal satisfies ANY first-fence fixture, so a predicate that
    # refused everything (or refused this program for an unrelated reason) still passed.
    named = FIRST_FENCE_CONSTRUCT.search(serr)
    if want_fence == "first":
        if not named:
            problems.append(
                "expected the FIRST fence (the Rust frozen-profile predicate) to refuse "
                "and name the construct, but it did not — this fixture gives the first "
                "fence no coverage"
            )
        elif want_construct is not None and named.group(1) != want_construct:
            problems.append(
                f"first fence named construct '{named.group(1)}' but this fixture is the "
                f"coverage for '{want_construct}' — it is being refused for the wrong "
                "reason, so the rule it is supposed to test is untested"
            )
    elif want_fence == "second":
        if named:
            problems.append(
                f"expected the SECOND fence (the pure-MIND compiler) to refuse, but the "
                f"Rust predicate caught it first as '{named.group(1)}' — the fixture no "
                "longer exercises the compiler's own fail-closed path"
            )
    return problems, rc, execs


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
    PINNED_IN_PROFILE = 18
    PINNED_OUT_PROFILE = 9
    # EXACT, not floors. Both lists are known GAPS, so neither may grow silently (more
    # refusal = capability regression) nor shrink silently (something became buildable,
    # which owes an IN_PROFILE entry with a real expected exit before anyone may call it
    # proven). A gap that can be closed by deleting its fixture is not tracked.
    PINNED_DEFERRED = 3
    PINNED_ELF_LAG = 3
    if len(IN_PROFILE) < PINNED_IN_PROFILE or len(OUT_PROFILE) < PINNED_OUT_PROFILE:
        print(
            f"FAIL  corpus shrank below pinned floor: in={len(IN_PROFILE)}<{PINNED_IN_PROFILE} "
            f"or out={len(OUT_PROFILE)}<{PINNED_OUT_PROFILE} — a deleted fixture is a failure."
        )
        return 1
    if len(DEFERRED_OVER_REJECTED) != PINNED_DEFERRED:
        print(
            f"FAIL  deferred over-rejection list is {len(DEFERRED_OVER_REJECTED)}, pinned at "
            f"{PINNED_DEFERRED}. Growing it is a capability regression; shrinking it means a "
            "construct became admitted and owes an IN_PROFILE entry with an expected exit."
        )
        return 1
    if len(DEFERRED_FROZEN_ELF_LAG) != PINNED_ELF_LAG:
        print(
            f"FAIL  frozen-ELF lag list is {len(DEFERRED_FROZEN_ELF_LAG)}, pinned at "
            f"{PINNED_ELF_LAG}. A reseed that LIFTS one must promote it to IN_PROFILE with "
            "an expected exit; a reseed that breaks something new must add it deliberately."
        )
        return 1

    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)

        for name, src, expect_exit in IN_PROFILE:
            rc, art, serr, execs = strace_native_build(name, src, td)
            th, ub = toolchain_hits(execs), unexpected_bins(execs)
            if rc != 0 or not art:
                # Quote mindc's OWN diagnostic, not the last line of the stream: stderr
                # here is the strace log, whose final line is always "+++ exited with N
                # +++" — which says nothing about why.
                why = next(
                    (ln.strip() for ln in serr.splitlines() if "error[backend-native]" in ln),
                    "(no error[backend-native] line in the strace log)",
                )
                fails.append(
                    f"in/{name}: build rc={rc} artifact={len(art)}B (want rc=0, ELF)\n"
                    f"        {why}"
                )
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
                # A negative returncode is a signal death, not an answer — name it, so a
                # SIGFPE from an unguarded idiv can never read as "returned the wrong int".
                how = f"died by signal {-got}" if got < 0 else f"exit={got}"
                fails.append(f"in/{name}: emitted ELF {how}, expected exit {expect_exit}")
            else:
                print(f"  ok   in-profile  {name:20} rc=0 exit={got} execve={execs} 0-toolchain")

        for name, src, want_construct in DEFERRED_OVER_REJECTED:
            problems, rc, _execs = check_refusal(name, src, td, "first", want_construct)
            if problems:
                fails.append(f"deferred/{name}: " + "; ".join(problems))
            else:
                print(f"  ok   DEFERRED    {name:20} rc={rc} over-rejected as "
                      f"'{want_construct}' (known gap, pinned)")

        for name, src in DEFERRED_FROZEN_ELF_LAG:
            # want_fence="second": fence 1 must ADMIT it (no "out-of-profile construct:"
            # in the diagnostic) and the frozen ELF must be the one refusing. If fence 1
            # ever starts catching one of these, the fixture stops measuring the lag and
            # the gate says so instead of quietly passing.
            problems, rc, _execs = check_refusal(name, src, td, "second", None)
            if problems:
                fails.append(f"elf-lag/{name}: " + "; ".join(problems))
            else:
                print(f"  ok   ELF-LAG     {name:20} rc={rc} admitted by fence 1, "
                      "refused by the frozen ELF (known lag, pinned)")

        for name, src, want_fence, want_construct in OUT_PROFILE:
            problems, rc, _execs = check_refusal(name, src, td, want_fence, want_construct)
            if problems:
                fails.append(f"out/{name}: " + "; ".join(problems))
            else:
                where = want_construct if want_construct else f"{want_fence} fence"
                print(f"  ok   out-profile {name:20} rc={rc} fail-closed ({where}), "
                      "no artifact, 0-toolchain")

    if fails:
        print("\nFAIL  RI-D1 frozen-profile readiness gate:")
        for f in fails:
            print("   -", f)
        return 1
    print(
        f"\nPASS  RI-D1 readiness: {len(IN_PROFILE)}/{len(IN_PROFILE)} in-profile programs "
        f"build+run toolchain-free (execve ⊆ {{mindc, stage1.elf}}), "
        f"{len(OUT_PROFILE)}/{len(OUT_PROFILE)} out-of-profile fail-closed with no MLIR "
        f"fallback and the declared fence naming the declared construct, "
        f"{len(DEFERRED_OVER_REJECTED)}/{PINNED_DEFERRED} known fence-1 over-rejections and "
        f"{len(DEFERRED_FROZEN_ELF_LAG)}/{PINNED_ELF_LAG} known frozen-ELF lags pinned.\n"
        "Native backend is READY to be default FOR THIS FROZEN PROFILE (default NOT flipped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
