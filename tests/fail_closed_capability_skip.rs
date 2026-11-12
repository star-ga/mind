// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Fail-CLOSED contract for the shared capability-skip helper.
//!
//! # The defect this gate defends against
//!
//! 24 integration-test sites carried the shape
//!
//! ```text
//! if !status.success() { println!("... build failed -> skipped"); return; }
//! ```
//!
//! with no panic and no `MIND_BENCH_REQUIRE` consultation. On a tier that
//! installs and PATH-verifies the MLIR toolchain, the only remaining way that
//! branch fires is a real compiler regression — which then graded as a PASS.
//! A determinism gate, eleven std-surface runtime-value gates and nine build
//! and cache gates were all fail-OPEN in exactly this way.
//!
//! Every assertion below is on `common::gate`, the single helper those sites
//! now route through, so the contract is proven once instead of 24 times.
//! The legitimate capability skip — a host genuinely lacking `mlir-build` or
//! the `mlir-opt`/`clang` binaries — is asserted to SURVIVE, because removing
//! it would be a different bug, not a fix.
//!
//! The BACKLOG of unrouted sites is drained:
//! `tests/fail_open_skip_site_ratchet.rs` now holds the tree to a FLAT
//! prohibition (floor zero, no backlog table). This file owns the helper's
//! CONTRACT; that one owns its COVERAGE.

mod common;

use common::gate::{self, Outcome};
use std::path::Path;
use std::process::Command;

// Every fixture below is the VERBATIM stderr of a real `mindc` refusal, and
// `fixture_codes_are_the_compilers_own` pins each one's code token against the
// compiler's own constant — so a code change cannot leave these fixtures
// asserting about a wire shape that no longer exists.

/// Verbatim stderr of `mindc --emit-shared` built without `mlir-build`
/// (`src/bin/mindc.rs`). A capability gap: this binary has no native backend.
const CAP_FEATURE: &str =
    "error[build][E5003]: --emit-shared requires building with the 'mlir-build' feature\n";

/// Verbatim stderr shape when the feature is on but the toolchain binary is
/// absent from PATH (`MlirBuildError::ToolMissing`, `src/eval/mlir_build.rs`).
const CAP_TOOL: &str = "error[build]: [E5004] tool not found: mlir-opt\n";

/// Verbatim stderr of the `mindc build` PROJECT route on a binary with no
/// native backend (`src/build/mod.rs`). Same condition as `CAP_FEATURE`,
/// completely different wording — which is exactly why the classifier may not
/// read wording. Before the cause code existed this refusal was graded a
/// compiler regression and PANICKED on every host without `mlir-build`,
/// reddening `mindc_cache_phase_f` in four live CI commands.
const CAP_PROJECT_BUILD: &str = "error[build][E5003]: entry module was not natively compiled \
     (embedded as a runtime-JIT fallback -- see the [WARN] above); refusing to report a \
     successful build for an artifact that is a launcher deferring to the installed \
     mind-runtime, which may exit 0 without executing your program\n";

/// The SAME prose as `CAP_PROJECT_BUILD`, carrying the real-failure cause code:
/// the module did not compile. Prose is identical, verdict is opposite — the
/// negative twin that proves the classifier reads the code.
const REAL_PROJECT_BUILD: &str = "error[build][E5005]: entry module was not natively compiled \
     (embedded as a runtime-JIT fallback -- see the [WARN] above); refusing to report a \
     successful build for an artifact that is a launcher deferring to the installed \
     mind-runtime, which may exit 0 without executing your program\n";

/// Verbatim stderr of `mindc x.mind --target gpu` (`pipeline.rs`): this build
/// carries no backend for the requested target, a HOST-capability fact. Its
/// code sits in the reserved cause namespace, but was not REGISTERED as a
/// cause until the namespace reservation was made mechanical — so the
/// classifier read it as an unknown cause, vetoed its own skip, and graded a
/// capability gap as a compiler regression on every backend-less host.
const CAP_TARGET_BACKEND: &str = "error[backend][E6002]: no backend available for target gpu\n";

/// An ordinary user error that used to occupy the cause namespace and now sits
/// outside it (`E6xxx`): an invalid `Mind.toml [exports] c_abi` entry. It must
/// neither forge a capability verdict nor veto one.
const REAL_MANIFEST_EXPORT: &str = "error[manifest][E6010]: invalid Mind.toml [exports] c_abi entry `bad name`: \
     not a C identifier\n";

/// A capability-SOUNDING refusal with no cause code at all — the pre-fix wire
/// shape. An uncoded refusal must fail closed.
const UNCODED_PROSE: &str =
    "error[build]: --emit-shared requires building with the 'mlir-build' feature\n";

/// A real compiler regression: the class that must NEVER grade as a pass.
const REAL_FAILURE: &str = "error[E0308]: mismatched types in `idiv`\n";

/// Verbatim stderr SHAPE of `mindc build` over a WORKSPACE (`run_workspace_build`,
/// `src/bin/mindc.rs`): the loop prints one `error[workspace][<member>][<code>]:`
/// per failing member and CONTINUES, so ONE stderr can carry a host-capability
/// refusal for one member and a REAL source failure for another. The verdict
/// must be the real failure — the same fail-closed merge `FallbackReason::merge`
/// already applies one layer down, which a "does any capability code appear?"
/// scan of the wire silently disagreed with.
const MIXED_WORKSPACE: &str = "error[workspace][core][E5003]: entry module was not natively \
     compiled (embedded as a runtime-JIT fallback -- see the [WARN] above)\n\
     error[workspace][tools][E5005]: entry module was not natively compiled (embedded as a \
     runtime-JIT fallback -- see the [WARN] above)\n";

/// The words of a capability refusal SCATTERED across unrelated lines: the
/// feature name in a hint, "requires" in an unrelated type error. Nothing here
/// refuses for a host-capability reason, so the words must not add up to a skip.
const SCATTERED_PROSE: &str = "error[E0308]: `mm` requires operands of the same rank\n\
     note: this build has no 'mlir-build' backend compiled in; \
     rebuild with --features mlir-build\n";

/// VERBATIM stderr of `mindc build` on a project whose SOURCE spells a cause
/// code: `fn main() -> i64 { let arr: [i64; 2] = [1, 2]; arr[E5003] }`.
///
/// Every `E5003` here is USER text — an identifier quoted back in the message
/// and the echoed source snippet — and the build failed for an entirely real
/// reason. A classifier that harvests any bracketed token from anywhere on the
/// wire reads a capability cause here and hands the program a free pass, so a
/// program can FORGE its own skip just by naming a code.
const FORGED_BY_SOURCE_TEXT: &str = "\
error[type-check][E2002]: unknown identifier `E5003`\n\
  --> /tmp/.tmpzY1E0G/src/main.mind:3:9\n\
   |     arr[E5003]\n\
   |         ^^^^^\n\
error[build]: /tmp/.tmpzY1E0G/src/main.mind: unresolved identifier(s) — refusing \
to embed a module that would crash at lowering\n";

/// A genuine host-capability refusal sharing one stderr with a real failure
/// that carries an ORDINARY diagnostic code (`E2002`), not a cause code.
///
/// No ordinary diagnostic is minted inside the cause namespace, so a classifier
/// that only merges CAUSE tokens cannot see this failure at all and grades the
/// run a tolerated skip. An `error[...]` header with no cause code is an
/// undiagnosed refusal, and undiagnosed fails closed.
const CAP_BESIDE_UNCODED_REAL: &str = "\
error[build][E5003]: entry module was not natively compiled (embedded as a \
runtime-JIT fallback)\n\
error[type-check][E2002]: unknown identifier `helper`\n";

// --- the pure decision core -------------------------------------------------

#[test]
fn success_is_compiled() {
    assert_eq!(gate::classify(true, "", false), Outcome::Compiled);
    assert_eq!(gate::classify(true, "", true), Outcome::Compiled);
}

#[test]
fn genuine_capability_gap_still_skips_when_not_enforcing() {
    // MUST NOT CHANGE: a host genuinely lacking the toolchain keeps skipping.
    assert_eq!(
        gate::classify(false, CAP_FEATURE, false),
        Outcome::CapabilitySkip
    );
    assert_eq!(
        gate::classify(false, CAP_TOOL, false),
        Outcome::CapabilitySkip
    );
}

#[test]
fn fixture_codes_are_the_compilers_own() {
    use libmind::diagnostics::capability as cap;
    let no_backend = format!("[{}]", cap::NO_NATIVE_BACKEND);
    let no_tool = format!("[{}]", cap::NATIVE_TOOLCHAIN_ABSENT);
    let real = format!("[{}]", cap::SOURCE_NOT_NATIVELY_COMPILABLE);
    assert!(CAP_FEATURE.contains(&no_backend), "{CAP_FEATURE}");
    assert!(
        CAP_PROJECT_BUILD.contains(&no_backend),
        "{CAP_PROJECT_BUILD}"
    );
    assert!(CAP_TOOL.contains(&no_tool), "{CAP_TOOL}");
    assert!(REAL_PROJECT_BUILD.contains(&real), "{REAL_PROJECT_BUILD}");
    assert!(!UNCODED_PROSE.contains('[') || !UNCODED_PROSE.contains("E50"));
    // The mixed fixture must carry BOTH a capability cause and the real one,
    // or it stops testing the merge it exists for.
    assert!(MIXED_WORKSPACE.contains(&no_backend), "{MIXED_WORKSPACE}");
    assert!(MIXED_WORKSPACE.contains(&real), "{MIXED_WORKSPACE}");
    let no_target_backend = format!("[{}]", cap::TARGET_BACKEND_UNAVAILABLE);
    assert!(
        CAP_TARGET_BACKEND.contains(&no_target_backend),
        "{CAP_TARGET_BACKEND}"
    );
    // The manifest fixture must carry NO cause code: an ordinary user error
    // inside the cause namespace is the defect this pins shut.
    assert!(
        cap::cause_codes(REAL_MANIFEST_EXPORT).is_empty(),
        "{REAL_MANIFEST_EXPORT}"
    );
    // The scattered fixture must carry NO cause code at all: its whole point is
    // that prose alone decides nothing.
    for code in [&no_backend, &no_tool, &real] {
        assert!(
            !SCATTERED_PROSE.contains(code.as_str()),
            "{SCATTERED_PROSE}"
        );
    }
}

#[test]
fn project_build_capability_gap_skips_when_not_enforcing() {
    // MUST NOT CHANGE: `mindc build` on a host with no native backend is a
    // capability gap, not a compiler regression.
    assert_eq!(
        gate::classify(false, CAP_PROJECT_BUILD, false),
        Outcome::CapabilitySkip
    );
}

#[test]
fn project_build_capability_gap_fails_under_enforcement() {
    match gate::classify(false, CAP_PROJECT_BUILD, true) {
        Outcome::Failed(s) => assert!(s.contains("not natively compiled")),
        other => panic!("MIND_BENCH_REQUIRE=1 must forbid a skip, got {other:?}"),
    }
}

#[test]
fn identical_prose_with_the_real_failure_code_never_skips() {
    // The whole point of coding the cause: these two differ ONLY in the code.
    match gate::classify(false, REAL_PROJECT_BUILD, false) {
        Outcome::Failed(s) => assert!(s.contains("E5005")),
        other => panic!("a module that did not compile must fail closed, got {other:?}"),
    }
}

#[test]
fn a_capability_code_beside_a_real_failure_code_fails_closed() {
    // A workspace build prints one refusal per member and keeps going, so both
    // causes reach one stderr. "Any capability code present" graded that as a
    // tolerated SKIP and the real failure passed — the same fail-open the coded
    // classifier was introduced to close, one level up.
    match gate::classify(false, MIXED_WORKSPACE, false) {
        Outcome::Failed(s) => assert!(s.contains("E5005"), "{s}"),
        other => panic!("a real failure beside a capability gap must fail closed, got {other:?}"),
    }
}

#[test]
fn an_unavailable_target_backend_skips_when_not_enforcing() {
    // A host/build capability gap: `--target gpu` on a build with no GPU
    // backend. Grading it `Failed` panics the call site with "this is a
    // compiler regression" — the forbidden outcome.
    assert_eq!(
        gate::classify(false, CAP_TARGET_BACKEND, false),
        Outcome::CapabilitySkip
    );
}

#[test]
fn an_unavailable_target_backend_fails_under_enforcement() {
    // Still a gap, but a tier that demands a real backend may not skip it.
    match gate::classify(false, CAP_TARGET_BACKEND, true) {
        Outcome::Failed(s) => assert!(s.contains("no backend available"), "{s}"),
        other => panic!("MIND_BENCH_REQUIRE=1 must forbid a skip, got {other:?}"),
    }
}

#[test]
fn an_ordinary_manifest_error_is_never_a_capability_skip() {
    // The other direction of the same reservation: a user error renumbered out
    // of the cause namespace must fail closed, and must never be readable as a
    // refusal CAUSE.
    use libmind::diagnostics::capability as cap;
    match gate::classify(false, REAL_MANIFEST_EXPORT, false) {
        Outcome::Failed(s) => assert!(s.contains("E6010"), "{s}"),
        other => panic!("an invalid manifest entry must fail closed, got {other:?}"),
    }
    assert!(
        cap::cause_codes(REAL_MANIFEST_EXPORT).is_empty(),
        "an out-of-namespace user error must not forge a refusal cause"
    );
    // It cannot forge a capability verdict — but it DOES veto one. A build that
    // refused for a real reason is a real failure, whatever else shares the
    // wire; grading it a tolerated skip was the fail-open this gate exists for.
    let beside = format!("{REAL_MANIFEST_EXPORT}{CAP_FEATURE}");
    match gate::classify(false, &beside, false) {
        Outcome::Failed(s) => assert!(s.contains("E6010"), "{s}"),
        other => panic!(
            "a real manifest error sharing the wire with a capability gap must \
             fail closed, got {other:?}"
        ),
    }
}

#[test]
fn source_text_that_spells_a_cause_code_can_never_forge_a_skip() {
    // A cause code is POSITIONAL: it counts only in a diagnostic header, never
    // in a message body, a `-->` path or an echoed snippet. Otherwise any
    // program that names `E5003` buys itself a pass from every gate.
    use libmind::diagnostics::capability as cap;
    assert!(
        cap::cause_codes(FORGED_BY_SOURCE_TEXT).is_empty(),
        "source text was read as a refusal cause: {:?}",
        cap::cause_codes(FORGED_BY_SOURCE_TEXT)
    );
    match gate::classify(false, FORGED_BY_SOURCE_TEXT, false) {
        Outcome::Failed(s) => assert!(s.contains("E2002"), "{s}"),
        other => panic!(
            "a program that spells a cause code in its own source forged a \
             capability skip, got {other:?}"
        ),
    }
}

#[test]
fn an_uncoded_error_header_beside_a_capability_cause_fails_closed() {
    // The merge must see refusals it has no code for. An ordinary type error
    // sharing the wire with a genuine capability refusal is still a real
    // failure of this build, so the verdict is `Failed`, never a skip.
    match gate::classify(false, CAP_BESIDE_UNCODED_REAL, false) {
        Outcome::Failed(s) => assert!(s.contains("E2002"), "{s}"),
        other => panic!(
            "an uncoded real failure sharing the wire with a capability cause \
             graded as a skip, got {other:?}"
        ),
    }
}

#[test]
fn scattered_capability_prose_is_never_a_skip() {
    // The pre-code classifier tested two INDEPENDENT substrings against the
    // whole stderr, so a hint naming the feature plus an unrelated "requires"
    // graded as a capability gap. Prose decides nothing; only a cause code does.
    match gate::classify(false, SCATTERED_PROSE, false) {
        Outcome::Failed(s) => assert!(s.contains("E0308"), "{s}"),
        other => panic!("scattered prose must never grade as a capability gap, got {other:?}"),
    }
}

#[test]
fn an_uncoded_capability_sounding_refusal_fails_closed() {
    // Prose is not the contract. A refusal that was never given a cause code
    // is undiagnosed, and undiagnosed fails closed.
    match gate::classify(false, UNCODED_PROSE, false) {
        Outcome::Failed(_) => {}
        other => panic!("an uncoded refusal must fail closed, got {other:?}"),
    }
}

#[test]
fn real_compiler_failure_is_never_a_skip() {
    // The whole point: today this printed "skipping" and the test PASSED.
    match gate::classify(false, REAL_FAILURE, false) {
        Outcome::Failed(s) => assert!(s.contains("mismatched types")),
        other => panic!("a real compiler failure must fail closed, got {other:?}"),
    }
}

#[test]
fn empty_stderr_failure_is_never_a_skip() {
    // `.status()` sites captured no stderr at all; an empty diagnostic must
    // not be mistaken for a capability gap.
    match gate::classify(false, "", false) {
        Outcome::Failed(_) => {}
        other => panic!("an undiagnosed failure must fail closed, got {other:?}"),
    }
}

#[test]
fn enforcement_forbids_even_a_genuine_capability_skip() {
    // MIND_BENCH_REQUIRE=1 -> the exec tier cannot pass vacuously.
    for stderr in [CAP_FEATURE, CAP_TOOL] {
        match gate::classify(false, stderr, true) {
            Outcome::Failed(_) => {}
            other => panic!("MIND_BENCH_REQUIRE=1 must forbid a skip, got {other:?}"),
        }
    }
}

// --- the call-site wrapper the 24 converted sites use -----------------------
//
// `gate::compiled_with`'s four end-to-end tests spawn a REAL stub compiler, and
// the stub is a POSIX shell script. They therefore live in
// `tests/fail_closed_capability_skip_stub_exec.rs`, which carries a file-scope
// `#![cfg(unix)]`: `ci.yml`'s `build_test` matrix also runs `windows-latest`,
// where exec'ing a non-PE image fails with ERROR_BAD_EXE_FORMAT (os error 193).
// Everything in THIS file is platform-independent and runs on all four rows —
// which is why the split is by portability, not by subject.
// `tests/harness_portability.rs` keeps that boundary mechanically.

// --- the capability-probe marker (which::which / mlir_available sites) ------

#[test]
fn probe_skip_panics_under_enforcement_and_marks_otherwise() {
    let r = std::panic::catch_unwind(|| gate::skipped_with("probe-target", "no mlir-opt", true));
    assert!(
        r.is_err(),
        "MIND_BENCH_REQUIRE=1 must forbid a capability probe skip"
    );
    // Not enforcing: returns normally and emits the `ran=0` marker the tier
    // gate's SKIP-MARKER CONSUMER treats as fatal-unless-accounted-for.
    //
    // Into a BUFFER, not this harness's stdout: the marker sink is the process
    // stdout handle (so a real skip survives libtest's capture), and
    // `probe-target` is a fixture name no gate carries — printing it for real
    // would hand the tier script a `ran=0` for a gate that does not exist.
    // Asserting the bytes is also strictly more than the old call proved.
    let mut sink: Vec<u8> = Vec::new();
    gate::skipped_because_to(
        &mut sink,
        "probe-target",
        "no mlir-opt",
        gate::Absent::Toolchain,
        false,
    );
    assert_eq!(
        String::from_utf8(sink).expect("marker is utf-8").trim_end(),
        concat!(
            "SDLC-GATE probe-target ran=0 fail=0 class=toolchain",
            "  (capability skip: no mlir-opt)"
        ),
        "a probe skip must report ran=0 and name its absence CLASS"
    );
}

// --- the ENV path lives next door ------------------------------------------
//
// Every test in THIS file supplies `enforce` explicitly, which is what makes
// the contract provable without touching process-global state — and is exactly
// why none of them can see the flag's READER. `gate::enforce_real_backend`,
// `gate::skipped`, `gate::skipped_optional` and `gate::compiled` are pinned in
// `tests/fail_closed_capability_skip_env.rs`, which varies `MIND_BENCH_REQUIRE`
// the only sound way: in a child process. Measured before it existed: mutating
// `enforce_real_backend` to `false` left this file at 22 passed; 0 failed.

// --- the cause-code anti-drift scan ----------------------------------------
//
// Lives in its own file: `tests/capability_refusal_cause_scan.rs`. THIS file
// owns the classifier's runtime contract (given a wire, what is the verdict);
// that one owns the compiler's SOURCE discipline (does every host-capability
// refusal in `src/` mint its cause). Different subject, different inputs, and
// keeping them together pushed this file past the size limit.

// --- end-to-end: the REAL compiler's refusals, not a fixture of them --------
//
// The fixtures above prove the classifier's contract; they cannot prove the
// compiler still emits that wire shape. These two drive the actual `mindc`
// binary and are feature-INDEPENDENT: whichever backend this build carries,
// a well-formed project must never grade as a compiler regression, and a
// program that does not compile must never grade as a skip.

/// Write a minimal single-source project into `dir`.
fn write_project(dir: &Path, name: &str, source: &str) {
    std::fs::create_dir_all(dir.join("src")).expect("mkdir src");
    std::fs::write(
        dir.join("Mind.toml"),
        format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n"),
    )
    .expect("write Mind.toml");
    std::fs::write(dir.join("src/main.mind"), source).expect("write entry");
}

fn run_mindc_build(dir: &Path) -> std::process::Output {
    Command::new(common::mindc_bin())
        .args(["build", "--backend", "mlir"])
        .current_dir(dir)
        .output()
        .expect("spawn mindc")
}

#[test]
fn a_well_formed_project_never_grades_as_a_compiler_regression() {
    // THE REGRESSION THIS GATE EXISTS FOR: on a binary without `mlir-build`,
    // `mindc build` refuses (the artifact would be a runtime-JIT launcher) —
    // a HOST capability gap. Grading it `Failed` panicked every converted call
    // site on every backend-less host.
    let tmp = tempfile::tempdir().expect("tempdir");
    write_project(tmp.path(), "cap_ok", "fn main() -> i64 { 42 }\n");
    let out = run_mindc_build(tmp.path());
    let stderr = String::from_utf8_lossy(&out.stderr);
    let verdict = gate::classify(out.status.success(), &stderr, false);
    assert!(
        !matches!(verdict, Outcome::Failed(_)),
        "a well-formed project graded as a compiler regression: {verdict:?}\nstderr:\n{stderr}"
    );
}

#[test]
fn a_program_that_does_not_compile_is_never_a_capability_skip() {
    // The negative twin: same refusal wording, different cause. If this ever
    // grades as a skip, the classifier has been widened back into a fail-open.
    let tmp = tempfile::tempdir().expect("tempdir");
    write_project(tmp.path(), "cap_broken", "fn broken( -> {\n");
    let out = run_mindc_build(tmp.path());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a broken program must not build clean"
    );
    match gate::classify(out.status.success(), &stderr, false) {
        Outcome::Failed(_) => {}
        other => panic!(
            "a program that does not compile must fail closed, got {other:?}\nstderr:\n{stderr}"
        ),
    }
}

#[test]
fn a_program_that_names_a_cause_code_is_never_a_capability_skip_end_to_end() {
    // The fixture above is only as good as the wire shape it copies. This
    // drives the REAL binary on a program whose source text spells `E5003`,
    // so the forgery is attempted against whatever the emitter actually
    // prints. Feature-INDEPENDENT: the program does not compile on any host.
    let tmp = tempfile::tempdir().expect("tempdir");
    write_project(
        tmp.path(),
        "cap_forge",
        "fn main() -> i64 {\n    let arr: [i64; 2] = [1, 2];\n    arr[E5003]\n}\n",
    );
    let out = run_mindc_build(tmp.path());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a program with an unresolved identifier must not build clean: {stderr}"
    );
    assert!(
        stderr.contains("E5003"),
        "the fixture stopped exercising the forgery: the compiler no longer \
         echoes the identifier. stderr:\n{stderr}"
    );
    match gate::classify(out.status.success(), &stderr, false) {
        Outcome::Failed(_) => {}
        other => panic!(
            "a program that names a cause code in its own source forged a \
             capability skip, got {other:?}\nstderr:\n{stderr}"
        ),
    }
}

#[test]
fn an_unavailable_target_backend_is_a_capability_gap_end_to_end() {
    // The fixtures above prove the classifier's contract; only the real binary
    // proves the compiler still stamps that code. Feature-INDEPENDENT: every
    // non-CPU target is refused in this crate regardless of the build's
    // features, so this asserts on every host.
    let tmp = tempfile::tempdir().expect("tempdir");
    let src = tmp.path().join("t.mind");
    std::fs::write(&src, "fn main() -> i64 { 0 }\n").expect("write source");
    let out = Command::new(common::mindc_bin())
        .arg(&src)
        .args(["--target", "gpu"])
        .output()
        .expect("spawn mindc");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a target with no backend must refuse: {stderr}"
    );
    match gate::classify(out.status.success(), &stderr, false) {
        Outcome::CapabilitySkip => {}
        other => panic!(
            "a host without the target's backend must grade as a capability gap, \
             got {other:?}\nstderr:\n{stderr}"
        ),
    }
}
