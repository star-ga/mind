// Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0.
//
//! `mindc build` / `mindc run` must consult the runnable-artifact ABI gate (#54).
//!
//! `products.runnable_blockers` — the constructs the shipped i64-scalar backend
//! would SILENTLY MISCOMPILE — was consulted ONLY on the single-file
//! `--emit-obj` / `--emit-shared` path (`src/bin/mindc.rs`). The `mindc build` /
//! `mindc run` compile path (`build::run_build` -> `project::compile_sources` ->
//! `compile_single_source`, and the cdylib `build_cdylib_from_entry`) NEVER
//! checked it — so a program that `--emit-shared` fail-loud REJECTS built GREEN
//! (rc=0) and ran WRONG via the primary commands. The fix consults
//! `runnable_blockers` in the build/run compile path and fails non-zero.
//!
//! This test uses the enum-handle-in-scalar-return blocker (`divide(_,0)` returns
//! `Res::Err(0)` where `-> i64` is declared — a leaked heap-record pointer),
//! confirms `--emit-shared` rejects it, then asserts `mindc build` AND `mindc run`
//! now exit NON-ZERO (they returned rc=0 before).
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test build_run_runnable_blocker_gate`

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

// Bare-scalar `-> i64` return that yields an enum constructor handle on one path
// — a `runnable_blocker` (`lower::enum_handle_in_scalar_return`).
const BLOCKER_MIND: &str = r#"enum Res { Ok(i64), Err(i64) }
fn divide(a: i64, b: i64) -> i64 {
    if b == 0 { return Res::Err(0); }
    return a / b;
}
fn main() -> i64 { return divide(4, 2); }
"#;

fn manifest(entry: &str) -> String {
    format!(
        "[package]\nname = \"blockerproj\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"{entry}\"\noutput = \"blockerproj\"\n"
    )
}

fn linked_manifest(name: &str) -> String {
    format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"src/main.mind\"\noutput = \"{name}\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/main.mind\", \"src/helper.mind\"]\n"
    )
}

fn write_linked_project(root: &Path, name: &str, main_source: &str, helper_source: &str) {
    let src_dir = root.join("src");
    fs::create_dir_all(&src_dir).expect("mkdir project src");
    fs::write(src_dir.join("main.mind"), main_source).expect("write project entry");
    fs::write(src_dir.join("helper.mind"), helper_source).expect("write project helper");
    fs::write(root.join("Mind.toml"), linked_manifest(name)).expect("write project manifest");
}

fn output_text(output: &Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_project_fallback_refusal(output: &Output, operation: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "{operation} must refuse a non-entry fallback: {}",
        output_text(output)
    );
    assert!(
        stderr.contains("[E5005]"),
        "{operation} must preserve the source-fallback cause code: {}",
        output_text(output)
    );
    assert!(
        stderr.contains("helper") && stderr.contains("not natively compiled"),
        "{operation} must identify the non-entry fallback module: {}",
        output_text(output)
    );
}

#[test]
fn build_and_run_reject_runnable_blocker() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "build_run_runnable_blocker_gate",
            "build-run-runnable-blocker-gate: mindc not found; skipping",
        );
        return;
    }
    let td = tempfile::tempdir().expect("tempdir");
    let src_dir = td.path().join("src");
    fs::create_dir_all(&src_dir).expect("mkdir src");
    fs::write(src_dir.join("main.mind"), BLOCKER_MIND).expect("write src");
    fs::write(td.path().join("Mind.toml"), manifest("src/main.mind")).expect("write manifest");

    // (0) Baseline: the single-file `--emit-shared` path already rejects it.
    let so = td.path().join("blocker.so");
    let es = Command::new(&mindc)
        .args([
            src_dir.join("main.mind").to_str().unwrap(),
            "--emit-shared",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc --emit-shared");
    let es_err = String::from_utf8_lossy(&es.stderr);
    if crate::common::gate::is_capability_gap(&es_err) {
        crate::common::gate::skipped(
            "build_run_runnable_blocker_gate",
            "build-run-runnable-blocker-gate: needs mlir-build; skipping",
        );
        return;
    }
    assert!(
        !es.status.success(),
        "--emit-shared must reject the runnable_blocker but exited 0"
    );
    assert!(
        es_err.contains("enum_handle_in_scalar_return"),
        "expected the enum-handle-scalar-return diagnostic; got:\n{es_err}"
    );

    // (1) `mindc build` must now ALSO reject it (was rc=0 before the fix).
    let build = Command::new(&mindc)
        .args(["build", "--backend", "mlir"])
        .current_dir(td.path())
        .output()
        .expect("run mindc build");
    assert!(
        !build.status.success(),
        "mindc build must fail non-zero on a runnable_blocker; stdout={} stderr={}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr),
    );
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("runnable artifact"),
        "mindc build error should name the runnable-artifact refusal; got:\n{}",
        String::from_utf8_lossy(&build.stderr),
    );

    // (2) `mindc run` must now ALSO reject it (was rc=0 before the fix).
    let run = Command::new(&mindc)
        .args(["run", "--backend", "mlir"])
        .current_dir(td.path())
        .output()
        .expect("run mindc run");
    assert!(
        !run.status.success(),
        "mindc run must fail non-zero on a runnable_blocker; stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
}

/// #244 project coverage: a warning-only, type-check-clean self-host intrinsic
/// in a LINKED non-entry source must not be turned into a green project build
/// or run. This exercises the source-fallback accounting in
/// `compile_sources`, rather than the entry-only single-file guard above.
#[test]
fn project_build_and_run_reject_non_entry_source_fallback() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "build_run_runnable_blocker_gate",
            "project-fallback-gate: mindc not found; skipping",
        );
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    write_linked_project(
        td.path(),
        "project_fallback",
        "import helper;\nfn main() -> i64 {\n    return degraded();\n}\n",
        "pub fn degraded() -> i64 {\n    let h: i64 = __mind_nerve_route(1);\n    return h;\n}\n",
    );

    // The helper is intentionally check-clean: E2024 is an advisory warning
    // for an unregistered __mind_* intrinsic. The project build is the seam
    // under test; a check failure here would exercise the wrong contract.
    let check = Command::new(&mindc)
        .args(["check", "src/helper.mind"])
        .current_dir(td.path())
        .output()
        .expect("run project check");
    assert!(
        check.status.success(),
        "fallback fixture must pass check with an advisory diagnostic: {}",
        output_text(&check)
    );
    let check_text = output_text(&check);
    assert!(
        check_text.contains("E2024") && check_text.contains("__mind_nerve_route"),
        "check must expose the advisory unsupported intrinsic: {check_text}"
    );

    // `--out` is outside the project target directory so this assertion binds
    // the user-visible artifact path, not an intermediate object/cache.
    let requested = td.path().join("requested-fallback");
    let build = Command::new(&mindc)
        .args(["build", "--backend", "mlir", "--no-cache", "--out"])
        .arg(&requested)
        .current_dir(td.path())
        .output()
        .expect("run project build");
    let build_text = output_text(&build);
    assert_project_fallback_refusal(&build, "project build");
    assert!(
        build_text.contains("E2024") && build_text.contains("__mind_nerve_route"),
        "build refusal must retain the helper's E2024 cause, rather than accepting an arbitrary resolver failure: {build_text}"
    );
    assert!(
        !requested.exists(),
        "a refused non-entry fallback build must leave no requested runnable artifact at {}",
        requested.display()
    );

    // Remove any intermediate target output before `run`, so a positive
    // absence assertion cannot be satisfied by stale state from the build
    // invocation above.
    if td.path().join("target").exists() {
        fs::remove_dir_all(td.path().join("target")).expect("remove failed-build target");
    }
    let run = Command::new(&mindc)
        .args(["run", "--backend", "mlir"])
        .current_dir(td.path())
        .output()
        .expect("run project run");
    let run_text = output_text(&run);
    assert_project_fallback_refusal(&run, "project run");
    assert!(
        run_text.contains("E2024") && run_text.contains("__mind_nerve_route"),
        "run refusal must retain the helper's E2024 cause, rather than accepting an arbitrary resolver failure: {run_text}"
    );
    assert!(
        !td.path().join("target/debug/project_fallback").exists(),
        "a refused non-entry fallback run must leave no fresh runnable artifact"
    );
}

/// A valid two-file project remains a real native build/run control for the
/// non-entry fallback gate. This prevents the refusal from becoming a blanket
/// rejection of linked dependencies.
#[test]
fn project_valid_linked_dependency_builds_and_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "build_run_runnable_blocker_gate",
            "project-link-control: mindc not found; skipping",
        );
        return;
    }

    let td = tempfile::tempdir().expect("tempdir");
    write_linked_project(
        td.path(),
        "project_link_control",
        "import helper;\nfn main() -> i64 { return answer(); }\n",
        "pub fn answer() -> i64 { return 42; }\n",
    );
    let requested = td.path().join("linked-project");
    let build = Command::new(&mindc)
        .args(["build", "--backend", "mlir", "--no-cache", "--out"])
        .arg(&requested)
        .current_dir(td.path())
        .output()
        .expect("run valid project build");
    assert!(
        build.status.success(),
        "valid linked project build failed: {}",
        output_text(&build)
    );
    assert!(requested.is_file(), "valid build produced no artifact");

    let run = Command::new(&mindc)
        .args(["run", "--backend", "mlir"])
        .current_dir(td.path())
        .output()
        .expect("run valid project run");
    assert_eq!(
        run.status.code(),
        Some(42),
        "valid linked project did not execute its dependency result: {}",
        output_text(&run)
    );
}

/// A native-only intrinsic is refused by name (E2041) before the MLIR backend
/// lowers or links, on every runnable MLIR path. It used to write a shared
/// object with an undefined symbol, or fail in the linker. `mindc check` and
/// `--emit-evidence` stay backend-neutral (the native backend lowers the row),
/// and `__mind_now_ns`, which no native row defines, keeps its E2024 refusal.
#[test]
fn a_native_only_intrinsic_is_refused_by_name_before_the_mlir_link() {
    let mindc = mindc_bin();
    let td = tempfile::tempdir().expect("tempdir");
    // Positive control first: a program with no native-only call links. On a
    // host without the MLIR toolchain the gate reports the skip (and fails
    // under MIND_BENCH_REQUIRE=1) instead of returning silently.
    let ok = td.path().join("ok.mind");
    fs::write(
        &ok,
        "fn main() -> i64 {\n    let p: i64 = __mind_alloc(64);\n    return 0;\n}\n",
    )
    .expect("write control");
    let so = td.path().join("ok.so");
    let built = Command::new(&mindc)
        .arg(&ok)
        .arg("--emit-shared")
        .arg(&so)
        .output()
        .expect("control");
    if crate::common::gate::is_capability_gap(&output_text(&built)) {
        crate::common::gate::skipped(
            "build_run_runnable_blocker_gate",
            "native-only intrinsic refusal: needs the MLIR toolchain",
        );
        return;
    }
    assert!(
        built.status.success() && so.exists(),
        "control: {}",
        output_text(&built)
    );
    let refused: [(&str, &str, &str); 4] = [
        ("ymm", "__mind_ymm_vmovdqu_ld(0, p, 0);", "E2041"),
        ("vp", "let q: i64 = __mind_vpaddd_i32x8(p, p);", "E2041"),
        ("relax", "let q: i64 = __mind_cpu_relax();", "E2041"),
        ("now", "let q: i64 = __mind_now_ns(0);", "E2024"),
    ];
    for (case, stmt, code) in refused {
        let name = stmt
            .split(['(', ' '])
            .find(|w| w.starts_with("__mind_"))
            .expect("intrinsic name");
        let src = td.path().join(format!("{case}.mind"));
        fs::write(
            &src,
            format!("fn main() -> i64 {{\n    let p: i64 = __mind_alloc(64);\n    {stmt}\n    return 0;\n}}\n"),
        )
        .expect("write source");
        for flag in ["--emit-shared", "--emit-obj"] {
            let out = td.path().join(format!("{case}{flag}.bin"));
            let built = Command::new(&mindc)
                .arg(&src)
                .arg(flag)
                .arg(&out)
                .arg("--allow-nondeterministic")
                .output()
                .expect("run mindc");
            let text = output_text(&built);
            assert_eq!(built.status.code(), Some(1), "{case} {flag}: {text}");
            assert!(
                text.contains(code) && text.contains(name),
                "{case} {flag}: {text}"
            );
            assert!(
                !out.exists(),
                "{case} {flag}: a refused build wrote {out:?}"
            );
        }
        if code == "E2041" {
            let check = Command::new(&mindc)
                .arg("check")
                .arg(&src)
                .output()
                .expect("check");
            assert!(
                check.status.success(),
                "{case}: check is backend-neutral: {}",
                output_text(&check)
            );
            let evidence = td.path().join(format!("{case}.mic3"));
            let emitted = Command::new(&mindc)
                .arg(&src)
                .args(["--emit-evidence"])
                .arg(&evidence)
                .arg("--allow-nondeterministic")
                .output()
                .expect("emit evidence");
            assert!(
                emitted.status.success(),
                "{case}: {}",
                output_text(&emitted)
            );
        }
    }
}

/// A native-only call in a project SIBLING module is refused (E2041) on the
/// shared-library routes, `mindc build --emit=cdylib` and the flat
/// `--emit-shared`. The sibling object compile never read the runnable
/// blockers, so both wrote a `.so` with `__mind_vpaddd_i32x8` undefined.
#[test]
fn a_native_only_call_in_a_sibling_module_is_refused_on_the_cdylib_routes() {
    let mindc = mindc_bin();
    let td = tempfile::tempdir().expect("tempdir");
    write_linked_project(
        td.path(),
        "siblingvp",
        "import helper;\nfn main() -> i64 {\n    return degraded();\n}\n",
        "pub fn degraded() -> i64 {\n    let p: i64 = __mind_alloc(64);\n    let q: i64 = __mind_vpaddd_i32x8(p, p);\n    return q;\n}\n",
    );
    let cdylib = td.path().join("libsiblingvp.so");
    let flat = td.path().join("flat.so");
    let routes: [(&str, Vec<String>, &Path); 2] = [
        (
            "build --emit=cdylib",
            vec![
                "build".into(),
                "--backend".into(),
                "mlir".into(),
                "--emit=cdylib".into(),
                format!("--out={}", cdylib.display()),
            ],
            &cdylib,
        ),
        (
            "--emit-shared",
            vec![
                "src/main.mind".into(),
                "--emit-shared".into(),
                flat.display().to_string(),
            ],
            &flat,
        ),
    ];
    for (route, args, artifact) in routes {
        let out = Command::new(&mindc)
            .args(&args)
            .current_dir(td.path())
            .output()
            .expect("run mindc");
        let text = output_text(&out);
        if crate::common::gate::is_capability_gap(&text) {
            crate::common::gate::skipped(
                "build_run_runnable_blocker_gate",
                "sibling native-only refusal: needs the MLIR toolchain",
            );
            return;
        }
        assert!(!out.status.success(), "{route}: must refuse: {text}");
        assert!(
            text.contains("E2041") && text.contains("__mind_vpaddd_i32x8"),
            "{route}: the refusal must name E2041 and the intrinsic: {text}"
        );
        assert!(
            !artifact.exists(),
            "{route}: a refused build wrote {artifact:?}"
        );
    }
}

/// Build `source` as a one-file project with `mindc build`; the output and
/// the artifact path.
fn build_project(source: &str, name: &str) -> (Output, tempfile::TempDir, std::path::PathBuf) {
    let td = tempfile::tempdir().expect("tempdir");
    let src_dir = td.path().join("src");
    fs::create_dir_all(&src_dir).expect("mkdir src");
    fs::write(src_dir.join("main.mind"), source).expect("write src");
    fs::write(td.path().join("Mind.toml"), manifest("src/main.mind")).expect("write manifest");
    let artifact = td.path().join(name);
    let build = Command::new(mindc_bin())
        .args(["build", "--backend", "mlir", "--no-cache", "--out"])
        .arg(&artifact)
        .current_dir(td.path())
        .output()
        .expect("run mindc build");
    (build, td, artifact)
}

/// `print` of a `String` value writes its bytes, never its record address.
/// Before the fix every non-literal string operand went through `printI64`, so
/// the program below wrote one decimal handle per string.
const PRINT_STRINGS_MIND: &str = r#"import std.string;

fn long_text(n: i64) -> String {
    let mut s: String = string_new();
    let mut i: i64 = 0;
    while i < n {
        s = string_push_byte(s, 97 + i % 26);
        i = i + 1;
    }
    return s;
}

fn label() -> String {
    return "fn-result";
}

fn show(s: &string) -> i64 {
    print("<", s, ">");
    return 0;
}

fn main() -> i64 {
    let empty: String = string_new();
    let ascii: String = "hello";
    let utf8: String = "héllo €";
    let built: String = string_push_str(string_new(), "built");
    let lower: string = "lower";
    let long: String = long_text(5000);
    print("[", empty, "]");
    print("[", ascii, "]");
    print("[", utf8, "]");
    print("[", built, "]");
    print("[", lower, "]");
    print("[", label(), "]");
    print("[", if string_len(ascii) > 0 { ascii } else { lower }, "]");
    let shown: i64 = show(&ascii);
    print("[", long, "]");
    print("|", 42 + shown);
    return 0;
}
"#;

#[test]
fn print_of_string_values_writes_their_bytes() {
    let (build, _td, artifact) = build_project(PRINT_STRINGS_MIND, "print_strings");
    if !crate::common::gate::compiled("build_run_runnable_blocker_gate", &build) {
        return;
    }
    let run = Command::new(&artifact)
        .output()
        .expect("run the built program");
    assert_eq!(run.status.code(), Some(0), "{}", output_text(&run));
    let long: String = (0..5000usize)
        .map(|i| char::from(b'a' + (i % 26) as u8))
        .collect();
    let want = format!("[][hello][héllo €][built][lower][fn-result][hello]<hello>[{long}]|42\n");
    assert_eq!(
        run.stdout,
        want.as_bytes(),
        "stdout bytes differ (a handle printed as an integer is the old defect)"
    );
}

/// Each `print` operand a runnable artifact cannot write correctly is refused
/// by name at build time, while `mindc check` (shared with the interpreter,
/// which prints every value) still accepts the source.
#[test]
fn print_of_aggregate_or_untyped_value_is_refused_by_name() {
    // A `&[T]` slice operand never reaches this gate: type-check refuses it (E2033).
    let cases: [(&str, &str, &str); 7] = [
        (
            "struct P { x: i64, y: i64 }\nfn main() -> i64 {\n    let p: P = P { x: 1, y: 2 };\n    print(p);\n    return 0;\n}\n",
            "struct",
            "a struct value",
        ),
        (
            "fn main() -> i64 {\n    let a: [i64; 3] = [1, 2, 3];\n    print(a);\n    return 0;\n}\n",
            "array",
            "an array, slice, set or tensor",
        ),
        (
            "fn show(t: tensor<f32[4]>) -> i64 {\n    print(t);\n    return 0;\n}\nfn main() -> i64 { return 0; }\n",
            "tensor",
            "an array, slice, set or tensor",
        ),
        (
            "fn show(m: map<i64, i64>) -> i64 {\n    print(m);\n    return 0;\n}\nfn main() -> i64 { return 0; }\n",
            "map",
            "a map",
        ),
        (
            "fn show(t: (i64, i64)) -> i64 {\n    print(t);\n    return 0;\n}\nfn main() -> i64 { return 0; }\n",
            "tuple",
            "a tuple",
        ),
        (
            "fn show(o: Option<i64>) -> i64 {\n    print(o);\n    return 0;\n}\nfn main() -> i64 { return 0; }\n",
            "option",
            "an `Option` or `Result` value",
        ),
        (
            "fn show<T>(x: T) -> i64 {\n    print(x);\n    return 0;\n}\nfn main() -> i64 { return 0; }\n",
            "untyped",
            "whose type is not known",
        ),
    ];
    for (source, kind, phrase) in cases {
        let (build, td, artifact) = build_project(source, kind);
        let text = output_text(&build);
        if crate::common::gate::is_capability_gap(&text) {
            crate::common::gate::skipped(
                "build_run_runnable_blocker_gate",
                "print-operand-refusal: needs mlir-build; skipping",
            );
            return;
        }
        assert!(!build.status.success(), "{kind}: build must refuse: {text}");
        assert!(
            text.contains("lower.print_operand") && text.contains(phrase),
            "{kind}: refusal must name the key and `{phrase}`: {text}"
        );
        assert!(
            !artifact.exists(),
            "{kind}: a refused build wrote {artifact:?}"
        );
        let check = Command::new(mindc_bin())
            .args(["check", "src/main.mind"])
            .current_dir(td.path())
            .output()
            .expect("run mindc check");
        assert!(
            !output_text(&check).contains("lower.print_operand"),
            "{kind}: check must not refuse the print: {}",
            output_text(&check)
        );
    }
}
