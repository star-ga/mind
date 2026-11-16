// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Owner-aware bare-name resolution for manifest-captured project imports.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libloading::{Library, Symbol};

fn manifest() -> &'static str {
    "[package]\nname = \"owner_resolution\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\noutput = \"owner_resolution\"\n"
}

fn check(mindc: &Path, project: &Path) -> Output {
    Command::new(mindc)
        .current_dir(project)
        .args(["check", "src/consumer.mind"])
        .output()
        .expect("run owner-resolution check")
}

fn project_with_modules(temp: &Path) -> PathBuf {
    let project = temp.join("project");
    fs::create_dir_all(project.join("src")).expect("create project src");
    fs::write(project.join("Mind.toml"), manifest()).expect("write manifest");
    fs::write(
        project.join("main.mind"),
        "fn main() -> i64 {\n    return 0;\n}\n",
    )
    .expect("write entry");
    fs::write(
        project.join("src/a.mind"),
        "fn f(x: i64) -> i64 {\n    return x + 1;\n}\nfn g(x: i64) -> i64 {\n    return x + 3;\n}\nstruct Widget {\n    a: i64,\n}\n",
    )
    .expect("write module a");
    fs::write(
        project.join("src/b.mind"),
        "fn f(x: f64) -> f64 {\n    return x + 2.0;\n}\nstruct Widget {\n    b: i64,\n}\n",
    )
    .expect("write module b");
    project
}

#[test]
fn captured_import_owner_controls_check_and_emit() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("owner-resolution scratch");
    let project = project_with_modules(temp.path());

    // The imported owner is authoritative: the unimported a.f signature must
    // not reinterpret the f64 call as an i64 call.
    fs::write(
        project.join("src/consumer.mind"),
        "import b;\nfn run(x: f64) -> f64 {\n    return f(x);\n}\n",
    )
    .expect("write only-b consumer");
    let out = check(&mindc, &project);
    assert!(
        out.status.success(),
        "only-b owner check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let artifact = project.join("only-b.so");
    let emit = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "--emit-shared",
            artifact.to_str().unwrap(),
            "src/consumer.mind",
        ])
        .output()
        .expect("run only-b emit");
    assert!(
        emit.status.success() && artifact.is_file(),
        "only-b emit failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr)
    );

    // Both imported owners export f. A bare call has no unique ABI owner and
    // must be refused by check, so emit cannot reach an MLIR/link failure.
    fs::write(
        project.join("src/consumer.mind"),
        "import a;\nimport b;\nfn run(x: i64) -> i64 {\n    return f(x);\n}\n",
    )
    .expect("write both-function consumer");
    let out = check(&mindc, &project);
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "ambiguous function was accepted: {rendered}"
    );
    assert!(rendered.contains("E2003"), "missing E2003: {rendered}");
    let ambiguous_artifact = project.join("ambiguous.so");
    let emit = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "--emit-shared",
            ambiguous_artifact.to_str().unwrap(),
            "src/consumer.mind",
        ])
        .output()
        .expect("run ambiguous emit");
    assert!(!emit.status.success(), "ambiguous emit succeeded");
    assert!(
        !ambiguous_artifact.exists(),
        "ambiguous emit published an artifact"
    );
    assert!(
        String::from_utf8_lossy(&emit.stderr).contains("E2003"),
        "ambiguous emit lost E2003: {}",
        String::from_utf8_lossy(&emit.stderr)
    );

    // A captured import that does not export the requested name must not
    // reopen the whole project and borrow an unimported sibling's ABI.
    fs::write(
        project.join("src/consumer.mind"),
        "import b;\nfn run(x: i64) -> i64 {\n    return g(x);\n}\n",
    )
    .expect("write missing-import consumer");
    let out = check(&mindc, &project);
    let rendered = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "unimported sibling was accepted: {rendered}"
    );
    assert!(rendered.contains("E2003"), "missing E2003: {rendered}");
    let missing_artifact = project.join("missing-owner.so");
    let emit = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "--emit-shared",
            missing_artifact.to_str().unwrap(),
            "src/consumer.mind",
        ])
        .output()
        .expect("run missing-import emit");
    assert!(!emit.status.success(), "unimported sibling emit succeeded");
    assert!(
        !missing_artifact.exists(),
        "unimported sibling emit published an artifact"
    );
    assert!(
        String::from_utf8_lossy(&emit.stderr).contains("E2003"),
        "missing-import emit lost E2003: {}",
        String::from_utf8_lossy(&emit.stderr)
    );

    // The same owner rule applies to bare types. Two imported Widget owners
    // refuse. Local declarations retain precedence during check; the native
    // artifact gate then refuses the unsupported linker collision before clang.
    fs::write(
        project.join("src/consumer.mind"),
        "import a;\nimport b;\nfn run(w: Widget) -> i64 {\n    return w.a;\n}\n",
    )
    .expect("write both-type consumer");
    let out = check(&mindc, &project);
    let rendered = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "ambiguous type was accepted: {rendered}"
    );
    assert!(rendered.contains("E2002"), "missing E2002: {rendered}");
    assert!(
        !rendered.contains("panicked at"),
        "ambiguous type panicked: {rendered}"
    );

    fs::write(
        project.join("src/consumer.mind"),
        "import a;\nfn f(x: i64) -> i64 {\n    return x + 7;\n}\nfn run(x: i64) -> i64 {\n    return f(x);\n}\n",
    )
    .expect("write local-shadow consumer");
    let out = check(&mindc, &project);
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "local declaration did not take precedence: {rendered}"
    );
    let local_shadow_artifact = project.join("local-shadow.so");
    let emit = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "--emit-shared",
            local_shadow_artifact.to_str().unwrap(),
            "src/consumer.mind",
        ])
        .output()
        .expect("run local-shadow emit");
    assert!(!emit.status.success(), "local shadow emit succeeded");
    assert!(
        !local_shadow_artifact.exists(),
        "local shadow emit published an artifact"
    );
    assert!(
        String::from_utf8_lossy(&emit.stderr).contains("lower::imported_local_fn_collision"),
        "local shadow emit lost structured lowering refusal: {}",
        String::from_utf8_lossy(&emit.stderr)
    );
}

#[test]
fn captured_single_owner_same_abi_executes_selected_body() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("same-ABI owner scratch");
    let project = project_with_modules(temp.path());

    // Keep the two declarations ABI-identical so a successful result proves
    // owner selection rather than a type mismatch. Their bodies differ, which
    // makes selecting the unimported sibling observable at runtime.
    fs::write(
        project.join("src/a.mind"),
        "fn f(x: i64) -> i64 { return x + 100; }\n",
    )
    .expect("write same-ABI module a");
    fs::write(
        project.join("src/b.mind"),
        "fn f(x: i64) -> i64 { return x + 200; }\n",
    )
    .expect("write same-ABI module b");

    for (owner, expected) in [("a", 105_i64), ("b", 205_i64)] {
        fs::write(
            project.join("src/consumer.mind"),
            format!("import {owner};\nfn run(x: i64) -> i64 {{ return f(x); }}\n"),
        )
        .expect("write single-owner consumer");
        let artifact = project.join(format!("single-{owner}.so"));
        let emit = Command::new(&mindc)
            .current_dir(&project)
            .args([
                "--emit-shared",
                artifact.to_str().unwrap(),
                "src/consumer.mind",
            ])
            .output()
            .expect("run same-ABI owner emit");
        assert!(
            emit.status.success() && artifact.is_file(),
            "single-{owner} emit failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&emit.stdout),
            String::from_utf8_lossy(&emit.stderr)
        );
        unsafe {
            let lib = Library::new(&artifact).expect("load same-ABI owner artifact");
            let run: Symbol<unsafe extern "C" fn(i64) -> i64> =
                lib.get(b"run\0").expect("load run");
            assert_eq!(
                run(5),
                expected,
                "imported owner {owner} body was not selected"
            );
        }
    }
}

fn collide_manifest() -> &'static str {
    "[package]\nname = \"collide_probe\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\noutput = \"collide_probe\"\n"
}

fn write_collide_project(root: &Path, main_src: &str, encoder_src: &str) -> PathBuf {
    let project = root.join("project");
    fs::create_dir_all(project.join("src")).expect("create collide src");
    fs::write(project.join("Mind.toml"), collide_manifest()).expect("write collide manifest");
    fs::write(project.join("src/main.mind"), main_src).expect("write collide main");
    fs::write(project.join("src/encoder.mind"), encoder_src).expect("write encoder");
    project
}

#[test]
fn executable_route_refuses_local_name_that_collides_with_import() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("collision scratch");
    let local_v: i64 = 1;
    let imported_v: i64 = 2;
    let encoder = format!("pub fn encode(x: i64) -> i64 {{\n    return {imported_v};\n}}\n");
    let main_src = format!(
        "import encoder;\n\nfn encode(x: i64) -> i64 {{\n    return {local_v};\n}}\n\npub fn main() -> i64 {{\n    return encoder.encode(0) * 10 + encode(0);\n}}\n"
    );
    let project = write_collide_project(temp.path(), &main_src, &encoder);
    let build = common::run_build_captured(&mindc, &project, &["--no-cache"]);
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    if build.status.success() {
        let artifact = common::reported_artifact(&build);
        let run = Command::new(&artifact)
            .output()
            .expect("run colliding binary");
        panic!(
            "executable route built a colliding program and returned {:?}; the wrong body is 11\n{rendered}",
            run.status.code()
        );
    }
    assert!(
        rendered.contains("lower::imported_local_fn_collision"),
        "executable route lost the collision refusal:\n{rendered}"
    );

    let cdylib = project.join("collide.so");
    let emit = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "build",
            "--backend",
            "mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            cdylib.to_str().unwrap(),
        ])
        .output()
        .expect("run cdylib build");
    let cdylib_text = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(
        !emit.status.success(),
        "cdylib route accepted the collision:\n{cdylib_text}"
    );
    assert!(
        cdylib_text.contains("lower::imported_local_fn_collision"),
        "cdylib route changed its refusal:\n{cdylib_text}"
    );
    assert!(!cdylib.exists(), "cdylib route published an artifact");

    let native_out = project.join("native-bin");
    let native = Command::new(&mindc)
        .current_dir(&project)
        .args([
            "build",
            "--backend",
            "native",
            "--no-cache",
            "--out",
            native_out.to_str().unwrap(),
        ])
        .output()
        .expect("run native build");
    let native_text = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&native.stdout),
        String::from_utf8_lossy(&native.stderr)
    );
    assert!(
        !native.status.success(),
        "native route accepted the collision:\n{native_text}"
    );
    assert!(
        native_text.contains("closure.duplicate_definition"),
        "native route changed its refusal:\n{native_text}"
    );
    assert!(
        !native_text.contains("lower::imported_local_fn_collision"),
        "native route took the executable preflight:\n{native_text}"
    );
}

#[test]
fn distinct_local_name_still_runs_imported_body() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("distinct-name scratch");
    let local_v: i64 = 1;
    let imported_v: i64 = 2;
    let expect = imported_v * 10 + local_v;
    let encoder = format!("pub fn encode(x: i64) -> i64 {{\n    return {imported_v};\n}}\n");
    let main_src = format!(
        "import encoder;\n\nfn tag(x: i64) -> i64 {{\n    return {local_v};\n}}\n\npub fn main() -> i64 {{\n    return encoder.encode(0) * 10 + tag(0);\n}}\n"
    );
    let project = write_collide_project(temp.path(), &main_src, &encoder);
    let build = common::run_build_captured(&mindc, &project, &["--no-cache"]);
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        build.status.success(),
        "distinct names failed to build:\n{rendered}"
    );
    let artifact = common::reported_artifact(&build);
    let run = Command::new(&artifact)
        .output()
        .expect("run distinct binary");
    assert_eq!(
        run.status.code(),
        Some(i32::try_from(expect).expect("expect fits the process status")),
        "distinct names returned the wrong body\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

const LOCAL_ENCODE: &str = "fn encode(x: i64) -> i64 {\n    return 1;\n}\n";
const IMPORTED_ENCODE: &str = "pub fn encode(x: i64) -> i64 {\n    return 2;\n}\n";

fn main_calling(import_line: &str, expr: &str) -> String {
    format!("{import_line}\n\n{LOCAL_ENCODE}\npub fn main() -> i64 {{\n    return {expr};\n}}\n")
}

fn write_project_files(root: &Path, files: &[(&str, &str)]) -> PathBuf {
    let project = root.join("project");
    for (rel, body) in files {
        let path = project.join(rel);
        fs::create_dir_all(path.parent().expect("project file has a parent"))
            .expect("create parent");
        fs::write(&path, body).expect("write project file");
    }
    fs::write(project.join("Mind.toml"), collide_manifest()).expect("write collide manifest");
    project
}

fn assert_three_routes_refuse_collision(mindc: &Path, project: &Path) {
    let build = common::run_build_captured(mindc, project, &["--no-cache"]);
    let rendered = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    if build.status.success() {
        let artifact = common::reported_artifact(&build);
        let run = Command::new(&artifact)
            .output()
            .expect("run colliding binary");
        panic!(
            "executable route built a colliding program and returned {:?}; the wrong body is 11\n{rendered}",
            run.status.code()
        );
    }
    assert!(
        rendered.contains("lower::imported_local_fn_collision"),
        "executable route lost the collision refusal:\n{rendered}"
    );

    let cdylib = project.join("collide.so");
    let emit = Command::new(mindc)
        .current_dir(project)
        .args([
            "build",
            "--backend",
            "mlir",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            cdylib.to_str().unwrap(),
        ])
        .output()
        .expect("run cdylib build");
    let cdylib_text = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&emit.stdout),
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(
        !emit.status.success(),
        "cdylib route accepted the collision:\n{cdylib_text}"
    );
    assert!(
        cdylib_text.contains("lower::imported_local_fn_collision"),
        "cdylib route changed its refusal:\n{cdylib_text}"
    );
    assert!(!cdylib.exists(), "cdylib route published an artifact");

    let native_out = project.join("native-bin");
    let native = Command::new(mindc)
        .current_dir(project)
        .args([
            "build",
            "--backend",
            "native",
            "--no-cache",
            "--out",
            native_out.to_str().unwrap(),
        ])
        .output()
        .expect("run native build");
    let native_text = format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&native.stdout),
        String::from_utf8_lossy(&native.stderr)
    );
    assert!(
        !native.status.success(),
        "native route accepted the collision:\n{native_text}"
    );
    assert!(
        native_text.contains("closure.duplicate_definition"),
        "native route changed its refusal:\n{native_text}"
    );
    assert!(
        !native_text.contains("lower::imported_local_fn_collision"),
        "native route took the executable preflight:\n{native_text}"
    );
}

fn refuse_item_project(root: &Path, import_line: &str, expr: &str, encoder_rel: &str) -> PathBuf {
    let main_src = main_calling(import_line, expr);
    write_project_files(
        root,
        &[
            ("src/main.mind", main_src.as_str()),
            (encoder_rel, IMPORTED_ENCODE),
        ],
    )
}

#[test]
fn item_import_longer_than_one_segment_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("item path scratch");
    let project = refuse_item_project(
        temp.path(),
        "import encoder.encode;",
        "encode.encode(0) * 10 + encode(0)",
        "src/encoder.mind",
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}

#[test]
fn item_import_alias_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("alias scratch");
    let project = refuse_item_project(
        temp.path(),
        "import encoder.encode as enc;",
        "enc.encode(0) * 10 + encode(0)",
        "src/encoder.mind",
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}

#[test]
fn nested_item_import_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("nested item scratch");
    let project = refuse_item_project(
        temp.path(),
        "import encoder.codec.encode;",
        "encode.encode(0) * 10 + encode(0)",
        "src/encoder/codec.mind",
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}

#[test]
fn use_colon_item_import_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("use colon scratch");
    let project = refuse_item_project(
        temp.path(),
        "use encoder::encode;",
        "encode.encode(0) * 10 + encode(0)",
        "src/encoder.mind",
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}

#[test]
fn reexport_item_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("reexport item scratch");
    let main_src = main_calling("import middle.encode;", "encode.encode(0) * 10 + encode(0)");
    let project = write_project_files(
        temp.path(),
        &[
            ("src/main.mind", main_src.as_str()),
            ("src/encoder.mind", IMPORTED_ENCODE),
            ("src/middle.mind", "pub import encoder.encode;\n"),
        ],
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}

#[test]
fn reexport_middle_refuses_local_encode() {
    let mindc = common::require_mindc();
    let temp = tempfile::tempdir().expect("reexport middle scratch");
    let main_src = main_calling("import middle;", "middle.encode(0) * 10 + encode(0)");
    let project = write_project_files(
        temp.path(),
        &[
            ("src/main.mind", main_src.as_str()),
            ("src/encoder.mind", IMPORTED_ENCODE),
            ("src/middle.mind", "pub import encoder;\n"),
        ],
    );
    assert_three_routes_refuse_collision(&mindc, &project);
}
