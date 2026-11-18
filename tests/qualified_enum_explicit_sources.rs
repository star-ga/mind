// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! A qualified enum variant resolves with and without a Mind.toml sources list.
//!
//! `src/config.mind` under an explicit sources list is module `crate.src.config`.
//! `config.Boundary.Reflective` must still resolve. The same program with no
//! sources list already resolved. Both checks must pass.
//!
//! A bare enum head that two imports export is ambiguous however each import
//! reaches its owner: directly (`crate.a.T`) or through the sources-list
//! fallback (`crate.src.b.T`).

#![cfg(all(unix, feature = "std-surface", feature = "cross-module-imports"))]

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

fn mindc() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_mindc"))
}

fn write_sources(root: &Path) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/config.mind"),
        "module config {\n    enum Boundary {\n        Reflective = 0,\n        Zero = 1,\n    }\n    export enum Boundary\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/main.mind"),
        "import config;\nfn tag(bc: config.Boundary) -> i64 {\n    match bc {\n        config.Boundary.Reflective => { return 1 },\n        config.Boundary.Zero => { return 2 },\n    }\n}\nfn main() -> i64 {\n    return tag(config.Boundary.Reflective)\n}\n",
    )
    .unwrap();
}

fn check(root: &Path) -> (bool, String) {
    let out = Command::new(mindc())
        .current_dir(root)
        .args(["check", "--no-fmt", "--no-lint", "src/main.mind"])
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn qualified_variant_resolves_without_a_sources_list() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    write_sources(root);
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-src\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n",
    )
    .unwrap();
    let (ok, text) = check(root);
    assert!(ok, "no sources list:\n{text}");
}

#[test]
fn qualified_variant_resolves_with_a_sources_list() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    write_sources(root);
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-src\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/config.mind\", \"src/main.mind\"]\n",
    )
    .unwrap();
    let (ok, text) = check(root);
    assert!(
        ok,
        "explicit sources list must resolve config.Boundary.Reflective:\n{text}"
    );
    assert!(
        !text.contains("unknown module-qualified enum variant"),
        "{text}"
    );
}

fn enum_t(module: &str) -> String {
    format!(
        "module {module} {{\n    enum T {{\n        X = 0,\n        Y = 1,\n    }}\n    export enum T\n}}\n"
    )
}

fn two_owner_project(root: &Path, main: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("a.mind"), enum_t("a")).unwrap();
    fs::write(root.join("src/b.mind"), enum_t("b")).unwrap();
    fs::write(root.join("src/main.mind"), main).unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"enum-two\"\nversion = \"0.0.1\"\n[build]\nentry = \"src/main.mind\"\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"a.mind\", \"src/b.mind\", \"src/main.mind\"]\n",
    )
    .unwrap();
}

#[test]
fn a_bare_head_exported_by_a_direct_and_a_fallback_import_is_ambiguous() {
    // `a.mind` is `crate.a`, reached by `import a` directly; `src/b.mind` is
    // `crate.src.b`, reached by `import b` only through the fallback.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    two_owner_project(
        root,
        "import a;\nimport b;\nfn main() -> i64 {\n    let v = T::Y;\n    match v {\n        T::X => { return 1 },\n        T::Y => { return 2 },\n    }\n}\n",
    );
    let (ok, text) = check(root);
    assert!(
        !ok,
        "a bare T named by two imports must not resolve to one:\n{text}"
    );
    assert!(text.contains("ambiguous enum `T`"), "{text}");
}

#[test]
fn the_qualified_head_resolves_in_the_same_layout() {
    // Positive control for the refusal above: the same two owners, each named
    // through its module, check clean.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    two_owner_project(
        root,
        "import a;\nimport b;\nfn main() -> i64 {\n    let v = a.T.Y;\n    let w = b.T.X;\n    match v {\n        a.T.X => { return 1 },\n        a.T.Y => { return 2 },\n    }\n}\n",
    );
    let (ok, text) = check(root);
    assert!(ok, "qualified a.T / b.T must resolve:\n{text}");
}

/// `mindc check --no-fmt <files>` from `root`, as a user checks sibling files.
fn check_files(root: &Path, files: &[(&str, &str)]) -> (bool, String) {
    for (name, src) in files {
        fs::write(root.join(name), src).unwrap();
    }
    let mut args = vec!["check", "--no-fmt"];
    args.extend(files.iter().map(|(name, _)| *name));
    let out = Command::new(mindc())
        .current_dir(root)
        .args(&args)
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

const FIXED_POINT: &str = "module fixed_point {\n    type Q16_16 = i32\n    type Hidden = i32\n    fn q16_add(a: Q16_16, b: Q16_16) -> Q16_16 {\n        a + b\n    }\n    export type Q16_16\n    export fn q16_add\n}\n";

const CONFIG: &str = "module config {\n    enum NormMode {\n        GroupNorm = 0,\n        Clamp = 1,\n    }\n    enum Hidden {\n        A = 0,\n    }\n    export enum NormMode\n}\n";

fn alias_user(ty: &str) -> String {
    format!(
        "module user {{\n    use fixed_point\n    fn twice(x: fixed_point.{ty}) -> fixed_point.{ty} {{\n        fixed_point.q16_add(x, x)\n    }}\n}}\n"
    )
}

#[test]
fn an_exported_alias_of_a_module_block_resolves_as_a_qualified_type() {
    // The `use` sits inside `module user { }`; its import must count.
    let temp = tempfile::tempdir().unwrap();
    let user = alias_user("Q16_16");
    let files = [("fixed_point.mind", FIXED_POINT), ("user.mind", &user)];
    let (ok, text) = check_files(temp.path(), &files);
    assert!(ok, "fixed_point.Q16_16 must resolve:\n{text}");
}

#[test]
fn an_undeclared_or_unexported_qualified_type_stays_refused() {
    for ty in ["NOPE", "Hidden"] {
        let temp = tempfile::tempdir().unwrap();
        let user = alias_user(ty);
        let files = [("fixed_point.mind", FIXED_POINT), ("user.mind", &user)];
        let (ok, text) = check_files(temp.path(), &files);
        assert!(!ok, "fixed_point.{ty} must stay refused:\n{text}");
        let expected = format!("unknown module-qualified type `fixed_point.{ty}`");
        assert!(text.contains(&expected), "{text}");
    }
}

#[test]
fn a_qualified_variant_of_a_module_block_resolves_in_a_match_pattern() {
    let temp = tempfile::tempdir().unwrap();
    let user = "module user {\n    use config\n    fn pick(m: config.NormMode) -> i32 {\n        match m {\n            config.NormMode.GroupNorm => {\n                1\n            },\n            config.NormMode.Clamp => {\n                2\n            },\n        }\n    }\n}\n";
    let (ok, text) = check_files(temp.path(), &[("config.mind", CONFIG), ("user.mind", user)]);
    assert!(ok, "config.NormMode.GroupNorm must resolve:\n{text}");
}

#[test]
fn a_variant_of_an_unexported_enum_stays_refused() {
    let temp = tempfile::tempdir().unwrap();
    let user = "module user {\n    use config\n    fn pick(m: i32) -> i32 {\n        match m {\n            config.Hidden.A => {\n                1\n            },\n            _ => {\n                0\n            },\n        }\n    }\n}\n";
    let (ok, text) = check_files(temp.path(), &[("config.mind", CONFIG), ("user.mind", user)]);
    assert!(!ok, "config.Hidden is not exported:\n{text}");
    assert!(text.contains("config.Hidden.A"), "{text}");
}

#[test]
fn a_flat_alias_of_a_src_module_resolves_in_a_project_check() {
    // Root `main.mind` makes the project root the repo, so `src/fixed_point.mind`
    // is `crate.src.fixed_point` while `import fixed_point` reads `crate.fixed_point`.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/fixed_point.mind"), "type Q16_16 = i32\n").unwrap();
    fs::write(
        root.join("src/kernel.mind"),
        "import fixed_point;\nfn clamp_q16(v: fixed_point.Q16_16) -> fixed_point.Q16_16 {\n    return v;\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("main.mind"),
        "import kernel;\nimport fixed_point;\nfn main() -> i64 {\n    return 0;\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"alias-src\"\nversion = \"0.0.1\"\n[build]\nentry = \"main.mind\"\n",
    )
    .unwrap();
    let out = Command::new(mindc())
        .current_dir(root)
        .args(["check", "--no-fmt"])
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.status.success(),
        "fixed_point.Q16_16 from src/kernel.mind:\n{text}"
    );
}

/// `other` re-exports config's enum under the same name; `strange` exports a
/// different type of that name.
const OTHER_ALIAS: &str = "module other {\n    use config\n    type NormMode = config.NormMode\n    export type NormMode\n}\n";
const OTHER_STRUCT: &str = "module other {\n    struct NormMode {\n        v: i64,\n    }\n    export struct NormMode\n}\n";

fn user_of(ty: &str, pattern: &str) -> String {
    format!(
        "module user {{\n    use config\n    use other\n    fn pick(m: {ty}) -> i32 {{\n        match m {{\n            {pattern}GroupNorm => {{\n                1\n            }},\n            {pattern}Clamp => {{\n                2\n            }},\n        }}\n    }}\n}}\n"
    )
}

#[test]
fn a_qualified_type_and_variant_resolve_beside_a_same_named_import() {
    // The qualified spelling names its module; another import's same-named
    // type (a re-export or a different type) cannot make it ambiguous.
    for (tag, other) in [("alias", OTHER_ALIAS), ("struct", OTHER_STRUCT)] {
        for pattern in ["config.NormMode.", "config.NormMode::"] {
            let temp = tempfile::tempdir().unwrap();
            let user = user_of("config.NormMode", pattern);
            let files = [
                ("config.mind", CONFIG),
                ("other.mind", other),
                ("user.mind", &user),
            ];
            let (ok, text) = check_files(temp.path(), &files);
            assert!(
                ok,
                "{tag} {pattern}: the qualified spelling was refused:\n{text}"
            );
        }
    }
}

#[test]
fn a_bare_name_two_imports_denote_as_one_type_resolves() {
    let temp = tempfile::tempdir().unwrap();
    let user = user_of("NormMode", "NormMode::");
    let files = [
        ("config.mind", CONFIG),
        ("other.mind", OTHER_ALIAS),
        ("user.mind", &user),
    ];
    let (ok, text) = check_files(temp.path(), &files);
    assert!(
        ok,
        "a re-export of the same enum is not an ambiguity:\n{text}"
    );
}

#[test]
fn a_bare_name_two_imports_denote_as_different_types_stays_refused() {
    let temp = tempfile::tempdir().unwrap();
    let user = user_of("NormMode", "NormMode::");
    let files = [
        ("config.mind", CONFIG),
        ("other.mind", OTHER_STRUCT),
        ("user.mind", &user),
    ];
    let (ok, text) = check_files(temp.path(), &files);
    assert!(!ok, "a bare NormMode names an enum and a struct:\n{text}");
    assert!(text.contains("ambiguous bare type `NormMode`"), "{text}");
    assert!(text.contains("ambiguous enum `NormMode`"), "{text}");
}

#[test]
fn a_flat_qualified_variant_resolves_beside_an_alias_reexport() {
    let temp = tempfile::tempdir().unwrap();
    let config = "enum NormMode {\n    GroupNorm = 0,\n    Clamp = 1,\n}\nexport enum NormMode\n";
    let other = "use config\ntype NormMode = config.NormMode\nexport type NormMode\n";
    let user = "use config\nuse other\nfn pick(m: config.NormMode) -> i32 {\n    match m {\n        config.NormMode::GroupNorm => {\n            1\n        },\n        config.NormMode::Clamp => {\n            2\n        },\n    }\n}\n";
    let files = [
        ("config.mind", config),
        ("other.mind", other),
        ("user.mind", user),
    ];
    let (ok, text) = check_files(temp.path(), &files);
    assert!(ok, "config.NormMode::GroupNorm in a flat layout:\n{text}");
}

/// Two modules whose aliases name each other form a cycle. The checker must
/// stop on it and still accept the program, not loop. The deadline turns a hang
/// into a failure instead of a stalled test run. It is in the test (no `timeout`
/// program: macOS has none), and the output goes to a file, so a full pipe
/// cannot stall the child.
#[test]
fn a_cyclic_alias_pair_does_not_hang_the_check() {
    let temp = tempfile::tempdir().unwrap();
    let a = "module a {\n    use b\n    type NormMode = b.NormMode\n    export type NormMode\n}\n";
    let b = "module b {\n    use a\n    type NormMode = a.NormMode\n    export type NormMode\n}\n";
    let user = "module user {\n    use config\n    use a\n    fn id(m: config.NormMode) -> config.NormMode {\n        m\n    }\n}\n";
    let files = [
        ("config.mind", CONFIG),
        ("a.mind", a),
        ("b.mind", b),
        ("user.mind", user),
    ];
    for (name, src) in files {
        fs::write(temp.path().join(name), src).unwrap();
    }
    let log = temp.path().join("check.log");
    let sink = fs::File::create(&log).unwrap();
    let mut child = Command::new(mindc())
        .current_dir(temp.path())
        .args(["check", "--no-fmt"])
        .args(files.iter().map(|(name, _)| *name))
        .stdout(sink.try_clone().unwrap())
        .stderr(sink)
        .spawn()
        .expect("run mindc check");
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("wait on mindc check") {
            break status;
        }
        if start.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            let _ = child.wait();
            let text = fs::read_to_string(&log).unwrap_or_default();
            panic!("the check hung:\n{text}");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let text = fs::read_to_string(&log).unwrap_or_default();
    assert!(status.success(), "a cyclic alias pair:\n{text}");
}
