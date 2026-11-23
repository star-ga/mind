//! Default-feature-profile controls for the native backend's import admission.
//!
//! These exist because the 24 controls in `native_module_bridge_controls.rs` are
//! erased wholesale by `#![cfg(all(feature = "cross-module-imports", feature =
//! "std-surface"))]`, so NONE of them can observe the branch a default
//! `cargo build` actually ships. This file is their complement: it compiles only
//! when the project resolver is absent, which is the profile CI builds.
//!
//! What it guards, measured on the pre-fix binary before the fix landed:
//! a source declaring a dependency the build cannot resolve COMPILED AND RAN.
//! `import\tabsent_module;` and `use absent_module;` each reached the native
//! bridge's 397-byte ELF fixture, which ran with exit 1. The old detector was a
//! source-text scan, `lines().map(trim_start).any(|l| l.starts_with("import "))`,
//! and it missed both spellings: the grammar accepts any whitespace after the
//! keyword, and `use` is parsed by `parse_use` into the very same `Node::Import`.
//! Lowering treats every `Node::Import` as a compile-time no-op, so no later
//! fence recovers the dropped fact.
//!
//! The same scan was also wrong in the other direction: it refused
//! `import std.math;`, which is satisfied by the seed std blob and is not a
//! local dependency at all.
//!
//! The current fixture is a host executable that drains and captures the input
//! image, then returns fixed bytes beginning with ELF magic. It does not compile
//! MIND or produce an executable ELF. These controls prove admission and source
//! transport, not native code generation or cross-platform byte identity. The
//! host runs only a compiler that carries the native-lane-v1 mark `nblane02`, so
//! the fixture is built with that mark (`--cfg native_lane_mark`).

#![cfg(all(not(feature = "cross-module-imports"), feature = "std-surface"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

const MINDC: &str = env!("CARGO_BIN_EXE_mindc");

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn native_fixture() -> PathBuf {
    static FIXTURE: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let dir = tempfile::tempdir().expect("native fixture directory");
            let root = dir.keep();
            let output = root.join(if cfg!(windows) {
                "draining_native_compiler.exe"
            } else {
                "draining_native_compiler"
            });
            let status = Command::new("rustc")
                .args([
                    "--edition",
                    "2024",
                    "--crate-name",
                    "draining_native_compiler",
                    "--cfg",
                    "native_lane_mark",
                ])
                .arg(repo_path(
                    "tests/native_bridge_support/draining_native_compiler.rs",
                ))
                .arg("-o")
                .arg(&output)
                .status()
                .expect("compile native fixture");
            assert!(
                status.success(),
                "native fixture compilation failed: {status}"
            );
            output
        })
        .clone()
}

struct Project {
    dir: tempfile::TempDir,
}

impl Project {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::create_dir_all(root.join("src")).expect("src");
        std::fs::write(
            root.join("Mind.toml"),
            "[package]\nname = \"dp\"\nversion = \"0.1.0\"\n\n\
             [build]\ntarget = \"cpu\"\nemit = \"binary\"\nentry = \"src/main.mind\"\n",
        )
        .expect("manifest");
        assert!(
            Command::new("git")
                .args(["init", "-q"])
                .current_dir(root)
                .status()
                .expect("git init")
                .success(),
            "git init must succeed"
        );
        Self { dir }
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }

    fn captured_image(&self) -> PathBuf {
        self.root().join("captured-native-image.bin")
    }

    /// (exit code, stderr, artifact bytes)
    fn build(&self, main_src: &str) -> (i32, String, Option<Vec<u8>>) {
        std::fs::write(self.root().join("src/main.mind"), main_src).expect("write main");
        let out = self.root().join("out.elf");
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(self.captured_image());
        let res = Command::new(MINDC)
            .args([
                "build",
                "src/main.mind",
                "--release",
                "--backend",
                "native",
                "--emit=binary",
                "--out",
            ])
            .arg(&out)
            .current_dir(self.root())
            .env("MINDC_STD_DIR", repo_path("std"))
            .env("MINDC_NATIVE_ELF", native_fixture())
            .env("MIND_NATIVE_TEST_CAPTURE", self.captured_image())
            .output()
            .expect("spawn mindc");
        (
            res.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&res.stderr).into_owned(),
            std::fs::read(&out).ok(),
        )
    }
}

/// Assert the shape root's review requires of a wrong-acceptance fix: nonzero
/// exit, NO artifact, and the refusal owned by the named rule rather than by
/// whatever else might happen to fail.
fn assert_refused(label: &str, got: (i32, String, Option<Vec<u8>>)) {
    let (code, err, art) = got;
    assert_ne!(code, 0, "{label} must refuse, got success: {err}");
    assert!(
        art.is_none(),
        "{label} must write NO artifact; a refusal that still emits is the wrong acceptance itself"
    );
    assert!(
        err.contains("scope.imports_need_feature"),
        "{label} must be refused by the unresolved-local-import rule, not by an \
         unrelated failure that happens to be nonzero: {err}"
    );
}

fn assert_fixture_drained(p: &Project, source: &str) {
    let captured = std::fs::read(p.captured_image())
        .expect("the successful fixture must capture the complete image");
    assert!(
        captured.ends_with(source.as_bytes()),
        "the fixture must consume the complete user source before emitting"
    );
}

/// The exact spelling from the independent review. A tab after the keyword is a
/// real `Node::Import`; the old text scan required a literal space.
///
/// MUTATION: restore `lines().any(|l| l.starts_with("import "))` in the
/// featureless `resolve_native_sources` and this test goes red by BUILDING --
/// measured, 397-byte ELF fixture, run exit 1.
#[test]
fn a_tab_after_the_import_keyword_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "import<TAB>absent_module",
        p.build("import\tabsent_module;\n\nfn main() -> i64 {\n    return 1;\n}\n"),
    );
}

/// `use` is parsed by `parse_use` into the same `Node::Import` as `import`.
/// The old detector never looked at the `use` spelling at all.
#[test]
fn the_use_spelling_is_the_same_node_and_is_refused() {
    let p = Project::new();
    assert_refused(
        "use absent_module",
        p.build("use absent_module;\n\nfn main() -> i64 {\n    return 1;\n}\n"),
    );
}

/// Positive control for the pair above: the plain spelling was ALREADY refused
/// before the fix. Without this, the two tests above could pass on a build that
/// refuses everything, and prove nothing about whitespace or `use`.
#[test]
fn the_plain_spelling_is_refused_too() {
    let p = Project::new();
    assert_refused(
        "import absent_module",
        p.build("import absent_module;\n\nfn main() -> i64 {\n    return 1;\n}\n"),
    );
}

/// The other direction, and the reason a third whitespace case would have been
/// the wrong fix: `std` imports are satisfied by the seed blob and are NOT local
/// dependencies. The old scan refused this program; the featured profile accepts
/// it. A default build must accept it too, or the two profiles disagree about
/// what a valid program is. The module is a `seed` row of the std manifest
/// (`std.sha256`). `std.math` is no seed module: the lane refuses it by name
/// (`lane.import.not_seed`), and that is still not a local-dependency verdict.
#[test]
fn a_std_only_import_is_not_a_local_dependency_and_builds() {
    let p = Project::new();
    let src = "import std.sha256;\n\nfn main() -> i64 {\n    return 7;\n}\n";
    let (code, err, art) = p.build(src);
    assert_eq!(
        code, 0,
        "a std-only import must build in any profile: {err}"
    );
    assert!(art.is_some(), "a successful build must emit an artifact");
    assert_fixture_drained(&p, src);
    let (code, err, art) = p.build("import std.math;\n\nfn main() -> i64 {\n    return 7;\n}\n");
    assert_eq!(code, 3, "a std module outside the seed is refused: {err}");
    assert!(art.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("(lane.import.not_seed)") && !err.contains("scope.imports_need_feature"),
        "`std.math` is refused by the seed rule, never as a local dependency: {err}"
    );
}

/// A source with no local imports must reach the fixture, and its fixed output
/// must pass through unchanged. Native byte identity has separate compiler gates.
#[test]
fn a_plain_single_file_reaches_the_fixture_and_preserves_its_output() {
    let p = Project::new();
    let src = "fn main() -> i64 {\n    return 7;\n}\n";
    let (code, err, art) = p.build(src);
    assert_eq!(code, 0, "a single file must build with no resolver: {err}");
    let bytes = art.expect("artifact");
    let mut expected = b"\x7fELF".to_vec();
    expected.resize(397, 0);
    assert_eq!(
        bytes, expected,
        "fixture output must pass through unchanged"
    );
    assert_fixture_drained(&p, src);
}

/// A small source can fit in a pipe even when a child exits before reading it.
/// This source is deliberately larger than a typical pipe buffer, so the
/// positive control proves the fixture drains the stream before it emits.
#[test]
fn a_large_source_is_drained_before_the_fixture_emits() {
    let p = Project::new();
    let src = format!(
        "{}fn main() -> i64 {{\n    return 7;\n}}\n",
        "// native image drain control\n".repeat(8192)
    );
    let (code, err, art) = p.build(&src);
    assert_eq!(code, 0, "a large source must build: {err}");
    assert_eq!(art.expect("artifact").len(), 397);
    assert_fixture_drained(&p, &src);
}

/// A source this build cannot parse has no enumerable imports, so it is refused
/// rather than compiled as a single file on the assumption it declared none.
/// The featured profile already refuses the same source through
/// `discover_with_source`; this keeps the profiles agreeing.
#[test]
fn an_unparseable_entry_is_refused_rather_than_assumed_import_free() {
    let p = Project::new();
    let (code, err, art) = p.build("fn main( -> { this is not mind source\n");
    assert_ne!(code, 0, "an unparseable entry must refuse: {err}");
    assert!(art.is_none(), "a refusal must write no artifact");
}

// ---------------------------------------------------------------------------
// NESTED shapes. `parse_stmt` accepts `import`/`use` wherever a statement is
// accepted, so an import can sit inside a function body, a branch, or a loop.
//
// Before these landed, the collector recursed into `Node::Block` ONLY, so
// `fn main() -> i64 { import absent_module; return 1; }` escaped it entirely.
// That program is check-clean, and native refused it only because the FROZEN
// compiler rejects a nested import as an unsupported construct -- the right
// answer from the wrong component, which stops protecting anything the moment
// the frozen profile widens.
//
// Each of these asserts the refusal is owned by `scope.imports_need_feature`,
// NOT merely that the exit code was nonzero, precisely because the accidental
// frozen-compiler refusal would satisfy a bare nonzero-exit assertion.
// ---------------------------------------------------------------------------

/// The exact fixture from the independent audit.
#[test]
fn an_import_inside_a_function_body_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "fn body import",
        p.build("fn main() -> i64 {\n    import absent_module;\n    return 1;\n}\n"),
    );
}

#[test]
fn a_use_inside_a_function_body_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "fn body use",
        p.build("fn main() -> i64 {\n    use absent_module;\n    return 1;\n}\n"),
    );
}

#[test]
fn an_import_inside_an_if_branch_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "if branch import",
        p.build(
            "fn main() -> i64 {\n    if 1 == 1 {\n        import absent_module;\n    }\n    \
             return 1;\n}\n",
        ),
    );
}

#[test]
fn an_import_inside_an_else_branch_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "else branch import",
        p.build(
            "fn main() -> i64 {\n    if 1 == 0 {\n        return 2;\n    } else {\n        \
             import absent_module;\n    }\n    return 1;\n}\n",
        ),
    );
}

#[test]
fn an_import_inside_a_while_body_is_still_an_import() {
    let p = Project::new();
    assert_refused(
        "while body import",
        p.build(
            "fn main() -> i64 {\n    while 0 == 1 {\n        import absent_module;\n    }\n    \
             return 1;\n}\n",
        ),
    );
}

/// Negative control for the nested set: widening the traversal must not start
/// treating `std` as a LOCAL dependency.
///
/// CORRECTED after measuring. This first asserted the program builds. It does
/// not, and the reason is not this collector: a nested import of ANY kind is
/// outside the frozen compiler's native profile, so it is refused as an
/// unsupported construct. That is a construct limit, not a dependency verdict.
///
/// The property that actually matters here is therefore stated directly: however
/// this program fares, it must never be refused by `scope.imports_need_feature`,
/// because `std` is satisfied by the seed blob. Asserting "it builds" would have
/// been asserting something about the frozen profile that this change neither
/// controls nor promises.
#[test]
fn a_nested_std_import_is_never_called_a_local_dependency() {
    let p = Project::new();
    let (_code, err, _art) =
        p.build("fn main() -> i64 {\n    import std.math;\n    return 7;\n}\n");
    assert!(
        !err.contains("scope.imports_need_feature"),
        "`std` is satisfied by the seed blob and must never be reported as an \
         unresolvable local dependency, whatever the frozen profile does with a \
         nested import: {err}"
    );
}
