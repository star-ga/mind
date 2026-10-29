//! Native bridge controls: an `import`/`use` item the frozen compiler would
//! read DIFFERENTLY from the admission parser.
//!
//! The Rust parser treats the `;` after an import path as optional. The frozen
//! pure-MIND compiler does not: its `parse_use` skips to the first `;` TOKEN,
//! so `import helper` followed by `fn main() -> i64 { return seven() + 35 }`
//! swallows `main` (and, with no `;` anywhere after it, every remaining item).
//! With no `main` left it synthesizes the library entry stub, so the build
//! "succeeded" with a 349-byte ELF whose only code was `exit(0)`, while the
//! same project exits 42 under `--backend mlir`.
//!
//! The bridge now refuses such a program by name before the frozen compiler
//! runs, and independently refuses any emitted artifact whose entry is the
//! no-`main` stub (the backstop for any other way `main` could vanish).
#![cfg(all(feature = "cross-module-imports", feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{Project, assert_native_result};

const KIND: &str = "native.import_not_semicolon_terminated";

fn reproducer(p: &Project, import_line: &str) {
    p.write(
        "src/helper.mind",
        "export { seven }\npub fn seven() -> i64 {\n    return 7\n}\n",
    );
    p.write(
        "src/main.mind",
        &format!("{import_line}\n\nfn main() -> i64 {{\n    return seven() + 35\n}}\n"),
    );
}

/// THE AUDITED REPRODUCER (ML-3 / NATXMOD), byte-for-byte: a two-module project
/// whose entry `import helper` has no `;` and whose `main` returns 42. Built
/// with no path argument (the manifest entry drives it), as the auditor did.
/// Before the fix: rc=0 and a 349-byte ELF exiting 0.
#[test]
fn two_module_semicolonless_import_never_succeeds_with_wrong_code() {
    let p = Project::new("v11_import_bare");
    reproducer(&p, "import helper");
    let (code, err, artifact) = p.build_native_no_path();
    if code == 0 {
        // A correct build would also satisfy the axis; a wrong one never may.
        let bytes = artifact.expect("a successful build must write its artifact");
        assert_native_result(&p, &bytes, 42, "seven() + 35 via the native backend");
        panic!("expected the named refusal, but the build succeeded correctly: {err}");
    }
    assert!(
        artifact.is_none(),
        "a refusal must write no artifact: {err}"
    );
    assert!(
        err.contains(KIND),
        "the refusal must be owned by the import-terminator rule: {err}"
    );
    assert!(
        err.contains("import helper"),
        "the diagnostic must quote the offending item: {err}"
    );
}

/// POSITIVE CONTROL for the refusal above: the identical project with the `;`
/// written builds and returns 42, so the rule refuses the dialect gap only,
/// not imports, not multi-module projects, not semicolon-less statements.
#[test]
fn the_same_project_with_a_terminated_import_builds_and_returns_42() {
    let p = Project::new("v11_import_semi");
    reproducer(&p, "import helper;");
    let (code, err, artifact) = p.build_native_no_path();
    assert_eq!(code, 0, "a `;`-terminated import must still build: {err}");
    let bytes = artifact.expect("artifact");
    assert_native_result(&p, &bytes, 42, "seven() + 35 with `import helper;`");
}

/// A `;` separated from the path only by whitespace is the SAME item to the
/// frozen compiler (it scans tokens), so it must not be over-refused.
#[test]
fn whitespace_before_the_semicolon_is_still_terminated() {
    let p = Project::new("v11_import_ws");
    reproducer(&p, "import helper ;");
    let (code, err, artifact) = p.build_native_no_path();
    assert_eq!(code, 0, "`import helper ;` must still build: {err}");
    let bytes = artifact.expect("artifact");
    assert_native_result(&p, &bytes, 42, "seven() + 35 with `import helper ;`");
}

/// The same gap on a SINGLE file: a std import needs no project, so this never
/// reached the resolver and was a silent `exit(0)` too (measured: 349 bytes).
#[test]
fn single_file_semicolonless_std_import_is_refused() {
    for line in ["import std.vec", "use std.vec", "use std::vec"] {
        let p = Project::new("single_std");
        p.write(
            "src/main.mind",
            &format!("{line}\nfn main() -> i64 {{\n    return 42\n}}\n"),
        );
        let (code, err, artifact) = p.build_native("src/main.mind");
        assert_ne!(code, 0, "`{line}` without `;` must refuse: {err}");
        assert!(
            artifact.is_none(),
            "`{line}`: refusal writes nothing: {err}"
        );
        assert!(err.contains(KIND), "`{line}`: wrong refusal owner: {err}");
    }
}

/// An unterminated import NESTED in a function body is refused too: the frozen
/// parser consumes statement-position imports with the same `;` scan.
#[test]
fn nested_semicolonless_import_is_refused() {
    let p = Project::new("nested");
    p.write(
        "src/main.mind",
        "fn main() -> i64 {\n    import std.vec\n    return 42\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "nested unterminated import must refuse: {err}");
    assert!(artifact.is_none(), "refusal writes nothing: {err}");
    assert!(err.contains(KIND), "wrong refusal owner: {err}");
}
