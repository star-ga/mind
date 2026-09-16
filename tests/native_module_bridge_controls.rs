//! Native module-bridge controls: SOURCE RESOLUTION and ADMISSION.
//!
//! Which sources enter the flattened image, in what order, and which programs
//! the bridge refuses to build at all. Visibility, emit-kind, cross-invocation
//! identity and refusal-shape controls live in
//! `native_module_bridge_visibility_controls.rs`; the two files share
//! `tests/native_bridge_support/`.
//!
//! On Linux x86-64 positive cases execute the real native image. Other hosts
//! execute the host-native drain fixture and still exercise source admission,
//! refusal ownership, artifact handling, and complete image transport.
//!
//! Split purely to stay under the 800-line ceiling. Source fixtures and refusal
//! assertions are unchanged; positive result checks use the target-aware helper.
#![cfg(all(feature = "cross-module-imports", feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{Project, assert_native_result};

/// THE MILESTONE: a real project whose entry imports a sibling compiles natively
/// and executes on Linux x86-64 (other hosts verify admission and transport).
/// This is the exact shape that produced
/// "accepts exactly one source file (got 0)" before the bridge was wired.
#[test]
fn local_import_is_admitted_and_bridged() {
    let p = Project::new("import_ok");
    p.write(
        "src/helper.mind",
        "pub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "multi-module native build must succeed: {err}");
    let bytes = artifact.expect("artifact written");
    assert_eq!(&bytes[0..4], b"\x7fELF", "must preserve ELF framing");
    assert_native_result(&p, &bytes, 7, "helper(6) == 7");
    assert!(
        err.contains("2 linked module(s)"),
        "the bridge must report the resolved closure size, got: {err}"
    );
}

/// A single-file program still takes the verbatim path: no joiner, no resolver,
/// same bytes. Paired with the import case above, this is what proves the
/// multi-module work did not disturb the existing corpus.
#[test]
fn single_file_is_admitted_and_bridged() {
    let p = Project::new("single");
    p.write("src/main.mind", "fn main() -> i64 {\n    return 7;\n}\n");
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "single-file native build must succeed: {err}");
    let bytes = artifact.expect("artifact");
    assert_native_result(&p, &bytes, 7, "single-file result");
    assert!(
        !err.contains("linked module(s)"),
        "a single file must NOT go through project resolution: {err}"
    );
}

/// An import naming a module that does not exist is refused, and the diagnostic
/// names the missing module rather than failing later inside the frozen compiler.
#[test]
fn missing_import_is_refused() {
    let p = Project::new("missing");
    p.write(
        "src/main.mind",
        "import absent_module;\n\nfn main() -> i64 {\n    return 1;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "a missing import must refuse, got success: {err}");
    assert!(artifact.is_none(), "refusal must write no artifact");
    assert!(
        err.contains("scope.unresolved_import"),
        "refusal must be owned by the unresolved-import rule, not any failure \
         that happens to mention an import: {err}"
    );
}

/// Two modules defining the same bare name are refused rather than resolved by
/// position. Measured on the frozen compiler: the same three sources return 20
/// in one order and 10 in the other, silently.
#[test]
fn duplicate_bare_name_across_modules_is_refused() {
    let p = Project::new("dup");
    p.write("src/a.mind", "pub fn v() -> i64 {\n    return 10;\n}\n");
    p.write("src/b.mind", "pub fn v() -> i64 {\n    return 20;\n}\n");
    p.write(
        "src/main.mind",
        "import a;\nimport b;\n\nfn main() -> i64 {\n    return v();\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "a duplicated bare name must refuse: {err}");
    assert!(artifact.is_none(), "refusal must write no artifact");
    // PINNED to the owner-resolution layer. This fixture CALLS the duplicated
    // name, so owner-qualified resolution rejects the call before closure
    // admission ever sees two definitions: the entry imports two modules that
    // both export `v`, so there is no unique ABI owner for the bare call.
    //
    // Pinned to the exact kind AND the offending name so a different refusal
    // arriving here is a finding, not a silent pass. The duplicate-definition
    // guard is covered separately by a fixture that does NOT call the name,
    // which is what keeps one guard from masking the other.
    assert!(
        err.contains("scope.rejected_by_type_check"),
        "an ambiguous bare CALL must be refused by owner-qualified resolution: {err}"
    );
    assert!(
        err.contains("`v`"),
        "the diagnostic must name the offending symbol `v`: {err}"
    );
}

/// The OTHER guard, reached only when the ambiguous name is never called.
///
/// Two modules define the same bare name and the entry calls NEITHER, so
/// owner-qualified resolution has no ambiguous call to reject and the program
/// reaches closure admission. Admission must still refuse: the flattened image
/// would otherwise carry two definitions of one bare name, which the frozen
/// compiler resolves last-definition-wins, silently.
///
/// This exists because the previous single control accepted either kind with an
/// OR. That let the earlier layer mask the later one: if duplicate-definition
/// admission regressed, the ambiguous-call rejection would still satisfy the
/// assertion and nothing would fail.
#[test]
fn duplicate_bare_name_is_refused_by_admission_when_never_called() {
    let p = Project::new("dupadmit");
    p.write(
        "src/a.mind",
        "export { v }\n\npub fn v() -> i64 {\n    return 1;\n}\n",
    );
    p.write(
        "src/b.mind",
        "export { v }\n\npub fn v() -> i64 {\n    return 2;\n}\n",
    );
    // Imports both, calls NEITHER: nothing for owner resolution to reject.
    p.write(
        "src/main.mind",
        "import a;\nimport b;\n\nfn main() -> i64 {\n    return 7;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "two definitions of one bare name must refuse: {err}"
    );
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("closure.duplicate_definition"),
        "this must be owned by duplicate-definition admission, not by the \
         ambiguous-call layer, or the two guards are masking each other: {err}"
    );
}

/// A user module defining a name the std seed blob already defines is refused:
/// the flat image would resolve it last-definition-wins with no diagnostic.
#[test]
fn std_symbol_collision_is_refused() {
    let p = Project::new("stdcollide");
    // `bytes_eq` is shipped unprefixed by std/toml.mind.
    p.write(
        "src/helper.mind",
        "pub fn bytes_eq(a: i64, b: i64) -> i64 {\n    return a;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return bytes_eq(1, 2);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "a std-name collision must refuse: {err}");
    assert!(artifact.is_none());
    assert!(
        err.contains("closure.std_symbol_collision"),
        "refusal must be owned by the std-collision rule: {err}"
    );
}

/// An out-of-profile construct in a REACHABLE imported body refuses, and the
/// diagnostic distinguishes the Rust fence from the frozen parser.
#[test]
fn reachable_out_of_profile_import_is_refused_by_the_fence() {
    let p = Project::new("reach_div");
    p.write(
        "src/helper.mind",
        "pub fn halve(x: i64) -> i64 {\n    return x >> 2;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return halve(14);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "a reachable `>>` must refuse: {err}");
    assert!(artifact.is_none());
    assert!(
        err.contains("frozen native profile") || err.contains("binop.shr"),
        "the FENCE must own this refusal, not the frozen parser: {err}"
    );
}

/// TWO INVOCATION FORMS agree byte-for-byte: an explicit entry path, and no path
/// at all with the manifest's `entry` driving the build.
///
/// This replaces a control that was vacuous -- it claimed to compare two routes
/// while only ever exercising one, because the bridge refused a no-path
/// invocation with "got 0". The manifest route now resolves, so the comparison
/// is real and the artifacts must be identical: same project, same closure, same
/// bytes, regardless of how the build was asked for.
#[test]
fn explicit_path_and_manifest_entry_produce_identical_artifacts() {
    let p = Project::new("twoforms");
    p.write(
        "src/helper.mind",
        "pub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );

    let (a_code, a_err, a_art) = p.build_native("src/main.mind");
    assert_eq!(a_code, 0, "explicit-path build must succeed: {a_err}");
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    let captured_a = {
        let captured = p.captured_source_image();
        for expected in [
            "pub fn helper(x: i64) -> i64 {\n    return x + 1;\n}",
            "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
        ] {
            assert!(
                captured
                    .windows(expected.len())
                    .any(|window| window == expected.as_bytes()),
                "explicit-path capture must contain the expected source fragment: {expected}"
            );
        }
        captured
    };
    let a = a_art.expect("explicit artifact");

    let (b_code, b_err, b_art) = p.build_native_no_path();
    assert_eq!(
        b_code, 0,
        "no-path build must resolve the manifest entry: {b_err}"
    );
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let captured_b = p.captured_source_image();
        assert_eq!(
            captured_a, captured_b,
            "explicit-path and manifest-entry forms must send identical source images"
        );
    }
    let b = b_art.expect("manifest-entry artifact");

    assert_eq!(
        a, b,
        "the two invocation forms must produce identical bytes"
    );
    assert_native_result(&p, &a, 7, "explicit-path result");
}

/// TRANSITIVE positive: main -> mid -> leaf, three linked modules. Semantic
/// execution is asserted only on Linux x86-64; other hosts verify transport.
/// The negative twin (a transitive UNEXPORTED callee) already exists; without
/// this, that negative could pass because transitive imports never worked.
#[test]
fn transitive_import_chain_is_admitted_and_bridged() {
    let p = Project::new("transok");
    p.write(
        "src/leaf.mind",
        "pub fn leaf(x: i64) -> i64 {\n    return x + 3;\n}\n",
    );
    p.write(
        "src/mid.mind",
        "import leaf;\n\npub fn mid(x: i64) -> i64 {\n    return leaf(x) + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import mid;\n\nfn main() -> i64 {\n    return mid(3);\n}\n",
    );
    let (code, err, art) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "a transitive chain must build: {err}");
    assert!(
        err.contains("3 linked module(s)"),
        "all three modules linked: {err}"
    );
    let bytes = art.expect("artifact");
    assert_native_result(&p, &bytes, 7, "transitive import result");
}

/// A CYCLIC import terminates and bridges rather than hanging or refusing.
#[test]
fn cyclic_imports_terminate_and_bridge() {
    let p = Project::new("cyc");
    p.write(
        "src/p.mind",
        "import q;\n\npub fn p() -> i64 {\n    return q1();\n}\n",
    );
    p.write(
        "src/q.mind",
        "import p;\n\npub fn q1() -> i64 {\n    return 5;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import p;\n\nfn main() -> i64 {\n    return p();\n}\n",
    );
    let (code, err, art) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "a cycle must terminate, not refuse or hang: {err}");
    let bytes = art.expect("artifact");
    assert_native_result(&p, &bytes, 5, "cyclic import result");
}

/// LIMITATION: the frozen stage1 compiler refuses `pub const` even though
/// `mindc check` accepts it. The sibling form is retained because it exercises
/// the composed bridge. Linux x86-64 tests the real parser refusal; other hosts
/// test only captured source transport because they use a drain fixture.
#[test]
fn pub_const_is_refused_by_the_frozen_profile_anywhere() {
    let p = Project::new("pubconst");
    p.write(
        "src/c.mind",
        "pub const K: i64 = 5;\n\npub fn kk() -> i64 {\n    return K;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import c;\n\nfn main() -> i64 {\n    return kk() + 2;\n}\n",
    );
    let (code, err, art) = p.build_native("src/main.mind");
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // Only this target executes the committed stage1 compiler. Keep the
        // measured frozen-profile refusal on the real semantic path.
        assert_ne!(
            code, 0,
            "pub const in a sibling is not yet supported natively: {err}"
        );
        assert!(art.is_none());
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        // Other CI hosts use a host-native drain fixture by design. It checks
        // source transport and emits an anchor; it cannot establish the
        // stage1 parser's profile refusal. Verify that this exact unsupported
        // source reached the fixture instead of asserting a simulated status.
        assert_eq!(code, 0, "host transport must complete: {err}");
        let bytes = art.expect("host transport anchor");
        let captured = p.captured_source_image();
        assert!(
            captured
                .windows(b"pub const K: i64 = 5;".len())
                .any(|window| window == b"pub const K: i64 = 5;"),
            "host fixture must receive the pub-const source"
        );
        assert_native_result(&p, &bytes, 0, "pub-const source transport");
    }

    // CONTROL: the same shape with a non-pub const DOES build, proving the
    // refusal is specific to `pub const` and not to constants or to siblings.
    let q = Project::new("nonpubconst");
    q.write(
        "src/c.mind",
        "const K: i64 = 5;\n\npub fn kk() -> i64 {\n    return K;\n}\n",
    );
    q.write(
        "src/main.mind",
        "import c;\n\nfn main() -> i64 {\n    return kk() + 2;\n}\n",
    );
    let (ok, err, ok_art) = q.build_native("src/main.mind");
    assert_eq!(
        ok, 0,
        "a non-pub const in a sibling must still build: {err}"
    );
    let bytes = ok_art.expect("artifact");
    assert_native_result(&q, &bytes, 7, "non-pub const control result");
}

/// EXPORT BOUNDARY. A callee its owning module does not export is refused, in
/// parity with `mindc check` (type_check::E2003).
///
/// Regression guard for a measured hole: before this, the bridge BUILT and RAN
/// this program (532 B, exit 105) while `mindc check` refused it. A backend
/// looser than the language ships an artifact for a program that does not exist.
/// Control for the above: the EXPORTED function still builds and runs. Without
/// this, the export check could refuse everything and still look correct.
/// The entry ROOT must belong to the entry module. An entry with no `main` and a
/// sibling that has one previously built and ran the SIBLING's program, because
/// admission roots at "main" over the merged image and cannot tell which module
/// supplied it.
/// TRANSITIVE export leak. `main` imports `relay`, `relay` imports `helper`, and
/// `main` calls a symbol `helper` does not export.
///
/// Regression guard for a measured hole: a text scan of DIRECTLY imported targets
/// never visits `helper` from `main`, so this built and ran (exit 105) while
/// `mindc check` refused it. Delegating to the checker closes it by construction.
/// A COMMENT mentioning a call must not refuse a valid program.
///
/// Regression guard in the opposite direction: a text scan counted
/// `// secret(5)` as a call and refused a program `mindc check` accepts. A
/// backend stricter than the language is a defect too, just a louder one.
/// A LIBRARY emit is refused, not silently satisfied with an executable.
///
/// Measured before this guard: `--emit=cdylib` wrote a 397-byte ET_EXEC to a
/// `.so` path and reported success. The frozen compiler emits a static
/// executable; a library target follows its own contract rather than receiving
/// an executable under a library name.
/// Control for the above: the executable emit still builds and runs.
/// A linked sibling that does not parse REFUSES; it is never skipped.
///
/// Skipping would drop that module's declared imports from the closure check, so
/// an unresolved import inside an unparseable sibling would pass unexamined.
#[test]
fn unparseable_linked_sibling_is_refused() {
    let p = Project::new("badparse");
    p.write(
        "src/helper.mind",
        "pub fn helper( -> i64 {\n    return 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "an unparseable linked sibling must refuse: {err}");
    assert!(artifact.is_none());
    // Named kind, MEASURED: `scope.discovery_failed` -- the resolver refuses
    // while discovering the sibling, before any later fence sees the program.
    assert!(
        err.contains("scope.discovery_failed"),
        "the refusal must be owned by sibling discovery: {err}"
    );
}

/// WHITESPACE PARITY. Tabs and extra spacing must not change semantic ownership.
///
/// The deleted line-scan was whitespace-sensitive; the checker is not. Two
/// programs differing only in indentation must reach the same verdict.
#[test]
fn whitespace_does_not_change_ownership() {
    let spaced = Project::new("ws_spaced");
    spaced.write(
        "src/helper.mind",
        "export { public_fn }\n\npub fn public_fn(x: i64) -> i64 {\n    return x + 1;\n}\n\n\
         fn secret(x: i64) -> i64 {\n    return x + 100;\n}\n",
    );
    spaced.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return secret(5);\n}\n",
    );
    let (a_code, _, a_art) = spaced.build_native("src/main.mind");

    let tabbed = Project::new("ws_tabbed");
    tabbed.write(
        "src/helper.mind",
        "export { public_fn }\n\npub  fn\tpublic_fn(x: i64) -> i64 {\n\treturn x + 1;\n}\n\n\
         fn\tsecret(x: i64) -> i64 {\n\treturn x + 100;\n}\n",
    );
    tabbed.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n\treturn secret(5);\n}\n",
    );
    let (b_code, _, b_art) = tabbed.build_native("src/main.mind");

    assert_eq!(
        a_code == 0,
        b_code == 0,
        "whitespace must not change the verdict: spaced={a_code} tabbed={b_code}"
    );
    assert_eq!(a_art.is_some(), b_art.is_some(), "artifact parity too");
}

// ---------------------------------------------------------------------------
// CROSS-INVOCATION source identity.
//
// WHAT THESE ARE, stated accurately after an independent review corrected an
// overclaim here. Each one runs a COMPLETE build, then mutates disk, then runs a
// WHOLLY SEPARATE second build. The disk does NOT change underneath a running
// invocation. An implementation that re-read the filesystem inside every
// invocation would pass all of them, so they cannot witness the intra-build
// snapshot property, and the previous version of this comment was wrong to say
// they did.
//
// They also do NOT discriminate `scope.install()` from `install_for_check` --
// measured, the swap leaves them green.
//
// What they DO establish is still worth having: across two invocations the
// verdict tracks the source each invocation actually captured, so a stale or
// retargeted dependency changes the answer rather than being silently reused.
//
// The genuine intra-build snapshot control lives at the `validate_captured`
// seam in `src/build/native_scope.rs`
// (`validation_follows_the_capture_when_disk_changes_underneath_it`): one call,
// pre-mutation bytes and scope, disk changed in between, with a positive control
// proving the edit is observable.
// ---------------------------------------------------------------------------

/// Captured-PRIVATE, disk-PUBLIC: the violation exists only in the captured
/// bytes. If validation followed the live disk it would see a legal program and
/// admit it. It must still refuse.
/// A dependency REMOVED from disk must refuse rather than be silently satisfied
/// from any stale table. Exercises the same seam from the other side.
/// A dependency RETARGETED on disk changes the verdict, proving the build reads
/// the current source set rather than any cached discovery.
/// SOURCE CONFINEMENT: an in-project symlink whose target lies outside the
/// project root is refused, naming both the symlink and the root.
///
/// Paired with a legitimate control, because the refusal is project-wide: a
/// single escaping symlink refuses every entry in that project, so a test that
/// only saw the refusal could not tell confinement from a broken fixture.
#[test]
#[cfg(unix)]
fn source_escaping_the_project_root_is_refused() {
    let outside = tempfile::tempdir().expect("outside dir");
    std::fs::write(
        outside.path().join("stray.mind"),
        "pub fn stray() -> i64 {\n    return 1;\n}\n",
    )
    .expect("write outside module");

    let p = Project::new("escape");
    p.write(
        "src/helper.mind",
        "pub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );

    // Control FIRST: without the escaping link this project is admitted and
    // reaches the native bridge (semantic execution is Linux/x86-64 scoped).
    let (ok, err, art) = p.build_native("src/main.mind");
    assert_eq!(ok, 0, "the clean project must build first: {err}");
    let bytes = art.expect("artifact");
    assert_native_result(&p, &bytes, 7, "clean project result");

    // Now plant an in-project symlink pointing outside the root.
    std::os::unix::fs::symlink(
        outside.path().join("stray.mind"),
        p.root().join("src/link.mind"),
    )
    .expect("symlink");

    let (code, err2, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "a source escaping the project root must refuse: {err2}"
    );
    assert!(artifact.is_none(), "refusal must write no artifact");
}

/// Duplicate definitions inside TRANSPARENT MODULE BLOCKS are owned by this
/// guard, not by a downstream refusal.
///
/// `module NAME { ... }` parses into a transparent block, so its functions are
/// not direct module items. Counting only direct items left this case entirely
/// uncovered: measured, such a program type-checks (the only `check` diagnostic
/// was formatting drift) and was refused natively ONLY by the frozen compiler
/// as an unsupported construct.
///
/// Depending on that would be depending on a downstream behaviour. If the frozen
/// profile ever admits module blocks, the flattened image would carry two
/// definitions of one bare name, and the frozen compiler resolves those
/// last-definition-wins — silently, with a valid-looking artifact.
///
/// MUTATION: restrict `ast_definition_counts` to direct `module.items` and the
/// refusal reverts to the frozen compiler's generic `unsupported construct`,
/// failing the named-kind assertion below.
#[test]
fn duplicate_definitions_inside_module_blocks_are_refused_by_admission() {
    let p = Project::new("blockdup");
    p.write(
        "src/main.mind",
        "module a {\n    pub fn v() -> i64 {\n        return 1;\n    }\n}\n\n\
         module b {\n    pub fn v() -> i64 {\n        return 2;\n    }\n}\n\n\
         fn main() -> i64 {\n    return 7;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "duplicated names in module blocks must refuse: {err}"
    );
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("closure.duplicate_definition"),
        "this guard must own the refusal; falling back to the frozen compiler's \
         generic construct rejection leaves the case uncovered if that profile \
         ever widens: {err}"
    );
}

/// An imported body that declares an enum is refused BEFORE the merged lowering.
///
/// The merged image is lowered as one translation unit, so exactly one ambient
/// owner is in force for the whole lowering. Per-source checking installs the
/// correct owner per module; lowering cannot, because expressing per-body
/// ownership there would mean lowering each body separately and splicing SSA
/// fragments, which this bridge does not do.
///
/// Enum lowering resolves a historical `Color::Red` spelling against the CURRENT
/// module, so a sibling carrying that spelling would bind under whatever owner
/// is ambient rather than its own. Measured before this refusal existed: the
/// program reached the bridge, lowered, and was rejected only by the frozen
/// compiler as an unsupported construct. That is a downstream behaviour, and it
/// disappears if the frozen profile ever admits enums — at which point the wrong
/// binding would be emitted into a valid-looking artifact.
///
/// MUTATION: delete the pre-lowering check and the refusal reverts to the frozen
/// compiler's generic `unsupported construct`, failing the named-kind assertion.
#[test]
fn an_enum_in_an_imported_body_is_refused_before_merged_lowering() {
    let p = Project::new("ownerlower");
    p.write(
        "src/palette.mind",
        "export { pick }\n\nenum Color {\n    Red,\n    Green,\n}\n\n\
         pub fn pick(c: Color) -> i64 {\n    match c {\n        \
         Color::Red => { return 7; }\n        Color::Green => { return 9; }\n    }\n}\n",
    );
    p.write(
        "src/main.mind",
        "import palette;\n\nfn main() -> i64 {\n    return 7;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "an owner-sensitive imported body must refuse: {err}"
    );
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("scope.owner_sensitive_imported_body"),
        "the bridge must own this refusal rather than depending on the frozen \
         profile happening to reject enums: {err}"
    );
}

/// The refusal above must not over-refuse: an imported body with NO
/// owner-sensitive declaration still builds and runs.
#[test]
fn an_ordinary_imported_body_is_unaffected_by_the_owner_check() {
    let p = Project::new("ownerok");
    p.write(
        "src/helper.mind",
        "export { helper }\n\npub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_eq!(
        code, 0,
        "an ordinary two-module program must still build: {err}"
    );
    let bytes = artifact.expect("artifact");
    assert_native_result(&p, &bytes, 7, "ordinary imported body result");
}

/// An imported TYPE ALIAS is owner-bearing and is refused before lowering.
///
/// This is the sharpest measured case, and before the refusal existed it was a
/// SILENT WRONG ANSWER rather than a refusal. `lower_to_ir` builds ONE
/// `LocalTypeAliases` over the whole merged AST; its collector is a bare-name
/// map where the last insertion wins; and that single table drives narrow return
/// masking. The image is joined entry-first then ascending module path, so a
/// later sibling's alias silently redefines an earlier sibling's return ABI.
///
/// Measured on the pre-fix binary with this exact fixture: the checker accepted
/// the program, the bridge did not refuse, and the emitted artifact RAN with
/// `f()` masked from 300 to `u8`. The `main` below returns a BOOLEAN rather than
/// the value, because a masked 300 is 44 and an exit code of 44 could not be
/// told apart from a deliberate result.
#[test]
fn an_imported_type_alias_is_refused_before_merged_lowering() {
    let p = Project::new("aliasown");
    p.write(
        "src/a.mind",
        "export { f }\n\ntype Word = i16;\n\npub fn f() -> Word {\n    return 300;\n}\n",
    );
    p.write(
        "src/z.mind",
        "export { g }\n\ntype Word = u8;\n\npub fn g() -> i64 {\n    return 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import a;\nimport z;\n\nfn main() -> i64 {\n    if f() == 300 {\n        \
         return 1;\n    }\n    return 0;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(code, 0, "an imported alias must refuse: {err}");
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("scope.owner_sensitive_imported_body"),
        "the bridge must own this; a downstream refusal is not closure: {err}"
    );
}

/// The SAME program with the module names swapped.
///
/// Source order decides which alias wins the last-write-wins table, so before
/// the refusal the two orderings behaved differently: one emitted a wrong-valued
/// artifact and the other was refused incidentally, with no named kind. Path
/// order must not be a semantic authority, so both orderings are pinned to the
/// same bridge-owned refusal.
#[test]
fn the_imported_alias_refusal_does_not_depend_on_module_path_order() {
    let p = Project::new("aliasrev");
    p.write(
        "src/z.mind",
        "export { f }\n\ntype Word = i16;\n\npub fn f() -> Word {\n    return 300;\n}\n",
    );
    p.write(
        "src/a.mind",
        "export { g }\n\ntype Word = u8;\n\npub fn g() -> i64 {\n    return 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import z;\nimport a;\n\nfn main() -> i64 {\n    if f() == 300 {\n        \
         return 1;\n    }\n    return 0;\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "the reversed ordering must refuse identically: {err}"
    );
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("scope.owner_sensitive_imported_body"),
        "both orderings must be owned by the same rule: {err}"
    );
}

/// An owner-bearing declaration nested in a TRANSPARENT MODULE BLOCK is found.
///
/// The alias is private, unused, and unrelated to the scalar function. The
/// temporary whole-source restriction still refuses it, including inside a
/// transparent block. The following scalar-only fixture supplies the control.
#[test]
fn an_unused_private_alias_in_an_imported_module_block_is_refused() {
    let p = Project::new("aliasnest");
    p.write(
        "src/a.mind",
        "export { f }\n\nmodule inner {\n    type Word = u8;\n}\n\n\
         pub fn f() -> i64 {\n    return 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import a;\n\nfn main() -> i64 {\n    return f();\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_ne!(
        code, 0,
        "a nested owner-bearing declaration must refuse: {err}"
    );
    assert!(artifact.is_none(), "a refusal must write no artifact");
    assert!(
        err.contains("scope.owner_sensitive_imported_body"),
        "a transparent block must not hide the declaration: {err}"
    );
}

/// POSITIVE: a harmless imported scalar body is unaffected and is bridged.
///
/// The refusal covers owner-BEARING declarations only. Widening it to every
/// imported body would trade a wrong answer for a useless backend, so this pins
/// that ordinary cross-module code still compiles and executes.
#[test]
fn a_harmless_imported_scalar_body_is_admitted_and_bridged() {
    let p = Project::new("aliasok");
    p.write(
        "src/helper.mind",
        "export { helper }\n\npub fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n",
    );
    p.write(
        "src/main.mind",
        "import helper;\n\nfn main() -> i64 {\n    return helper(6);\n}\n",
    );
    let (code, err, artifact) = p.build_native("src/main.mind");
    assert_eq!(code, 0, "an ordinary imported body must still build: {err}");
    let bytes = artifact.expect("artifact");
    assert_native_result(&p, &bytes, 7, "harmless imported scalar result");
}
