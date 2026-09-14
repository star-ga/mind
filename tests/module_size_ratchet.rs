// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Structural gate: a source file that is already over the size ceiling may not
//! grow, and no new file may cross the ceiling.
//!
//! # Why this file exists
//!
//! The house rule — "200-400 lines typical, 800 max; do not grow an over-limit
//! file, add a new focused module" — was carried only as PROSE. Prose does not
//! fail a build, and three separate landings in one wave added logic to
//! `src/build/mod.rs`, `src/bin/mindc.rs` and `src/project/mod.rs`, each of
//! which was already well past the ceiling. Nothing noticed, because nothing
//! could.
//!
//! # Why the scan is not `src/` only
//!
//! It was, and that hole cost exactly what a half-scoped gate always costs. A
//! later wave pushed `scripts/exec_semantics_gate.sh` from 729 to 843 lines —
//! ACROSS the ceiling, in the file whose entire subject is a check that
//! asserted nothing — and grew two already-over-limit test files, while this
//! gate reported clean. It was looking somewhere else. A scan scope that stops
//! where one wave happened to work is the same drift as a lint whose glob and
//! whose reference detection disagree: it reports on the tree it can see and is
//! silent about the tree it cannot.
//!
//! So the scope is now every TRACKED source file under [`SCANNED_ROOTS`] with
//! an extension in [`SCANNED_EXTENSIONS`] — Rust and the harness alike, because
//! the ceiling is a rule about files a human has to read, not about which
//! toolchain compiles them.
//!
//! The scope is read off `git ls-files`, i.e. off the INDEX, for two reasons a
//! working-tree walk gets wrong: an untracked scratch file would make the
//! verdict depend on what happens to be lying around, and a generated artifact
//! under a scanned root would be judged as source. `scripts/check_gate_wiring.py`
//! reads its harness ratchet the same way, for the same reason.
//!
//! # The two directions, both mechanical
//!
//! * **No file exceeds its budget.** A file not in the table below must be at
//!   or under [`CEILING`]. A file in the table is a pre-existing over-limit
//!   file and must be at or under its PINNED line count — so the only way to
//!   add code to one is to raise its number deliberately, in the diff, where a
//!   reviewer sees it.
//! * **No budget outlives its file.** Every pinned entry must still name a file
//!   that is over the ceiling. An entry whose file was split (or deleted) is a
//!   stale second list, and a stale list is how a gate quietly stops guarding.
//!
//! The budget is a MAXIMUM, not an equality: shrinking an over-limit file is
//! the point of the rule and must never turn the gate red. Re-pinning a budget
//! DOWN after an extraction is what converts a one-time cleanup into a ratchet,
//! and is expected in the same commit as the extraction.
//!
//! This gate deliberately says nothing about whether a file SHOULD be split —
//! only that the over-limit set may not get worse by accident.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The house ceiling. A file at or under this needs no entry below.
const CEILING: usize = 800;

/// The source roots the ceiling governs.
///
/// Kept as roots rather than as a file list: the file set is read off
/// `git ls-files` under these, so adding a source file cannot silently escape
/// the ceiling by not being mentioned here. A root that matches NOTHING fails
/// the gate (see `every_scanned_root_contributes_files`) — a renamed or
/// mistyped root would otherwise shrink the scan in total silence, which is the
/// `ran=0` shape this repository keeps paying for.
const SCANNED_ROOTS: &[&str] = &["benches", "examples", "scripts", "src", "tests", "tools"];

/// The source extensions the ceiling governs: the product (Rust) and the
/// harness (Python, shell). `.mind` is deliberately absent — its files are
/// conformance fixtures and corpora, sized by what they must cover.
const SCANNED_EXTENSIONS: &[&str] = &["py", "rs", "sh"];

/// Pre-existing over-limit files, pinned at the line count they may not
/// exceed. Sorted by path; one entry per file.
///
/// Adding a row is admitting a NEW over-limit file and should be rejected in
/// review in favour of a focused module. Raising a row is admitting growth of
/// one that already exists, and needs the same justification the prose rule
/// always asked for — the difference is that now it cannot happen silently.
///
/// The `examples/` and `tests/` rows entered with the scope widening above.
/// They are the state the widened scan FOUND, pinned so it cannot get worse;
/// pinning them is not a licence to have grown them. Three of them
/// (`tests/mindc_cache_phase_f.rs`, `tests/verify_ssa.rs`,
/// `tests/std_surface_net_fs_process.rs`) grew in the very wave that could not
/// see them, and were carried into shared helpers rather than re-pinned at the
/// grown count — the first of the three left this table altogether, which is
/// what a ratchet is FOR.
const LEGACY_BUDGETS: &[(&str, usize)] = &[
    ("examples/mindc_mind/mic3_primitives_smoke.py", 1665),
    ("examples/mindc_mind/mindfuzz_self_host.py", 1098),
    ("examples/mindc_mind/self_host_native_elf_smoke.py", 869),
    (
        "examples/mindc_mind/self_host_tc_fn_value_call_smoke.py",
        921,
    ),
    (
        "examples/mindc_mind/self_host_tc_undeclared_assign_smoke.py",
        851,
    ),
    (
        "examples/mindc_mind/self_host_tc_unknown_call_smoke.py",
        1090,
    ),
    (
        "examples/mindc_mind/self_host_tc_unknown_ident_smoke.py",
        1413,
    ),
    ("examples/mindc_mind/tc_differential_fuzz.py", 1203),
    // Feature-boundary declarations grow existing AST visitors. The parser also
    // retains fail-closed escapes and adjacent-macro refusal; lowering retains
    // annotated-string environments, and the test runner retains root-cause
    // diagnostics. Pin only this reviewed growth; the 800-line ceiling stays.
    ("src/ast/mod.rs", 1161),
    ("src/bin/mind-ai.rs", 1101),
    ("src/bin/mindc.rs", 3343),
    ("src/build/mod.rs", 943),
    ("src/check/mod.rs", 945),
    ("src/deps/mod.rs", 1060),
    ("src/doc/mod.rs", 888),
    ("src/eval/abi_gate.rs", 1036),
    ("src/eval/autodiff.rs", 1643),
    ("src/eval/lower.rs", 12744),
    ("src/eval/mlir_build.rs", 1002),
    ("src/eval/mlir_export.rs", 1447),
    ("src/eval/mod.rs", 3918),
    ("src/eval/stdlib/tensor.rs", 1097),
    ("src/fmt/printer.rs", 2229),
    ("src/ir/compact/parse.rs", 967),
    ("src/ir/compact/v2/binary.rs", 960),
    ("src/ir/compact/v2/evidence.rs", 1052),
    ("src/ir/compact/v2/map_tests.rs", 917),
    ("src/ir/compact/v3/emit.rs", 1399),
    ("src/ir/compact/v3/evidence.rs", 4005),
    ("src/ir/compact/v3/mod.rs", 2218),
    ("src/ir/compact/v3/parse.rs", 1354),
    ("src/ir/evidence.rs", 1216),
    ("src/ir/fp_mode.rs", 1217),
    ("src/ir/mod.rs", 1519),
    ("src/ir/verify.rs", 1849),
    ("src/mlir/lowering.rs", 12230),
    ("src/opt/collapse.rs", 1228),
    ("src/opt/native_opt.rs", 1137),
    ("src/opt/scev.rs", 871),
    ("src/parser/expand_bimap.rs", 1836),
    ("src/parser/mod.rs", 6170),
    ("src/project/mod.rs", 3587),
    ("src/type_checker/mod.rs", 6577),
    ("src/type_checker/resolve.rs", 1319),
    ("tests/cross_substrate_identity.rs", 3288),
    ("tests/g2_differential_mlir.rs", 1020),
    ("tests/mindc_deps_phase_de.rs", 851),
    ("tests/mindfuzz_cross_substrate.rs", 1486),
    ("tests/return_cond_type_reject.rs", 1007),
    ("tests/std_surface_net_fs_process.rs", 803),
];

fn manifest_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Every TRACKED source file under [`SCANNED_ROOTS`] whose extension is in
/// [`SCANNED_EXTENSIONS`], as `(repo-relative path, line count)`, sorted.
///
/// `git ls-files` emits paths with `/` separators on every host and relative to
/// the directory it runs in, which is the manifest root — so the strings here
/// match [`LEGACY_BUDGETS`] verbatim without any per-host normalisation.
fn tracked_source_sizes() -> Vec<(String, usize)> {
    let root = manifest_root();
    let out = Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--"])
        .args(SCANNED_ROOTS)
        .output()
        .expect(
            "`git ls-files` must run: this gate reads the INDEX, so that an untracked \
             scratch file cannot red it and a generated artifact cannot be judged as \
             source. Run it from a checkout.",
        );
    assert!(
        out.status.success(),
        "`git ls-files` failed ({}). A gate that cannot enumerate its scope has not \
         found a clean tree; it has found nothing.\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let listing = String::from_utf8(out.stdout).expect("git emits utf-8 paths");
    let mut sizes: Vec<(String, usize)> = listing
        .split('\0')
        .filter(|p| !p.is_empty())
        .filter(|p| {
            let ext = Path::new(p)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            SCANNED_EXTENSIONS.contains(&ext)
        })
        .filter_map(|p| {
            // A path in the index with no file on disk is a staged deletion and
            // has no line count. Anything else that fails to read is a real
            // problem, but it cannot be distinguished here without guessing, so
            // the emptiness asserts below are what keep a silent miss visible.
            std::fs::read_to_string(root.join(p))
                .ok()
                .map(|text| (p.to_string(), text.lines().count()))
        })
        .collect();
    sizes.sort();
    sizes
}

/// The budget `path` may not exceed: its pinned entry, else the ceiling.
fn budget_for(path: &str) -> usize {
    LEGACY_BUDGETS
        .iter()
        .find(|(p, _)| *p == path)
        .map_or(CEILING, |(_, n)| *n)
}

/// The over-budget files of `sizes`, as reviewer-readable lines.
fn over_budget(sizes: &[(String, usize)]) -> Vec<String> {
    sizes
        .iter()
        .filter(|(p, n)| *n > budget_for(p))
        .map(|(p, n)| format!("{p}: {n} lines (budget {})", budget_for(p)))
        .collect()
}

#[test]
fn no_source_file_exceeds_its_budget() {
    let sizes = tracked_source_sizes();
    assert!(
        !sizes.is_empty(),
        "the scan found no tracked source file at all, so this gate proves nothing"
    );
    let bad = over_budget(&sizes);
    assert!(
        bad.is_empty(),
        "these files are over the {CEILING}-line ceiling (or over the line count \
         pinned for an already-over-limit file). Carry the new logic into a \
         focused module instead of growing one that is already too large; if the \
         growth is genuinely unavoidable, raise the number in `LEGACY_BUDGETS` in \
         the same commit so the decision is visible.\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn every_scanned_root_contributes_files() {
    // The scope assert. `0 <= ceiling` is the shape of a vacuous pass, and a
    // root that has been renamed, moved or mistyped shrinks the scan to exactly
    // that shape without any other symptom. Per ROOT and per EXTENSION, because
    // either one alone can be silently emptied.
    let sizes = tracked_source_sizes();
    for root in SCANNED_ROOTS {
        let prefix = format!("{root}/");
        let n = sizes.iter().filter(|(p, _)| p.starts_with(&prefix)).count();
        assert!(
            n > 0,
            "`{root}` contributed no tracked {SCANNED_EXTENSIONS:?} file to the scan. \
             Either the directory moved and `SCANNED_ROOTS` was not updated, or the \
             pathspec is wrong — both leave the ceiling unenforced there while this \
             gate keeps reporting clean."
        );
    }
    for ext in SCANNED_EXTENSIONS {
        let suffix = format!(".{ext}");
        let n = sizes.iter().filter(|(p, _)| p.ends_with(&suffix)).count();
        assert!(
            n > 0,
            "no `.{ext}` file reached the scan, so that whole source class is \
             unguarded. Drop the extension from `SCANNED_EXTENSIONS` if it is \
             genuinely gone, rather than leaving a rule with nothing to apply to."
        );
    }
}

#[test]
fn no_budget_outlives_its_file() {
    // The other direction. A pinned row for a file that was split, renamed or
    // deleted is a stale second list: it guards nothing, and it hides the fact
    // that the over-limit set changed. Re-pin it or drop it.
    let sizes = tracked_source_sizes();
    for (path, budget) in LEGACY_BUDGETS {
        let actual = sizes.iter().find(|(p, _)| p == path).map(|(_, n)| *n);
        let Some(actual) = actual else {
            panic!(
                "`LEGACY_BUDGETS` pins {path} at {budget} lines, but no such tracked \
                 source file exists under `SCANNED_ROOTS`. Drop the row — a budget \
                 for a file that is gone is a stale list, not a gate."
            );
        };
        assert!(
            actual > CEILING,
            "`LEGACY_BUDGETS` pins {path} at {budget}, but it is now {actual} lines \
             — at or under the {CEILING}-line ceiling. Drop the row so the ceiling \
             itself guards it; leaving the row would license growing it back."
        );
    }
}

#[test]
fn the_budget_table_is_sorted_and_has_no_duplicates() {
    let mut seen: Vec<&str> = Vec::new();
    for (path, _) in LEGACY_BUDGETS {
        assert!(
            !seen.contains(path),
            "{path} is pinned twice; the second row would be dead and the first \
             would silently win"
        );
        seen.push(path);
    }
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    assert_eq!(
        seen, sorted,
        "keep `LEGACY_BUDGETS` sorted by path so a row is found by reading, and \
         two commits adding a row do not collide on the same line"
    );
}

#[test]
fn the_ratchet_can_still_see_growth() {
    // Positive control: the check is only evidence if it fails on the shape it
    // forbids. Both defect shapes, against a synthetic set.
    let pinned = LEGACY_BUDGETS[0];
    let grown = vec![(pinned.0.to_string(), pinned.1 + 1)];
    assert_eq!(
        over_budget(&grown).len(),
        1,
        "an over-limit file growing by one line must be seen"
    );
    let fresh = vec![("src/brand/new.rs".to_string(), CEILING + 1)];
    assert_eq!(
        over_budget(&fresh).len(),
        1,
        "a NEW file over the ceiling must be seen even though it is unpinned"
    );
    // The shapes the widened scope exists to catch: a shell gate and a test
    // harness crossing the ceiling were both invisible while the scan was
    // `src/`-only, so they are pinned here as cases, not just as scope.
    let harness = vec![
        ("scripts/exec_semantics_gate.sh".to_string(), CEILING + 43),
        ("tests/some_gate.rs".to_string(), CEILING + 1),
        ("examples/mindc_mind/some_smoke.py".to_string(), CEILING + 1),
    ];
    assert_eq!(
        over_budget(&harness).len(),
        3,
        "a shell gate, a test harness and a smoke script over the ceiling must \
         each be seen — the scan being `src/`-only is what let the first one \
         cross it unnoticed"
    );
    // ... and does not cry wolf on the compliant shapes.
    let at_budget = vec![
        (pinned.0.to_string(), pinned.1),
        ("src/brand/new.rs".to_string(), CEILING),
    ];
    assert!(over_budget(&at_budget).is_empty());
    let shrunk = vec![(pinned.0.to_string(), pinned.1 - 1)];
    assert!(
        over_budget(&shrunk).is_empty(),
        "shrinking an over-limit file is the point of the rule and must not \
         turn the gate red"
    );
}
