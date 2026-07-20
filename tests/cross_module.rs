// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Cross-module import resolution — D2 end-to-end (Phase 10.6 item 9 /
//! Phase 15 self-hosting prerequisite).
//!
//! Proves that with the `cross-module-imports` feature a `use crate.a`
//! makes module `crate.a`'s exported symbols resolve in module B's
//! type-checker, and that WITHOUT cross-module wiring the same symbol
//! is "unknown identifier" — i.e. the default path is unchanged.
//!
//! Run: `cargo test --features cross-module-imports --test cross_module`

#![cfg(feature = "cross-module-imports")]

use libmind::parser::parse;
use libmind::project::module_table::build_module_table;
use libmind::type_checker::{
    check_module_types_in_file, check_module_types_with_modules, cm_set_project_table,
};

/// Module B references `helper`, which is only declared+exported by A.
const MODULE_B_SRC: &str = "use crate.a\nlet x = helper\n";

#[test]
fn unresolved_without_module_table() {
    // Baseline: B alone, no cross-module wiring → `helper` is unknown.
    let b = parse(MODULE_B_SRC).expect("parse B");
    let errs = check_module_types_in_file(&b, MODULE_B_SRC, Some("b.mind"), &Default::default());
    assert!(
        errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "expected an 'unknown identifier helper' diagnostic without the module table; got: {errs:?}"
    );
}

#[test]
fn resolved_with_module_table() {
    let a_src = "export { helper }";
    let a = parse(a_src).expect("parse A");
    let b = parse(MODULE_B_SRC).expect("parse B");

    let table = build_module_table(&[("crate.a".to_string(), &a)]);

    let errs = check_module_types_with_modules(
        &b,
        MODULE_B_SRC,
        Some("b.mind"),
        &Default::default(),
        &table,
    );

    assert!(
        !errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "`use crate.a` should resolve `helper` exported by A; got: {errs:?}"
    );
}

#[test]
fn wrong_module_path_does_not_resolve() {
    let a_src = "export { helper }";
    let a = parse(a_src).expect("parse A");
    let b = parse(MODULE_B_SRC).expect("parse B"); // imports `crate.a`

    // Table registers the symbol under a DIFFERENT path.
    let table = build_module_table(&[("crate.other".to_string(), &a)]);

    let errs = check_module_types_with_modules(
        &b,
        MODULE_B_SRC,
        Some("b.mind"),
        &Default::default(),
        &table,
    );
    assert!(
        errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "import path `crate.a` must not resolve a symbol registered under `crate.other`"
    );
}

#[test]
fn with_modules_restores_prior_project_table_not_none() {
    // Regression for the CM_TABLE panic-unsafety/clobber fix: a project-scope
    // table installed via `cm_set_project_table` (the whole-project pattern
    // documented on that function) must survive a `check_module_types_with_modules`
    // call that installs its OWN table for the duration — the guard restores
    // whatever was there before, not an unconditional `None`.
    let a = parse("export { helper }").expect("parse A");
    let project_table = build_module_table(&[("crate.a".to_string(), &a)]);
    cm_set_project_table(Some(project_table));

    let b = parse(MODULE_B_SRC).expect("parse B");
    // A different, unrelated per-file table (registers nothing useful) —
    // `helper` must NOT resolve during this call.
    let unrelated = build_module_table(&[]);
    let during = check_module_types_with_modules(
        &b,
        MODULE_B_SRC,
        Some("b.mind"),
        &Default::default(),
        &unrelated,
    );
    assert!(
        during.iter().any(|e| format!("{e:?}").contains("helper")),
        "the per-file table passed to check_module_types_with_modules must be in \
         effect DURING the call, not the pre-set project table"
    );

    // After the call, the project table set by `cm_set_project_table` must be
    // back in effect (restored, not cleared to `None`).
    let after = check_module_types_in_file(&b, MODULE_B_SRC, Some("b.mind"), &Default::default());
    assert!(
        !after.iter().any(|e| format!("{e:?}").contains("helper")),
        "the project-scope table must be RESTORED after check_module_types_with_modules \
         returns, not clobbered to None; got: {after:?}"
    );

    cm_set_project_table(None); // clean up the thread-local for other tests
}

#[test]
fn thread_local_table_does_not_leak_between_calls() {
    let a = parse("export { helper }").expect("parse A");
    let b = parse(MODULE_B_SRC).expect("parse B");
    let table = build_module_table(&[("crate.a".to_string(), &a)]);

    // First call uses the table.
    let _ = check_module_types_with_modules(
        &b,
        MODULE_B_SRC,
        Some("b.mind"),
        &Default::default(),
        &table,
    );
    // Second call WITHOUT the table must behave like the baseline
    // (table cleared after the prior call — no leakage).
    let errs = check_module_types_in_file(&b, MODULE_B_SRC, Some("b.mind"), &Default::default());
    assert!(
        errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "module table must not leak into a subsequent plain check call"
    );
}
