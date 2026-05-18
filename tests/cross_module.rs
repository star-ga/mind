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
use libmind::type_checker::{check_module_types_in_file, check_module_types_with_modules};
use std::collections::HashMap;

/// Module B references `helper`, which is only declared+exported by A.
const MODULE_B_SRC: &str = "use crate.a\nlet x = helper\n";

#[test]
fn unresolved_without_module_table() {
    // Baseline: B alone, no cross-module wiring → `helper` is unknown.
    let b = parse(MODULE_B_SRC).expect("parse B");
    let errs = check_module_types_in_file(&b, MODULE_B_SRC, Some("b.mind"), &HashMap::new());
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

    let errs =
        check_module_types_with_modules(&b, MODULE_B_SRC, Some("b.mind"), &HashMap::new(), &table);

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

    let errs =
        check_module_types_with_modules(&b, MODULE_B_SRC, Some("b.mind"), &HashMap::new(), &table);
    assert!(
        errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "import path `crate.a` must not resolve a symbol registered under `crate.other`"
    );
}

#[test]
fn thread_local_table_does_not_leak_between_calls() {
    let a = parse("export { helper }").expect("parse A");
    let b = parse(MODULE_B_SRC).expect("parse B");
    let table = build_module_table(&[("crate.a".to_string(), &a)]);

    // First call uses the table.
    let _ =
        check_module_types_with_modules(&b, MODULE_B_SRC, Some("b.mind"), &HashMap::new(), &table);
    // Second call WITHOUT the table must behave like the baseline
    // (table cleared after the prior call — no leakage).
    let errs = check_module_types_in_file(&b, MODULE_B_SRC, Some("b.mind"), &HashMap::new());
    assert!(
        errs.iter().any(|e| format!("{e:?}").contains("helper")),
        "module table must not leak into a subsequent plain check call"
    );
}
