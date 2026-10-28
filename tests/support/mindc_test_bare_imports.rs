// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Bare calls into imported sibling modules on the tree-evaluated `mindc test`
//! path: resolution order, export visibility and ambiguity refusal. Included
//! by `tests/mindc_test_imports.rs`, whose project and run helpers it uses.

use super::*;

#[test]
fn bare_call_to_imported_exported_function_executes() {
    // `mindc check` and native builds resolve a bare call to an exported
    // function of an imported sibling; the test evaluator must too.
    let root = project(
        "eval_import_bare_call",
        r#"
import arithmetic;
import auto_export;

#[test]
fn bare_imported_calls_execute() {
    assert times(6, 7) == 42, "bare call through an export block";
    assert la_add(40, 2) == 42, "bare call to an auto-exported pub fn";
    assert la_chain(5) == 30, "bare call from a dependency into its own import";
}
"#,
    );
    fs::write(
        root.join("src/auto_export.mind"),
        "import arithmetic;\n\
         pub fn la_add(a: i64, b: i64) -> i64 { return a + b; }\n\
         pub fn la_chain(x: i64) -> i64 { return times(x, 6); }\n",
    )
    .expect("write auto-export module");
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("1 passed; 0 failed"), "{text}");
    }
}

#[test]
fn bare_call_keeps_local_definition_first() {
    let root = project(
        "eval_import_bare_local_first",
        r#"
import arithmetic;
fn times(a: i64, b: i64) -> i64 { return 7; }

#[test]
fn local_definition_shadows_import() {
    assert times(3, 4) == 7, "lexical function wins over the import";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn bare_call_to_unexported_function_is_refused() {
    let root = project(
        "eval_import_bare_private",
        r#"
import arithmetic;

#[test]
fn private_bare_call_is_refused() {
    assert private_times(3, 4) == 12, "unexported helper must not be reachable";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn bare_call_exported_by_two_imports_is_refused() {
    // The entry file's own ambiguous bare call is already refused by the type
    // checker (E2003). A dependency's is deferred to evaluation, so it must be
    // refused there as ambiguous, never resolved to either exporter.
    let root = project(
        "eval_import_bare_ambiguous",
        r#"
import mid;

#[test]
fn ambiguous_bare_call_in_dependency_is_refused() {
    assert mid.pick() == 41, "two exporters";
}
"#,
    );
    fs::write(
        root.join("src/mid.mind"),
        "export { pick }\nimport left;\nimport right;\nfn pick() -> i64 { return value(); }\n",
    )
    .expect("write dependency module");
    fs::write(
        root.join("src/left.mind"),
        "export { value }\nfn value() -> i64 { return 41; }\n",
    )
    .expect("write left module");
    fs::write(
        root.join("src/right.mind"),
        "export { value }\nfn value() -> i64 { return 77; }\n",
    )
    .expect("write right module");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
    assert!(text.contains("call to `value` is ambiguous"), "{text}");
}
