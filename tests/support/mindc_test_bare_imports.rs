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

#[test]
fn bare_imported_constant_executes() {
    // `mindc check` and native builds resolve a bare reference to an exported
    // constant of an imported sibling; the test evaluator must too (request 9).
    let root = project(
        "eval_import_bare_const",
        r#"
import arithmetic;

#[test]
fn bare_imported_const_executes() {
    assert UNIT == 8, "bare imported constant";
    assert times(UNIT, 5) == 40, "bare call with bare imported constant";
}
"#,
    );
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("1 passed; 0 failed"), "{text}");
    }
}

#[test]
fn bare_constant_keeps_local_definition_first() {
    let root = project(
        "eval_import_bare_const_local_first",
        r#"
import arithmetic;
const UNIT: i64 = 99;

#[test]
fn local_constant_shadows_import() {
    assert UNIT == 99, "lexical const wins over the import";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn bare_constant_exported_by_two_imports_is_refused() {
    // A bare constant exported by two of a module's imports is refused ahead
    // of evaluation: the name is not uniquely visible, so name resolution
    // rejects it as unknown (E2002) before lowering can see it. The evaluator
    // keeps its own ambiguity refusal as defence in depth (see the
    // `module_bindings` unit test), but this is where a real project meets it.
    let root = project(
        "eval_import_bare_const_ambiguous",
        r#"
import mid;

#[test]
fn ambiguous_bare_const_in_dependency_is_refused() {
    assert mid.pick() == 41, "two exporters";
}
"#,
    );
    fs::write(
        root.join("src/mid.mind"),
        "export { pick }\nimport left;\nimport right;\nfn pick() -> i64 { return LIMIT; }\n",
    )
    .expect("write dependency module");
    fs::write(
        root.join("src/left.mind"),
        "export { LIMIT }\nconst LIMIT: i64 = 41;\n",
    )
    .expect("write left module");
    fs::write(
        root.join("src/right.mind"),
        "export { LIMIT }\nconst LIMIT: i64 = 77;\n",
    )
    .expect("write right module");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("unknown identifier `LIMIT`"), "{text}");
    assert!(text.contains("E2002"), "{text}");
}

#[test]
fn unexported_bare_constant_is_refused() {
    let root = project(
        "eval_import_bare_const_private",
        r#"
import arithmetic;

#[test]
fn private_bare_const_is_refused() {
    assert SECRET == 99, "unexported const must not be reachable";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]
#[test]
fn bare_imported_constant_lowers_to_its_value() {
    // Lowering must inline the imported constant's VALUE (never const 0 — the
    // lower.rs fail-closed panic and the silent-miscompile mutant both die
    // here): the MLIR for the importing module carries the real value, and the
    // inline `#[test]` asserts it so a 0-resolving mutant fails the test run.
    let root = project(
        "eval_import_bare_const_lower",
        r#"
import arithmetic;

fn use_unit() -> i64 {
    return UNIT;
}

#[test]
fn bare_imported_const_lowers() {
    assert use_unit() == 8, "imported constant lowers to its value";
}
"#,
    );
    let test_output = Command::new(common::mindc_bin())
        .args(["test", "tests/import_eval.mind"])
        .current_dir(&root)
        .output()
        .expect("run mindc test");
    let test_text = output_text(&test_output);
    assert!(test_output.status.success(), "{test_text}");
    assert!(test_text.contains("1 passed; 0 failed"), "{test_text}");

    let emit_output = Command::new(common::mindc_bin())
        .args(["tests/import_eval.mind", "--emit-mlir"])
        .current_dir(&root)
        .output()
        .expect("run mindc --emit-mlir");
    let emit_text = output_text(&emit_output);
    assert!(emit_output.status.success(), "{emit_text}");
    assert!(
        emit_text.contains("arith.constant 8") || emit_text.contains("const.i64 8"),
        "imported constant must inline its value, never 0:\n{emit_text}"
    );
    assert!(
        !emit_text.contains("refusing to emit const 0"),
        "imported constant must not hit the lowering fail-closed panic:\n{emit_text}"
    );
}

/// Import order between a bundled `std` export and a sibling export of the
/// same name must not decide the winner: the sibling's export binds whether
/// the `std` import is written first or second, and a local definition still
/// wins over both imports. A `std` binding inserted first must be replaced by
/// the sibling's later export, not ambiguified by it.
#[test]
fn std_import_order_does_not_shadow_sibling_export() {
    let sibling = "export { sha256 }\n\
        fn sha256(in_addr: i64, in_len: i64, out_addr: i64) -> i64 { return 42; }\n";
    let imports_std_first = "import std.sha256;\nimport a;\nexport { run }\n";
    let imports_sibling_first = "import a;\nimport std.sha256;\nexport { run }\n";
    let local_sha = "fn sha256(in_addr: i64, in_len: i64, out_addr: i64) -> i64 { return 7; }\n";
    let run_fn = "fn run() -> i64 { return sha256(0, 0, 0); }\n";
    let cases = [
        (
            "eval_order_std_first",
            format!("{imports_std_first}{run_fn}"),
            42i64,
        ),
        (
            "eval_order_sibling_first",
            format!("{imports_sibling_first}{run_fn}"),
            42,
        ),
        (
            "eval_order_local_wins_std_first",
            format!("{imports_std_first}{local_sha}{run_fn}"),
            7,
        ),
        (
            "eval_order_local_wins_sibling_first",
            format!("{imports_sibling_first}{local_sha}{run_fn}"),
            7,
        ),
    ];
    for (name, probe, want) in cases {
        let root = project(
            name,
            &format!(
                "import arithmetic;\n\
                 #[test]\n\
                 fn order_resolves() {{\n\
                 \x20   assert arithmetic.run() == {want}, \"import order must not pick the callee\";\n\
                 }}\n"
            ),
        );
        fs::write(root.join("src/arithmetic.mind"), &probe).expect("write probe module");
        fs::write(root.join("src/a.mind"), sibling).expect("write sibling module");
        let output = run(&root);
        let text = output_text(&output);
        assert!(output.status.success(), "{name}: {text}");
        assert!(text.contains("1 passed; 0 failed"), "{name}: {text}");
    }
}
