// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! End-to-end import resolution for the tree-evaluated `mindc test` path.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MODULE: &str = r#"
export { UNIT, times }
const UNIT: i64 = 8;
const SECRET: i64 = 99;
fn times(a: i64, b: i64) -> i64 { return private_times(a, b); }
fn private_times(a: i64, b: i64) -> i64 { return a * b; }
"#;

fn project(name: &str, test_source: &str) -> PathBuf {
    let root = common::scratch_dir("mindc-test-imports").join(name);
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::create_dir_all(root.join("src")).expect("create source dir");
    fs::create_dir_all(root.join("tests")).expect("create tests dir");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
             [build]\nentry = \"main.mind\"\n"
        ),
    )
    .expect("write manifest");
    fs::write(root.join("main.mind"), "fn main() -> i32 { return 0; }\n").expect("write entry");
    fs::write(root.join("src/arithmetic.mind"), MODULE).expect("write imported module");
    fs::write(root.join("tests/import_eval.mind"), test_source).expect("write test source");
    root
}

fn run(root: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(["test", "tests/import_eval.mind"])
        .current_dir(root)
        .output()
        .expect("run mindc test")
}

fn explicit_source_project(name: &str, test_source: &str) -> PathBuf {
    let root = common::scratch_dir("mindc-test-imports-explicit").join(name);
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::write(
        root.join("Mind.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n\
             [build]\nentry = \"consumer.mind\"\n\n\
             [targets.cpu]\nbackend = \"cpu\"\nsources = [\"provider.mind\", \"consumer.mind\"]\n"
        ),
    )
    .expect("write explicit source manifest");
    fs::write(
        root.join("provider.mind"),
        "export { VALUE, double }\nconst VALUE: i64 = 11;\nfn double(x: i64) -> i64 { return x + x; }\nfn hidden(x: i64) -> i64 { return x + x; }\n",
    )
    .expect("write explicit imported module");
    fs::write(root.join("consumer.mind"), test_source).expect("write explicit test source");
    root
}

fn run_explicit(root: &Path) -> Output {
    Command::new(common::mindc_bin())
        .args(["test", "consumer.mind"])
        .current_dir(root)
        .output()
        .expect("run explicit-source mindc test")
}

fn run_with_threads(root: &Path, threads: usize) -> Output {
    Command::new(common::mindc_bin())
        .args([
            "test",
            "tests/import_eval.mind",
            "--threads",
            &threads.to_string(),
        ])
        .current_dir(root)
        .output()
        .expect("run mindc test")
}

fn output_text(output: &Output) -> String {
    format!(
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn imported_const_and_function_execute_from_manifest_scope() {
    let root = project(
        "eval_import_ok",
        r#"
import arithmetic;

#[test]
fn imported_const_executes() {
    assert arithmetic.UNIT == 8, "imported const";
}

#[test]
fn imported_function_executes() {
    assert arithmetic.times(arithmetic.UNIT, 5) == 40, "imported function";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("running 2 tests"), "{text}");
    assert!(text.contains("2 passed; 0 failed"), "{text}");
}

#[test]
fn path_syntax_imported_call_and_value_execute_from_explicit_file_entry() {
    let root = explicit_source_project(
        "eval_explicit_path_syntax",
        r#"
import provider;

#[test]
fn imported_path_call_executes() {
    assert provider::double(7) == 14, "path imported function";
    assert provider::VALUE == 11, "path imported constant";
}
"#,
    );
    let output = run_explicit(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("running 1 test"), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn nested_path_alias_and_local_enum_remain_distinct() {
    let root = common::scratch_dir("mindc-test-imports-nested-path")
        .join(format!("nested_path_{}", std::process::id()));
    fs::create_dir_all(root.join(".git")).expect("create project boundary");
    fs::create_dir_all(root.join("src")).expect("create nested source directory");
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"nested_path\"\nversion = \"0.1.0\"\n\n\
         [build]\nentry = \"consumer.mind\"\n\n\
         [targets.cpu]\nbackend = \"cpu\"\nsources = [\"src/provider.mind\", \"consumer.mind\"]\n",
    )
    .expect("write nested path manifest");
    fs::write(
        root.join("src/provider.mind"),
        "export { double }\nfn double(x: i64) -> i64 { return x + x; }\n",
    )
    .expect("write nested provider");
    fs::write(
        root.join("consumer.mind"),
        r#"
use crate.src.provider;
enum Side { Left(i64), Right }
fn local_constructor() -> i64 {
    let side = Side::Left(3);
    match side {
        Side::Left(value) => value,
        Side::Right => 0,
    }
}

#[test]
fn nested_path_call_executes() {
    assert crate::src::provider::double(8) == 16, "nested path imported function";
}
"#,
    )
    .expect("write nested path test source");
    let output = Command::new(common::mindc_bin())
        .args(["test", "consumer.mind"])
        .current_dir(&root)
        .output()
        .expect("run nested path mindc test");
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn path_syntax_private_and_wrong_arity_fail_closed() {
    let private_root = explicit_source_project(
        "eval_explicit_path_private",
        r#"
import provider;

#[test]
fn private_path_symbol_is_refused() {
    assert provider::hidden(7) == 14, "must not execute";
}
"#,
    );
    let private_output = run_explicit(&private_root);
    let private_text = output_text(&private_output);
    assert!(!private_output.status.success(), "{private_text}");
    assert!(private_text.contains("E2003"), "{private_text}");

    let arity_root = explicit_source_project(
        "eval_explicit_path_arity",
        r#"
import provider;

#[test]
fn path_wrong_arity_is_refused() {
    assert provider::double() == 0, "must not execute";
}
"#,
    );
    let arity_output = run_explicit(&arity_root);
    let arity_text = output_text(&arity_output);
    assert!(!arity_output.status.success(), "{arity_text}");
    assert!(
        arity_text.contains("arity") || arity_text.contains("expects 1"),
        "{arity_text}"
    );
}

#[test]
fn standalone_path_import_still_requires_a_manifest() {
    let root = common::scratch_dir("mindc-test-imports-standalone-path")
        .join(format!("standalone_path_{}", std::process::id()));
    fs::create_dir_all(&root).expect("create standalone project");
    fs::write(
        root.join("provider.mind"),
        "export { double }\nfn double(x: i64) -> i64 { return x + x; }\n",
    )
    .expect("write standalone provider");
    fs::write(
        root.join("consumer.mind"),
        r#"
import provider;

#[test]
fn standalone_path_import_is_refused() {
    assert provider::double(7) == 14, "must not execute";
}
"#,
    )
    .expect("write standalone path test");
    let output = Command::new(common::mindc_bin())
        .args(["test", "consumer.mind"])
        .current_dir(&root)
        .output()
        .expect("run standalone path mindc test");
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(
        text.contains("E2003") && text.contains("Mind.toml"),
        "{text}"
    );
}

#[test]
fn nested_archive_entry_uses_invocation_manifest_for_import_execution() {
    let root = common::scratch_dir("mindc-test-imports")
        .join(format!("archive_root_no_git_{}", std::process::id()));
    fs::create_dir_all(root.join("src")).expect("create source dir");
    fs::create_dir_all(root.join("docs/mindc-repros")).expect("create nested repro dir");
    assert!(
        !root.join(".git").exists(),
        "archive fixture must have no git marker"
    );
    fs::write(
        root.join("Mind.toml"),
        "[package]\nname = \"archive_imports\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\n",
    )
    .expect("write archive manifest");
    fs::write(root.join("main.mind"), "fn main() -> i32 { return 0; }\n")
        .expect("write archive entry");
    fs::write(root.join("src/arithmetic.mind"), MODULE).expect("write imported module");
    fs::write(
        root.join("docs/mindc-repros/nested_imports.mind"),
        r#"
import arithmetic;

#[test]
fn imported_const_executes() {
    assert arithmetic.UNIT == 8, "imported const";
}

#[test]
fn imported_function_executes() {
    assert arithmetic.times(arithmetic.UNIT, 5) == 40, "imported function";
}
"#,
    )
    .expect("write nested archive test");

    let output = Command::new(common::mindc_bin())
        .args(["test", "docs/mindc-repros/nested_imports.mind"])
        .current_dir(&root)
        .output()
        .expect("run nested archive import test");
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("running 2 tests"), "{text}");
    assert!(text.contains("2 passed; 0 failed"), "{text}");
}

#[test]
fn unexported_imported_const_is_refused() {
    let root = project(
        "eval_import_private_const",
        r#"
import arithmetic;
#[test]
fn private_const_is_refused() {
    assert arithmetic.SECRET == 99, "must not execute";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("E2002"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn unexported_imported_function_is_refused() {
    let root = project(
        "eval_import_private_fn",
        r#"
import arithmetic;
#[test]
fn private_function_is_refused() {
    assert arithmetic.private_times(3, 4) == 12, "must not execute";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("E2003"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn missing_imported_function_is_refused() {
    let root = project(
        "eval_import_missing",
        r#"
import absent_module;
#[test]
fn missing_function_is_refused() {
    assert absent_module.no_such_function(1) == 1, "must not execute";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("E2003"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn declared_but_unimported_module_is_refused() {
    let root = project(
        "eval_import_unimported",
        r#"
#[test]
fn unimported_function_is_refused() {
    assert arithmetic.times(3, 4) == 12, "must not execute";
}
"#,
    );
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("unknown variable: arithmetic"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn malformed_imported_module_is_refused() {
    let root = project(
        "eval_import_malformed",
        r#"
import arithmetic;
#[test]
fn malformed_dependency_is_refused() {
    assert arithmetic.times(3, 4) == 12, "must not execute";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "fn times( -> i64 { return 0; }\n",
    )
    .expect("write malformed imported module");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(
        text.contains("imported sibling") && text.contains("does not parse"),
        "{text}"
    );
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn lexical_module_ownership_survives_function_and_const_collisions() {
    let root = project(
        "eval_import_lexical_owner",
        r#"
import arithmetic;
const UNIT: i64 = 99;
fn private_times(a: i64, b: i64) -> i64 { return 99; }

#[test]
fn dependency_keeps_its_private_bindings() {
    assert arithmetic.times(3, 4) == 12, "dependency helper owner";
    assert arithmetic.UNIT == 8, "dependency const owner";
    assert UNIT == 99, "consumer const owner";
    assert private_times(3, 4) == 99, "consumer helper owner";
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
fn qualified_aliases_select_same_named_exports_deterministically() {
    let root = project(
        "eval_import_same_exports",
        r#"
use crate.src.left;
use crate.src.right;

#[test]
fn left_alias_keeps_owner() {
    assert left.value() == 41, "left exported function";
    assert left.VALUE == 41, "left exported const";
}

#[test]
fn right_alias_keeps_owner() {
    assert right.value() == 77, "right exported function";
    assert right.VALUE == 77, "right exported const";
}
"#,
    );
    fs::write(
        root.join("src/left.mind"),
        "export { VALUE, value }\nconst VALUE: i64 = 41;\nfn helper() -> i64 { return VALUE; }\nfn value() -> i64 { return helper(); }\n",
    )
    .expect("write left module");
    fs::write(
        root.join("src/right.mind"),
        "export { VALUE, value }\nconst VALUE: i64 = 77;\nfn helper() -> i64 { return VALUE; }\nfn value() -> i64 { return helper(); }\n",
    )
    .expect("write right module");
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("2 passed; 0 failed"), "{text}");
    }
}

#[test]
fn direct_qualifiers_select_same_named_exports() {
    let root = project(
        "eval_import_direct_qualifiers",
        r#"
import left;
import right;
let value: i64 = 0;

#[test]
fn direct_qualifiers_keep_owners() {
    assert left.value() == 41, "left direct qualifier";
    assert right.value() == 77, "right direct qualifier";
}
"#,
    );
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
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn transitive_dependency_keeps_private_helper_and_local_shadows() {
    let root = project(
        "eval_import_transitive_owner",
        r#"
import arithmetic;
#[test]
fn transitive_helper_executes() {
    assert arithmetic.times(3, 4) == 14, "transitive private helper and const";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "import helper;\nexport { times }\nconst FACTOR: i64 = helper.SCALE;\nfn times(a: i64, b: i64) -> i64 { return helper.times(a, b) + FACTOR; }\n",
    )
    .expect("write importing module");
    fs::write(
        root.join("src/helper.mind"),
        "export { SCALE, times }\nconst SCALE: i64 = 2;\nfn local_mul(a: i64, b: i64) -> i64 { let SCALE: i64 = b; return a * SCALE; }\nfn times(a: i64, b: i64) -> i64 { return local_mul(a, b); }\n",
    )
    .expect("write private helper module");
    let output = run(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn qualified_values_and_callee_constants_ignore_caller_locals() {
    let root = project(
        "eval_import_caller_local_collision",
        r#"
import arithmetic;
import right;
let BASE: i64 = 90;
fn local_value() -> i64 { return BASE; }
fn local_caller() -> i64 { let BASE: i64 = 100; return local_value(); }
#[test]
fn caller_locals_do_not_override_qualified_or_callee_values() {
    let UNIT: f64 = 99.0;
    let BASE: i64 = 5;
    assert arithmetic.UNIT == 8, "qualified value owner";
    assert arithmetic.unit() == 8, "callee lexical owner";
    assert arithmetic.global_value() == 41, "dependency module globals";
    assert arithmetic.param(12) == 12, "parameter shadows module global";
    assert right.global_value() == 77, "sibling module globals";
    assert local_value() == 90, "entry module global";
    assert local_caller() == 90, "caller local isolation";
    assert UNIT == 99.0, "ordinary local shadow";
    assert BASE == 5, "module global can be shadowed locally";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "export { UNIT, unit, global_value, param }\nconst UNIT: i64 = 8;\nlet FIRST: i64 = 40;\nlet BASE: i64 = FIRST + 1;\nfn unit() -> i64 { return UNIT; }\nfn global_value() -> i64 { return BASE; }\nfn param(BASE: i64) -> i64 { return BASE; }\n",
    )
    .expect("write collision dependency");
    fs::write(
        root.join("src/right.mind"),
        "export { global_value }\nlet BASE: i64 = 77;\nfn global_value() -> i64 { return BASE; }\n",
    )
    .expect("write sibling globals");
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("1 passed; 0 failed"), "{text}");
    }
}

#[test]
fn transitive_private_function_and_constant_are_refused() {
    let root = project(
        "eval_import_transitive_private",
        r#"
import arithmetic;
fn hidden_fn() -> i64 { return 99; }
const SECRET: i64 = 99;
#[test]
fn private_transitive_symbols_do_not_capture_consumer_names() {
    assert arithmetic.value() == 77, "must not execute";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "import hidden;\nexport { value }\nfn value() -> i64 { return hidden.hidden_fn() + hidden.SECRET; }\n",
    )
    .expect("write transitive importer");
    fs::write(
        root.join("src/hidden.mind"),
        "export { visible }\nconst SECRET: i64 = 36;\nfn hidden_fn() -> i64 { return 41; }\nfn visible() -> i64 { return 0; }\n",
    )
    .expect("write private transitive dependency");
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(!output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("E2002") || text.contains("E2003"), "{text}");
        assert!(text.contains("0 passed; 1 failed"), "{text}");
    }
}

#[test]
fn cyclic_dependency_constants_fail_closed() {
    let root = project(
        "eval_import_const_cycle",
        r#"
import arithmetic;
#[test]
fn cycle_is_not_a_value() {
    assert arithmetic.UNIT == 1, "must not execute";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "export { UNIT }\nconst UNIT: i64 = NEXT;\nconst NEXT: i64 = UNIT;\n",
    )
    .expect("write cyclic constants");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("not const-evaluable"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn missing_transitive_helper_does_not_capture_consumer_function() {
    let root = project(
        "eval_import_missing_private",
        r#"
import arithmetic;
fn absent_private() -> i64 { return 99; }
#[test]
fn missing_private_is_refused() {
    assert arithmetic.times(3, 4) == 12, "must not capture consumer";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "export { times }\nfn times(a: i64, b: i64) -> i64 { return absent_private(); }\n",
    )
    .expect("write dependency with missing helper");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("unsupported operation"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn manifest_resolved_qualified_type_does_not_block_evaluation() {
    let root = project(
        "eval_import_qualified_type",
        r#"
import arithmetic;
fn read_item(item: arithmetic.Item) -> i64 { return item.value; }
#[test]
fn qualified_exported_type_is_prepared() {
    assert arithmetic.answer() == 41, "qualified type setup";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "export { Item, answer }\nstruct Item { value: i64 }\nfn answer() -> i64 { return 41; }\n",
    )
    .expect("write typed dependency");
    let output = run(&root);
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    assert!(text.contains("1 passed; 0 failed"), "{text}");
}

#[test]
fn manifest_resolved_private_qualified_type_is_refused() {
    let root = project(
        "eval_import_private_qualified_type",
        r#"
import arithmetic;
fn read_item(item: arithmetic.Item) -> i64 { return item.value; }
#[test]
fn private_type_does_not_run() {
    assert arithmetic.answer() == 41, "must not execute";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        "export { answer }\nstruct Item { value: i64 }\nfn answer() -> i64 { return 41; }\n",
    )
    .expect("write private typed dependency");
    let output = run(&root);
    let text = output_text(&output);
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("E2002"), "{text}");
    assert!(text.contains("0 passed; 1 failed"), "{text}");
}

#[test]
fn imported_module_standard_library_dependency_executes() {
    let root = project(
        "eval_import_transitive_std",
        r#"
import arithmetic;
#[test]
fn dependency_standard_library_call_runs() {
    assert arithmetic.empty_digest_matches() == 1, "transitive std import";
}
"#,
    );
    fs::write(
        root.join("src/arithmetic.mind"),
        r#"
import std.sha256;
export { empty_digest_matches }
fn empty_digest_matches() -> i64 {
    let input: i64 = __mind_alloc(1);
    let output: i64 = __mind_alloc(32);
    let status: i64 = sha256.sha256(input, 0, output);
    if status != 0 { return -1; }
    let expected: [i64; 32] = [227, 176, 196, 66, 152, 252, 28, 20, 154, 251, 244, 200, 153, 111, 185, 36, 39, 174, 65, 228, 100, 155, 147, 76, 164, 149, 153, 27, 120, 82, 184, 85];
    let mut i: i64 = 0;
    while i < 32 {
        if __mind_load_i8(output + i) != expected[i] { return 0; }
        i = i + 1;
    }
    return 1;
}
"#,
    )
    .expect("write dependency with std import");
    for threads in [1, 4] {
        let output = run_with_threads(&root, threads);
        let text = output_text(&output);
        assert!(output.status.success(), "threads={threads}\n{text}");
        assert!(text.contains("1 passed; 0 failed"), "{text}");
    }
}
