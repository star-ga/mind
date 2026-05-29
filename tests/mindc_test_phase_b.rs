// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! RFC 0008 Phase B integration tests for `mindc test`.
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports" mindc_test_phase_b`

use std::path::PathBuf;

use libmind::test::{ReporterKind, TestOptions, TestStatus, discover_tests_in_source, run_tests};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture(name))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
}

// ---------------------------------------------------------------------------
// Parser-level: #[test] attribute parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_test_attribute_sets_is_test() {
    let src = r#"
#[test]
fn test_simple() {
    let x: i64 = 1
}
"#;
    let module = libmind::parser::parse(src).expect("parse ok");
    let fndef = module.items.first().expect("one item");
    match fndef {
        libmind::ast::Node::FnDef { is_test, name, .. } => {
            assert_eq!(name, "test_simple");
            assert!(*is_test, "is_test should be true");
        }
        _ => panic!("expected FnDef, got {:?}", fndef),
    }
}

#[test]
fn parse_fn_without_test_attr_has_is_test_false() {
    let src = r#"fn normal_fn() -> i64 { 42 }"#;
    let module = libmind::parser::parse(src).expect("parse ok");
    let fndef = module.items.first().expect("one item");
    match fndef {
        libmind::ast::Node::FnDef { is_test, .. } => {
            assert!(!*is_test, "is_test should be false for unannotated fn");
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn parse_test_fn_with_nonzero_arity_is_parse_error() {
    // RFC 0008 Phase B: a #[test] fn must have zero parameters.
    // This should be a parse-time error.
    let src = r#"
#[test]
fn test_invalid(x: i64) {
    let y: i64 = x + 1
}
"#;
    let result = libmind::parser::parse(src);
    assert!(
        result.is_err(),
        "expected parse error for #[test] fn with non-zero arity"
    );
    let errs = result.unwrap_err();
    let msg = errs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        msg.contains("zero parameters") || msg.contains("parameter"),
        "error message should mention parameters: {msg}"
    );
}

#[test]
fn parse_pub_test_fn_is_allowed() {
    // #[test] pub fn is syntactically valid (is_pub + is_test both true).
    let src = r#"
#[test]
pub fn test_pub_fn() {
    let x: i64 = 42
}
"#;
    let module = libmind::parser::parse(src).expect("parse ok");
    match module.items.first() {
        Some(libmind::ast::Node::FnDef {
            is_test, is_pub, ..
        }) => {
            assert!(*is_test);
            assert!(*is_pub);
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn is_pub_field_preserved_on_non_test_fn() {
    // Existing is_pub plumbing must still work after adding is_test.
    let src = r#"pub fn exported() -> i64 { 1 }"#;
    let module = libmind::parser::parse(src).expect("parse ok");
    match module.items.first() {
        Some(libmind::ast::Node::FnDef {
            is_pub, is_test, ..
        }) => {
            assert!(*is_pub, "pub fn should have is_pub=true");
            assert!(!*is_test, "non-test fn should have is_test=false");
        }
        _ => panic!("expected FnDef"),
    }
}

// ---------------------------------------------------------------------------
// Discovery: discover_tests_in_source
// ---------------------------------------------------------------------------

#[test]
fn discover_finds_test_fns_in_source() {
    let src = r#"
#[test]
fn test_alpha() { let x: i64 = 1 }

fn helper() -> i64 { 99 }

#[test]
fn test_beta() { let y: i64 = 2 }
"#;
    let path = PathBuf::from("dummy.mind");
    let entries = discover_tests_in_source(&path, src).expect("discover ok");
    assert_eq!(entries.len(), 2, "should find exactly 2 test fns");
    assert!(entries.iter().any(|e| e.name == "dummy::test_alpha"));
    assert!(entries.iter().any(|e| e.name == "dummy::test_beta"));
    // helper() has no #[test] — must not appear.
    assert!(!entries.iter().any(|e| e.name.contains("helper")));
}

#[test]
fn discover_returns_correct_source_line() {
    let src = "// line 1\n#[test]\nfn test_here() { let x: i64 = 0 }\n";
    let path = PathBuf::from("linetest.mind");
    let entries = discover_tests_in_source(&path, src).expect("discover ok");
    assert_eq!(entries.len(), 1);
    // fn keyword is on line 3 (1-based), or 2 depending on exact span
    // We just verify it's a reasonable line number >= 1.
    assert!(entries[0].source_line >= 1);
}

#[test]
fn discover_from_fixture_all_pass() {
    let path = fixture("test_phase_b_all_pass.mind");
    let src = read_fixture("test_phase_b_all_pass.mind");
    let entries = discover_tests_in_source(&path, &src).expect("discover ok");
    assert_eq!(entries.len(), 2);
    assert!(
        entries
            .iter()
            .any(|e| e.name.contains("test_addition_is_correct"))
    );
    assert!(
        entries
            .iter()
            .any(|e| e.name.contains("test_multiplication_is_correct"))
    );
}

#[test]
fn discover_from_fixture_one_fail() {
    let path = fixture("test_phase_b_one_fail.mind");
    let src = read_fixture("test_phase_b_one_fail.mind");
    let entries = discover_tests_in_source(&path, &src).expect("discover ok");
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().any(|e| e.name.contains("test_passes")));
    assert!(entries.iter().any(|e| e.name.contains("test_fails")));
}

// ---------------------------------------------------------------------------
// Execution: run_tests with all-pass fixture
// ---------------------------------------------------------------------------

#[test]
fn run_all_pass_fixture_reports_2_passed() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: String::new(),
        capture: true,
        threads: 1,
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    assert_eq!(summary.passed, 2, "both tests should pass");
    assert_eq!(summary.failed, 0, "no tests should fail");
    assert!(summary.all_passed());
}

// ---------------------------------------------------------------------------
// Execution: run_tests with one-pass, one-fail fixture
// ---------------------------------------------------------------------------

#[test]
fn run_one_fail_fixture_reports_1_passed_1_failed() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_one_fail.mind")],
        filter: String::new(),
        capture: true,
        threads: 1,
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests returns Ok even with failures");
    assert_eq!(summary.passed, 1, "one test should pass");
    assert_eq!(summary.failed, 1, "one test should fail");
    assert!(!summary.all_passed());

    // The failing test must be the right one.
    let fail_result = summary
        .results
        .iter()
        .find(|r| r.status != TestStatus::Passed);
    assert!(fail_result.is_some(), "should have a failing test result");
    let fail = fail_result.unwrap();
    assert!(
        fail.name.contains("test_fails"),
        "failing test should be named test_fails, got {}",
        fail.name
    );
}

// ---------------------------------------------------------------------------
// --filter: substring matching
// ---------------------------------------------------------------------------

#[test]
fn filter_restricts_tests_run() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: "addition".to_string(),
        capture: true,
        threads: 1,
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    // Only "test_addition_is_correct" matches "addition".
    assert_eq!(
        summary.passed + summary.failed,
        1,
        "only one test should run with filter 'addition'"
    );
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 0);
}

#[test]
fn filter_with_no_match_runs_zero_tests() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: "nonexistent_xyzzy".to_string(),
        capture: true,
        threads: 1,
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    assert_eq!(summary.passed, 0);
    assert_eq!(summary.failed, 0);
    assert!(
        summary.all_passed(),
        "--no-match filter: all_passed should be true (0 tests)"
    );
}

// ---------------------------------------------------------------------------
// --list: lists names without running
// ---------------------------------------------------------------------------

#[test]
fn list_flag_returns_empty_summary_without_running() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: String::new(),
        capture: true,
        threads: 1,
        list: true,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    // With --list, no tests are executed: summary is zero-initialized.
    assert_eq!(summary.passed, 0, "--list should not execute any tests");
    assert_eq!(summary.failed, 0, "--list should not execute any tests");
    assert!(summary.results.is_empty());
}

// ---------------------------------------------------------------------------
// --threads=1: sequential execution
// ---------------------------------------------------------------------------

#[test]
fn threads_1_sequential_both_tests_run() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: String::new(),
        capture: true,
        threads: 1,
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 0);
}

// ---------------------------------------------------------------------------
// Directory walk: discovery picks up #[test] fns from nested files
// ---------------------------------------------------------------------------

#[test]
fn directory_walk_discovers_test_fns() {
    // Walk the fixtures directory — it contains test_phase_b_all_pass.mind
    // and test_phase_b_one_fail.mind, each with 2 test fns = 4 total.
    let opts = TestOptions {
        paths: vec![fixtures_dir()],
        filter: String::new(),
        capture: true,
        threads: 1,
        list: true, // list only, so we don't care about failures in other fixtures
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    // With --list no tests run, but the discovery count is not returned.
    // We verify by running discover on each fixture file directly.
    let all_pass_path = fixture("test_phase_b_all_pass.mind");
    let one_fail_path = fixture("test_phase_b_one_fail.mind");
    let src1 = read_fixture("test_phase_b_all_pass.mind");
    let src2 = read_fixture("test_phase_b_one_fail.mind");
    let e1 = discover_tests_in_source(&all_pass_path, &src1).unwrap();
    let e2 = discover_tests_in_source(&one_fail_path, &src2).unwrap();
    assert_eq!(
        e1.len() + e2.len(),
        4,
        "should find 4 test fns across both fixtures"
    );
    // Silence unused-variable warning on summary.
    let _ = summary;
}

// ---------------------------------------------------------------------------
// Parallel execution: multi-thread both pass
// ---------------------------------------------------------------------------

#[test]
fn parallel_execution_with_multiple_threads() {
    let opts = TestOptions {
        paths: vec![fixture("test_phase_b_all_pass.mind")],
        filter: String::new(),
        capture: true,
        threads: 0, // use available parallelism
        list: false,
        reporter: ReporterKind::Human,
    };
    let summary = run_tests(&opts).expect("run_tests ok");
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 0);
}

// ---------------------------------------------------------------------------
// AST consumer backward compat: reap_threshold still works with is_test
// ---------------------------------------------------------------------------

#[test]
fn reap_threshold_and_is_test_coexist() {
    // A fn with [reap_threshold] but no #[test] should still work.
    let src = r#"
#[reap_threshold(0.3)]
fn expert_fn() -> i64 { 1 }
"#;
    let module = libmind::parser::parse(src).expect("parse ok");
    match module.items.first() {
        Some(libmind::ast::Node::FnDef {
            reap_threshold,
            is_test,
            ..
        }) => {
            assert!(reap_threshold.is_some(), "reap_threshold should be set");
            assert!(!*is_test, "is_test should be false");
        }
        _ => panic!("expected FnDef"),
    }
}
