// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mindc test` must report the ROOT CAUSE, not the downstream symptom (#242).
//!
//! The harness evaluates the test body once, then walks it a SECOND time
//! evaluating asserts against the env the first pass populated. When the first
//! pass aborts on an unresolved reference inside a CALLEE, every later binding
//! is missing, and the second pass blamed a correctly-bound name in the caller
//! (`unknown variable: rc`) while the real error — the undefined symbol — was
//! discarded unread. That sent a downstream debugging session chasing test bugs
//! that did not exist.
//!
//! Gate: `cargo test --test test_harness_root_cause_error`

#![cfg(unix)]

mod common;
use common::mindc_bin;

use std::process::Command;

const POISONED: &str = r#"
fn called_with_undefined_ref(out: i64) -> i64 {
    __mind_store_i8(out, 7)
    return UNDEFINED_CONST_XYZ
}

#[test]
fn test_caller_env_poisoned() {
    let out: i64 = __mind_alloc(8)
    let rc: i64 = called_with_undefined_ref(out)
    assert rc == 7, "rc is bound; the real error is the undefined const"
    __mind_free(out)
}
"#;

// A module whose first pass SUCCEEDS: a genuinely failing assert must still
// report its own message verbatim, unchanged by the root-cause plumbing.
const HEALTHY: &str = r#"
#[test]
fn test_real_failure() {
    let a: i64 = 2
    assert a == 3, "a should be 3"
}
"#;

fn run_mindc_test(name: &str, src: &str) -> String {
    let mindc = mindc_bin();
    if !mindc.exists() {
        return String::new();
    }
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, src).expect("write src");
    let out = Command::new(&mindc)
        .args(["test", path.to_str().unwrap()])
        .output()
        .expect("run mindc test");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn unresolved_callee_ref_names_the_undefined_symbol() {
    let combined = run_mindc_test("mind_r10_poisoned.mind", POISONED);
    if combined.is_empty() {
        println!("test-harness-root-cause: mindc not found; skipping");
        return;
    }
    assert!(
        combined.contains("UNDEFINED_CONST_XYZ"),
        "the failure must name the undefined symbol that actually aborted the \
         body, not only the downstream symptom; got:\n{combined}"
    );
}

#[test]
fn a_genuine_assert_failure_is_unchanged() {
    let combined = run_mindc_test("mind_r10_healthy.mind", HEALTHY);
    if combined.is_empty() {
        println!("test-harness-root-cause: mindc not found; skipping");
        return;
    }
    assert!(
        combined.contains("a should be 3"),
        "a real assertion failure must still report its own message:\n{combined}"
    );
    assert!(
        !combined.contains("the test body aborted at the error above"),
        "the root-cause note must NOT be attached when the first pass \
         succeeded:\n{combined}"
    );
}
