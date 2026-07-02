// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Function-value-call fail-loud regression gate (E2012).
//!
//! Confirmed accept-what-cannot-be-emitted bug: calling a function-valued local
//! binding (`let f = add1  return f(41)`) passed `mindc check` and `--emit-shared`
//! WROTE an .so, but the callee was synthesised as a call to an undefined
//! `func.func private @f`, so the .so carried an UNDEFINED symbol (`nm -D` shows
//! `U f`) and dlopen/call failed at link/load time. First-class functions do not
//! exist yet, so the front-end must REJECT the value-call at compile (E2012)
//! rather than emit a dangling extern.
//!
//! This gate asserts the fn-value call shape is REJECTED with E2012 and that a
//! normal direct call to the SAME module fn (`add1(41)`) does NOT trip E2012 —
//! the reject fires only on the local-binding-only callee, never on a real
//! module fn call.
//!
//! We assert on the presence/absence of the diagnostic CODE in the combined
//! stdout+stderr rather than the process exit status: `mindc check` also emits a
//! `fmt::drift` diagnostic for an unformatted file, so whether E2012 fires is the
//! behavior under test (mirrors tests/typo_reject.rs).

use std::io::Write;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn write_tmp(name: &str, src: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&p).expect("create tmp");
    f.write_all(src.as_bytes()).expect("write tmp");
    p
}

/// Combined stdout+stderr of `mindc check <path>`. The default `human` reporter
/// prints diagnostics to stdout, so both streams are read.
fn check_out(path: &std::path::Path) -> String {
    let out = mindc()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("spawn mindc");
    let mut s = String::from_utf8_lossy(&out.stdout).to_string();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    s
}

#[test]
fn function_value_call_rejected() {
    let bad = write_tmp(
        "mind_fn_value_call_bad.mind",
        "pub fn add1(x: i64) -> i64 {\n\
         \x20   return x + 1\n\
         }\n\
         pub fn go() -> i64 {\n\
         \x20   let f = add1\n\
         \x20   return f(41)\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2012"),
        "function-value call not rejected (undefined-symbol miscompile); out: {out}"
    );
}

#[test]
fn direct_call_not_flagged_as_fn_value() {
    let good = write_tmp(
        "mind_fn_value_call_good.mind",
        "pub fn add1(x: i64) -> i64 {\n\
         \x20   return x + 1\n\
         }\n\
         pub fn go() -> i64 {\n\
         \x20   return add1(41)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2012"),
        "direct call to a real module fn falsely rejected as a function value: {out}"
    );
}
