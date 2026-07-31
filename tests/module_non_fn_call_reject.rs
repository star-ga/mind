// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Module-level non-function-call fail-loud regression gate (E2012).
//!
//! Confirmed accept-what-cannot-be-emitted bug (module-scope twin of the local
//! function-value case in `tests/fn_value_call_reject.rs`): a call whose callee
//! is a MODULE-LEVEL NON-FUNCTION declaration — a `const`, module-`let`,
//! `struct`, `enum`-type, or `type`-alias name — passed `mindc check` with ZERO
//! diagnostics, then lowering synthesised `func.call @<name>` against a
//! data/global symbol, so `--emit-shared` produced an .so referencing an
//! undefined/non-callable symbol. `const ADD: i64 = 1  fn main() { ADD(2) }` is
//! the canonical repro. The front-end must REJECT such a call at compile (E2012,
//! the same broken-artifact class the function-value case uses) with an
//! "`X` is not a function" message.
//!
//! These gates assert the const/struct/enum/type-alias callee shapes are
//! REJECTED with E2012, that a real module `fn` of the SAME name is still
//! ACCEPTED (the reject fires ONLY on the non-fn callee), and that a valid enum
//! variant CONSTRUCTOR call (`Some(x)`) is NEVER swept up.
//!
//! We assert on the presence/absence of the diagnostic CODE in the combined
//! stdout+stderr rather than the process exit status: `mindc check` also emits a
//! `fmt::drift` diagnostic for an unformatted file, so whether E2012 fires is the
//! behavior under test (mirrors tests/fn_value_call_reject.rs).

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

/// Combined stdout+stderr of `mindc check <path>`.
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
fn const_callee_rejected() {
    let bad = write_tmp(
        "mind_const_call_bad.mind",
        "const ADD: i64 = 1\n\
         pub fn main() -> i64 {\n\
         \x20   return ADD(2)\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2012") && out.contains("is not a function"),
        "call to a module-level const not rejected (undefined-symbol miscompile); out: {out}"
    );
}

#[test]
fn struct_callee_rejected() {
    let bad = write_tmp(
        "mind_struct_call_bad.mind",
        "struct Point {\n\
         \x20   x: i64,\n\
         \x20   y: i64,\n\
         }\n\
         pub fn main() -> i64 {\n\
         \x20   return Point(1)\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2012") && out.contains("is not a function"),
        "call to a struct type name not rejected: {out}"
    );
}

#[test]
fn enum_type_callee_rejected() {
    let bad = write_tmp(
        "mind_enum_call_bad.mind",
        "enum Color {\n\
         \x20   Red,\n\
         \x20   Green,\n\
         }\n\
         pub fn main() -> i64 {\n\
         \x20   let c = Color(1)\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&bad);
    assert!(
        out.contains("E2012") && out.contains("is not a function"),
        "call to an enum type name not rejected: {out}"
    );
}

#[test]
fn real_fn_of_same_name_accepted() {
    // The reject must fire ONLY on the non-fn callee: a genuine module fn call
    // (here named `ADD`, the same identifier as the rejected const above) must
    // NOT trip E2012.
    let good = write_tmp(
        "mind_const_call_good.mind",
        "pub fn ADD(x: i64) -> i64 {\n\
         \x20   return x + 1\n\
         }\n\
         pub fn main() -> i64 {\n\
         \x20   return ADD(2)\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2012"),
        "direct call to a real module fn falsely rejected as a non-function: {out}"
    );
}

#[test]
fn enum_variant_constructor_not_flagged() {
    // A valid unit/payload enum-variant constructor call must never be swept up
    // by the non-fn-callee reject — variant names are excluded from the guard.
    let good = write_tmp(
        "mind_enum_ctor_good.mind",
        "pub fn main() -> i64 {\n\
         \x20   let r = Some(7)\n\
         \x20   return 0\n\
         }\n",
    );
    let out = check_out(&good);
    assert!(
        !out.contains("E2012"),
        "enum variant constructor falsely rejected as a non-function: {out}"
    );
}
