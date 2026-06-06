// Regression test for the silent i64->i32 narrowing miscompile found by MIND-Fuzz
// (tools/mindfuzz). The spec requires an explicit `as` cast for integer
// conversions (grammar AsCast; language.md "operands MUST share a dtype"), so an
// implicit narrowing that loses data must be a type error (E2004), not a silent
// truncation. The explicit `as i32` form is accepted by the COMPILER (verified via
// the mindc compile path); the tree-walking eval interpreter does not implement
// `as` casts, so the positive guard below checks that ordinary same-type code is
// NOT over-rejected by the new check.

use libmind::eval;
use libmind::parser;

#[test]
fn implicit_narrowing_i64_to_i32_is_rejected() {
    // 4294967297 = 2^32 + 1 — does not fit in i32; previously truncated to 1
    // with no diagnostic (a silent miscompile).
    let src = "let big: i64 = 4294967297; let small: i32 = big; small";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let err = eval::eval_module_with_env(&m, &mut env, Some(src))
        .expect_err("implicit i64->i32 narrowing must be a type error, not silent truncation");
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("narrow"),
        "expected a narrowing diagnostic (E2004), got: {err}"
    );
}

#[test]
fn same_type_assignment_not_over_rejected() {
    // The new narrowing check must not flag ordinary same-dtype bindings.
    let src = "let n: i32 = 3; n + 1";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let out = eval::eval_module_with_env(&m, &mut env, Some(src))
        .expect("same-dtype binding must still type-check and evaluate");
    assert_eq!(out, 4);
}
