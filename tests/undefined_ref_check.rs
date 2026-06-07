// Regression tests for #23: the type-checker must detect undefined variable and
// undefined call references in `mindc check`. The motivating gap was that an
// identifier used only in an `if`-CONDITION expression was skipped while the
// branch bodies were checked — so an undefined reference in a condition slipped
// through with no diagnostic.

use std::collections::HashMap;

use libmind::diagnostics;
use libmind::parser;
use libmind::type_checker;

fn rendered_diags(src: &str) -> String {
    let module = parser::parse_with_diagnostics(src).expect("parse failed");
    let diags = type_checker::check_module_types(&module, src, &HashMap::new());
    diags
        .iter()
        .map(|d| diagnostics::render(src, d))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn undefined_ident_in_if_condition_is_detected() {
    // `UNDEF` is referenced only in the if-condition; it is not a param, let,
    // or fn. Before the fix this slipped through (the condition was not checked).
    let src = "fn f() -> i64 { if UNDEF == 1 { return 0 as i64; } 1 as i64 }";
    let out = rendered_diags(src);
    assert!(
        out.contains("UNDEF"),
        "undefined identifier in if-condition must be diagnosed, got: {out}"
    );
}

#[test]
fn undefined_bare_ident_is_detected() {
    let src = "fn f() -> i64 { return r; }";
    let out = rendered_diags(src);
    assert!(
        out.contains('r') && out.to_lowercase().contains("unknown"),
        "undefined `return r` must be diagnosed, got: {out}"
    );
}

#[test]
fn defined_param_in_if_condition_not_flagged() {
    // The condition references a real parameter — must NOT be flagged (no
    // false positive from the new condition check).
    let src = "fn f(x: i64) -> i64 { if x == 1 { return 0 as i64; } 1 as i64 }";
    let out = rendered_diags(src);
    assert!(
        !out.to_lowercase().contains("unknown identifier"),
        "a defined parameter used in an if-condition must not be flagged, got: {out}"
    );
}
