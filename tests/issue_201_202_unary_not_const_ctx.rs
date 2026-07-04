// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

// Regression coverage for issues #201 (unary logical NOT) and #202 (i64 const
// literal context-typing). Both features already landed in the front-end; these
// tests lock the behaviour in.

use std::collections::HashMap;

use libmind::parser;
use libmind::type_checker;
use libmind::types::ValueType;

// ---- Issue #201: unary `!` (logical NOT) prefix operator ----

#[test]
fn unary_not_in_if_condition_parses() {
    // The exact repro from the issue must parse without error.
    let src =
        "fn g() -> bool { true }\nfn f() -> bool {\n    if !g() { return false; }\n    true\n}\n";
    parser::parse(src).expect("`if !g()` must parse (issue #201)");
}

#[test]
fn unary_not_type_checks() {
    let src =
        "fn g() -> bool { true }\nfn f() -> bool {\n    if !g() { return false; }\n    true\n}\n";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    let errs: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.severity, libmind::diagnostics::Severity::Error))
        .collect();
    assert!(errs.is_empty(), "unexpected type errors: {errs:?}");
}

#[test]
fn unary_not_on_bool_literal_parses() {
    // `!` in a bare expression position, disambiguated from the `!=` binop.
    let src = "let a = true\nlet b = !a\nb";
    parser::parse(src).expect("`!a` must parse");
}

// ---- Issue #202: i64 const literal context-typing ----

#[test]
fn const_i64_from_bare_literal_type_checks() {
    // `const X: i64 = 1;` — the bare literal must context-type to i64.
    let src = "const X: i64 = 1\nfn f() -> i64 { X }\n";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    let errs: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.severity, libmind::diagnostics::Severity::Error))
        .collect();
    assert!(
        errs.is_empty(),
        "`const X: i64 = 1` must type-check (issue #202), got: {errs:?}"
    );
}

#[test]
fn const_i64_from_explicit_cast_still_type_checks() {
    // The pre-existing accepted form must keep working.
    let src = "const Z: i64 = 1 as i64\nfn f() -> i64 { Z }\n";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    let errs: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.severity, libmind::diagnostics::Severity::Error))
        .collect();
    assert!(errs.is_empty(), "`1 as i64` must still work, got: {errs:?}");
}

#[test]
fn const_widening_covers_the_unsigned_and_signed_int_family() {
    let src = "const A: i64 = 1\nconst B: u64 = 2\nfn f() -> i64 { A }\n";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    let errs: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.severity, libmind::diagnostics::Severity::Error))
        .collect();
    assert!(
        errs.is_empty(),
        "int const widening must hold, got: {errs:?}"
    );
}

#[test]
fn const_bool_from_int_literal_still_rejected() {
    // Context-typing must not become a blanket accept: a genuine mismatch
    // (bool annotation, integer literal) still errors.
    let src = "const X: bool = 1\nfn f() -> bool { X }\n";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    let errs: Vec<_> = diags
        .iter()
        .filter(|d| matches!(d.severity, libmind::diagnostics::Severity::Error))
        .collect();
    assert!(
        !errs.is_empty(),
        "`const X: bool = 1` must still be rejected"
    );
}
