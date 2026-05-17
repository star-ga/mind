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

// Part of the MIND project (Machine Intelligence Native Design).

//! Parser and type-checker tests for Phase 10.7 surface constructs:
//! `match` expressions and `&expr` / `&mut expr` reference-taking expressions.

use libmind::{compile_source, CompileOptions};

fn parses(src: &str) -> bool {
    compile_source(src, &CompileOptions::default()).is_ok()
}

// ──────────────────────────────────────────────────────────────────────
//  match expressions — parser
// ──────────────────────────────────────────────────────────────────────

#[test]
fn match_on_enum_variants() {
    // Basic variant patterns, no wildcard.
    let src = "module m {\n\
                 enum Mode { On, Off }\n\
                 fn f(mode: Mode) -> i32 {\n\
                   match mode {\n\
                     Mode::On  => 1,\n\
                     Mode::Off => 0,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match on enum variants must parse");
}

#[test]
fn match_with_wildcard_arm() {
    let src = "module m {\n\
                 enum Mode { On, Off }\n\
                 fn f(mode: Mode) -> i32 {\n\
                   match mode {\n\
                     Mode::On => 1,\n\
                     _        => 0,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match with wildcard arm must parse");
}

#[test]
fn match_on_integer_literals() {
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   match x {\n\
                     0 => 10,\n\
                     1 => 20,\n\
                     _ => 30,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match on integer literals must parse");
}

#[test]
fn match_with_string_literal_arm() {
    let src = "module m {\n\
                 fn f(s: i32) -> i32 {\n\
                   match s {\n\
                     0 => 1,\n\
                     _ => 2,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match with literal arms must parse");
}

#[test]
fn match_with_block_body() {
    let src = "module m {\n\
                 enum Side { Left, Right }\n\
                 fn f(s: Side) -> i32 {\n\
                   match s {\n\
                     Side::Left  => { let x: i32 = 1\n x },\n\
                     Side::Right => { let x: i32 = 2\n x },\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match with block body arms must parse");
}

#[test]
fn match_identifier_binding_arm() {
    // Bare `x` binding pattern — matches anything, binds the name.
    let src = "module m {\n\
                 fn f(v: i32) -> i32 {\n\
                   match v {\n\
                     0 => 0,\n\
                     x => x,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match with bare ident binding arm must parse");
}

#[test]
fn match_nested_enum_variant_with_payload() {
    // `Result::Ok(x)` style — variant with payload patterns.
    let src = "module m {\n\
                 enum Res { Ok, Err }\n\
                 fn f(r: Res) -> i32 {\n\
                   match r {\n\
                     Res::Ok  => 1,\n\
                     Res::Err => 0,\n\
                     _        => -1,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "match with enum variant with payload must parse"
    );
}

#[test]
fn match_negative_integer_pattern() {
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   match x {\n\
                     -1 => 10,\n\
                     0  => 20,\n\
                     _  => 30,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "match with negative integer pattern must parse"
    );
}

#[test]
fn match_without_trailing_comma() {
    // Trailing comma after the last arm is optional.
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   match x {\n\
                     0 => 1,\n\
                     _ => 2\n\
                   }\n\
                 }\n\
               }\n";
    assert!(parses(src), "match without trailing comma must parse");
}

#[test]
fn match_qualified_enum_variant() {
    // `config.Mode::On` — module-qualified variant path.
    let src = "module config { enum Mode { On, Off } }\n\
               module m {\n\
                 use config\n\
                 fn f(mode: config.Mode) -> i32 {\n\
                   match mode {\n\
                     config.Mode::On  => 1,\n\
                     config.Mode::Off => 0,\n\
                     _                => -1,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "match on module-qualified enum variant must parse"
    );
}

// ──────────────────────────────────────────────────────────────────────
//  match expressions — type-checker
// ──────────────────────────────────────────────────────────────────────

#[test]
fn match_type_checker_accepts_consistent_arms() {
    // All arms return the same scalar type; the type-checker should accept.
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   match x {\n\
                     0 => 1,\n\
                     _ => 0,\n\
                   }\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "match with consistent arm types must be accepted by the type-checker"
    );
}

// ──────────────────────────────────────────────────────────────────────
//  &expr / &mut expr — parser
// ──────────────────────────────────────────────────────────────────────

#[test]
fn ref_expr_in_fn_call_arg() {
    // `reduce(&buf)` — passing a reference to a fn call.
    let src = "module m {\n\
                 fn reduce(buf: &Vec<i32>) -> i32 { 0 }\n\
                 fn caller() -> i32 {\n\
                   let buf: Vec<i32> = Vec::new()\n\
                   reduce(&buf)\n\
                 }\n\
               }\n";
    assert!(parses(src), "`&expr` in fn call arg must parse");
}

#[test]
fn mut_ref_expr_in_fn_call_arg() {
    // `update(&mut buf)` — passing a mutable reference.
    let src = "module m {\n\
                 fn update(buf: &mut Vec<i32>) { }\n\
                 fn caller() {\n\
                   let mut buf: Vec<i32> = Vec::new()\n\
                   update(&mut buf)\n\
                 }\n\
               }\n";
    assert!(parses(src), "`&mut expr` in fn call arg must parse");
}

#[test]
fn ref_expr_in_let_binding() {
    // `let r = &x` — binding a reference expression.
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   let r = &x\n\
                   0\n\
                 }\n\
               }\n";
    assert!(parses(src), "`&expr` in let binding must parse");
}

#[test]
fn mut_ref_expr_in_let_binding() {
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   let mut y: i32 = x\n\
                   let r = &mut y\n\
                   0\n\
                 }\n\
               }\n";
    assert!(parses(src), "`&mut expr` in let binding must parse");
}

#[test]
fn ref_expr_does_not_conflict_with_bitwise_and() {
    // `a & b` (bitwise-AND) must still parse when `&` appears as infix.
    let src = "module m { fn f(a: i32, b: i32) -> i32 { a & b } }\n";
    assert!(
        parses(src),
        "bitwise-AND `&` must still parse after adding ref-take prefix"
    );
}

#[test]
fn ref_expr_and_bitwise_and_coexist() {
    // Both `&expr` prefix and `a & b` infix in the same fn.
    let src = "module m {\n\
                 fn g(x: &i32) -> i32 { 0 }\n\
                 fn f(a: i32, b: i32) -> i32 {\n\
                   let c = a & b\n\
                   let r = &a\n\
                   c\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "`&expr` prefix and bitwise-AND infix must coexist in the same fn"
    );
}

#[test]
fn ref_expr_struct_arg() {
    // `f(&point)` where the parameter type is `&Point`.
    let src = "module m {\n\
                 struct Point { x: i32, y: i32 }\n\
                 fn f(p: &Point) -> i32 { 0 }\n\
                 fn caller() -> i32 {\n\
                   let pt = Point { x: 1, y: 2 }\n\
                   f(&pt)\n\
                 }\n\
               }\n";
    assert!(parses(src), "`&struct_expr` in fn call must parse");
}

#[test]
fn ref_expr_type_is_checked() {
    // The type-checker must accept `&expr` without erroring.
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   let r = &x\n\
                   x\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "`&expr` in let binding must pass the type-checker"
    );
}

#[test]
fn mut_ref_expr_type_is_checked() {
    let src = "module m {\n\
                 fn f(x: i32) -> i32 {\n\
                   let mut y: i32 = x\n\
                   let r = &mut y\n\
                   x\n\
                 }\n\
               }\n";
    assert!(
        parses(src),
        "`&mut expr` in let binding must pass the type-checker"
    );
}
