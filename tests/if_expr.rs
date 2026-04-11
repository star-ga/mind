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

use libmind::eval;
use libmind::parser;

#[test]
fn if_expr_true_branch() {
    let m = parser::parse("let x = if 1 > 0 { 42 } else { 0 }; x").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 42);
}

#[test]
fn if_expr_false_branch() {
    let m = parser::parse("let x = if 0 > 1 { 42 } else { 7 }; x").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 7);
}

#[test]
fn if_expr_in_arithmetic() {
    let m = parser::parse("1 + if 1 > 0 { 10 } else { 20 }").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 11);
}

#[test]
fn if_stmt_standalone_parses() {
    let m = parser::parse("if 1 > 0 { 42 }");
    assert!(m.is_ok());
}

#[test]
fn if_no_else_returns_zero_on_false() {
    let m = parser::parse("if 0 > 1 { 99 }").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 0);
}

#[test]
fn if_else_if_chain() {
    let m = parser::parse("let x = if 0 > 1 { 1 } else if 0 > 2 { 2 } else { 3 }; x").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 3);
}
