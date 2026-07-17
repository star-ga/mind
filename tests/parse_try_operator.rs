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

//! Parser + type-checker tests for the W1.5f postfix `?` operator.
//!
//! `expr?` parses to a first-class `ast::Node::Try` (so `mindc fmt` can
//! round-trip the sugar); the lowering desugars it into the W1.5a `match`
//! machinery. These tests pin: (1) `?` parses to a `Node::Try`, (2) the
//! Result/Option family is resolved from the enclosing fn's return type, and
//! (3) `?` in a fn that returns neither Result nor Option is a hard parse
//! error (so no broken artifact is ever emitted).

use libmind::ast::Node;
use libmind::parser::parse;

/// Recursively search a node for the first `Node::Try`, returning its
/// `is_option` family flag. Covers the container shapes the fixtures use.
fn find_try(node: &Node) -> Option<bool> {
    match node {
        Node::Try { is_option, .. } => Some(*is_option),
        Node::FnDef { body, .. } | Node::Block { stmts: body, .. } => {
            body.iter().find_map(find_try)
        }
        Node::Let { value, .. } => find_try(value),
        Node::Return {
            value: Some(inner), ..
        } => find_try(inner),
        Node::Call { args, .. } => args.iter().find_map(find_try),
        Node::Binary { left, right, .. } => find_try(left).or_else(|| find_try(right)),
        Node::Paren(inner, _) => find_try(inner),
        Node::Match { arms, .. } => arms.iter().find_map(|a| find_try(&a.body)),
        _ => None,
    }
}

fn first_try_family(src: &str) -> Option<bool> {
    let module = parse(src).expect("source must parse");
    module.items.iter().find_map(find_try)
}

#[test]
fn question_mark_parses_to_try_node_result_family() {
    // `?` inside a Result-returning fn parses to `Node::Try { is_option: false }`.
    let src = "fn g(x: i64) -> Result<i64, i64> { Ok(x) }\n\
               fn f(x: i64) -> Result<i64, i64> {\n\
                 let v = g(x)?\n\
                 Ok(v + 1)\n\
               }\n";
    assert_eq!(
        first_try_family(src),
        Some(false),
        "`g(x)?` in a Result fn must parse to a Node::Try with the Ok/Err family"
    );
}

#[test]
fn question_mark_parses_to_try_node_option_family() {
    // `?` inside an Option-returning fn parses to `Node::Try { is_option: true }`.
    let src = "fn g(x: i64) -> Option<i64> { Some(x) }\n\
               fn f(x: i64) -> Option<i64> {\n\
                 let v = g(x)?\n\
                 Some(v + 1)\n\
               }\n";
    assert_eq!(
        first_try_family(src),
        Some(true),
        "`g(x)?` in an Option fn must parse to a Node::Try with the Some/None family"
    );
}

#[test]
fn question_mark_in_last_expression_position_parses() {
    // `?` in the final expression, nested inside a call + arithmetic.
    let src = "fn g(x: i64) -> Result<i64, i64> { Ok(x) }\n\
               fn f(x: i64) -> Result<i64, i64> {\n\
                 Ok(g(x)? + 5)\n\
               }\n";
    assert_eq!(
        first_try_family(src),
        Some(false),
        "`g(x)?` in last-expression position must still parse to a Node::Try"
    );
}

#[test]
fn question_mark_rejected_outside_result_or_option_fn() {
    // A fn returning a plain scalar must reject `?` at parse time — a clear
    // error, so lowering never runs and no broken artifact is emitted.
    let src = "fn g(x: i64) -> Result<i64, i64> { Ok(x) }\n\
               fn bad(x: i64) -> i64 {\n\
                 let v = g(x)?\n\
                 v + 1\n\
               }\n";
    let err = parse(src)
        .err()
        .expect("`?` in an i64-returning fn must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("`?`") && msg.contains("Result") && msg.contains("Option"),
        "the rejection must name `?` and the Result/Option requirement, got: {msg}"
    );
}

#[test]
fn question_mark_rejected_when_fn_has_no_return_type() {
    // A fn with no `-> T` at all must also reject `?`.
    let src = "fn g(x: i64) -> Result<i64, i64> { Ok(x) }\n\
               fn bad(x: i64) {\n\
                 let v = g(x)?\n\
               }\n";
    assert!(
        parse(src).is_err(),
        "`?` in a fn with no declared return type must be rejected"
    );
}
