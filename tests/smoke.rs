// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

use mind::lexer;
use mind::parser;
use mind::type_checker;
use mind::types::ValueType;

#[test]
fn lex_parse_check_minimal() {
    let toks = lexer::lex("x 123");
    assert!(toks.len() >= 2);
    let m = parser::parse("x 123").expect("parse");
    let mut env = std::collections::HashMap::new();
    env.insert("x".to_string(), ValueType::ScalarI32);
    let diags = type_checker::check_module_types(&m, "x 123", &env);
    assert!(diags.is_empty());
}
