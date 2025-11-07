use mind::{lexer, parser, type_checker, types::ValueType};

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
