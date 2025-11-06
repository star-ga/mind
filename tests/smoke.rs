use mind::{lexer, parser, type_checker};

#[test]
fn lex_parse_check_minimal() {
    let toks = lexer::lex("x 123");
    assert!(toks.len() >= 2);
    let m = parser::parse("x 123").expect("parse");
    type_checker::check(&m).expect("check");
}
