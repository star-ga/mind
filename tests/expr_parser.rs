use mind::eval;
use mind::parser;

#[test]
fn precedence_and_parens() {
    let module = parser::parse("1 + 2 * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 7);

    let module = parser::parse("(1 + 2) * 3").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 9);
}

#[test]
fn division_and_zero_guard() {
    let module = parser::parse("8 / 2").unwrap();
    assert_eq!(eval::eval_first_expr(&module).unwrap(), 4);

    let module = parser::parse("1 / 0").unwrap();
    assert!(eval::eval_first_expr(&module).is_err());
}
