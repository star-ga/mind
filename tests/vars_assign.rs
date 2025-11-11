use mind::eval;
use mind::parser;

#[test]
fn let_and_use_variable() {
    let m = parser::parse("let x = 2; x * 3 + 1").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 7);
}

#[test]
fn assign_updates_value() {
    let m = parser::parse("let x = 1; x = x + 4; x * 2").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 10);
}
