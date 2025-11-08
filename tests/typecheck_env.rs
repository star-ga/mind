use mind::{eval, parser};

#[test]
fn env_prevents_unknown() {
    let src = "let x = 2; x + 3";
    let module = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let result = eval::eval_module_with_env(&module, &mut env, Some(src)).unwrap();
    assert_eq!(result, 5);
}
