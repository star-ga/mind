use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn transpose_reverse() {
    let src = r#" let x: Tensor[f32,(2,3,4)] = 0; tensor.transpose(x) "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("(4,3,2)"));
}
