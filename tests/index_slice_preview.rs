use std::collections::HashMap;

use mind::eval;
use mind::parser;
#[test]
fn index_drops_axis() {
    let src = r#" let x: Tensor[f32,(2,5)] = 1; tensor.index(x, axis=1, i=0) "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("(2)"));
    assert!(s.contains("fill=1"));
}

#[test]
fn slice_changes_len() {
    let src = r#" let x: Tensor[f32,(3,6)] = 2; tensor.slice(x, axis=1, start=1, end=4) "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("(3,3)"));
    assert!(s.contains("fill=2"));
}
