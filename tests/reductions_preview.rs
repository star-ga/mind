use std::collections::HashMap;

use mind::eval;
use mind::parser;
#[test]
fn sum_all_axes_preview() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 1;
        tensor.sum(x)
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("Tensor[F32,()]"));
    assert!(s.contains("fill=6"));
}

#[test]
fn mean_keepdims_preview() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 2;
        tensor.mean(x, axes=[1], keepdims=true)
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("Tensor[F32,(2,1)]"));
    assert!(s.contains("fill=2"));
}
