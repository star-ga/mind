use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn grad_sum_is_ones() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        grad(tensor.sum(x), wrt=[x])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("x: Tensor[F32,(2,3)]"));
    assert!(s.contains("fill=1"));
}

#[test]
fn grad_mean_is_1_over_n() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        grad(tensor.mean(x), wrt=[x])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("fill=0.166666") || s.contains("fill=0.166667"));
}
