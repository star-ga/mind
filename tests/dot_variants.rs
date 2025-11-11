use std::collections::HashMap;

use mind::eval;
use mind::parser;
#[test]
fn dot_vec_vec_scalar() {
    let src = r#" let v: Tensor[f32,(3)] = 1; let w: Tensor[f32,(3)] = 2; tensor.dot(v,w) "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("Tensor[F32,()]"));
    assert!(s.contains("fill=6"));
}
