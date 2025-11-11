use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn grad_add_is_ones() {
    let src = "let x: Tensor[f32,(2,3)] = 0; grad(tensor.sum(x + 1), wrt=[x])";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("grad{"));
    assert!(rendered.contains("x: Tensor[F32,(2,3)]"));
    assert!(rendered.contains("fill=1"));
}

#[test]
fn grad_mul_scalar_fill() {
    let src = "let x: Tensor[f32,(2,3)] = 0; grad(tensor.sum(2 * x), wrt=[x])";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("fill=2"));
}
