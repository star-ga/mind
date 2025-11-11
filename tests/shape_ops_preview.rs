use std::collections::HashMap;

use mind::eval;
use mind::parser;
#[test]
fn reshape_and_back_grad_shape_only() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 5;
        let y = tensor.reshape(x, (3,2));
        grad(tensor.sum(y), wrt=[x])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("x: Tensor[F32,(2,3)]"));
    assert!(s.contains("fill=1"));
}
