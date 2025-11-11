use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn grad_sum_matmul_is_ones() {
    let src = r#"
        let A: Tensor[f32,(2,3)] = 0;
        let B: Tensor[f32,(3,4)] = 0;
        grad(tensor.sum(tensor.matmul(A,B)), wrt=[A,B])
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("A: Tensor[F32,(2,3)]"));
    assert!(s.contains("B: Tensor[F32,(3,4)]"));
    assert!(s.contains("fill=1"));
}
