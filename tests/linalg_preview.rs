use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn matmul_preview_fill() {
    let src = r#"
        let a: Tensor[f32,(2,3)] = 1;
        let b: Tensor[f32,(3,4)] = 2;
        tensor.matmul(a,b)
    "#;
    let m = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let v = eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = eval::format_value_human(&v);
    assert!(s.contains("(2,4)"));
    assert!(s.contains("fill=6"));
}
