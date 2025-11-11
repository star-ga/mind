use std::collections::HashMap;

use mind::eval;
use mind::parser;
#[test]
fn relu_preview_keeps_shape() {
    let src = "let x: Tensor[f32,(2,3)] = 0; tensor.relu(x - 1)";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("Tensor[F32,(2,3)]"));
}
