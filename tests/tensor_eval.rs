use mind::eval;
use mind::parser;

use std::collections::HashMap;

#[test]
fn annotated_tensor_plus_scalar_yields_tensor_preview() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let preview = eval::format_value_human(&value);
    assert!(preview.contains("Tensor["));
    assert!(preview.contains("(2,3)"));
    assert!(preview.contains("F32") || preview.contains("f32"));
    if cfg!(feature = "cpu-buffers") {
        assert!(preview.contains("materialized"), "got: {preview}");
    } else {
        assert!(preview.contains("fill=1"), "got: {preview}");
    }
}

#[test]
fn tensor_tensor_broadcast_preview() {
    let src = "let a: Tensor[f32,(2,1,3)] = 0; let b: Tensor[f32,(1,4,3)] = 0; a + b";
    let module = parser::parse(src).unwrap();
    let mut env = HashMap::new();
    let value = eval::eval_module_value_with_env(&module, &mut env, Some(src)).unwrap();
    let preview = eval::format_value_human(&value);
    assert!(preview.contains("(2,4,3)"), "got: {preview}");
}
