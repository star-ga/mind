use mind::{eval, parser};

#[test]
fn scalar_annotation_matches() {
    let src = "let n: i32 = 3; n + 1";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let out = eval::eval_module_with_env(&m, &mut env, Some(src)).unwrap();
    assert_eq!(out, 4);
}

#[test]
fn tensor_ann_blocks_scalar_ops() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let m = parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let err = eval::eval_module_with_env(&m, &mut env, Some(src)).unwrap_err();
    let s = format!("{err}");
    assert!(s.to_lowercase().contains("type"), "got: {s}");
}
