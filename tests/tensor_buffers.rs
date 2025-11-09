#[cfg(feature = "cpu-buffers")]
#[test]
fn materializes_small_filled_tensor() {
    let src = "let x: Tensor[f32,(2,3)] = 1; tensor.materialize(x)";
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("materialized"));
    assert!(s.contains("(2,3)"));
}

#[cfg(feature = "cpu-buffers")]
#[test]
fn tensor_sample_uses_materialized_data() {
    let src = "let x: Tensor[f32,(2,3)] = 1; tensor.sample(tensor.materialize(x), 4)";
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("materialized"));
    assert!(s.contains("(4)"));
}

#[cfg(not(feature = "cpu-buffers"))]
#[test]
fn preview_only_without_buffers() {
    let src = "let x: Tensor[f32,(2,3)] = 1; x";
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("fill=1"));
    assert!(!s.contains("materialized"));
}
