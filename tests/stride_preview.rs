#[test]
fn stride_pos_step_len() {
    let src = r#" let x: Tensor[f32,(2,10)] = 7; tensor.slice_stride(x, axis=1, start=0, end=10, step=2) "#;
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("(2,5)"));
    assert!(s.contains("fill=7"));
}
