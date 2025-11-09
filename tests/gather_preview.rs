#[test]
fn gather_inserts_idx_shape() {
    let src = r#"
        let x: Tensor[f32,(3,4)] = 5;
        let idx: Tensor[i32,(2)] = 0;
        tensor.gather(x, axis=1, idx)
    "#;
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("(3,2)"));
    assert!(s.contains("fill=5"));
}
