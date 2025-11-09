#[test]
fn grad_through_slice_shapes_ok() {
    let src = r#"
        let X: Tensor[f32,(3,6)] = 0;
        let y = tensor.sum(tensor.slice(X, axis=1, start=1, end=4));
        grad(y, wrt=[X])
    "#;
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("X: Tensor[F32,(3,6)]"));
}
