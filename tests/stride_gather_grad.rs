#[test]
fn grad_through_stride_and_gather_shapes() {
    let src = r#"
        let X: Tensor[f32,(5,10)] = 0;
        let I: Tensor[i32,(4)] = 0;
        let y = tensor.sum(tensor.gather(tensor.slice_stride(X, axis=1, start=0, end=10, step=2), axis=1, idx=I));
        grad(y, wrt=[X])
    "#;
    let m = mind::parser::parse(src).unwrap();
    let mut env = std::collections::HashMap::new();
    let v = mind::eval::eval_module_value_with_env(&m, &mut env, Some(src)).unwrap();
    let s = mind::eval::format_value_human(&v);
    assert!(s.contains("X: Tensor[F32,(5,10)]"));
}
