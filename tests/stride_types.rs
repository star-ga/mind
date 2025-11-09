#[test]
fn zero_step_is_error() {
    let src = r#" let x: Tensor[f32,(2,10)] = 0; tensor.slice_stride(x, axis=1, start=0, end=10, step=0) "#;
    let m = mind::parser::parse(src).unwrap();
    let diags = mind::type_checker::check_module_types(&m, src, &std::collections::HashMap::new());
    assert!(!diags.is_empty());
}
