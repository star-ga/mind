#[test]
fn index_axis_bounds_checked() {
    let src = r#" let x: Tensor[f32,(2,5)] = 0; tensor.index(x, axis=2, i=0) "#;
    let m = mind::parser::parse(src).unwrap();
    let tenv = std::collections::HashMap::new();
    let diags = mind::type_checker::check_module_types(&m, src, &tenv);
    assert!(!diags.is_empty());
}
