#[test]
fn conv2d_channel_mismatch_errors() {
    let src = r#"
        let x: Tensor[f32,(1,2,2,2)] = 0;
        let w: Tensor[f32,(1,1,3,1)] = 0;
        tensor.conv2d(x, w)
    "#;
    let module = mind::parser::parse(src).unwrap();
    let diags = mind::type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(!diags.is_empty());
}

#[test]
fn conv2d_same_padding_symbolic_shapes() {
    let src = r#"
        let x: Tensor[f32,(N,H,W,C)] = 0;
        let w: Tensor[f32,(3,3,C,F)] = 0;
        tensor.conv2d(x, w, stride_h=2, stride_w=2, padding="same")
    "#;
    let module = mind::parser::parse(src).unwrap();
    let diags = mind::type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(diags.is_empty());
}
use std::collections::HashMap;
