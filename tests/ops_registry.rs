use mind::ops::core_v1;

#[test]
fn core_v1_contains_expected_ops() {
    let ops = core_v1::core_v1_ops();
    assert!(core_v1::is_core_v1_op("tensor.sum"));
    assert!(core_v1::is_core_v1_op("tensor.relu"));
    assert_eq!(ops.len(), core_v1::core_v1_ops().len());
}

#[test]
fn lookup_returns_metadata() {
    let relu = core_v1::core_v1_op_signature("tensor.relu").unwrap();
    assert!(relu.differentiable);
    assert!(matches!(relu.arity, core_v1::Arity::Fixed(1)));
}
