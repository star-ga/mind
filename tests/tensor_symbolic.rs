use mind::{
    parser, type_checker,
    types::{DType, ShapeDim, TensorType, ValueType},
};
use std::collections::HashMap;

#[test]
fn broadcast_with_symbols_equal_symbols_ok() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(1)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(diags.is_empty(), "{:?}", diags);
}

#[test]
fn broadcast_with_symbols_mismatch_fails() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Sym("C"), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(!diags.is_empty(), "expected symbol mismatch");
}
