use mind::parser;
use mind::type_checker;
use mind::types::DType;
use mind::types::ShapeDim;
use mind::types::TensorType;
use mind::types::ValueType;
use std::collections::HashMap;

#[test]
fn tensor_plus_tensor_same_shape() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(diags.is_empty(), "{:?}", diags);
}

#[test]
fn tensor_plus_tensor_broadcast_tail() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(1), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(1), ShapeDim::Known(4), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(diags.is_empty(), "{:?}", diags);
}

#[test]
fn tensor_plus_scalar_promote_i32_to_f32() {
    let src = "a + 1";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(diags.is_empty(), "{:?}", diags);
}

#[test]
fn dtype_mismatch_rejected() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::I32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(!diags.is_empty(), "expected dtype mismatch");
}

#[test]
fn shape_mismatch_rejected() {
    let src = "a + b";
    let m = parser::parse(src).unwrap();
    let mut env: HashMap<String, ValueType> = HashMap::new();
    env.insert(
        "a".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        )),
    );
    env.insert(
        "b".to_string(),
        ValueType::Tensor(TensorType::new(
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(3)],
        )),
    );
    let diags = type_checker::check_module_types(&m, src, &env);
    assert!(!diags.is_empty(), "expected shape mismatch");
}
