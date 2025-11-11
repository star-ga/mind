use mind::types::infer::unify;
use mind::types::DType;
use mind::types::ShapeDim;
use mind::types::TensorType;

#[test]
fn unify_known_and_sym() {
    let a = TensorType::new(DType::F32, vec![ShapeDim::Known(32), ShapeDim::Sym("B")]);
    let b = TensorType::new(DType::F32, vec![ShapeDim::Known(32), ShapeDim::Known(8)]);
    let u = unify(&a, &b);
    assert_eq!(u.shape, vec![ShapeDim::Known(32), ShapeDim::Known(8)]);
}

#[test]
fn unify_rank_mismatch_left_biased() {
    let a = TensorType::new(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Sym("N")]);
    let b = TensorType::new(DType::F32, vec![ShapeDim::Known(4)]);
    let u = unify(&a, &b);
    assert_eq!(u.shape[0], ShapeDim::Known(4));
}
