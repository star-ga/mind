use libmind::shapes::engine::{
    broadcast_shapes, infer_output_shape, is_elementwise, rule_for_op, ShapeErrorKind,
    ShapeRuleKind,
};

#[test]
fn rule_for_known_ops() {
    assert_eq!(
        rule_for_op("tensor.relu"),
        Some(ShapeRuleKind::ElementwiseUnary)
    );
    assert_eq!(
        rule_for_op("tensor.add"),
        Some(ShapeRuleKind::ElementwiseBinary)
    );
    assert_eq!(
        rule_for_op("tensor.sum_all"),
        Some(ShapeRuleKind::ReduceAll)
    );
    assert_eq!(rule_for_op("tensor.matmul"), Some(ShapeRuleKind::MatMul2D));
    assert!(rule_for_op("tensor.unknown").is_none());
}

#[test]
fn elementwise_flag_matches_rules() {
    assert!(is_elementwise("tensor.add"));
    assert!(is_elementwise("tensor.relu"));
    assert!(!is_elementwise("tensor.matmul"));
    assert!(!is_elementwise("tensor.sum_all"));
}

#[test]
fn broadcast_shapes_simple() {
    let a = [2, 3];
    let b = [1, 3];
    let out = broadcast_shapes(&a, &b).expect("broadcast should succeed");
    assert_eq!(out, vec![2, 3]);
}

#[test]
fn broadcast_shapes_error() {
    let a = [2, 3];
    let b = [4, 3];
    let err = broadcast_shapes(&a, &b).unwrap_err();
    match err {
        ShapeErrorKind::BroadcastError { lhs, rhs } => {
            assert_eq!(lhs, vec![2, 3]);
            assert_eq!(rhs, vec![4, 3]);
        }
        _ => panic!("expected BroadcastError"),
    }
}

#[test]
fn infer_elementwise_binary_broadcast() {
    let out = infer_output_shape("tensor.add", &[&[2, 3][..], &[1, 3][..]])
        .expect("elementwise add should broadcast");
    assert_eq!(out, vec![2, 3]);
}

#[test]
fn infer_elementwise_unary_identity() {
    let out =
        infer_output_shape("tensor.relu", &[&[4, 5, 6][..]]).expect("relu should preserve shape");
    assert_eq!(out, vec![4, 5, 6]);
}

#[test]
fn infer_matmul_2d_ok() {
    let out = infer_output_shape("tensor.matmul", &[&[2, 3][..], &[3, 4][..]])
        .expect("matmul should work");
    assert_eq!(out, vec![2, 4]);
}

#[test]
fn infer_matmul_mismatched_inner_dim() {
    let err = infer_output_shape("tensor.matmul", &[&[2, 3][..], &[4, 5][..]]).unwrap_err();
    match err.kind {
        ShapeErrorKind::RankMismatch { .. } => {}
        _ => panic!("expected RankMismatch for mismatched inner dims"),
    }
}

#[test]
fn infer_reduce_all_to_scalar() {
    let out = infer_output_shape("tensor.sum_all", &[&[2, 2][..]])
        .expect("sum_all should reduce to scalar");
    // Rank-0 scalar represented as an empty shape.
    assert_eq!(out, Vec::<usize>::new());
}

#[test]
fn infer_unknown_op_reports_error() {
    let err = infer_output_shape("tensor.unknown", &[&[1, 2][..]]).unwrap_err();
    match err.kind {
        ShapeErrorKind::UnknownOp => {}
        _ => panic!("expected UnknownOp error"),
    }
}
