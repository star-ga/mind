#![cfg(feature = "mlir-lowering")]

use libmind::compile_ir_to_mlir_text;
use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::types::{ConvPadding, DType, ShapeDim};

fn tensor_const(
    module: &mut IRModule,
    dtype: DType,
    shape: Vec<ShapeDim>,
    fill: Option<f64>,
) -> ValueId {
    let id = module.fresh();
    module
        .instrs
        .push(Instr::ConstTensor(id, dtype, shape, fill));
    id
}

fn scalar_const(module: &mut IRModule, value: i64) -> ValueId {
    let id = module.fresh();
    module.instrs.push(Instr::ConstI64(id, value));
    id
}

#[test]
fn lowers_basic_arithmetic() {
    let mut module = IRModule::new();
    let a = tensor_const(
        &mut module,
        DType::F32,
        vec![ShapeDim::Known(2), ShapeDim::Known(2)],
        Some(1.0),
    );
    let b = tensor_const(
        &mut module,
        DType::F32,
        vec![ShapeDim::Known(2), ShapeDim::Known(2)],
        Some(2.0),
    );
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Add,
        lhs: a,
        rhs: b,
    });
    module.instrs.push(Instr::Output(dst));

    let mut cloned = module.clone();
    let text_a = compile_ir_to_mlir_text(&mut module).expect("lowering should succeed");
    let text_b =
        compile_ir_to_mlir_text(&mut cloned).expect("lowering should succeed deterministically");

    assert!(text_a.contains("func.func @main() -> (tensor<2x2xf32>)"));
    assert!(text_a.contains("linalg.fill ins(%fill"));
    assert!(text_a.contains("arith.addf"));
    assert_eq!(text_a, text_b, "lowering must be deterministic");
}

#[test]
fn lowers_matmul() {
    let mut module = IRModule::new();
    let lhs = tensor_const(
        &mut module,
        DType::F32,
        vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        Some(0.0),
    );
    let rhs = tensor_const(
        &mut module,
        DType::F32,
        vec![ShapeDim::Known(3), ShapeDim::Known(4)],
        Some(0.0),
    );
    let dst = module.fresh();
    module.instrs.push(Instr::MatMul {
        dst,
        a: lhs,
        b: rhs,
    });
    module.instrs.push(Instr::Output(dst));

    let text = compile_ir_to_mlir_text(&mut module).expect("matmul lowering");
    assert!(text.contains("func.func @main() -> (tensor<2x4xf32>)"));
    assert!(text.contains("%tmp2 = tensor.empty() : tensor<2x4xf32>"));
    assert!(text.contains(
        "linalg.matmul ins(%0 : tensor<2x3xf32> , %1 : tensor<3x4xf32>) outs(%tmp2 : tensor<2x4xf32>) -> tensor<2x4xf32>"
    ));
}

#[test]
fn lowers_conv2d() {
    let mut module = IRModule::new();
    let input = tensor_const(
        &mut module,
        DType::F32,
        vec![
            ShapeDim::Known(1),
            ShapeDim::Known(8),
            ShapeDim::Known(8),
            ShapeDim::Known(3),
        ],
        Some(0.0),
    );
    let filter = tensor_const(
        &mut module,
        DType::F32,
        vec![
            ShapeDim::Known(3),
            ShapeDim::Known(3),
            ShapeDim::Known(3),
            ShapeDim::Known(4),
        ],
        Some(0.0),
    );
    let dst = module.fresh();
    module.instrs.push(Instr::Conv2d {
        dst,
        input,
        filter,
        stride_h: 1,
        stride_w: 1,
        padding: ConvPadding::Same,
    });
    module.instrs.push(Instr::Output(dst));

    let text = compile_ir_to_mlir_text(&mut module).expect("conv2d lowering");
    assert!(text.contains("func.func @main() -> (tensor<1x8x8x4xf32>)"));
    assert!(text.contains("%tmp2 = tensor.empty() : tensor<1x8x8x4xf32>"));
    assert!(text.contains(
        "linalg.conv_2d_nhwc_hwcf ins(%0 : tensor<1x8x8x3xf32>, %1 : tensor<3x3x3x4xf32>) outs(%tmp2 : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>"
    ));
}

#[test]
fn reports_unsupported_ops() {
    let mut module = IRModule::new();
    let src = tensor_const(&mut module, DType::F32, vec![ShapeDim::Known(2)], Some(0.0));
    let dst = module.fresh();
    module.instrs.push(Instr::Gather {
        dst,
        src,
        indices: src,
        axis: 0,
    });
    module.instrs.push(Instr::Output(dst));

    let err = compile_ir_to_mlir_text(&mut module).expect_err("gather is not lowered in phase 1");
    assert!(matches!(err, libmind::MlirLowerError::UnsupportedOp { .. }));
}

#[test]
fn lowers_multiple_outputs() {
    let mut module = IRModule::new();
    let a = scalar_const(&mut module, 1);
    let b = scalar_const(&mut module, 2);
    let sum = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst: sum,
        op: BinOp::Add,
        lhs: a,
        rhs: b,
    });
    module.instrs.push(Instr::Output(sum));
    module.instrs.push(Instr::Output(a));

    let text = compile_ir_to_mlir_text(&mut module).expect("lowering should work");
    assert!(text.contains("func.func @main() -> (i64, i64)"));
    assert!(text.contains("return %"));
    assert!(text.contains("i64"));
}

#[test]
fn conv2d_mismatched_channels() {
    let mut module = IRModule::new();
    let input = tensor_const(
        &mut module,
        DType::F32,
        vec![
            ShapeDim::Known(1),
            ShapeDim::Known(4),
            ShapeDim::Known(4),
            ShapeDim::Known(3),
        ],
        Some(0.0),
    );
    let filter = tensor_const(
        &mut module,
        DType::F32,
        vec![
            ShapeDim::Known(3),
            ShapeDim::Known(3),
            ShapeDim::Known(4),
            ShapeDim::Known(8),
        ],
        Some(0.0),
    );
    let dst = module.fresh();
    module.instrs.push(Instr::Conv2d {
        dst,
        input,
        filter,
        stride_h: 1,
        stride_w: 1,
        padding: ConvPadding::Same,
    });
    module.instrs.push(Instr::Output(dst));

    let err = compile_ir_to_mlir_text(&mut module).expect_err("channel mismatch should error");
    assert!(
        matches!(err, libmind::MlirLowerError::ShapeError(msg) if msg.contains("input channels"))
    );
}
