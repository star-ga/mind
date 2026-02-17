// Audit coverage tests for the IR verifier (C1: SSA verification, conv2d stride/axis validation).
// Added 2026-02-17 per consensus gap analysis.

use libmind::ir::{verify_module, BinOp, IRModule, Instr, IrVerifyError, ValueId};
use libmind::types::ConvPadding;

fn scalar_const(ir: &mut IRModule, value: i64) -> ValueId {
    let id = ir.fresh();
    ir.instrs.push(Instr::ConstI64(id, value));
    id
}

// --- Conv2d stride validation ---

#[test]
fn conv2d_valid_strides_pass_verification() {
    let mut m = IRModule::new();
    let input = scalar_const(&mut m, 0);
    let filter = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2d {
        dst,
        input,
        filter,
        stride_h: 1,
        stride_w: 1,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    assert!(verify_module(&m).is_ok());
}

#[test]
fn conv2d_stride_h_zero_rejected() {
    let mut m = IRModule::new();
    let input = scalar_const(&mut m, 0);
    let filter = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2d {
        dst,
        input,
        filter,
        stride_h: 0,
        stride_w: 1,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("strides must be positive"))
    );
}

#[test]
fn conv2d_stride_w_zero_rejected() {
    let mut m = IRModule::new();
    let input = scalar_const(&mut m, 0);
    let filter = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2d {
        dst,
        input,
        filter,
        stride_h: 2,
        stride_w: 0,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("strides must be positive"))
    );
}

#[test]
fn conv2d_grad_input_stride_zero_rejected() {
    let mut m = IRModule::new();
    let dy = scalar_const(&mut m, 0);
    let filter = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2dGradInput {
        dst,
        dy,
        filter,
        input_shape: [1, 4, 4, 1],
        stride_h: 0,
        stride_w: 1,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("conv2d_grad_input"))
    );
}

#[test]
fn conv2d_grad_filter_stride_zero_rejected() {
    let mut m = IRModule::new();
    let input = scalar_const(&mut m, 0);
    let dy = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2dGradFilter {
        dst,
        input,
        dy,
        filter_shape: [3, 3, 1, 1],
        stride_h: 1,
        stride_w: 0,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("conv2d_grad_filter"))
    );
}

// --- Reduction axis validation ---

#[test]
fn sum_negative_axis_rejected() {
    let mut m = IRModule::new();
    let src = scalar_const(&mut m, 42);
    let dst = m.fresh();
    m.instrs.push(Instr::Sum {
        dst,
        src,
        axes: vec![-1],
        keepdims: false,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("non-negative"))
    );
}

#[test]
fn mean_negative_axis_rejected() {
    let mut m = IRModule::new();
    let src = scalar_const(&mut m, 42);
    let dst = m.fresh();
    m.instrs.push(Instr::Mean {
        dst,
        src,
        axes: vec![0, -2],
        keepdims: false,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("non-negative"))
    );
}

#[test]
fn sum_valid_axes_pass() {
    let mut m = IRModule::new();
    let src = scalar_const(&mut m, 42);
    let dst = m.fresh();
    m.instrs.push(Instr::Sum {
        dst,
        src,
        axes: vec![0, 1],
        keepdims: false,
    });
    m.instrs.push(Instr::Output(dst));
    assert!(verify_module(&m).is_ok());
}

// --- Gather axis validation ---

#[test]
fn gather_negative_axis_rejected() {
    let mut m = IRModule::new();
    let src = scalar_const(&mut m, 0);
    let indices = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Gather {
        dst,
        src,
        indices,
        axis: -1,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(
        matches!(err, IrVerifyError::InvalidOperand { message, .. } if message.contains("non-negative"))
    );
}

#[test]
fn gather_valid_axis_pass() {
    let mut m = IRModule::new();
    let src = scalar_const(&mut m, 0);
    let indices = scalar_const(&mut m, 0);
    let dst = m.fresh();
    m.instrs.push(Instr::Gather {
        dst,
        src,
        indices,
        axis: 0,
    });
    m.instrs.push(Instr::Output(dst));
    assert!(verify_module(&m).is_ok());
}

// --- FnDef body scope validation ---

#[test]
fn fndef_body_use_before_def_rejected() {
    let mut m = IRModule::new();
    let phantom = ValueId(99);
    let body_dst = ValueId(100);
    // Body references phantom (99) which is not defined inside the body
    m.instrs.push(Instr::FnDef {
        name: "test_fn".to_string(),
        params: vec![],
        ret_id: None,
        body: vec![Instr::BinOp {
            dst: body_dst,
            op: BinOp::Add,
            lhs: phantom, // not defined in body
            rhs: phantom,
        }],
    });
    // Need output for the module
    let out = scalar_const(&mut m, 0);
    m.instrs.push(Instr::Output(out));
    let err = verify_module(&m).unwrap_err();
    assert!(matches!(err, IrVerifyError::UseBeforeDefinition { .. }));
}

#[test]
fn fndef_body_with_valid_defs_passes() {
    let mut m = IRModule::new();
    let body_a = ValueId(50);
    let body_b = ValueId(51);
    let body_sum = ValueId(52);
    m.instrs.push(Instr::FnDef {
        name: "valid_fn".to_string(),
        params: vec![("x".to_string(), body_a)],
        ret_id: None,
        body: vec![
            Instr::ConstI64(body_a, 1),
            Instr::ConstI64(body_b, 2),
            Instr::BinOp {
                dst: body_sum,
                op: BinOp::Add,
                lhs: body_a,
                rhs: body_b,
            },
        ],
    });
    let out = scalar_const(&mut m, 0);
    m.instrs.push(Instr::Output(out));
    assert!(verify_module(&m).is_ok());
}

// --- Duplicate definition ---

#[test]
fn duplicate_definition_rejected() {
    let mut m = IRModule::new();
    let id = m.fresh();
    m.instrs.push(Instr::ConstI64(id, 1));
    m.instrs.push(Instr::ConstI64(id, 2)); // duplicate
    m.instrs.push(Instr::Output(id));
    let err = verify_module(&m).unwrap_err();
    assert!(matches!(err, IrVerifyError::DuplicateDefinition(_)));
}

// --- Conv2d with undefined operands ---

#[test]
fn conv2d_undefined_input_rejected() {
    let mut m = IRModule::new();
    let filter = scalar_const(&mut m, 0);
    let phantom = ValueId(99);
    let dst = m.fresh();
    m.instrs.push(Instr::Conv2d {
        dst,
        input: phantom,
        filter,
        stride_h: 1,
        stride_w: 1,
        padding: ConvPadding::Valid,
    });
    m.instrs.push(Instr::Output(dst));
    let err = verify_module(&m).unwrap_err();
    assert!(matches!(
        err,
        IrVerifyError::UseBeforeDefinition { value, .. } if value == phantom
    ));
}
