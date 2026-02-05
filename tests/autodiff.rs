#![cfg(feature = "autodiff")]

use std::collections::BTreeMap;

use libmind::differentiate_function;
use libmind::ir::{BinOp, IRModule, Instr, ValueId};
use libmind::types::{DType, ShapeDim};

fn scalar_const(ir: &mut IRModule, value: i64) -> ValueId {
    let id = ir.fresh();
    ir.instrs.push(Instr::ConstI64(id, value));
    id
}

fn scalar_tensor(ir: &mut IRModule) -> ValueId {
    let id = ir.fresh();
    ir.instrs
        .push(Instr::ConstTensor(id, DType::F32, vec![], None));
    id
}

fn assert_deterministic(ir: &IRModule) {
    let one = differentiate_function(ir, "main").expect("first differentiation");
    let two = differentiate_function(ir, "main").expect("second differentiation");
    assert_eq!(
        format!("{}", one.gradient_module),
        format!("{}", two.gradient_module)
    );
    assert_eq!(one.gradients, two.gradients);
}

#[test]
fn grad_of_square() {
    let mut ir = IRModule::new();
    let x = scalar_tensor(&mut ir);
    let y = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: y,
        op: BinOp::Mul,
        lhs: x,
        rhs: x,
    });
    ir.instrs.push(Instr::Output(y));

    let result = differentiate_function(&ir, "main").expect("gradients");
    let grad_x = result.gradients.get(&x).expect("gradient for x");
    assert_deterministic(&ir);

    // Gradient for x*x accumulates two paths: d/dx (x*x) = x + x.
    let seen_add =
        result.gradient_module.instrs.iter().any(
            |instr| matches!(instr, Instr::BinOp { dst, op: BinOp::Add, .. } if dst == grad_x),
        );
    assert!(seen_add, "gradient accumulation for x should use addition");
}

#[test]
fn grad_of_bilinear() {
    let mut ir = IRModule::new();
    let x = scalar_tensor(&mut ir);
    let y = scalar_tensor(&mut ir);
    let xy = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: xy,
        op: BinOp::Mul,
        lhs: x,
        rhs: y,
    });
    let sum = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst: sum,
        op: BinOp::Add,
        lhs: xy,
        rhs: y,
    });
    ir.instrs.push(Instr::Output(sum));

    let result = differentiate_function(&ir, "main").expect("gradients");
    assert_deterministic(&ir);

    // dy should accumulate contributions from both paths.
    let dy = result.gradients.get(&y).expect("gradient for y");
    let mut seen_add = false;
    for instr in &result.gradient_module.instrs {
        if let Instr::BinOp {
            dst,
            op: BinOp::Add,
            ..
        } = instr
        {
            if dst == dy {
                seen_add = true;
            }
        }
    }
    assert!(seen_add, "expected additive accumulation for dy");
}

#[test]
fn matmul_rule_applied() {
    let mut ir = IRModule::new();
    let a = ir.fresh();
    ir.instrs.push(Instr::ConstTensor(
        a,
        DType::F32,
        vec![ShapeDim::Known(2), ShapeDim::Known(2)],
        None,
    ));
    let b = ir.fresh();
    ir.instrs.push(Instr::ConstTensor(
        b,
        DType::F32,
        vec![ShapeDim::Known(2), ShapeDim::Known(2)],
        None,
    ));
    let out = ir.fresh();
    ir.instrs.push(Instr::MatMul { dst: out, a, b });
    ir.instrs.push(Instr::Output(out));

    let result = differentiate_function(&ir, "main").expect("gradients");
    assert_deterministic(&ir);

    let gradients: BTreeMap<_, _> = result.gradients.clone();
    assert!(gradients.contains_key(&a));
    assert!(gradients.contains_key(&b));
    // Expect at least one transpose in the gradient IR for matmul.
    assert!(result
        .gradient_module
        .instrs
        .iter()
        .any(|instr| matches!(instr, Instr::Transpose { .. })));
}

#[test]
fn multiple_outputs_rejected() {
    let mut ir = IRModule::new();
    let x = scalar_const(&mut ir, 1);
    ir.instrs.push(Instr::Output(x));
    ir.instrs.push(Instr::Output(x));

    let err = differentiate_function(&ir, "main").unwrap_err();
    assert!(format!("{}", err).contains("multiple outputs"));
}

#[test]
fn unsupported_division_errors() {
    let mut ir = IRModule::new();
    let x = scalar_const(&mut ir, 4);
    let y = scalar_const(&mut ir, 2);
    let dst = ir.fresh();
    ir.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Div,
        lhs: x,
        rhs: y,
    });
    ir.instrs.push(Instr::Output(dst));

    let err = differentiate_function(&ir, "main").unwrap_err();
    assert!(format!("{}", err).contains("unsupported"));
}
