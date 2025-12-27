use mind::ir::{
    canonicalize_module, format_ir_module, prepare_ir_for_backend, verify_module, BinOp, IRModule,
    Instr, IrVerifyError, ValueId,
};

fn scalar_const(ir: &mut IRModule, value: i64) -> ValueId {
    let id = ir.fresh();
    ir.instrs.push(Instr::ConstI64(id, value));
    id
}

#[test]
fn canonicalization_is_deterministic_and_idempotent() {
    let mut module = IRModule::new();
    let unused = scalar_const(&mut module, 7);
    let a = scalar_const(&mut module, 1);
    let b = scalar_const(&mut module, 2);
    let add = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst: add,
        op: BinOp::Add,
        lhs: b,
        rhs: a,
    });
    module.instrs.push(Instr::Output(add));
    // Ensure the unused const is kept alive in the SSA namespace but removed from code.
    module.next_id = unused.0.max(module.next_id);

    let mut second = module.clone();

    canonicalize_module(&mut module);
    canonicalize_module(&mut second);

    assert_eq!(format_ir_module(&module), format_ir_module(&second));

    let snapshot = format_ir_module(&module);
    canonicalize_module(&mut module);
    assert_eq!(snapshot, format_ir_module(&module));
}

#[test]
fn canonicalization_constant_folds_simple_ints() {
    let mut module = IRModule::new();
    let a = scalar_const(&mut module, 4);
    let b = scalar_const(&mut module, 6);
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Mul,
        lhs: a,
        rhs: b,
    });
    module.instrs.push(Instr::Output(dst));

    canonicalize_module(&mut module);
    let printed = format_ir_module(&module);
    assert!(
        printed.contains("const.i64 24"),
        "expected constant folding: {}",
        printed
    );
    assert!(
        !printed.contains("const.i64 4\n  %1 = const.i64 6"),
        "dead inputs were not removed after folding: {}",
        printed
    );
}

#[test]
fn verifier_catches_missing_output() {
    let module = IRModule::new();
    let err = verify_module(&module).unwrap_err();
    assert!(matches!(err, IrVerifyError::MissingOutput));
}

#[test]
fn verifier_rejects_use_before_definition() {
    let mut module = IRModule::new();
    let phantom = ValueId(99);
    module.instrs.push(Instr::Output(phantom));

    let err = verify_module(&module).unwrap_err();
    assert!(matches!(err, IrVerifyError::UseBeforeDefinition { .. }));
}

#[test]
fn verifier_checks_next_id_sync() {
    let mut module = IRModule::new();
    let id = module.fresh();
    module.instrs.push(Instr::ConstI64(id, 1));
    module.instrs.push(Instr::Output(id));
    module.next_id = 0; // deliberately stale

    let err = verify_module(&module).unwrap_err();
    assert!(matches!(err, IrVerifyError::NextIdOutOfSync { .. }));
}

#[test]
fn prepare_pipeline_runs_verifier_and_canonicalizer() {
    let mut module = IRModule::new();
    let a = scalar_const(&mut module, 2);
    let b = scalar_const(&mut module, 3);
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Add,
        lhs: b,
        rhs: a,
    });
    module.instrs.push(Instr::Output(dst));

    prepare_ir_for_backend(&mut module).expect("backend prep");
    let printed = format_ir_module(&module);
    assert!(
        printed.contains("const.i64 5"),
        "expected folded sum: {}",
        printed
    );
}
