use libmind::ir::{
    BinOp, IRModule, Instr, IrVerifyError, ValueId, canonicalize_module, format_ir_module,
    prepare_ir_for_backend, verify_module,
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

/// Regression: a value consumed ONLY as a `Call` argument must be marked live
/// and survive dead-code elimination in `canonicalize_module`.
///
/// Before the fix, `instruction_operands` returned an empty operand set for
/// `Instr::Call` (the `_ => vec![]` catch-all), so the argument value was never
/// inserted into the `used` set and was pruned — a silent miscompile. The
/// std-surface UFCS bricks lower `v.push(x)` into exactly such a `Call`, so a
/// pruned argument drops a real side-effecting/identity operand.
#[test]
fn canonicalization_keeps_value_used_only_as_call_arg() {
    let mut module = IRModule::new();
    // %0 is referenced ONLY as the argument to the call below.
    let arg = scalar_const(&mut module, 42);
    let call_dst = module.fresh();
    module.instrs.push(Instr::Call {
        dst: call_dst,
        name: "sink".to_string(),
        args: vec![arg],
    });
    module.instrs.push(Instr::Output(call_dst));
    module.next_id = module.next_id.max(call_dst.0 + 1);

    canonicalize_module(&mut module);

    let arg_kept = module
        .instrs
        .iter()
        .any(|i| matches!(i, Instr::ConstI64(id, 42) if *id == arg));
    assert!(
        arg_kept,
        "value used only as a Call argument was pruned: {}",
        format_ir_module(&module)
    );
    // And the call still carries the argument.
    let call_keeps_arg = module.instrs.iter().any(|i| {
        matches!(i, Instr::Call { args, name, .. }
            if name == "sink" && args.first() == Some(&arg))
    });
    assert!(
        call_keeps_arg,
        "Call lost its argument after canonicalization: {}",
        format_ir_module(&module)
    );
}

// --- P1-A: IR-canonical const-fold must be EXACT-OR-SKIP, never saturate (live-miscompile) ---

#[test]
fn overflowing_const_binop_is_not_saturate_folded() {
    // i64::MAX + 1 in const position previously SATURATED to i64::MAX, silently disagreeing
    // with the runtime's two's-complement wrap to i64::MIN. It must now stay a BinOp so the
    // runtime computes the (wrapping) value — exact-or-skip.
    let mut module = IRModule::new();
    let a = scalar_const(&mut module, i64::MAX);
    let b = scalar_const(&mut module, 1);
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Add,
        lhs: a,
        rhs: b,
    });
    module.instrs.push(Instr::Output(dst));
    canonicalize_module(&mut module);
    assert!(
        module
            .instrs
            .iter()
            .any(|i| matches!(i, Instr::BinOp { op: BinOp::Add, .. })),
        "overflowing const Add must remain a BinOp, not saturate-fold to i64::MAX"
    );
}

#[test]
fn nonoverflowing_const_binop_still_folds() {
    // No behavior change for representable results: 2 + 3 folds away as before.
    let mut module = IRModule::new();
    let a = scalar_const(&mut module, 2);
    let b = scalar_const(&mut module, 3);
    let dst = module.fresh();
    module.instrs.push(Instr::BinOp {
        dst,
        op: BinOp::Add,
        lhs: a,
        rhs: b,
    });
    module.instrs.push(Instr::Output(dst));
    canonicalize_module(&mut module);
    assert!(
        !module
            .instrs
            .iter()
            .any(|i| matches!(i, Instr::BinOp { op: BinOp::Add, .. })),
        "representable const Add (2+3) must still fold away"
    );
}
