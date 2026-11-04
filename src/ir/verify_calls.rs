// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Call-site checks that run with module verification.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity, at any nesting depth. The C runtime reads that many
//! arguments, so a short call hands it an undefined trailing argument.
//!
//! The type checker reports the same rule as E2005 where its body pass visits
//! the call (function and `for` bodies, match arms, `let` initializers, return
//! expressions). It does not visit every shape: `while` bodies and conditions
//! (RFC 0005 Gap 1) and `else if` branches are skipped
//! (`tests/intrinsic_call_arity_build.rs` pins each through the build, where
//! only this verifier refuses them). This verifier runs on
//! every `mindc` path that lowers through MLIR (`build`, `run`, and the
//! `--emit-ir`, `--emit-mic`, `--emit-mic3`, `--emit-mlir` and `--emit-obj`
//! flags) and refuses the call at any depth (E3001), so no wrong-arity call
//! builds. Two paths do not reach this verifier and are fenced
//! elsewhere: `--backend native` refuses every intrinsic call at closure
//! admission, and `mind eval` emits from unverified lowered IR (its own deferred
//! marker in `src/main.rs`) but no intrinsic call reaches its output today: a
//! top-level call is a type error and function definitions are not emitted.
//! The check runs in every build, with or without the `std-surface` feature:
//! the intrinsic table is not feature-gated, so neither is the refusal.

use super::IRModule;
use super::Instr;
use super::verify::IrVerifyError;

/// Verify `module`: the intrinsic call arity, then every structural rule of the
/// module verifier (SSA definition order, operands, output).
pub fn verify_module(module: &IRModule) -> Result<(), IrVerifyError> {
    validate_intrinsic_arity(&module.instrs)?;
    super::verify::verify_module(module)
}

/// The declared i64 arity of a registered intrinsic in any profile, or `None`
/// if `name` is not in the table. `std_surface_intrinsic_arity` answers only
/// for the Rust/MLIR profile; a call site must match in every profile.
fn declared_intrinsic_arity(name: &str) -> Option<usize> {
    crate::intrinsics::STD_SURFACE_INTRINSICS
        .iter()
        .find(|spec| spec.physical_name == name)
        .map(|spec| spec.arity)
}

fn validate_intrinsic_arity(instrs: &[Instr]) -> Result<(), IrVerifyError> {
    for (instr_index, instr) in instrs.iter().enumerate() {
        if let Instr::Call { name, args, .. } = instr
            && name.starts_with("__mind_")
            && let Some(arity) = declared_intrinsic_arity(name)
            && args.len() != arity
        {
            return Err(IrVerifyError::InvalidOperand {
                instr_index,
                message: format!(
                    "intrinsic `{name}` expects {arity} i64 argument(s) (i64 ABI — RFC 0005); got {}",
                    args.len()
                ),
            });
        }
        // `instr_bodies` is the one list of body-carrying variants (function
        // bodies, while, if, region), so a new variant cannot be missed here.
        for body in super::instr_bodies(instr) {
            validate_intrinsic_arity(body)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ValueId;

    /// `n` constants and an `n`-argument `__mind_write`, at top level or inside
    /// a function body: the two shapes every feature set can build.
    fn write_call(n: usize, in_fn: bool) -> IRModule {
        let mut calls: Vec<Instr> = (0..n).map(|i| Instr::ConstI64(ValueId(i), 1)).collect();
        calls.push(Instr::Call {
            dst: ValueId(n),
            name: "__mind_write".to_string(),
            args: (0..n).map(ValueId).collect(),
            resolved_callee: None,
        });
        let mut m = IRModule::new();
        if in_fn {
            calls.push(Instr::Return {
                value: Some(ValueId(n)),
            });
            m.instrs.push(Instr::FnDef {
                name: "main".to_string(),
                params: vec![],
                ret_id: Some(ValueId(n)),
                body: calls,
                reap_threshold: None,
                semantic_types: None,
                #[cfg(feature = "std-surface")]
                value_types: std::collections::BTreeMap::new(),
            });
            m.instrs.push(Instr::ConstI64(ValueId(n + 1), 0));
            m.instrs.push(Instr::Output(ValueId(n + 1)));
            m.next_id = n + 2;
        } else {
            m.instrs.extend(calls);
            m.instrs.push(Instr::Output(ValueId(n)));
            m.next_id = n + 1;
        }
        m
    }

    #[test]
    fn a_wrong_arity_intrinsic_call_is_refused_in_every_build() {
        for in_fn in [false, true] {
            for n in [3, 5] {
                let err = verify_module(&write_call(n, in_fn))
                    .expect_err("a wrong-arity __mind_write must not verify");
                assert!(
                    err.to_string()
                        .contains("intrinsic `__mind_write` expects 4 i64 argument(s)"),
                    "{err}"
                );
            }
            verify_module(&write_call(4, in_fn)).expect("a 4-argument __mind_write verifies");
        }
    }
}
