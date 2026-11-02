// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Call-site checks that run with module verification.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity, at any nesting depth. The C runtime reads that many
//! arguments, so a short call hands it an undefined trailing argument.
//!
//! The rule runs here, not in the type checker: `mindc check` does not
//! type-check function bodies, so a check there would miss these calls. Every
//! `mindc` path that lowers through MLIR (`build`, `run`, and the `--emit-ir`,
//! `--emit-mic`, `--emit-mic3`, `--emit-mlir` and `--emit-obj` flags) verifies
//! the module first. Two paths do not reach this verifier and are fenced
//! elsewhere: `--backend native` refuses every intrinsic call at closure
//! admission, and `mind eval` emits from unverified lowered IR (its own deferred
//! marker in `src/main.rs`) but no intrinsic call reaches its output today: a
//! top-level call is a type error and function definitions are not emitted.

// deferred: would have `mindc check` type-check function bodies (this rule and every other
// body error), not done here because that changes what `check` reports on every existing
// program — upgrade path: a separate change that type-checks bodies under `check`, measured
// by a sweep of std/, examples/ and tests/.

use super::IRModule;
#[cfg(feature = "std-surface")]
use super::Instr;
use super::verify::IrVerifyError;

/// Verify `module`: the intrinsic call arity, then every structural rule of the
/// module verifier (SSA definition order, operands, output).
pub fn verify_module(module: &IRModule) -> Result<(), IrVerifyError> {
    #[cfg(feature = "std-surface")]
    validate_intrinsic_arity(&module.instrs)?;
    super::verify::verify_module(module)
}

/// The declared i64 arity of a registered intrinsic in any profile, or `None`
/// if `name` is not in the table. `std_surface_intrinsic_arity` answers only
/// for the Rust/MLIR profile; a call site must match in every profile.
#[cfg(feature = "std-surface")]
fn declared_intrinsic_arity(name: &str) -> Option<usize> {
    crate::intrinsics::STD_SURFACE_INTRINSICS
        .iter()
        .find(|spec| spec.physical_name == name)
        .map(|spec| spec.arity)
}

#[cfg(feature = "std-surface")]
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
