// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Call-site checks that run with module verification.
//!
//! A call to a registered `__mind_*` intrinsic carries exactly the intrinsic's
//! declared arity, at any nesting depth. The C runtime reads that many
//! arguments, so a short call hands it an undefined trailing argument.

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
