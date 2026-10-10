// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
//! Operand types for wave-three admission. Unknown is never an integer proof.
use crate::ast::TypeAnn;
use crate::ir::{BinOp, IRModule, Instr, ValueId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scalar {
    Signed,
    Unsigned,
    Float,
}

pub(crate) fn declared(ty: &TypeAnn) -> Option<Scalar> {
    match ty {
        TypeAnn::ScalarI64 => Some(Scalar::Signed),
        TypeAnn::ScalarU32 => Some(Scalar::Unsigned),
        TypeAnn::ScalarF64 => Some(Scalar::Float),
        TypeAnn::Named(n) => match n.as_str() {
            // The current lowering spells both pointer-width aliases as i64.
            "i64" | "isize" | "usize" => Some(Scalar::Signed),
            "u8" | "u16" | "u32" | "u64" => Some(Scalar::Unsigned),
            "f64" => Some(Scalar::Float),
            _ => None,
        },
        _ => None,
    }
}

/// Census conversion names, plus the unsigned tag needed by unsigned operands.
/// This is separate from the pinned old seed's numeric intrinsic-id table.
pub(crate) fn conversion(name: &str) -> Option<Scalar> {
    match name {
        "__mind_conv_i32" | "__mind_conv_i64" => Some(Scalar::Signed),
        "__mind_conv_u64" => Some(Scalar::Unsigned),
        "__mind_conv_f64" => Some(Scalar::Float),
        _ => None,
    }
}

pub(crate) fn types(
    module: &IRModule,
    name: Option<&str>,
    body: &[Instr],
) -> BTreeMap<ValueId, Scalar> {
    let mut map = BTreeMap::new();
    // Each iteration only adds facts; cyclic values without a typed incoming
    // definition remain unknown. FnDef bodies have independent SSA namespaces.
    loop {
        let before = map.len();
        infer(module, name, body, &mut map);
        if before == map.len() {
            break;
        }
    }
    map
}

fn infer(
    module: &IRModule,
    name: Option<&str>,
    body: &[Instr],
    map: &mut BTreeMap<ValueId, Scalar>,
) {
    for instr in body {
        let ty = match instr {
            Instr::ConstI64(..) => Some(Scalar::Signed),
            Instr::ConstF64(..) => Some(Scalar::Float),
            Instr::Param { index, .. } => parameter(module, name, *index),
            Instr::Call { name, .. } => conversion(name)
                .or_else(|| {
                    crate::ir::frozen_intrinsics::frozen_native_intrinsic(name)
                        .filter(|r| r.result == crate::ir::frozen_intrinsics::FrozenResult::Value)
                        .map(|_| Scalar::Signed)
                })
                .or_else(|| returned(module, name)),
            #[cfg(feature = "std-surface")]
            Instr::BinOp {
                op: BinOp::Shl | BinOp::Shr,
                lhs,
                rhs,
                ..
            } if matches!(map.get(rhs), Some(Scalar::Signed | Scalar::Unsigned)) => {
                map.get(lhs).copied()
            }
            Instr::BinOp { op, lhs, rhs, .. }
                if !matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) =>
            {
                same(map, *lhs, *rhs)
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                then_result,
                else_result,
                ..
            } => same(map, *then_result, *else_result),
            _ => None,
        };
        if let (Some(id), Some(ty)) = (crate::ir::instruction_dst(instr), ty) {
            map.entry(id).or_insert(ty);
        }
        if matches!(instr, Instr::FnDef { .. }) {
            continue;
        }
        #[cfg(feature = "std-surface")]
        {
            for nested in crate::ir::instr_bodies(instr) {
                infer(module, name, nested, map);
            }
            if let Instr::If { merges, .. } = instr {
                for (dst, a, b) in merges {
                    if let Some(ty) = same(map, *a, *b) {
                        map.entry(*dst).or_insert(ty);
                    }
                }
            }
            if let Instr::While {
                live_vars,
                init_ids,
                exit_ids,
                ..
            } = instr
            {
                for (((_, back), initial), exit) in live_vars.iter().zip(init_ids).zip(exit_ids) {
                    if let Some(ty) = same(map, *back, *initial) {
                        map.entry(*exit).or_insert(ty);
                    }
                }
            }
        }
    }
}

fn same(map: &BTreeMap<ValueId, Scalar>, a: ValueId, b: ValueId) -> Option<Scalar> {
    let a = *map.get(&a)?;
    (map.get(&b) == Some(&a)).then_some(a)
}

#[cfg(feature = "std-surface")]
fn parameter(module: &IRModule, name: Option<&str>, index: usize) -> Option<Scalar> {
    declared(module.fn_signatures.get(name?)?.0.get(index)?)
}
#[cfg(not(feature = "std-surface"))]
fn parameter(_: &IRModule, _: Option<&str>, _: usize) -> Option<Scalar> {
    None
}
#[cfg(feature = "std-surface")]
fn returned(module: &IRModule, name: &str) -> Option<Scalar> {
    if let Some((_, ret)) = module.fn_signatures.get(name) {
        return ret.as_ref().and_then(declared);
    }
    // Extern declarations are represented separately from local/bundled std
    // signatures. Closure admission has already required an exact POSIX match;
    // the whole-program proof still checks the linked body's actual returns.
    module.instrs.iter().find_map(|instr| match instr {
        #[cfg(feature = "std-surface")]
        Instr::ExternFnDecl {
            name: callee,
            ret_type,
            is_varargs: false,
            ..
        } if callee == name => match ret_type.as_deref() {
            Some("i64") => Some(Scalar::Signed),
            Some("f64") => Some(Scalar::Float),
            _ => None,
        },
        _ => None,
    })
}
#[cfg(not(feature = "std-surface"))]
fn returned(_: &IRModule, _: &str) -> Option<Scalar> {
    None
}
