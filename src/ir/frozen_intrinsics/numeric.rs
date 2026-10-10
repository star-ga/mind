// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
//! Whole-program proof for newly admitted numeric operations and conversions.
use super::program::{Key, Program};
use super::{reject, visit_fn_instrs};
use crate::ir::frozen_profile::FrozenProfileRejection;
use crate::ir::frozen_profile::scalar::{self as native_scalar, Scalar};
use crate::ir::{BinOp, Instr};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn admit(prog: &Program<'_>) -> Result<(), FrozenProfileRejection> {
    let maps: Vec<_> = prog
        .scopes
        .iter()
        .map(|s| native_scalar::types(s.module, s.func.map(|f| f.name), s.body))
        .collect();
    let kind = |k: Key| maps[k.0].get(&k.1).copied();
    let mut sites: BTreeMap<usize, Vec<(usize, Vec<crate::ir::ValueId>)>> = BTreeMap::new();
    let mut returns: BTreeMap<usize, Vec<Option<crate::ir::ValueId>>> = BTreeMap::new();
    for (s, scope) in prog.scopes.iter().enumerate() {
        if let Some(r) = scope.func.and_then(|f| f.ret_id) {
            returns.entry(s).or_default().push(Some(r));
        }
        visit_fn_instrs(scope.body, &mut |i| match i {
            Instr::Call { name, args, .. } => {
                if let Some(c) = prog.callee(name) {
                    sites.entry(c).or_default().push((s, args.clone()));
                }
            }
            Instr::Return { value } => {
                returns.entry(s).or_default().push(*value);
            }
            _ => {}
        });
    }
    let mut needs: BTreeMap<Key, Vec<Key>> = BTreeMap::new();
    let mut invalid = BTreeSet::new();
    for (s, scope) in prog.scopes.iter().enumerate() {
        visit_fn_instrs(scope.body, &mut |i| {
            let Some(dst) = crate::ir::instruction_dst(i) else {
                return;
            };
            let key = (s, dst);
            let Some(ty) = kind(key) else { return };
            let mut deps = Vec::new();
            let valid = match i {
                Instr::ConstI64(..) => ty == Scalar::Signed,
                Instr::ConstF64(..) => ty == Scalar::Float,
                Instr::Param { index, .. } => scope.func.is_some_and(|f| {
                    prog.callee(f.name) == Some(s)
                        && sites.get(&s).into_iter().flatten().all(|(caller, args)| {
                            if args.len() != f.n_params {
                                return false;
                            }
                            let Some(arg) = args.get(*index) else {
                                return false;
                            };
                            let k = (*caller, *arg);
                            deps.push(k);
                            kind(k) == Some(ty)
                        })
                }),
                Instr::Call { name, args, .. } if native_scalar::conversion(name).is_some() => {
                    if args.len() == 1 && native_scalar::conversion(name) == Some(ty) {
                        deps.push((s, args[0]));
                        kind((s, args[0])).is_some()
                    } else {
                        false
                    }
                }
                Instr::Call { name, args, .. } => {
                    if super::frozen_native_intrinsic(name).is_some() {
                        true
                    } else {
                        prog.callee(name).is_some_and(|c| {
                            prog.scopes[c]
                                .func
                                .is_some_and(|f| f.n_params == args.len())
                                && returns.get(&c).is_some_and(|rs| {
                                    !rs.is_empty()
                                        && rs.iter().all(|r| {
                                            r.is_some_and(|r| {
                                                deps.push((c, r));
                                                kind((c, r)) == Some(ty)
                                            })
                                        })
                                })
                        })
                    }
                }
                Instr::BinOp { op, lhs, rhs, .. } => {
                    deps.extend([(s, *lhs), (s, *rhs)]);
                    #[cfg(feature = "std-surface")]
                    let shift = matches!(op, BinOp::Shl | BinOp::Shr);
                    #[cfg(not(feature = "std-surface"))]
                    let shift = {
                        let _ = op;
                        false
                    };
                    kind((s, *lhs)) == Some(ty)
                        && (kind((s, *rhs)) == Some(ty)
                            || (shift
                                && matches!(
                                    kind((s, *rhs)),
                                    Some(Scalar::Signed | Scalar::Unsigned)
                                )))
                }
                #[cfg(feature = "std-surface")]
                Instr::If {
                    then_result,
                    else_result,
                    ..
                } => {
                    deps.extend([(s, *then_result), (s, *else_result)]);
                    kind((s, *then_result)) == Some(ty) && kind((s, *else_result)) == Some(ty)
                }
                _ => false,
            };
            needs.entry(key).or_default().extend(deps);
            if !valid {
                invalid.insert(key);
            }
        });
        // Region identities are additional definitions, not independent proofs.
        visit_fn_instrs(scope.body, &mut |i| {
            for (dst, dependencies) in super::operand_types::region_facts(i) {
                let key = (s, dst);
                let Some(ty) = kind(key) else { continue };
                if let Some(ds) = dependencies {
                    for d in ds {
                        needs.entry(key).or_default().push((s, d));
                        if kind((s, d)) != Some(ty) {
                            invalid.insert(key);
                        }
                    }
                } else {
                    invalid.insert(key);
                }
            }
        });
    }
    let mut proven: BTreeSet<Key> = needs
        .keys()
        .filter(|k| !invalid.contains(k))
        .copied()
        .collect();
    loop {
        let before = proven.len();
        let lost: Vec<_> = proven
            .iter()
            .filter(|k| needs[*k].iter().any(|d| !proven.contains(d)))
            .copied()
            .collect();
        for k in lost {
            proven.remove(&k);
        }
        if before == proven.len() {
            break;
        }
    }
    for (s, scope) in prog.scopes.iter().enumerate() {
        let mut error = None;
        visit_fn_instrs(scope.body, &mut |i| {
            if error.is_some() {
                return;
            }
            match i {
                Instr::Call { name, args, .. } if native_scalar::conversion(name).is_some() => {
                    if args.len() != 1 || !proven.contains(&(s, args[0])) {
                        error = Some("call.conversion_operand_type");
                    }
                }
                Instr::BinOp { op, lhs, rhs, .. } => {
                    let label = match op {
                        BinOp::Div => Some("binop.div_operand_type"),
                        BinOp::Mod => Some("binop.mod_operand_type"),
                        #[cfg(feature = "std-surface")]
                        BinOp::Shl => Some("binop.shl_operand_type"),
                        #[cfg(feature = "std-surface")]
                        BinOp::Shr => Some("binop.shr_operand_type"),
                        _ => None,
                    };
                    if label.is_some()
                        && (!proven.contains(&(s, *lhs))
                            || !proven.contains(&(s, *rhs))
                            || !matches!(kind((s, *lhs)), Some(Scalar::Signed | Scalar::Unsigned))
                            || !matches!(kind((s, *rhs)), Some(Scalar::Signed | Scalar::Unsigned)))
                    {
                        error = label;
                    }
                }
                _ => {}
            }
        });
        if let Some(e) = error {
            return reject(e);
        }
    }
    Ok(())
}
