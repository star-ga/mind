// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! NARROW-width values for the frozen native profile: which values the MLIR backend
//! computes at 32 bits (`u32` / `i32`) or 1 bit (`bool`) while the frozen native ELF
//! computes everything at 64 bits.
//!
//! Audit 2026-09-18: `fn f(a: u32, b: u32) -> i64 { let s = a + b  return s / 2 }` with
//! `f(4294967295, 1)` — MLIR adds in i32 (wraps to 0), native adds in i64; MLIR also
//! TRUNCATES a `u32` argument at the call site (`f(-1)` arrives as 4294967295) where
//! native passes it raw. So a narrow operand of `/` or `%` can always diverge (every
//! such division is refused — main refused every native division, nothing is lost),
//! and a compare diverges when a narrow operand may have WRAPPED (the result of narrow
//! arithmetic) or meets a literal MLIR truncates (outside the kind's range).
//!
//! The kind tracking mirrors `src/mlir/lowering.rs` (narrow-mode vs widen-mode): a
//! binop stays narrow only when every non-literal operand has the SAME narrow kind
//! (`u32 + u32`, `u32 + literal`); a `u32` beside a non-literal `i64`, an `i32`, or an
//! `i64` merge arm / loop init WIDENS to a signed i64 and is not narrow. Refusing those
//! widened compares lost programs main compiles correctly (audit round 5, seven
//! measured).

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::TypeAnn;
use crate::ir::{BinOp, Instr, ValueId};

/// A 32- or 1-bit kind, with the literal range MLIR represents exactly in it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NarrowKind {
    U32,
    I32,
    Bool,
}

impl NarrowKind {
    pub(crate) fn of(ty: &TypeAnn) -> Option<Self> {
        match ty {
            TypeAnn::ScalarU32 => Some(Self::U32),
            TypeAnn::ScalarI32 => Some(Self::I32),
            TypeAnn::ScalarBool => Some(Self::Bool),
            _ => None,
        }
    }

    /// Is `v` representable exactly after MLIR truncates it to this width?
    pub(crate) fn holds(self, v: i64) -> bool {
        match self {
            Self::U32 => (0..1 << 32).contains(&v),
            Self::I32 => (-(1 << 31)..1 << 31).contains(&v),
            Self::Bool => (0..2).contains(&v),
        }
    }
}

/// The narrow values of one body.
#[derive(Default)]
pub(crate) struct Narrow {
    /// Value -> its narrow kind (absent = MLIR kinds it i64/f64/tensor).
    pub(crate) kind: BTreeMap<ValueId, NarrowKind>,
    /// Narrow values that are the RESULT of narrow arithmetic (may have wrapped).
    pub(crate) wrapped: BTreeSet<ValueId>,
    /// Callee -> the narrow kind of each formal, for the call-site literal check.
    pub(crate) formals: BTreeMap<String, Vec<Option<NarrowKind>>>,
}

impl Narrow {
    /// Compute the narrow values of `body`, seeded by `params` (index -> kind of that
    /// parameter, `None` = unknown signature: treated as not narrow, the Exact/Wide
    /// sets already refuse such bodies' unsigned operators) and `rets` (callee -> kind).
    pub(crate) fn of_body(
        fn_params: &[(String, ValueId)],
        param_kind: &dyn Fn(usize) -> Option<NarrowKind>,
        rets: &BTreeMap<String, NarrowKind>,
        formals: &BTreeMap<String, Vec<Option<NarrowKind>>>,
        consts: &BTreeMap<ValueId, i64>,
        body: &[Instr],
    ) -> Self {
        let mut n = Self {
            formals: formals.clone(),
            ..Self::default()
        };
        for (index, (_, id)) in fn_params.iter().enumerate() {
            if let Some(k) = param_kind(index) {
                n.kind.insert(*id, k);
            }
        }
        while n.pass(body, rets, consts) {}
        n
    }

    fn pass(
        &mut self,
        body: &[Instr],
        rets: &BTreeMap<String, NarrowKind>,
        consts: &BTreeMap<ValueId, i64>,
    ) -> bool {
        let mut grew = false;
        for instr in body {
            match instr {
                Instr::Call { dst, name, .. } => {
                    if let Some(k) = rets.get(name) {
                        grew |= self.kind.insert(*dst, *k).is_none();
                    }
                }
                Instr::BinOp { dst, op, lhs, rhs } if !is_compare(op) => {
                    if let Some(k) = self.narrow_mode(&[*lhs, *rhs], consts) {
                        grew |= self.kind.insert(*dst, k).is_none();
                        grew |= self.wrapped.insert(*dst);
                    }
                }
                Instr::If {
                    cond_instrs,
                    then_instrs,
                    else_instrs,
                    then_result,
                    else_result,
                    dst,
                    merges,
                    ..
                } => {
                    for nested in [cond_instrs, then_instrs, else_instrs] {
                        grew |= self.pass(nested, rets, consts);
                    }
                    grew |= self.join(*dst, &[*then_result, *else_result]);
                    for (merge, then_val, else_val) in merges {
                        grew |= self.join(*merge, &[*then_val, *else_val]);
                    }
                }
                Instr::While {
                    cond_instrs,
                    body: loop_body,
                    live_vars,
                    init_ids,
                    exit_ids,
                    ..
                } => {
                    for nested in [cond_instrs, loop_body] {
                        grew |= self.pass(nested, rets, consts);
                    }
                    // Header and exit block args are typed by the INIT kind; a narrow
                    // loop variable whose post-body value may have wrapped carries that
                    // into the header argument (its init id) and the exit id.
                    for (k, (_, post)) in live_vars.iter().enumerate() {
                        let Some(init) = init_ids.get(k).copied() else {
                            continue;
                        };
                        let Some(kind) = self.kind.get(&init).copied() else {
                            continue;
                        };
                        if let Some(exit) = exit_ids.get(k).copied() {
                            grew |= self.kind.insert(exit, kind).is_none();
                        }
                        if self.wrapped.contains(post) {
                            grew |= self.wrapped.insert(init);
                            if let Some(exit) = exit_ids.get(k).copied() {
                                grew |= self.wrapped.insert(exit);
                            }
                        }
                    }
                }
                Instr::FnDef { .. } => {}
                _ => {}
            }
        }
        grew
    }

    /// The kind MLIR computes a binop in, if it stays narrow: at least one operand is
    /// narrow and every non-literal operand has that same kind (a literal is truncated
    /// to it). Mixed kinds and a non-literal i64 widen to i64 (`None`).
    fn narrow_mode(&self, ids: &[ValueId], consts: &BTreeMap<ValueId, i64>) -> Option<NarrowKind> {
        let mut kind = None;
        for id in ids {
            match self.kind.get(id) {
                Some(k) => match kind {
                    None => kind = Some(*k),
                    Some(prev) if prev == *k => {}
                    Some(_) => return None,
                },
                None if consts.contains_key(id) => {}
                None => return None,
            }
        }
        kind
    }

    /// An `if` join stays narrow only when both arms are narrow with the same kind.
    fn join(&mut self, dst: ValueId, arms: &[ValueId; 2]) -> bool {
        let (Some(a), Some(b)) = (self.kind.get(&arms[0]), self.kind.get(&arms[1])) else {
            return false;
        };
        if a != b {
            return false;
        }
        let kind = *a;
        let mut grew = self.kind.insert(dst, kind).is_none();
        if arms.iter().any(|v| self.wrapped.contains(v)) {
            grew |= self.wrapped.insert(dst);
        }
        grew
    }

    /// Is a LITERAL argument of `callee` outside the range of its narrow formal? MLIR
    /// truncates it at the call site (`f(-1)` into `u32` arrives as 4294967295); the
    /// frozen ELF passes it raw. A computed out-of-range argument is not visible here —
    /// that is the front end's job (it admits any i64 into a `u32` formal today).
    pub(crate) fn arg_out_of_range(
        &self,
        callee: &str,
        args: &[ValueId],
        consts: &BTreeMap<ValueId, i64>,
    ) -> bool {
        let Some(kinds) = self.formals.get(callee) else {
            return false;
        };
        args.iter().enumerate().any(|(i, arg)| {
            let (Some(Some(kind)), Some(v)) = (kinds.get(i), consts.get(arg)) else {
                return false;
            };
            !kind.holds(*v)
        })
    }

    /// May `op` over `ids` diverge because of a narrow operand?
    ///
    /// `/` and `%`: any narrow operand (32-bit vs 64-bit arithmetic, and the untruncated
    /// call boundary). Compares (ordered AND `==`/`!=`): a narrow operand that may have
    /// wrapped, or a narrow operand beside a literal outside its kind's range (MLIR
    /// truncates the literal: `a == 4294967296` is `a == 0` there).
    pub(crate) fn diverges(
        &self,
        op: &BinOp,
        ids: &[ValueId],
        consts: &BTreeMap<ValueId, i64>,
    ) -> bool {
        let kinds: Vec<Option<NarrowKind>> =
            ids.iter().map(|id| self.kind.get(id).copied()).collect();
        let Some(kind) = kinds.iter().flatten().next().copied() else {
            return false;
        };
        if matches!(op, BinOp::Div | BinOp::Mod) {
            return true;
        }
        ids.iter().any(|id| self.wrapped.contains(id))
            || ids
                .iter()
                .any(|id| consts.get(id).is_some_and(|v| !kind.holds(*v)))
    }
}

fn is_compare(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
    )
}
