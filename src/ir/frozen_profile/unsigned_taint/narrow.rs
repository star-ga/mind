// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
//! NARROW-width values for the frozen native profile: which values the MLIR backend
//! computes at 32 bits (`u32` / `i32`), 16/8 bits (`u16` / `u8`, masked after every op
//! by the Rust lowering's width masks) or 1 bit (`bool`) while the frozen native ELF —
//! a separate front end compiling the SOURCE — computes everything at 64 bits.
//!
//! Audit 2026-09-18: `fn f(a: u32, b: u32) -> i64 { let s = a + b  return s / 2 }` with
//! `f(4294967295, 1)` — MLIR adds in i32 (wraps to 0), native adds in i64; MLIR also
//! TRUNCATES a `u32` argument at the call site (`f(-1)` arrives as 4294967295) where
//! native passes it raw. So a narrow operand of `/` or `%` can always diverge (every
//! such division is refused — main refused every native division, nothing is lost),
//! and a compare diverges when a narrow operand may have WRAPPED (the result of narrow
//! arithmetic) or meets a literal MLIR truncates (outside the kind's range). A wrapped
//! value may not ESCAPE either — returned, passed to a call, used as an array index, or
//! widened into an i64 operation or join — because MLIR carries the wrapped value and
//! native the 64-bit one (`u32 + u32` returned as i64: main native 0 vs MLIR 1 at
//! 4294967295 + 1, audit 2026-09-18 round 6).
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
    U16,
    U8,
    Bool,
}

impl NarrowKind {
    pub(crate) fn of(ty: &TypeAnn) -> Option<Self> {
        match ty {
            TypeAnn::ScalarU32 => Some(Self::U32),
            TypeAnn::ScalarI32 => Some(Self::I32),
            TypeAnn::Named(n) if n == "u16" => Some(Self::U16),
            TypeAnn::Named(n) if n == "u8" => Some(Self::U8),
            TypeAnn::ScalarBool => Some(Self::Bool),
            _ => None,
        }
    }

    /// Is `v` representable exactly after MLIR truncates / masks it to this width?
    pub(crate) fn holds(self, v: i64) -> bool {
        match self {
            Self::U32 => (0..1 << 32).contains(&v),
            Self::I32 => (-(1 << 31)..1 << 31).contains(&v),
            Self::U16 => (0..1 << 16).contains(&v),
            Self::U8 => (0..1 << 8).contains(&v),
            Self::Bool => (0..2).contains(&v),
        }
    }

    /// Same physical width as `other` (MLIR joins `i32 ⊔ u32` to the unsigned kind).
    fn same_width(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::U32, Self::I32) | (Self::I32, Self::U32)
        )
    }
}

/// The narrow values of one body.
#[derive(Default)]
pub(crate) struct Narrow {
    /// Value -> its narrow kind (absent = MLIR kinds it i64/f64/tensor).
    pub(crate) kind: BTreeMap<ValueId, NarrowKind>,
    /// Narrow values that are the RESULT of narrow arithmetic (may have wrapped).
    pub(crate) wrapped: BTreeSet<ValueId>,
    /// `x - y` of two UNWRAPPED same-kind narrows: |x - y| < 2^width, so the wrapped
    /// difference is 0 iff x == y — `== 0` / `!= 0` on it never diverges (round 6).
    pub(crate) diff: BTreeSet<ValueId>,
    /// Callee -> the narrow kind of each formal, for the call-site literal check.
    pub(crate) formals: BTreeMap<String, Vec<Option<NarrowKind>>>,
    /// This function's declared narrow return kind, if any.
    pub(crate) ret_kind: Option<NarrowKind>,
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
        ret_kind: Option<NarrowKind>,
        consts: &BTreeMap<ValueId, i64>,
        body: &[Instr],
    ) -> Self {
        let mut n = Self {
            formals: formals.clone(),
            ret_kind,
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
                        let operands = [*lhs, *rhs];
                        let literal_in_range =
                            |id: &ValueId| consts.get(id).is_none_or(|v| k.holds(*v));
                        let all_unwrapped = operands.iter().all(|id| !self.wrapped.contains(id));
                        let all_narrow = operands.iter().all(|id| self.kind.contains_key(id));
                        // `& | ^` cannot leave the kind's range (with an in-range
                        // literal), so MLIR's masked result equals native's.
                        let bitwise = matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor);
                        if bitwise && operands.iter().all(literal_in_range) {
                            if !all_unwrapped {
                                grew |= self.wrapped.insert(*dst);
                            }
                        } else {
                            grew |= self.wrapped.insert(*dst);
                            if matches!(op, BinOp::Sub) && all_unwrapped && all_narrow {
                                grew |= self.diff.insert(*dst);
                            }
                        }
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
                    // Header and exit block args are typed by the INIT kind. A narrow
                    // loop counter (`i = a; while i < n { i = i + 1 }`) is NOT marked
                    // wrapped from its post-body value: main compiles it correctly, and
                    // MLIR cannot build a narrow-seeded loop at all ('i64' vs 'i32'
                    // block-arg type error), so there is no measured divergence to refuse
                    // (round 6). deferred: if MLIR ever builds such loops, an unbounded
                    // narrow counter wraps there and not natively — revisit then.
                    for (k, _) in live_vars.iter().enumerate() {
                        let Some(init) = init_ids.get(k).copied() else {
                            continue;
                        };
                        let Some(kind) = self.kind.get(&init).copied() else {
                            continue;
                        };
                        if let Some(exit) = exit_ids.get(k).copied() {
                            grew |= self.kind.insert(exit, kind).is_none();
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

    /// An `if` join stays narrow when both arms are narrow with the same kind. MLIR
    /// absorbs a same-WIDTH `i32 ⊔ u32` join to `u32` (signedness absorption, and then
    /// compares it `ult`), so that join is narrow AND treated as wrapped: a negative
    /// i32 arm compared as u32 diverges from native's signed compare (round 6).
    fn join(&mut self, dst: ValueId, arms: &[ValueId; 2]) -> bool {
        let (Some(a), Some(b)) = (self.kind.get(&arms[0]), self.kind.get(&arms[1])) else {
            return false;
        };
        let (a, b) = (*a, *b);
        let kind = if a == b {
            a
        } else if a.same_width(b) {
            NarrowKind::U32
        } else {
            return false;
        };
        let mut grew = self.kind.insert(dst, kind).is_none();
        if a != b || arms.iter().any(|v| self.wrapped.contains(v)) {
            grew |= self.wrapped.insert(dst);
        }
        grew
    }

    /// Does a WRAPPED narrow value escape the narrow world in `instr`: returned, passed
    /// as a call argument, used as an array index, widened into a non-narrow binop, or
    /// joined with a non-narrow arm? MLIR carries the wrapped value on, native the
    /// 64-bit one (`u32 + u32` returned as i64: native 0 vs MLIR 1 at 2^32-1 + 1;
    /// `bool + bool` returned: native 2 vs MLIR 0 — round 6).
    pub(crate) fn wrapped_escapes(&self, instr: &Instr, consts: &BTreeMap<ValueId, i64>) -> bool {
        let wrapped = |id: &ValueId| self.wrapped.contains(id);
        match instr {
            // Native TRUNCATES a `-> u32` / `-> i32` return (measured, round 6), so a
            // wrapped value of exactly that kind leaves through it intact. `-> bool` is
            // not truncated natively (`bool + bool` returned: 2 vs 0), nor is `-> i64`.
            Instr::Return { value: Some(v) } => {
                wrapped(v)
                    && !(matches!(self.ret_kind, Some(NarrowKind::U32 | NarrowKind::I32))
                        && self.kind.get(v).copied() == self.ret_kind)
            }
            Instr::Call { args, .. } => args.iter().any(wrapped),
            Instr::ArrayLoad { index, .. } => wrapped(index),
            // A wrapped bool as a CONDITION: `true + true` is 0 on MLIR (i1) and 2
            // natively, so `if s { .. }` takes different branches.
            Instr::While { cond_id, .. } => wrapped(cond_id),
            Instr::BinOp { op, lhs, rhs, .. } if !is_compare(op) => {
                self.narrow_mode(&[*lhs, *rhs], consts).is_none()
                    && [lhs, rhs].into_iter().any(wrapped)
            }
            Instr::If {
                cond_id,
                then_result,
                else_result,
                dst,
                merges,
                ..
            } => {
                let leaks = |d: &ValueId, arms: [&ValueId; 2]| {
                    !self.kind.contains_key(d) && arms.into_iter().any(wrapped)
                };
                wrapped(cond_id)
                    || leaks(dst, [then_result, else_result])
                    || merges.iter().any(|(m, a, b)| leaks(m, [a, b]))
            }
            _ => false,
        }
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
    /// wrapped (except `x - y == 0` on unwrapped operands, see `diff`), or a narrow
    /// operand beside a literal outside its kind's range (MLIR truncates the literal:
    /// `a == 4294967296` is `a == 0` there). Ordered compares on `bool`: MLIR compares
    /// i1 SIGNED (true is -1 there: `false < true` is 0 on MLIR, 1 natively).
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
        let ordered = matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge);
        if ordered && kinds.iter().flatten().any(|k| *k == NarrowKind::Bool) {
            return true;
        }
        let eq_zero_of_diff = matches!(op, BinOp::Eq | BinOp::Ne)
            && ids.iter().any(|id| self.diff.contains(id))
            && ids.iter().any(|id| consts.get(id) == Some(&0));
        (!eq_zero_of_diff && ids.iter().any(|id| self.wrapped.contains(id)))
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
