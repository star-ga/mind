// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Intermediate narrow-arithmetic re-masking for the tree evaluator — the
//! oracle mirror of `lower.rs::infer_narrow_arith_ty` / its
//! `mask_narrow_binop_result` + the narrow-left shift-COUNT masking.
//!
//! # The fifth declared-width locus
//!
//! An 8/16-bit binary/shift RESULT re-wraps at a width the SOURCE never
//! declares, inferred from a 3-valued operand lattice (`NarrowLat`). The
//! compiled artifact narrows it — `let a: u8 = 250; (a + 10) / 2` is **2**, the
//! intermediate `a + 10` wrapping to 4 first — while the tree evaluator carried
//! every result full width and answered **130**. Because this evaluator is the
//! `mindc test` ORACLE, that silent wide answer graded correct code wrong. This
//! module closes the divergence by mirroring the lattice and reusing
//! `declared_width`'s ONE narrow-width table (`narrow_scalar_kind` /
//! `narrow_kind` / `apply_narrow_kind`) for the masking — no second width table.
//!
//! # Faithful to `lower.rs`
//!
//! * only the 8/16-bit widths re-mask (`lower.rs::is_named_narrow_sig_ty`); the
//!   32-bit ABI wraps intermediates by a different mechanism and is NOT
//!   re-masked here — every classification below filters to `8 | 16`;
//! * a `Wide` operand (a non-narrow `i64`/`u64` var, an `as i64`/`as i32` cast)
//!   widens the whole expression — the result is NOT re-masked (#329);
//! * an integer literal is width-NEUTRAL; an all-literal expression is `Wide`;
//! * `Add`/`Sub`/`Mul`/`Div`/`Mod` JOIN both operands (any `Wide` ⇒ `Wide`;
//!   two `Narrow`s ⇒ the WIDEST, equal keeps the LEFT); `Shl`/`Shr` take the
//!   LEFT operand's width; comparisons and `And`/`Or`/`Xor` never re-mask;
//! * a narrow-left SHIFT masks the COUNT to (width-1) BEFORE shifting
//!   (`lower.rs` Finding 3: `1u8 << 8` == `1 << (8 & 7)` == 1), then re-masks
//!   the result to that width.
//!
//! # Leaves classified from the runtime `env`
//!
//! `lower.rs` classifies Call / FieldAccess / IndexAccess leaves from static IR
//! context (`ir.fn_signatures`, `struct_field_types`, receiver types). The tree
//! evaluator has no such static context, but it DOES have the runtime `env` and
//! the sibling `declared_width` registries: a Call's return type from the
//! `FN_TABLE`, a `p.f` from the receiver's `Value::Struct` name in `env` plus
//! `declared_width`'s narrow-field schema, an `a[i]` element from — nothing,
//! since `Value::Tuple` carries no element type. Every unresolved leaf (a
//! non-ident receiver, an unannotated array element, an aggregate param) is
//! width-NEUTRAL. Neutral never OVER-narrows: it can only miss a
//! both-narrow-non-local case, the direction the evaluator already erred, so it
//! strictly reduces divergence. See `declared_width`'s "Not covered" section.

use std::collections::HashMap;

use super::{EvalError, Value};
use crate::ast::{BinOp, Node};

#[cfg(feature = "std-surface")]
use super::declared_width::{self, NarrowScalar};
#[cfg(feature = "std-surface")]
use crate::ast::{self, Literal, TypeAnn};

/// Re-mask an ARITHMETIC binop result (`Add`/`Sub`/`Mul`/`Div`/`Mod`) to its
/// inferred narrow width, exactly as the compiled artifact does. Comparisons
/// yield `i1`/bool and are never re-masked. A non-narrow (`Wide`/`Neutral`)
/// inference returns `value` untouched — the byte-identical all-i64 path.
#[cfg(feature = "std-surface")]
pub(crate) fn narrow_binary_result(
    op: BinOp,
    left: &Node,
    right: &Node,
    value: Value,
    env: &HashMap<String, Value>,
) -> Result<Value, EvalError> {
    let lat = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            join(infer(left, env), infer(right, env))
        }
        // Comparisons produce bool; never truncate them.
        _ => NarrowLat::Neutral,
    };
    apply_lat(lat, value)
}

/// Non-`std-surface` builds have no narrow scalar types and their arithmetic
/// intermediate masking is a no-op (`mask_narrow_binop_result` is `dst`
/// verbatim there), so the result is returned unchanged — byte-identical.
#[cfg(not(feature = "std-surface"))]
pub(crate) fn narrow_binary_result(
    _op: BinOp,
    _left: &Node,
    _right: &Node,
    value: Value,
    _env: &HashMap<String, Value>,
) -> Result<Value, EvalError> {
    Ok(value)
}

/// Mask a SHIFT COUNT to (left-operand width - 1) BEFORE the shift when the
/// LEFT operand is a narrow (8/16-bit) value — the artifact masks the count mod
/// the DECLARED width, so `1u8 << 8` shifts by `8 & 7 == 0` and stays 1. The
/// count IS masked here (this is not a post-shift result mask); a wide/neutral
/// left, or a non-shift op, returns `count` unchanged. Only reachable in
/// `std-surface` (the `Node::Bitwise` variant is itself `std-surface`-gated).
#[cfg(feature = "std-surface")]
pub(crate) fn narrow_shift_count(
    op: ast::BitOp,
    left: &Node,
    count: i64,
    env: &HashMap<String, Value>,
) -> i64 {
    match op {
        ast::BitOp::Shl | ast::BitOp::Shr => match infer(left, env) {
            NarrowLat::Narrow(kind) => count & ((width_of(kind) as i64) - 1),
            NarrowLat::Wide | NarrowLat::Neutral => count,
        },
        // The count concept applies only to shifts; `And`/`Or`/`Xor` use the
        // right operand as a VALUE, which must never be masked.
        ast::BitOp::And | ast::BitOp::Or | ast::BitOp::Xor => count,
    }
}

/// Re-mask a SHIFT result (`Shl`/`Shr`) to the LEFT operand's inferred narrow
/// width (Finding 3). `And`/`Or`/`Xor` keep in-range narrow operands in range
/// and are never re-masked. Only reachable in `std-surface`.
#[cfg(feature = "std-surface")]
pub(crate) fn narrow_shift_result(
    op: ast::BitOp,
    left: &Node,
    value: Value,
    env: &HashMap<String, Value>,
) -> Result<Value, EvalError> {
    let lat = match op {
        ast::BitOp::Shl | ast::BitOp::Shr => infer(left, env),
        ast::BitOp::And | ast::BitOp::Or | ast::BitOp::Xor => NarrowLat::Neutral,
    };
    apply_lat(lat, value)
}

/// The evaluator mirror of `lower.rs::NarrowLat`. Carries the resolved
/// `NarrowScalar` (width + signedness) rather than a `TypeAnn`, because the
/// only things done with it are a width JOIN and a mask.
#[cfg(feature = "std-surface")]
#[derive(Clone, Copy)]
enum NarrowLat {
    Narrow(NarrowScalar),
    Wide,
    Neutral,
}

/// Join two operand lattice values (`lower.rs::join_narrow_lat`): any `Wide` ⇒
/// `Wide`; two `Narrow`s ⇒ the WIDEST (equal width keeps the LEFT for
/// stability); `Narrow` + `Neutral` ⇒ that `Narrow`; all-`Neutral` ⇒ `Wide`.
#[cfg(feature = "std-surface")]
fn join(l: NarrowLat, r: NarrowLat) -> NarrowLat {
    use NarrowLat::*;
    match (l, r) {
        (Wide, _) | (_, Wide) => Wide,
        (Narrow(a), Narrow(b)) => {
            if width_of(b) > width_of(a) {
                Narrow(b)
            } else {
                Narrow(a)
            }
        }
        (Narrow(a), Neutral) | (Neutral, Narrow(a)) => Narrow(a),
        (Neutral, Neutral) => Wide,
    }
}

/// Classify an operand subtree (`lower.rs::infer_narrow_arith_ty`). A narrow
/// LOCAL ⇒ `Narrow`; any other variable ⇒ `Wide` (a wide var widens the
/// arithmetic — the #329 fix). An integer literal is `Neutral`. A narrow `as`
/// cast ⇒ `Narrow`; a wide cast ⇒ `Wide`. Nested arithmetic joins; a shift
/// carries the LEFT width. A Call / FieldAccess leaf is `Narrow` when it
/// resolves to an 8/16-bit type, else the historical `Neutral` (the `lower.rs`
/// classify with `nonnarrow = Neutral`). Every other leaf is `Neutral`.
#[cfg(feature = "std-surface")]
fn infer(node: &Node, env: &HashMap<String, Value>) -> NarrowLat {
    match node {
        Node::Lit(Literal::Ident(name), _) => match local_kind(name) {
            Some(kind) => NarrowLat::Narrow(kind),
            None => NarrowLat::Wide,
        },
        Node::Lit(Literal::Int(_), _) => NarrowLat::Neutral,
        Node::Paren(inner, _) => infer(inner, env),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => join(infer(left, env), infer(right, env)),
        Node::Bitwise {
            op: ast::BitOp::Shl | ast::BitOp::Shr,
            left,
            ..
        } => infer(left, env),
        // A narrow `as u8`/`as u16` cast ⇒ `Narrow`; ANY other cast target (a
        // wide `i64`/`i32`/`u32`, an unresolved type) ⇒ `Wide` — the explicit
        // widen forces the whole arithmetic wide (#329). Mirrors `lower.rs`
        // `classify(Some(ty), Wide)`; the Neutral fallback here silently
        // OVER-narrowed `a_u8 + (10 as i64)` to 4 instead of widening to 260.
        Node::As { ty, .. } => classify_wide(cast_kind(ty)),
        // Finding 2 leaves: `Narrow` when resolved to an 8/16-bit type, else the
        // historical non-forcing `Neutral`.
        Node::Call { callee, .. } => classify_neutral(call_ret_kind(callee)),
        Node::FieldAccess {
            receiver, field, ..
        } => classify_neutral(field_access_kind(receiver, field, env)),
        _ => NarrowLat::Neutral,
    }
}

/// The `lower.rs` `classify(t, Neutral)` arm: a resolved narrow (8/16-bit) kind
/// ⇒ `Narrow`; an unresolved / non-narrow leaf keeps the historical `Neutral`.
#[cfg(feature = "std-surface")]
fn classify_neutral(kind: Option<NarrowScalar>) -> NarrowLat {
    match kind {
        Some(kind) => NarrowLat::Narrow(kind),
        None => NarrowLat::Neutral,
    }
}

/// The `lower.rs` `classify(Some(ty), Wide)` arm for an `as` cast: a resolved
/// narrow (8/16-bit) target ⇒ `Narrow`; ANY other cast target (a wide
/// `i64`/`i32`/`u32`, an unresolved type) ⇒ `Wide`. NOT `Neutral`: a wide cast
/// must WIDEN the whole arithmetic (#329), so `let a: u8 = 250; a + (10 as i64)`
/// stays 260 (join(Narrow, Wide) = Wide, no re-mask) rather than re-masking to
/// u8 (4, from join(Narrow, Neutral) = Narrow).
#[cfg(feature = "std-surface")]
fn classify_wide(kind: Option<NarrowScalar>) -> NarrowLat {
    match kind {
        Some(kind) => NarrowLat::Narrow(kind),
        None => NarrowLat::Wide,
    }
}

/// Apply an inferred lattice value to the computed result. A `Narrow` width
/// masks an `Int` through `declared_width::apply_narrow_kind`; a `Narrow`
/// inference over a non-`Int` value is REFUSED (fail-closed — the one thing
/// this layer must never do is answer at full width). `Wide`/`Neutral` pass
/// through untouched.
#[cfg(feature = "std-surface")]
fn apply_lat(lat: NarrowLat, value: Value) -> Result<Value, EvalError> {
    match lat {
        NarrowLat::Narrow(kind) => match value {
            Value::Int(n) => Ok(Value::Int(declared_width::apply_narrow_kind(n, kind))),
            other => Err(EvalError::UnsupportedMsg(format!(
                "intermediate arithmetic inferred a narrow `{}` width, but evaluation produced a \
                 {}, which this evaluator cannot materialise at that width; refusing rather than \
                 answering at full width",
                kind_name(kind),
                value_kind(&other)
            ))),
        },
        NarrowLat::Wide | NarrowLat::Neutral => Ok(value),
    }
}

/// Width of a resolved narrow kind, for the join's widest-wins comparison.
#[cfg(feature = "std-surface")]
fn width_of(kind: NarrowScalar) -> u32 {
    match kind {
        NarrowScalar::Signed(w) | NarrowScalar::Unsigned(w) => w,
    }
}

/// A resolved narrow kind, filtered to the 8/16-bit arith admitted set
/// (`lower.rs::is_named_narrow_sig_ty` — the 32-bit widths are deliberately
/// excluded from intermediate re-masking).
#[cfg(feature = "std-surface")]
fn arith_admitted(kind: Option<NarrowScalar>) -> Option<NarrowScalar> {
    kind.filter(|&k| matches!(width_of(k), 8 | 16))
}

/// Human name of a `Value` kind, for the fail-closed refusal message (never
/// dumps a whole tensor/struct into the diagnostic).
#[cfg(feature = "std-surface")]
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "string",
        Value::Tuple(_) => "array/tuple",
        Value::Tensor(_) => "tensor",
        Value::GradMap(_) => "gradient map",
        Value::Enum { .. } => "enum variant",
        Value::Struct { .. } => "struct",
    }
}

/// Source spelling of a narrow kind, for the fail-closed refusal message.
#[cfg(feature = "std-surface")]
fn kind_name(kind: NarrowScalar) -> String {
    match kind {
        NarrowScalar::Signed(w) => format!("i{w}"),
        NarrowScalar::Unsigned(w) => format!("u{w}"),
    }
}

/// The recorded narrow kind of a scalar local, filtered to the 8/16-bit arith
/// admitted set. Reads `declared_width`'s narrow-locals registry, so the
/// lattice never keeps a second copy of which locals are narrow.
#[cfg(feature = "std-surface")]
fn local_kind(name: &str) -> Option<NarrowScalar> {
    let kind = declared_width::NARROW_LOCALS.with(|m| {
        let map = m.borrow();
        if map.is_empty() {
            None
        } else {
            map.get(name).copied()
        }
    });
    arith_admitted(kind)
}

/// The narrow kind of an `as`-cast TARGET, alias-resolved and filtered to the
/// 8/16-bit arith admitted set — `x as u8` ⇒ `Narrow(u8)`, `x as i32`/`i64` ⇒
/// wide. Reuses the shared `narrow_scalar_kind` table; no second table.
#[cfg(feature = "std-surface")]
fn cast_kind(ty: &TypeAnn) -> Option<NarrowScalar> {
    let resolved = super::type_aliases::resolve_active(ty);
    arith_admitted(declared_width::narrow_scalar_kind(&resolved))
}

/// The narrow kind of a Call leaf's declared return type, read from the
/// evaluator's `FN_TABLE` (the artifact reads `ir.fn_signatures`). `None` for a
/// wide/unknown return.
#[cfg(feature = "std-surface")]
fn call_ret_kind(callee: &str) -> Option<NarrowScalar> {
    let ret = super::FN_TABLE.with(|t| t.borrow().get(callee).and_then(|f| f.ret_type.clone()))?;
    let resolved = super::type_aliases::resolve_active(&ret);
    arith_admitted(declared_width::narrow_scalar_kind(&resolved))
}

/// The narrow kind of a `receiver.field` leaf, resolved only for an ident
/// receiver whose runtime `env` value is a `Value::Struct`: the struct name
/// comes from that value (the evaluator's stand-in for the artifact's static
/// receiver type), and the field's declared type from `declared_width`'s
/// narrow-field schema. A non-ident receiver, an unbound receiver, or a
/// non-struct value is unresolved ⇒ `None` ⇒ Neutral.
#[cfg(feature = "std-surface")]
fn field_access_kind(
    receiver: &Node,
    field: &str,
    env: &HashMap<String, Value>,
) -> Option<NarrowScalar> {
    let Node::Lit(Literal::Ident(name), _) = receiver else {
        return None;
    };
    let Some(Value::Struct {
        name: struct_name, ..
    }) = env.get(name)
    else {
        return None;
    };
    let field_ty = declared_width::STRUCT_FIELD_TYPES.with(|m| {
        let schema = m.borrow();
        if schema.is_empty() {
            None
        } else {
            schema.get(struct_name).and_then(|f| f.get(field)).cloned()
        }
    })?;
    arith_admitted(declared_width::narrow_kind(&field_ty))
}
