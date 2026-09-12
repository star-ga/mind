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

//! Declared-width materialisation for the tree-walking evaluator — the layer
//! every `mindc test` verdict is graded against.
//!
//! # Why this exists
//!
//! The shipped language contract is two's-complement wrap AT THE DECLARED
//! WIDTH, identical on every substrate (CHANGELOG, the narrow-int entries).
//! Both shipped backends honour it through three DISTINCT mechanisms:
//!
//! * **RETURN** `-> i32` — the callee truncates at the return boundary and the
//!   use site re-widens (`arith.extsi` for a signed width, `arith.extui` for
//!   an unsigned one, so unsignedness is tracked separately and stays that
//!   way);
//! * **PARAM** `(x: i32)` — the declared width is materialised into the
//!   binding as the argument is bound (for `i32`/`u32` the signature IS the
//!   narrow width and the CALLER truncates; for the 8/16-bit widths the callee
//!   masks on entry — see below, the observable value is the same);
//! * **LOCAL** `let a: i32` — a `shli 32` / `shrsi 32` pair in the i64 slot
//!   (`lower.rs::mask_narrow_let`), re-applied on every later reassignment of
//!   that binding through `lower.rs`'s per-fn `NARROW_LOCALS` registry
//!   (`mask_narrow_assign`).
//!
//! The tree evaluator carried every scalar full width and therefore answered
//! WIDE at all three: `fn f() -> i32 { return 50000 * 50000 }` evaluated to
//! 2500000000 where both backends produce -1794967296. Because this evaluator
//! is the ORACLE, a silent wide answer is not a small divergence — it grades
//! correct code as wrong and wrong code as right. So the rule here is: honour
//! the declared width, or refuse loudly. Never answer wide.
//!
//! # One admitted set, two lowering strategies
//!
//! MEASURED (a `.so` emitted by this compiler, read back at full width by a C
//! probe, with non-constant operands so nothing is literal-folding): all six
//! sub-64-bit declared widths narrow at all three mechanisms. The compiled
//! artifact answers `-> u8 { 300 }` = 44, `(c: u8)` given 300 = 44,
//! `let a: u8 = 300` = 44, `-> i16` given 40000 = -25536.
//!
//! The two lowering STRATEGIES behind that differ, and the difference is worth
//! knowing before touching this file:
//!
//! * `i32`/`u32` take the PHYSICAL-i32 MLIR ABI —
//!   `type_ann_to_abi_mlir` gives the `func.func` slot the real `i32` type, the
//!   CALLER emits `arith.trunci` on each argument, and the callee emits
//!   `arith.trunci` at the return;
//! * `i8`/`u8`/`i16`/`u16` take the i64-SLOT narrow-signature ABI
//!   (`lower.rs::is_named_narrow_sig_ty`) — the `func.func` slot stays `i64`
//!   and the CALLEE masks instead: the parameter is materialised at its
//!   declared width on fn entry, and `mask_narrow_ret` masks every return
//!   site.
//!
//! Caller-truncates and callee-masks-on-entry are observationally identical at
//! the value level, which is the only level this evaluator has, so one
//! admitted set is correct here. It is NOT correct to infer the set from the
//! ABI slot type: reading `type_ann_to_abi_mlir` alone says `-> u8` is an i64
//! slot and would have left the oracle answering 300 where the artifact
//! answers 44 — the same divergence this module exists to remove, pointing the
//! other way. The set below is the one the measurement supports; re-measure
//! before changing it.
//!
//! # No fifth width representation
//!
//! The declared `TypeAnn` IS the width. Nothing here spells a bit count:
//! `narrow_kind` resolves the annotation through the module's own type
//! aliases and then asks `narrow_scalar_kind` — the evaluator's one
//! narrow-width table, shared with the `as`-cast path — and the actual
//! truncate/extend is `apply_narrow_kind`, the mirror of
//! `lower.rs::mask_narrow_let`. Adding a width means adding it THERE, once,
//! where both readers see it.
//!
//! # The fourth mechanism: STRUCT FIELDS
//!
//! `Node::StructLit` had no field types, so a `[u8; N]` / `u8` field was stored
//! and read at full width — a `[u8; 4]` holding 300 read 44 from the artifact
//! but 300 here (MEASURED). `Value::Struct` carries no field types, so
//! `install_struct_defs` builds a per-struct narrow-field SCHEMA once per module
//! eval from its `StructDef` nodes, and the `Node::StructLit` arm runs each
//! field through `narrow_struct_field` at CONSTRUCTION (so every later read/copy
//! sees the wrapped value, mirroring the artifact's stored byte). A scalar
//! narrow field and a narrow-element aggregate (`[u8; N]`, `&[u8]`, nested)
//! share the one `narrow_scalar_kind` table; a known-narrow field whose value
//! cannot be materialised at that width is REFUSED, never returned wide.
//!
//! # The fifth mechanism: INTERMEDIATE NARROW ARITHMETIC
//!
//! An 8/16-bit binop/shift RESULT re-wraps at a width the SOURCE never declares
//! (`a: u8 = 250; (a + 10) / 2` is 2); mirrored in the sibling `narrow_arith`
//! module, reusing this file's `pub(crate)` width tables and registries.
//!
//! # Not covered (declared open, not silently half-done)
//!
//! A `p.f`/`a[i]` via a non-`env`-ident receiver, an unannotated array element, or an aggregate param stays width-NEUTRAL — see `narrow_arith`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::ast::{Node, TypeAnn};

use super::{EvalError, Value};

// ---------------------------------------------------------------------------
// The evaluator's ONE narrow-width table, and the materialisation it drives.
// Both readers live here: the `as`-cast path (`apply_scalar_cast`, called from
// the `Node::As` arm) and the declared-width path below. Adding a width means
// adding it to `narrow_scalar_kind`, once, where both readers see it.
// ---------------------------------------------------------------------------

/// Apply a scalar `as`-cast to an i64-carried interpreter value, matching the
/// codegen path (`src/eval/lower.rs` `Node::As` + `scalar_int_cast_width` /
/// `scalar_uint_cast_width`) bit-for-bit so the interpreter never diverges from
/// the runnable artifact.
///
///   * SIGNED narrow (`i8`/`i16`/`i32`, also `TypeAnn::ScalarI32`): truncate to
///     the low `W` bits then sign-extend — `(n << (64-W)) >> (64-W)` with an
///     arithmetic right shift, exactly the codegen shift pair.
///   * UNSIGNED narrow (`u8`/`u16`/`u32`, also `TypeAnn::ScalarU32`): zero-extend
///     by masking the low `W` bits — `n & ((1<<W)-1)`.
///   * `i64`/`u64`/floats/pointers/aliases / any non-narrowing target: returned
///     unchanged (the transparent codegen fall-through).
pub(crate) fn apply_scalar_cast(n: i64, ty: &TypeAnn) -> i64 {
    match narrow_scalar_kind(ty) {
        Some(kind) => apply_narrow_kind(n, kind),
        None => n,
    }
}

/// The sub-64-bit materialisation a DECLARED scalar integer type demands, and
/// its signedness. `None` for every type the i64 slot already represents
/// exactly — `i64`/`u64`, floats, bools, pointers, aggregates, unresolved
/// names — which the backends leave untouched, so the evaluator must too.
///
/// This is the evaluator's SINGLE narrow-width table: the `as`-cast path
/// (`apply_scalar_cast`) and the declared-width path
/// (`declared_width::narrow`) both read the width out of it, so the admitted
/// set cannot drift between "a cast to u8 narrows" and "a binding declared u8
/// narrows". It mirrors `lower.rs::scalar_int_cast_width` /
/// `scalar_uint_cast_width` — the widths the compiled shift-pair / mask
/// actually materialise — and deliberately reports `None` (not `Signed(64)`)
/// for the full widths, because a 64-bit "narrowing" is the identity.
///
/// Alias resolution is NOT done here: callers that must see through a
/// module-local `type` alias resolve first and call this on the target (see
/// `declared_width::narrow_kind`). Keeping resolution out preserves the
/// existing `as`-cast behaviour exactly.
pub(crate) fn narrow_scalar_kind(ty: &TypeAnn) -> Option<NarrowScalar> {
    match ty {
        TypeAnn::ScalarI32 => Some(NarrowScalar::Signed(32)),
        TypeAnn::ScalarU32 => Some(NarrowScalar::Unsigned(32)),
        TypeAnn::Named(name) => match name.as_str() {
            "i8" => Some(NarrowScalar::Signed(8)),
            "i16" => Some(NarrowScalar::Signed(16)),
            "i32" => Some(NarrowScalar::Signed(32)),
            "u8" => Some(NarrowScalar::Unsigned(8)),
            "u16" => Some(NarrowScalar::Unsigned(16)),
            "u32" => Some(NarrowScalar::Unsigned(32)),
            _ => None,
        },
        _ => None,
    }
}

/// A declared scalar integer width below 64 bits, carrying the signedness that
/// decides how the value is re-widened back into the i64 slot: a signed width
/// sign-extends (the compiled `arith.extsi`, i.e. the `(x << k) >> k` shift
/// pair), an unsigned width zero-extends (`arith.extui`, i.e. the low-bit
/// mask). Unsignedness is tracked here, alongside the width and never merged
/// into it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NarrowScalar {
    Signed(u32),
    Unsigned(u32),
}

/// Materialise `n` at a declared narrow width, bit-for-bit as the compiled
/// path does: truncate + sign-extend for a signed width, mask for an unsigned
/// one.
pub(crate) fn apply_narrow_kind(n: i64, kind: NarrowScalar) -> i64 {
    match kind {
        NarrowScalar::Signed(width) => {
            let shift = 64 - width as i64;
            (n << shift) >> shift
        }
        NarrowScalar::Unsigned(width) => {
            let mask: i64 = if width == 32 {
                0xFFFF_FFFF
            } else {
                (1i64 << width) - 1
            };
            n & mask
        }
    }
}

// ---------------------------------------------------------------------------
// The three declared-width mechanisms.
// ---------------------------------------------------------------------------

/// The narrow materialisation a declared type demands — the single admitted
/// set, shared by all three mechanisms (see the module docs for the
/// measurement that says it is one set and not two).
///
/// The canonical spellings (`i32`, `u8`, …) are answered by the shared table
/// directly; only a name the table does not know pays for alias resolution, so
/// the overwhelmingly common `i64`/float/aggregate annotation costs one failed
/// match and nothing else.
pub(crate) fn narrow_kind(ty: &TypeAnn) -> Option<NarrowScalar> {
    if let Some(kind) = narrow_scalar_kind(ty) {
        return Some(kind);
    }
    resolve_named_alias(ty)
}

/// A `Named` annotation may be a module-local `type` alias OF a narrow scalar,
/// so resolve it before giving up. `lower.rs::collect_signatures` resolves
/// aliases the same way before a signature reaches the ABI mapper.
#[cfg(feature = "std-surface")]
fn resolve_named_alias(ty: &TypeAnn) -> Option<NarrowScalar> {
    match ty {
        TypeAnn::Named(_) => narrow_scalar_kind(&super::type_aliases::resolve_active(ty)),
        _ => None,
    }
}

/// The alias table (`eval::type_aliases`) is part of the `std-surface`, so the
/// low-level subset has no aliases to resolve. That subset also cannot lower a
/// user `FnDef` at all — measured, `error[mlir]: unsupported instruction at
/// index 0: FnDef` — so there is no artifact for the oracle to disagree with
/// here, and the canonical spellings above still materialise identically in
/// both configurations. No assertion changes with the feature.
#[cfg(not(feature = "std-surface"))]
fn resolve_named_alias(_ty: &TypeAnn) -> Option<NarrowScalar> {
    None
}

/// Human name of a `Value` kind, for refusal messages that must not dump a
/// whole tensor or struct into a diagnostic.
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

/// Source spelling of a narrow materialisation, for refusal messages.
fn kind_name(kind: NarrowScalar) -> String {
    match kind {
        NarrowScalar::Signed(w) => format!("i{w}"),
        NarrowScalar::Unsigned(w) => format!("u{w}"),
    }
}

/// Materialise `value` at the width `ty` declares.
///
/// * a non-narrowing declared type returns `value` verbatim;
/// * an i64-carried scalar under a narrowing type is truncated and re-extended
///   by `apply_narrow_kind` — sign-extend for `i8`/`i16`/`i32`,
///   zero-extend for `u8`/`u16`/`u32`;
/// * anything else is REFUSED. `what` + `name` point the refusal at the
///   DECLARATION (``return value of `f` ``, ``parameter `x` ``, ``binding
///   `a` ``) rather than at the interpreter, and refusing is the whole point:
///   the one thing this layer must never do is answer at full width.
pub(crate) fn narrow(
    ty: &TypeAnn,
    value: Value,
    what: What,
    name: &str,
) -> Result<Value, EvalError> {
    let Some(kind) = narrow_kind(ty) else {
        return Ok(value);
    };
    narrow_with(kind, value, what, name)
}

fn narrow_with(
    kind: NarrowScalar,
    value: Value,
    what: What,
    name: &str,
) -> Result<Value, EvalError> {
    match value {
        Value::Int(n) => Ok(Value::Int(apply_narrow_kind(n, kind))),
        other => Err(EvalError::UnsupportedMsg(format!(
            "{} `{name}` is declared `{}`, but evaluation produced a {}, which this evaluator \
             cannot materialise at that width; refusing rather than answering at full width",
            what.label(),
            kind_name(kind),
            value_kind(&other)
        ))),
    }
}

/// Which of the three declared-width mechanisms a refusal came from. Carried
/// as a tag rather than a pre-formatted string so the message — and its
/// allocation — exists only on the refusal path.
#[derive(Clone, Copy)]
pub(crate) enum What {
    Binding,
    Parameter,
    Return,
}

impl What {
    fn label(self) -> &'static str {
        match self {
            What::Binding => "binding",
            What::Parameter => "parameter",
            What::Return => "return value of",
        }
    }
}

/// The RETURN mechanism: materialise a callee's result at its declared width
/// before the use site sees it, so the use site reads exactly what an
/// `arith.extsi` / `arith.extui` of the truncated callee result would carry.
/// Applied at BOTH exits of a body — an early `return` and fall-through.
pub(crate) fn narrow_return(
    ret: &Option<TypeAnn>,
    callee: &str,
    value: Value,
) -> Result<Value, EvalError> {
    narrow_opt(ret, value, What::Return, callee)
}

/// The LOCAL mechanism for a `let`, both halves at once: materialise the value
/// at the declared width, then RECORD that width so a later unannotated
/// `name = …` re-masks through `narrow_reassign`. The two always go together
/// — a site that narrows without recording goes wide again one statement
/// later — so they are one call.
pub(crate) fn bind_let(
    ann: &Option<TypeAnn>,
    name: &str,
    value: Value,
) -> Result<Value, EvalError> {
    let value = narrow_opt(ann, value, What::Binding, name)?;
    record(name, ann.as_ref());
    Ok(value)
}

/// `narrow` over an optional annotation (`let` bindings, return types).
pub(crate) fn narrow_opt(
    ty: &Option<TypeAnn>,
    value: Value,
    what: What,
    name: &str,
) -> Result<Value, EvalError> {
    match ty {
        Some(t) => narrow(t, value, what, name),
        None => Ok(value),
    }
}

// ---------------------------------------------------------------------------
// Narrow-locals registry (the LOCAL mechanism's second half)
// ---------------------------------------------------------------------------
//
// `let a: i32 = …` masks its initializer, but a later `a = a * 50000` carries
// no annotation. `lower.rs` keeps a per-fn `NARROW_LOCALS` map for exactly
// this and re-applies `mask_narrow_let` on `Assign` (`mask_narrow_assign`);
// the evaluator needs the same map or it goes wide again one statement after
// the declaration. The scoping rules below are the ones `lower.rs` already
// uses: a fn body starts from an EMPTY map seeded with its own narrow params
// (`enter_narrow_scope`), and a block snapshots/restores
// (`NarrowLocalsGuard`).
//
// Only genuinely-narrowing declarations are recorded, so the map is EMPTY for
// the overwhelming majority of modules and every scope operation below is an
// `is_empty()` test with no allocation. A non-narrow re-`let` of the same name
// REMOVES the entry — a stale narrow decl would be a new class of silent
// wrongness, which is precisely what this module exists to prevent (the same
// shadow-clear rule `lower.rs` documents as Finding 1(a)).

thread_local! {
    pub(crate) static NARROW_LOCALS: RefCell<HashMap<String, NarrowScalar>> = RefCell::new(HashMap::new());
}

/// Restores the enclosing scope's narrow-locals map on every exit path,
/// including an error and the `ReturnFlow` control-flow signal.
pub(crate) struct ScopeGuard(Option<HashMap<String, NarrowScalar>>);

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            NARROW_LOCALS.with(|m| *m.borrow_mut() = previous);
        }
    }
}

/// Enter a FUNCTION scope: the callee sees none of the caller's narrow locals.
pub(crate) fn enter_function_scope() -> ScopeGuard {
    ScopeGuard(Some(
        NARROW_LOCALS.with(|m| std::mem::take(&mut *m.borrow_mut())),
    ))
}

/// Enter a BLOCK scope: declarations inside are visible, and are dropped again
/// at block exit so a block-local `let a: i32` never re-types an enclosing
/// `a: i64` for the statements after the block.
pub(crate) fn enter_block_scope() -> ScopeGuard {
    ScopeGuard(Some(NARROW_LOCALS.with(|m| m.borrow().clone())))
}

/// Record (or clear) the declared narrow width of `name`.
pub(crate) fn record(name: &str, ty: Option<&TypeAnn>) {
    let declared = ty.and_then(narrow_kind);
    NARROW_LOCALS.with(|m| {
        let mut map = m.borrow_mut();
        match declared {
            Some(kind) => {
                map.insert(name.to_string(), kind);
            }
            None => {
                // Removing from an EMPTY map still hashes `name`; skipping the
                // call keeps the narrow-free hot path allocation- and
                // hash-free (the same reason `lower.rs` guards its remove).
                if !map.is_empty() {
                    map.remove(name);
                }
            }
        }
    });
}

/// Re-apply a binding's declared narrow width to an unannotated reassignment.
/// A binding with no recorded narrow declaration passes through untouched.
pub(crate) fn narrow_reassign(name: &str, value: Value) -> Result<Value, EvalError> {
    let declared = NARROW_LOCALS.with(|m| {
        let map = m.borrow();
        if map.is_empty() {
            None
        } else {
            map.get(name).copied()
        }
    });
    match declared {
        Some(kind) => narrow_with(kind, value, What::Binding, name),
        None => Ok(value),
    }
}

// ---------------------------------------------------------------------------
// The fourth mechanism: STRUCT FIELDS
// ---------------------------------------------------------------------------
//
// `Value::Struct` carries no field types, so narrowing a `u8` / `[u8; N]` field
// needs a schema keyed by struct name, built ONCE per module eval from its
// `StructDef` nodes (`install_struct_defs`) and read at construction
// (`narrow_struct_field`). Only ACTUALLY-narrowing fields are recorded, so the
// schema is empty for most modules and construction pays one `is_empty()` test
// with no allocation — the same discipline the narrow-locals registry uses.

thread_local! {
    /// `struct name -> (field name -> declared narrowing type)`, for the fields
    /// whose declared type narrows only. Restored on scope exit by
    /// `StructDefsGuard` so a nested module eval neither sees nor leaks the
    /// caller's schema.
    pub(crate) static STRUCT_FIELD_TYPES: RefCell<HashMap<String, HashMap<String, TypeAnn>>> =
        RefCell::new(HashMap::new());
}

/// Restores the enclosing eval's struct-field schema on every exit path.
pub(crate) struct StructDefsGuard(HashMap<String, HashMap<String, TypeAnn>>);

impl Drop for StructDefsGuard {
    fn drop(&mut self) {
        STRUCT_FIELD_TYPES.with(|m| *m.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Collect the narrow-field schema from a module's `StructDef` items and make
/// it the active schema for the eval, returning a guard that restores the
/// previous one. Only fields whose declared type actually narrows (a narrow
/// scalar, or an array/slice whose element narrows, recursively) are recorded,
/// so the common non-narrow struct contributes nothing and construction stays
/// on the empty-map fast path.
pub(crate) fn install_struct_defs(items: &[Node]) -> StructDefsGuard {
    let mut schema: HashMap<String, HashMap<String, TypeAnn>> = HashMap::new();
    for item in items {
        if let Node::StructDef { name, fields, .. } = item {
            let mut narrow_fields: HashMap<String, TypeAnn> = HashMap::new();
            for f in fields {
                if type_needs_narrowing(&f.ty) {
                    narrow_fields.insert(f.name.clone(), f.ty.clone());
                }
            }
            if !narrow_fields.is_empty() {
                // Last definition wins, matching the fn table; a module's own
                // duplicate `StructDef` is a checker concern, not this layer's.
                schema.insert(name.clone(), narrow_fields);
            }
        }
    }
    STRUCT_FIELD_TYPES.with(|m| {
        let previous = std::mem::replace(&mut *m.borrow_mut(), schema);
        StructDefsGuard(previous)
    })
}

/// Does a declared field type require any sub-64-bit materialisation? True for
/// a narrow scalar and for an array/slice whose element (recursively) narrows;
/// false for everything the i64 slot already represents exactly. Alias
/// resolution rides on `narrow_kind`, so a `type Byte = u8` field is seen.
fn type_needs_narrowing(ty: &TypeAnn) -> bool {
    if narrow_kind(ty).is_some() {
        return true;
    }
    match ty {
        TypeAnn::Array { element, .. } | TypeAnn::Slice { element, .. } => {
            type_needs_narrowing(element)
        }
        _ => false,
    }
}

/// Narrow a struct field's constructed value to its declared width, at struct
/// construction. A field with no recorded narrow declaration — and every field
/// of an unregistered struct — passes through untouched; a recorded field is
/// materialised by `narrow_field_value`, which refuses rather than answering
/// wide when the value cannot be materialised at the declared width.
pub(crate) fn narrow_struct_field(
    struct_name: &str,
    field: &str,
    value: Value,
) -> Result<Value, EvalError> {
    STRUCT_FIELD_TYPES.with(|m| {
        let schema = m.borrow();
        if schema.is_empty() {
            return Ok(value);
        }
        match schema.get(struct_name).and_then(|f| f.get(field)) {
            Some(ty) => narrow_field_value(ty, value, struct_name, field),
            None => Ok(value),
        }
    })
}

/// Materialise `value` at the declared field type `ty`, recursively:
///
/// * a narrow scalar type narrows an `Int` (as the three scalar mechanisms do);
/// * an array/slice of a narrowing element narrows each element of a `Tuple`
///   (`[u8; N]`, `&[u8]`, and nesting thereof — the interpreter carries every
///   array/slice value as `Value::Tuple`);
/// * a non-narrowing declared type returns `value` verbatim;
/// * a narrow-declared field whose value cannot be materialised at that width
///   (a non-int scalar, a non-tuple aggregate) is REFUSED — the one thing this
///   layer must never do is answer at full width.
fn narrow_field_value(
    ty: &TypeAnn,
    value: Value,
    struct_name: &str,
    field: &str,
) -> Result<Value, EvalError> {
    if let Some(kind) = narrow_kind(ty) {
        return match value {
            Value::Int(n) => Ok(Value::Int(apply_narrow_kind(n, kind))),
            other => Err(refuse_field(struct_name, field, &kind_name(kind), &other)),
        };
    }
    match ty {
        TypeAnn::Array { element, .. } | TypeAnn::Slice { element, .. } => {
            if !type_needs_narrowing(element) {
                return Ok(value);
            }
            match value {
                Value::Tuple(items) => {
                    let mut narrowed = Vec::with_capacity(items.len());
                    for item in items {
                        narrowed.push(narrow_field_value(element, item, struct_name, field)?);
                    }
                    Ok(Value::Tuple(narrowed))
                }
                other => Err(refuse_field(struct_name, field, &type_display(ty), &other)),
            }
        }
        _ => Ok(value),
    }
}

/// Fail-closed refusal for a struct field the schema says narrows but whose
/// value cannot be materialised at the declared width.
fn refuse_field(struct_name: &str, field: &str, declared: &str, got: &Value) -> EvalError {
    EvalError::UnsupportedMsg(format!(
        "field `{field}` of struct `{struct_name}` is declared `{declared}`, but evaluation \
         produced a {}, which this evaluator cannot materialise at that width; refusing rather \
         than answering at full width",
        value_kind(got)
    ))
}

/// Best-effort source spelling of an aggregate field type, for refusals.
fn type_display(ty: &TypeAnn) -> String {
    match ty {
        TypeAnn::Array { element, .. } => format!("[{}; N]", type_display(element)),
        TypeAnn::Slice { element, .. } => format!("[{}]", type_display(element)),
        TypeAnn::Named(name) => name.clone(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(n: &str) -> TypeAnn {
        TypeAnn::Named(n.to_string())
    }

    #[test]
    fn signed_declaration_wraps_at_the_declared_width() {
        let v = narrow(
            &TypeAnn::ScalarI32,
            Value::Int(2_500_000_000),
            What::Binding,
            "t",
        )
        .unwrap();
        assert_eq!(v.as_int(), Some(-1_794_967_296));
        // Every signed width in the shared table, not just i32.
        let v = narrow(&named("i8"), Value::Int(200), What::Binding, "t").unwrap();
        assert_eq!(v.as_int(), Some(-56));
        let v = narrow(&named("i16"), Value::Int(40000), What::Binding, "t").unwrap();
        assert_eq!(v.as_int(), Some(-25536));
    }

    #[test]
    fn every_admitted_width_narrows_at_every_mechanism() {
        // The compiled artifact narrows all six widths at all three
        // mechanisms (measured — see the module docs). The two ABI strategies
        // differ (caller-truncates for i32/u32, callee-masks-on-entry for the
        // 8/16-bit widths) but the VALUE does not, so making the admitted set
        // depend on the mechanism would reintroduce a divergence.
        for spelling in [
            named("i8"),
            named("u8"),
            named("i16"),
            named("u16"),
            TypeAnn::ScalarI32,
            TypeAnn::ScalarU32,
        ] {
            assert!(narrow_kind(&spelling).is_some(), "{spelling:?}");
        }
        let u8_ty = named("u8");
        for mechanism in [What::Binding, What::Parameter, What::Return] {
            assert_eq!(
                narrow(&u8_ty, Value::Int(300), mechanism, "a")
                    .unwrap()
                    .as_int(),
                Some(44),
                "a `u8` must materialise as 44 at every mechanism"
            );
        }
    }

    #[test]
    fn unsigned_zero_extends_rather_than_sign_extends() {
        // 65536 * 65536 == 2^32 -> 0 under u32, and the result must be
        // non-negative (extui), never sign fill.
        let v = narrow(
            &TypeAnn::ScalarU32,
            Value::Int(4_294_967_296),
            What::Binding,
            "t",
        )
        .unwrap();
        assert_eq!(v.as_int(), Some(0));
        let v = narrow(&named("u32"), Value::Int(-1), What::Binding, "t").unwrap();
        assert_eq!(v.as_int(), Some(4_294_967_295));
        let v = narrow(&named("u8"), Value::Int(-1), What::Binding, "t").unwrap();
        assert_eq!(v.as_int(), Some(255));
        let v = narrow(&TypeAnn::ScalarU32, Value::Int(-1), What::Return, "f").unwrap();
        assert_eq!(v.as_int(), Some(4_294_967_295));
        // Same bit pattern, signed declaration: -1 stays -1.
        let v = narrow(&TypeAnn::ScalarI32, Value::Int(-1), What::Binding, "t").unwrap();
        assert_eq!(v.as_int(), Some(-1));
    }

    #[test]
    fn full_width_and_aggregate_types_pass_through() {
        for ty in [
            TypeAnn::ScalarI64,
            named("u64"),
            named("usize"),
            TypeAnn::ScalarBool,
            TypeAnn::ScalarF64,
            named("Packet"),
        ] {
            assert!(narrow_kind(&ty).is_none(), "{ty:?}");
        }
        let v = narrow(
            &TypeAnn::ScalarI64,
            Value::Int(5_000_000_000),
            What::Binding,
            "t",
        )
        .unwrap();
        assert_eq!(v.as_int(), Some(5_000_000_000));
        // A non-narrow declaration must not refuse a non-int value either.
        assert!(narrow(&named("Packet"), Value::Str("s".into()), What::Binding, "t").is_ok());
    }

    #[test]
    fn narrow_declared_non_scalar_is_refused_not_widened() {
        let err = narrow(
            &TypeAnn::ScalarI32,
            Value::Str("handle".to_string()),
            What::Return,
            "f",
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("return value of `f`"), "{msg}");
        assert!(msg.contains("refusing"), "{msg}");
        let err = narrow(&named("u8"), Value::Float(1.5), What::Binding, "x").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("binding `x`"), "{msg}");
        assert!(msg.contains("`u8`"), "{msg}");
        let err = narrow(&TypeAnn::ScalarU32, Value::Float(1.5), What::Parameter, "x").unwrap_err();
        assert!(err.to_string().contains("parameter `x`"), "{err}");
    }

    #[test]
    fn reassignment_reapplies_the_declared_width() {
        let _scope = enter_function_scope();
        record("a", Some(&TypeAnn::ScalarI32));
        let v = narrow_reassign("a", Value::Int(5_000_000_000)).unwrap();
        assert_eq!(v.as_int(), Some(705_032_704));
        // A non-narrow re-`let` clears the entry: no stale narrowing.
        record("a", Some(&TypeAnn::ScalarI64));
        let v = narrow_reassign("a", Value::Int(5_000_000_000)).unwrap();
        assert_eq!(v.as_int(), Some(5_000_000_000));
        // An untracked name is untouched.
        assert_eq!(
            narrow_reassign("never_declared", Value::Int(5_000_000_000))
                .unwrap()
                .as_int(),
            Some(5_000_000_000)
        );
    }

    #[test]
    fn block_scope_does_not_leak_a_narrow_declaration() {
        let _scope = enter_function_scope();
        record("a", Some(&TypeAnn::ScalarI64));
        {
            let _block = enter_block_scope();
            record("a", Some(&TypeAnn::ScalarI32));
            assert_eq!(
                narrow_reassign("a", Value::Int(5_000_000_000))
                    .unwrap()
                    .as_int(),
                Some(705_032_704)
            );
        }
        assert_eq!(
            narrow_reassign("a", Value::Int(5_000_000_000))
                .unwrap()
                .as_int(),
            Some(5_000_000_000)
        );
    }

    #[test]
    fn function_scope_does_not_inherit_caller_declarations() {
        let _scope = enter_function_scope();
        record("a", Some(&TypeAnn::ScalarI32));
        {
            let _callee = enter_function_scope();
            assert_eq!(
                narrow_reassign("a", Value::Int(5_000_000_000))
                    .unwrap()
                    .as_int(),
                Some(5_000_000_000)
            );
        }
        assert_eq!(
            narrow_reassign("a", Value::Int(5_000_000_000))
                .unwrap()
                .as_int(),
            Some(705_032_704)
        );
    }
}
