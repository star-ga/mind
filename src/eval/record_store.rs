// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct field stores in the tree-walking evaluator: `r.f = v` and
//! `r.f[i] = v` (a fixed-array or dynamic-array field).
//!
//! A struct value is a shared [`Record`], so the store goes to the record the
//! receiver denotes and every other reference to it sees the new value
//! (types.md "Record identity"): a record passed to a fn that stores a field,
//! or bound to a second name, is the same record. A fixed-array field stays a
//! value container: replacing one element of it changes that field of THIS
//! record and no array copied out of it earlier. A field DECLARED `array<T>`
//! (a dynamic array) is different: the compiled build stores a std.vec HANDLE
//! in it, a copy of the handle shares the heap backing, and an element store
//! goes through that backing — so an alias copied out before the store sees
//! it. The interpreter models that with the shared element run
//! ([`Value::Array`], [`super::value::ArrayBuf`]): construction and a
//! whole-field store wrap the field's value into a run (a fresh one, the way
//! a new literal allocates a new backing; an already-shared run stays the
//! same run, the way a handle copy aliases), and the element store writes
//! through it. The receiver is evaluated before the right-hand side, as the
//! compiled store evaluates it.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::ast::{Node, TypeAnn};

use super::autodiff::TensorEnvEntry;
use super::value::{ArrayBuf, Record};
use super::{EvalError, ExecMode, Value, declared_width, eval_value_expr_mode};

type Env = HashMap<String, Value>;
type TensorEnv = HashMap<String, TensorEnvEntry>;

// ---------------------------------------------------------------------------
// Which struct fields are DYNAMIC arrays (`array<T>`)
// ---------------------------------------------------------------------------
//
// `Value` carries no field types, so "is this field declared `array<T>`"
// needs a schema keyed by struct name, built once per module eval from its
// `StructDef` nodes — the same shape `declared_width::install_struct_defs`
// uses for narrow fields. Only `array<T>` fields are recorded (a fixed
// `[T; N]` field or a `&[T]` slice is NOT a dynamic array), so the common
// struct contributes nothing.

thread_local! {
    /// `struct name -> fields declared array<T>`. Restored on scope exit by
    /// [`DynamicArrayFieldsGuard`] so a nested module eval neither sees nor
    /// leaks the caller's schema.
    static DYNAMIC_ARRAY_FIELDS: RefCell<HashMap<String, HashSet<String>>> =
        RefCell::new(HashMap::new());
}

/// Restores the enclosing eval's dynamic-array-field schema on every exit
/// path.
pub(crate) struct DynamicArrayFieldsGuard(HashMap<String, HashSet<String>>);

impl Drop for DynamicArrayFieldsGuard {
    fn drop(&mut self) {
        DYNAMIC_ARRAY_FIELDS.with(|m| *m.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Collect the `array<T>`-declared fields of a module's `StructDef`s and make
/// them the active schema for the eval, returning a guard that restores the
/// previous one. Last definition wins, matching the fn table and the
/// narrow-field schema.
pub(crate) fn install_dynamic_array_fields(items: &[Node]) -> DynamicArrayFieldsGuard {
    let mut schema: HashMap<String, HashSet<String>> = HashMap::new();
    for item in items {
        if let Node::StructDef { name, fields, .. } = item {
            let dynamic: BTreeSet<&str> = fields
                .iter()
                .filter(|f| is_dynamic_array_ann(&f.ty))
                .map(|f| f.name.as_str())
                .collect();
            if !dynamic.is_empty() {
                schema
                    .entry(name.clone())
                    .or_default()
                    .extend(dynamic.into_iter().map(str::to_string));
            }
        }
    }
    DYNAMIC_ARRAY_FIELDS.with(|m| {
        let previous = std::mem::replace(&mut *m.borrow_mut(), schema);
        DynamicArrayFieldsGuard(previous)
    })
}

/// `array<T>` parses as `TypeAnn::Generic { name: "array", .. }` — the
/// surface the lowering routes to the std.vec runtime (`lower.rs`); a fixed
/// `[T; N]` (`TypeAnn::Array`) and a slice (`TypeAnn::Slice`) are not it.
fn is_dynamic_array_ann(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::Generic { name, .. } if name == "array")
}

/// Is `struct_name.field` declared `array<T>` in the active schema?
fn field_is_dynamic_array(struct_name: &str, field: &str) -> bool {
    DYNAMIC_ARRAY_FIELDS.with(|m| {
        m.borrow()
            .get(struct_name)
            .is_some_and(|fields| fields.contains(field))
    })
}

/// Materialise the shared run for one store of a dynamic-array field's value:
/// a plain array value becomes a FRESH run (a new literal allocates a new
/// backing, natively), and a value that already shares a run keeps sharing it
/// (a handle copy aliases). Every other field — fixed array, scalar, struct,
/// unregistered — passes through untouched.
pub(crate) fn wrap_struct_field(struct_name: &str, field: &str, value: Value) -> Value {
    if !field_is_dynamic_array(struct_name, field) {
        return value;
    }
    match value {
        Value::Array(_) => value,
        Value::Tuple(items) => Value::Array(ArrayBuf::new(items)),
        other => other,
    }
}

/// Evaluate `receiver` to the record it denotes, refusing anything else.
fn record_of(
    receiver: &Node,
    field: &str,
    env: &Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<(String, Record), EvalError> {
    match eval_value_expr_mode(receiver, env, tensor_env, mode)? {
        Value::Struct { name, fields } => Ok((name, fields)),
        other => Err(EvalError::UnsupportedMsg(format!(
            "struct field assignment `.{field}` needs a struct receiver, got {}",
            super::assert_check::value_kind_name(&other)
        ))),
    }
}

fn store(name: &str, record: &Record, field: &str, value: Value) -> Result<Value, EvalError> {
    // The declared width of a narrow field (`u8`, `[u8; N]`) applies to a
    // store exactly as it does to the literal that built the record.
    let value = declared_width::narrow_struct_field(name, field, value)?;
    // A field declared `array<T>` holds a shared element run: the new value
    // becomes a FRESH run (a new literal allocates a new backing natively) or
    // keeps the run it already shares (a handle copy aliases). Aliases of the
    // old field keep the old run, exactly as they keep the old handle.
    let value = wrap_struct_field(name, field, value);
    if record.set(field, value.clone()) {
        Ok(value)
    } else {
        Err(EvalError::UnsupportedMsg(format!(
            "struct `{name}` has no field .{field}"
        )))
    }
}

/// `receiver.field = value`.
pub(super) fn assign_field(
    receiver: &Node,
    field: &str,
    value: &Node,
    env: &Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let (name, record) = record_of(receiver, field, env, tensor_env, mode.clone())?;
    let value = eval_value_expr_mode(value, env, tensor_env, mode)?;
    store(&name, &record, field, value)
}

/// `owner.field[index] = value`, with `value` already evaluated (the indexed
/// store keeps its established right-hand-side-first order).
pub(super) fn assign_field_element(
    owner: &Node,
    field: &str,
    index: &Node,
    value: Value,
    env: &Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let (name, record) = record_of(owner, field, env, tensor_env, mode.clone())?;
    // A field holding a shared element run (`array<T>`, see
    // `wrap_struct_field`) stores THROUGH the run: the compiled build calls
    // vec_set on the field's handle, so an alias copied out before the store
    // — or a for-each walking the same field — sees the new element.
    if let Some(Value::Array(run)) = record.get(field) {
        let idx = match eval_value_expr_mode(index, env, tensor_env, mode)? {
            Value::Int(i) => i,
            other => {
                return Err(EvalError::UnsupportedMsg(format!(
                    "array index must be an integer, got {other:?}"
                )));
            }
        };
        let len = run.len();
        if usize::try_from(idx).is_ok_and(|i| run.set(i, value.clone())) {
            return Ok(value);
        }
        return Err(EvalError::BoundsTrap(format!(
            "array index {idx} out of bounds (len {len})"
        )));
    }
    let Some(Value::Tuple(mut items)) = record.get(field) else {
        return Err(EvalError::UnsupportedMsg(format!(
            "indexed assignment receiver `.{field}` of struct `{name}` must be an array field"
        )));
    };
    let idx = match eval_value_expr_mode(index, env, tensor_env, mode)? {
        Value::Int(i) => i,
        other => {
            return Err(EvalError::UnsupportedMsg(format!(
                "array index must be an integer, got {other:?}"
            )));
        }
    };
    let len = items.len();
    let Some(slot) = usize::try_from(idx).ok().and_then(|i| items.get_mut(i)) else {
        return Err(EvalError::BoundsTrap(format!(
            "array index {idx} out of bounds (len {len})"
        )));
    };
    *slot = value.clone();
    store(&name, &record, field, Value::Tuple(items))?;
    Ok(value)
}
