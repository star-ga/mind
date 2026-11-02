// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Struct field stores in the tree-walking evaluator: `r.f = v` and
//! `r.f[i] = v` (a fixed-array field).
//!
//! A struct value is a shared [`Record`], so the store goes to the record the
//! receiver denotes and every other reference to it sees the new value
//! (types.md "Record identity"): a record passed to a fn that stores a field,
//! or bound to a second name, is the same record. A fixed-array field stays a
//! value container: replacing one element of it changes that field of THIS
//! record and no array copied out of it earlier. The receiver is evaluated
//! before the right-hand side, as the compiled store evaluates it.

use std::collections::HashMap;

use crate::ast::Node;

use super::autodiff::TensorEnvEntry;
use super::value::Record;
use super::{EvalError, ExecMode, Value, declared_width, eval_value_expr_mode};

type Env = HashMap<String, Value>;
type TensorEnv = HashMap<String, TensorEnvEntry>;

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
    let Some(slot) = usize::try_from(idx).ok().and_then(|i| items.get_mut(i)) else {
        return Err(EvalError::BoundsTrap(format!(
            "array index {idx} out of bounds (len {})",
            items.len()
        )));
    };
    *slot = value.clone();
    store(&name, &record, field, Value::Tuple(items))?;
    Ok(value)
}
