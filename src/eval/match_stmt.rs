// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `match` in statement position in the tree-walking evaluator.
//!
//! The taken arm's body is a block scope over the SAME threaded env, like an
//! `if` branch: an assignment to an outer name in the arm survives the
//! `match`, and a `break` / `continue` in the arm reaches the enclosing loop,
//! as in the compiled build. The names the pattern binds are local to the arm
//! and are restored when it ends. A `match` that is an operand (`let x = match
//! ..`, a call argument) keeps the expression form in `eval_value_expr_mode`.

use std::collections::HashMap;

use crate::ast::{MatchArm, Node, Pattern};

use super::autodiff::TensorEnvEntry;
use super::{EvalError, ExecMode, Value, eval_value_expr_mode, exec_block_scoped};

type Env = HashMap<String, Value>;
type TensorEnv = HashMap<String, TensorEnvEntry>;

/// The bindings `pattern` makes against `value`, or `None` when it does not
/// match: `_`, a binding name, a literal, an enum variant with its payload
/// sub-patterns, a tuple, and a struct-variant pattern.
fn arm_bindings(pattern: &Pattern, value: &Value) -> Option<Vec<(String, Value)>> {
    let mut out = Vec::new();
    super::match_enum_payload(
        std::slice::from_ref(pattern),
        std::slice::from_ref(value),
        &mut out,
    )
    .then_some(out)
}

/// Bind `bindings` for the duration of `run`, then restore what each name
/// denoted before (or remove it) however `run` ends.
fn with_bindings<R>(
    env: &mut Env,
    bindings: Vec<(String, Value)>,
    run: impl FnOnce(&mut Env) -> Result<R, EvalError>,
) -> Result<R, EvalError> {
    let mut saved: Vec<(String, Option<Value>)> = Vec::with_capacity(bindings.len());
    for (name, value) in bindings {
        let previous = env.insert(name.clone(), value);
        saved.push((name, previous));
    }
    let outcome = run(env);
    for (name, previous) in saved.into_iter().rev() {
        match previous {
            Some(value) => {
                env.insert(name, value);
            }
            None => {
                env.remove(&name);
            }
        }
    }
    outcome
}

/// `match scrutinee { arms }` in statement position: the first arm whose
/// pattern matches and whose guard (with the pattern's bindings in scope)
/// holds runs as a block scope over `env`.
pub(super) fn exec_match(
    scrutinee: &Node,
    arms: &[MatchArm],
    env: &mut Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let value = eval_value_expr_mode(scrutinee, env, tensor_env, mode.clone())?;
    for arm in arms {
        let Some(bindings) = arm_bindings(&arm.pattern, &value) else {
            continue;
        };
        let taken = with_bindings(env, bindings, |env| {
            if let Some(guard) = &arm.guard {
                let holds = eval_value_expr_mode(guard, env, tensor_env, mode.clone())?;
                if matches!(holds, Value::Int(0)) {
                    return Ok(None);
                }
            }
            exec_block_scoped(
                std::slice::from_ref(&arm.body),
                env,
                tensor_env,
                mode.clone(),
            )
            .map(Some)
        })?;
        if let Some(result) = taken {
            return Ok(result);
        }
    }
    Err(EvalError::UnsupportedMsg(
        "no match arm matched (non-exhaustive match)".into(),
    ))
}
