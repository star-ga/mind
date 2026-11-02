// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Loop execution for the tree-walking evaluator (the `mindc test` oracle):
//! range `for`, `while`, for-each, and the `break` / `continue` signals.
//!
//! Every loop form runs its body through [`exec_block_scoped`], one block scope
//! per ITERATION, over the caller's threaded env: an `assign` to an outer name
//! survives into the next iteration and past the loop, a body `let` dies at
//! iteration end. `break` and `continue` travel as [`EvalError::BreakFlow`] /
//! [`EvalError::ContinueFlow`] up through the `if` / block / `match` arms that
//! enclose them (each restores its own scope on the way) to the innermost loop
//! here, which stops or starts its next iteration. They mirror the compiled
//! path: `break` branches to the loop exit, `continue` to the loop header (a
//! range or for-each `continue` takes the index step first). A function body
//! is a fence ([`fence`]): a signal never leaves the function that raised it.

use std::collections::HashMap;

use crate::ast::{Literal, Node};

use super::autodiff::TensorEnvEntry;
use super::closures::node_children_ref;
use super::{EvalError, ExecMode, Value, eval_value_expr_mode, exec_block_scoped};

type Env = HashMap<String, Value>;
type TensorEnv = HashMap<String, TensorEnvEntry>;

/// Upper bound on the iterations of one `while` at compile-time evaluation, so
/// a non-terminating loop fails loudly instead of hanging the compiler.
#[cfg(feature = "std-surface")]
const MAX_WHILE_ITERS: u64 = 1_000_000;

/// What one iteration of a loop body asks the loop to do next.
enum Flow {
    Next(Value),
    Continue,
    Break,
}

/// Run one iteration of `body` as its own block scope.
fn iteration(
    body: &[Node],
    env: &mut Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Flow, EvalError> {
    match exec_block_scoped(body, env, tensor_env, mode) {
        Ok(value) => Ok(Flow::Next(value)),
        Err(EvalError::ContinueFlow) => Ok(Flow::Continue),
        Err(EvalError::BreakFlow) => Ok(Flow::Break),
        Err(other) => Err(other),
    }
}

/// Bind the loop variable `var` for the duration of `run`, then restore the
/// binding it shadowed (or remove it) however `run` ends, so a statement after
/// the loop reads the OUTER binding, as compiled scoping does.
fn with_loop_var(
    var: &str,
    env: &mut Env,
    run: impl FnOnce(&mut Env) -> Result<Value, EvalError>,
) -> Result<Value, EvalError> {
    let saved = env.get(var).cloned();
    let outcome = run(env);
    match saved {
        Some(value) => {
            env.insert(var.to_string(), value);
        }
        None => {
            env.remove(var);
        }
    }
    outcome
}

fn int_bound(
    node: &Node,
    which: &str,
    env: &Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<i64, EvalError> {
    match eval_value_expr_mode(node, env, tensor_env, mode)? {
        Value::Int(n) => Ok(n),
        _ => Err(EvalError::UnsupportedMsg(format!(
            "for-loop {which} must be int"
        ))),
    }
}

/// `for var in start..end { body }`: both bounds are evaluated once, before
/// the first iteration; the loop value is the last completed iteration's.
pub(super) fn run_range_for(
    var: &str,
    start: &Node,
    end: &Node,
    body: &[Node],
    env: &mut Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let s = int_bound(start, "start", env, tensor_env, mode.clone())?;
    let e = int_bound(end, "end", env, tensor_env, mode.clone())?;
    with_loop_var(var, env, |env| {
        let mut result = Value::Int(0);
        for i in s..e {
            env.insert(var.to_string(), Value::Int(i));
            match iteration(body, env, tensor_env, mode.clone())? {
                Flow::Next(value) => result = value,
                Flow::Continue => {}
                Flow::Break => break,
            }
        }
        Ok(result)
    })
}

/// `while cond { body }`: `Value::Int(0)` is false, anything else is true.
#[cfg(feature = "std-surface")]
pub(super) fn run_while(
    cond: &Node,
    body: &[Node],
    env: &mut Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let mut result = Value::Int(0);
    let mut iters: u64 = 0;
    loop {
        if matches!(
            eval_value_expr_mode(cond, env, tensor_env, mode.clone())?,
            Value::Int(0)
        ) {
            return Ok(result);
        }
        iters += 1;
        if iters > MAX_WHILE_ITERS {
            return Err(EvalError::UnsupportedMsg(
                "`while` exceeded the compile-time evaluation iteration cap \
                 (possible non-terminating loop)"
                    .into(),
            ));
        }
        match iteration(body, env, tensor_env, mode.clone())? {
            Flow::Next(value) => result = value,
            Flow::Continue => {}
            Flow::Break => return Ok(result),
        }
    }
}

/// `for var in collection { body }` over an array value.
///
/// The compiled path evaluates the collection handle and its length ONCE,
/// then reads `collection[idx]` at the start of each iteration. A store
/// `collection[i] = v` in the body goes through that same handle, so a later
/// iteration reads the stored element. When the collection is a variable, the
/// element is therefore read from the variable's CURRENT array each
/// iteration (an indexed store rebinds it here); any other collection
/// expression is evaluated once. A body that rebinds the collection variable
/// itself is refused: the compiled loop keeps walking the old handle, and an
/// interpreter that cannot tell the two apart must not guess.
pub(super) fn run_for_each(
    var: &str,
    collection: &Node,
    body: &[Node],
    env: &mut Env,
    tensor_env: &TensorEnv,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    // A loop variable named like its collection shadows it inside the body;
    // the compiled loop still walks its hidden handle, i.e. the snapshot.
    let live = match collection {
        Node::Lit(Literal::Ident(name), _) if name != var => Some(name.as_str()),
        _ => None,
    };
    if let Some(name) = live {
        if body.iter().any(|stmt| rebinds(stmt, name)) {
            return Err(EvalError::UnsupportedMsg(format!(
                "for-each body reassigns its collection `{name}`; the compiled loop \
                 keeps reading the handle it started with, which the interpreter \
                 does not model — iterate over a copy or index with a range `for`"
            )));
        }
    }
    let snapshot = eval_value_expr_mode(collection, env, tensor_env, mode.clone())?;
    let len = match &snapshot {
        Value::Tuple(items) => items.len(),
        other => {
            return Err(EvalError::UnsupportedMsg(format!(
                "for-each collection must be an array, got {}",
                super::assert_check::value_kind_name(other)
            )));
        }
    };
    with_loop_var(var, env, |env| {
        let mut result = Value::Int(0);
        for idx in 0..len {
            let element = match live.and_then(|name| env.get(name)) {
                Some(Value::Tuple(items)) => items.get(idx).cloned(),
                _ => match &snapshot {
                    Value::Tuple(items) => items.get(idx).cloned(),
                    _ => None,
                },
            };
            let Some(element) = element else {
                return Err(EvalError::BoundsTrap(format!(
                    "for-each index {idx} out of bounds of its collection"
                )));
            };
            env.insert(var.to_string(), element);
            match iteration(body, env, tensor_env, mode.clone())? {
                Flow::Next(value) => result = value,
                Flow::Continue => {}
                Flow::Break => break,
            }
        }
        Ok(result)
    })
}

/// Does `node` (or anything nested in it, short of a nested fn) assign the
/// whole variable `name`?
fn rebinds(node: &Node, name: &str) -> bool {
    match node {
        Node::Assign { name: target, .. } if target == name => true,
        Node::FnDef(..) => false,
        other => node_children_ref(other)
            .into_iter()
            .any(|child| rebinds(child, name)),
    }
}

/// The function-body fence: a `break` / `continue` that reached a function
/// boundary had no enclosing loop inside that function. It must never stop a
/// loop in the CALLER, so it becomes an ordinary evaluation error here.
pub(super) fn fence(err: EvalError) -> EvalError {
    match err {
        EvalError::BreakFlow => EvalError::UnsupportedMsg("`break` outside of a loop".into()),
        EvalError::ContinueFlow => EvalError::UnsupportedMsg("`continue` outside of a loop".into()),
        other => other,
    }
}
