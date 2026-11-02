// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Scoped assertion checking for the interpreter-backed test runner.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::ast::Node;

use super::{EvalError, ExecMode, TensorEnvEntry, Value};

#[derive(Default)]
struct State {
    enabled: bool,
    failure: Option<String>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Restores the prior thread-local state when a test evaluation ends or unwinds.
pub(crate) struct Guard {
    previous: Option<State>,
}

/// Enable assertion checking for one interpreter evaluation on this thread.
pub(crate) fn enter() -> Guard {
    let previous = STATE.with(|state| {
        std::mem::replace(
            &mut *state.borrow_mut(),
            State {
                enabled: true,
                failure: None,
            },
        )
    });
    Guard {
        previous: Some(previous),
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            STATE.with(|state| *state.borrow_mut() = previous);
        }
    }
}

fn enabled() -> bool {
    STATE.with(|state| state.borrow().enabled)
}

fn fail(message: String) -> EvalError {
    STATE.with(|state| state.borrow_mut().failure = Some(message));
    EvalError::Unsupported
}

pub(crate) fn take_failure() -> Option<String> {
    STATE.with(|state| state.borrow_mut().failure.take())
}

/// Evaluate an assertion only while the test runner's scoped guard is active.
pub(super) fn eval(
    cond: &Node,
    message: &Option<String>,
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    if !enabled() {
        return Ok(Value::Int(0));
    }
    let failure = |default: &str| fail(message.clone().unwrap_or_else(|| default.to_string()));
    match super::eval_value_expr_mode(cond, env, tensor_env, mode)? {
        Value::Int(0) => Err(failure("assertion failed")),
        Value::Float(0.0) => Err(failure("assertion failed (float)")),
        Value::Int(_) | Value::Float(_) => Ok(Value::Int(0)),
        Value::Tuple(items) => Err(EvalError::UnsupportedMsg(format!(
            "assert condition is a {}-tuple, not a boolean: \
             `assert(cond, \"msg\")` parses the parenthesised pair as one \
             tuple expression; write `assert cond, \"msg\"`",
            items.len()
        ))),
        other => Err(EvalError::UnsupportedMsg(format!(
            "assert condition must evaluate to a boolean or integer, got {}",
            value_kind_name(&other)
        ))),
    }
}

pub(super) fn value_kind_name(value: &Value) -> &'static str {
    match value {
        Value::Int(_) => "an integer",
        Value::Float(_) => "a float",
        Value::Str(_) => "a string",
        Value::Tuple(_) => "a tuple",
        Value::Tensor(_) => "a tensor",
        Value::GradMap(_) => "a gradient map",
        Value::Enum { .. } => "an enum value",
        Value::Struct { .. } => "a struct value",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_and_unwinding_guards_restore_prior_state() {
        assert!(!enabled());
        {
            let _outer = enter();
            assert!(enabled());
            let _ = fail("outer".into());
            {
                let _inner = enter();
                assert!(enabled());
                assert_eq!(take_failure(), None);
            }
            assert_eq!(take_failure().as_deref(), Some("outer"));

            let unwind = std::panic::catch_unwind(|| {
                let _inner = enter();
                panic!("probe");
            });
            assert!(unwind.is_err());
            assert!(enabled());
        }
        assert!(!enabled());
        assert_eq!(take_failure(), None);
    }
}
