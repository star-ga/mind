// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Interpreter model of native string records, kept in step with lowering.
//!
//! Compiled code passes every string as a pointer to a `[addr | len | cap]`
//! record. The interpreter holds a string in three forms: a Rust-side
//! [`Value::Str`] literal, a tagged arena record (`interp_mem::materialize_str`,
//! made at a `string` parameter), and a std `String` struct value. This module
//! gives the interpreter the compiled meaning of the operations that cross
//! those forms:
//!
//! * the module is normalized by the same `string_equality` pass lowering runs,
//!   so `==`/`!=` on two strings becomes `__mind_string_eq` (byte-wise) here
//!   too, and a refused comparison fails with the same `E2038` text;
//! * `__mind_string_eq` compares bytes exactly like the runtime C function;
//! * a string bound to a `String` parameter becomes the std `String` struct
//!   value read from its record, so std.string functions (`string_len`,
//!   `string_slice`, `string_eq`, ...) execute on it as compiled code does.

use std::collections::HashMap;

use super::autodiff::TensorEnvEntry;
use super::{EvalError, ExecMode, Value, eval_value_expr_mode, interp_mem};
use crate::ast::{Module, Node, TypeAnn};

/// Apply the shared string-equality normalization to a module about to be
/// interpreted. A refused comparison is an evaluation error carrying the
/// check-time code, so `mindc test` and `mindc build` refuse the same source.
#[cfg(feature = "std-surface")]
pub(crate) fn normalize_module(module: &Module) -> Result<Option<Module>, EvalError> {
    let analysis = super::string_equality::analyze(module);
    if let Some(refused) = analysis.refused.first() {
        return Err(EvalError::UnsupportedMsg(format!(
            "[{}] {}. {}",
            crate::type_checker::lowering_refusals::E_STRING_EQUALITY_OPERAND,
            super::string_equality::refusal_message(refused.op),
            super::string_equality::REFUSAL_HELP
        )));
    }
    Ok(analysis.rewritten)
}

/// Without the std surface no string comparison is rewritten by lowering
/// either, so the interpreter evaluates the module as written.
#[cfg(not(feature = "std-surface"))]
pub(crate) fn normalize_module(_module: &Module) -> Result<Option<Module>, EvalError> {
    Ok(None)
}

fn load_i64(addr: i64) -> Result<i64, EvalError> {
    interp_mem::eval_intrinsic("__mind_load_i64", &[addr]).map_err(EvalError::UnsupportedMsg)
}

/// The `[addr, len, cap]` fields of a tagged string record.
#[cfg(feature = "std-surface")]
fn record_fields(rec: i64) -> Result<[i64; 3], EvalError> {
    Ok([load_i64(rec)?, load_i64(rec + 8)?, load_i64(rec + 16)?])
}

/// The std `String` struct value with the given record fields.
#[cfg(feature = "std-surface")]
fn string_struct([addr, len, cap]: [i64; 3]) -> Value {
    Value::Struct {
        name: "String".to_string(),
        fields: vec![
            ("addr".to_string(), Value::Int(addr)),
            ("len".to_string(), Value::Int(len)),
            ("cap".to_string(), Value::Int(cap)),
        ],
    }
}

/// Convert an argument at a parameter declared `string` or `String`.
///
/// A `string` parameter receives a native record, as before. A `String`
/// parameter receives the std struct value that compiled code reads through
/// the same record, so its `.addr`/`.len`/`.cap` fields resolve. Values that
/// are not known strings pass unchanged.
pub(crate) fn bind_param(ty: &TypeAnn, value: Value) -> Result<Value, EvalError> {
    let TypeAnn::Named(name) = ty else {
        return Ok(value);
    };
    match (name.as_str(), value) {
        ("string", Value::Str(text)) => interp_mem::materialize_str(&text)
            .map(Value::Int)
            .map_err(EvalError::UnsupportedMsg),
        #[cfg(feature = "std-surface")]
        ("String", Value::Str(text)) => {
            let rec = interp_mem::materialize_str(&text).map_err(EvalError::UnsupportedMsg)?;
            Ok(string_struct(record_fields(rec)?))
        }
        #[cfg(feature = "std-surface")]
        ("String", Value::Int(rec)) if interp_mem::is_str_record(rec) => {
            Ok(string_struct(record_fields(rec)?))
        }
        (_, value) => Ok(value),
    }
}

/// The bytes of one string operand of `__mind_string_eq`.
fn operand_bytes(value: &Value) -> Result<Vec<u8>, EvalError> {
    let (addr, len) = match value {
        Value::Str(text) => return Ok(text.as_bytes().to_vec()),
        Value::Int(rec) => (load_i64(*rec)?, load_i64(rec + 8)?),
        Value::Struct { name, fields } if name == "String" => {
            let field = |wanted: &str| match fields.iter().find(|(f, _)| f == wanted) {
                Some((_, Value::Int(n))) => Ok(*n),
                _ => Err(EvalError::UnsupportedMsg(format!(
                    "__mind_string_eq: `String` value has no integer field `{wanted}`"
                ))),
            };
            (field("addr")?, field("len")?)
        }
        other => {
            return Err(EvalError::UnsupportedMsg(format!(
                "__mind_string_eq: operand is not a string: {other:?}"
            )));
        }
    };
    if len <= 0 {
        return Ok(Vec::new());
    }
    (0..len)
        .map(|i| {
            interp_mem::eval_intrinsic("__mind_load_i8", &[addr + i])
                .map(|b| b as u8)
                .map_err(EvalError::UnsupportedMsg)
        })
        .collect()
}

/// `__mind_string_eq(a, b)`: 1 when both strings hold the same bytes, else 0
/// (the runtime `mind_string_eq_impl`: lengths first, then every byte).
pub(crate) fn eval_string_eq(
    args: &[Node],
    env: &HashMap<String, Value>,
    tensor_env: &HashMap<String, TensorEnvEntry>,
    mode: ExecMode,
) -> Result<Value, EvalError> {
    let [left, right] = args else {
        return Err(EvalError::UnsupportedMsg(format!(
            "__mind_string_eq: expected 2 arguments, got {}",
            args.len()
        )));
    };
    let left = eval_value_expr_mode(left, env, tensor_env, mode.clone())?;
    let right = eval_value_expr_mode(right, env, tensor_env, mode)?;
    let equal = operand_bytes(&left)? == operand_bytes(&right)?;
    Ok(Value::Int(i64::from(equal)))
}

#[cfg(all(test, feature = "std-surface"))]
mod tests {
    use super::*;

    fn eq(left: Value, right: Value) -> i64 {
        i64::from(operand_bytes(&left).unwrap() == operand_bytes(&right).unwrap())
    }

    #[test]
    fn every_string_form_compares_by_bytes() {
        interp_mem::reset();
        let rec_a = interp_mem::materialize_str("abc").unwrap();
        let rec_b = interp_mem::materialize_str("abc").unwrap();
        assert_ne!(
            rec_a, rec_b,
            "two records, so a handle compare would differ"
        );
        assert_eq!(eq(Value::Int(rec_a), Value::Int(rec_b)), 1);
        assert_eq!(eq(Value::Int(rec_a), Value::Str("abc".into())), 1);
        assert_eq!(eq(Value::Int(rec_a), Value::Str("abd".into())), 0);
        assert_eq!(eq(Value::Int(rec_a), Value::Str("ab".into())), 0);
        let view = string_struct(record_fields(rec_b).unwrap());
        assert_eq!(eq(view, Value::Str("abc".into())), 1);
        assert_eq!(eq(Value::Str(String::new()), string_struct([0, 0, 0])), 1);
        assert!(operand_bytes(&Value::Float(1.0)).is_err());
    }

    #[test]
    fn string_parameters_bind_records_and_string_structs() {
        interp_mem::reset();
        let string_ty = TypeAnn::Named("string".into());
        let Value::Int(rec) = bind_param(&string_ty, Value::Str("hey".into())).unwrap() else {
            panic!("a `string` parameter binds a native record");
        };
        assert!(interp_mem::is_str_record(rec));
        {
            let struct_ty = TypeAnn::Named("String".into());
            let expected = string_struct(record_fields(rec).unwrap());
            assert_eq!(bind_param(&struct_ty, Value::Int(rec)).unwrap(), expected);
            let Value::Struct { fields, .. } =
                bind_param(&struct_ty, Value::Str("hey".into())).unwrap()
            else {
                panic!("a `String` parameter binds the std struct value");
            };
            assert_eq!(fields[1], ("len".to_string(), Value::Int(3)));
        }
        let untouched = bind_param(&TypeAnn::ScalarI64, Value::Str("x".into())).unwrap();
        assert_eq!(untouched, Value::Str("x".into()));
    }
}
