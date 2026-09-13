// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Stable names used in type and shape diagnostics.

use crate::ast::BinOp;
use crate::types::{DType, ShapeDim, TensorType, ValueType};

pub(super) fn dtype_name(dtype: &DType) -> &'static str {
    match dtype {
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::Q16 => "q16",
    }
}

pub(super) fn format_shape(shape: &[ShapeDim]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(sym) => sym.to_string(),
        })
        .collect();
    format!("({})", dims.join(","))
}

pub(super) fn format_usize_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", dims.join(","))
}

pub(super) fn describe_tensor(tensor: &TensorType) -> String {
    format!(
        "Tensor[{}, {}]",
        dtype_name(&tensor.dtype),
        format_shape(&tensor.shape)
    )
}

pub(super) fn describe_value_type(v: &ValueType) -> String {
    match v {
        ValueType::ScalarI32 => "Scalar[i32]".to_string(),
        ValueType::ScalarI64 => "Scalar[i64]".to_string(),
        ValueType::ScalarF32 => "Scalar[f32]".to_string(),
        ValueType::ScalarF64 => "Scalar[f64]".to_string(),
        ValueType::ScalarBool => "Scalar[bool]".to_string(),
        ValueType::Ref { mutable, target } => {
            let m = if *mutable { "mut " } else { "" };
            let t = if target.is_empty() {
                "<ref>"
            } else {
                target.as_str()
            };
            format!("&{m}{t}")
        }
        ValueType::Tensor(tensor) => describe_tensor(tensor),
        ValueType::GradMap(entries) => {
            let mut parts = Vec::new();
            for (name, tensor) in entries {
                parts.push(format!("{}: {}", name, describe_tensor(tensor)));
            }
            format!("GradMap{{{}}}", parts.join(", "))
        }
    }
}

pub(super) fn dim_display(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

pub(super) fn binop_display(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
    }
}
