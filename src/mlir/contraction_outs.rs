// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Zero the output of a named linalg contraction.
//!
//! `linalg.matmul` and `linalg.conv_2d_nhwc_hwcf` add each product into
//! the output element. `tensor.empty` does not write 0. A sum into that
//! tensor includes the bytes `malloc` returned. Fill 0 before the
//! contraction. The caller passes the filled SSA name as `outs`.

use std::fmt::Write;

use crate::types::DType;

/// Append `arith.constant`, `tensor.empty` and `linalg.fill`. Return the filled name.
pub(super) fn zeroed_empty(body: &mut String, id: usize, dtype: &DType, tensor_ty: &str) -> String {
    let elem = dtype.as_str();
    let zero_lit = match dtype {
        DType::F64 | DType::F32 | DType::F16 | DType::BF16 => "0.0",
        _ => "0",
    };
    emit_line(
        body,
        &format!("    %z{id} = arith.constant {zero_lit} : {elem}"),
    );
    emit_line(
        body,
        &format!("    %tmp{id} = tensor.empty() : {tensor_ty}"),
    );
    emit_line(
        body,
        &format!(
            "    %acc{id} = linalg.fill ins(%z{id} : {elem}) \
             outs(%tmp{id} : {tensor_ty}) -> {tensor_ty}"
        ),
    );
    format!("%acc{id}")
}

fn emit_line(body: &mut String, line: &str) {
    writeln!(body, "{line}").expect("write to string cannot fail");
}
