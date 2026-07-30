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

//! Finding #273 — a `tensor.reshape` dimension that names a *runtime value*
//! binding (e.g. `let n = 6; tensor.reshape(x, (2, n))`) is a silent
//! miscompile: the parser stores the identifier as a shape SYMBOL, so the
//! runtime value is discarded, the element-count check is bypassed (a symbolic
//! product is unknown), and `lower_expr` emits `Instr::Reshape` with a bogus
//! `ShapeDim::Sym("n")` dim. This must fail closed at CHECK time with the
//! `shape::reshape_runtime_dim` diagnostic — a reshape dim must be a
//! compile-time-known extent (integer literal or a static shape symbol).
//!
//! A legitimate static shape symbol (`batch`, in scope from a tensor type) is
//! NOT a value binding, so it is absent from the type env and never flagged —
//! the discriminator that keeps this fix from false-positiving.

use libmind::parser;
use libmind::type_checker::check_module_types_in_file;

fn diag_codes(src: &str) -> Vec<String> {
    let module = parser::parse(src).expect("parse");
    check_module_types_in_file(&module, src, None, &Default::default())
        .into_iter()
        .map(|d| d.code.to_string())
        .collect()
}

/// The bug: a runtime `i64` param used as a reshape extent must be rejected.
#[test]
fn reshape_runtime_dim_rejected() {
    let src = "\
fn f(x: tensor<f32[2, 3]>, n: i64) -> tensor<f32[2, 3]> {
    let y = tensor.reshape(x, (2, n));
    y
}
";
    let codes = diag_codes(src);
    assert!(
        codes.iter().any(|c| c == "shape::reshape_runtime_dim"),
        "expected shape::reshape_runtime_dim for a runtime-value reshape dim; got: {codes:?}"
    );
}

/// A static shape symbol (`batch`, from the param's tensor type) is a valid
/// compile-time-known extent and must NOT be flagged.
#[test]
fn reshape_static_shape_symbol_ok() {
    let src = "\
fn g(x: tensor<f32[batch, 6]>) -> tensor<f32[batch, 6]> {
    let y = tensor.reshape(x, (batch, 6));
    y
}
";
    let codes = diag_codes(src);
    assert!(
        !codes.iter().any(|c| c == "shape::reshape_runtime_dim"),
        "static shape symbol `batch` must not be flagged; got: {codes:?}"
    );
}

/// Integer-literal extents (the ordinary case) must never be flagged.
#[test]
fn reshape_numeric_dims_ok() {
    let src = "\
fn h(x: tensor<f32[2, 3]>) -> tensor<f32[3, 2]> {
    let y = tensor.reshape(x, (3, 2));
    y
}
";
    let codes = diag_codes(src);
    assert!(
        !codes.iter().any(|c| c == "shape::reshape_runtime_dim"),
        "numeric reshape dims must not be flagged; got: {codes:?}"
    );
}
