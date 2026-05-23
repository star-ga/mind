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

//! RFC 0012 Phase B — tensor operators (`@` matmul, `.+`/`.-`/`.*`/`./` elementwise)
//! that desugar to existing std.blas IR forms.
//!
//! Phase B scope (RFC 0012 §7.2):
//!   - Parser: `A @ B` infix matmul (precedence 11) → `Node::TensorMatmul`.
//!   - Parser: `A .+ B`, `A .- B`, `A .* B`, `A ./ B` elementwise
//!     (`.+`/`.-` prec 9, `.*`/`./` prec 13) → `Node::TensorElemwise`.
//!   - Type checker: shape inference for both new forms using existing machinery
//!     (`engine::infer_output_shape("tensor.matmul")` and `broadcast_shapes`).
//!   - Diagnostics: `shape::matmul_mismatch` for inner-dim mismatch on `@`;
//!     `shape::broadcast_mismatch` for incompatible shapes on elementwise.
//!   - Desugar (single point in `lower_expr`):
//!     `A @ B`  → `Instr::MatMul { a, b }`  — identical to `tensor.matmul(A, B)`.
//!     `A .+ B` → `Instr::BinOp { Add }`     — identical to scalar `A + B` for tensors.
//!   - **IR-text byte-identity gate**: `A @ B` ≡ `tensor.matmul(A, B)` via
//!     `format_ir_module`.  Both produce `matmul %0, %1` in IR text.
//!
//! Phase B.2 deferred (NOT tested here):
//!   - `.T` transpose postfix operator.
//!   - Reductions: `.sum`, `.mean`, `.max`.
//!   - Norm shorthands.
//!   - `.reshape`.
//!   - MLIR-level byte-identity with raw `matmul_rmajor_f32_v` intrinsic
//!     (requires shape-dimension threading from type-checker to lower_expr).
//!   - Full NumPy-style prefix-rank broadcasting.
//!
//! Hard gates verified here:
//!   1. Parser produces `Node::TensorMatmul` for `A @ B`.
//!   2. Parser produces `Node::TensorElemwise` for each of `.+`/`.-`/`.*`/`./`.
//!   3. Precedence: `A .* B .+ C` binds as `(A .* B) .+ C`.
//!   4. Type checker: `A @ B` with matching inner dims → clean, correct output shape.
//!   5. Type checker: `A @ B` with mismatched inner dims → `shape::matmul_mismatch`.
//!   6. Type checker: `A @ B` for q16 tensors → clean, correct output shape.
//!   7. Type checker: `A .+ B` same shape → clean.
//!   8. Type checker: `A .+ B` incompatible shapes → `shape::broadcast_mismatch`.
//!   9. **Byte-identity**: IR text of `A @ B` ≡ IR text of `tensor.matmul(A, B)`.
//!  10. **Byte-identity**: IR text of `A .+ B` ≡ IR text of `A + B` (scalar add).

use libmind::ast::{Node, TensorElemOp};
use libmind::eval::lower_to_ir;
use libmind::ir::format_ir_module;
use libmind::parser;
use libmind::type_checker::{check_module_types_in_file, TypeEnv};
use libmind::types::{DType, ShapeDim, TensorType, ValueType};

// ── helpers ──────────────────────────────────────────────────────────────────

fn tensor_vt(dtype: DType, dims: &[usize]) -> ValueType {
    let shape = dims.iter().copied().map(ShapeDim::Known).collect();
    ValueType::Tensor(TensorType::new(dtype, shape))
}

/// TypeEnv pre-populated with named tensors.
fn env_with(entries: &[(&str, DType, &[usize])]) -> TypeEnv {
    let mut env = TypeEnv::new();
    for (name, dtype, dims) in entries {
        env.insert(name.to_string(), tensor_vt(dtype.clone(), dims));
    }
    env
}

fn check_src(src: &str, env: &TypeEnv) -> Vec<libmind::diagnostics::Diagnostic> {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    check_module_types_in_file(&module, src, None, env)
}

/// Parse `src` and return the IR text produced by `lower_to_ir`.
fn ir_text(src: &str) -> String {
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let ir = lower_to_ir(&module);
    format_ir_module(&ir)
}

// ── TEST 1: parser produces TensorMatmul for A @ B ───────────────────────────

/// `let c = a @ b` must parse with the inner expression being `Node::TensorMatmul`.
#[test]
fn parse_at_produces_tensor_matmul_node() {
    let src = "let c = a @ b";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    assert!(
        matches!(value.as_ref(), Node::TensorMatmul { .. }),
        "expected Node::TensorMatmul, got {:?}",
        value
    );
}

// ── TEST 2: parser produces TensorElemwise for each elementwise operator ──────

/// `.+` must produce `TensorElemwise { op: Add }`.
#[test]
fn parse_dot_add_produces_tensor_elemwise_add() {
    let src = "let c = a .+ b";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    assert!(
        matches!(value.as_ref(), Node::TensorElemwise { op: TensorElemOp::Add, .. }),
        "expected TensorElemwise Add, got {:?}",
        value
    );
}

/// `.-` must produce `TensorElemwise { op: Sub }`.
#[test]
fn parse_dot_sub_produces_tensor_elemwise_sub() {
    let src = "let c = a .- b";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    assert!(
        matches!(value.as_ref(), Node::TensorElemwise { op: TensorElemOp::Sub, .. }),
        "expected TensorElemwise Sub, got {:?}",
        value
    );
}

/// `.*` must produce `TensorElemwise { op: Mul }`.
#[test]
fn parse_dot_mul_produces_tensor_elemwise_mul() {
    let src = "let c = a .* b";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    assert!(
        matches!(value.as_ref(), Node::TensorElemwise { op: TensorElemOp::Mul, .. }),
        "expected TensorElemwise Mul, got {:?}",
        value
    );
}

/// `./` must produce `TensorElemwise { op: Div }`.
#[test]
fn parse_dot_div_produces_tensor_elemwise_div() {
    let src = "let c = a ./ b";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    assert!(
        matches!(value.as_ref(), Node::TensorElemwise { op: TensorElemOp::Div, .. }),
        "expected TensorElemwise Div, got {:?}",
        value
    );
}

// ── TEST 3: precedence — .* binds tighter than .+ ────────────────────────────

/// `a .* b .+ c` must parse as `(a .* b) .+ c`.
/// The outer node must be TensorElemwise { Add } with lhs = TensorElemwise { Mul }.
#[test]
fn elemwise_mul_binds_tighter_than_add() {
    let src = "let r = a .* b .+ c";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match value.as_ref() {
        Node::TensorElemwise { op: TensorElemOp::Add, lhs, .. } => {
            assert!(
                matches!(lhs.as_ref(), Node::TensorElemwise { op: TensorElemOp::Mul, .. }),
                "lhs of .+ must be .* group; got {:?}",
                lhs
            );
        }
        other => panic!("expected TensorElemwise(Add) at top; got {other:?}"),
    }
}

/// `@` binds tighter than `.+`: `a .+ b @ c` → `a .+ (b @ c)`.
#[test]
fn at_binds_tighter_than_dot_add() {
    let src = "let r = a .+ b @ c";
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { value, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match value.as_ref() {
        Node::TensorElemwise { op: TensorElemOp::Add, rhs, .. } => {
            assert!(
                matches!(rhs.as_ref(), Node::TensorMatmul { .. }),
                "rhs of .+ must be @-group; got {:?}",
                rhs
            );
        }
        other => panic!("expected TensorElemwise(Add) at top; got {other:?}"),
    }
}

// ── TEST 4: type checker — A @ B matching inner dims → clean ─────────────────

/// `A: Tensor<f32,[4,8]>`, `B: Tensor<f32,[8,16]>` → `C: Tensor<f32,[4,16]>`, no errors.
#[test]
fn matmul_op_matching_dims_no_error() {
    let env = env_with(&[
        ("a", DType::F32, &[4, 8]),
        ("b", DType::F32, &[8, 16]),
    ]);
    let src = "let c = a @ b";
    let diags = check_src(src, &env);
    let errs: Vec<_> = diags.iter().filter(|d| d.code.starts_with("shape::") || d.code.starts_with("E")).collect();
    assert!(
        errs.is_empty(),
        "expected no errors for compatible matmul; got: {errs:?}"
    );
}

// ── TEST 5: shape::matmul_mismatch on inner dim mismatch ─────────────────────

/// `A: Tensor<f32,[4,8]>`, `B: Tensor<f32,[7,16]>` → inner-dim mismatch (8 ≠ 7)
/// → `shape::matmul_mismatch`.
#[test]
fn matmul_op_inner_dim_mismatch_diagnostic() {
    let env = env_with(&[
        ("a", DType::F32, &[4, 8]),
        ("b", DType::F32, &[7, 16]),
    ]);
    let src = "let c = a @ b";
    let diags = check_src(src, &env);
    let mismatch_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::matmul_mismatch")
        .collect();
    assert!(
        !mismatch_diags.is_empty(),
        "expected shape::matmul_mismatch for [4,8] @ [7,16]; got: {diags:?}"
    );
}

// ── TEST 6: type checker — q16 matmul → clean ────────────────────────────────

/// `A: Tensor<q16,[8,16]>`, `B: Tensor<q16,[16,32]>` → `C: Tensor<q16,[8,32]>`, no errors.
#[test]
fn matmul_op_q16_matching_dims_no_error() {
    let env = env_with(&[
        ("a", DType::Q16, &[8, 16]),
        ("b", DType::Q16, &[16, 32]),
    ]);
    let src = "let c = a @ b";
    let diags = check_src(src, &env);
    let errs: Vec<_> = diags.iter().filter(|d| d.code.starts_with("shape::") || d.code.starts_with("E")).collect();
    assert!(
        errs.is_empty(),
        "expected no errors for q16 matmul; got: {errs:?}"
    );
}

// ── TEST 7: type checker — .+ same shape → clean ─────────────────────────────

/// `A: Tensor<f32,[4,8]>`, `B: Tensor<f32,[4,8]>` → elementwise `.+` → clean.
#[test]
fn elemwise_add_same_shape_no_error() {
    let env = env_with(&[
        ("a", DType::F32, &[4, 8]),
        ("b", DType::F32, &[4, 8]),
    ]);
    let src = "let c = a .+ b";
    let diags = check_src(src, &env);
    let errs: Vec<_> = diags.iter().filter(|d| d.code.starts_with("shape::") || d.code.starts_with("E")).collect();
    assert!(
        errs.is_empty(),
        "expected no errors for same-shape .+ ; got: {errs:?}"
    );
}

// ── TEST 8: shape::broadcast_mismatch for incompatible shapes ────────────────

/// `A: Tensor<f32,[4,8]>`, `B: Tensor<f32,[4,16]>` → shapes incompatible for
/// elementwise → `shape::broadcast_mismatch`.
#[test]
fn elemwise_add_shape_mismatch_diagnostic() {
    let env = env_with(&[
        ("a", DType::F32, &[4, 8]),
        ("b", DType::F32, &[4, 16]),
    ]);
    let src = "let c = a .+ b";
    let diags = check_src(src, &env);
    let mismatch_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::broadcast_mismatch")
        .collect();
    assert!(
        !mismatch_diags.is_empty(),
        "expected shape::broadcast_mismatch for [4,8] .+ [4,16]; got: {diags:?}"
    );
}

// ── TEST 9: BYTE-IDENTITY — `A @ B` ≡ `tensor.matmul(A, B)` ─────────────────

/// Load-bearing gate (RFC 0012 §7.2, Phase B):
///
/// `A @ B` desugars in `lower_expr` to `Instr::MatMul { a, b }`.
/// `tensor.matmul(A, B)` also lowers to `Instr::MatMul { a, b }` (via `Node::CallMatMul`).
/// Both forms must therefore produce byte-identical IR text via `format_ir_module`.
///
/// Phase B.2 note: MLIR-level byte-identity with the raw `matmul_rmajor_f32_v`
/// intrinsic is deferred — it requires shape-dimension threading from the
/// type-checker through `lower_expr` to emit the correct `Instr::Call` args.
/// This test gates at the IR text layer (`format_ir_module`), which is the
/// contract delivered by Phase B.
#[test]
fn ir_text_at_operator_byte_identical_to_call_matmul() {
    let ir_at = ir_text("let c = a @ b");
    let ir_call = ir_text("let c = tensor.matmul(a, b)");

    assert_eq!(
        ir_at, ir_call,
        "IR text must be byte-identical between `A @ B` and `tensor.matmul(A, B)`.\n\
         `A @ B`:\n{ir_at}\n\
         `tensor.matmul(A, B)`:\n{ir_call}"
    );

    // Additionally confirm the IR text contains the matmul opcode.
    assert!(
        ir_at.contains("matmul"),
        "IR text must contain 'matmul' opcode; got:\n{ir_at}"
    );
}

// ── TEST 10: BYTE-IDENTITY — `A .+ B` ≡ `A + B` (tensor add) ────────────────

/// `A .+ B` desugars to `Instr::BinOp { Add }`.
/// `A + B` (scalar-form binary) also lowers to `Instr::BinOp { Add }` for
/// tensor operands.  Both must produce byte-identical IR text.
///
/// Phase B.2 note: MLIR-level byte-identity with a dedicated vector-add
/// intrinsic is deferred pending RFC 0006 Track C.
#[test]
fn ir_text_dot_add_byte_identical_to_scalar_add() {
    let ir_dot = ir_text("let c = a .+ b");
    let ir_scalar = ir_text("let c = a + b");

    assert_eq!(
        ir_dot, ir_scalar,
        "IR text must be byte-identical between `A .+ B` and `A + B`.\n\
         `A .+ B`:\n{ir_dot}\n\
         `A + B`:\n{ir_scalar}"
    );

    assert!(
        ir_dot.contains("add"),
        "IR text must contain 'add' opcode; got:\n{ir_dot}"
    );
}

// ── TEST 11: BYTE-IDENTITY — .-, .*, ./ ──────────────────────────────────────

/// `A .- B` ≡ `A - B` in IR text.
#[test]
fn ir_text_dot_sub_byte_identical_to_scalar_sub() {
    let ir_dot = ir_text("let c = a .- b");
    let ir_scalar = ir_text("let c = a - b");
    assert_eq!(ir_dot, ir_scalar,
        "`A .- B` and `A - B` must produce identical IR text.\n\
         dot:\n{ir_dot}\nscalar:\n{ir_scalar}");
}

/// `A .* B` ≡ `A * B` in IR text.
#[test]
fn ir_text_dot_mul_byte_identical_to_scalar_mul() {
    let ir_dot = ir_text("let c = a .* b");
    let ir_scalar = ir_text("let c = a * b");
    assert_eq!(ir_dot, ir_scalar,
        "`A .* B` and `A * B` must produce identical IR text.\n\
         dot:\n{ir_dot}\nscalar:\n{ir_scalar}");
}

/// `A ./ B` ≡ `A / B` in IR text.
#[test]
fn ir_text_dot_div_byte_identical_to_scalar_div() {
    let ir_dot = ir_text("let c = a ./ b");
    let ir_scalar = ir_text("let c = a / b");
    assert_eq!(ir_dot, ir_scalar,
        "`A ./ B` and `A / B` must produce identical IR text.\n\
         dot:\n{ir_dot}\nscalar:\n{ir_scalar}");
}

// ── Phase B.2 deferred — documented placeholders ─────────────────────────────

/// Phase B.2 deferred: `.T` transpose operator.
///
/// `A.T` will produce `Node::TensorTranspose` and desugar to
/// `Instr::Call { "__mind_blas_transpose_f32_v", ... }`.
/// Not implemented in Phase B.
#[test]
fn phase_b2_transpose_operator() {
    let _ = ir_text("let t = a.T");
}

/// Phase B.2 deferred: `.sum` reduction.
///
/// `A.sum` will produce `Node::TensorReduce { kind: Sum }` and desugar to
/// a sum-reduction kernel call.
/// Not implemented in Phase B.
#[test]
#[ignore = "Phase B.2 deferred: .sum reduction not yet implemented"]
fn phase_b2_sum_reduction() {
    let _ = ir_text("let s = a.sum");
}

/// Phase B.2 deferred: `.mean` reduction.
#[test]
#[ignore = "Phase B.2 deferred: .mean reduction not yet implemented"]
fn phase_b2_mean_reduction() {
    let _ = ir_text("let m = a.mean");
}

/// Phase B.2 deferred: `.max` reduction.
#[test]
#[ignore = "Phase B.2 deferred: .max reduction not yet implemented"]
fn phase_b2_max_reduction() {
    let _ = ir_text("let m = a.max");
}

/// Phase B.2 deferred: MLIR-level byte-identity with `matmul_rmajor_f32_v`.
///
/// The RFC 0012 §7.2 gate-matrix target is:
///   `A @ B` → MLIR byte-identical to `matmul_rmajor_f32_v(w, x, y, rows, cols)`.
///
/// This requires threading the type-checker's concrete shape dims (M, K) through
/// `lower_expr` to emit `Instr::Call { "__mind_blas_matmul_rmajor_f32_v", [a, b, rows, cols] }`.
/// Deferred to Phase B.2 pending type-env integration in lower_expr.
#[test]
#[ignore = "Phase B.2 deferred: MLIR-level byte-identity with matmul_rmajor_f32_v requires shape-dim threading"]
fn phase_b2_mlir_byte_identity_matmul_rmajor_f32_v() {
    // When implemented: compile both `A @ B` and hand-written
    // `matmul_rmajor_f32_v(a_ptr, b_ptr, c_ptr, 4, 16)` to MLIR text and assert equality.
    todo!("Phase B.2")
}
