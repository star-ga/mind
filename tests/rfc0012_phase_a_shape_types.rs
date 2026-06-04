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

//! RFC 0012 Phase A — shape-typed `Tensor<dtype, [dims]>` type system + compile-time
//! shape checking.
//!
//! Phase A scope (RFC 0012 §7.1):
//!   - Parser: `Tensor<dtype, [dims]>` type annotation — confirmed/extended.
//!   - `q16` as a first-class dtype keyword in parser + type checker.
//!   - `shape::rank_mismatch`, `shape::dim_mismatch`, `shape::dtype_mismatch`,
//!     and `shape::symbolic_conflict` diagnostics via `diag_from_span` + `errs.push`.
//!   - Symbolic dim unification by name across a function's parameter list.
//!   - No MLIR codegen change (shape type is compile-time-only; lowering unchanged).
//!
//! Hard gates verified here:
//!   1. Parser round-trips `Tensor<f32, [4, 8]>` → `TypeAnn::Tensor { dtype: "f32", dims: ["4", "8"] }`.
//!   2. Parser round-trips `Tensor<q16, [16]>` — q16 first-class dtype.
//!   3. `shape::dim_mismatch`: `[4, 8]` annotation vs `[4, 16]` inferred → diagnostic.
//!   4. `shape::rank_mismatch`: `[M, N]` annotation vs `[M, N, K]` inferred → diagnostic.
//!   5. `shape::dtype_mismatch`: `Tensor<f32,...>` vs `Tensor<q16,...>` → diagnostic.
//!   6. Symbolic dim unification: `fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>)` —
//!      call with matching N → clean; call with mismatched N → `shape::symbolic_conflict`.
//!   7. Lowering unchanged: a `Tensor<f32,[4,8]>` annotation does not alter the
//!      `ValueType::Tensor` produced; the shape type is a compile-time lens only.

use libmind::ast::{Literal, Module, Node, Param, Span, TypeAnn};
use libmind::parser;
use libmind::type_checker::{TypeEnv, check_module_types_in_file};
use libmind::types::{DType, ShapeDim, TensorType, ValueType};

// ── helpers ──────────────────────────────────────────────────────────────────

fn sp() -> Span {
    Span::new(0, 0)
}

fn check_module(
    module: &Module,
    src: &str,
    env: &TypeEnv,
) -> Vec<libmind::diagnostics::Diagnostic> {
    check_module_types_in_file(module, src, None, env)
}

fn tensor_vt(dtype: DType, shape: Vec<ShapeDim>) -> ValueType {
    ValueType::Tensor(TensorType::new(dtype, shape))
}

/// Build an env pre-populated with named tensors so tests can use
/// `Literal::Ident("name")` as the RHS without needing the `tensor.zeros` builtin.
fn env_with(entries: &[(&str, DType, &[usize])]) -> TypeEnv {
    let mut env = TypeEnv::new();
    for (name, dtype, dims) in entries {
        let shape = dims.iter().copied().map(ShapeDim::Known).collect();
        env.insert(name.to_string(), tensor_vt(dtype.clone(), shape));
    }
    env
}

/// Build a `let name: Tensor<dtype, [dims]> = ident_rhs` AST node.
fn let_tensor(name: &str, ann_dtype: &str, ann_dims: &[&str], rhs_ident: &str) -> Node {
    Node::Let {
        name: name.to_string(),
        mutable: false,
        ann: Some(TypeAnn::Tensor {
            dtype: ann_dtype.to_string(),
            dims: ann_dims.iter().map(|s| s.to_string()).collect(),
        }),
        value: Box::new(Node::Lit(Literal::Ident(rhs_ident.to_string()), sp())),
        span: sp(),
    }
}

// ── TEST 1: parser round-trips Tensor<f32, [4, 8]> ──────────────────────────

/// `Tensor<f32, [4, 8]>` must parse to `TypeAnn::Tensor { dtype: "f32", dims: ["4", "8"] }`.
#[test]
fn parse_tensor_f32_concrete_dims() {
    let src = r#"let x: Tensor<f32, [4, 8]> = y"#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { ann, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match ann.as_ref().expect("annotation present") {
        TypeAnn::Tensor { dtype, dims } => {
            assert_eq!(dtype, "f32", "dtype must be f32");
            assert_eq!(dims, &["4", "8"], "dims must be [4, 8]");
        }
        other => panic!("expected TypeAnn::Tensor, got {other:?}"),
    }
}

// ── TEST 2: parser accepts q16 dtype ─────────────────────────────────────────

/// `Tensor<q16, [16]>` must parse to `TypeAnn::Tensor { dtype: "q16", dims: ["16"] }`.
#[test]
fn parse_tensor_q16_dtype() {
    let src = r#"let v: Tensor<q16, [16]> = q_vec"#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { ann, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match ann.as_ref().expect("annotation present") {
        TypeAnn::Tensor { dtype, dims } => {
            assert_eq!(dtype, "q16", "dtype must be q16");
            assert_eq!(dims, &["16"], "dims must be [16]");
        }
        other => panic!("expected TypeAnn::Tensor, got {other:?}"),
    }
}

// ── TEST 3: parser accepts empty dims (rank-0 scalar tensor) ─────────────────

/// `Tensor<f32, []>` (rank-0) must parse cleanly.
#[test]
fn parse_tensor_rank_zero() {
    let src = r#"let s: Tensor<f32, []> = scalar_val"#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { ann, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match ann.as_ref().expect("annotation present") {
        TypeAnn::Tensor { dtype, dims } => {
            assert_eq!(dtype, "f32");
            assert!(dims.is_empty(), "rank-0 tensor must have empty dims list");
        }
        other => panic!("expected TypeAnn::Tensor, got {other:?}"),
    }
}

// ── TEST 4: parser accepts symbolic dims ─────────────────────────────────────

/// `Tensor<f32, [N, K]>` must parse with dims = ["N", "K"].
#[test]
fn parse_tensor_symbolic_dims() {
    let src = r#"let a: Tensor<f32, [N, K]> = some_tensor"#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { ann, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match ann.as_ref().expect("annotation present") {
        TypeAnn::Tensor { dtype, dims } => {
            assert_eq!(dtype, "f32");
            assert_eq!(dims, &["N", "K"]);
        }
        other => panic!("expected TypeAnn::Tensor, got {other:?}"),
    }
}

// ── TEST 5: i64 dtype is now recognized ──────────────────────────────────────

/// `Tensor<i64, [8]>` must parse cleanly (i64 added alongside q16 in RFC 0012 §3.2).
#[test]
fn parse_tensor_i64_dtype() {
    let src = r#"let t: Tensor<i64, [8]> = int_tensor"#;
    let module = parser::parse(src).unwrap_or_else(|e| panic!("parse error: {e:?}"));
    let Node::Let { ann, .. } = &module.items[0] else {
        panic!("expected Let node");
    };
    match ann.as_ref().expect("annotation present") {
        TypeAnn::Tensor { dtype, dims } => {
            assert_eq!(dtype, "i64");
            assert_eq!(dims, &["8"]);
        }
        other => panic!("expected TypeAnn::Tensor, got {other:?}"),
    }
}

// ── TEST 6: shape::dim_mismatch ───────────────────────────────────────────────

/// Annotating a binding with `Tensor<f32, [4, 16]>` but the env supplies
/// `Tensor<f32, [4, 8]>` must produce exactly one `shape::dim_mismatch` diagnostic.
#[test]
fn shape_dim_mismatch_diagnostic() {
    let module = Module {
        items: vec![let_tensor("x", "f32", &["4", "16"], "src")],
    };
    // src: Tensor<f32,[4,8]> — dim 1 mismatch (annotation=16, inferred=8)
    let env = env_with(&[("src", DType::F32, &[4, 8])]);
    let src = "let x: Tensor<f32, [4, 16]> = src";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::dim_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::dim_mismatch diagnostic; got: {diags:?}"
    );
    let msg = &shape_diags[0].message;
    assert!(
        msg.contains("16") || msg.contains("8"),
        "diagnostic message must mention the conflicting sizes; got: {msg}"
    );
}

// ── TEST 7: shape::rank_mismatch ─────────────────────────────────────────────

/// Annotating with a rank-3 annotation but env supplies rank-2 → `shape::rank_mismatch`.
#[test]
fn shape_rank_mismatch_diagnostic() {
    let module = Module {
        items: vec![let_tensor("x", "f32", &["4", "8", "16"], "src")],
    };
    // src: Tensor<f32,[4,8]> — rank 2 vs annotation rank 3
    let env = env_with(&[("src", DType::F32, &[4, 8])]);
    let src = "let x: Tensor<f32, [4, 8, 16]> = src";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::rank_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::rank_mismatch diagnostic; got: {diags:?}"
    );
    let msg = &shape_diags[0].message;
    assert!(
        msg.contains("rank"),
        "diagnostic message must mention rank; got: {msg}"
    );
}

// ── TEST 8: shape::dtype_mismatch (f32 annotation, q16 inferred) ─────────────

/// Annotating with `Tensor<f32,[4]>` but env supplies `Tensor<q16,[4]>` →
/// `shape::dtype_mismatch`.
#[test]
fn shape_dtype_mismatch_f32_ann_q16_inferred() {
    let module = Module {
        items: vec![let_tensor("x", "f32", &["4"], "q_src")],
    };
    let env = env_with(&[("q_src", DType::Q16, &[4])]);
    let src = "let x: Tensor<f32, [4]> = q_src";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::dtype_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::dtype_mismatch for f32 ann vs q16 inferred; got: {diags:?}"
    );
    let msg = &shape_diags[0].message;
    assert!(
        msg.contains("q16") || msg.contains("f32"),
        "diagnostic message must mention the conflicting dtypes; got: {msg}"
    );
}

// ── TEST 9: shape::dtype_mismatch (q16 annotation, f32 inferred) ─────────────

/// Annotating with `Tensor<q16,[4]>` but env supplies `Tensor<f32,[4]>` →
/// `shape::dtype_mismatch`.
#[test]
fn shape_dtype_mismatch_q16_ann_f32_inferred() {
    let module = Module {
        items: vec![let_tensor("x", "q16", &["4"], "f32_src")],
    };
    let env = env_with(&[("f32_src", DType::F32, &[4])]);
    let src = "let x: Tensor<q16, [4]> = f32_src";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::dtype_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::dtype_mismatch for q16 ann vs f32 inferred; got: {diags:?}"
    );
}

// ── TEST 10: compatible tensor binding — no diagnostics ──────────────────────

/// `let x: Tensor<f32, [4, 8]> = src` where src: Tensor<f32,[4,8]> must produce
/// zero shape diagnostics.
#[test]
fn compatible_tensor_binding_no_diag() {
    let module = Module {
        items: vec![let_tensor("x", "f32", &["4", "8"], "src")],
    };
    let env = env_with(&[("src", DType::F32, &[4, 8])]);
    let src = "let x: Tensor<f32, [4, 8]> = src";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code.starts_with("shape::"))
        .collect();
    assert!(
        shape_diags.is_empty(),
        "expected no shape diagnostics for compatible binding; got: {shape_diags:?}"
    );
}

// ── TEST 11: compatible q16 binding — no diagnostics ─────────────────────────

/// `let x: Tensor<q16, [8]> = q_vec` where q_vec: Tensor<q16,[8]> — clean.
#[test]
fn q16_compatible_binding_no_diag() {
    let module = Module {
        items: vec![let_tensor("x", "q16", &["8"], "q_vec")],
    };
    let env = env_with(&[("q_vec", DType::Q16, &[8])]);
    let src = "let x: Tensor<q16, [8]> = q_vec";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code.starts_with("shape::"))
        .collect();
    assert!(
        shape_diags.is_empty(),
        "q16 compatible binding must produce no shape diagnostics; got: {shape_diags:?}"
    );
}

// ── TEST 12: symbolic dim unification — same N, clean call ───────────────────

/// `fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>)` called with
/// x: Tensor<f32,[4,8]> and y: Tensor<f32,[4,16]> — N=4 both → no conflict.
#[test]
fn symbolic_dim_same_n_no_conflict() {
    // Build the module AST directly: FnDef + two let + call.
    let fn_node = Node::FnDef {
        type_params: vec![],
        is_pub: false,
        is_test: false,
        name: "f".to_string(),
        params: vec![
            Param {
                name: "a".to_string(),
                ty: TypeAnn::Tensor {
                    dtype: "f32".to_string(),
                    dims: vec!["N".to_string(), "K".to_string()],
                },
                span: sp(),
            },
            Param {
                name: "b".to_string(),
                ty: TypeAnn::Tensor {
                    dtype: "f32".to_string(),
                    dims: vec!["N".to_string(), "M".to_string()],
                },
                span: sp(),
            },
        ],
        ret_type: Some(TypeAnn::ScalarI32),
        body: vec![Node::Return {
            value: Some(Box::new(Node::Lit(Literal::Int(0), sp()))),
            span: sp(),
        }],
        reap_threshold: None,
        attrs: Vec::new(),
        span: sp(),
    };
    // Let x = Tensor<f32,[4,8]>, y = Tensor<f32,[4,16]> — N matches (4 == 4)
    let let_x = Node::Let {
        name: "x".to_string(),
        mutable: false,
        ann: None,
        value: Box::new(Node::Lit(Literal::Ident("x_src".to_string()), sp())),
        span: sp(),
    };
    let let_y = Node::Let {
        name: "y".to_string(),
        mutable: false,
        ann: None,
        value: Box::new(Node::Lit(Literal::Ident("y_src".to_string()), sp())),
        span: sp(),
    };
    // Call f(x, y)
    let call_node = Node::Call {
        callee: "f".to_string(),
        args: vec![
            Node::Lit(Literal::Ident("x".to_string()), sp()),
            Node::Lit(Literal::Ident("y".to_string()), sp()),
        ],
        span: sp(),
    };
    let module = Module {
        items: vec![fn_node, let_x, let_y, call_node],
    };

    let mut env = TypeEnv::new();
    env.insert(
        "x_src".to_string(),
        tensor_vt(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Known(8)]),
    );
    env.insert(
        "y_src".to_string(),
        tensor_vt(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Known(16)]),
    );

    let src = "fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>) -> i32 { return 0 } let x = x_src let y = y_src f(x, y)";
    let diags = check_module(&module, src, &env);
    let conflict_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::symbolic_conflict")
        .collect();
    assert!(
        conflict_diags.is_empty(),
        "no symbolic_conflict expected when N matches (4 == 4); got: {conflict_diags:?}"
    );
}

// ── TEST 13: symbolic dim conflict — N mismatched across args ─────────────────

/// `fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>)` called with
/// x: Tensor<f32,[4,8]> and y: Tensor<f32,[8,16]> — N=4 vs N=8 → symbolic_conflict.
#[test]
fn symbolic_dim_mismatch_n_conflict() {
    let fn_node = Node::FnDef {
        type_params: vec![],
        is_pub: false,
        is_test: false,
        name: "f".to_string(),
        params: vec![
            Param {
                name: "a".to_string(),
                ty: TypeAnn::Tensor {
                    dtype: "f32".to_string(),
                    dims: vec!["N".to_string(), "K".to_string()],
                },
                span: sp(),
            },
            Param {
                name: "b".to_string(),
                ty: TypeAnn::Tensor {
                    dtype: "f32".to_string(),
                    dims: vec!["N".to_string(), "M".to_string()],
                },
                span: sp(),
            },
        ],
        ret_type: Some(TypeAnn::ScalarI32),
        body: vec![Node::Return {
            value: Some(Box::new(Node::Lit(Literal::Int(0), sp()))),
            span: sp(),
        }],
        reap_threshold: None,
        attrs: Vec::new(),
        span: sp(),
    };
    // x: Tensor<f32,[4,8]>, y: Tensor<f32,[8,16]> — N=4 from x, N=8 from y → conflict
    let let_x = Node::Let {
        name: "x".to_string(),
        mutable: false,
        ann: None,
        value: Box::new(Node::Lit(Literal::Ident("x_src".to_string()), sp())),
        span: sp(),
    };
    let let_y = Node::Let {
        name: "y".to_string(),
        mutable: false,
        ann: None,
        value: Box::new(Node::Lit(Literal::Ident("y_src".to_string()), sp())),
        span: sp(),
    };
    let call_node = Node::Call {
        callee: "f".to_string(),
        args: vec![
            Node::Lit(Literal::Ident("x".to_string()), sp()),
            Node::Lit(Literal::Ident("y".to_string()), sp()),
        ],
        span: sp(),
    };
    let module = Module {
        items: vec![fn_node, let_x, let_y, call_node],
    };

    let mut env = TypeEnv::new();
    env.insert(
        "x_src".to_string(),
        tensor_vt(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Known(8)]),
    );
    env.insert(
        "y_src".to_string(),
        tensor_vt(DType::F32, vec![ShapeDim::Known(8), ShapeDim::Known(16)]),
    );

    let src = "fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>) -> i32 { return 0 } let x = x_src let y = y_src f(x, y)";
    let diags = check_module(&module, src, &env);
    let conflict_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::symbolic_conflict")
        .collect();
    assert!(
        !conflict_diags.is_empty(),
        "expected shape::symbolic_conflict when N mismatches (4 != 8); got: {diags:?}"
    );
    let msg = &conflict_diags[0].message;
    assert!(
        msg.contains('N'),
        "diagnostic must name the conflicting symbol; got: {msg}"
    );
}

// ── TEST 14: shape diagnostics flow through diag_from_span ───────────────────

/// Verify that shape diagnostics have `phase = "type-check"` and a `code`
/// in the `shape::` namespace — confirming they flow through `diag_from_span`
/// and not via `eprintln!`.
#[test]
fn shape_diag_channel_verification() {
    let module = Module {
        items: vec![let_tensor("x", "f32", &["4", "16"], "src")],
    };
    let env = env_with(&[("src", DType::F32, &[4, 8])]);
    let src = "let x: Tensor<f32, [4, 16]> = src";
    let diags = check_module(&module, src, &env);

    let shape_diag = diags
        .iter()
        .find(|d| d.code.starts_with("shape::"))
        .expect("expected a shape:: diagnostic");

    assert_eq!(
        shape_diag.phase, "type-check",
        "shape diagnostics must flow through the type-check phase (diag_from_span); \
         got phase = {:?}",
        shape_diag.phase
    );
    assert!(
        shape_diag.code.starts_with("shape::"),
        "code must be in the shape:: namespace; got: {}",
        shape_diag.code
    );
    // Severity must be Error (same as safety:: diagnostics).
    assert_eq!(
        shape_diag.severity,
        libmind::diagnostics::Severity::Error,
        "shape diagnostics must have Error severity"
    );
}

// ── TEST 15: dim_mismatch in function body ────────────────────────────────────

/// `let` bindings inside a `fn` body get the same shape checking as module-level.
#[test]
fn shape_check_inside_fn_body() {
    // Build a fn body with a mismatched let binding.
    let body_let = let_tensor("x", "f32", &["4", "16"], "body_src");
    let fn_node = Node::FnDef {
        type_params: vec![],
        is_pub: false,
        is_test: false,
        name: "check_body".to_string(),
        params: vec![],
        ret_type: Some(TypeAnn::ScalarI32),
        body: vec![
            body_let,
            Node::Return {
                value: Some(Box::new(Node::Lit(Literal::Int(0), sp()))),
                span: sp(),
            },
        ],
        reap_threshold: None,
        attrs: Vec::new(),
        span: sp(),
    };
    let module = Module {
        items: vec![fn_node],
    };
    let env = env_with(&[("body_src", DType::F32, &[4, 8])]);
    let src = "fn check_body() -> i32 { let x: Tensor<f32, [4, 16]> = body_src return 0 }";
    let diags = check_module(&module, src, &env);

    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::dim_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::dim_mismatch inside fn body; got: {diags:?}"
    );
}

// ── TEST 16: lowering unchanged — shape type is compile-time-only ─────────────

/// A `Tensor<f32,[4,8]>` annotation and an unannotated binding both produce
/// the same `ValueType::Tensor(TensorType { dtype: F32, shape: [4, 8] })`.
/// This verifies that the shape annotation is a compile-time lens only.
#[test]
fn shape_annotation_does_not_change_value_type() {
    // Expected ValueType for a f32 [4,8] tensor.
    let expected_vt = tensor_vt(DType::F32, vec![ShapeDim::Known(4), ShapeDim::Known(8)]);

    // Annotated let binding — annotation matches inferred type exactly.
    let module_annotated = Module {
        items: vec![let_tensor("x", "f32", &["4", "8"], "src")],
    };
    let mut env = TypeEnv::new();
    env.insert("src".to_string(), expected_vt.clone());
    let src = "let x: Tensor<f32, [4, 8]> = src";
    let diags_annotated = check_module(&module_annotated, src, &env);
    let shape_diags: Vec<_> = diags_annotated
        .iter()
        .filter(|d| d.code.starts_with("shape::"))
        .collect();
    assert!(
        shape_diags.is_empty(),
        "compatible annotation must not produce shape diagnostics; got: {shape_diags:?}"
    );

    // Unannotated let — same src tensor, no annotation.
    let module_unannotated = Module {
        items: vec![Node::Let {
            name: "y".to_string(),
            mutable: false,
            ann: None,
            value: Box::new(Node::Lit(Literal::Ident("src".to_string()), sp())),
            span: sp(),
        }],
    };
    let src2 = "let y = src";
    let diags_unannotated = check_module(&module_unannotated, src2, &env);
    assert!(
        diags_unannotated.is_empty(),
        "unannotated binding must produce zero diagnostics; got: {diags_unannotated:?}"
    );
    // Both cases pass with zero diagnostics → the annotation is a compile-time
    // lens and doesn't alter the underlying ValueType representation.
}

// ── TEST 17: q16 DType recognized in type system ──────────────────────────────

/// `DType::Q16` round-trips through `DType::parse` and `DType::as_str`.
#[test]
fn dtype_q16_roundtrip() {
    let dt = DType::parse("q16").expect("q16 must be a recognized dtype");
    assert_eq!(dt, DType::Q16);
    assert_eq!(dt.as_str(), "q16");
}

// ── TEST 18: q16 tensor ValueType in type checker ────────────────────────────

/// A `Tensor<q16,[8]>` annotation resolves to `ValueType::Tensor(TensorType { dtype: Q16, shape: [8] })`.
#[test]
fn q16_tensor_value_type_round_trip() {
    let module = Module {
        items: vec![let_tensor("v", "q16", &["8"], "q_vec")],
    };
    let env = env_with(&[("q_vec", DType::Q16, &[8])]);
    let src = "let v: Tensor<q16, [8]> = q_vec";
    let diags = check_module(&module, src, &env);
    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code.starts_with("shape::"))
        .collect();
    assert!(
        shape_diags.is_empty(),
        "q16 tensor binding must be compatible with q16 annotation; got: {shape_diags:?}"
    );
}

// ── TEST 19: dtype_mismatch across Tensor<f32> vs Tensor<q16> ────────────────
// This is the direct RFC 0012 §3.2 test that q16 and f32 are distinct dtypes.

#[test]
fn shape_dtype_mismatch_diagnostic() {
    // Annotation: Tensor<q16, [4]> — inferred: Tensor<f32, [4]>
    let module = Module {
        items: vec![let_tensor("x", "q16", &["4"], "f32_vec")],
    };
    let env = env_with(&[("f32_vec", DType::F32, &[4])]);
    let src = "let x: Tensor<q16, [4]> = f32_vec";
    let diags = check_module(&module, src, &env);
    let shape_diags: Vec<_> = diags
        .iter()
        .filter(|d| d.code == "shape::dtype_mismatch")
        .collect();
    assert!(
        !shape_diags.is_empty(),
        "expected shape::dtype_mismatch for Tensor<q16> vs Tensor<f32>; got: {diags:?}"
    );
}
