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

use libmind::ast::{Node, SparseLayout, TypeAnn};
use libmind::parser;
use libmind::type_checker::check_module_types;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Parser: SparseTensor type surface
// ---------------------------------------------------------------------------

#[test]
fn parse_sparse_csr_fn_param() {
    let src = "fn route(w: tensor<sparse[csr], i32>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    assert_eq!(m.items.len(), 1);
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let params = &fd.params;
            assert_eq!(params.len(), 1);
            assert!(
                matches!(
                    &params[0].ty,
                    TypeAnn::SparseTensor {
                        layout: SparseLayout::Csr,
                        ..
                    }
                ),
                "expected SparseTensor(Csr), got {:?}",
                &params[0].ty
            );
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_csr_return_type() {
    let src = "fn make_sparse() -> tensor<sparse[csr], i32> { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    assert_eq!(m.items.len(), 1);
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let ret_type = &fd.ret_type;
            assert!(
                matches!(
                    ret_type,
                    Some(TypeAnn::SparseTensor {
                        layout: SparseLayout::Csr,
                        ..
                    })
                ),
                "expected SparseTensor(Csr) return type, got {ret_type:?}"
            );
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_csc_layout() {
    let src = "fn f(x: tensor<sparse[csc], f32>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let params = &fd.params;
            assert!(matches!(
                &params[0].ty,
                TypeAnn::SparseTensor {
                    layout: SparseLayout::Csc,
                    ..
                }
            ));
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_coo_layout() {
    let src = "fn f(x: tensor<sparse[coo], f32>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let params = &fd.params;
            assert!(matches!(
                &params[0].ty,
                TypeAnn::SparseTensor {
                    layout: SparseLayout::Coo,
                    ..
                }
            ));
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_bsr_layout() {
    let src = "fn f(x: tensor<sparse[bsr], f32>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let params = &fd.params;
            assert!(matches!(
                &params[0].ty,
                TypeAnn::SparseTensor {
                    layout: SparseLayout::Bsr,
                    ..
                }
            ));
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_with_shape() {
    // tensor<sparse[csr], i32[128, 256]> — explicit shape dims
    let src = "fn f(x: tensor<sparse[csr], i32[128, 256]>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef(fd, _) => match &fd.params[0].ty {
            TypeAnn::SparseTensor { layout, shape, .. } => {
                assert_eq!(*layout, SparseLayout::Csr);
                assert_eq!(shape.len(), 2);
            }
            other => panic!("expected SparseTensor, got {other:?}"),
        },
        other => panic!("expected FnDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_in_let_binding() {
    let src = "let w: tensor<sparse[csr], i32> = 0;";
    let m = parser::parse(src).expect("parse failed");
    assert_eq!(m.items.len(), 1);
    match &m.items[0] {
        Node::Let { ann, .. } => {
            assert!(matches!(
                ann,
                Some(TypeAnn::SparseTensor {
                    layout: SparseLayout::Csr,
                    ..
                })
            ));
        }
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn parse_sparse_in_struct_field() {
    let src = "struct Weights { data: tensor<sparse[csr], f32> }";
    let m = parser::parse(src).expect("parse failed");
    assert_eq!(m.items.len(), 1);
    match &m.items[0] {
        Node::StructDef { fields, .. } => {
            assert_eq!(fields.len(), 1);
            assert!(matches!(
                &fields[0].ty,
                TypeAnn::SparseTensor {
                    layout: SparseLayout::Csr,
                    ..
                }
            ));
        }
        other => panic!("expected StructDef, got {other:?}"),
    }
}

#[test]
fn parse_sparse_unknown_layout_is_error() {
    let src = "fn f(x: tensor<sparse[unknown_layout], i32>) -> i32 { return 0; }";
    let result = parser::parse(src);
    assert!(result.is_err(), "expected parse error for unknown layout");
}

// ---------------------------------------------------------------------------
// Type checker: SparseTensor returns None from valuetype_from_ann
// ---------------------------------------------------------------------------

#[test]
fn typecheck_sparse_fn_param_no_crash() {
    // The type checker must accept `tensor<sparse[csr], i32>` in a fn signature
    // without panicking. It returns no diags because SparseTensor resolves to
    // None in valuetype_from_ann, which is the documented v1 behaviour.
    let src = "fn route(w: tensor<sparse[csr], i32>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    let diags = check_module_types(&m, src, &HashMap::new());
    // No type errors expected (the param just passes through as opaque).
    assert!(diags.is_empty(), "unexpected type diags: {diags:?}");
}

#[test]
fn typecheck_sparse_return_type_no_crash() {
    let src = "fn make_sparse() -> tensor<sparse[csr], i32> { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    let diags = check_module_types(&m, src, &HashMap::new());
    assert!(diags.is_empty(), "unexpected type diags: {diags:?}");
}

// ---------------------------------------------------------------------------
// Existing dense tensor parsing unaffected (regression guard)
// ---------------------------------------------------------------------------

#[test]
fn dense_tensor_parse_unaffected() {
    let src = "fn f(x: tensor<f32[32, 64]>) -> i32 { return 0; }";
    let m = parser::parse(src).expect("parse failed");
    match &m.items[0] {
        Node::FnDef(fd, _) => {
            let params = &fd.params;
            assert!(
                matches!(&params[0].ty, TypeAnn::Tensor { .. }),
                "expected dense Tensor, got {:?}",
                &params[0].ty
            );
        }
        other => panic!("expected FnDef, got {other:?}"),
    }
}
