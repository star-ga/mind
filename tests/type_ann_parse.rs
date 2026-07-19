// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use libmind::parser;

#[test]
fn parses_scalar_annotation() {
    let m = parser::parse("let n: i32 = 3; n + 1").unwrap();
    assert_eq!(m.items.len(), 2);
}

#[test]
fn parses_tensor_annotation() {
    let m = parser::parse("let x: Tensor[f32,(B,3,224,224)] = 0;").unwrap();
    assert_eq!(m.items.len(), 1);
}

/// Bare `[T]` is sugar for `&[T]`: the parameter's *type* must parse to a
/// structurally identical `TypeAnn` (the span-free node). This is the root
/// invariant behind two downstream guarantees — `mindc fmt` canonicalises
/// `[T]`→`&[T]`, and the mic@3 codec emits identical bytes (same
/// `TypeAnn::Slice` node) — so neither the formatter nor the trace_hash
/// can observe the surface spelling. (Whole-AST equality would differ only
/// on source spans; the type node itself carries none.)
#[test]
fn bare_slice_parses_identically_to_borrowed_slice() {
    fn first_param_ty(src: &str) -> libmind::ast::TypeAnn {
        let m = parser::parse(src).unwrap();
        match &m.items[0] {
            libmind::ast::Node::FnDef(fd, _) => fd.params[0].ty.clone(),
            other => panic!("expected a fn item, got {other:?}"),
        }
    }

    let bare = first_param_ty("fn f(xs: [u32]) -> u32 { 0 }");
    let borrowed = first_param_ty("fn f(xs: &[u32]) -> u32 { 0 }");
    assert_eq!(
        bare, borrowed,
        "`[T]` and `&[T]` must parse to the identical Slice TypeAnn node"
    );

    // The fixed-size array `[T; N]` must NOT collapse to that node.
    let array = first_param_ty("fn f(xs: [u32; 4]) -> u32 { 0 }");
    assert_ne!(
        bare, array,
        "`[T]` (slice) and `[T; N]` (array) must remain distinct"
    );
}
