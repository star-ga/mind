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

//! Exhaustive AST child enumeration for the Q16.16 numerics lint
//! (`nerve_lint`, diagnostic codes `E_NERVE_001`…`E_NERVE_005`).
//!
//! WHY THIS IS A SEPARATE MODULE, AND WHY THE MATCH HAS NO `_` ARM.
//! The five rules are *negative* properties: "there is no reassociating
//! reduction here", "there is no IEEE-754 operand here", "there is no
//! hand-rolled `(a * b) >> 16` here". A walker that silently skips an AST
//! variant it does not recognise turns every one of those into a false
//! "clean" — the single most dangerous failure mode a bit-identity gate can
//! have, because it looks exactly like a pass.
//!
//! So [`for_each_child`] matches every `Node` variant explicitly and has NO
//! catch-all arm. Adding a variant to `ast::Node` breaks this build until the
//! author classifies it, which is the same fail-closed registry discipline the
//! FP-mode classifier uses for float-returning intrinsics. Variants that
//! genuinely carry no sub-expression (`Break`, `Import`, `Export`, …) are
//! listed together in one no-child arm so the intent is auditable rather than
//! inferred from absence.

use crate::ast::{MatchArm, Node, StructLitField};

/// Invoke `f` once for every direct sub-expression / sub-statement of `node`.
///
/// Order is source order (left-to-right, then body). Declaration items nested
/// inside `node` (a `FnDef` inside an `ImplBlock`, a closure body) are yielded
/// too — the lint walks whole modules, so skipping them would leave a method
/// body unchecked.
pub fn for_each_child<'a, F: FnMut(&'a Node)>(node: &'a Node, f: &mut F) {
    match node {
        // ── Leaves: literals, identifiers, and markers with no expression ──
        Node::Lit(_, _)
        | Node::Import { .. }
        | Node::Export { .. }
        | Node::StructDef { .. }
        | Node::EnumDef { .. }
        | Node::TypeAlias { .. }
        | Node::ExternConst { .. }
        | Node::ExternBlock { .. }
        | Node::TraitDef { .. }
        | Node::CallTensorRand { .. } => {}
        #[cfg(feature = "std-surface")]
        Node::Break { .. } | Node::Continue { .. } => {}

        // ── One boxed operand ──────────────────────────────────────────────
        Node::Paren(inner, _)
        | Node::Neg { operand: inner, .. }
        | Node::Not { operand: inner, .. }
        | Node::BitNot { operand: inner, .. }
        | Node::Try { inner, .. }
        | Node::Ref { inner, .. }
        | Node::Assert { cond: inner, .. }
        | Node::As { expr: inner, .. }
        | Node::Deref { operand: inner, .. }
        | Node::Let { value: inner, .. }
        | Node::LetTuple { value: inner, .. }
        | Node::Assign { value: inner, .. }
        | Node::Const { value: inner, .. }
        | Node::CallGrad { loss: inner, .. }
        | Node::CallTensorSum { x: inner, .. }
        | Node::CallTensorMean { x: inner, .. }
        | Node::CallReshape { x: inner, .. }
        | Node::CallExpandDims { x: inner, .. }
        | Node::CallSqueeze { x: inner, .. }
        | Node::CallTranspose { x: inner, .. }
        | Node::CallIndex { x: inner, .. }
        | Node::CallSlice { x: inner, .. }
        | Node::CallSliceStride { x: inner, .. }
        | Node::CallTensorRelu { x: inner, .. }
        | Node::FieldAccess {
            receiver: inner, ..
        } => f(inner),

        // ── Two boxed operands ─────────────────────────────────────────────
        Node::Binary { left, right, .. }
        | Node::Logical { left, right, .. }
        | Node::TensorElemwise {
            lhs: left,
            rhs: right,
            ..
        }
        | Node::TensorMatmul {
            lhs: left,
            rhs: right,
            ..
        }
        | Node::CallDot {
            a: left, b: right, ..
        }
        | Node::CallMatMul {
            a: left, b: right, ..
        }
        | Node::CallGather {
            x: left,
            idx: right,
            ..
        }
        | Node::CallTensorConv2d {
            x: left, w: right, ..
        }
        | Node::IndexAccess {
            receiver: left,
            index: right,
            ..
        } => {
            f(left);
            f(right);
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            f(left);
            f(right);
        }
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            f(receiver);
            f(start);
            f(end);
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            f(receiver);
            f(index);
            f(value);
        }
        Node::FieldAssign {
            receiver, value, ..
        } => {
            f(receiver);
            f(value);
        }
        Node::DerefAssign { target, value, .. } => {
            f(target);
            f(value);
        }

        // ── Sequences ──────────────────────────────────────────────────────
        Node::Tuple { elements, .. }
        | Node::ArrayLit { elements, .. }
        | Node::SetLit { elements, .. } => elements.iter().for_each(&mut *f),
        Node::Call { args, .. } | Node::Print { args, .. } => args.iter().for_each(&mut *f),
        Node::Block { stmts, .. } => stmts.iter().for_each(&mut *f),
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => body.iter().for_each(&mut *f),
        Node::ImplBlock { methods, .. } => methods.iter().for_each(&mut *f),
        Node::MapLit { entries, .. } => {
            for (k, v) in entries {
                f(k);
                f(v);
            }
        }
        Node::StructLit { fields, .. } => {
            for StructLitField { value, .. } in fields {
                f(value);
            }
        }
        Node::Return { value, .. } => {
            if let Some(v) = value {
                f(v);
            }
        }

        // ── Compound: receiver + args / cond + bodies ──────────────────────
        Node::MethodCall { receiver, args, .. } => {
            f(receiver);
            args.iter().for_each(&mut *f);
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            f(cond);
            then_branch.iter().for_each(&mut *f);
            if let Some(e) = else_branch {
                e.iter().for_each(&mut *f);
            }
        }
        Node::For {
            start, end, body, ..
        } => {
            f(start);
            f(end);
            body.iter().for_each(&mut *f);
        }
        Node::ForEach {
            collection, body, ..
        } => {
            f(collection);
            body.iter().for_each(&mut *f);
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            f(cond);
            body.iter().for_each(&mut *f);
        }
        Node::Match {
            scrutinee, arms, ..
        } => {
            f(scrutinee);
            for MatchArm { guard, body, .. } in arms {
                if let Some(g) = guard {
                    f(g);
                }
                f(body);
            }
        }
        Node::FnDef(fd, _) => fd.body.iter().for_each(&mut *f),
        Node::Closure(cd, _) => cd.body.iter().for_each(&mut *f),
    }
}

/// Depth-first pre-order walk of `node` and everything under it.
///
/// The visitor sees `node` itself first, so a rule can match on the enclosing
/// form (`Bitwise { op: Shr, .. }`) and then inspect its own subtree.
pub fn walk<'a, F: FnMut(&'a Node)>(node: &'a Node, f: &mut F) {
    f(node);
    for_each_child(node, &mut |child| walk(child, f));
}

/// `true` iff any node in the subtree rooted at `node` satisfies `pred`.
pub fn any<F: Fn(&Node) -> bool>(node: &Node, pred: &F) -> bool {
    if pred(node) {
        return true;
    }
    let mut found = false;
    for_each_child(node, &mut |child| {
        if !found && any(child, pred) {
            found = true;
        }
    });
    found
}
