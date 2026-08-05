//! Module-wide narrow-int SURFACE prescan (compile-speed early-skip).
//!
//! `infer_narrow_arith_ty` (src/eval/lower.rs) recurses through every
//! `Binary`/`Bitwise`/`Paren` operand subtree, and it is invoked once per
//! lowered binop from `mask_narrow_binop_result` — so every ANCESTOR binop
//! re-walks its whole operand subtree. For nested arithmetic with no narrow-int
//! surface (the common all-i64/float hot path, e.g. a matmul accumulation
//! `acc + a[i] * b[j]`), that is O(n²) tree-walk work in which every leaf
//! returns `None` — pure waste.
//!
//! This prescan computes, ONCE per `lower_to_ir` call in a single O(n) walk of
//! the module AST, whether the module mentions ANY type from which
//! `infer_narrow_arith_ty` could ever derive a narrow (`i8`/`u8`/`i16`/`u16`)
//! result. If not, `infer_narrow_arith_ty` early-returns `None` without
//! recursing (see the `MODULE_HAS_NARROW_SURFACE` gate in lower.rs), restoring
//! O(n) lowering.
//!
//! # Byte-identity proof
//!
//! The skip is provably byte-identical, not heuristically so:
//!
//! * `infer_narrow_arith_ty` can only return `Some` when one of its LEAF arms
//!   produces a `TypeAnn` passing `is_named_narrow_sig_ty` (a `Named` 8/16-bit
//!   integer). Its leaf sources are, exhaustively:
//!   1. `NARROW_LOCALS` entries — inserted by `record_narrow_let` /
//!      `enter_narrow_scope` / the match-payload bind desugar, ALWAYS from a
//!      declared `TypeAnn` occurring in the module AST (a `let`/param/`const`
//!      annotation, or a `Generic` argument of a declared `Option<…>`/
//!      `Result<…>`/tuple annotation).
//!   2. `Node::As { ty, .. }` — the cast annotation, an AST `TypeAnn` directly.
//!   3. `ir.fn_signatures` returns — inserted only from `FnDef` param/return
//!      annotations of this module's AST (the entry pre-pass and the per-FnDef
//!      insert in lower.rs).
//!   4. `ir.struct_field_types` — inserted only from `StructDef` field
//!      annotations of this module's AST (plus intra-module clones).
//!   5. `__elem__…` struct_env sentinels (via `index_element_narrow_ty`) —
//!      bare type-NAME strings derived from `array<T>` annotation arguments,
//!      struct field types, or the fixed non-narrow `"String"`/`"i64"`
//!      method-return sentinels.
//! * Therefore: if NO `TypeAnn` anywhere in the module AST mentions a narrow
//!   integer name (recursively through `Generic` args, tuple elements, slice/
//!   array/ref/pointer element types, fn-pointer signatures — and, extra-
//!   conservatively, tensor dtype STRINGS), then every one of those sources is
//!   narrow-free and EVERY `infer_narrow_arith_ty` call returns `None`.
//!   Early-returning `None` is byte-for-byte the same decision, minus the walk.
//! * If the scan DOES find a narrow mention, the flag stays `true` and the
//!   lowering path is literally unchanged (the O(n²) walk remains for narrow-
//!   containing modules — correctness over speed there).
//!
//! The scan is deliberately an OVER-approximation (e.g. a narrow type in a
//! trait signature or tensor dtype sets the flag even though neither can reach
//! `infer_narrow_arith_ty` today): a false `true` only forgoes the speedup,
//! never changes bytes. The `Node`/`TypeAnn` matches are EXHAUSTIVE (no `_`
//! catch-all over node kinds), so adding a new AST variant is a compile error
//! here — the walker can never silently under-scan a future construct.

use crate::ast::{self, Node, TypeAnn};

/// True when `name` is one of the four narrow integer type names that
/// `is_named_narrow_sig_ty` (lower.rs) recognises.
fn is_narrow_name(name: &str) -> bool {
    matches!(name, "i8" | "u8" | "i16" | "u16")
}

/// True when a `TypeAnn` mentions a narrow integer type anywhere inside it
/// (recursively through generic args, tuple elements, element/target types and
/// fn-pointer signatures). Tensor dtypes are STRINGS, checked with the same
/// name test (over-approximation — see module doc).
fn ty_mentions_narrow(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32 => false,
        TypeAnn::Named(n) => is_narrow_name(n),
        TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. } => is_narrow_name(dtype),
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => {
            ty_mentions_narrow(element)
        }
        TypeAnn::Ref { target, .. } => ty_mentions_narrow(target),
        TypeAnn::RawPtr { pointee, .. } => ty_mentions_narrow(pointee),
        TypeAnn::SparseTensor { element, .. } => ty_mentions_narrow(element),
        TypeAnn::Generic { args, .. } => args.iter().any(ty_mentions_narrow),
        TypeAnn::Tuple { elements } => elements.iter().any(ty_mentions_narrow),
        TypeAnn::FnPtr { params, ret } => {
            params.iter().any(ty_mentions_narrow) || ret.as_deref().is_some_and(ty_mentions_narrow)
        }
    }
}

/// Optional-annotation helper (`let` / `const` / return types).
fn opt_ty_mentions_narrow(ty: &Option<TypeAnn>) -> bool {
    ty.as_ref().is_some_and(ty_mentions_narrow)
}

/// Params helper (fn defs, extern fns, closures, trait method signatures).
fn params_mention_narrow(params: &[ast::Param]) -> bool {
    params.iter().any(|p| ty_mentions_narrow(&p.ty))
}

/// True when any `TypeAnn` reachable from `node` (annotations first, then all
/// child expressions/statements, recursively) mentions a narrow integer type.
fn node_mentions_narrow(node: &Node) -> bool {
    let any = |nodes: &[Node]| nodes.iter().any(node_mentions_narrow);
    match node {
        Node::Lit(_, _) => false,
        Node::Binary { left, right, .. } => {
            node_mentions_narrow(left) || node_mentions_narrow(right)
        }
        Node::Paren(inner, _) => node_mentions_narrow(inner),
        Node::Tuple { elements, .. } => any(elements),
        Node::Call { args, .. } => any(args),
        Node::CallGrad { loss, .. } => node_mentions_narrow(loss),
        Node::CallTensorSum { x, .. }
        | Node::CallTensorMean { x, .. }
        | Node::CallReshape { x, .. }
        | Node::CallExpandDims { x, .. }
        | Node::CallSqueeze { x, .. }
        | Node::CallTranspose { x, .. }
        | Node::CallIndex { x, .. }
        | Node::CallSlice { x, .. }
        | Node::CallSliceStride { x, .. }
        | Node::CallTensorRelu { x, .. } => node_mentions_narrow(x),
        Node::CallGather { x, idx, .. } => node_mentions_narrow(x) || node_mentions_narrow(idx),
        Node::CallDot { a, b, .. } | Node::CallMatMul { a, b, .. } => {
            node_mentions_narrow(a) || node_mentions_narrow(b)
        }
        Node::TensorMatmul { lhs, rhs, .. } | Node::TensorElemwise { lhs, rhs, .. } => {
            node_mentions_narrow(lhs) || node_mentions_narrow(rhs)
        }
        Node::CallTensorRand { .. } => false,
        Node::CallTensorConv2d { x, w, .. } => node_mentions_narrow(x) || node_mentions_narrow(w),
        Node::Let { ann, value, .. } => opt_ty_mentions_narrow(ann) || node_mentions_narrow(value),
        Node::LetTuple { value, .. } | Node::Assign { value, .. } => node_mentions_narrow(value),
        Node::FnDef(fd, _) => {
            params_mention_narrow(&fd.params)
                || opt_ty_mentions_narrow(&fd.ret_type)
                || any(&fd.body)
        }
        Node::Return { value, .. } => value.as_deref().is_some_and(node_mentions_narrow),
        Node::Block { stmts, .. } => any(stmts),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            node_mentions_narrow(cond)
                || any(then_branch)
                || else_branch.as_deref().is_some_and(any)
        }
        Node::Import { .. } => false,
        Node::ArrayLit { elements, .. } | Node::SetLit { elements, .. } => any(elements),
        Node::MapLit { entries, .. } => entries
            .iter()
            .any(|(k, v)| node_mentions_narrow(k) || node_mentions_narrow(v)),
        Node::For {
            start, end, body, ..
        } => node_mentions_narrow(start) || node_mentions_narrow(end) || any(body),
        Node::ForEach {
            collection, body, ..
        } => node_mentions_narrow(collection) || any(body),
        Node::While { cond, body, .. } => node_mentions_narrow(cond) || any(body),
        Node::Break { .. } | Node::Continue { .. } => false,
        Node::Print { args, .. } => any(args),
        Node::Neg { operand, .. } | Node::Not { operand, .. } | Node::BitNot { operand, .. } => {
            node_mentions_narrow(operand)
        }
        Node::MethodCall { receiver, args, .. } => node_mentions_narrow(receiver) || any(args),
        Node::FieldAccess { receiver, .. } => node_mentions_narrow(receiver),
        Node::Const { ty, value, .. } => opt_ty_mentions_narrow(ty) || node_mentions_narrow(value),
        Node::ExternConst { ty, .. } => ty_mentions_narrow(ty),
        Node::TypeAlias { target, .. } => ty_mentions_narrow(target),
        Node::Export { .. } => false,
        Node::StructDef { fields, .. } => fields.iter().any(|f| ty_mentions_narrow(&f.ty)),
        Node::EnumDef { variants, .. } => variants
            .iter()
            .any(|v| v.payload.iter().any(ty_mentions_narrow)),
        Node::Assert { cond, .. } => node_mentions_narrow(cond),
        Node::As { expr, ty, .. } => ty_mentions_narrow(ty) || node_mentions_narrow(expr),
        Node::Logical { left, right, .. } | Node::Bitwise { left, right, .. } => {
            node_mentions_narrow(left) || node_mentions_narrow(right)
        }
        Node::StructLit { fields, .. } => fields.iter().any(|f| node_mentions_narrow(&f.value)),
        Node::IndexAccess {
            receiver, index, ..
        } => node_mentions_narrow(receiver) || node_mentions_narrow(index),
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            node_mentions_narrow(receiver)
                || node_mentions_narrow(start)
                || node_mentions_narrow(end)
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            node_mentions_narrow(receiver)
                || node_mentions_narrow(index)
                || node_mentions_narrow(value)
        }
        Node::FieldAssign {
            receiver, value, ..
        } => node_mentions_narrow(receiver) || node_mentions_narrow(value),
        Node::Match {
            scrutinee, arms, ..
        } => {
            node_mentions_narrow(scrutinee)
                || arms.iter().any(|arm| {
                    arm.guard.as_ref().is_some_and(node_mentions_narrow)
                        || node_mentions_narrow(&arm.body)
                })
        }
        Node::Try { inner, .. } => node_mentions_narrow(inner),
        Node::Ref { inner, .. } => node_mentions_narrow(inner),
        Node::ExternBlock { fns, .. } => fns
            .iter()
            .any(|ef| params_mention_narrow(&ef.params) || opt_ty_mentions_narrow(&ef.ret_type)),
        Node::Region { body, .. } => any(body),
        Node::Closure {
            params,
            ret_type,
            body,
            ..
        } => params_mention_narrow(params) || opt_ty_mentions_narrow(ret_type) || any(body),
        Node::TraitDef { methods, .. } => methods
            .iter()
            .any(|m| params_mention_narrow(&m.params) || opt_ty_mentions_narrow(&m.ret_type)),
        Node::ImplBlock { methods, .. } => any(methods),
    }
}

/// One O(n) walk of the (post-desugar) module AST `lower_to_ir` receives.
/// `true` when ANY narrow-int type is mentioned anywhere — see the module doc
/// for why `false` proves every `infer_narrow_arith_ty` call returns `None`.
pub(crate) fn module_mentions_narrow(module: &ast::Module) -> bool {
    module.items.iter().any(node_mentions_narrow)
}
