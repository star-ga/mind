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

/// A type name the walk reaches at a leaf. Built-in scalar variants arrive as their
/// canonical spelling, so one walk serves every name-keyed scan; a tensor element
/// dtype arrives separately because not every scan treats it as a scalar mention.
#[derive(Clone, Copy)]
enum TypeLeaf<'a> {
    Scalar(&'a str),
    TensorDtype(&'a str),
}

type LeafPred = fn(TypeLeaf<'_>) -> bool;

/// True when `name` is one of the four narrow integer type names that
/// `is_named_narrow_sig_ty` (lower.rs) recognises.
fn is_narrow_name(name: &str) -> bool {
    matches!(name, "i8" | "u8" | "i16" | "u16")
}

/// The narrow scan counts tensor dtypes too (over-approximation — see module doc).
fn is_narrow_leaf(leaf: TypeLeaf<'_>) -> bool {
    match leaf {
        TypeLeaf::Scalar(n) | TypeLeaf::TensorDtype(n) => is_narrow_name(n),
    }
}

/// A SCALAR floating-point type other than `f64`. Tensor element dtypes are not
/// counted: a tensor is out of the frozen profile on its own, and the refusal
/// should name the tensor rather than its element type.
fn is_non_f64_float_scalar(leaf: TypeLeaf<'_>) -> bool {
    match leaf {
        TypeLeaf::Scalar(n) => matches!(n, "f32" | "F32" | "f16" | "F16" | "bf16" | "BF16"),
        TypeLeaf::TensorDtype(_) => false,
    }
}

/// True when a `TypeAnn` mentions a type `pred` accepts anywhere inside it
/// (recursively through generic args, tuple elements, element/target types and
/// fn-pointer signatures). Tensor dtypes are STRINGS, checked with the same
/// name test (over-approximation — see module doc).
fn ty_mentions(pred: LeafPred, ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarI32 => pred(TypeLeaf::Scalar("i32")),
        TypeAnn::ScalarI64 => pred(TypeLeaf::Scalar("i64")),
        TypeAnn::ScalarF32 => pred(TypeLeaf::Scalar("f32")),
        TypeAnn::ScalarF64 => pred(TypeLeaf::Scalar("f64")),
        TypeAnn::ScalarBool => pred(TypeLeaf::Scalar("bool")),
        TypeAnn::ScalarU32 => pred(TypeLeaf::Scalar("u32")),
        TypeAnn::Named(n) => pred(TypeLeaf::Scalar(n)),
        TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. } => {
            pred(TypeLeaf::TensorDtype(dtype))
        }
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => {
            ty_mentions(pred, element)
        }
        TypeAnn::Ref { target, .. } => ty_mentions(pred, target),
        TypeAnn::RawPtr { pointee, .. } => ty_mentions(pred, pointee),
        TypeAnn::SparseTensor { element, .. } => ty_mentions(pred, element),
        TypeAnn::Generic { args, .. } => args.iter().any(|t| ty_mentions(pred, t)),
        TypeAnn::Tuple { elements } => elements.iter().any(|t| ty_mentions(pred, t)),
        TypeAnn::FnPtr { params, ret } => {
            params.iter().any(|t| ty_mentions(pred, t))
                || ret.as_deref().is_some_and(|t| ty_mentions(pred, t))
        }
    }
}

/// Optional-annotation helper (`let` / `const` / return types).
fn opt_ty_mentions(pred: LeafPred, ty: &Option<TypeAnn>) -> bool {
    ty.as_ref().is_some_and(|t| ty_mentions(pred, t))
}

/// Params helper (fn defs, extern fns, closures, trait method signatures).
fn params_mention(pred: LeafPred, params: &[ast::Param]) -> bool {
    params.iter().any(|p| ty_mentions(pred, &p.ty))
}

/// True when any `TypeAnn` reachable from `node` (annotations first, then all
/// child expressions/statements, recursively) mentions a type `pred` accepts.
fn node_mentions(pred: LeafPred, node: &Node) -> bool {
    let any = |nodes: &[Node]| nodes.iter().any(|n| node_mentions(pred, n));
    match node {
        Node::Lit(_, _) => false,
        Node::Binary { left, right, .. } => node_mentions(pred, left) || node_mentions(pred, right),
        Node::Paren(inner, _) => node_mentions(pred, inner),
        Node::Tuple { elements, .. } => any(elements),
        Node::Call { args, .. } => any(args),
        Node::CallGrad { loss, .. } => node_mentions(pred, loss),
        Node::CallTensorSum { x, .. }
        | Node::CallTensorMean { x, .. }
        | Node::CallReshape { x, .. }
        | Node::CallExpandDims { x, .. }
        | Node::CallSqueeze { x, .. }
        | Node::CallTranspose { x, .. }
        | Node::CallIndex { x, .. }
        | Node::CallSlice { x, .. }
        | Node::CallSliceStride { x, .. }
        | Node::CallTensorRelu { x, .. } => node_mentions(pred, x),
        Node::CallGather { x, idx, .. } => node_mentions(pred, x) || node_mentions(pred, idx),
        Node::CallDot { a, b, .. } | Node::CallMatMul { a, b, .. } => {
            node_mentions(pred, a) || node_mentions(pred, b)
        }
        Node::TensorMatmul { lhs, rhs, .. } | Node::TensorElemwise { lhs, rhs, .. } => {
            node_mentions(pred, lhs) || node_mentions(pred, rhs)
        }
        Node::CallTensorRand { .. } => false,
        Node::CallTensorConv2d { x, w, .. } => node_mentions(pred, x) || node_mentions(pred, w),
        Node::Let { ann, value, .. } => opt_ty_mentions(pred, ann) || node_mentions(pred, value),
        Node::LetTuple { value, .. } | Node::Assign { value, .. } => node_mentions(pred, value),
        Node::FnDef(fd, _) => {
            params_mention(pred, &fd.params) || opt_ty_mentions(pred, &fd.ret_type) || any(&fd.body)
        }
        Node::Return { value, .. } => value.as_deref().is_some_and(|n| node_mentions(pred, n)),
        Node::Block { stmts, .. } => any(stmts),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            node_mentions(pred, cond)
                || any(then_branch)
                || else_branch.as_deref().is_some_and(&any)
        }
        Node::Import { .. } => false,
        Node::ArrayLit { elements, .. } | Node::SetLit { elements, .. } => any(elements),
        Node::MapLit { entries, .. } => entries
            .iter()
            .any(|(k, v)| node_mentions(pred, k) || node_mentions(pred, v)),
        Node::For {
            start, end, body, ..
        } => node_mentions(pred, start) || node_mentions(pred, end) || any(body),
        Node::ForEach {
            collection, body, ..
        } => node_mentions(pred, collection) || any(body),
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => node_mentions(pred, cond) || any(body),
        #[cfg(feature = "std-surface")]
        Node::Break { .. } | Node::Continue { .. } => false,
        Node::Print { args, .. } => any(args),
        Node::Neg { operand, .. } | Node::Not { operand, .. } | Node::BitNot { operand, .. } => {
            node_mentions(pred, operand)
        }
        Node::MethodCall { receiver, args, .. } => node_mentions(pred, receiver) || any(args),
        Node::FieldAccess { receiver, .. } => node_mentions(pred, receiver),
        Node::Const { ty, value, .. } => opt_ty_mentions(pred, ty) || node_mentions(pred, value),
        Node::ExternConst { ty, .. } => ty_mentions(pred, ty),
        Node::TypeAlias { target, .. } => ty_mentions(pred, target),
        Node::Export { .. } => false,
        Node::StructDef { fields, .. } => fields.iter().any(|f| ty_mentions(pred, &f.ty)),
        Node::EnumDef { variants, .. } => variants
            .iter()
            .any(|v| v.payload.iter().any(|t| ty_mentions(pred, t))),
        Node::Assert { cond, .. } => node_mentions(pred, cond),
        Node::As { expr, ty, .. } => ty_mentions(pred, ty) || node_mentions(pred, expr),
        Node::Logical { left, right, .. } => {
            node_mentions(pred, left) || node_mentions(pred, right)
        }
        #[cfg(feature = "std-surface")]
        Node::Bitwise { left, right, .. } => {
            node_mentions(pred, left) || node_mentions(pred, right)
        }
        Node::StructLit { fields, .. } => fields.iter().any(|f| node_mentions(pred, &f.value)),
        Node::IndexAccess {
            receiver, index, ..
        } => node_mentions(pred, receiver) || node_mentions(pred, index),
        Node::SliceRange {
            receiver,
            start,
            end,
            ..
        } => {
            node_mentions(pred, receiver) || node_mentions(pred, start) || node_mentions(pred, end)
        }
        Node::IndexAssign {
            receiver,
            index,
            value,
            ..
        } => {
            node_mentions(pred, receiver)
                || node_mentions(pred, index)
                || node_mentions(pred, value)
        }
        Node::FieldAssign {
            receiver, value, ..
        } => node_mentions(pred, receiver) || node_mentions(pred, value),
        Node::Match {
            scrutinee, arms, ..
        } => {
            node_mentions(pred, scrutinee)
                || arms.iter().any(|arm| {
                    arm.guard.as_ref().is_some_and(|n| node_mentions(pred, n))
                        || node_mentions(pred, &arm.body)
                })
        }
        Node::Try { inner, .. } => node_mentions(pred, inner),
        Node::Ref { inner, .. } => node_mentions(pred, inner),
        Node::ExternBlock { fns, .. } => fns
            .iter()
            .any(|ef| params_mention(pred, &ef.params) || opt_ty_mentions(pred, &ef.ret_type)),
        #[cfg(feature = "std-surface")]
        Node::Region { body, .. } => any(body),
        Node::Closure(data, _) => {
            params_mention(pred, &data.params)
                || opt_ty_mentions(pred, &data.ret_type)
                || any(&data.body)
        }
        Node::TraitDef { methods, .. } => methods
            .iter()
            .any(|m| params_mention(pred, &m.params) || opt_ty_mentions(pred, &m.ret_type)),
        Node::ImplBlock { methods, .. } => any(methods),
    }
}

/// One O(n) walk of the (post-desugar) module AST `lower_to_ir` receives.
/// `true` when ANY narrow-int type is mentioned anywhere — see the module doc
/// for why `false` proves every `infer_narrow_arith_ty` call returns `None`.
pub(crate) fn module_mentions_narrow(module: &ast::Module) -> bool {
    module
        .items
        .iter()
        .any(|n| node_mentions(is_narrow_leaf, n))
}

/// `true` when ANY type annotation in the module names a scalar float type other
/// than `f64` (`f32`, `f16`, `bf16`), including cast targets and the call-form cast
/// `f32(x)`, which the parser desugars to `As`. The frozen native profile uses it:
/// its IR carries no scalar float width (`ConstF64` is the only float producer
/// and a declared `f32` lives only in the AST), and the native comparison and
/// arithmetic lowerings are proven for f64 alone.
pub(crate) fn module_mentions_non_f64_float(module: &ast::Module) -> bool {
    module
        .items
        .iter()
        .any(|n| node_mentions(is_non_f64_float_scalar, n))
}
