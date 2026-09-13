//! Tensor-shape admission rules for the runnable artifact ABI gate.
//!
//! Static-shape tensor parameters use the real memref/tensor boundary. Dynamic
//! shapes, diff tensors, tensor returns, and tensors nested in type-carrying
//! composites remain fail-closed until their ABI lowering is implemented.

use crate::ast::TypeAnn;

/// Reason a function PARAMETER `TypeAnn` cannot lower in the runnable ABI, or
/// `None` when it lowers correctly. Identical to [`sig_non_i64`] EXCEPT that a
/// STATIC-SHAPE (all extents compile-time-known) `tensor` parameter is now
/// allowed: it lowers to a real memref/tensor C ABI (ptr + baked-in static
/// extents), not the erased i64 scalar. The `func.func` signature carries the
/// true `tensor<..>` type (`type_ann_to_abi_mlir`), the param seeds a real
/// `ValueKind::Tensor` (`type_ann_to_value_kind`), the build routes to the
/// `arith-linalg` preset whose `one-shot-bufferize{bufferize-function-
/// boundaries=true}` converts the boundary to a memref, and the pinned
/// reduction fold reads it via `tensor.extract %param[..]`. A DYNAMIC/symbolic
/// dim still gates (no static extent to bake into the memref descriptor), and a
/// `diff tensor` param still gates (autodiff boundary is a separate change).
/// Tensor RETURNS are unchanged — still routed through [`sig_non_i64`] and
/// gated (the out-param C ABI is a separate, larger slice).
pub(crate) fn param_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        TypeAnn::Tensor { dims, .. } if tensor_dims_all_static(dims) => None,
        _ => sig_non_i64(ty),
    }
}

/// `true` when a tensor annotation's dims are all statically-known numeric
/// extents (`tensor<f64[4]>` → `["4"]`), so the shape can be baked into a memref
/// descriptor with no dynamic dim. A rank-0 tensor (`dims` empty) is NOT treated
/// as static-lowerable here (there is no reduction surface for it yet), and a
/// symbolic dim (`["N"]`) makes the whole annotation dynamic.
fn tensor_dims_all_static(dims: &[String]) -> bool {
    !dims.is_empty() && dims.iter().all(|d| d.parse::<usize>().is_ok())
}

/// Reason a function parameter/return `TypeAnn` cannot lower in the runnable
/// i64 ABI, or `None` when it lowers correctly (`i64`, `f32`, `f64`, `bool`,
/// a struct handle (`Named`), slice/ref/array/tuple — handled elsewhere or
/// already loud).
pub(crate) fn sig_non_i64(ty: &TypeAnn) -> Option<&'static str> {
    match ty {
        // `i32`/`u32` params & returns now lower correctly (real i32 MLIR with
        // signed/unsigned op selection + deterministic two's-complement wrap), so
        // they are no longer gated.
        TypeAnn::Tensor { .. } => Some(
            "a tensor-typed parameter/return erases to the i64 ABI and is treated as a scalar \
             integer",
        ),
        TypeAnn::DiffTensor { .. } => Some("a diff-tensor parameter/return erases to the i64 ABI"),
        // A tensor NESTED inside a composite type also erases to the raw i64
        // slot: `type_ann_to_abi_mlir` lowers the composite's ABI to `i64` and
        // the tensor is never materialised, so `--emit-shared` wrote an rc=0
        // `.so` whose C signature does not match the declared type (measured:
        // `&tensor`, `(tensor, i64)`, `Option<tensor>`, `[tensor; N]`, and a
        // `-> i64 { return t }` that leaks the raw slot). The top-level match
        // above misses every one of these. Recurse structurally so any tensor
        // anywhere in the annotation fails LOUD — the third instance of the
        // silent sub-i64-ABI miscompile class, and a fail-closed breach.
        //
        // Recursion is over the type-CARRYING composites only. RawPtr / FnPtr
        // are deliberately NOT recursed: they lower to an opaque `!llvm.ptr`
        // (the extern-C ABI contract), so `*const tensor` is a real pointer,
        // not an erased tensor. `Named` is opaque here (resolved later in
        // typecheck) — an alias/struct that itself contains a tensor is a
        // separate, unaudited slice, left as-is. `SparseTensor` is likewise
        // out of this measured slice (runtime-resolved layout).
        TypeAnn::Slice { element, .. } | TypeAnn::Array { element, .. } => sig_non_i64(element),
        TypeAnn::Ref { target, .. } => sig_non_i64(target),
        TypeAnn::Generic { args, .. } => args.iter().find_map(|a| sig_non_i64(a)),
        TypeAnn::Tuple { elements } => elements.iter().find_map(sig_non_i64),
        // i8/u8/i16/u16 (as `Named`) params + returns now lower correctly via
        // the i64-SLOT narrow-signature ABI (`src/eval/lower.rs`): a param is
        // materialised at its declared width on fn entry (zext mask / sext
        // shift-pair) and a return is masked to its declared width at every
        // return site, so the width and unsigned semantics are preserved —
        // they are no longer gated.
        _ => None,
    }
}
