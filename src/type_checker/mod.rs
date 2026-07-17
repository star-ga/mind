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

mod resolve;

use std::collections::BTreeSet;
use std::collections::HashMap;

use crate::ast::BinOp;

use crate::ast::Literal;

use crate::ast::Module;

use crate::ast::Node;

use crate::ast::Span as AstSpan;
use crate::ast::TypeAnn;

use crate::diagnostics::{Diagnostic as Pretty, Severity, Span};

use crate::linalg;
use crate::shapes::engine;
use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;
use crate::types::TensorType;
use crate::types::ValueType;

#[derive(Debug)]
pub struct TypeErrSpan {
    pub msg: String,
    pub span: AstSpan,
}

pub type TypeEnv = HashMap<String, ValueType>;

/// A tensor parameter signature: `(arg_position, param_name, dims, dtype)`.
/// `arg_position` is the 0-based index in the full param list so call arguments
/// (some of which may be non-tensor) can be correlated (RFC 0012 Phase A).
type TensorParamSig = (usize, String, Vec<String>, String);

/// Maps a function name to its tensor-param signatures, used by call-site
/// symbolic-dim checks without a second module walk.
type FnTensorSigs = HashMap<String, Vec<TensorParamSig>>;

/// RFC 0005 Phase B — an intra-module function's call signature, captured
/// from its `Node::FnDef` so call sites can validate arity.
///
/// Only the parameter count is recorded: arity is the one call property that
/// is *always* soundly knowable from the AST regardless of the loose i64 ABI
/// (every intra-module call result and most locals collapse to `ScalarI64`,
/// literals widen freely, struct/aggregate values are i64 heap addresses, and
/// tensor-shape agreement is the dedicated RFC 0012 symbolic pass's job). It
/// covers generic functions too — `fn id<T>(x: T)` has a known arity of 1
/// even though `T` binds to anything at the call site. See `build_intra_fn_sigs`.
#[derive(Debug, Clone)]
struct IntraFnSig {
    param_count: usize,
    /// Declared parameter TypeAnns, carried for the Bug #38 fixed-buffer /
    /// growable-`bytes` mismatch check ONLY. The general per-arg type check
    /// stays disabled under the loose i64 ABI; see `check_intra_fn_call`.
    param_types: Vec<TypeAnn>,
    /// Declared return TypeAnn (`None` when the fn omits `-> T`). Read by
    /// `confident_scalar_class` so a `Call` to an intra-module fn resolves to
    /// its *declared* return class — never the loose i64 default. Used only for
    /// the RFC 0011 int↔float class checks; a `None`/non-scalar return keeps a
    /// call opaque (class `None`), so untyped callees never false-positive.
    ret_type: Option<TypeAnn>,
}

/// Maps an intra-module function name to its captured call signature.
type IntraFnSigs = HashMap<String, IntraFnSig>;

const TYPE_ERR_CODE: &str = "E2001";
const SHAPE_BROADCAST_CODE: &str = "E2101";
const SHAPE_RANK_CODE: &str = "E2102";
const SHAPE_INNER_DIM_CODE: &str = "E2103";
/// Implicit integer narrowing (data-loss) without an explicit `as` cast.
/// The spec forbids silent dtype changes across an assignment boundary
/// (`grammar-syntax.ebnf:240` provides `AsCast = Expression "as" Type` for
/// explicit width changes; `language.md:181` requires operands to share a
/// dtype). A wider integer flowing into a narrower slot truncates at runtime
/// (confirmed miscompile: `i64 4294967297` into an `i32` slot yields `1`).
const NARROWING_CODE: &str = "E2004";

/// Intra-module call arity mismatch: a call to a module-level `fn` passes the
/// wrong number of arguments. Arity is always knowable from the AST regardless
/// of the loose i64 ABI, so this is enforced for every intra-module call (RFC
/// 0005 Phase B — closes the pre-registration signature soundness hole).
const CALL_ARITY_CODE: &str = "E2005";

/// Non-exhaustive `match` on a sum/enum type: the arms are enum-variant
/// patterns of a known `enum` that omit at least one variant and provide no
/// wildcard `_` / binding catch-all arm. A non-exhaustive match silently
/// returns a value at runtime instead of failing, so this closes that
/// soundness hole. Scoped to enum matches only — integer/literal matches
/// cannot be exhaustive without a wildcard and are not flagged.
const MATCH_NONEXHAUSTIVE_CODE: &str = "match::non_exhaustive";

/// Match arm SCALAR-CLASS mismatch (Finding 19): two arms of one `match`
/// resolve to different scalar classes (e.g. an integer tag arm and a float
/// payload arm) once payload sub-patterns bind at their declared variant type.
/// This is a real type error surfaced from function bodies, distinct from the
/// demoted exact-width `E2001` mismatch (which i32-vs-i64 siblings trigger and
/// which is intentionally tolerated as both are i64-backed).
const MATCH_ARM_MISMATCH_CODE: &str = "match::arm_mismatch";

/// Bug #38 — a fixed-size buffer value (`bytes[N]`, an opaque i64 handle to N
/// raw bytes, e.g. from `bytes[N].zero()` or `std.sha256.hash`) flows into a
/// parameter declared as the GROWABLE `bytes` vec record `[addr|len|cap]`.
/// The two share the i64 ABI but have incompatible layouts: reading `.length`
/// on a fixed buffer reads raw payload bytes as the vec len field — a silent
/// miscompile. Fail loud rather than emit it (consistent with the #306
/// fail-closed philosophy). NARROW by construction: fires ONLY on the exact
/// growable-`bytes`-param + fixed-`bytes[N]`-arg pairing.
const FIXED_BYTES_INTO_VEC_CODE: &str = "E2006";

/// A `return <value>` whose value is a FLOAT-class scalar (`f32`/`f64`) while
/// the enclosing function's declared return type is an INTEGER-class scalar
/// (`i32`/`i64`/`bool`). Without this the check phase passes and `mlir-opt`
/// later rejects the artifact with an opaque
/// "type of return operand 0 ('f64') doesn't match function result type
/// ('i64')". Fired ONLY on the int-vs-float scalar CLASS mismatch — NEVER on
/// an exact-width sibling (`i32` return from an `i64` fn stays valid) and
/// NEVER symmetrically: only a *confidently-float* return value triggers it,
/// because the loose i64 ABI defaults every untyped same-file call to
/// `ScalarI64`, so a symmetric "int value into f64 fn" rule would
/// false-positive on `return helper()` in an `-> f64` body. A float value is
/// never a default, so this direction is sound.
const RETURN_TYPE_MISMATCH_CODE: &str = "E2010";

/// An `if`/`while` condition that infers to a FLOAT-class scalar (`f32`/`f64`)
/// and is NOT a boolean-intent expression (a comparison `<,<=,>,>=,==,!=`, a
/// logical `&&`/`||`, or a `!`). A raw float in condition position lowers to
/// `arith.cmpi` on an `f64` at `mlir-opt` — a late, opaque failure. Comparisons
/// are excluded because `infer_expr` reports `f64 > f64` as `ScalarF64` (the
/// operand class, not `bool`), so a naive "float condition" rule would reject
/// the valid `if a > b` over floats. Fires only on a directly-float condition
/// (`if 1.5`, `while fparam`), never on a comparison/logical/negation.
const COND_TYPE_MISMATCH_CODE: &str = "E2011";

/// A `Node::Binary` (arithmetic OR comparison) that mixes a confidently-Int
/// operand with a confidently-Float operand (`1 + 2.0`, `i64param < f64param`).
/// MIND has no implicit numeric coercion (RFC 0011); such an expression lowers
/// to an ill-typed `arith` op and fails late/opaquely at `mlir-opt`. Fires ONLY
/// when BOTH operand classes are annotation/literal-derived (see
/// `confident_scalar_class`) — a loose-typed operand keeps the pair `None` and
/// never triggers it. Closes the `i64 < f64` condition hole without touching
/// `cond_is_boolean_intent`.
const MIXED_CLASS_BINOP_CODE: &str = "E2013";

/// A `let`/assignment whose scalar-annotation class disagrees with the class of
/// its value (`let x: f64 = 5`, `let y: i64 = 1.5`). No implicit int↔float
/// conversion exists (RFC 0011). Fires ONLY when BOTH the annotation and the
/// value classes are annotation/literal-derived.
///
/// NOTE: the design note proposed `E2012` for this diagnostic, but that code is
/// already taken by `resolve.rs::FN_VALUE_CALL_CODE`; `E2015` is the free
/// successor. (`E2013` mixed-binop keeps its proposed value.)
const LET_CLASS_MISMATCH_CODE: &str = "E2015";

/// A numeric `as bool` cast (`3 as bool`). Casting a number to `bool` is not a
/// defined conversion in MIND; the check phase rejected it here instead of
/// letting the raw integer/float bits pass through unnormalised. Use `x != 0`.
const AS_BOOL_CODE: &str = "E2016";

/// An `import std.X` line was parsed but this `mindc` binary was built
/// WITHOUT the std surface compiled in (neither `std-surface` nor
/// `cross-module-imports`). Without that surface the bundled stdlib
/// export set is empty, so every `map_new` / `jv_parse` / `vec_*` call
/// resolves nowhere and the user otherwise gets a confusing storm of
/// per-call `E2003 unsupported call` errors. We instead fail LOUD ONCE
/// at the import span pointing at the real cause: the feature-stripped
/// build. This code is only ever emitted on the featureless path, so a
/// `std-surface`-enabled (default) build is byte-for-byte unchanged.
#[cfg(not(any(feature = "std-surface", feature = "cross-module-imports")))]
const STD_IMPORT_NO_SURFACE_CODE: &str = "E2007";

/// True for the RFC 0012 shape-diagnostic codes. Used to keep the
/// additive FnDef-body shape pass from contributing non-shape errors
/// (see the FnDef arm in `check_module_types_in_file`).
fn is_shape_diag_code(code: &str) -> bool {
    matches!(
        code,
        "E2101" | "E2102" | "E2103" | "shape::matmul_mismatch" | "shape::broadcast_mismatch"
    ) || code.starts_with("shape::")
}

/// Bit-width of an integer scalar `ValueType`, or `None` for non-integer
/// scalars / aggregates. `ScalarBool` is intentionally excluded — a
/// `bool`-into-integer flow is a distinct type mismatch, not a narrowing.
/// Note: `u32` collapses to `ScalarI32` in v1 (`valuetype_from_ann`), so it
/// ranks at width 32 here, which is what we want for width-narrowing.
fn int_scalar_bits(v: &ValueType) -> Option<u32> {
    match v {
        ValueType::ScalarI32 => Some(32),
        ValueType::ScalarI64 => Some(64),
        _ => None,
    }
}

/// True when assigning a value of `from` into a slot declared `to` would be
/// an *implicit integer narrowing* — i.e. both are integer scalars and the
/// destination is strictly narrower than the source. Such an assignment
/// silently truncates at runtime, so the spec requires an explicit `as` cast.
///
/// Widening (e.g. `i32` -> `i64`) is value-preserving and is *not* flagged:
/// it loses no data, the keystone/self-host already rely on it, and the
/// spec's "operands MUST share a dtype" rule (`language.md:181`) constrains
/// arithmetic operand dtypes, not value-preserving assignment widening — for
/// which `AsCast` remains available but is not mandated.
fn is_implicit_narrowing(to: &ValueType, from: &ValueType) -> bool {
    match (int_scalar_bits(to), int_scalar_bits(from)) {
        (Some(to_bits), Some(from_bits)) => to_bits < from_bits,
        _ => false,
    }
}

fn dtype_name(dtype: &DType) -> &'static str {
    match dtype {
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::Q16 => "q16",
    }
}

fn format_shape(shape: &[ShapeDim]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(sym) => sym.to_string(),
        })
        .collect();
    format!("({})", dims.join(","))
}

fn format_usize_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", dims.join(","))
}

fn describe_tensor(tensor: &TensorType) -> String {
    format!(
        "Tensor[{}, {}]",
        dtype_name(&tensor.dtype),
        format_shape(&tensor.shape)
    )
}

fn describe_value_type(v: &ValueType) -> String {
    match v {
        ValueType::ScalarI32 => "Scalar[i32]".to_string(),
        ValueType::ScalarI64 => "Scalar[i64]".to_string(),
        ValueType::ScalarF32 => "Scalar[f32]".to_string(),
        ValueType::ScalarF64 => "Scalar[f64]".to_string(),
        ValueType::ScalarBool => "Scalar[bool]".to_string(),
        ValueType::Tensor(tensor) => describe_tensor(tensor),
        ValueType::GradMap(entries) => {
            let mut parts = Vec::new();
            for (name, tensor) in entries {
                parts.push(format!("{}: {}", name, describe_tensor(tensor)));
            }
            format!("GradMap{{{}}}", parts.join(", "))
        }
    }
}

fn dim_display(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

fn binop_display(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
    }
}

fn shape_op_for_binop(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "tensor.add",
        BinOp::Sub => "tensor.sub",
        BinOp::Mul => "tensor.mul",
        BinOp::Div => "tensor.div",
        BinOp::Mod => "tensor.mod",
        BinOp::Lt => "tensor.lt",
        BinOp::Le => "tensor.le",
        BinOp::Gt => "tensor.gt",
        BinOp::Ge => "tensor.ge",
        BinOp::Eq => "tensor.eq",
        BinOp::Ne => "tensor.ne",
    }
}

fn concrete_shape(shape: &[ShapeDim]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => out.push(*n),
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(out)
}

fn shape_from_usize(shape: &[usize]) -> Vec<ShapeDim> {
    shape.iter().copied().map(ShapeDim::Known).collect()
}

fn shape_engine_error(op_display: &str, err: engine::ShapeError, span: AstSpan) -> TypeErrSpan {
    match err.kind {
        engine::ShapeErrorKind::UnknownOp => TypeErrSpan {
            msg: format!("shape rule not defined for `{op_display}`"),
            span,
        },
        engine::ShapeErrorKind::RankMismatch {
            expected,
            actual_lhs,
            actual_rhs,
        } => {
            let expected_display = if op_display.contains("matmul") && actual_rhs.is_some() {
                format!(
                    "matmul inner dimension mismatch (lhs.shape[1]={} vs rhs.shape[0]={})",
                    actual_lhs.get(1).copied().unwrap_or(0),
                    actual_rhs
                        .as_ref()
                        .and_then(|rhs| rhs.first().copied())
                        .unwrap_or(0)
                )
            } else {
                expected
            };
            let rhs_str = actual_rhs
                .as_ref()
                .map(|rhs| format!(", rhs={}", format_usize_shape(rhs)))
                .unwrap_or_default();
            TypeErrSpan {
                msg: format!(
                    "rank mismatch for `{op_display}`: expected {expected_display}, got lhs={}{}",
                    format_usize_shape(&actual_lhs),
                    rhs_str
                ),
                span,
            }
        }
        engine::ShapeErrorKind::BroadcastError { lhs, rhs } => TypeErrSpan {
            msg: format!(
                "cannot broadcast shapes {} and {} for `{op_display}`",
                format_usize_shape(&lhs),
                format_usize_shape(&rhs)
            ),
            span,
        },
    }
}

fn promote_scalar_to(dtype: DType) -> Option<ValueType> {
    match dtype {
        DType::F32 => Some(ValueType::ScalarI32),
        DType::I32 => Some(ValueType::ScalarI32),
        _ => None,
    }
}

fn combine_dtypes(lhs: &ValueType, rhs: &ValueType) -> Option<DType> {
    match (lhs, rhs) {
        (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
            if tl.dtype == tr.dtype {
                Some(tl.dtype.clone())
            } else {
                None
            }
        }
        (ValueType::Tensor(t), ValueType::ScalarI32)
        | (ValueType::ScalarI32, ValueType::Tensor(t))
        | (ValueType::Tensor(t), ValueType::ScalarF32)
        | (ValueType::ScalarF32, ValueType::Tensor(t)) => {
            if t.dtype == DType::F32 {
                promote_scalar_to(t.dtype.clone()).map(|_| DType::F32)
            } else {
                None
            }
        }
        (ValueType::ScalarI32, ValueType::ScalarI32) => None,
        (ValueType::ScalarF32, ValueType::ScalarF32) => Some(DType::F32),
        (ValueType::ScalarF64, ValueType::ScalarF64) => Some(DType::F32),
        (ValueType::ScalarI64, ValueType::ScalarI64) => None,
        (ValueType::ScalarF64, ValueType::ScalarI64)
        | (ValueType::ScalarI64, ValueType::ScalarF64) => Some(DType::F32),
        (ValueType::ScalarF32, ValueType::ScalarI32)
        | (ValueType::ScalarI32, ValueType::ScalarF32) => Some(DType::F32),
        (ValueType::ScalarBool, ValueType::ScalarBool) => None,
        (ValueType::GradMap(_), _) | (_, ValueType::GradMap(_)) => None,
        _ => None, // Other scalar combinations
    }
}

fn linalg_type_err(op: &str, span: AstSpan, msg: String) -> TypeErrSpan {
    TypeErrSpan {
        msg: format!("`{op}`: {msg}"),
        span,
    }
}

fn normalize_axis(axis: i32, rank: usize, span: AstSpan, op: &str) -> Result<usize, TypeErrSpan> {
    let rank_i32 = rank as i32;
    let idx = if axis < 0 { rank_i32 + axis } else { axis };
    if idx < 0 || idx >= rank_i32 {
        Err(TypeErrSpan {
            msg: format!("axis {axis} out of range for `{op}` (rank {rank})"),
            span,
        })
    } else {
        Ok(idx as usize)
    }
}

fn dim_len(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

fn slice_len(start: i32, end: i32) -> Option<usize> {
    if start < 0 || end < start {
        None
    } else {
        Some((end - start) as usize)
    }
}

fn slice_len_with_step(len: Option<usize>, start: i32, end: i32, step: i32) -> Option<usize> {
    if step == 0 {
        return None;
    }
    let len = len?;
    let len_i = len as i64;
    let step_i = step as i64;

    let mut start_i = start as i64;
    let mut end_i = end as i64;

    if step_i > 0 {
        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < 0 {
            start_i = 0;
        }
        if start_i > len_i {
            start_i = len_i;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < 0 {
            end_i = 0;
        }
        if end_i > len_i {
            end_i = len_i;
        }

        if start_i >= end_i {
            Some(0)
        } else {
            let diff = end_i - start_i;
            Some(((diff + step_i.abs() - 1) / step_i.abs()) as usize)
        }
    } else {
        if len == 0 {
            return Some(0);
        }

        if start_i < 0 {
            start_i += len_i;
        }
        if start_i < -1 {
            start_i = -1;
        }
        if start_i >= len_i {
            start_i = len_i - 1;
        }

        if end_i < 0 {
            end_i += len_i;
        }
        if end_i < -1 {
            end_i = -1;
        }
        if end_i >= len_i {
            end_i = len_i - 1;
        }

        if start_i <= end_i {
            Some(0)
        } else {
            let diff = start_i - end_i;
            Some(((diff + (-step_i) - 1) / (-step_i)) as usize)
        }
    }
}

fn conv_channels_compatible(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(sa), ShapeDim::Sym(sb)) => sa == sb,
        _ => true,
    }
}

fn conv_output_dim(
    input: &ShapeDim,
    kernel: Option<&ShapeDim>,
    stride: usize,
    padding: ConvPadding,
    span: AstSpan,
    axis: &str,
) -> Result<ShapeDim, TypeErrSpan> {
    let input_known = dim_len(input);
    let kernel_known = kernel.and_then(dim_len);
    let result = match padding {
        ConvPadding::Valid => linalg::conv_output_dim_valid(input_known, kernel_known, stride),
        ConvPadding::Same => linalg::conv_output_dim_same(input_known, stride),
    };
    match result {
        Ok(Some(v)) => Ok(ShapeDim::Known(v)),
        Ok(None) => Ok(ShapeDim::Sym(fresh_symbol(&format!("_conv_{axis}"), span))),
        Err(msg) => Err(TypeErrSpan {
            msg: format!("`tensor.conv2d`: {msg} ({axis})"),
            span,
        }),
    }
}

fn normalize_axes_list(
    axes: &[i32],
    rank: usize,
    span: AstSpan,
    op: &str,
) -> Result<Vec<usize>, TypeErrSpan> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut normalized = Vec::new();
    for &axis in axes {
        let idx = normalize_axis(axis, rank, span, op)?;
        if !seen.insert(idx) {
            return Err(TypeErrSpan {
                msg: format!("duplicate axis {axis} in `{op}`"),
                span,
            });
        }
        normalized.push(idx);
    }
    normalized.sort_unstable();
    Ok(normalized)
}

fn normalize_reduce_axes(
    axes: &[i32],
    rank: usize,
    span: AstSpan,
    op: &str,
) -> Result<Vec<usize>, TypeErrSpan> {
    if axes.is_empty() {
        return Ok((0..rank).collect());
    }
    normalize_axes_list(axes, rank, span, op)
}

fn reduce_shape(shape: &[ShapeDim], axes: &[usize], keepdims: bool) -> Vec<ShapeDim> {
    if keepdims {
        let mut out = shape.to_vec();
        for &axis in axes {
            if axis < out.len() {
                out[axis] = ShapeDim::Known(1);
            }
        }
        out
    } else {
        let axis_set: BTreeSet<usize> = axes.iter().cloned().collect();
        let mut out = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if !axis_set.contains(&idx) {
                out.push(dim.clone());
            }
        }
        out
    }
}

fn known_product(shape: &[ShapeDim]) -> Option<usize> {
    let mut total = 1usize;
    for dim in shape {
        match dim {
            ShapeDim::Known(n) => {
                total = total.checked_mul(*n)?;
            }
            ShapeDim::Sym(_) => return None,
        }
    }
    Some(total)
}

fn normalize_expand_axis(axis: i32, rank: usize, span: AstSpan) -> Result<usize, TypeErrSpan> {
    let extended = rank + 1;
    let idx = if axis < 0 {
        (extended as i32) + axis
    } else {
        axis
    };
    if idx < 0 || idx > extended as i32 - 1 {
        Err(TypeErrSpan {
            msg: format!("axis {axis} out of range for `tensor.expand_dims` (rank {rank})"),
            span,
        })
    } else {
        Ok(idx as usize)
    }
}

fn compute_squeeze_axes(
    shape: &[ShapeDim],
    axes: &[i32],
    span: AstSpan,
) -> Result<Vec<usize>, TypeErrSpan> {
    if axes.is_empty() {
        let mut remove = Vec::new();
        for (idx, dim) in shape.iter().enumerate() {
            if matches!(dim, ShapeDim::Known(1)) {
                remove.push(idx);
            }
        }
        return Ok(remove);
    }
    let normalized = normalize_axes_list(axes, shape.len(), span, "tensor.squeeze")?;
    for &axis in &normalized {
        match shape.get(axis) {
            Some(ShapeDim::Known(1)) => {}
            Some(_) => {
                return Err(TypeErrSpan {
                    msg: format!("cannot squeeze axis {axis}: dimension is not 1"),
                    span,
                });
            }
            None => {
                return Err(TypeErrSpan {
                    msg: format!("axis {axis} out of range for `tensor.squeeze`"),
                    span,
                });
            }
        }
    }
    Ok(normalized)
}

fn broadcast_shapes(a: &[ShapeDim], b: &[ShapeDim]) -> Option<Vec<ShapeDim>> {
    let mut out = Vec::new();
    let mut i = a.len() as isize - 1;
    let mut j = b.len() as isize - 1;

    while i >= 0 || j >= 0 {
        let da = if i >= 0 {
            a[i as usize].clone()
        } else {
            ShapeDim::Known(1)
        };
        let db = if j >= 0 {
            b[j as usize].clone()
        } else {
            ShapeDim::Known(1)
        };

        let dim = match (da, db) {
            (ShapeDim::Known(x), ShapeDim::Known(y)) => {
                if x == y {
                    ShapeDim::Known(x)
                } else if x == 1 {
                    ShapeDim::Known(y)
                } else if y == 1 {
                    ShapeDim::Known(x)
                } else {
                    return None;
                }
            }
            (ShapeDim::Sym(s1), ShapeDim::Sym(s2)) => {
                if s1 == s2 {
                    ShapeDim::Sym(s1)
                } else {
                    return None;
                }
            }
            (ShapeDim::Sym(sym), ShapeDim::Known(n)) | (ShapeDim::Known(n), ShapeDim::Sym(sym)) => {
                if n == 1 {
                    ShapeDim::Sym(sym)
                } else {
                    return None;
                }
            }
        };

        out.push(dim);
        i -= 1;
        j -= 1;
    }

    out.reverse();
    Some(out)
}

/// Levenshtein edit distance between two identifiers (for "did you mean" hints).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// The known identifier closest to `name` within a small edit distance, for a
/// "did you mean" hint. Deterministic regardless of `env` iteration order: picks the
/// minimum distance, breaking ties by the lexicographically-smallest name — so the
/// suggestion never depends on HashMap ordering.
fn closest_identifier(name: &str, env: &TypeEnv) -> Option<String> {
    let max_dist = (name.chars().count() / 2).clamp(1, 2);
    let mut best: Option<(usize, &String)> = None;
    for known in env.keys() {
        let d = levenshtein(name, known);
        if d == 0 || d > max_dist {
            continue;
        }
        let better = match &best {
            None => true,
            Some((bd, bk)) => d < *bd || (d == *bd && known < *bk),
        };
        if better {
            best = Some((d, known));
        }
    }
    best.map(|(_, k)| k.clone())
}

fn infer_expr(node: &Node, env: &TypeEnv) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    match node {
        Node::Lit(Literal::Int(_), span) => Ok((ValueType::ScalarI32, *span)),
        Node::Lit(Literal::Float(_), span) => Ok((ValueType::ScalarF64, *span)),
        Node::Lit(Literal::Str(_), span) => Ok((ValueType::ScalarI32, *span)), // strings treated as opaque
        // Loop-control statements carry no value; type as the benign scalar.
        #[cfg(feature = "std-surface")]
        Node::Break { span } | Node::Continue { span } => Ok((ValueType::ScalarI32, *span)),
        Node::Lit(Literal::Ident(name), span) => {
            if let Some(t) = env.get(name).cloned() {
                Ok((t, *span))
            } else if split_enum_variant_path(name)
                .and_then(|(e, v)| enum_variants_of(&e).map(|vs| vs.iter().any(|x| x == &v)))
                .unwrap_or(false)
            {
                // An enum-variant constructor used as a *value* (`Mode::On`) is a
                // valid expression but is not a value-env binding. Resolve it via
                // the enum registry to the i64 ABI (the discriminant) rather than
                // raising E2002 — matches the loose i64 ABI every other value
                // form lowers to. (Unit variants only; payload variants such as
                // `Some(x)` are call expressions handled by `infer_call`.)
                Ok((ValueType::ScalarI64, *span))
            } else {
                Err(TypeErrSpan {
                    msg: match closest_identifier(name, env) {
                        Some(s) => format!("unknown identifier `{name}` — did you mean `{s}`?"),
                        None => format!("unknown identifier `{name}`"),
                    },
                    span: *span,
                })
            }
        }
        Node::Paren(inner, span) => {
            let (ty, _) = infer_expr(inner, env)?;
            Ok((ty, *span))
        }
        // W1.5f: postfix `?`. The PARSER is the enforcement point for "?
        // outside a Result/Option-returning fn" — it rejects that BEFORE
        // type-check with a clear parse error (no broken artifact is ever
        // emitted), because `?` is only produced when the enclosing fn's
        // declared return type is `Result`/`Option`. Here we type-check `inner`
        // (surfacing any error in the operand) and type the unwrapped Ok/Some
        // value as the loose i64 ABI every enum payload lowers to.
        Node::Try { inner, span, .. } => {
            let _ = infer_expr(inner, env)?;
            Ok((ValueType::ScalarI64, *span))
        }
        Node::Tuple { elements, span } => {
            // Infer type from the last element (matches lowering semantics).
            if let Some(last) = elements.last() {
                let (ty, _) = infer_expr(last, env)?;
                Ok((ty, *span))
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::Call { callee, args, span } => infer_call(callee, args, *span, env),
        Node::CallGrad { loss, wrt, span } => infer_grad(loss, wrt, *span, env),
        Node::CallTensorSum {
            x,
            axes,
            keepdims,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_norm =
                        normalize_reduce_axes(axes, tensor.shape.len(), *span, "tensor.sum")?;
                    let shape = reduce_shape(&tensor.shape, &axes_norm, *keepdims);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sum` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTensorMean {
            x,
            axes,
            keepdims,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_norm =
                        normalize_reduce_axes(axes, tensor.shape.len(), *span, "tensor.mean")?;
                    let shape = reduce_shape(&tensor.shape, &axes_norm, *keepdims);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.mean` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallReshape { x, dims, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let new_shape = shape_from_dims(dims);
                    if new_shape.len() != tensor.shape.len() {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.reshape` expects {} dimensions but got {}",
                                tensor.shape.len(),
                                new_shape.len()
                            ),
                            span: *span,
                        });
                    }
                    if let (Some(old), Some(new)) =
                        (known_product(&tensor.shape), known_product(&new_shape))
                    {
                        if old != new {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.reshape` element count mismatch: {old} vs {new}"
                                ),
                                span: *span,
                            });
                        }
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, new_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.reshape` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallExpandDims { x, axis, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let rank = tensor.shape.len();
                    let axis = normalize_expand_axis(*axis, rank, *span)?;
                    let mut shape = tensor.shape;
                    shape.insert(axis, ShapeDim::Known(1));
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.expand_dims` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSqueeze { x, axes, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axes_to_remove = compute_squeeze_axes(&tensor.shape, axes, *span)?;
                    let axis_set: BTreeSet<usize> = axes_to_remove.iter().cloned().collect();
                    let mut shape = Vec::new();
                    for (idx, dim) in tensor.shape.iter().enumerate() {
                        if !axis_set.contains(&idx) {
                            shape.push(dim.clone());
                        }
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.squeeze` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTranspose { x, axes, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let rank = tensor.shape.len();
                    let perm = if let Some(list) = axes {
                        linalg::normalize_permutation(list, rank)
                            .map_err(|msg| linalg_type_err("tensor.transpose", *span, msg))?
                    } else {
                        linalg::default_transpose(rank)
                    };
                    if perm.len() != rank {
                        return Err(linalg_type_err(
                            "tensor.transpose",
                            *span,
                            format!("expected {} axes but got {}", rank, perm.len()),
                        ));
                    }
                    let shape = linalg::permute_shape(&tensor.shape, &perm);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.transpose` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallIndex { x, axis, i, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    if tensor.shape.is_empty() {
                        return Err(TypeErrSpan {
                            msg: "`tensor.index` requires a tensor with rank >= 1".to_string(),
                            span: *span,
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.index")?;
                    if let Some(len) = dim_len(&tensor.shape[axis_norm]) {
                        if *i < 0 || (*i as usize) >= len {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.index`: index {i} out of bounds for axis {axis_norm} (len {len})"
                                ),
                                span: *span,
                            });
                        }
                    }
                    let mut shape = tensor.shape.clone();
                    shape.remove(axis_norm);
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.index` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSlice {
            x,
            axis,
            start,
            end,
            span,
        } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    if *start < 0 || *end < *start {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.slice` requires 0 <= start <= end (got start={start}, end={end})"
                            ),
                            span: *span,
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.slice")?;
                    if let Some(len) = dim_len(&tensor.shape[axis_norm]) {
                        if *end as usize > len {
                            return Err(TypeErrSpan {
                                msg: format!(
                                    "`tensor.slice`: end {end} out of bounds for axis {axis_norm} (len {len})"
                                ),
                                span: *span,
                            });
                        }
                    }
                    let new_dim = match (dim_len(&tensor.shape[axis_norm]), slice_len(*start, *end))
                    {
                        (Some(_), Some(len)) => ShapeDim::Known(len),
                        _ => ShapeDim::Sym(fresh_symbol("_slice", *span)),
                    };
                    let mut shape = tensor.shape.clone();
                    shape[axis_norm] = new_dim;
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.slice` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallSliceStride {
            x,
            axis,
            start,
            end,
            step,
            span,
        } => {
            if *step == 0 {
                return Err(TypeErrSpan {
                    msg: "`tensor.slice_stride` requires step != 0".to_string(),
                    span: *span,
                });
            }
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => {
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.slice_stride")?;
                    let dim = tensor.shape[axis_norm].clone();
                    let new_dim = if let Some(len) = dim_len(&dim) {
                        let Some(result_len) = slice_len_with_step(Some(len), *start, *end, *step)
                        else {
                            return Err(TypeErrSpan {
                                msg: "`tensor.slice_stride` bounds are invalid for the axis"
                                    .to_string(),
                                span: *span,
                            });
                        };
                        ShapeDim::Known(result_len)
                    } else if (*step > 0 && *start >= *end) || (*step < 0 && *start <= *end) {
                        ShapeDim::Known(0)
                    } else {
                        ShapeDim::Sym(fresh_symbol("_slice_stride", *span))
                    };
                    let mut shape = tensor.shape.clone();
                    shape[axis_norm] = new_dim;
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.slice_stride` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallGather { x, axis, idx, span } => {
            let (x_ty, _) = infer_expr(x, env)?;
            let (idx_ty, idx_span) = infer_expr(idx, env)?;
            match (x_ty, idx_ty) {
                (ValueType::Tensor(tensor), ValueType::Tensor(idx_tensor)) => {
                    if idx_tensor.dtype != DType::I32 {
                        return Err(TypeErrSpan {
                            msg: "`tensor.gather` requires `idx` to be an i32 tensor".to_string(),
                            span: idx.span(),
                        });
                    }
                    let axis_norm =
                        normalize_axis(*axis, tensor.shape.len(), *span, "tensor.gather")?;
                    let mut shape = Vec::new();
                    shape.extend_from_slice(&tensor.shape[..axis_norm]);
                    shape.extend(idx_tensor.shape.iter().cloned());
                    if axis_norm < tensor.shape.len() {
                        shape.extend_from_slice(&tensor.shape[axis_norm + 1..]);
                    }
                    Ok((
                        ValueType::Tensor(TensorType::new(tensor.dtype, shape)),
                        *span,
                    ))
                }
                (ValueType::Tensor(_), _) => Err(TypeErrSpan {
                    msg: "`tensor.gather` requires `idx` to be a tensor".to_string(),
                    span: idx_span,
                }),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.gather` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallDot { a, b, span } => {
            let (lt, _) = infer_expr(a, env)?;
            let (rt, _) = infer_expr(b, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.dot` dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    let info = linalg::compute_matmul_shape_info(&tl.shape, &tr.shape)
                        .map_err(|msg| linalg_type_err("tensor.dot", *span, msg))?;
                    Ok((
                        ValueType::Tensor(TensorType::new(tl.dtype.clone(), info.result_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.dot` requires tensor arguments".to_string(),
                    span: *span,
                }),
            }
        }
        Node::CallMatMul { a, b, span } => {
            let (lt, _) = infer_expr(a, env)?;
            let (rt, _) = infer_expr(b, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`tensor.matmul` dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    if let (Some(lhs), Some(rhs)) =
                        (concrete_shape(&tl.shape), concrete_shape(&tr.shape))
                    {
                        match engine::infer_output_shape("tensor.matmul", &[&lhs, &rhs]) {
                            Ok(out) => {
                                return Ok((
                                    ValueType::Tensor(TensorType::new(
                                        tl.dtype.clone(),
                                        shape_from_usize(&out),
                                    )),
                                    *span,
                                ));
                            }
                            Err(e) => return Err(shape_engine_error("tensor.matmul", e, *span)),
                        }
                    }
                    let info = linalg::compute_matmul_shape_info(&tl.shape, &tr.shape)
                        .map_err(|msg| linalg_type_err("tensor.matmul", *span, msg))?;
                    Ok((
                        ValueType::Tensor(TensorType::new(tl.dtype.clone(), info.result_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`tensor.matmul` requires tensor arguments".to_string(),
                    span: *span,
                }),
            }
        }
        // RFC 0012 Phase B — `A @ B` matmul operator.
        //
        // Type rule: A: Tensor<T,[M,K]>, B: Tensor<T,[K,N]> → Tensor<T,[M,N]>.
        // Inner-dimension mismatch → `shape::matmul_mismatch`.
        // Uses the same shape machinery as `CallMatMul` above.
        Node::TensorMatmul { lhs, rhs, span } => {
            let (lt, _) = infer_expr(lhs, env)?;
            let (rt, _) = infer_expr(rhs, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "`@` dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    // Inner-dimension check: A[M,K] @ B[K,N] requires K matches.
                    if let (Some(lhs_shape), Some(rhs_shape)) =
                        (concrete_shape(&tl.shape), concrete_shape(&tr.shape))
                    {
                        match engine::infer_output_shape("tensor.matmul", &[&lhs_shape, &rhs_shape])
                        {
                            Ok(out) => {
                                return Ok((
                                    ValueType::Tensor(TensorType::new(
                                        tl.dtype.clone(),
                                        shape_from_usize(&out),
                                    )),
                                    *span,
                                ));
                            }
                            Err(_) => {
                                // Inner-dimension mismatch: K on lhs vs K on rhs.
                                let lhs_k = lhs_shape.get(1).copied().unwrap_or(lhs_shape[0]);
                                let rhs_k = rhs_shape[0];
                                return Err(TypeErrSpan {
                                    msg: format!(
                                        "`@` inner dimension mismatch: \
                                         lhs inner dim {} vs rhs outer dim {}",
                                        lhs_k, rhs_k
                                    ),
                                    span: *span,
                                });
                            }
                        }
                    }
                    // Symbolic shapes: use linalg helper.
                    let info =
                        linalg::compute_matmul_shape_info(&tl.shape, &tr.shape).map_err(|msg| {
                            TypeErrSpan {
                                msg: format!("`@` shape error: {msg}"),
                                span: *span,
                            }
                        })?;
                    Ok((
                        ValueType::Tensor(TensorType::new(tl.dtype.clone(), info.result_shape)),
                        *span,
                    ))
                }
                _ => Err(TypeErrSpan {
                    msg: "`@` requires tensor operands on both sides".to_string(),
                    span: *span,
                }),
            }
        }
        // RFC 0012 Phase B — elementwise `.+ .- .* ./` operators.
        //
        // Type rule: same shape and dtype → same shape; scalar broadcast
        // is the strict subset supported in Phase B (full prefix-rank
        // broadcasting deferred to Phase B.2).
        // Shape mismatch → `shape::broadcast_mismatch`.
        Node::TensorElemwise { lhs, rhs, span, .. } => {
            let (lt, _) = infer_expr(lhs, env)?;
            let (rt, _) = infer_expr(rhs, env)?;
            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if tl.dtype != tr.dtype {
                        return Err(TypeErrSpan {
                            msg: format!(
                                "elementwise operator dtype mismatch: left {} vs right {}",
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        });
                    }
                    if let Some(shape) = broadcast_shapes(&tl.shape, &tr.shape) {
                        Ok((
                            ValueType::Tensor(TensorType::new(tl.dtype.clone(), shape)),
                            *span,
                        ))
                    } else {
                        Err(TypeErrSpan {
                            msg: format!(
                                "elementwise operator shape mismatch: \
                                 cannot broadcast {} and {}",
                                format_shape(&tl.shape),
                                format_shape(&tr.shape),
                            ),
                            span: *span,
                        })
                    }
                }
                // Scalar broadcast: tensor .op scalar or scalar .op tensor.
                (
                    ValueType::Tensor(t),
                    ValueType::ScalarI32
                    | ValueType::ScalarI64
                    | ValueType::ScalarF32
                    | ValueType::ScalarF64,
                )
                | (
                    ValueType::ScalarI32
                    | ValueType::ScalarI64
                    | ValueType::ScalarF32
                    | ValueType::ScalarF64,
                    ValueType::Tensor(t),
                ) => Ok((ValueType::Tensor(t.clone()), *span)),
                _ => Err(TypeErrSpan {
                    msg: "elementwise operator requires tensor operands".to_string(),
                    span: *span,
                }),
            }
        }
        Node::CallTensorRelu { x, span } => {
            let (arg_ty, _) = infer_expr(x, env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => Ok((ValueType::Tensor(tensor), *span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.relu` requires a tensor argument".to_string(),
                    span: x.span(),
                }),
            }
        }
        Node::CallTensorRand { shape, span } => {
            let dims: Vec<ShapeDim> = shape.iter().map(|&d| ShapeDim::Known(d)).collect();
            Ok((
                ValueType::Tensor(TensorType {
                    dtype: DType::F32,
                    shape: dims,
                }),
                *span,
            ))
        }
        Node::CallTensorConv2d {
            x,
            w,
            stride_h,
            stride_w,
            padding,
            span,
        } => {
            if *stride_h == 0 || *stride_w == 0 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d`: strides must be positive".to_string(),
                    span: *span,
                });
            }

            let (x_ty, _) = infer_expr(x, env)?;
            let (w_ty, _) = infer_expr(w, env)?;
            let (x_tensor, w_tensor) = match (x_ty, w_ty) {
                (ValueType::Tensor(a), ValueType::Tensor(b)) => (a, b),
                (ValueType::Tensor(_), other) => {
                    return Err(TypeErrSpan {
                        msg: format!(
                            "`tensor.conv2d`: expected tensor weights but found {}",
                            describe_value_type(&other)
                        ),
                        span: w.span(),
                    });
                }
                (other, _) => {
                    return Err(TypeErrSpan {
                        msg: format!(
                            "`tensor.conv2d`: expected tensor input but found {}",
                            describe_value_type(&other)
                        ),
                        span: x.span(),
                    });
                }
            };

            if x_tensor.shape.len() != 4 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d` expects input layout NHWC (rank 4)".to_string(),
                    span: x.span(),
                });
            }
            if w_tensor.shape.len() != 4 {
                return Err(TypeErrSpan {
                    msg: "`tensor.conv2d` expects filter layout HWIO (rank 4)".to_string(),
                    span: w.span(),
                });
            }

            let in_channels = &x_tensor.shape[3];
            let kernel_channels = &w_tensor.shape[2];
            if !conv_channels_compatible(in_channels, kernel_channels) {
                return Err(TypeErrSpan {
                    msg: format!(
                        "`tensor.conv2d`: channel mismatch {} vs {}",
                        dim_display(in_channels),
                        dim_display(kernel_channels)
                    ),
                    span: *span,
                });
            }

            if let Some(kh) = dim_len(&w_tensor.shape[0]) {
                if kh == 0 {
                    return Err(TypeErrSpan {
                        msg: "`tensor.conv2d`: kernel height must be positive".to_string(),
                        span: w.span(),
                    });
                }
            }
            if let Some(kw) = dim_len(&w_tensor.shape[1]) {
                if kw == 0 {
                    return Err(TypeErrSpan {
                        msg: "`tensor.conv2d`: kernel width must be positive".to_string(),
                        span: w.span(),
                    });
                }
            }

            let dtype = if x_tensor.dtype == w_tensor.dtype {
                x_tensor.dtype.clone()
            } else if matches!(x_tensor.dtype, DType::F32) || matches!(w_tensor.dtype, DType::F32) {
                DType::F32
            } else {
                return Err(TypeErrSpan {
                    msg: format!(
                        "`tensor.conv2d`: incompatible dtypes {} and {}",
                        dtype_name(&x_tensor.dtype),
                        dtype_name(&w_tensor.dtype)
                    ),
                    span: *span,
                });
            };

            let out_h = conv_output_dim(
                &x_tensor.shape[1],
                Some(&w_tensor.shape[0]),
                *stride_h,
                *padding,
                *span,
                "h",
            )?;
            let out_w = conv_output_dim(
                &x_tensor.shape[2],
                Some(&w_tensor.shape[1]),
                *stride_w,
                *padding,
                *span,
                "w",
            )?;

            let out_shape = vec![
                x_tensor.shape[0].clone(),
                out_h,
                out_w,
                w_tensor.shape[3].clone(),
            ];

            Ok((ValueType::Tensor(TensorType::new(dtype, out_shape)), *span))
        }
        Node::Binary {
            op,
            left,
            right,
            span,
        } => {
            let (lt, _) = infer_expr(left, env)?;
            let (rt, _) = infer_expr(right, env)?;
            // Same-type scalar binary op returns that scalar type. Pre-RFC-0012
            // only ScalarI32 was fast-pathed here (i64 annotations resolved to
            // ScalarI32). RFC 0012 Phase A introduced ScalarI64/ScalarF64 as
            // distinct types; without these arms `i64 + i64` (and f64/bool)
            // fell through to the `_ => incompatible types` catch-all (E2001).
            match (&lt, &rt) {
                (ValueType::ScalarI32, ValueType::ScalarI32) => {
                    return Ok((ValueType::ScalarI32, *span));
                }
                (ValueType::ScalarI64, ValueType::ScalarI64) => {
                    return Ok((ValueType::ScalarI64, *span));
                }
                (ValueType::ScalarF32, ValueType::ScalarF32) => {
                    return Ok((ValueType::ScalarF32, *span));
                }
                (ValueType::ScalarF64, ValueType::ScalarF64) => {
                    return Ok((ValueType::ScalarF64, *span));
                }
                (ValueType::ScalarBool, ValueType::ScalarBool) => {
                    return Ok((ValueType::ScalarBool, *span));
                }
                _ => {}
            }

            match (&lt, &rt) {
                (ValueType::Tensor(tl), ValueType::Tensor(tr)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        if let (Some(lhs), Some(rhs)) =
                            (concrete_shape(&tl.shape), concrete_shape(&tr.shape))
                        {
                            match engine::infer_output_shape(shape_op_for_binop(op), &[&lhs, &rhs])
                            {
                                Ok(out) => {
                                    return Ok((
                                        ValueType::Tensor(TensorType::new(
                                            dtype,
                                            shape_from_usize(&out),
                                        )),
                                        *span,
                                    ));
                                }
                                Err(e) => {
                                    return Err(shape_engine_error(binop_display(op), e, *span));
                                }
                            }
                        }

                        if let Some(shape) = broadcast_shapes(&tl.shape, &tr.shape) {
                            Ok((ValueType::Tensor(TensorType::new(dtype, shape)), *span))
                        } else {
                            Err(TypeErrSpan {
                                msg: format!(
                                    "cannot broadcast shapes {} and {} for `{}`",
                                    format_shape(&tl.shape),
                                    format_shape(&tr.shape),
                                    binop_display(op)
                                ),
                                span: *span,
                            })
                        }
                    } else {
                        Err(TypeErrSpan {
                            msg: format!(
                                "dtype mismatch for `{}`: left {} vs right {}",
                                binop_display(op),
                                describe_tensor(tl),
                                describe_tensor(tr)
                            ),
                            span: *span,
                        })
                    }
                }
                (ValueType::Tensor(t), ValueType::ScalarI32)
                | (ValueType::ScalarI32, ValueType::Tensor(t)) => {
                    if let Some(dtype) = combine_dtypes(&lt, &rt) {
                        Ok((
                            ValueType::Tensor(TensorType::new(dtype, t.shape.clone())),
                            *span,
                        ))
                    } else {
                        let dtype_str = dtype_name(&t.dtype);
                        let message = match promote_scalar_to(t.dtype.clone()) {
                            Some(_) => format!(
                                "cannot apply `{}`: scalar promotion to tensor dtype `{}` is not supported",
                                binop_display(op),
                                dtype_str
                            ),
                            None => format!(
                                "cannot apply `{}`: tensor dtype `{}` does not support scalar operands",
                                binop_display(op),
                                dtype_str
                            ),
                        };
                        Err(TypeErrSpan {
                            msg: message,
                            span: *span,
                        })
                    }
                }
                _ => Err(TypeErrSpan {
                    msg: "incompatible types in binary operation".to_string(),
                    span: *span,
                }),
            }
        }
        Node::Let { value, .. } | Node::Assign { value, .. } | Node::LetTuple { value, .. } => {
            infer_expr(value, env)
        }
        // Function definitions don't have a value type in expression context
        Node::FnDef { span, .. } => Ok((ValueType::ScalarI32, *span)), // Placeholder
        Node::Return { value, span } => {
            if let Some(v) = value {
                infer_expr(v, env)
            } else {
                Ok((ValueType::ScalarI32, *span)) // Void return
            }
        }
        Node::Block { stmts, span } => {
            if let Some(last) = stmts.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::If {
            then_branch, span, ..
        } => {
            if let Some(last) = then_branch.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        // Import statements don't have a value type; they're module-level declarations
        Node::Import { span, .. } => Ok((ValueType::ScalarI32, *span)),
        Node::ArrayLit { elements, span } => {
            if let Some(first) = elements.first() {
                infer_expr(first, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::MapLit { entries, span } => {
            // A map literal is an i64 std.map handle. Validate each key/value
            // expression, then type as the i64 handle (the map ABI).
            for (k, v) in entries {
                let _ = infer_expr(k, env);
                let _ = infer_expr(v, env);
            }
            Ok((ValueType::ScalarI64, *span))
        }
        Node::SetLit { elements, span } => {
            // A set literal is an i64 std.map handle (a map keyed by elements).
            for e in elements {
                let _ = infer_expr(e, env);
            }
            Ok((ValueType::ScalarI64, *span))
        }
        Node::For {
            var, body, span, ..
        } => {
            let env = &mut env.clone();
            // Register loop variable
            env.insert(var.clone(), ValueType::ScalarI32);
            // Register let-bindings inside body
            for stmt in body {
                if let Node::Let { name, .. } = stmt {
                    env.insert(name.clone(), ValueType::ScalarI32);
                }
                if let Node::Assign { name, .. } = stmt {
                    if !env.contains_key(name) {
                        env.insert(name.clone(), ValueType::ScalarI32);
                    }
                }
            }
            if let Some(last) = body.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::ForEach {
            var,
            collection,
            body,
            span,
        } => {
            // Validate the collection expression, then type-check the body with
            // the element bound. The element is the `vec_get` result (an i64
            // handle/value under the std.vec ABI), so it types as ScalarI64.
            let _ = infer_expr(collection, env);
            let env = &mut env.clone();
            env.insert(var.clone(), ValueType::ScalarI64);
            for stmt in body {
                if let Node::Let { name, .. } = stmt {
                    env.insert(name.clone(), ValueType::ScalarI64);
                }
                if let Node::Assign { name, .. } = stmt {
                    if !env.contains_key(name) {
                        env.insert(name.clone(), ValueType::ScalarI64);
                    }
                }
            }
            if let Some(last) = body.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
        Node::Print { span, .. } => Ok((ValueType::ScalarI32, *span)),
        Node::Neg { operand, .. } => infer_expr(operand, env),
        // `!expr` yields a 0/1 boolean. Validate the operand infers, then type
        // the result as the operand's scalar type (mirrors `Neg`; the result is
        // truthy/falsy and feeds `if`/`while` conditions — enum_match #9).
        Node::Not { operand, .. } => infer_expr(operand, env),
        Node::MethodCall { receiver, span, .. } => {
            // Static/associated type-name call (`string.from_utf8_bytes(..)`): the
            // receiver is a TYPE name, not a value — don't resolve it as an
            // identifier (it would error "unknown identifier `string`"). The
            // result is an opaque handle (i64). Only when no local shadows the
            // name (a real value receiver keeps the normal path).
            if let Node::Lit(Literal::Ident(tn), _) = receiver.as_ref() {
                if (tn == "string" || tn == "String") && env.get(tn).is_none() {
                    return Ok((ValueType::ScalarI64, *span));
                }
            }
            let (recv_ty, _) = infer_expr(receiver, env)?;
            Ok((recv_ty, *span))
        }
        Node::FieldAccess { receiver, span, .. } => {
            let _ = infer_expr(receiver, env)?;
            Ok((ValueType::ScalarI32, *span))
        }
        // Phase 10.5 declarations are statement-level, not expression-level.
        // The module walker handles them; infer_expr is unreachable for them
        // but a clear error here keeps any internal misuse from compiling.
        Node::Const { span, .. }
        | Node::ExternConst { span, .. }
        | Node::TypeAlias { span, .. }
        | Node::Export { span, .. }
        | Node::StructDef { span, .. }
        | Node::EnumDef { span, .. } => Err(TypeErrSpan {
            msg: "declaration cannot be used as an expression".to_string(),
            span: *span,
        }),
        // Phase 10.5 stretch: `assert cond[, "msg"]` — statement-level only.
        Node::Assert { span, .. } => Err(TypeErrSpan {
            msg: "`assert` is a statement, not an expression".to_string(),
            span: *span,
        }),
        // `expr as type` — the cast carries the annotation forward.
        Node::As { expr, ty, span } => {
            let _ = infer_expr(expr, env)?;
            // Use the annotated type as the cast result; if it's a Named
            // alias we don't yet have it resolved (v1 limitation), fall
            // back to the source expression's type.
            if let Some(vt) = valuetype_from_ann(ty) {
                Ok((vt, *span))
            } else {
                let (vt, _) = infer_expr(expr, env)?;
                Ok((vt, *span))
            }
        }
        // `a && b`, `a || b` — boolean. Both sides must be inferable.
        Node::Logical {
            left, right, span, ..
        } => {
            let _ = infer_expr(left, env)?;
            let _ = infer_expr(right, env)?;
            Ok((ValueType::ScalarBool, *span))
        }
        // Bitwise: integer-typed; result type matches the left operand's type.
        Node::Bitwise {
            left, right, span, ..
        } => {
            let (lt, _) = infer_expr(left, env)?;
            let _ = infer_expr(right, env)?;
            Ok((lt, *span))
        }
        // Phase 10.6: struct literal expression. Type-check each field's
        // value sub-expression to surface errors there; the aggregate
        // type is reported as a Named alias of the struct's identifier
        // so downstream consumers (e.g. function-return-type checks)
        // can match it against `TypeAnn::Named(name)`. Full structural
        // resolution against StructDef arrives in a follow-up.
        Node::StructLit { name, fields, span } => {
            for f in fields {
                infer_expr(&f.value, env)?;
            }
            // Return ScalarI32 as a stable placeholder until structural
            // typing lands; the field-value checks already ran above so
            // the bulk of the contract is validated.
            let _ = name;
            Ok((ValueType::ScalarI32, *span))
        }
        // Phase 10.6: index access `xs[i]`. Type-check the receiver +
        // index expressions for early error surfacing; return ScalarI32
        // as a placeholder until element-type extraction lands.
        Node::IndexAccess {
            receiver,
            index,
            span,
        } => {
            infer_expr(receiver, env)?;
            infer_expr(index, env)?;
            Ok((ValueType::ScalarI32, *span))
        }
        // Phase 10.6: index assignment `xs[i] = v`. Statement-style;
        // returns ScalarI32 placeholder (the value's type) so downstream
        // typecheck passes don't break.
        Node::IndexAssign {
            receiver,
            index,
            value,
            span,
        } => {
            infer_expr(receiver, env)?;
            infer_expr(index, env)?;
            infer_expr(value, env)?;
            Ok((ValueType::ScalarI32, *span))
        }
        // Phase 10.6: field assignment `obj.field = v`. Same pattern.
        Node::FieldAssign {
            receiver,
            field: _,
            value,
            span,
        } => {
            infer_expr(receiver, env)?;
            infer_expr(value, env)?;
            Ok((ValueType::ScalarI32, *span))
        }
        // Phase 10.7: `match scrutinee { arms }`.
        //
        // The result type is unified across all arms. Pattern type
        // compatibility is checked conservatively (literal patterns vs.
        // scrutinee scalar class). Exhaustiveness is ENFORCED as a blocking
        // error for a match on a known enum type (see check_match_exhaustiveness
        // below); integer-tag / literal matches are not flagged.
        Node::Match {
            scrutinee,
            arms,
            span,
        } => {
            let (scrutinee_ty, _) = infer_expr(scrutinee, env)?;
            // Exhaustiveness is a PURELY STRUCTURAL property of the arm patterns
            // + guards, independent of arm-body / guard types. Check it FIRST so
            // a later guard/body inference error can never abort the arm loop
            // before it runs (which would silently drop the load-bearing
            // non-exhaustive diagnostic — drift #131). A guarded arm never counts
            // toward exhaustiveness; see `check_match_exhaustiveness`.
            check_match_exhaustiveness(arms)?;
            let mut result_ty: Option<ValueType> = None;
            for arm in arms {
                match &arm.pattern {
                    crate::ast::Pattern::Literal(crate::ast::Literal::Int(_))
                        if !matches!(
                            scrutinee_ty,
                            ValueType::ScalarI32 | ValueType::ScalarI64 | ValueType::ScalarBool
                        ) =>
                    {
                        return Err(TypeErrSpan {
                            msg: "integer literal pattern does not match scrutinee type"
                                .to_string(),
                            span: arm.span,
                        });
                    }
                    crate::ast::Pattern::Literal(crate::ast::Literal::Float(_))
                        if !matches!(scrutinee_ty, ValueType::ScalarF32 | ValueType::ScalarF64) =>
                    {
                        return Err(TypeErrSpan {
                            msg: "float literal pattern does not match scrutinee type".to_string(),
                            span: arm.span,
                        });
                    }
                    _ => {}
                }
                let mut arm_env = env.clone();
                if let crate::ast::Pattern::Ident(bind_name) = &arm.pattern {
                    arm_env.insert(bind_name.clone(), scrutinee_ty.clone());
                }
                if let crate::ast::Pattern::EnumVariant { path, args } = &arm.pattern {
                    // Finding 19: bind each payload sub-pattern at its DECLARED
                    // payload type (not the scrutinee/enum type), so an arm body
                    // using the payload is checked at the right type and a
                    // cross-arm class mismatch surfaces. Fall back to the scrutinee
                    // type when the enum/variant/payload is unresolvable (imported
                    // enum, Named payload, arity mismatch) — same defer discipline
                    // as `check_match_exhaustiveness`.
                    let payloads =
                        split_enum_variant_path(path).and_then(|(e, v)| variant_payload_of(&e, &v));
                    for (i, sub) in args.iter().enumerate() {
                        if let crate::ast::Pattern::Ident(sub_name) = sub {
                            let bind_ty = payloads
                                .as_ref()
                                .and_then(|p| p.get(i))
                                .and_then(valuetype_from_ann)
                                .unwrap_or_else(|| scrutinee_ty.clone());
                            arm_env.insert(sub_name.clone(), bind_ty);
                        }
                    }
                }
                // Pattern-guards W1.5a: a guard is a boolean expression evaluated
                // with the arm's pattern bindings in scope. Type-check it in
                // `arm_env` so an ill-typed / undefined-name guard surfaces; the
                // loose i64 ABI carries bools as i64, so no class assertion here.
                if let Some(guard) = &arm.guard {
                    infer_expr(guard, &arm_env)?;
                }
                match infer_expr(&arm.body, &arm_env) {
                    Ok((arm_ty, _)) => match &result_ty {
                        None => result_ty = Some(arm_ty),
                        Some(existing) => {
                            // Finding 19: compare arms by SCALAR CLASS, not exact
                            // ValueType. Sibling arms differing only in int width
                            // (i32 vs i64) are both i64-backed and legitimately
                            // unify; what is unsound is mixing classes — an int
                            // tag arm vs a float arm, now that payload sub-patterns
                            // bind at their declared type. A class mismatch is a
                            // hard type error (its own code, surfaced from fn
                            // bodies unlike the demoted exact-width mismatch).
                            if !same_scalar_class(existing, &arm_ty) {
                                return Err(TypeErrSpan {
                                    msg: format!(
                                        "match arm type class mismatch: expected {} but found {}",
                                        describe_value_type(existing),
                                        describe_value_type(&arm_ty)
                                    ),
                                    span: arm.span,
                                });
                            }
                        }
                    },
                    Err(e) => return Err(e),
                }
            }
            // Exhaustiveness was checked up-front (see the note after the
            // scrutinee inference above) so it can never be skipped by an arm
            // inference error.
            Ok((result_ty.unwrap_or(ValueType::ScalarI32), *span))
        }
        // Phase 10.7: `&expr` / `&mut expr` reference-taking.
        //
        // v1: type-check the inner expression; return `ScalarI32` as a
        // stable placeholder. Full `ValueType::Ref { mutable, inner }`
        // propagation is a follow-up that requires extending `ValueType`.
        Node::Ref { inner, span, .. } => {
            infer_expr(inner, env)?;
            Ok((ValueType::ScalarI32, *span))
        }
        // RFC 0005 Gap 1: while loop type. The body may change mutable
        // variables; the while expression itself is unit-typed (ScalarI32
        // placeholder) until Gap 1 lands full control-flow typing.
        #[cfg(feature = "std-surface")]
        Node::While { span, .. } => Ok((ValueType::ScalarI32, *span)),
        // RFC 0010 Phase A: `extern "C"` blocks are declarations; they do not
        // produce a typed value. Return ScalarI32 placeholder for compatibility.
        Node::ExternBlock { span, .. } => Ok((ValueType::ScalarI32, *span)),
        // RFC 0010 Phase J-A: `region { ... }` has the type of its last
        // expression. For now return ScalarI64 (the typical region result is
        // an i64 scalar extracted from heap-allocated data). Full type
        // inference for the region result is Phase J-B.
        #[cfg(feature = "std-surface")]
        Node::Region { body, span, .. } => {
            if let Some(last) = body.last() {
                infer_expr(last, env)
            } else {
                Ok((ValueType::ScalarI32, *span))
            }
        }
    }
}

fn infer_grad(
    loss: &Node,
    wrt: &[String],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    let (loss_ty, _) = infer_expr(loss, env)?;
    match loss_ty {
        ValueType::Tensor(ref t) => {
            if !t.shape.is_empty() {
                return Err(TypeErrSpan {
                    msg: "`grad` expects a scalar loss with shape ()".to_string(),
                    span: loss.span(),
                });
            }
        }
        _ => {
            return Err(TypeErrSpan {
                msg: "`grad` requires the loss to be a tensor expression".to_string(),
                span: loss.span(),
            });
        }
    };

    let mut entries = Vec::new();
    for name in wrt {
        match env.get(name) {
            Some(ValueType::Tensor(t)) => entries.push((name.clone(), t.clone())),
            Some(_) => {
                return Err(TypeErrSpan {
                    msg: format!("`{}` is not a tensor variable", name),
                    span,
                });
            }
            None => {
                return Err(TypeErrSpan {
                    msg: format!("unknown tensor `{}` in `wrt`", name),
                    span,
                });
            }
        }
    }

    Ok((ValueType::GradMap(entries), span))
}

// RFC 0005 Phase 1 + 1.5 — pure-MIND standard surface intrinsics.
//
// The primitives the std surface (`Vec`, `String`, `Map`, `io`)
// is allowed to bottom out into. All take and return `i64` only (no
// `Ptr` type — see RFC 0005 P0a; an address is a 64-bit integer).
// The pair (`__mind_load_i64`, `__mind_store_i64`) was added at Phase
// 1.5 to resolve P0c — without scalar load/store at address, `vec.push`
// cannot write the new value into the `__mind_alloc`-returned backing
// store. Lowered by the gated Phase-0 `Instr::Call` arm in
// `src/mlir/lowering.rs` to `func.call @__mind_*(%a..) : (i64..) -> i64`,
// with a matching `func.func private` declaration emitted once per
// distinct callee in sorted order. Default builds compile out the
// recogniser entirely.
#[cfg(feature = "std-surface")]
const STD_SURFACE_INTRINSICS: &[(&str, usize)] = &[
    ("__mind_alloc", 1),
    ("__mind_blas_dot_f32", 3),
    // RFC 0006 Track B (increment 1): native MLIR vector-dialect
    // `dot_f32`. Same i64 ABI / arity (3) as the Track A scalar bridge;
    // the difference is purely in lowering — the `Instr::Call` for this
    // name emits a `vector`-dialect reduction loop, not a `func.call` to
    // the runtime-support C shim. Track A's `__mind_blas_dot_f32` stays
    // registered and is the unchanged scalar/AVX2 fallback.
    ("__mind_blas_dot_f32_v", 3),
    // "int-dot" tier: native MLIR vector-dialect int16 dot product. Same i64
    // ABI / arity (3) as the other vector dots. Inputs are i16 row-major;
    // byte-identical to the scalar oracle `(i32) sum_k ((i32)a[k]*(i32)b[k])`
    // for ALL int16 inputs (i64-lane accumulate, no shift, no saturation, no
    // early narrow). The widen-multiply-accumulate loop is the AVX2 vpmaddwd
    // idiom at -march=x86-64-v3 — the fast deterministic int GEMM tier.
    ("__mind_blas_dot_i16_v", 3),
    ("__mind_blas_dot_l1_f32", 3),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect f32 L1
    // (sum-of-abs) reduction. Same i64 ABI / arity (3) as the Track A
    // scalar bridge; lowering interception emits an abs-diff + add
    // reduction loop. Track A's `__mind_blas_dot_l1_f32` is unchanged.
    ("__mind_blas_dot_l1_f32_v", 3),
    ("__mind_blas_dot_l1_q16", 3),
    // RFC 0006 Track B (increment 3): native MLIR vector-dialect Q16.16 L1
    // (Manhattan, sum-of-abs) reduction. Byte-identical to the Track A
    // scalar oracle `__mind_blas_dot_l1_q16` at every length (task #57
    // cross-arch bit-identity gate); closes the Q16.16 vector-path metric
    // parity deferred in increment 2. Track A's `__mind_blas_dot_l1_q16`
    // is unchanged.
    ("__mind_blas_dot_l1_q16_v", 3),
    ("__mind_blas_dot_linf_f32", 3),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect f32 L∞
    // (max-of-abs) reduction. Track A's `__mind_blas_dot_linf_f32` is
    // unchanged.
    ("__mind_blas_dot_linf_f32_v", 3),
    ("__mind_blas_dot_q16", 3),
    // RFC 0006 Track B (increment 2): native MLIR vector-dialect Q16.16
    // dot product. Byte-identical to the Track A scalar oracle
    // `__mind_blas_dot_q16` at every length (task #57 cross-arch
    // bit-identity gate). Track A's `__mind_blas_dot_q16` is unchanged.
    ("__mind_blas_dot_q16_v", 3),
    ("__mind_blas_matmul_rmajor_f32", 5),
    // RFC 0006 Track B (increment 3b): native MLIR vector-dialect row-major
    // f32 matmul.  Outer scf.for over rows, inner vectorised dot_f32_v
    // (8-lane FMA + scalar tail) inlined per row, stores to caller-allocated
    // y buffer, returns 0.  Same arity (5) and i64 ABI as Track A.
    ("__mind_blas_matmul_rmajor_f32_v", 5),
    // "int-dot" tier: native MLIR vector-dialect row-major int16 matmul.
    // Outer scf.for over rows, inner int16 dot reduction from emit_vec_dot_i16
    // (sext i16->i64, i64-lane accumulate, vector.reduction <add>, scalar
    // tail, trunc) inlined per row, stores i32 to the caller-allocated y
    // buffer, returns 0. Same arity (5) and i64 ABI as the f32/q16 matmuls.
    // Byte-identical to the scalar oracle applied per row, for all int16
    // inputs. Track B vector-dialect only — no Track A i16 matmul extern.
    ("__mind_blas_matmul_rmajor_i16_v", 5),
    // RFC 0006 Track B (increment 4): native MLIR vector-dialect row-major
    // Q16.16 matmul.  Outer scf.for over rows, inner Q16.16 dot reduction
    // from emit_vec_dot_q16 (widen i32→i64, >> 16, i64-lane accumulate,
    // vector.reduction <add>, scalar tail, trunc+extsi) inlined per row,
    // stores i32 to caller-allocated y buffer, returns 0.  Byte-identical
    // to the scalar oracle __mind_blas_dot_q16 applied per row (cross-arch
    // bit-identity gate, task #57).  Track B vector-dialect only — there is
    // no Track A q16 matmul extern; the per-row oracle is __mind_blas_dot_q16.
    ("__mind_blas_matmul_rmajor_q16_v", 5),
    // "det.igemm" tier: fused int8 GEMM. A is M×K row-major int8 (1 byte), B
    // is K×N row-major int8, C is M×N row-major INT32 caller-allocated; arity 6
    // (a, b, c, m, k, n), i64 ABI, returns 0. Same BLIS-blocked register-tiled
    // kernel as the Q16 path with i8→i32 sign-extension during the pack and NO
    // >> 16 shift (int8 is integer, not fixed-point). The C-tile accumulates
    // i64; the i64→i32 truncation happens once at the store — byte-identical to
    // the per-element scalar int32 oracle (i32) Σ_k (i32)A[i,k]*(i32)B[k,j] for
    // all shapes. The same MLIR lowers to vpmaddwd (AVX2) / SDOT (aarch64),
    // both yielding the identical exact int32 sum.
    ("__mind_blas_matmul_mm_i8_v", 6),
    // Multithreaded fused int8 GEMM. Same ABI (arity 6: a, b, c, m, k, n; i64;
    // returns 0) and byte-for-byte output as __mind_blas_matmul_mm_i8_v,
    // parallelised over contiguous owner-computes M-row bands with raw POSIX
    // threads. Output is independent of the thread count (no cross-thread
    // reduction), so cross-substrate bit-identity holds.
    ("__mind_blas_matmul_mm_i8_mt_v", 6),
    // RFC 0006 Track B: fused outer-product Q16.16 GEMM. A is M×K row-major,
    // B is K×N row-major (un-transposed), C is M×N row-major caller-allocated;
    // arity 6 (a, b, c, m, k, n), i64 ABI, returns 0. Register-tiled
    // outer-product microkernel (no horizontal reduction) — byte-identical to
    // the per-element scalar oracle Σ_k (A[i,k]*B[k,j])>>16 for all shapes.
    ("__mind_blas_matmul_mm_q16_v", 6),
    // Multithreaded fused outer-product Q16.16 GEMM. Same ABI (arity 6:
    // a, b, c, m, k, n; i64; returns 0) and byte-for-byte output as
    // __mind_blas_matmul_mm_q16_v, parallelised over contiguous owner-computes
    // M-row bands with raw POSIX threads. Output is independent of the thread
    // count (no cross-thread reduction), so cross-substrate bit-identity holds.
    ("__mind_blas_matmul_mm_q16_mt_v", 6),
    ("__mind_free", 1),
    ("__mind_load_i64", 1),
    // RFC 0005 Phase 1.6 (task #306) — single-byte load/store. The
    // (`__mind_store_i64(base + i, b)` writes one byte / `__mind_load_i64(base + i) & 255`
    // reads one byte) convention used by `std.string` / `std.sha256` / `std.toml`
    // / `std.tui` clobbers 7 bytes per store and can read past the buffer. The
    // store form is currently masked by a 7-byte backing-store pad in runtime-support
    // (commit `cc5a513`), but the garbage past `len` is a cross-substrate
    // bit-identity landmine (NEON / RVV may not have the same pad). These two
    // intrinsics provide a proper one-byte ABI; `load_i8` zero-extends to i64 so
    // call sites preserve the `& 255` mask semantics during migration.
    ("__mind_load_i8", 1),
    ("__mind_load_i32", 1),
    ("__mind_load_i16", 1),
    ("__mind_read", 4),
    ("__mind_realloc", 2),
    ("__mind_store_i64", 2),
    ("__mind_store_i8", 2),
    ("__mind_store_i32", 2),
    ("__mind_store_i16", 2),
    ("__mind_write", 4),
    // `c.byte()` — the byte (low 8 bits) of a char/int receiver. The method-call
    // type-check validates it as a 1-arg call `byte(recv)`; lowering desugars it
    // to `recv & 0xFF` (see the MethodCall arm in eval/lower.rs). mind-flow's
    // lexer relies on it (`'0'.byte()`).
    ("byte", 1),
];

#[cfg(feature = "std-surface")]
fn std_surface_intrinsic_arity(name: &str) -> Option<usize> {
    STD_SURFACE_INTRINSICS
        .iter()
        .find_map(|(n, arity)| (*n == name).then_some(*arity))
}

#[cfg(feature = "std-surface")]
fn check_std_surface_intrinsic(
    callee: &str,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    let arity = std_surface_intrinsic_arity(callee).ok_or_else(|| TypeErrSpan {
        msg: format!("unsupported call to `{callee}`"),
        span,
    })?;
    if args.len() != arity {
        return Err(TypeErrSpan {
            msg: format!(
                "intrinsic `{callee}` expects {arity} i64 argument(s) (i64 ABI — RFC 0005); got {}",
                args.len()
            ),
            span,
        });
    }
    for (i, a) in args.iter().enumerate() {
        let (ty, _) = infer_expr(a, env)?;
        // Integer literals come in as `ScalarI32` (line 562) but lower
        // to `ConstI64` in IR — accept both; reject anything else with
        // a clear i64-ABI message that points to RFC 0005 phase 2+.
        if !matches!(ty, ValueType::ScalarI32 | ValueType::ScalarI64) {
            return Err(TypeErrSpan {
                msg: format!(
                    "argument {i} to intrinsic `{callee}` must be i64 (got {}); the std-surface intrinsics use the i64 ABI — aggregate types are RFC 0005 phase 2+",
                    describe_value_type(&ty)
                ),
                span: a.span(),
            });
        }
    }
    Ok((ValueType::ScalarI64, span))
}

/// Bug #38 — true when `ann` is a fixed-size buffer type `bytes[N]`
/// (parser renders it `Named("bytes[N]")`), distinct from the growable
/// `Named("bytes")` vec record. The `[` suffix is the sole discriminator.
fn typeann_is_fixed_bytes(ann: &TypeAnn) -> bool {
    matches!(ann, TypeAnn::Named(n) if n.starts_with("bytes["))
}

/// Bug #38 — recognise an argument EXPRESSION that evaluates to a fixed-size
/// `bytes[N]` buffer handle. NARROW by construction: returns `true` only for
/// the two value forms that produce such a handle, and `None`-equivalent
/// (`false`) for every other expression, so Phase-A loose typing is fully
/// preserved and no valid program can be rejected.
///
///   1. `bytes[N].zero()` — parses as `MethodCall { receiver:
///      IndexAccess { receiver: Ident("bytes"), .. }, method: "zero" }`,
///      the same shape the lowerer matches at lower.rs (`__mind_calloc`).
///   2. A call to a fn whose declared return type is `Named("bytes[...]")`
///      (e.g. `std.sha256.hash(...) -> bytes[32]`), looked up via the
///      cross-module signature table.
fn arg_is_fixed_bytes_buffer(arg: &Node) -> bool {
    match arg {
        Node::MethodCall {
            receiver, method, ..
        } if method == "zero" => matches!(
            receiver.as_ref(),
            Node::IndexAccess { receiver: base, .. }
                if matches!(base.as_ref(), Node::Lit(Literal::Ident(n), _) if n == "bytes")
        ),
        // A bare identifier bound to a fixed `bytes[N]` buffer (`let buf:
        // bytes[32] = …; take_bytes(buf)`) is the SAME i64 handle as the value
        // forms above — binding it is not a copy into a growable vec. Resolve it
        // against the fn-body's `bytes[N]` local set so the #38 miscompile is
        // refused through an alias too, not just when written directly.
        Node::Lit(Literal::Ident(n), _) => fixed_bytes_local(n),
        #[cfg(feature = "cross-module-imports")]
        Node::Call { callee, .. } => cm_lookup_fn(callee)
            .and_then(|sig| sig.ret_type)
            .as_ref()
            .is_some_and(typeann_is_fixed_bytes),
        _ => false,
    }
}

/// Bug #38 — given a callee's declared parameter TypeAnns and the call's
/// argument expressions, return the first `(index, span)` where a growable
/// `bytes` parameter receives a fixed-size `bytes[N]` buffer handle. This is
/// the only arg/param type comparison performed under the loose i64 ABI, and
/// it fires ONLY on this exact mismatch — see `arg_is_fixed_bytes_buffer`.
fn fixed_bytes_into_vec_violation(
    param_types: &[TypeAnn],
    args: &[Node],
) -> Option<(usize, AstSpan)> {
    for (i, (param, arg)) in param_types.iter().zip(args.iter()).enumerate() {
        if matches!(param, TypeAnn::Named(n) if n == "bytes") && arg_is_fixed_bytes_buffer(arg) {
            return Some((i, arg.span()));
        }
    }
    None
}

fn infer_call(
    callee: &str,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    match callee {
        "tensor.zeros" | "tensor.ones" => {
            if args.len() != 2 {
                return Err(TypeErrSpan {
                    msg: format!("`{callee}` expects (dtype, shape) arguments"),
                    span,
                });
            }
            let dtype = infer_dtype_arg(&args[0])?;
            let shape = infer_shape_arg(&args[1])?;
            Ok((ValueType::Tensor(TensorType::new(dtype, shape)), span))
        }
        "tensor.shape" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.shape` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.shape` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.sum" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.sum` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(tensor) => Ok((
                    ValueType::Tensor(TensorType::new(tensor.dtype, Vec::new())),
                    span,
                )),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sum` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.dtype" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.dtype` expects a single tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.dtype` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        "tensor.print" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.print` expects a single argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            if matches!(arg_ty, ValueType::Tensor(_) | ValueType::ScalarI32) {
                Ok((arg_ty, span))
            } else {
                Err(TypeErrSpan {
                    msg: "`tensor.print` requires a tensor or scalar argument".to_string(),
                    span,
                })
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.materialize" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.materialize` expects a tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty.clone() {
                ValueType::Tensor(_) => Ok((arg_ty, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.materialize` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.is_materialized" => {
            if args.len() != 1 {
                return Err(TypeErrSpan {
                    msg: "`tensor.is_materialized` expects a tensor argument".to_string(),
                    span,
                });
            }
            let (arg_ty, _) = infer_expr(&args[0], env)?;
            match arg_ty {
                ValueType::Tensor(_) => Ok((ValueType::ScalarI32, span)),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.is_materialized` requires a tensor argument".to_string(),
                    span,
                }),
            }
        }
        #[cfg(feature = "cpu-buffers")]
        "tensor.sample" => {
            if args.len() != 2 {
                return Err(TypeErrSpan {
                    msg: "`tensor.sample` expects (tensor, count) arguments".to_string(),
                    span,
                });
            }
            let (tensor_ty, _) = infer_expr(&args[0], env)?;
            let (count_ty, _) = infer_expr(&args[1], env)?;
            match (tensor_ty, count_ty) {
                (ValueType::Tensor(tensor), ValueType::ScalarI32) => Ok((
                    ValueType::Tensor(TensorType::new(
                        tensor.dtype,
                        vec![ShapeDim::Sym("_sample")],
                    )),
                    span,
                )),
                (ValueType::Tensor(_), _) => Err(TypeErrSpan {
                    msg: "`tensor.sample` requires the second argument to be an integer"
                        .to_string(),
                    span,
                }),
                _ => Err(TypeErrSpan {
                    msg: "`tensor.sample` requires a tensor and an integer argument".to_string(),
                    span,
                }),
            }
        }
        _ => {
            // Cross-module imports (RFC 0005 Phase 2 ergonomics): a
            // `use std.vec` resolver-injects the file's `pub fn`/struct
            // names into `env` as `ScalarI32` placeholders.  Calls to
            // those names need to type-check as i64-result intrinsic-
            // style calls — there's no per-fn-signature lookup yet
            // (that's Phase B), so we accept the call and return i64
            // when the callee is present in env via the resolver path.
            //
            // The std-surface intrinsic check still runs *after* this
            // for `__mind_*` callees that aren't in env (the std files
            // themselves use them directly).  Default build (neither
            // feature) keeps the byte-identical "unsupported call"
            // error — moat held.
            #[cfg(feature = "cross-module-imports")]
            if env.get(callee).is_some() {
                // RFC 0005 Phase B — try the signature-aware path
                // first.  If the project table has a typed
                // declaration for `callee`, validate arity + per-arg
                // types against it and return the declared return
                // type.  If the lookup fails (the table has the name
                // but no signature, e.g. an `export { ... }` block
                // surface, or the imported `pub fn` was unparsed),
                // fall back to Phase-A loose ScalarI64 behavior.
                if let Some(sig) = cm_lookup_fn(callee) {
                    return check_imported_fn_call(&sig, args, span, env);
                }
                // RFC 0005 Phase B — a same-file module-level fn (not an
                // imported one) is checked against the intra-module side-table
                // for arity. Cross-module imported signatures took precedence
                // above; this only fires for callees the import table doesn't
                // resolve.
                if let Some(sig) = intra_lookup_fn(callee) {
                    return check_intra_fn_call(callee, &sig, args, span, env);
                }
                // No typed signature for this same-file callee: accept the call
                // at the loose i64 ABI WITHOUT re-inferring the argument
                // expressions. A re-walk via `infer_expr` false-positives on
                // valid value forms the outer type-check pass already accepts —
                // notably enum-variant constructors in argument position
                // (`f(Mode::On)`), which are not value-env identifiers and would
                // wrongly raise E2002. Arity is already enforced via the typed
                // paths above. (Regression fix: this re-walk reddened CI's
                // `enum_tag_match_writes_field_in_each_arm`.)
                return Ok((ValueType::ScalarI64, span));
            }
            // Intra-module function call: if the callee is a function defined
            // in the same module, the pre-scan in check_module_types_in_file
            // registers it in env as ScalarI64. Validate the call's arity
            // against the captured signature before accepting it (arity is the
            // one property always knowable under the loose i64 ABI).
            if env.get(callee).is_some() {
                if let Some(sig) = intra_lookup_fn(callee) {
                    return check_intra_fn_call(callee, &sig, args, span, env);
                }
                // No typed signature for this same-file callee: accept the call
                // at the loose i64 ABI WITHOUT re-inferring the argument
                // expressions. A re-walk via `infer_expr` false-positives on
                // valid value forms the outer type-check pass already accepts —
                // notably enum-variant constructors in argument position
                // (`f(Mode::On)`), which are not value-env identifiers and would
                // wrongly raise E2002. Arity is already enforced via the typed
                // paths above. (Regression fix: this re-walk reddened CI's
                // `enum_tag_match_writes_field_in_each_arm`.)
                return Ok((ValueType::ScalarI64, span));
            }
            #[cfg(feature = "std-surface")]
            {
                check_std_surface_intrinsic(callee, args, span, env)
            }
            #[cfg(not(feature = "std-surface"))]
            {
                let _ = (args, env); // silence warnings on default build
                Err(TypeErrSpan {
                    msg: format!("unsupported call to `{callee}`"),
                    span,
                })
            }
        }
    }
}

fn infer_dtype_arg(node: &Node) -> Result<DType, TypeErrSpan> {
    match node {
        Node::Lit(Literal::Ident(name), span) => name.parse().map_err(|_| TypeErrSpan {
            msg: format!("unknown dtype `{name}`"),
            span: *span,
        }),
        _ => Err(TypeErrSpan {
            msg: "expected dtype identifier".to_string(),
            span: node.span(),
        }),
    }
}

fn infer_shape_arg(node: &Node) -> Result<Vec<ShapeDim>, TypeErrSpan> {
    match node {
        Node::Tuple { .. } | Node::Paren(..) | Node::Lit(..) => infer_shape_node(node),
        _ => infer_shape_node(node),
    }
}

fn infer_shape_node(node: &Node) -> Result<Vec<ShapeDim>, TypeErrSpan> {
    match node {
        Node::Tuple { elements, .. } => {
            let mut dims = Vec::new();
            for el in elements {
                dims.extend(infer_shape_node(el)?);
            }
            Ok(dims)
        }
        Node::Paren(inner, _) => infer_shape_node(inner),
        Node::Lit(Literal::Int(n), span) => {
            if *n < 0 {
                Err(TypeErrSpan {
                    msg: "shape dimensions must be non-negative".to_string(),
                    span: *span,
                })
            } else {
                Ok(vec![ShapeDim::Known(*n as usize)])
            }
        }
        Node::Lit(Literal::Ident(name), _span) => Ok(vec![ShapeDim::Sym(leak_symbol(name))]),
        _ => Err(TypeErrSpan {
            msg: "unsupported shape literal".to_string(),
            span: node.span(),
        }),
    }
}

fn leak_symbol(name: &str) -> &'static str {
    crate::types::intern::intern_str(name)
}

/// A fresh symbolic shape-dim name, *content-addressed* by the source span of
/// the callsite that produced it.
///
/// Determinism (CRIT): a process-global `AtomicUsize` counter made the emitted
/// `ShapeDim::Sym` name depend on how many prior compilations ran in the same
/// process — two modules type-checked in one process drew different names by
/// order, breaking run-to-run / in-process reproducibility for any symbolic
/// shape that reaches MLIR. Deriving the name from the callsite's `(start,end)`
/// byte span makes it a pure function of the source: the SAME source always
/// yields the SAME name regardless of prior compiles, and two DISTINCT
/// callsites have distinct spans so no two symbolic dims collide. The `prefix`
/// already encodes the role (`_slice`, `_slice_stride`, `_conv_<axis>`), so a
/// single callsite that yields one symbol per axis stays collision-free.
fn fresh_symbol(prefix: &str, span: AstSpan) -> &'static str {
    crate::types::intern::intern_str(&format!("{prefix}_{}_{}", span.start(), span.end()))
}

fn dtype_from_str(s: &str) -> Option<DType> {
    match s {
        "i32" => Some(DType::I32),
        "i64" => Some(DType::I64),
        "f32" => Some(DType::F32),
        "f64" => Some(DType::F64),
        "bf16" => Some(DType::BF16),
        "f16" => Some(DType::F16),
        // RFC 0012 §3.2 — Q16.16 first-class dtype keyword.
        // Storage: i32 (same as DType::I32 at runtime); the dtype tag is the
        // compile-time signal for byte-identity semantics (task #57).
        "q16" => Some(DType::Q16),
        _ => None,
    }
}

fn shape_from_dims(dims: &[String]) -> Vec<ShapeDim> {
    dims.iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                ShapeDim::Known(n)
            } else {
                ShapeDim::Sym(Box::leak(d.clone().into_boxed_str()))
            }
        })
        .collect()
}

fn valuetype_from_ann(ann: &crate::ast::TypeAnn) -> Option<ValueType> {
    match ann {
        crate::ast::TypeAnn::ScalarI32 => Some(ValueType::ScalarI32),
        crate::ast::TypeAnn::ScalarI64 => Some(ValueType::ScalarI64),
        crate::ast::TypeAnn::ScalarF32 => Some(ValueType::ScalarF32),
        crate::ast::TypeAnn::ScalarF64 => Some(ValueType::ScalarF64),
        crate::ast::TypeAnn::ScalarBool => Some(ValueType::ScalarBool),
        crate::ast::TypeAnn::Tensor { dtype, dims }
        | crate::ast::TypeAnn::DiffTensor { dtype, dims } => {
            let dt = dtype_from_str(dtype)?;
            let shape = shape_from_dims(dims);
            Some(ValueType::Tensor(TensorType::new(dt, shape)))
        }
        // Phase 10.5 Tier-1: user-defined type names resolve via the env at
        // typecheck time; structural lookup returns None and lets the caller
        // perform alias resolution.
        crate::ast::TypeAnn::Named(_) => None,
        // Phase 10.5 Tier-2: u32 maps to i32 in v1 (no separate unsigned
        // ValueType yet); sign correctness is enforced at use sites.
        crate::ast::TypeAnn::ScalarU32 => Some(ValueType::ScalarI32),
        // Phase 10.6: borrowed slice, fixed-size array, single-value
        // reference, and generic-application types are aggregates that
        // don't have a direct ValueType today. Return None and let the
        // caller decide; most call sites use these in fn signatures,
        // where shape validation runs against the underlying element
        // type at call sites.
        crate::ast::TypeAnn::Slice { .. }
        | crate::ast::TypeAnn::Array { .. }
        | crate::ast::TypeAnn::Ref { .. }
        | crate::ast::TypeAnn::Generic { .. }
        | crate::ast::TypeAnn::Tuple { .. } => None,
        // SparseTensor: runtime resolves layout; return None so callers
        // fall through to the runtime resolver path (same pattern as Slice).
        crate::ast::TypeAnn::SparseTensor { .. } => None,
        // RFC 0010 Phase A: raw pointer types are opaque handles in the MIND
        // type system — they don't map to a ValueType. Return None to let
        // callers treat them as unresolved (typically, extern fn signatures).
        crate::ast::TypeAnn::RawPtr { .. } => None,
        // RFC 0010 Phase B: callback function pointers are opaque handles.
        crate::ast::TypeAnn::FnPtr { .. } => None,
    }
}

/// Walk statements; extend env on let/assign; return pretty diags for any errors.
pub fn check_module_types(module: &Module, src: &str, env: &TypeEnv) -> Vec<Pretty> {
    check_module_types_in_file(module, src, None, env)
}

// ── Intra-module call signatures (RFC 0005 Phase B) ───────────────────
//
// A side-table of every module-level fn's call signature, threaded via a
// thread-local set by `check_module_types_in_file` (for the duration of the
// per-file check) rather than by a new parameter on the shared
// `check_module_types*` signature — same moat-preserving discipline as the
// cross-module table below. Built for ALL fns before ANY call is checked, so
// forward references and recursion are resolved correctly. Cleared at the end
// of the check (no leakage across files / threads).
thread_local! {
    static INTRA_FN_SIGS: std::cell::RefCell<Option<IntraFnSigs>> =
        const { std::cell::RefCell::new(None) };
}

/// Look up an intra-module fn's captured signature by name. Returns `None`
/// when the side-table isn't populated (no module context) or the name isn't
/// a module-level fn (e.g. an imported symbol or an env placeholder).
fn intra_lookup_fn(name: &str) -> Option<IntraFnSig> {
    INTRA_FN_SIGS.with(|cell| {
        cell.borrow()
            .as_ref()
            .and_then(|sigs| sigs.get(name).cloned())
    })
}

// ── Enum-variant registry (match exhaustiveness) ──────────────────────
//
// Maps every module-level `enum` name to its ordered set of variant names so
// the `Node::Match` arm in `infer_expr` can check exhaustiveness WITHOUT a
// scrutinee enum-type (the loose model collapses an enum value to a scalar tag,
// losing its identity). Exhaustiveness is driven entirely by the arm patterns:
// the enum is identified from the `EnumVariant` arms' `Name::Variant` paths.
// Threaded via the same set-on-entry / clear-on-exit thread-local discipline as
// the fn side-table. Built once per `check_module_types_in_file` (and merged
// into the parent during the FnDef-body recursion, so a body's match still sees
// the module's enums). Maps `enum name -> Vec<variant name>` (Vec for stable,
// deterministic ordering of any "missing variants" list).
type EnumVariantTable = HashMap<String, Vec<String>>;

/// Finding-19 soundness: each variant's declared PAYLOAD types, keyed by the
/// bare `"Enum::Variant"` path (matching `split_enum_variant_path` output). Lets
/// a match arm bind a payload sub-pattern at its declared type instead of the
/// scrutinee/enum type, so `E::A(x) => x` checks at the payload type.
type EnumPayloadTable = HashMap<String, Vec<crate::ast::TypeAnn>>;

thread_local! {
    static ENUM_VARIANTS: std::cell::RefCell<Option<EnumVariantTable>> =
        const { std::cell::RefCell::new(None) };
    static ENUM_PAYLOADS: std::cell::RefCell<Option<EnumPayloadTable>> =
        const { std::cell::RefCell::new(None) };
}

/// Look up a variant's declared payload `TypeAnn`s by bare `"Enum::Variant"`.
fn variant_payload_of(enum_name: &str, variant: &str) -> Option<Vec<crate::ast::TypeAnn>> {
    let key = format!("{enum_name}::{variant}");
    ENUM_PAYLOADS.with(|cell| cell.borrow().as_ref().and_then(|t| t.get(&key).cloned()))
}

/// Build the payload registry from a module's `Node::EnumDef` items.
fn build_enum_payloads(module: &Module) -> EnumPayloadTable {
    let mut table = EnumPayloadTable::new();
    for item in &module.items {
        if let Node::EnumDef { name, variants, .. } = item {
            for v in variants {
                table.insert(format!("{name}::{}", v.name), v.payload.clone());
            }
        }
    }
    table
}

/// Whether two scalar `ValueType`s belong to the same broad CLASS (int family
/// {i32,i64,bool} vs float family {f32,f64} vs tensor vs gradmap). Match arms
/// must agree on class; exact width is deliberately NOT required so an i64
/// payload value and an i32-typed literal `0` in sibling arms stay compatible.
fn same_scalar_class(a: &ValueType, b: &ValueType) -> bool {
    fn class(v: &ValueType) -> u8 {
        match v {
            ValueType::ScalarI32 | ValueType::ScalarI64 | ValueType::ScalarBool => 0,
            ValueType::ScalarF32 | ValueType::ScalarF64 => 1,
            ValueType::Tensor(_) => 2,
            ValueType::GradMap(_) => 3,
        }
    }
    class(a) == class(b)
}

/// True for a float-family scalar (`f32`/`f64`). A float `ValueType` is never a
/// loose-ABI default (the default fallback is always `ScalarI64`), so a value
/// that infers to float was genuinely inferred as float — never guessed.
fn is_float_scalar(v: &ValueType) -> bool {
    matches!(v, ValueType::ScalarF32 | ValueType::ScalarF64)
}

/// True for an integer-family scalar (`i32`/`i64`/`bool`), mirroring the int
/// class in `same_scalar_class`.
fn is_int_scalar(v: &ValueType) -> bool {
    matches!(
        v,
        ValueType::ScalarI32 | ValueType::ScalarI64 | ValueType::ScalarBool
    )
}

/// Whether a node in `if`/`while` condition position is a boolean-intent
/// expression whose runtime value is a truth value regardless of its operand
/// class. `infer_expr` reports `f64 > f64` as `ScalarF64` (it returns the
/// operand class, not `bool`), so these forms must be excluded from the
/// float-condition check (E2011) or a valid `if a > b` over floats would be
/// wrongly rejected.
fn cond_is_boolean_intent(node: &Node) -> bool {
    match node {
        Node::Binary { op, .. } => matches!(
            op,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        ),
        Node::Logical { .. } | Node::Not { .. } => true,
        _ => false,
    }
}

/// Early check-phase diagnostics that need the enclosing function's declared
/// return type in scope (E2010) or that inspect condition position (E2011).
/// Walks the function body's statement positions, recursing through control
/// flow but stopping at nested `FnDef`s (which carry their own return type).
/// Uses `env` = params + module symbols (NO body-local bindings): a value that
/// can't be resolved yields `Err` from `infer_expr` and is skipped, so this
/// only ever fires on confidently-typed expressions — no false positives on
/// locals. Additive: it only pushes E2010/E2011 on the sound conditions
/// documented at each code constant.
fn check_return_and_cond_types(
    stmts: &[Node],
    ret_ty: Option<&ValueType>,
    env: &TypeEnv,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    for stmt in stmts {
        check_return_and_cond_node(stmt, ret_ty, env, src, file, errs);
    }
}

fn check_cond_type(
    cond: &Node,
    env: &TypeEnv,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    if cond_is_boolean_intent(cond) {
        return;
    }
    if let Ok((vt, _)) = infer_expr(cond, env) {
        if is_float_scalar(&vt) {
            errs.push(diag_from_span(
                src,
                file,
                format!(
                    "condition must be a boolean or integer expression, but this is {}",
                    describe_value_type(&vt)
                ),
                cond.span(),
                COND_TYPE_MISMATCH_CODE,
            ));
        }
    }
}

fn check_return_and_cond_node(
    node: &Node,
    ret_ty: Option<&ValueType>,
    env: &TypeEnv,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        Node::Return { value: Some(v), .. } => {
            if let Some(rt) = ret_ty {
                if is_int_scalar(rt) {
                    if let Ok((vt, _)) = infer_expr(v, env) {
                        if is_float_scalar(&vt) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "return type mismatch: function returns {} but this returns {}",
                                    describe_value_type(rt),
                                    describe_value_type(&vt)
                                ),
                                v.span(),
                                RETURN_TYPE_MISMATCH_CODE,
                            ));
                        }
                    }
                }
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            check_cond_type(cond, env, src, file, errs);
            check_return_and_cond_types(then_branch, ret_ty, env, src, file, errs);
            if let Some(eb) = else_branch {
                check_return_and_cond_types(eb, ret_ty, env, src, file, errs);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            check_cond_type(cond, env, src, file, errs);
            check_return_and_cond_types(body, ret_ty, env, src, file, errs);
        }
        Node::For { body, .. } | Node::ForEach { body, .. } => {
            check_return_and_cond_types(body, ret_ty, env, src, file, errs);
        }
        Node::Block { stmts, .. } => {
            check_return_and_cond_types(stmts, ret_ty, env, src, file, errs);
        }
        // Nested function definitions carry their own return type; the outer
        // `ret_ty` does not apply, so do not descend into them here.
        _ => {}
    }
}

// ── Confidence-gated scalar-class checks (RFC 0011 — no implicit int↔float) ──
//
// A syntactic class checker that runs alongside the E2010/E2011 pass. It does
// NOT use `infer_expr` (whose `ScalarI64`/`ScalarI32` results are polluted by
// loose defaults: fn names, field access, loop vars). Instead it derives a
// scalar CLASS *only* from literals, declared annotations, explicit `as`
// targets, and intra-module callees' declared return annotations, so every
// diagnostic fires on a fact — never a guess. Under-coverage (fields, methods,
// untyped calls) is by design; zero over-coverage is the load-bearing property.

/// Broad scalar CLASS: the two families that must never mix without an explicit
/// `as` (RFC 0011). Coarser than `ValueType` and derived only from annotations
/// and literals, so a `Some(_)` is always provable, never a loose default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarClass {
    Int,
    Float,
}

/// The scalar CLASS of a type annotation, or `None` for non-scalar / unknown
/// annotations. `bool` and every integer width map to `Int`; `f32`/`f64` map to
/// `Float`. Name-exact on the `Named` fallback (the parser emits `Named("u64")`,
/// `Named("u8")`, `Named("i8")`, … for the widths without a dedicated variant).
fn scalar_class_of_ann(ty: &TypeAnn) -> Option<ScalarClass> {
    match ty {
        TypeAnn::ScalarI32 | TypeAnn::ScalarI64 | TypeAnn::ScalarBool | TypeAnn::ScalarU32 => {
            Some(ScalarClass::Int)
        }
        TypeAnn::ScalarF32 | TypeAnn::ScalarF64 => Some(ScalarClass::Float),
        TypeAnn::Named(n) => match n.as_str() {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "isize" | "usize"
            | "bool" => Some(ScalarClass::Int),
            "f32" | "f64" => Some(ScalarClass::Float),
            _ => None,
        },
        _ => None,
    }
}

/// Whether an annotation names `bool` (dedicated variant or `Named("bool")`).
fn is_bool_ann(ty: &TypeAnn) -> bool {
    matches!(ty, TypeAnn::ScalarBool) || matches!(ty, TypeAnn::Named(n) if n == "bool")
}

/// Declared-scalar bindings in scope for the class checker. `classes` maps a
/// binding name to its annotation-derived `ScalarClass`, seeded with the
/// enclosing fn's params and grown at each annotated `let`. Names are only ever
/// added from annotations — never the loose i64 default — so membership is
/// always a fact. (Point-lookup only; nothing iterates it into output.)
#[derive(Debug, Clone, Default)]
struct ClassCtx {
    classes: HashMap<String, ScalarClass>,
}

/// Confidence-gated scalar class of an expression: `Some` ONLY when provable
/// from a literal, a declared binding, an explicit `as` target, or an
/// intra-module callee's declared return annotation. Returns `None` for every
/// loose-typed construct (field/method/index access, untyped call, match,
/// bitwise, comparison, …), so the checks it feeds can fire only on
/// annotation/literal-derived mismatches — the exact programs that die at
/// `mlir-opt` today.
fn confident_scalar_class(node: &Node, ctx: &ClassCtx) -> Option<ScalarClass> {
    match node {
        Node::Lit(Literal::Int(_), _) => Some(ScalarClass::Int),
        Node::Lit(Literal::Float(_), _) => Some(ScalarClass::Float),
        Node::Lit(Literal::Ident(name), _) => ctx.classes.get(name).copied(),
        Node::As { ty, .. } => scalar_class_of_ann(ty),
        Node::Paren(inner, _) => confident_scalar_class(inner, ctx),
        Node::Neg { operand, .. } => confident_scalar_class(operand, ctx),
        Node::Binary {
            op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
            left,
            right,
            ..
        } => {
            let l = confident_scalar_class(left, ctx)?;
            let r = confident_scalar_class(right, ctx)?;
            (l == r).then_some(l)
        }
        Node::Call { callee, .. } => intra_lookup_fn(callee)
            .and_then(|sig| sig.ret_type.as_ref().and_then(scalar_class_of_ann)),
        _ => None,
    }
}

/// Entry point: run the confidence-gated scalar-class checks over a fn body.
/// `ret_class` is the enclosing fn's declared-return class (drives the E2010
/// new direction); `ctx` is pre-seeded with the fn's scalar params.
fn check_scalar_classes(
    stmts: &[Node],
    ret_class: Option<ScalarClass>,
    ctx: &mut ClassCtx,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    for stmt in stmts {
        check_scalar_class_stmt(stmt, ret_class, ctx, src, file, errs);
    }
}

fn check_scalar_class_stmt(
    node: &Node,
    ret_class: Option<ScalarClass>,
    ctx: &mut ClassCtx,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        Node::Let {
            name, ann, value, ..
        } => {
            walk_expr_class_checks(value, ctx, src, file, errs);
            match ann.as_ref().and_then(scalar_class_of_ann) {
                Some(ann_class) => {
                    if let Some(val_class) = confident_scalar_class(value, ctx) {
                        if ann_class != val_class {
                            errs.push(diag_from_span(
                                src,
                                file,
                                "no implicit int↔float conversion (RFC 0011); write the value in the annotated type (e.g. `5.0`) or use an explicit `as` cast".to_string(),
                                value.span(),
                                LET_CLASS_MISMATCH_CODE,
                            ));
                        }
                    }
                    ctx.classes.insert(name.clone(), ann_class);
                }
                None => {
                    // A non-scalar / unannotated (or unknown) binding shadows any
                    // same-named param: drop its tracked class so a later use
                    // resolves to `None`, never a stale fact.
                    ctx.classes.remove(name);
                }
            }
        }
        Node::Assign { name, value, .. } => {
            walk_expr_class_checks(value, ctx, src, file, errs);
            if let Some(&ann_class) = ctx.classes.get(name) {
                if let Some(val_class) = confident_scalar_class(value, ctx) {
                    if ann_class != val_class {
                        errs.push(diag_from_span(
                            src,
                            file,
                            "no implicit int↔float conversion (RFC 0011); write the value in the declared type or use an explicit `as` cast".to_string(),
                            value.span(),
                            LET_CLASS_MISMATCH_CODE,
                        ));
                    }
                }
            }
        }
        Node::Return { value: Some(v), .. } => {
            walk_expr_class_checks(v, ctx, src, file, errs);
            // E2010, NEW direction: float-declared return + confident-Int value.
            // The existing infer-based pass handles the float-value-into-int-fn
            // direction; the loose-call case stays `None` here and never fires.
            if ret_class == Some(ScalarClass::Float)
                && confident_scalar_class(v, ctx) == Some(ScalarClass::Int)
            {
                errs.push(diag_from_span(
                    src,
                    file,
                    "return type mismatch: function returns a float but this returns an integer value (RFC 0011 — no implicit int→float conversion; write `<value>.0` or use an explicit `as` cast)".to_string(),
                    v.span(),
                    RETURN_TYPE_MISMATCH_CODE,
                ));
            }
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_expr_class_checks(cond, ctx, src, file, errs);
            let mut inner = ctx.clone();
            check_scalar_classes(then_branch, ret_class, &mut inner, src, file, errs);
            if let Some(eb) = else_branch {
                let mut inner2 = ctx.clone();
                check_scalar_classes(eb, ret_class, &mut inner2, src, file, errs);
            }
        }
        #[cfg(feature = "std-surface")]
        Node::While { cond, body, .. } => {
            walk_expr_class_checks(cond, ctx, src, file, errs);
            let mut inner = ctx.clone();
            check_scalar_classes(body, ret_class, &mut inner, src, file, errs);
        }
        Node::For { body, .. } | Node::ForEach { body, .. } => {
            let mut inner = ctx.clone();
            check_scalar_classes(body, ret_class, &mut inner, src, file, errs);
        }
        Node::Block { stmts, .. } => {
            let mut inner = ctx.clone();
            check_scalar_classes(stmts, ret_class, &mut inner, src, file, errs);
        }
        // Nested fns carry their own return type; the mini-module recursion
        // checks them, and their diagnostics ride the body-filter whitelist.
        Node::FnDef { .. } => {}
        // Any other statement is an expression position (e.g. a bare call):
        // walk it for the expression-level checks.
        other => walk_expr_class_checks(other, ctx, src, file, errs),
    }
}

/// Recursively check an expression for the mixed-class binop (E2013), u64
/// unsigned-context (E2014), and `as bool` (E2016) diagnostics.
fn walk_expr_class_checks(
    node: &Node,
    ctx: &ClassCtx,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        Node::Binary {
            op: _,
            left,
            right,
            span,
        } => {
            let l = confident_scalar_class(left, ctx);
            let r = confident_scalar_class(right, ctx);
            if matches!(
                (l, r),
                (Some(ScalarClass::Int), Some(ScalarClass::Float))
                    | (Some(ScalarClass::Float), Some(ScalarClass::Int))
            ) {
                errs.push(diag_from_span(
                    src,
                    file,
                    "no implicit int↔float conversion (RFC 0011): this operator mixes an integer and a float operand; convert one side with an explicit `as` cast".to_string(),
                    *span,
                    MIXED_CLASS_BINOP_CODE,
                ));
            }
            // issue #99: `u64` is a first-class value kind — the sign-sensitive
            // comparison / `/` / `%` operators lower to the UNSIGNED variants
            // (`ult`/`ule`/`ugt`/`uge`/`divui`/`remui`) when an operand is
            // `ScalarU64`, and `<u64> as f32|f64` lowers via `uitofp`, so u64 no
            // longer needs a fail-closed reject in any of these contexts.
            walk_expr_class_checks(left, ctx, src, file, errs);
            walk_expr_class_checks(right, ctx, src, file, errs);
        }
        Node::Bitwise {
            op: _, left, right, ..
        } => {
            // issue #99: `u64 >> n` lowers to the LOGICAL shift (`arith.shrui`)
            // when an operand is `ScalarU64`. (`<< & | ^` are sign-agnostic.)
            walk_expr_class_checks(left, ctx, src, file, errs);
            walk_expr_class_checks(right, ctx, src, file, errs);
        }
        Node::As { expr, ty, span } => {
            if is_bool_ann(ty) {
                errs.push(diag_from_span(
                    src,
                    file,
                    "cast to `bool` is not permitted; use `x != 0`".to_string(),
                    *span,
                    AS_BOOL_CODE,
                ));
            }
            walk_expr_class_checks(expr, ctx, src, file, errs);
        }
        Node::Paren(inner, _) => walk_expr_class_checks(inner, ctx, src, file, errs),
        Node::Neg { operand, .. } | Node::Not { operand, .. } => {
            walk_expr_class_checks(operand, ctx, src, file, errs)
        }
        Node::Logical { left, right, .. } => {
            walk_expr_class_checks(left, ctx, src, file, errs);
            walk_expr_class_checks(right, ctx, src, file, errs);
        }
        Node::Call { args, .. } => {
            for a in args {
                walk_expr_class_checks(a, ctx, src, file, errs);
            }
        }
        _ => {}
    }
}

/// Look up an enum's full variant-name list by enum name. Returns `None` when
/// the registry isn't populated or `name` isn't a module-level enum.
fn enum_variants_of(name: &str) -> Option<Vec<String>> {
    ENUM_VARIANTS.with(|cell| cell.borrow().as_ref().and_then(|t| t.get(name).cloned()))
}

/// Build the enum-variant registry from a module's `Node::EnumDef` items.
fn build_enum_variants(module: &Module) -> EnumVariantTable {
    let mut table = EnumVariantTable::new();
    for item in &module.items {
        if let Node::EnumDef { name, variants, .. } = item {
            table.insert(
                name.clone(),
                variants.iter().map(|v| v.name.clone()).collect(),
            );
        }
    }
    table
}

/// Split an `EnumVariant` pattern path into `(enum_name, variant_name)`.
///
/// Patterns carry a dotted+`::` path such as `"Mode::On"` or, when the enum
/// was imported, `"config.Mode::On"`. The enum is the segment immediately
/// before the final `::Variant`; any leading `module.` qualifier is stripped so
/// the bare enum name matches the registry (which keys on the declared name).
/// Returns `None` for a path without a `::` separator (not an enum variant).
fn split_enum_variant_path(path: &str) -> Option<(String, String)> {
    let (head, variant) = path.rsplit_once("::")?;
    // `head` may be `Mode` or `config.Mode`; take the last dotted segment.
    let enum_name = head.rsplit('.').next().unwrap_or(head);
    Some((enum_name.to_string(), variant.to_string()))
}

/// RAII guard that installs the enum-variant registry into the thread-local for
/// the duration of a `check_module_types_in_file` call, merging onto (not
/// replacing) any parent registry — exactly like `IntraSigGuard` — so an enum
/// match inside a fn body (whose recursive sub-module has no `EnumDef`s) still
/// resolves the module's enums. Restores the previous registry on drop.
struct EnumVariantsGuard {
    prev: Option<EnumVariantTable>,
    prev_payloads: Option<EnumPayloadTable>,
}

impl EnumVariantsGuard {
    fn install(table: EnumVariantTable, payloads: EnumPayloadTable) -> Self {
        let prev = ENUM_VARIANTS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let prev = slot.clone();
            match slot.as_mut() {
                Some(existing) => existing.extend(table),
                None => *slot = Some(table),
            }
            prev
        });
        // Merge-onto-parent + restore the payload registry in lockstep, so a
        // match inside a fn body still resolves the module's variant payloads.
        let prev_payloads = ENUM_PAYLOADS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let prev = slot.clone();
            match slot.as_mut() {
                Some(existing) => existing.extend(payloads),
                None => *slot = Some(payloads),
            }
            prev
        });
        EnumVariantsGuard {
            prev,
            prev_payloads,
        }
    }
}

impl Drop for EnumVariantsGuard {
    fn drop(&mut self) {
        ENUM_VARIANTS.with(|cell| *cell.borrow_mut() = self.prev.take());
        ENUM_PAYLOADS.with(|cell| *cell.borrow_mut() = self.prev_payloads.take());
    }
}

/// Check a `match`'s arms for exhaustiveness over a sum/enum type.
///
/// Returns `Err` only when the match is provably non-exhaustive on an enum:
///   * no arm is a catch-all (`_` wildcard or a bare `Ident` binding), AND
///   * every non-catch-all arm is an `EnumVariant` of a SINGLE known enum, AND
///   * the covered variants are a strict subset of that enum's full variant set.
///
/// Anything outside that shape is accepted (returns `Ok`): integer / float /
/// bool / string literal matches (which can't be exhaustive without a wildcard
/// and legitimately rely on a `_` arm — and main.mind's 51 tag-matches are all
/// of this kind), matches mixing variants of different enums, matches whose
/// enum isn't in the registry, and matches that already include a catch-all.
/// This keeps the check sound and false-positive-free on the existing corpus.
fn check_match_exhaustiveness(arms: &[crate::ast::MatchArm]) -> Result<(), TypeErrSpan> {
    use crate::ast::Pattern;
    // An UNGUARDED wildcard or bare-ident arm makes any match exhaustive. A
    // GUARDED irrefutable arm (`_ if g`, `x if g`) does NOT: its guard can fail,
    // leaving the value uncovered (the Rust rule; pattern-guards W1.5a / drift
    // #131). So a guarded arm never satisfies exhaustiveness.
    let has_catch_all = arms
        .iter()
        .any(|a| a.guard.is_none() && matches!(a.pattern, Pattern::Wildcard | Pattern::Ident(_)));
    if has_catch_all {
        return Ok(());
    }
    // Collect the enum name + covered variants from the EnumVariant arms. If
    // any arm is not an EnumVariant (e.g. a literal), this is not a pure-enum
    // match and exhaustiveness is not enforced here.
    let mut enum_name: Option<String> = None;
    let mut covered: BTreeSet<String> = BTreeSet::new();
    let mut last_span = AstSpan::new(0, 0);
    for arm in arms {
        last_span = arm.span;
        match &arm.pattern {
            Pattern::EnumVariant { path, .. } => {
                let Some((e, v)) = split_enum_variant_path(path) else {
                    return Ok(());
                };
                match &enum_name {
                    None => enum_name = Some(e),
                    // Variants of two different enums in one match -> not a
                    // single-enum exhaustiveness question; defer.
                    Some(existing) if *existing != e => return Ok(()),
                    _ => {}
                }
                // A GUARDED variant arm does NOT cover its variant — the guard
                // can fail, leaving that variant unmatched (drift #131). Only an
                // unguarded variant arm marks the variant covered.
                if arm.guard.is_none() {
                    covered.insert(v);
                }
            }
            // Any non-enum, non-catch-all pattern (a literal) -> not enforced.
            _ => return Ok(()),
        }
    }
    let Some(enum_name) = enum_name else {
        // No arms at all, or no enum arms — nothing to check.
        return Ok(());
    };
    // Only enforce when the enum is actually declared (registry hit); an
    // unknown enum name means we can't know the full variant set, so defer.
    let Some(all_variants) = enum_variants_of(&enum_name) else {
        return Ok(());
    };
    let missing: Vec<String> = all_variants
        .iter()
        .filter(|v| !covered.contains(*v))
        .cloned()
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(TypeErrSpan {
        msg: format!(
            "non-exhaustive `match` on enum `{enum_name}`: missing variant(s) {}; \
             add the missing arm(s) or a wildcard `_` arm",
            missing
                .iter()
                .map(|v| format!("`{enum_name}::{v}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        span: last_span,
    })
}

/// Validate an intra-module call against a captured `IntraFnSig`: enforce
/// arity (always knowable from the AST). Returns the loose i64-ABI result type
/// on success (signature-aware return inference is a later slice — the
/// soundness hole this closes is the *unchecked argument count*, not the return
/// type). The arg expressions are still walked so their own errors surface.
fn check_intra_fn_call(
    callee: &str,
    sig: &IntraFnSig,
    args: &[Node],
    span: AstSpan,
    _env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    if args.len() != sig.param_count {
        return Err(TypeErrSpan {
            msg: format!(
                "function `{callee}` expects {expected} argument(s); got {got}",
                expected = sig.param_count,
                got = args.len(),
            ),
            span,
        });
    }
    // Bug #38 — the ONE per-arg type comparison done under the loose i64 ABI:
    // refuse a fixed-size `bytes[N]` buffer flowing into a growable `bytes`
    // parameter (a silent miscompile). NARROW — fires only on that exact
    // pairing (see `fixed_bytes_into_vec_violation`), so the loose typing the
    // rest of this fn relies on is untouched.
    if let Some((i, arg_span)) = fixed_bytes_into_vec_violation(&sig.param_types, args) {
        return Err(TypeErrSpan {
            msg: format!(
                "function `{callee}` parameter {i} is a growable `bytes` vec, but the \
                 argument is a fixed-size `bytes[N]` buffer handle. These share the i64 \
                 ABI but have incompatible layouts — reading `.length` on the buffer \
                 would read raw payload as the vec length (silent miscompile). Copy the \
                 buffer into a `bytes` vec, or declare the parameter `bytes[N]`."
            ),
            span: arg_span,
        });
    }
    // Arity is otherwise the ONLY property validated here — it is always soundly
    // knowable from the AST regardless of the loose i64 ABI (see the module note
    // on `build_intra_fn_sigs`). We deliberately do NOT re-infer the argument
    // expressions: per-arg type checking is not soundly determinable under the
    // i64 ABI, and a re-walk via `infer_expr` false-positives on valid value
    // forms the outer type-check pass already accepts — notably enum-variant
    // constructors in argument position (`f(Mode::On)`), which are not
    // value-env identifiers and would wrongly raise E2002. (Regression fix: an
    // 0.8.x arg re-walk reddened the compositional suite's
    // `enum_tag_match_writes_field_in_each_arm`; arity tests assert E2005 only.)
    Ok((ValueType::ScalarI64, span))
}

/// RAII guard that installs an intra-module signature table into the
/// thread-local for the duration of a `check_module_types_in_file` call and
/// restores the previous value on drop.
///
/// Installation MERGES `sigs` onto any table the parent already installed
/// rather than replacing it. This is load-bearing: the FnDef-body checker
/// recurses into `check_module_types_in_file` with a sub-module that holds
/// only the body's statements (NO module-level `FnDef`s), so a plain replace
/// would wipe the table for the duration of the body check and disable call
/// validation exactly where most calls live. Merging keeps the outer module's
/// signatures visible inside bodies while letting any (currently nonexistent)
/// nested fn shadow. The previous table is restored on drop, so there is no
/// leakage across files / threads / sibling fns. Mirrors the moat-preserving
/// thread-local discipline of `CM_TABLE`.
struct IntraSigGuard {
    prev: Option<IntraFnSigs>,
}

impl IntraSigGuard {
    fn install(sigs: IntraFnSigs) -> Self {
        let prev = INTRA_FN_SIGS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let prev = slot.clone();
            match slot.as_mut() {
                // Merge onto the parent table (child entries win).
                Some(existing) => existing.extend(sigs),
                None => *slot = Some(sigs),
            }
            prev
        });
        IntraSigGuard { prev }
    }
}

impl Drop for IntraSigGuard {
    fn drop(&mut self) {
        INTRA_FN_SIGS.with(|cell| *cell.borrow_mut() = self.prev.take());
    }
}

// ── Bug #38 follow-up — fixed-`bytes[N]` LOCAL bindings ───────────────
//
// The #38 guard (`arg_is_fixed_bytes_buffer`) recognised a fixed buffer only in
// its two SYNTACTIC value forms — `bytes[N].zero()` written DIRECTLY in argument
// position, or a call whose declared return type is `bytes[N]`. It missed the
// fixed buffer flowing through a LOCAL BINDING:
//
//     let buf: bytes[32] = bytes[32].zero()   // a fixed buffer handle
//     take_bytes(buf)                          // `bytes` param — SAME miscompile
//
// `let buf: bytes[32] = …` is NOT a copy into a growable vec (the guard message's
// own suggested fix); it keeps the identical i64 handle, so passing `buf` to a
// `bytes` parameter reads the buffer's raw payload at +8 as the vec `.length` —
// exactly the silent miscompile #38 fails loud on, but via a one-line alias.
//
// We close it by recording every body-local binding whose DECLARED annotation is
// `bytes[N]`, then resolving an `Ident` argument against that set in
// `arg_is_fixed_bytes_buffer`. Threaded via the same set-on-entry / restore-on-
// drop thread-local discipline as `INTRA_FN_SIGS` so the shared
// `check_module_types*` signature (the moat) is untouched. Built once per FnDef
// body from its `let`/`let mut` statements and merged onto any parent set so a
// nested block's body check still sees the enclosing fn's bytes[N] locals.
thread_local! {
    static FIXED_BYTES_LOCALS: std::cell::RefCell<Option<BTreeSet<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// True iff `name` is a body-local binding declared with a fixed `bytes[N]`
/// annotation in the currently-checked fn body. `false` when the side-table is
/// unpopulated (no fn-body context) — preserving the loose i64 ABI everywhere
/// the narrow #38 pairing does not apply.
fn fixed_bytes_local(name: &str) -> bool {
    FIXED_BYTES_LOCALS.with(|cell| cell.borrow().as_ref().is_some_and(|set| set.contains(name)))
}

/// Collect the names of `let`/`let mut` bindings whose declared annotation is a
/// fixed `bytes[N]` type, recursing into nested blocks/branches/loops so a
/// buffer declared in any sub-scope of the fn body is tracked. Pure and additive
/// — it only ever records names that are provably fixed buffers, so it can never
/// flag a growable `bytes` value (which never carries a `bytes[N]` annotation).
fn collect_fixed_bytes_locals(stmts: &[Node], out: &mut BTreeSet<String>) {
    for s in stmts {
        match s {
            Node::Let { name, ann, .. } => {
                if matches!(ann, Some(a) if typeann_is_fixed_bytes(a)) {
                    out.insert(name.clone());
                }
            }
            Node::Block { stmts, .. } => collect_fixed_bytes_locals(stmts, out),
            Node::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_fixed_bytes_locals(then_branch, out);
                if let Some(eb) = else_branch {
                    collect_fixed_bytes_locals(eb, out);
                }
            }
            Node::For { body, .. } | Node::ForEach { body, .. } => {
                collect_fixed_bytes_locals(body, out);
            }
            #[cfg(feature = "std-surface")]
            Node::While { body, .. } => collect_fixed_bytes_locals(body, out),
            #[cfg(feature = "std-surface")]
            Node::Region { body, .. } => collect_fixed_bytes_locals(body, out),
            Node::Match { arms, .. } => {
                for arm in arms {
                    if let Node::Block { stmts, .. } = &arm.body {
                        collect_fixed_bytes_locals(stmts, out);
                    }
                }
            }
            _ => {}
        }
    }
}

/// RAII guard that installs the fixed-`bytes[N]` local set into the thread-local
/// for the duration of a fn-body check and restores the previous value on drop.
/// Merges onto any parent set (child names win) so a nested block's recursive
/// body check still sees the enclosing fn's `bytes[N]` locals.
struct FixedBytesLocalsGuard {
    prev: Option<BTreeSet<String>>,
}

impl FixedBytesLocalsGuard {
    fn install(locals: BTreeSet<String>) -> Self {
        let prev = FIXED_BYTES_LOCALS.with(|cell| {
            let mut slot = cell.borrow_mut();
            let prev = slot.clone();
            match slot.as_mut() {
                Some(existing) => existing.extend(locals),
                None => *slot = Some(locals),
            }
            prev
        });
        FixedBytesLocalsGuard { prev }
    }
}

impl Drop for FixedBytesLocalsGuard {
    fn drop(&mut self) {
        FIXED_BYTES_LOCALS.with(|cell| *cell.borrow_mut() = self.prev.take());
    }
}

// ── Cross-module imports (Phase 10.6 item 9 / Phase 15) — D2 ──────────
//
// The module table is threaded via a thread-local set by the gated
// entrypoint below, NOT via a new parameter on the shared
// `check_module_types*` signature. This keeps the default hot-path
// signature byte-identical, so the µs frontend / headline criterion
// benches are provably untouched (the moat). When the feature is off,
// none of this compiles and the `Node::Import` arm is an inert binding.
#[cfg(feature = "cross-module-imports")]
thread_local! {
    static CM_TABLE: std::cell::RefCell<Option<crate::project::module_table::ModuleTable>> =
        const { std::cell::RefCell::new(None) };
}

/// Inject the exported names of the module referenced by `path` into
/// `tenv` as opaque scalars (`ValueType::ScalarI32` — the same type a
/// bare `Node::Import` already yields in `infer_expr`). Exact path
/// match only; globs / re-export chains are deliverable 3+.
#[cfg(feature = "cross-module-imports")]
fn cm_inject_imported_symbols(tenv: &mut TypeEnv, path: &[String]) {
    CM_TABLE.with(|cell| {
        if let Some(table) = cell.borrow().as_ref() {
            let key = path.join(".");
            if let Some(exports) = table.get(&key) {
                for sym in &exports.exported {
                    tenv.entry(sym.clone()).or_insert(ValueType::ScalarI32);
                }
            }
        }
    });
}

/// Project-scope setter (D3): the project builder sets the
/// whole-project module table once before the per-file compile loop so
/// the existing single-file pipeline (`check_module_types_in_file`)
/// resolves cross-file symbols WITHOUT a signature change to the shared
/// compile path. Pass `None` to clear after the loop. Keeping the
/// default pipeline signature byte-identical is what holds the moat.
#[cfg(feature = "cross-module-imports")]
pub fn cm_set_project_table(table: Option<crate::project::module_table::ModuleTable>) {
    CM_TABLE.with(|cell| *cell.borrow_mut() = table);
}

/// RFC 0005 Phase B — find an imported fn's signature by name across
/// the active project table.  Returns `None` when the table is empty
/// (default-feature build, no project context) or when the name
/// doesn't resolve to a typed `pub fn`.  The clone is cheap: each
/// `ExportedFn` is a name + a `Vec<TypeAnn>` + an `Option<TypeAnn>`.
#[cfg(feature = "cross-module-imports")]
fn cm_lookup_fn(name: &str) -> Option<crate::project::module_table::ExportedFn> {
    CM_TABLE.with(|cell| {
        cell.borrow()
            .as_ref()
            .and_then(|table| table.lookup_imported_fn(name).cloned())
    })
}

/// True iff ANY module in the active project table exports `name`. Unlike
/// `cm_lookup_fn` (which resolves only typed `fn` signatures), this answers the
/// bare resolvability question for EVERY exported symbol kind — `fn`, `const`,
/// `type`, `struct` — so a cross-module const (`fixed_point.Q16_ONE`, which the
/// parser has normalised to a bare `Q16_ONE` reference) or an explicitly
/// `export`ed fn is not a genuinely-undefined reference. Empty on the
/// default-feature / single-file (`mindc check`) path (table is `None`), so
/// that path is byte-identical.
#[cfg(feature = "cross-module-imports")]
pub(crate) fn cm_symbol_exported(name: &str) -> bool {
    CM_TABLE.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|table| table.exports_symbol(name))
    })
}

/// RFC 0005 Phase B — validate a call against an imported fn's
/// signature.  Compares arity then per-arg types against the
/// declared `param_types`; returns the declared `ret_type` as a
/// `ValueType` (default i64 ABI when the declaration is a Named
/// struct or an unsupported aggregate).
#[cfg(feature = "cross-module-imports")]
fn check_imported_fn_call(
    sig: &crate::project::module_table::ExportedFn,
    args: &[Node],
    span: AstSpan,
    env: &TypeEnv,
) -> Result<(ValueType, AstSpan), TypeErrSpan> {
    if args.len() != sig.param_types.len() {
        return Err(TypeErrSpan {
            msg: format!(
                "imported `{name}` expects {expected} argument(s); got {got}",
                name = sig.name,
                expected = sig.param_types.len(),
                got = args.len(),
            ),
            span,
        });
    }
    // Bug #38 — refuse a fixed-size `bytes[N]` buffer flowing into a growable
    // `bytes` parameter (narrow; see `fixed_bytes_into_vec_violation`). Runs
    // before the ValueType-level compatibility walk below, which cannot see
    // the distinction (both collapse to ScalarI64).
    if let Some((i, arg_span)) = fixed_bytes_into_vec_violation(&sig.param_types, args) {
        return Err(TypeErrSpan {
            msg: format!(
                "imported `{name}` parameter {i} is a growable `bytes` vec, but the \
                 argument is a fixed-size `bytes[N]` buffer handle. These share the i64 \
                 ABI but have incompatible layouts — reading `.length` on the buffer \
                 would read raw payload as the vec length (silent miscompile). Copy the \
                 buffer into a `bytes` vec, or declare the parameter `bytes[N]`.",
                name = sig.name,
            ),
            span: arg_span,
        });
    }
    for (i, (arg, declared)) in args.iter().zip(sig.param_types.iter()).enumerate() {
        let (actual, arg_span) = infer_expr(arg, env)?;
        let expected = cm_typeann_to_valuetype(declared);
        if !cm_arg_compatible(&expected, &actual) {
            // Phase D2 (light): if the declared parameter is a Named
            // struct (`Vec`, `String`, `Map`, ...), surface the
            // struct's name in the error message rather than the
            // lossy ScalarI64 it lowers to under the Option-C heap
            // ABI. The compatibility check itself stays permissive
            // (Named structs accept i64-typed values since that's
            // the on-wire ABI), but when widening fails on a
            // *different* arg the user now sees which Named-typed
            // parameter triggered it.
            return Err(TypeErrSpan {
                msg: format!(
                    "imported `{name}` argument {i} expects {exp}; got {got}",
                    name = sig.name,
                    i = i,
                    exp = describe_param_type(declared),
                    got = describe_value_type(&actual),
                ),
                span: arg_span,
            });
        }
    }
    let ret = sig
        .ret_type
        .as_ref()
        .map(cm_typeann_to_valuetype)
        .unwrap_or(ValueType::ScalarI64);
    Ok((ret, span))
}

/// RFC 0005 Phase B — map a `TypeAnn` to a `ValueType` for cross-
/// module call-site checking.  Reuses the type-checker's existing
/// `valuetype_from_ann`; falls back to `ScalarI64` for everything the
/// helper can't resolve (Named struct/enum types, Slice/Array/Ref
/// aggregates).  This matches RFC 0005's Option-C heap ABI where
/// struct values are i64 base-addresses on the wire.
#[cfg(feature = "cross-module-imports")]
fn cm_typeann_to_valuetype(ann: &crate::ast::TypeAnn) -> ValueType {
    valuetype_from_ann(ann).unwrap_or(ValueType::ScalarI64)
}

/// Phase D2 (light) — render a parameter's TypeAnn for an error
/// message in a way that *preserves* Named struct identity. The Phase
/// B compatibility check still operates on `ValueType` (where Named
/// structs collapse to `ScalarI64`), but when we hand an error string
/// to the user, "expected Vec (heap-record i64 addr)" is far more
/// debuggable than "expected scalar i64". Slice / Array / Ref
/// aggregates and primitive scalars fall through to the existing
/// `describe_value_type` rendering.
#[cfg(feature = "cross-module-imports")]
fn describe_param_type(ann: &crate::ast::TypeAnn) -> String {
    match ann {
        crate::ast::TypeAnn::Named(name) => {
            format!("{name} (heap-record i64 addr)")
        }
        _ => describe_value_type(&cm_typeann_to_valuetype(ann)),
    }
}

/// RFC 0005 Phase B — compatibility check for a single arg.  Accepts
/// exact matches plus the universal i32 -> i64 widening that integer
/// literals depend on (literals come in as `ScalarI32` from the
/// lexer; the call ABI is i64).
#[cfg(feature = "cross-module-imports")]
fn cm_arg_compatible(expected: &ValueType, actual: &ValueType) -> bool {
    if expected == actual {
        return true;
    }
    matches!(
        (expected, actual),
        (ValueType::ScalarI64, ValueType::ScalarI32) | (ValueType::ScalarI32, ValueType::ScalarI64)
    )
}

/// Gated entrypoint: type-check `module` with cross-module symbol
/// resolution against `table`. Sets the thread-local for the duration
/// of the check and clears it afterward (no leakage across calls).
#[cfg(feature = "cross-module-imports")]
pub fn check_module_types_with_modules(
    module: &Module,
    src: &str,
    file: Option<&str>,
    env: &TypeEnv,
    table: &crate::project::module_table::ModuleTable,
) -> Vec<Pretty> {
    CM_TABLE.with(|cell| *cell.borrow_mut() = Some(table.clone()));
    let result = check_module_types_in_file(module, src, file, env);
    CM_TABLE.with(|cell| *cell.borrow_mut() = None);
    result
}

pub fn check_module_types_in_file(
    module: &Module,
    src: &str,
    file: Option<&str>,
    env: &TypeEnv,
) -> Vec<Pretty> {
    let mut errs = Vec::new();
    let mut tenv = env.clone();

    // E2023 — the `__mind_` prefix is reserved for compiler intrinsics (RFC 0005
    // i64-ABI surface: `__mind_alloc` / `__mind_load_i8` / …). A user `fn` with
    // that prefix would shadow the intrinsic on the interpreter fn-table oracle
    // (a silent miscompile), so it is rejected fail-loud here on `check`/`build`,
    // and defensively refused again in the interpreter fn-table install and in
    // lowering (never merely skipped). Fires for ALL modules, independent of
    // `#[bimap]`.
    for item in &module.items {
        if let Node::FnDef { name, span, .. } = item {
            if name.starts_with("__mind_") {
                errs.push(diag_from_span(
                    src,
                    file,
                    format!(
                        "reserved intrinsic prefix: a user function may not be named `{name}`; the `__mind_` prefix is reserved for compiler intrinsics"
                    ),
                    *span,
                    "E2023",
                ));
            }
        }
    }

    // Single prologue pass: classify + collect all per-category data in one
    // walk instead of four separate passes (repr_c_struct_names, fn_tensor_sigs,
    // has_fn any(), has_enum any()) plus a conditional FnDef pre-register walk.
    // For modules with none of these item types (e.g. scalar_math), only one
    // iterator traversal runs instead of four — each with its own setup/teardown
    // overhead even on a one-item list. Insertion order within each container is
    // identical to the original (module.items order) → byte-identical output.
    let mut repr_c_struct_names = std::collections::BTreeSet::<String>::new();
    let mut fn_tensor_sigs: FnTensorSigs = FnTensorSigs::new();
    let mut intra_fn_sigs: IntraFnSigs = IntraFnSigs::new();
    let mut has_fn = false;
    let mut has_enum = false;

    for item in &module.items {
        match item {
            Node::StructDef { name, attrs, .. }
                if attrs
                    .iter()
                    .any(|a| a.name == "repr" && a.args.iter().any(|arg| arg == "C")) =>
            {
                repr_c_struct_names.insert(name.clone());
            }
            Node::FnDef {
                name,
                params,
                ret_type,
                ..
            } => {
                has_fn = true;
                // Pre-register name for intra-module call resolution.
                tenv.entry(name.clone()).or_insert(ValueType::ScalarI64);
                // Collect tensor-param signatures for RFC 0012 Phase A call-site checks.
                let tensor_params: Vec<TensorParamSig> = params
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, p)| match &p.ty {
                        TypeAnn::Tensor { dims, dtype } | TypeAnn::DiffTensor { dims, dtype } => {
                            Some((idx, p.name.clone(), dims.clone(), dtype.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                if !tensor_params.is_empty() {
                    fn_tensor_sigs.insert(name.clone(), tensor_params);
                }
                // Collect arity for RFC 0005 Phase B call-site arity checks.
                intra_fn_sigs.insert(
                    name.clone(),
                    IntraFnSig {
                        param_count: params.len(),
                        param_types: params.iter().map(|p| p.ty.clone()).collect(),
                        ret_type: ret_type.clone(),
                    },
                );
            }
            Node::EnumDef { .. } => {
                has_enum = true;
            }
            _ => {}
        }
    }

    // Install the intra-fn signature side-table (built above, not a separate walk).
    let _intra_sig_guard = has_fn.then(|| IntraSigGuard::install(intra_fn_sigs));

    // Install the enum-variant registry for match-exhaustiveness checking
    // (same merge-onto-parent discipline, so an enum match inside a fn body
    // still resolves the module's enums). Restored on return.
    let _enum_variants_guard = has_enum.then(|| {
        EnumVariantsGuard::install(build_enum_variants(module), build_enum_payloads(module))
    });

    for item in &module.items {
        match item {
            Node::Let {
                name,
                ann,
                value,
                span,
                ..
            } => match ann {
                // RFC 0005 Phase 6.2b Gap 2 — fixed-size array annotation
                // `[T; N]`.  Check that the RHS array literal has exactly N
                // elements; element-type compatibility defers to the element
                // type's own value-type inference.
                #[cfg(feature = "std-surface")]
                Some(crate::ast::TypeAnn::Array { length, .. }) => {
                    // Count elements in the RHS if it's an ArrayLit.
                    let rhs_len: Option<usize> = match value.as_ref() {
                        Node::ArrayLit { elements, .. } => Some(elements.len()),
                        _ => None,
                    };
                    if let Some(actual) = rhs_len {
                        if actual != *length as usize {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "array length mismatch for `{}`: annotation [_; {}] but \
                                     literal has {} elements",
                                    name, length, actual
                                ),
                                *span,
                                TYPE_ERR_CODE,
                            ));
                        }
                    }
                    // Register as ScalarI64 in the env (element type, v1 approximation).
                    tenv.insert(name.clone(), ValueType::ScalarI64);
                }
                Some(annotation) => match valuetype_from_ann(annotation) {
                    Some(vt_ann) => {
                        match infer_expr(value, &tenv) {
                            Ok((vt, _)) => {
                                let allow_scalar_fill = matches!(
                                    (&vt_ann, &vt),
                                    (ValueType::Tensor(_), ValueType::ScalarI32)
                                );
                                // Implicit integer narrowing (data-loss):
                                // a wider integer flowing into a narrower
                                // declared slot truncates at runtime. The
                                // spec requires an explicit `as` cast
                                // (AsCast, grammar-syntax.ebnf:240). Emit the
                                // precise NARROWING_CODE so the FnDef-body
                                // pass surfaces it inside function bodies too.
                                if is_implicit_narrowing(&vt_ann, &vt) {
                                    errs.push(diag_from_span(
                                        src,
                                        file,
                                        format!(
                                            "implicit narrowing {} -> {} loses data for `{}`; use an explicit `as {}` cast",
                                            describe_value_type(&vt),
                                            describe_value_type(&vt_ann),
                                            name,
                                            describe_value_type(&vt_ann),
                                        ),
                                        value.span(),
                                        NARROWING_CODE,
                                    ));
                                } else if vt_ann != vt && !allow_scalar_fill {
                                    // RFC 0012 Phase A: when both sides are
                                    // tensors, emit precise `shape::*`
                                    // diagnostics instead of the generic
                                    // `TYPE_ERR_CODE` mismatch.
                                    let shape_handled = match (&vt_ann, &vt) {
                                        (ValueType::Tensor(ann_t), ValueType::Tensor(inf_t)) => {
                                            check_tensor_shape_compat(
                                                ann_t,
                                                inf_t,
                                                name,
                                                src,
                                                file,
                                                value.span(),
                                                &mut errs,
                                            )
                                        }
                                        _ => false,
                                    };
                                    if !shape_handled {
                                        errs.push(diag_from_span(
                                            src,
                                            file,
                                            format!(
                                                "type mismatch for `{}`: annotation {} vs inferred {}",
                                                name,
                                                describe_value_type(&vt_ann),
                                                describe_value_type(&vt)
                                            ),
                                            value.span(),
                                            TYPE_ERR_CODE,
                                        ));
                                    }
                                }
                            }
                            Err(e) => errs.push(diag_from_type_err(src, file, e)),
                        }
                        tenv.insert(name.clone(), vt_ann);
                    }
                    None => errs.push(diag_from_span(
                        src,
                        file,
                        format!("unsupported annotation for `{}`", name),
                        *span,
                        TYPE_ERR_CODE,
                    )),
                },
                None => match infer_expr(value, &tenv) {
                    Ok((vt, _)) => {
                        tenv.insert(name.clone(), vt);
                    }
                    Err(e) => errs.push(diag_from_type_err(src, file, e)),
                },
            },
            Node::Assign { name, value, .. } => {
                let rhs = infer_expr(value, &tenv);
                match (tenv.get(name).cloned(), rhs) {
                    (Some(vt_lhs), Ok((vt_rhs, _))) => {
                        if is_implicit_narrowing(&vt_lhs, &vt_rhs) {
                            // Implicit integer narrowing on assignment: same
                            // data-loss hazard as a narrowing let-binding.
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "implicit narrowing {} -> {} loses data assigning to `{}`; use an explicit `as {}` cast",
                                    describe_value_type(&vt_rhs),
                                    describe_value_type(&vt_lhs),
                                    name,
                                    describe_value_type(&vt_lhs),
                                ),
                                value.span(),
                                NARROWING_CODE,
                            ));
                        } else if vt_lhs != vt_rhs {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "cannot assign `{}`: expected {} but found {}",
                                    name,
                                    describe_value_type(&vt_lhs),
                                    describe_value_type(&vt_rhs)
                                ),
                                value.span(),
                                TYPE_ERR_CODE,
                            ));
                        }
                    }
                    (None, Ok((vt_rhs, _))) => {
                        tenv.insert(name.clone(), vt_rhs);
                    }
                    (_, Err(e)) => errs.push(diag_from_type_err(src, file, e)),
                }
            }
            // Import statements are handled at module level. With the
            // `cross-module-imports` feature and a populated module
            // table, a `use crate::a.b` injects module `crate.a.b`'s
            // exported names into the local type env so later
            // identifier lookups resolve across the file boundary.
            // Default build: byte-identical no-op (the arm compiles to
            // a discarded pattern binding, zero runtime cost).
            Node::Import { path, span } => {
                #[cfg(feature = "cross-module-imports")]
                cm_inject_imported_symbols(&mut tenv, path);
                // FAIL LOUD at the import/std-path resolution layer: a
                // featureless `mindc` (built with neither `std-surface`
                // nor `cross-module-imports`) carries an EMPTY bundled
                // std surface, so an `import std.X` resolves no names and
                // every `map_new` / `jv_parse` / `vec_*` call would
                // otherwise produce a confusing per-call `E2003 unsupported
                // call`. Point the user at the real cause ONCE, at the
                // import span, instead. This block only compiles on the
                // featureless build; a `std-surface`/`cross-module-imports`
                // binary (the default + the prescribed release build) takes
                // the inject path above and is byte-for-byte unchanged.
                #[cfg(not(any(feature = "std-surface", feature = "cross-module-imports")))]
                if path.first().map(String::as_str) == Some("std") {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "`import {}` requires a mindc built with the \
                             `std-surface` (or `cross-module-imports`) feature; \
                             this binary has no std surface so its functions \
                             cannot be resolved — rebuild with \
                             `--features std-surface`",
                            path.join("."),
                        ),
                        *span,
                        STD_IMPORT_NO_SURFACE_CODE,
                    ));
                }
                // `path` is consumed by `cm_inject_imported_symbols` (CMI on)
                // or by the std-surface check above (both off); `span` is only
                // read on the both-off path. Discard both here so every
                // remaining feature combo — notably `std-surface` ON +
                // `cross-module-imports` OFF — compiles warning-free without
                // re-touching this arm.
                let _ = (path, span);
            }
            // Phase 10.5 Tier-1: const introduces a name into the type env.
            // Phase 10.5 Tier-1: type aliases, exports — recorded but not
            // type-checked at v1; type-alias resolution is a v1.1 follow-up.
            // Phase 10.5 Tier-2: struct, enum — recorded; v1 typechecker
            // does not yet resolve named types in field positions, so we
            // intentionally skip rather than error.
            Node::Const {
                name, ty, value, ..
            } => {
                let rhs = infer_expr(value, &tenv);
                match rhs {
                    Ok((vt_rhs, _)) => {
                        if let Some(ann) = ty {
                            if let Some(vt_ann) = valuetype_from_ann(ann) {
                                // An integer-literal / const-foldable integer RHS
                                // coerces to an integer annotation as long as it
                                // WIDENS (or matches) — the same rule a fn-body
                                // `let x: i64 = 5` already applies (a bare literal
                                // defaults to i32). A genuine narrowing or a
                                // non-integer mismatch still errors.
                                let int_widen = vt_ann != vt_rhs
                                    && int_scalar_bits(&vt_ann).is_some()
                                    && int_scalar_bits(&vt_rhs).is_some()
                                    && !is_implicit_narrowing(&vt_ann, &vt_rhs);
                                if vt_ann != vt_rhs && !int_widen {
                                    errs.push(diag_from_span(
                                        src,
                                        file,
                                        format!(
                                            "const `{}`: expected {} but found {}",
                                            name,
                                            describe_value_type(&vt_ann),
                                            describe_value_type(&vt_rhs)
                                        ),
                                        value.span(),
                                        TYPE_ERR_CODE,
                                    ));
                                    continue;
                                }
                                tenv.insert(name.clone(), vt_ann);
                            } else {
                                // Named/Aliased type: skip ann check, record rhs
                                tenv.insert(name.clone(), vt_rhs);
                            }
                        } else {
                            tenv.insert(name.clone(), vt_rhs);
                        }
                    }
                    Err(e) => errs.push(diag_from_type_err(src, file, e)),
                }
            }
            // `extern const NAME: TYPE` — record the name in the type env so
            // consumers resolve it. An array/aggregate annotation has no direct
            // ValueType (`valuetype_from_ann` returns None), so we only insert a
            // scalar type when one is derivable; the name is already made
            // resolvable by `collect_decl_names` in either case, and index use
            // sites validate against the element type at the index arm.
            Node::ExternConst { name, ty, .. } => {
                if let Some(vt) = valuetype_from_ann(ty) {
                    tenv.insert(name.clone(), vt);
                }
            }
            Node::TypeAlias { .. }
            | Node::Export { .. }
            | Node::StructDef { .. }
            | Node::EnumDef { .. } => {
                // Record-only at v1: parser shipped, typechecker hooks deferred.
            }
            // RFC 0010 Phase A: validate extern "C" block signatures.
            // All parameter and return types must be Copy-compatible C ABI
            // types: i8/i16/i32/i64, u8/u16/u32/u64, f32/f64, bool, or
            // raw pointers `*const T` / `*mut T`. Aggregate types (String,
            // Vec, user-defined structs) are rejected with `safety::extern_non_copy`.
            Node::ExternBlock { fns, .. } => {
                for efn in fns {
                    for param in &efn.params {
                        if let Err(msg) =
                            check_extern_type_with_repr_c(&param.ty, &repr_c_struct_names)
                        {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!(
                                    "extern \"C\" fn `{}` parameter `{}`: {}",
                                    efn.name, param.name, msg
                                ),
                                param.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                    if let Some(ret) = &efn.ret_type {
                        if let Err(msg) = check_extern_type_with_repr_c(ret, &repr_c_struct_names) {
                            errs.push(diag_from_span(
                                src,
                                file,
                                format!("extern \"C\" fn `{}` return type: {}", efn.name, msg),
                                efn.span,
                                "safety::extern_non_repr_c",
                            ));
                        }
                    }
                }
            }
            // `assert <cond>` at module-level: typecheck the condition.
            Node::Assert { cond, .. } => {
                if let Err(e) = infer_expr(cond, &tenv) {
                    errs.push(diag_from_type_err(src, file, e));
                }
            }
            // A Node::Block at module level is the unwrapped body of a
            // `module NAME { ... }` declaration (Phase 10.5). Walk its
            // statements as if they were at top level.
            Node::Block { stmts, .. } => {
                // Each inner item is checked in isolation (`items: vec![inner]`),
                // which strips its sibling module-level declarations. Pre-collect
                // the whole block's declaration names (fn / const / let / struct /
                // enum / type-alias / extern) and inject them into the env so that
                // the per-item recursion — and the #23 resolve pass it runs — sees
                // intra-module fn calls (`reduce(&buf)` calling a sibling `fn
                // reduce`) as resolvable rather than undefined. Sound: these are
                // genuine declarations; injecting them only adds resolvable names.
                // A single-item block has no siblings, so the decl-collection +
                // env clone below would be pure overhead — skip it and pass the
                // env straight through. Only multi-item blocks need sibling
                // injection so the #23 resolve pass sees intra-module calls
                // (`reduce(&buf)` → sibling `fn reduce`) as resolvable. Keeps the
                // compile hot path cheap (no per-check clone for the common case).
                if stmts.len() < 2 {
                    for inner in stmts {
                        let inner_module = Module {
                            items: vec![inner.clone()],
                        };
                        errs.extend(check_module_types_in_file(&inner_module, src, file, env));
                    }
                } else {
                    let mut sibling_decls: std::collections::BTreeSet<String> =
                        std::collections::BTreeSet::new();
                    let block_module = Module {
                        items: stmts.clone(),
                    };
                    resolve::collect_decl_names(&block_module, &mut sibling_decls);
                    let mut inner_env = env.clone();
                    for name in &sibling_decls {
                        inner_env
                            .entry(name.clone())
                            .or_insert(ValueType::ScalarI64);
                    }
                    for inner in stmts {
                        let inner_module = Module {
                            items: vec![inner.clone()],
                        };
                        let inner_errs =
                            check_module_types_in_file(&inner_module, src, file, &inner_env);
                        errs.extend(inner_errs);
                    }
                }
            }
            // RFC 0010 Phase J-A: `region { ... }` at module level.
            // Delegate to the recursive region-escape checker so that the
            // diagnostic flows through the structured surface (consistent
            // with the Phase A/B `safety::extern_non_repr_c` pattern).
            #[cfg(feature = "std-surface")]
            Node::Region { .. } => {
                check_region_escapes_in_node(item, src, file, &mut errs);
                check_genref_unchecked_deref(item, src, file, &mut errs);
            }
            // RFC 0012 Phase A — function definitions: check tensor-param shape
            // annotations and recurse into the body with a param-extended env
            // so that `let` bindings inside the function body get the same
            // `shape::*` diagnostics as module-level bindings.
            Node::FnDef {
                params,
                body,
                ret_type,
                type_params,
                span: fn_span,
                ..
            } => {
                // Run the symbolic-conflict scan on the parameter list.
                check_fn_param_shape_conflicts(params, *fn_span, src, file, &mut errs);

                // Build a local env that extends the module env with the
                // function's parameters, mapping each param name to its
                // ValueType (defaulting to ScalarI64 for unsupported anns).
                let mut fn_env = tenv.clone();
                for param in params {
                    let vt = valuetype_from_ann(&param.ty).unwrap_or(ValueType::ScalarI64);
                    fn_env.insert(param.name.clone(), vt);
                }
                // Walk the function body as a mini-module so shape checks on
                // `let` bindings inside the function body fire the same
                // `shape::*` diagnostics.  This reuses the existing module-
                // level walker recursively (same path as Node::Block above).
                let body_module = Module {
                    items: body.clone(),
                };
                // Bug #38 follow-up — record this body's fixed-`bytes[N]` local
                // bindings so a buffer flowed through an alias (`let buf:
                // bytes[32] = …; take_bytes(buf)`) is caught by the same E2006
                // guard as the direct `bytes[N].zero()` form. Restored on drop.
                let _fixed_bytes_guard = {
                    let mut locals = BTreeSet::new();
                    collect_fixed_bytes_locals(body, &mut locals);
                    (!locals.is_empty()).then(|| FixedBytesLocalsGuard::install(locals))
                };
                let body_errs = check_module_types_in_file(&body_module, src, file, &fn_env);
                // RFC 0012 Phase A is PURELY ADDITIVE: this recursion exists
                // solely to fire `shape::*` diagnostics on tensor `let`
                // bindings inside fn bodies. It must NOT contribute generic
                // type errors — the fn body is already type-checked by the
                // main pass, and the mini-module env here defaults non-tensor
                // params (e.g. `&Point` refs) to ScalarI64, which produces
                // false positives on `&expr` / match-block / struct-arg
                // constructs that have nothing to do with shapes. Keep only
                // shape diagnostics so a valid program never fails to compile
                // because of Phase A. (Regression fix 2026-05-22: this
                // recursion previously extended ALL body errors, breaking
                // ref-expr + match-block parsing tests in parse_match_and_ref.)
                // Keep shape diagnostics (RFC 0012 Phase A) AND the precise
                // implicit-narrowing diagnostic (E2004). Narrowing is a
                // data-loss miscompile that must surface inside fn bodies, and
                // unlike the generic mismatch path it does not false-positive
                // on ref/match constructs (it fires only when both the
                // declared and inferred types are integer scalars of known
                // width).
                // Also keep the intra-module call arity diagnostic (E2005, RFC
                // 0005 Phase B) and the non-exhaustive-match diagnostic. Like
                // narrowing, each fires only on a concrete, sound condition
                // (wrong argument count; an enum match missing a variant with no
                // wildcard) — never on the ref/match-binding/struct-arg
                // constructs that made the generic mismatch path unsafe inside
                // bodies — so admitting them surfaces these soundness holes
                // where most calls and matches actually live (fn bodies).
                // Bug #38 — the fixed-buffer-into-`bytes`-vec diagnostic (E2006)
                // joins this whitelist for the same reason as arity: it fires
                // ONLY on the narrow, concrete `bytes[N]`-arg + `bytes`-param
                // pairing (gated by `arg_is_fixed_bytes_buffer`), never on the
                // ref/match-binding/struct-arg constructs that made the generic
                // mismatch path unsafe inside bodies. Most such calls live in fn
                // bodies, so without this the silent miscompile would slip
                // through exactly where it occurs.
                errs.extend(body_errs.into_iter().filter(|d| {
                    is_shape_diag_code(d.code)
                        || d.code == NARROWING_CODE
                        || d.code == CALL_ARITY_CODE
                        || d.code == MATCH_NONEXHAUSTIVE_CODE
                        || d.code == MATCH_ARM_MISMATCH_CODE
                        || d.code == FIXED_BYTES_INTO_VEC_CODE
                        || d.code == RETURN_TYPE_MISMATCH_CODE
                        || d.code == COND_TYPE_MISMATCH_CODE
                        || d.code == MIXED_CLASS_BINOP_CODE
                        || d.code == LET_CLASS_MISMATCH_CODE
                        || d.code == AS_BOOL_CODE
                }));

                // Early check-phase diagnostics that fail closed LATE at
                // `mlir-opt` otherwise: a float-class value returned from an
                // int-class fn (E2010) and a float-class `if`/`while` condition
                // (E2011). Both need the enclosing fn's declared return type /
                // condition position in scope, which the mini-module recursion
                // above does not carry, so they run here directly against
                // `fn_env` (params + module symbols). Sound-condition only —
                // see RETURN_TYPE_MISMATCH_CODE / COND_TYPE_MISMATCH_CODE.
                let ret_vt = ret_type.as_ref().and_then(valuetype_from_ann);
                check_return_and_cond_types(body, ret_vt.as_ref(), &fn_env, src, file, &mut errs);

                // Confidence-gated scalar-class checks (RFC 0011 — no implicit
                // int↔float coercion): E2010 float-return + int-value direction,
                // E2013 mixed-class binop, E2015 let/assign class mismatch,
                // E2016 numeric `as bool`. Purely syntactic (never `infer_expr`),
                // seeded with the fn's scalar params — fires only on
                // annotation/literal-derived mismatches.
                let mut class_ctx = ClassCtx::default();
                for param in params {
                    if let Some(c) = scalar_class_of_ann(&param.ty) {
                        class_ctx.classes.insert(param.name.clone(), c);
                    }
                }
                let ret_class = ret_type.as_ref().and_then(scalar_class_of_ann);
                check_scalar_classes(body, ret_class, &mut class_ctx, src, file, &mut errs);

                // Issue #23 — scoped name resolution for the fn body. Replaces
                // the dropped unknown-identifier / undefined-call diagnostics
                // (the mini-module filter above keeps only shape + narrowing
                // codes) with a purpose-built pass that unions all symbol
                // sources and tracks nested scope, so it reports E2002/E2003
                // only for genuinely-unresolvable references. The injected set
                // is the module env's keys (module fns/consts/lets + any
                // cross-module symbols merged by the import resolver); passing
                // them can only make MORE names resolve, never false-positive.
                let mut injected: BTreeSet<String> = tenv.keys().cloned().collect();
                // Symbolic tensor shape-dimension identifiers declared in the
                // parameter and return types (`fn f(x: tensor<f32[batch, 32]>)
                // -> tensor<f32[batch, 10]>`) are in scope throughout the body
                // wherever the shape is referenced (`reshape(x, [batch, 800])`).
                // Pre-bind them so they resolve; numeric dims (`32`, `800`) are
                // literals and are skipped. This only adds names, never removes,
                // so it cannot turn a genuine undefined ident into a pass.
                for param in params {
                    resolve::collect_shape_vars(&param.ty, &mut injected);
                }
                if let Some(rt) = ret_type {
                    resolve::collect_shape_vars(rt, &mut injected);
                }
                // Generic type-parameter names declared as `fn id<T, U>(...)` are
                // in scope throughout the body wherever the type is referenced as
                // a value (e.g. `T::default()`, `size_of::<T>()`, or a `Named("T")`
                // shape var). Without this, a generic body false-positives E2002
                // on its own type params. Like shape vars, this only ADDS names —
                // it can never turn a genuine undefined ident into a pass.
                for tp in type_params {
                    injected.insert(tp.clone());
                }
                let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                for u in resolve::resolve_fn_body(body, &param_names, module, &injected) {
                    let (msg, code) = if let Some(enum_name) = &u.variant_of {
                        // Unknown variant of a locally-declared enum. `u.name` is
                        // the full `Enum::Variant` path; show just the variant.
                        let variant = u
                            .name
                            .rsplit_once("::")
                            .map(|(_, v)| v)
                            .unwrap_or(u.name.as_str());
                        let hint = match &u.suggestion {
                            Some(s) => format!(" — did you mean `{s}`?"),
                            None => String::new(),
                        };
                        (
                            format!("unknown variant `{variant}` of enum `{enum_name}`{hint}"),
                            resolve::UNKNOWN_VARIANT_CODE,
                        )
                    } else if u.undeclared_assign {
                        let hint = match &u.suggestion {
                            Some(s) => format!(" — did you mean `{s}`?"),
                            None => String::new(),
                        };
                        (
                            format!("assignment to undeclared variable `{}`{hint}", u.name),
                            resolve::UNDECLARED_ASSIGN_CODE,
                        )
                    } else if u.fn_value_call {
                        (
                            format!(
                                "cannot call a function value `{}`: first-class functions are not yet supported",
                                u.name
                            ),
                            resolve::FN_VALUE_CALL_CODE,
                        )
                    } else if u.is_call {
                        (
                            format!("unsupported call to `{}`", u.name),
                            resolve::UNKNOWN_CALL_CODE,
                        )
                    } else {
                        let msg = match &u.suggestion {
                            Some(s) => {
                                format!("unknown identifier `{}` — did you mean `{}`?", u.name, s)
                            }
                            None => format!("unknown identifier `{}`", u.name),
                        };
                        (msg, resolve::UNKNOWN_IDENT_CODE)
                    };
                    errs.push(diag_from_span(src, file, msg, u.span, code));
                }

                // RFC 0010 Phase J-A/B: safety passes over the fn node.
                #[cfg(feature = "std-surface")]
                check_region_escapes_in_node(item, src, file, &mut errs);
                #[cfg(feature = "std-surface")]
                check_genref_unchecked_deref(item, src, file, &mut errs);
            }
            other => {
                if let Err(e) = infer_expr(other, &tenv) {
                    errs.push(diag_from_type_err(src, file, e));
                }
                // RFC 0010 Phase J-A: recurse into fn bodies and other
                // compound nodes to find embedded `region { }` blocks and
                // emit `safety::region_escape` diagnostics for any that
                // return a region-interior allocation.
                #[cfg(feature = "std-surface")]
                check_region_escapes_in_node(other, src, file, &mut errs);
                // RFC 0010 Phase J-B: flag `gen_deref` results used without
                // a zero-check guard (`safety::genref_unchecked_deref`).
                #[cfg(feature = "std-surface")]
                check_genref_unchecked_deref(other, src, file, &mut errs);
            }
        }
    }

    // RFC 0012 Phase A — second pass: check call-site symbolic dim consistency.
    // Walk all items in the module; any Node::Call whose callee is in fn_tensor_sigs
    // gets the unification check.  This pass is O(items × depth) and runs after
    // the main loop so that function definitions are fully registered in tenv first.
    if !fn_tensor_sigs.is_empty() {
        for item in &module.items {
            walk_calls_for_symbolic_check(item, &fn_tensor_sigs, &tenv, src, file, &mut errs);
        }
    }

    // RFC 0012 Phase C.1 — function-annotation checks: `[deterministic]`,
    // `[target(...)]`, and `[q16]`, reading the attribute list recorded inert
    // by Phase C.0. Purely additive — only functions that opt in are checked;
    // un-annotated code and foreign attributes are untouched.
    check_determinism_annotations(module, src, file, &mut errs);

    errs
}

/// RFC 0010 Phase J-A — `region { }` escape checker.
///
/// Walks `node` recursively to find every `Node::Region` and checks whether
/// its result expression is a region-interior allocation (a direct call to a
/// known allocating function, or an identifier bound to such a call).  When
/// an escape is detected, a `safety::region_escape` diagnostic is appended to
/// `errs` using the same `diag_from_span` surface as Phase A/B.
///
/// Conservative (Phase J-A): only the direct-return and simple-binding cases
/// are flagged.  Aliasing through struct fields is Phase J-B.  Scalar results
/// (i64 / f64 / bool literals and arithmetic) are never flagged.
#[cfg(feature = "std-surface")]
fn check_region_escapes_in_node(
    node: &Node,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        Node::Region { body, span } => {
            // Collect names bound to known allocating calls inside the body.
            // Known allocating functions: __mind_alloc and any *_new constructor.
            let mut alloc_bound: std::collections::BTreeSet<String> = Default::default();
            for stmt in body {
                if let Node::Let { name, value, .. } = stmt {
                    if is_allocating_call(value) {
                        alloc_bound.insert(name.clone());
                    }
                }
                // Recurse into nested regions within the body.
                check_region_escapes_in_node(stmt, src, file, errs);
            }
            // Check the result expression (the last item in body).
            if let Some(last) = body.last() {
                let escapes = match last {
                    Node::Lit(Literal::Ident(name), _) => alloc_bound.contains(name.as_str()),
                    other => is_allocating_call(other),
                };
                if escapes {
                    errs.push(diag_from_span(
                        src,
                        file,
                        "region-interior allocation escapes region block \
                         (RFC 0010 §3.2); the pointer will be freed at \
                         region exit — use a scalar result or copy the data \
                         out before the region closes"
                            .to_string(),
                        *span,
                        "safety::region_escape",
                    ));
                }
            }
        }
        // Recurse into fn bodies.
        Node::FnDef { body, .. } => {
            for stmt in body {
                check_region_escapes_in_node(stmt, src, file, errs);
            }
        }
        // Recurse into block nodes (unwrapped module declarations).
        Node::Block { stmts, .. } => {
            for stmt in stmts {
                check_region_escapes_in_node(stmt, src, file, errs);
            }
        }
        // Recurse into let bindings whose RHS might contain a region.
        Node::Let { value, .. } => {
            check_region_escapes_in_node(value, src, file, errs);
        }
        _ => {}
    }
}

/// Return `true` when `node` is a direct call to a known heap-allocating
/// function.  The set covers `__mind_alloc` and all `*_new` constructors
/// that return a heap handle (`vec_new`, `string_new`, `map_new`, etc.).
/// Scalars and arithmetic nodes always return `false`.
#[cfg(feature = "std-surface")]
fn is_allocating_call(node: &Node) -> bool {
    if let Node::Call { callee, .. } = node {
        callee == "__mind_alloc" || callee.ends_with("_new")
    } else {
        false
    }
}

/// RFC 0010 Phase J-B — GenRef unchecked-deref checker.
///
/// Walks `node` recursively to find every `gen_deref(handle)` call that is
/// used without a zero-check guard, and emits `safety::genref_unchecked_deref`
/// through the `diag_from_span` channel (NEVER via `eprintln!`).
///
/// The safe patterns are:
/// ```mind
/// let p = gen_deref(r)
/// match p { 0 => { /* dangling */ }, _ => { /* live */ } }
/// ```
/// ```mind
/// let p = gen_deref(r)
/// if p == 0 { /* dangling */ }
/// ```
///
/// The unsafe pattern — flagged — is a `let name = gen_deref(…)` binding
/// whose bound name is NOT the direct scrutinee of the immediately following
/// `match` or `if` in the same statement sequence.  A bare expression-statement
/// `gen_deref(…)` (not bound to any name) is also always flagged.
///
/// Conservative Phase J-B scope:
///   - `let x = gen_deref(r)` without a guard on `x` in the next stmt → flag.
///   - `gen_deref(r)` as a bare statement expression → flag.
///   - Does NOT perform full data-flow analysis across multiple bindings or
///     function returns.  Full aliasing is Phase J-C.
///
/// Emits through `diag_from_span` exactly like `safety::region_escape` and
/// `safety::extern_non_repr_c`.
#[cfg(feature = "std-surface")]
fn check_genref_unchecked_deref(
    node: &Node,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        // Walk function bodies: analyse each statement sequence then recurse.
        Node::FnDef { body, .. } => {
            check_genref_in_stmt_seq(body, src, file, errs);
            for stmt in body {
                check_genref_unchecked_deref(stmt, src, file, errs);
            }
        }
        Node::Block { stmts, .. } => {
            check_genref_in_stmt_seq(stmts, src, file, errs);
            for stmt in stmts {
                check_genref_unchecked_deref(stmt, src, file, errs);
            }
        }
        Node::Region { body, .. } => {
            check_genref_in_stmt_seq(body, src, file, errs);
            for stmt in body {
                check_genref_unchecked_deref(stmt, src, file, errs);
            }
        }
        // Let bindings are handled by check_genref_in_stmt_seq above; do NOT
        // recurse here or we would double-count / misattribute the diagnostic.
        // Recursing into if/match/while bodies for nested fn-level seq checks.
        Node::If {
            then_branch,
            else_branch,
            ..
        } => {
            check_genref_in_stmt_seq(then_branch, src, file, errs);
            for stmt in then_branch {
                check_genref_unchecked_deref(stmt, src, file, errs);
            }
            if let Some(eb) = else_branch {
                check_genref_in_stmt_seq(eb, src, file, errs);
                for stmt in eb {
                    check_genref_unchecked_deref(stmt, src, file, errs);
                }
            }
        }
        _ => {}
    }
}

/// Check a statement sequence for unguarded `gen_deref` patterns.
///
/// Two cases are handled:
///
/// 1. `let name = gen_deref(…)` where the next statement is NOT an
///    `if`/`match` whose scrutinee is `name` → emit diagnostic.
///
/// 2. A bare expression-statement that IS a `gen_deref(…)` call (not bound
///    to any name, so no guard is possible) → emit diagnostic.
///
/// Iterates in O(n) — one pass over the statement list.
#[cfg(feature = "std-surface")]
fn check_genref_in_stmt_seq(stmts: &[Node], src: &str, file: Option<&str>, errs: &mut Vec<Pretty>) {
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            // Case 1: let binding of a gen_deref result.
            Node::Let {
                name, value, span, ..
            } if is_genderef_call(value) => {
                let next = stmts.get(i + 1);
                let guarded = next.is_some_and(|n| is_zero_guard_on(n, name));
                if !guarded {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "`{name}` is the result of `gen_deref` and is used without \
                             a zero-check guard (RFC 0010 §3.3); the allocation may \
                             have been freed — add `match {name} \
                             {{ 0 => {{ /* dangling */ }}, _ => {{ /* live */ }} }}`"
                        ),
                        *span,
                        "safety::genref_unchecked_deref",
                    ));
                }
            }
            // Case 2: bare gen_deref expression statement — no guard possible.
            Node::Call { callee, span, .. } if callee == "gen_deref" => {
                errs.push(diag_from_span(
                    src,
                    file,
                    "result of `gen_deref` is used without a zero-check guard \
                     (RFC 0010 §3.3); the allocation may have been freed — \
                     bind the result and check for 0: \
                     `match gen_deref(r) { 0 => { /* dangling */ }, p => { /* live */ } }`"
                        .to_string(),
                    *span,
                    "safety::genref_unchecked_deref",
                ));
            }
            _ => {}
        }
    }
}

/// Return `true` when `node` is a direct `gen_deref(…)` call.
#[cfg(feature = "std-surface")]
fn is_genderef_call(node: &Node) -> bool {
    matches!(node, Node::Call { callee, .. } if callee == "gen_deref")
}

/// Return `true` when `node` is an `if`/`match` whose scrutinee is `binding`.
///
/// Covers:
///   - `match <binding> { … }` — the scrutinee is `Lit(Ident(binding))`.
///   - `if <binding> { … }` / `if <binding> == 0 { … }` — the condition
///     starts with an ident equal to `binding`.
///
/// This is a conservative surface-level check: it does not track aliases or
/// evaluate the condition deeply.  Phase J-C can refine this.
#[cfg(feature = "std-surface")]
fn is_zero_guard_on(node: &Node, binding: &str) -> bool {
    match node {
        Node::Match { scrutinee, .. } => is_ident_node(scrutinee, binding),
        Node::If { cond, .. } => is_ident_or_cmp_node(cond, binding),
        _ => false,
    }
}

/// Return `true` when `node` is `Lit(Ident(name))` with the given name.
#[cfg(feature = "std-surface")]
fn is_ident_node(node: &Node, name: &str) -> bool {
    matches!(node, Node::Lit(Literal::Ident(n), _) if n == name)
}

/// Return `true` when `node` is an identifier or a binary comparison
/// (`==`, `!=`, `<`, `>`) whose left-hand side is the named identifier.
/// This covers `if p { }`, `if p == 0 { }`, `if p != 0 { }`.
#[cfg(feature = "std-surface")]
fn is_ident_or_cmp_node(node: &Node, name: &str) -> bool {
    if is_ident_node(node, name) {
        return true;
    }
    // Binary: check whether either operand is the named identifier.
    if let Node::Binary { left, right, .. } = node {
        return is_ident_node(left, name) || is_ident_node(right, name);
    }
    false
}

/// RFC 0010 Phase A — verify that a type used in an `extern "C"` signature is
/// a C-ABI-compatible Copy type.
///
/// Accepted: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool, usize,
/// isize, and raw pointers (`*const T` / `*mut T` for any accepted T).
///
/// Rejected: Named user types (String, Vec, custom structs), tensors,
/// slices, references, tuples, generic types, etc.
///
/// Returns `Ok(())` on acceptance, `Err(reason)` on rejection.
/// RFC 0010 Phase B (audit fix F-06): check_extern_type with a registry of
/// known #[repr(C)] struct names.  Named types present in `repr_c` are accepted;
/// all other unknown Named types produce a `safety::extern_non_repr_c` diagnostic.
fn check_extern_type_with_repr_c(
    ann: &TypeAnn,
    repr_c: &std::collections::BTreeSet<String>,
) -> Result<(), String> {
    match ann {
        // All built-in scalar types are C-ABI-compatible.
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool
        | TypeAnn::ScalarU32 => Ok(()),
        // Raw pointers: validate the pointee type recursively. Phase A
        // accepts any built-in pointee; the pointer itself lowers to !llvm.ptr.
        TypeAnn::RawPtr { pointee, .. } => check_extern_type_with_repr_c(pointee, repr_c),
        // RFC 0010 Phase B: callback function pointers `extern "C" fn(T) -> R`
        // are C-ABI-compatible; they lower to !llvm.ptr. Validate that the
        // callback parameter and return types also satisfy Phase B rules.
        TypeAnn::FnPtr { params, ret } => {
            for p in params {
                check_extern_type_with_repr_c(p, repr_c)?;
            }
            if let Some(r) = ret {
                check_extern_type_with_repr_c(r, repr_c)?;
            }
            Ok(())
        }
        // Named types: accept primitives and known repr(C) structs.
        // RFC 0010 Phase B audit fix F-06: unknown Named types are rejected
        // with safety::extern_non_repr_c to prevent silent miscompilation.
        TypeAnn::Named(name) => match name.as_str() {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64"
            | "bool" | "usize" | "isize" => Ok(()),
            other => {
                if repr_c.contains(other) {
                    Ok(())
                } else {
                    Err(format!(
                        "type `{other}` is not a primitive C scalar and is not annotated                          with `#[repr(C)]`; add `#[repr(C)]` to the struct definition or                          use a raw pointer `*const {other}` / `*mut {other}`                          (safety::extern_non_repr_c)"
                    ))
                }
            }
        },
        // Aggregate and non-Copy types are rejected.
        TypeAnn::Tensor { .. }
        | TypeAnn::DiffTensor { .. }
        | TypeAnn::SparseTensor { .. }
        | TypeAnn::Slice { .. }
        | TypeAnn::Ref { .. }
        | TypeAnn::Array { .. }
        | TypeAnn::Tuple { .. }
        | TypeAnn::Generic { .. } => {
            Err("aggregate/non-Copy type is not allowed in `extern \"C\"` \
             signatures (safety::extern_non_copy); use a raw pointer `*const T` \
             or `*mut T` to pass aggregate data across the C ABI"
                .into())
        }
    }
}

// ── RFC 0012 Phase A — shape diagnostic codes ─────────────────────────────────
//
// RFC 0012 Phase B diagnostic codes — tensor operator shape errors.
// All flow through `diag_from_span`, never via `eprintln!`.
const SHAPE_MATMUL_MISMATCH: &str = "shape::matmul_mismatch";
const SHAPE_BROADCAST_MISMATCH: &str = "shape::broadcast_mismatch";

// All shape diagnostics flow through `diag_from_span` exactly like the existing
// `safety::extern_non_repr_c` / `safety::region_escape` diagnostics — NEVER via
// `eprintln!`.  The diagnostic codes are in the `shape::` namespace, compatible
// with the `mindc check` severity model (RFC 0007 §5).

const SHAPE_RANK_MISMATCH: &str = "shape::rank_mismatch";
const SHAPE_DIM_MISMATCH: &str = "shape::dim_mismatch";
const SHAPE_DTYPE_MISMATCH: &str = "shape::dtype_mismatch";
const SHAPE_SYMBOLIC_CONFLICT: &str = "shape::symbolic_conflict";

/// RFC 0012 Phase A — compare two tensor `ValueType`s and push precise
/// `shape::*` diagnostics when they differ.
///
/// Called at `let x: Tensor<...> = rhs` sites where both `ann` and `inferred`
/// are `ValueType::Tensor`.  A generic type-mismatch (`TYPE_ERR_CODE`) is
/// emitted by the existing path when the types are structurally unequal;
/// this function replaces that generic diagnostic with a precise shape one
/// when both sides are tensor types.
///
/// Returns `true` if a shape diagnostic was pushed (caller should skip the
/// generic mismatch path), `false` if the types are compatible or non-tensor
/// (let the existing path handle it).
fn check_tensor_shape_compat(
    ann: &TensorType,
    inferred: &TensorType,
    binding_name: &str,
    src: &str,
    file: Option<&str>,
    span: AstSpan,
    errs: &mut Vec<Pretty>,
) -> bool {
    // Dtype mismatch — checked first; shape comparison is meaningless across dtypes.
    if ann.dtype != inferred.dtype {
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "dtype mismatch for `{}`: annotation `{}` vs inferred `{}`",
                binding_name,
                dtype_name(&ann.dtype),
                dtype_name(&inferred.dtype),
            ),
            span,
            SHAPE_DTYPE_MISMATCH,
        ));
        return true;
    }
    // Rank mismatch — checked before individual dims.
    if ann.shape.len() != inferred.shape.len() {
        errs.push(diag_from_span(
            src,
            file,
            format!(
                "rank mismatch for `{}`: annotation rank {} vs inferred rank {}",
                binding_name,
                ann.shape.len(),
                inferred.shape.len(),
            ),
            span,
            SHAPE_RANK_MISMATCH,
        ));
        return true;
    }
    // Per-dimension mismatch — only for known (concrete) dims; symbolic dims
    // are always compatible with a concrete value at a single binding site
    // (unification happens across the function scope in check_fn_param_shape_conflicts).
    let mut pushed = false;
    for (i, (adim, idim)) in ann.shape.iter().zip(inferred.shape.iter()).enumerate() {
        if let (ShapeDim::Known(an), ShapeDim::Known(inf)) = (adim, idim) {
            if an != inf {
                errs.push(diag_from_span(
                    src,
                    file,
                    format!(
                        "dim mismatch for `{}` at axis {}: annotation {} vs inferred {}",
                        binding_name, i, an, inf,
                    ),
                    span,
                    SHAPE_DIM_MISMATCH,
                ));
                pushed = true;
            }
        }
    }
    pushed
}

/// RFC 0012 Phase A — check symbolic dim consistency at a call site.
///
/// Given `fn_sigs` (function name → Vec<(arg_position, param_name, dims, dtype)>),
/// the concrete argument types (resolved via `infer_expr`), and the callee name,
/// unify the symbolic dim names across arguments.  If the same symbol is bound
/// to two different concrete sizes, push a `shape::symbolic_conflict` diagnostic.
///
/// Example: `fn f(a: Tensor<f32,[N,K]>, b: Tensor<f32,[N,M]>)` called with
/// `f(x, y)` where `x: Tensor<f32,[4,8]>` and `y: Tensor<f32,[8,16]>` → `N`
/// is bound to `4` by `x` and `8` by `y` → `shape::symbolic_conflict`.
#[allow(clippy::too_many_arguments)]
fn check_call_symbolic_dims(
    callee: &str,
    args: &[Node],
    call_span: AstSpan,
    fn_sigs: &FnTensorSigs,
    env: &TypeEnv,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    let Some(tensor_params) = fn_sigs.get(callee) else {
        return;
    };
    // Bind symbol names to concrete sizes by walking (arg_position, tensor_param) entries.
    // sym_env: symbol_name → (bound_concrete_size, first_binding_param_name)
    let mut sym_env: HashMap<String, (usize, String)> = HashMap::new();
    for (arg_pos, param_name, dims, _dtype) in tensor_params {
        // Argument at the declared position — skip if out of range.
        let Some(arg) = args.get(*arg_pos) else {
            continue;
        };
        // Resolve the argument's ValueType to get concrete shape.
        let inferred = match infer_expr(arg, env) {
            Ok((ValueType::Tensor(t), _)) => t,
            _ => continue, // non-tensor arg or inference error: skip
        };
        // Walk each dimension of the param annotation.
        for (pos, dim_str) in dims.iter().enumerate() {
            // Only symbolic dims (non-parseable as usize).
            if dim_str.parse::<usize>().is_ok() {
                continue;
            }
            // Concrete value from the inferred shape at the same position.
            let concrete = match inferred.shape.get(pos) {
                Some(ShapeDim::Known(n)) => *n,
                _ => continue, // symbolic or out-of-bounds in inferred: skip
            };
            match sym_env.get(dim_str) {
                None => {
                    sym_env.insert(dim_str.clone(), (concrete, param_name.clone()));
                }
                Some((bound_size, bound_param)) if *bound_size != concrete => {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "symbolic dim `{sym}` conflict in call to `{callee}`: \
                             bound to {bound} via param `{bp}` but {actual} via param `{pp}`",
                            sym = dim_str,
                            bound = bound_size,
                            bp = bound_param,
                            actual = concrete,
                            pp = param_name,
                        ),
                        call_span,
                        SHAPE_SYMBOLIC_CONFLICT,
                    ));
                }
                _ => {}
            }
        }
    }
}

/// RFC 0012 Phase A — walk a node tree and run `check_call_symbolic_dims`
/// on every `Node::Call` that matches a function in `fn_sigs`.
fn walk_calls_for_symbolic_check(
    node: &Node,
    fn_sigs: &FnTensorSigs,
    env: &TypeEnv,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    match node {
        Node::Call { callee, args, span } => {
            check_call_symbolic_dims(callee, args, *span, fn_sigs, env, src, file, errs);
            for a in args {
                walk_calls_for_symbolic_check(a, fn_sigs, env, src, file, errs);
            }
        }
        Node::FnDef { body, .. } => {
            for stmt in body {
                walk_calls_for_symbolic_check(stmt, fn_sigs, env, src, file, errs);
            }
        }
        Node::Let { value, .. } => {
            walk_calls_for_symbolic_check(value, fn_sigs, env, src, file, errs);
        }
        Node::Return { value: Some(v), .. } => {
            walk_calls_for_symbolic_check(v, fn_sigs, env, src, file, errs);
        }
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            walk_calls_for_symbolic_check(cond, fn_sigs, env, src, file, errs);
            for s in then_branch {
                walk_calls_for_symbolic_check(s, fn_sigs, env, src, file, errs);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    walk_calls_for_symbolic_check(s, fn_sigs, env, src, file, errs);
                }
            }
        }
        Node::Block { stmts, .. } => {
            for s in stmts {
                walk_calls_for_symbolic_check(s, fn_sigs, env, src, file, errs);
            }
        }
        _ => {}
    }
}

// ── RFC 0012 Phase C.1 — function-annotation checks ───────────────────
//
// Reads the attribute list recorded inert on `FnDef` by Phase C.0 and
// enforces the determinism / target / q16 contracts of RFC 0012 §5.1-5.4.
// Purely additive: only functions that opt in via `[deterministic]`,
// `[target(...)]`, or `[q16]` are checked. Un-annotated code and foreign
// attributes are untouched, so no previously-compiling program can fail.

/// Backend-target vocabulary accepted by `[target(...)]`. Mirrors the
/// mindc CLI `parse_target` set (canonical names + documented aliases)
/// plus `q16`, the RFC 0012 §5.3 deterministic fixed-point pseudo-target.
const VALID_TARGET_NAMES: &[&str] = &[
    "cpu",
    "gpu",
    "cuda",
    "rocm",
    "metal",
    "webgpu",
    "tpu",
    "npu",
    "ane",
    "hexagon",
    "lpu",
    "groq",
    "dpu",
    "smartnic",
    "bluefield",
    "fpga",
    "hls",
    "cerebras",
    "wse",
    "wse2",
    "wse3",
    "q16",
];

const DET_NONDET_CODE: &str = "determinism::nondeterministic_in_deterministic";
const DET_UNKNOWN_TARGET_CODE: &str = "determinism::unknown_target";
const DET_FLOAT_IN_Q16_CODE: &str = "determinism::float_in_q16_fn";

/// A function's RFC 0012 annotation state, derived from its `attrs`.
#[derive(Default)]
struct FnAnnot {
    deterministic: bool,
    q16: bool,
}

/// `[q16]` is shorthand for `[deterministic] [target(q16)]` (§5.3), so a
/// `[q16]` function is deterministic by construction.
fn fn_annot(attrs: &[crate::ast::Attribute]) -> FnAnnot {
    let mut a = FnAnnot::default();
    for at in attrs {
        match at.name.as_str() {
            "deterministic" => a.deterministic = true,
            "q16" => {
                a.q16 = true;
                a.deterministic = true;
            }
            _ => {}
        }
    }
    a
}

/// Collect every `Call` callee name (with its span) reachable in `node`.
/// Mirrors the arm coverage of `walk_calls_for_symbolic_check` plus the
/// common expression positions. Under-collection is sound — it can only
/// miss a violation, never invent one.
/// Pre-order walk of a function body: invokes `f` on `node` and every node
/// structurally reachable through the statement/expression positions a fn body
/// uses. The descent is defined ONCE here; the determinism call-graph and the
/// `#[q16]` dtype scans are thin closures over it (under-collection of an
/// unmodelled position is sound — it can only miss a violation, never invent
/// one). Module-level item collection uses `collect_module_fndefs`, which has a
/// different scope (top-level items + module blocks, not fn-body internals).
fn for_each_body_node<'a>(node: &'a Node, f: &mut impl FnMut(&'a Node)) {
    f(node);
    match node {
        Node::Call { args, .. } => {
            for a in args {
                for_each_body_node(a, f);
            }
        }
        Node::MethodCall { receiver, args, .. } => {
            for_each_body_node(receiver, f);
            for a in args {
                for_each_body_node(a, f);
            }
        }
        Node::Binary { left, right, .. } => {
            for_each_body_node(left, f);
            for_each_body_node(right, f);
        }
        Node::Let { value, .. } => for_each_body_node(value, f),
        Node::Return { value: Some(v), .. } => for_each_body_node(v, f),
        Node::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            for_each_body_node(cond, f);
            for s in then_branch {
                for_each_body_node(s, f);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    for_each_body_node(s, f);
                }
            }
        }
        Node::Block { stmts, .. } => {
            for s in stmts {
                for_each_body_node(s, f);
            }
        }
        _ => {}
    }
}

/// Collect every `Call` callee name (with its span) reachable in `node`.
fn collect_call_targets(node: &Node, out: &mut Vec<(String, AstSpan)>) {
    for_each_body_node(node, &mut |n| {
        if let Node::Call { callee, span, .. } = n {
            out.push((callee.clone(), *span));
        }
    });
}

/// RFC 0012 §5.1 pt 3 (Phase C.2) — implicit determinism of a callee that is
/// NOT a user-defined function in the current module (std / imported /
/// intrinsic). Keyed on the dtype-suffix convention shared by std.blas and the
/// tensor surface:
///   `Some(true)`  — implicitly deterministic: `_q16` is the Q16.16
///                   byte-identical fixed-point path; `__mind_*` are integer
///                   intrinsics. A `#[deterministic]` caller may use these.
///   `Some(false)` — implicitly NON-deterministic: `_f32`/`_f64` floating
///                   reductions are not order-fixed under SIMD.
///   `None`        — unknown surface: conservatively NOT flagged.
/// Builtins that are nondeterministic by construction: PRNG draws (their result
/// depends on hidden entropy/seed state) and wall-clock / IO reads (observe the
/// world). A `#[deterministic]` function may not call any of these, so they map
/// to `Some(false)` even though they carry no `_f32`/`_f64` dtype suffix — the
/// suffix heuristic alone let `random`/`rand_*` slip through as `None` (unknown,
/// not flagged). Matched on the bare callee name AND the `std.rand.*` style
/// dotted path tail, so `random`, `std.rand.random`, and `rng.rand_uniform`
/// (recv.method desugar) all resolve to the same nondeterministic verdict.
const NONDETERMINISTIC_BUILTINS: &[&str] = &[
    // PRNG draws.
    "random",
    "rand",
    "rand_normal",
    "rand_uniform",
    "rand_int",
    "rand_range",
    "rand_bytes",
    "rand_seed",
    "randn",
    "shuffle",
    // Wall-clock / nondeterministic environment reads.
    "now",
    "time_now",
    "system_time",
    "monotonic_now",
    "read_line",
    "read_input",
];

fn implicit_external_determinism(callee: &str) -> Option<bool> {
    // A nondeterministic builtin is flagged regardless of any dtype suffix.
    // Match the bare name or the last dotted/`::`-qualified path segment so a
    // qualified call (`std.rand.random`, `rng::rand_uniform`) is caught too.
    let tail = callee.rsplit(['.', ':']).next().unwrap_or(callee);
    if NONDETERMINISTIC_BUILTINS.contains(&tail) || NONDETERMINISTIC_BUILTINS.contains(&callee) {
        return Some(false);
    }
    if callee.starts_with("__mind_") || callee.contains("_q16") {
        Some(true)
    } else if callee.contains("_f32") || callee.contains("_f64") {
        Some(false)
    } else {
        None
    }
}

/// Collect every `FnDef` node reachable as a top-level item or nested inside a
/// `module NAME { ... }` block. Module blocks parse to a transparent
/// `Node::Block` item (the module name is not a scope for call resolution —
/// the rest of the compiler treats it as a flat item list), so determinism
/// checks descend into them. Returns the `FnDef` nodes in source order.
fn collect_module_fndefs<'a>(items: &'a [Node], out: &mut Vec<&'a Node>) {
    for item in items {
        match item {
            Node::FnDef { .. } => out.push(item),
            Node::Block { stmts, .. } => collect_module_fndefs(stmts, out),
            _ => {}
        }
    }
}

/// Collect `(name, dtype, span)` for every `let` binding with a tensor type
/// annotation reachable in `node` (recursing into if/block bodies). Used by the
/// `#[q16]` dtype check to extend coverage from params/return to body locals.
fn collect_let_tensor_dtypes<'a>(node: &'a Node, out: &mut Vec<(&'a str, &'a str, AstSpan)>) {
    for_each_body_node(node, &mut |n| {
        if let Node::Let {
            name,
            ann: Some(TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. }),
            span,
            ..
        } = n
        {
            out.push((name.as_str(), dtype.as_str(), *span));
        }
    });
}

/// RFC 0012 Phase C.1/C.2 — enforce the function-annotation contracts.
///
/// Visits every `FnDef` at the top level AND inside `module { }` blocks
/// (Phase C.2 closed the prior module-block coverage gap). Purely additive:
/// only functions that opt in via annotations are checked.
fn check_determinism_annotations(
    module: &Module,
    src: &str,
    file: Option<&str>,
    errs: &mut Vec<Pretty>,
) {
    // Flatten all FnDefs (top-level + module-block-nested) once.
    let mut fndefs: Vec<&Node> = Vec::new();
    collect_module_fndefs(&module.items, &mut fndefs);

    // Pre-scan: every user-defined function's annotation state, so the
    // determinism call-graph check can resolve callees within this module.
    let mut annots: HashMap<&str, FnAnnot> = HashMap::new();
    for item in &fndefs {
        if let Node::FnDef { name, attrs, .. } = item {
            annots.insert(name.as_str(), fn_annot(attrs));
        }
    }

    for item in &fndefs {
        let Node::FnDef {
            name,
            params,
            ret_type,
            body,
            attrs,
            span,
            ..
        } = item
        else {
            continue;
        };
        let a = fn_annot(attrs);

        // Check 0 — float-bearing tuple return fail-loud. An all-i64 tuple is a
        // heap aggregate (`__mind_alloc(8*n)` + per-slot `__mind_store_i64`)
        // returned by its base pointer (i64) — exactly like a `#[repr(C)]` struct
        // return — and the caller takes it apart with `let (a, b) = …`, so that
        // case is legal. But the aggregate stores/loads every slot AS i64, so a
        // FLOAT element (`f32`/`f64`) would have its bits reinterpreted; reject
        // that case with a clear diagnostic (it otherwise surfaces as a cryptic
        // `non-i64 argument to __mind_store_i64` at MLIR build). Nested tuples and
        // struct elements are themselves heap pointers (i64), so they are safe.
        #[cfg(not(feature = "std-surface-experimental"))]
        if let Some(TypeAnn::Tuple { elements }) = ret_type {
            let has_float = elements.iter().any(|e| {
                matches!(e, TypeAnn::ScalarF32 | TypeAnn::ScalarF64)
                    || matches!(e, TypeAnn::Named(n) if n == "f32" || n == "f64")
            });
            if has_float {
                errs.push(diag_from_span(
                    src,
                    file,
                    format!(
                        "`{name}` returns a tuple containing a float element; the \
                         all-i64 tuple ABI stores every slot as `i64`, which would \
                         reinterpret the float's bits. Return a `#[repr(C)]` struct \
                         (passed by address) instead, or build with the \
                         `std-surface-experimental` feature \
                         (safety::tuple_return_unsupported)"
                    ),
                    *span,
                    "safety::tuple_return_unsupported",
                ));
            }
        }

        // Check 1 — `[target(...)]` / `[q16]` name validity. Local, no call
        // graph: a malformed or unknown backend name is unambiguously wrong.
        for at in attrs {
            if at.name == "target" {
                match at.args.first() {
                    None => errs.push(diag_from_span(
                        src,
                        file,
                        "`[target(...)]` requires a backend name, e.g. `[target(cpu)]`".to_string(),
                        *span,
                        DET_UNKNOWN_TARGET_CODE,
                    )),
                    Some(t) if !VALID_TARGET_NAMES.contains(&t.as_str()) => {
                        errs.push(diag_from_span(
                            src,
                            file,
                            format!(
                                "unknown target `{t}` on `{name}` (expected one of \
                                 cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras|q16)"
                            ),
                            *span,
                            DET_UNKNOWN_TARGET_CODE,
                        ));
                    }
                    _ => {}
                }
            }
        }

        // Check 2 — a `[q16]` function may only declare q16 tensors. Checked
        // on declared parameter and return types (sound: declared types are
        // unambiguous; expression-level dtype tracking is Phase C.2).
        if a.q16 {
            for p in params {
                if let TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. } = &p.ty {
                    if dtype != "q16" {
                        errs.push(diag_from_span(
                            src,
                            file,
                            format!(
                                "parameter `{}` of `#[q16]` fn `{name}` is `Tensor<{dtype},..>`; \
                                 a `#[q16]` function requires q16 tensors",
                                p.name
                            ),
                            p.span,
                            DET_FLOAT_IN_Q16_CODE,
                        ));
                    }
                }
            }
            if let Some(TypeAnn::Tensor { dtype, .. } | TypeAnn::DiffTensor { dtype, .. }) =
                ret_type
            {
                if dtype != "q16" {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "`#[q16]` fn `{name}` returns `Tensor<{dtype},..>`; a `#[q16]` \
                             function requires q16 tensors"
                        ),
                        *span,
                        DET_FLOAT_IN_Q16_CODE,
                    ));
                }
            }
            // C.2+: body-local `let` bindings with a tensor annotation must also
            // be q16 (declared types are unambiguous; full expression-level
            // dtype inference is future work).
            let mut lets = Vec::new();
            for stmt in body {
                collect_let_tensor_dtypes(stmt, &mut lets);
            }
            for (lname, dtype, lspan) in lets {
                if dtype != "q16" {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "`let {lname}` in `#[q16]` fn `{name}` is `Tensor<{dtype},..>`; \
                             a `#[q16]` function requires q16 tensors"
                        ),
                        lspan,
                        DET_FLOAT_IN_Q16_CODE,
                    ));
                }
            }
        }

        // Check 3 — a `[deterministic]` function may only call functions that
        // are themselves deterministic. A callee that resolves to a user fn in
        // THIS module is deterministic iff it is annotated. A callee outside
        // the module (std / imported / intrinsic) is judged by the implicit
        // predicate (RFC §5.1 pt 3): `_q16`/`__mind_*` are deterministic;
        // `_f32`/`_f64` floating reductions are not; anything else is unknown
        // and conservatively NOT flagged (no false positives).
        if a.deterministic {
            let mut calls = Vec::new();
            for stmt in body {
                collect_call_targets(stmt, &mut calls);
            }
            for (callee, cspan) in calls {
                let (nondeterministic, hint) = match annots.get(callee.as_str()) {
                    Some(an) if !an.deterministic => {
                        (true, "annotate it `#[deterministic]` or remove the call")
                    }
                    Some(_) => (false, ""),
                    None => match implicit_external_determinism(&callee) {
                        Some(false) => {
                            // PRNG / wall-clock / IO builtins get a precise hint;
                            // float reductions keep the q16-variant guidance.
                            let nondet_tail =
                                callee.rsplit(['.', ':']).next().unwrap_or(callee.as_str());
                            let hint = if NONDETERMINISTIC_BUILTINS.contains(&nondet_tail)
                                || NONDETERMINISTIC_BUILTINS.contains(&callee.as_str())
                            {
                                "it is nondeterministic (PRNG / wall-clock / IO) — \
                                 a `#[deterministic]` fn cannot draw entropy or read the world"
                            } else {
                                "its floating-point reductions are not deterministic — \
                                 use the q16 variant or remove the call"
                            };
                            (true, hint)
                        }
                        _ => (false, ""),
                    },
                };
                if nondeterministic {
                    errs.push(diag_from_span(
                        src,
                        file,
                        format!(
                            "`{name}` is `#[deterministic]` but calls non-deterministic \
                             `{callee}`; {hint}"
                        ),
                        cspan,
                        DET_NONDET_CODE,
                    ));
                }
            }
        }
    }
}

/// RFC 0012 Phase A — check that symbolic dim names are used consistently
/// across a function's parameter list.
///
/// Phase A scope: purely annotation-level structural check. Call-site
/// symbolic unification (`shape::symbolic_conflict` from concrete args)
/// is handled via `check_call_symbolic_dims` + `walk_calls_for_symbolic_check`.
fn check_fn_param_shape_conflicts(
    _params: &[crate::ast::Param],
    _fn_span: AstSpan,
    _src: &str,
    _file: Option<&str>,
    _errs: &mut Vec<Pretty>,
) {
    // Phase A: annotation-level structural conflict detection is deferred.
    // Concrete call-site conflicts are caught by check_call_symbolic_dims.
    // Full const-generic annotation arithmetic is Phase A-extended (§10.1).
}

fn diag_from_span(
    src: &str,
    file: Option<&str>,
    msg: String,
    span: AstSpan,
    code: &'static str,
) -> Pretty {
    let span = Span::from_offsets(src, span.start(), span.end(), file);
    Pretty {
        phase: "type-check",
        code,
        severity: Severity::Error,
        message: msg,
        span: Some(span),
        notes: Vec::new(),
        help: None,
    }
}

fn diag_from_type_err(src: &str, file: Option<&str>, err: TypeErrSpan) -> Pretty {
    let code = classify_error_code(&err.msg);
    diag_from_span(src, file, err.msg, err.span, code)
}

fn classify_error_code(msg: &str) -> &'static str {
    // RFC 0012 Phase B: the `@` operator gets the rule-id matmul-mismatch code.
    // Match the literal `` `@` `` marker that every `@`-operator diagnostic
    // carries, NOT a bare "matmul" substring — the legacy `tensor.matmul(a,b)`
    // intrinsic must keep its original E2103 inner-dimension code (it falls
    // through to the generic branch below).
    if msg.contains("`@`") && msg.contains("inner dimension") {
        SHAPE_MATMUL_MISMATCH
    } else if msg.contains("elementwise") && msg.contains("broadcast") {
        SHAPE_BROADCAST_MISMATCH
    } else if msg.contains("inner") && msg.contains("dimension") {
        SHAPE_INNER_DIM_CODE
    } else if msg.contains("broadcast") {
        SHAPE_BROADCAST_CODE
    } else if msg.contains("rank mismatch") {
        SHAPE_RANK_CODE
    } else if msg.starts_with("function `") && msg.contains("argument(s); got") {
        // Intra-module call arity mismatch (RFC 0005 Phase B).
        CALL_ARITY_CODE
    } else if msg.starts_with("non-exhaustive `match`") {
        // Non-exhaustive enum match.
        MATCH_NONEXHAUSTIVE_CODE
    } else if msg.starts_with("match arm type class mismatch") {
        // Finding 19: cross-class arm mismatch (int tag vs float payload).
        MATCH_ARM_MISMATCH_CODE
    } else if msg.contains("fixed-size `bytes[N]` buffer handle") {
        // Bug #38: fixed buffer flowing into a growable `bytes` parameter.
        FIXED_BYTES_INTO_VEC_CODE
    } else {
        TYPE_ERR_CODE
    }
}

#[cfg(test)]
mod did_you_mean_tests {
    use super::*;

    #[test]
    fn closest_identifier_suggests_near_typo() {
        let mut env: TypeEnv = HashMap::new();
        env.insert("count".to_string(), ValueType::ScalarI32);
        env.insert("amount".to_string(), ValueType::ScalarI32);
        // one-char typo -> the close name
        assert_eq!(closest_identifier("cout", &env).as_deref(), Some("count"));
        // nothing close enough -> no suggestion
        assert_eq!(closest_identifier("zzzzz", &env), None);
    }

    #[test]
    fn closest_identifier_is_deterministic_on_ties() {
        // Two equidistant candidates: the suggestion must be the lexicographically
        // smaller one regardless of HashMap iteration order (wedge: deterministic).
        let mut env: TypeEnv = HashMap::new();
        env.insert("bat".to_string(), ValueType::ScalarI32);
        env.insert("cat".to_string(), ValueType::ScalarI32);
        // "aat" is distance 1 from both "bat" and "cat"; "bat" < "cat".
        for _ in 0..20 {
            assert_eq!(closest_identifier("aat", &env).as_deref(), Some("bat"));
        }
    }
}
