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

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::ir::{BinOp, IRModule, Instr, ValueId};
use crate::opt::ir_canonical::canonicalize_module;
use crate::types::{ConvPadding, DType, ShapeDim};

/// RFC 0006 Track B (increment 1) — the pure-MIND surface name whose
/// `Instr::Call` lowers to a native MLIR `vector`-dialect reduction loop
/// instead of a `func.call` to the Track A runtime-support C bridge.
/// Declared unconditionally so the default-build catch-all stays
/// byte-identical; only the gated `Instr::Call` arm ever compares it.
#[cfg(feature = "std-surface")]
const VEC_DOT_F32_INTRINSIC: &str = "__mind_blas_dot_f32_v";

/// RFC 0006 Track B — the statically-known SIMD lane count emitted by the
/// `dot_f32_v` lowering. Eight f32 lanes is the AVX2 / NEON-pair width;
/// LLVM legalises wider/narrower targets from the same `vector<8xf32>`.
#[cfg(feature = "std-surface")]
const VEC_DOT_F32_LANES: usize = 8;

/// RFC 0006 Track B (increment 2) — the pure-MIND surface name whose
/// `Instr::Call` lowers to a native MLIR `vector`-dialect Q16.16
/// reduction. Byte-identical to the Track A scalar oracle
/// `__mind_blas_dot_q16` at every length (task #57 — integer reduction
/// is associative, the per-element arithmetic `>> 16` is replicated
/// exactly in `vector<8xi64>` lanes).
#[cfg(feature = "std-surface")]
const VEC_DOT_Q16_INTRINSIC: &str = "__mind_blas_dot_q16_v";

/// RFC 0006 Track B (increment 2) — native MLIR vector-dialect f32
/// L1 (Manhattan, sum of `|a-b|`) reduction surface name.
#[cfg(feature = "std-surface")]
const VEC_DOT_L1_F32_INTRINSIC: &str = "__mind_blas_dot_l1_f32_v";

/// RFC 0006 Track B (increment 3) — native MLIR vector-dialect Q16.16
/// L1 (Manhattan, sum of `|a-b|`) reduction surface name. Byte-identical
/// to the Track A scalar oracle `__mind_blas_dot_l1_q16` at every length
/// (task #57 — integer reduction is associative, and per-element
/// `|sext64(a) - sext64(b)|` is exact; this completes the Q16.16
/// vector-path metric parity left open in increment 2, RFC 0006 §9.3).
#[cfg(feature = "std-surface")]
const VEC_DOT_L1_Q16_INTRINSIC: &str = "__mind_blas_dot_l1_q16_v";

/// RFC 0006 Track B (increment 2) — native MLIR vector-dialect f32
/// L∞ (Chebyshev, max of `|a-b|`) reduction surface name.
#[cfg(feature = "std-surface")]
const VEC_DOT_LINF_F32_INTRINSIC: &str = "__mind_blas_dot_linf_f32_v";

/// RFC 0006 Track B (increment 2) — Q16.16 vector lane count. The
/// scalar Q16.16 oracle widens each `i32` product to `i64` before the
/// arithmetic `>> 16`; the vector path mirrors that with
/// `vector<8xi64>` accumulator lanes. Eight matches the f32 width so
/// LLVM legalises both metrics from a single tile shape.
#[cfg(feature = "std-surface")]
const VEC_Q16_LANES: usize = 8;

/// RFC 0006 Track B (increment 2) — which f32 distance metric the
/// vectorised reduction emits. L2 has its own `emit_vec_dot_f32`
/// (multiply-accumulate); L1/L∞ share `emit_vec_dot_metric_f32`
/// (abs-difference + add/max reduction).
#[cfg(feature = "std-surface")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VecMetric {
    /// Sum of `|a[i] - b[i]|` — `vector.reduction <add>`.
    L1,
    /// Max of `|a[i] - b[i]|` — `vector.reduction <maximumf>`.
    Linf,
}

/// Structured errors produced by the MLIR lowering pipeline.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MlirLowerError {
    /// The lowering pass does not know how to translate the given instruction.
    #[error("unsupported instruction at index {instr_index}: {op}")]
    UnsupportedOp { instr_index: usize, op: String },
    /// Lowering requires type information that is unavailable.
    #[error("missing type information for value {value:?} while lowering {context}")]
    MissingTypeInfo {
        value: ValueId,
        context: &'static str,
    },
    /// The lowering pipeline detected inconsistent shapes or operands.
    #[error("shape error: {0}")]
    ShapeError(String),
    /// IR verification failed before lowering.
    #[error("IR verification failed: {0}")]
    VerificationFailed(#[from] crate::ir::IrVerifyError),
    /// RFC 0002 C-ABI export wrapper codegen rejected an export.
    #[cfg(feature = "ffi-c-user")]
    #[error("C-ABI export codegen: {0}")]
    CExportError(String),
}

/// A lowered MLIR module in textual form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirModule {
    /// The fully formatted MLIR module text.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueKind {
    ScalarI64,
    Tensor {
        dtype: DType,
        shape: Vec<ShapeDim>,
    },
    /// RFC 0006 Track B — a `vector<lanes x f32>` SSA value produced by
    /// the SIMD primitives (`VecLoad` / `VecFma`). Gated; the default
    /// build never constructs the producing instructions so this variant
    /// is unreachable there.
    #[cfg(feature = "std-surface")]
    VectorF32 {
        lanes: usize,
    },
    /// RFC 0006 Track B (increment 2) — a `vector<lanes x i64>` SSA value
    /// produced by the Q16.16 SIMD primitives (`VecLoadI32` widened /
    /// `VecMulAddQ16`). Gated; default builds never construct the
    /// producing instructions so this variant is unreachable there.
    #[cfg(feature = "std-surface")]
    VectorI64 {
        lanes: usize,
    },
}

struct LoweringContext {
    values: BTreeMap<ValueId, ValueKind>,
    outputs: Vec<ValueId>,
    body: String,
    /// RFC 0005 Phase 0: callee name -> arity for every `Instr::Call`
    /// lowered, so the module assembler can emit one
    /// `func.func private @name(i64...) -> i64` declaration per
    /// distinct callee. `BTreeSet` keeps emission deterministic
    /// (stable MLIR text -> stable model_hash). Gated; the default
    /// build has no `Instr::Call` arm and never touches this.
    #[cfg(feature = "std-surface")]
    extern_calls: std::collections::BTreeSet<(String, usize)>,
    /// RFC 0005 P0d: pre-formatted `func.func @name(...) -> i64 { ... }`
    /// bodies for every `Instr::FnDef` seen at module top level. The
    /// assembler concatenates these *before* `@main` and excludes their
    /// names from `extern_calls` so we don't emit a forward decl that
    /// would clash with the definition.
    #[cfg(feature = "std-surface")]
    user_fns: String,
    /// Names defined locally (Instr::FnDef) — filter from extern decls.
    #[cfg(feature = "std-surface")]
    defined_fns: std::collections::BTreeSet<String>,
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            outputs: Vec::new(),
            body: String::new(),
            #[cfg(feature = "std-surface")]
            extern_calls: std::collections::BTreeSet::new(),
            #[cfg(feature = "std-surface")]
            user_fns: String::new(),
            #[cfg(feature = "std-surface")]
            defined_fns: std::collections::BTreeSet::new(),
        }
    }

    fn emit_line(&mut self, line: &str) {
        writeln!(&mut self.body, "{line}").expect("write to string cannot fail");
    }

    fn emit_instr(&mut self, instr_index: usize, instr: &Instr) -> Result<(), MlirLowerError> {
        match instr {
            Instr::ConstI64(id, value) => {
                self.emit_line(&format!("    %{} = arith.constant {} : i64", id.0, value));
                self.values.insert(*id, ValueKind::ScalarI64);
            }
            Instr::ConstTensor(id, dtype, shape, fill) => {
                let dtype_str = dtype.as_str();
                let tensor_ty = tensor_type(shape, dtype_str);
                let fill_value = format_fill(*fill, dtype);
                self.emit_line(&format!(
                    "    %fill{} = arith.constant {} : {}",
                    id.0, fill_value, dtype_str
                ));
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    id.0, tensor_ty
                ));
                self.emit_line(&format!(
                    "    %{} = linalg.fill ins(%fill{} : {}) outs(%tmp{} : {}) -> {}",
                    id.0, id.0, dtype_str, id.0, tensor_ty, tensor_ty
                ));
                self.values.insert(
                    *id,
                    ValueKind::Tensor {
                        dtype: dtype.clone(),
                        shape: shape.clone(),
                    },
                );
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let lhs_kind = self
                    .values
                    .get(lhs)
                    .ok_or(MlirLowerError::MissingTypeInfo {
                        value: *lhs,
                        context: "binop",
                    })?
                    .clone();
                let rhs_kind = self
                    .values
                    .get(rhs)
                    .ok_or(MlirLowerError::MissingTypeInfo {
                        value: *rhs,
                        context: "binop",
                    })?
                    .clone();
                let (result_kind, type_str, op_str) = match (&lhs_kind, &rhs_kind) {
                    (
                        ValueKind::Tensor { dtype, shape },
                        ValueKind::Tensor {
                            dtype: dtype_b,
                            shape: shape_b,
                        },
                    ) => {
                        if dtype != dtype_b || shape != shape_b {
                            return Err(MlirLowerError::ShapeError(
                                "tensor binary ops require matching shapes".into(),
                            ));
                        }
                        let mlir_op = select_arith_op(*op, dtype);
                        let ty = tensor_type(shape, dtype.as_str());
                        (
                            ValueKind::Tensor {
                                dtype: dtype.clone(),
                                shape: shape.clone(),
                            },
                            ty,
                            mlir_op,
                        )
                    }
                    _ => {
                        let mlir_op = match op {
                            BinOp::Add => "arith.addi",
                            BinOp::Sub => "arith.subi",
                            BinOp::Mul => "arith.muli",
                            BinOp::Div => "arith.divsi",
                            BinOp::Mod => "arith.remsi",
                            BinOp::Lt => "arith.cmpi \"slt\",",
                            BinOp::Le => "arith.cmpi \"sle\",",
                            BinOp::Gt => "arith.cmpi \"sgt\",",
                            BinOp::Ge => "arith.cmpi \"sge\",",
                            BinOp::Eq => "arith.cmpi \"eq\",",
                            BinOp::Ne => "arith.cmpi \"ne\",",
                            // Phase 6.5 Stage 1a — bitwise ops on i64.
                            #[cfg(feature = "std-surface")]
                            BinOp::BitAnd => "arith.andi",
                            #[cfg(feature = "std-surface")]
                            BinOp::BitOr => "arith.ori",
                            #[cfg(feature = "std-surface")]
                            BinOp::BitXor => "arith.xori",
                            #[cfg(feature = "std-surface")]
                            BinOp::Shl => "arith.shli",
                            // Arithmetic (signed) right-shift — matches Rust i64 >> i64.
                            #[cfg(feature = "std-surface")]
                            BinOp::Shr => "arith.shrsi",
                        };
                        (ValueKind::ScalarI64, "i64".to_string(), mlir_op)
                    }
                };

                self.emit_line(&format!(
                    "    %{} = {} %{}, %{} : {}",
                    dst.0, op_str, lhs.0, rhs.0, type_str
                ));
                self.values.insert(*dst, result_kind);
            }
            Instr::MatMul { dst, a, b } => {
                let a_info = self.tensor_info(a, "matmul lhs")?;
                let b_info = self.tensor_info(b, "matmul rhs")?;
                let (out_shape, m_ty, n_ty, result_ty) =
                    matmul_shapes(&a_info.shape, &b_info.shape, a_info.dtype.as_str())?;
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    dst.0, result_ty
                ));
                self.emit_line(&format!(
                    "    %{} = linalg.matmul ins(%{} : {} , %{} : {}) outs(%tmp{} : {}) -> {}",
                    dst.0, a.0, m_ty, b.0, n_ty, dst.0, result_ty, result_ty
                ));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: a_info.dtype.clone(),
                        shape: out_shape,
                    },
                );
            }
            Instr::Conv2d {
                dst,
                input,
                filter,
                stride_h,
                stride_w,
                padding,
            } => {
                let input_info = self.tensor_info(input, "conv2d input")?;
                let filter_info = self.tensor_info(filter, "conv2d filter")?;
                let (out_shape, input_ty, filter_ty, result_ty) =
                    conv2d_shapes(&input_info, &filter_info, *stride_h, *stride_w, *padding)?;
                self.emit_line(&format!(
                    "    %tmp{} = tensor.empty() : {}",
                    dst.0, result_ty
                ));
                self.emit_line(&format!(
                    "    %{} = linalg.conv_2d_nhwc_hwcf ins(%{} : {}, %{} : {}) outs(%tmp{} : {}) -> {}",
                    dst.0, input.0, input_ty, filter.0, filter_ty, dst.0, result_ty, result_ty
                ));
                self.values.insert(
                    *dst,
                    ValueKind::Tensor {
                        dtype: input_info.dtype.clone(),
                        shape: out_shape,
                    },
                );
            }
            Instr::Output(id) => {
                self.outputs.push(*id);
            }
            // RFC 0005 Phase 0: generic call -> `func.call`. Scoped to
            // the i64 ABI (every arg + result is i64) — exactly the
            // five `__mind_*` intrinsic signatures. Non-i64 args are a
            // clear error (tensor/aggregate call ABI is RFC 0005
            // phase 2+). Default build has no this arm and the
            // catch-all still errors `UnsupportedOp` on `Instr::Call`
            // exactly as before — byte-identical, moat held.
            #[cfg(feature = "std-surface")]
            Instr::Call { dst, name, args } => {
                for a in args {
                    match self.values.get(a) {
                        Some(ValueKind::ScalarI64) => {}
                        _ => {
                            return Err(MlirLowerError::UnsupportedOp {
                                instr_index,
                                op: format!(
                                    "non-i64 argument to call `{name}` \
                                     (RFC 0005 phase 2+ covers aggregate call ABI)"
                                ),
                            })
                        }
                    }
                }
                // RFC 0006 Track B (increment 1) — the `dot_f32_v` surface
                // fn lowers to a *native* MLIR `vector`-dialect reduction
                // loop instead of a `func.call` to the Track A
                // runtime-support C bridge.  Track A's `__mind_blas_dot_f32`
                // extern path is untouched and remains the scalar/AVX2
                // fallback; this is purely additive.  Any other callee
                // keeps the generic `func.call` lowering below.
                if name == VEC_DOT_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_f32(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 2) — the Q16.16 vector dot.
                // Byte-identical to Track A's scalar `__mind_blas_dot_q16`
                // oracle at every length (task #57 cross-arch bit-identity
                // gate): per-element widen -> multiply -> arithmetic
                // `>> 16` -> i64-lane accumulate -> associative lane sum
                // -> truncate-low-32 + sign-extend. Track A's
                // `__mind_blas_dot_q16` extern path is untouched.
                if name == VEC_DOT_Q16_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_q16(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 2) — f32 L1 / L∞ vector
                // reductions (sum-of-abs / max-of-abs). Same i64-packed-f32
                // ABI and ~1e-4-relative numerical contract as the f32 L2
                // `dot_f32_v` path; Track A's scalar/AVX2 L1/L∞ externs
                // are untouched.
                if name == VEC_DOT_L1_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_metric_f32(*dst, args[0], args[1], args[2], VecMetric::L1);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                // RFC 0006 Track B (increment 3) — the Q16.16 L1 vector
                // reduction. Byte-identical to Track A's scalar
                // `__mind_blas_dot_l1_q16` oracle at every length (task #57):
                // per-element widen -> signed subtract -> arith-only abs
                // (`maxsi(d, 0-d)`, mirroring the C oracle's `if (d<0) d=-d`)
                // -> i64-lane accumulate -> associative lane sum ->
                // truncate-low-32 + sign-extend. Completes the Q16.16
                // vector-path metric parity left open in increment 2
                // (RFC 0006 §9.3). Track A's `__mind_blas_dot_l1_q16` extern
                // path is untouched and remains the scalar/AVX2 fallback.
                if name == VEC_DOT_L1_Q16_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_l1_q16(*dst, args[0], args[1], args[2]);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                if name == VEC_DOT_LINF_F32_INTRINSIC && args.len() == 3 {
                    self.emit_vec_dot_metric_f32(*dst, args[0], args[1], args[2], VecMetric::Linf);
                    self.values.insert(*dst, ValueKind::ScalarI64);
                    return Ok(());
                }
                let arg_refs: Vec<String> = args.iter().map(|a| format!("%{}", a.0)).collect();
                let arg_tys: Vec<&str> = args.iter().map(|_| "i64").collect();
                self.emit_line(&format!(
                    "    %{} = func.call @{}({}) : ({}) -> i64",
                    dst.0,
                    name,
                    arg_refs.join(", "),
                    arg_tys.join(", ")
                ));
                self.values.insert(*dst, ValueKind::ScalarI64);
                self.extern_calls.insert((name.clone(), args.len()));
            }
            // RFC 0005 P0d: emit `func.func @name(%pN: i64...) -> i64 { ... }`
            // for each user-defined function. The body is lowered into a
            // sub-context so its locals get a clean SSA namespace; the
            // resulting text is appended to `user_fns` and emitted as a
            // sibling top-level symbol *before* `@main`. Gated.
            #[cfg(feature = "std-surface")]
            Instr::FnDef {
                name,
                params,
                ret_id,
                body,
                ..
            } => {
                let mut sub = LoweringContext::new();
                for (_pname, pid) in params {
                    sub.values.insert(*pid, ValueKind::ScalarI64);
                }
                for (idx, inner) in body.iter().enumerate() {
                    sub.emit_instr(idx, inner)?;
                }

                let sig_args: Vec<String> = params
                    .iter()
                    .map(|(_, pid)| format!("%{}: i64", pid.0))
                    .collect();
                let mut fn_text = String::new();
                fn_text.push_str(&format!(
                    "  func.func @{}({}) -> i64 {{\n",
                    name,
                    sig_args.join(", ")
                ));
                fn_text.push_str(&sub.body);
                // Every fn returns i64 under the std-surface ABI. If the last
                // emitted line in the body is not a block terminator (return,
                // cf.br, cf.cond_br), synthesise one from `ret_id`.
                //
                // We check the LAST non-empty line of the body, not whether
                // the body contains "return" anywhere.  The previous pattern
                // `.contains("    return ")` incorrectly suppressed the
                // synthetic return for functions that use early-return inside
                // `if` branches but end with a plain value expression — those
                // functions emit instructions after the final `^if_after_N:`
                // block label that have no terminator.
                let last_line = sub
                    .body
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("")
                    .trim();
                let already_terminated = last_line.starts_with("return")
                    || last_line.starts_with("cf.br ")
                    || last_line.starts_with("cf.cond_br ");
                if !already_terminated {
                    match ret_id {
                        Some(rid) => fn_text.push_str(&format!("    return %{} : i64\n", rid.0)),
                        None => fn_text
                            .push_str("    %z = arith.constant 0 : i64\n    return %z : i64\n"),
                    }
                }
                fn_text.push_str("  }\n");

                self.user_fns.push_str(&fn_text);
                self.defined_fns.insert(name.clone());
                // Bubble up any extern calls or nested definitions discovered
                // inside the body so the module-level assembler sees them.
                for ec in sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                for df in sub.defined_fns {
                    self.defined_fns.insert(df);
                }
                self.user_fns.push_str(&sub.user_fns);
            }
            // P0d: function parameters bind a ValueId to the i64 ABI; the
            // value is named in the enclosing `func.func` signature so we
            // do not emit anything for the Param itself. Gated.
            #[cfg(feature = "std-surface")]
            Instr::Param { dst, .. } => {
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            // P0d: explicit `return %v : i64` inside a user fn body.
            #[cfg(feature = "std-surface")]
            Instr::Return { value } => match value {
                Some(v) => self.emit_line(&format!("    return %{} : i64", v.0)),
                None => self.emit_line("    return"),
            },
            // RFC 0005 Gap 1: `while cond { body }` — basic-block loop lowering.
            //
            // MLIR structure emitted (cf dialect):
            //
            //   cf.br ^while_header_N
            // ^while_header_N:
            //   <cond_instrs>
            //   %cond_bool_N = arith.trunci %cond_id : i64 to i1
            //   cf.cond_br %cond_bool_N, ^while_body_N, ^while_after_N
            // ^while_body_N:
            //   <body_instrs>
            //   cf.br ^while_header_N
            // ^while_after_N:
            //
            // N = instr_index for uniqueness across nested whiles. Gated.
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_id,
                cond_instrs,
                body,
                ..
            } => {
                let lbl = instr_index;
                // Fall into the header from the entry block.
                self.emit_line(&format!("    cf.br ^while_header_{lbl}"));
                // Header block: evaluate condition.
                self.emit_line(&format!("  ^while_header_{lbl}:"));
                let mut cond_sub = LoweringContext::new();
                for (vid, kind) in &self.values {
                    cond_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ci) in cond_instrs.iter().enumerate() {
                    cond_sub.emit_instr(idx, ci)?;
                }
                self.body.push_str(&cond_sub.body);
                for (vid, kind) in cond_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the condition sub-context.
                #[cfg(feature = "std-surface")]
                for ec in cond_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                // i64 condition → i1 for cf.cond_br.
                self.emit_line(&format!(
                    "    %cond_bool_{lbl} = arith.trunci %{} : i64 to i1",
                    cond_id.0
                ));
                self.emit_line(&format!(
                    "    cf.cond_br %cond_bool_{lbl}, ^while_body_{lbl}, ^while_after_{lbl}"
                ));
                // Body block.
                self.emit_line(&format!("  ^while_body_{lbl}:"));
                let mut body_sub = LoweringContext::new();
                for (vid, kind) in &self.values {
                    body_sub.values.insert(*vid, kind.clone());
                }
                for (idx, bi) in body.iter().enumerate() {
                    body_sub.emit_instr(idx, bi)?;
                }
                self.body.push_str(&body_sub.body);
                for (vid, kind) in body_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the body sub-context.
                #[cfg(feature = "std-surface")]
                for ec in body_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                // Back-edge: loop back to the header.
                self.emit_line(&format!("    cf.br ^while_header_{lbl}"));
                // After block: execution continues here when the condition is false.
                self.emit_line(&format!("  ^while_after_{lbl}:"));
            }
            // RFC 0005 Phase 6.2b Gap 2 — `const NAME: [i64; N] = [...]`
            // lowers to an MLIR `arith.constant` dense attribute that is
            // stored to a `memref<N x i64>` alloca so fn bodies can load
            // from it.  The name is threaded through as an SSA comment so
            // textual IR round-trips retain the label.
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { dst, name, values } => {
                let label = name.as_deref().unwrap_or("__anon");
                // Emit a dense integer array constant as a tensor<Ni64> global.
                let n = values.len();
                let elems: String = values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                self.emit_line(&format!("  // const.array @{label} : [i64; {n}]"));
                self.emit_line(&format!(
                    "  {} = arith.constant dense<[{}]> : tensor<{}xi64>",
                    dst, elems, n
                ));
            }
            // RFC 0005 Phase 6.2b Gap 2 — `arr[idx]` element load.
            // Lowers to an `tensor.extract` from the base tensor constant.
            #[cfg(feature = "std-surface")]
            Instr::ArrayLoad { dst, base, index } => {
                self.emit_line(&format!(
                    "  {} = tensor.extract {}[{}] : tensor<?>",
                    dst, base, index
                ));
            }
            // RFC 0006 Track B (increment 1) — SIMD vector load.
            //
            // The Option-C ABI gives us i64 opaque addresses; native MLIR
            // memory access uses `llvm.inttoptr` to recover a pointer,
            // `llvm.getelementptr` (i8 element type) to apply the byte
            // offset, then a vector-typed `llvm.load` of
            // `vector<lanes x f32>`.  `convert-vector-to-llvm` +
            // `reconcile-unrealized-casts` legalise this to the host SIMD
            // width with no per-target code and no C shim — the Track B
            // thesis-pure property (vs Track A's runtime-support bridge).
            #[cfg(feature = "std-surface")]
            Instr::VecLoad {
                dst,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vptr_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    dst.0, base.0
                ));
                self.emit_line(&format!(
                    "    %vptr{0} = llvm.getelementptr %vptr_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    dst.0, offset.0
                ));
                self.emit_line(&format!(
                    "    %{0} = llvm.load %vptr{0} : !llvm.ptr -> vector<{1}xf32>",
                    dst.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorF32 { lanes: l });
            }
            // RFC 0006 Track B (increment 1) — element-wise fused
            // multiply-add: `dst = a * b + acc`.  Lowers to `vector.fma`,
            // which `convert-vector-to-llvm` turns into the
            // `llvm.intr.fmuladd` intrinsic (one hardware FMA per lane
            // group on targets that have one).
            #[cfg(feature = "std-surface")]
            Instr::VecFma {
                dst,
                a,
                b,
                acc,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %{0} = vector.fma %{1}, %{2}, %{3} : vector<{4}xf32>",
                    dst.0, a.0, b.0, acc.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorF32 { lanes: l });
            }
            // RFC 0006 Track B (increment 1) — horizontal sum to scalar.
            //
            // `vector.reduction <add>` becomes `llvm.intr.vector.reduce.fadd`.
            // The result is the f32 scalar bit-packed (zero-extended) into
            // an i64 so it travels the Option-C i64 ABI exactly like every
            // other `__mind_blas_*` return value.  The tree-shaped pairwise
            // reduction is NOT bit-identical to a sequential scalar sum —
            // the numerical contract bounds it to 1e-4 relative of the f64
            // oracle, matching Track A's AVX2 path.
            #[cfg(feature = "std-surface")]
            Instr::VecReduceAdd { dst, src, lanes } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vred{0} = vector.reduction <add>, %{1} : \
                     vector<{2}xf32> into f32",
                    dst.0, src.0, l
                ));
                self.emit_line(&format!(
                    "    %vbits{0} = arith.bitcast %vred{0} : f32 to i32",
                    dst.0
                ));
                self.emit_line(&format!(
                    "    %{0} = arith.extui %vbits{0} : i32 to i64",
                    dst.0
                ));
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            // RFC 0006 Track B (increment 2) — symmetric vector store:
            // `mem[base + offset .. +lanes] = src`.  Recovers the pointer
            // from the Option-C i64 address (`llvm.inttoptr`), applies the
            // byte offset (`llvm.getelementptr`, i8 element type) and emits
            // a vector-typed `llvm.store`.  Mirror image of `VecLoad`.
            #[cfg(feature = "std-surface")]
            Instr::VecStore {
                src,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vsp_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    src.0, base.0
                ));
                self.emit_line(&format!(
                    "    %vsp{0} = llvm.getelementptr %vsp_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    src.0, offset.0
                ));
                self.emit_line(&format!(
                    "    llvm.store %{0}, %vsp{0} : vector<{1}xf32>, !llvm.ptr",
                    src.0, l
                ));
            }
            // RFC 0006 Track B (increment 2) — i32 sibling of `VecLoad`
            // for the Q16.16 path.  Same address recovery; the loaded
            // value is `vector<lanes x i32>`.
            #[cfg(feature = "std-surface")]
            Instr::VecLoadI32 {
                dst,
                base,
                offset,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %viptr_b{0} = llvm.inttoptr %{1} : i64 to !llvm.ptr",
                    dst.0, base.0
                ));
                self.emit_line(&format!(
                    "    %viptr{0} = llvm.getelementptr %viptr_b{0}[%{1}] : \
                     (!llvm.ptr, i64) -> !llvm.ptr, i8",
                    dst.0, offset.0
                ));
                self.emit_line(&format!(
                    "    %{0} = llvm.load %viptr{0} : !llvm.ptr -> vector<{1}xi32>",
                    dst.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorI64 { lanes: l });
            }
            // RFC 0006 Track B (increment 2) — Q16.16 fused widening
            // multiply-shift-accumulate.  `dst = acc + ((sext64(a) *
            // sext64(b)) >>a 16)`, element-wise.  The shift is *arithmetic*
            // (`arith.shrsi`), exactly mirroring the Track A scalar oracle's
            // per-element `prod >> 16` under LLVM `ashr` semantics — this
            // is the operation the cross-arch bit-identity contract (#57)
            // pins.  `a`/`b` are `vector<lanes x i32>`; `acc`/`dst` are
            // `vector<lanes x i64>`.
            #[cfg(feature = "std-surface")]
            Instr::VecMulAddQ16 {
                dst,
                a,
                b,
                acc,
                lanes,
            } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %vqa{0} = arith.extsi %{1} : vector<{2}xi32> to vector<{2}xi64>",
                    dst.0, a.0, l
                ));
                self.emit_line(&format!(
                    "    %vqb{0} = arith.extsi %{1} : vector<{2}xi32> to vector<{2}xi64>",
                    dst.0, b.0, l
                ));
                self.emit_line(&format!(
                    "    %vqp{0} = arith.muli %vqa{0}, %vqb{0} : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %vqs16_{0} = arith.constant dense<16> : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %vqsh{0} = arith.shrsi %vqp{0}, %vqs16_{0} : vector<{1}xi64>",
                    dst.0, l
                ));
                self.emit_line(&format!(
                    "    %{0} = arith.addi %{1}, %vqsh{0} : vector<{2}xi64>",
                    dst.0, acc.0, l
                ));
                self.values.insert(*dst, ValueKind::VectorI64 { lanes: l });
            }
            // RFC 0006 Track B (increment 2) — horizontal i64 sum.
            // `vector.reduction <add>` over `vector<lanes x i64>` ->
            // `llvm.intr.vector.reduce.add`.  Integer addition is
            // associative, so this is bit-identical to a sequential scalar
            // accumulation no matter how LLVM groups the lanes — the
            // property the #57 Q16.16 gate relies on.
            #[cfg(feature = "std-surface")]
            Instr::VecReduceAddI64 { dst, src, lanes } => {
                let l = *lanes;
                self.emit_line(&format!(
                    "    %{0} = vector.reduction <add>, %{1} : \
                     vector<{2}xi64> into i64",
                    dst.0, src.0, l
                ));
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            // Phase 6.5 Stage 1a — `if cond { then } else { else }` lowering.
            //
            // MLIR structure emitted (cf dialect, matching the While pattern):
            //
            //   %cond_i1_N = arith.trunci %cond_id : i64 to i1
            //   cf.cond_br %cond_i1_N, ^if_then_N, ^if_else_N
            // ^if_then_N:
            //   <then_instrs>
            //   cf.br ^if_after_N(%then_result : i64)
            // ^if_else_N:
            //   <else_instrs>
            //   cf.br ^if_after_N(%else_result : i64)
            // ^if_after_N(%dst : i64):
            //
            // Block arguments are used to forward the branch value to the
            // join block — this is MLIR's standard pattern for if-as-value.
            // The previous placeholder (arith.constant 0) broke all if
            // expressions used as values (e.g. `let x = if cond { a } else { b }`),
            // replacing the selected value with 0.
            //
            // For branches that terminate with `Instr::Return`, the
            // `cf.br ^if_after_N` is omitted because the block is already
            // terminated by a `return` op.  If BOTH branches return, the
            // ^if_after_N block receives no predecessors but mlir-opt will
            // DCE it; the block arg is still declared for structural validity.
            //
            // N = dst.0 (globally unique SSA id) for uniqueness. Gated.
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_id,
                cond_instrs,
                then_instrs,
                then_result,
                else_instrs,
                else_result,
                dst,
                ..
            } => {
                // Use `dst.0` (the unique SSA id of the result value) as the
                // label suffix instead of `instr_index`.  `instr_index` resets
                // to 0 in every sub-context (e.g. inside a FnDef body loop),
                // causing block-label collisions when multiple `Instr::If`
                // nodes appear in the same function.  `dst.0` is globally
                // unique within the module.
                let lbl = dst.0;

                // Emit the condition sub-instructions into the current block.
                let mut cond_sub = LoweringContext::new();
                for (vid, kind) in &self.values {
                    cond_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ci) in cond_instrs.iter().enumerate() {
                    cond_sub.emit_instr(idx, ci)?;
                }
                self.body.push_str(&cond_sub.body);
                for (vid, kind) in cond_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the condition sub-context.
                for ec in cond_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }

                // Determine whether the condition value is already i1 (produced
                // by a comparison BinOp like `arith.cmpi`) or is a plain i64
                // that needs truncation.  MLIR's `cf.cond_br` requires an i1.
                //
                // We inspect the last instruction in `cond_instrs`: if it is a
                // comparison BinOp (Lt/Le/Gt/Ge/Eq/Ne), the result is already
                // i1 and we use it directly.  Otherwise we emit `arith.trunci`.
                let cond_already_i1 = cond_instrs
                    .last()
                    .map(|last| {
                        matches!(
                            last,
                            Instr::BinOp {
                                op: BinOp::Lt
                                    | BinOp::Le
                                    | BinOp::Gt
                                    | BinOp::Ge
                                    | BinOp::Eq
                                    | BinOp::Ne,
                                ..
                            }
                        )
                    })
                    .unwrap_or(false);

                if cond_already_i1 {
                    // Comparison result is already i1 — use it directly.
                    self.emit_line(&format!(
                        "    cf.cond_br %{}, ^if_then_{lbl}, ^if_else_{lbl}",
                        cond_id.0
                    ));
                } else {
                    // Plain i64 → truncate to i1 first.
                    self.emit_line(&format!(
                        "    %cond_i1_{lbl} = arith.trunci %{} : i64 to i1",
                        cond_id.0
                    ));
                    self.emit_line(&format!(
                        "    cf.cond_br %cond_i1_{lbl}, ^if_then_{lbl}, ^if_else_{lbl}"
                    ));
                }

                // Then block.
                self.emit_line(&format!("  ^if_then_{lbl}:"));
                let mut then_sub = LoweringContext::new();
                for (vid, kind) in &self.values {
                    then_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ti) in then_instrs.iter().enumerate() {
                    then_sub.emit_instr(idx, ti)?;
                }
                self.body.push_str(&then_sub.body);
                for (vid, kind) in then_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the then sub-context.
                for ec in then_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                // If the last instruction in the then-block was already a
                // `return`, do NOT emit a `cf.br` — the block is already
                // properly terminated.  Otherwise forward then_result to the
                // join block via a block-argument branch.
                let then_ends_with_return = then_instrs
                    .last()
                    .map(|i| matches!(i, Instr::Return { .. }))
                    .unwrap_or(false);
                if !then_ends_with_return {
                    self.emit_line(&format!(
                        "    cf.br ^if_after_{lbl}(%{} : i64)",
                        then_result.0
                    ));
                }

                // Else block.
                self.emit_line(&format!("  ^if_else_{lbl}:"));
                let mut else_sub = LoweringContext::new();
                for (vid, kind) in &self.values {
                    else_sub.values.insert(*vid, kind.clone());
                }
                for (idx, ei) in else_instrs.iter().enumerate() {
                    else_sub.emit_instr(idx, ei)?;
                }
                self.body.push_str(&else_sub.body);
                for (vid, kind) in else_sub.values {
                    self.values.insert(vid, kind);
                }
                // Bubble up extern_calls from the else sub-context.
                for ec in else_sub.extern_calls {
                    self.extern_calls.insert(ec);
                }
                let else_ends_with_return = else_instrs
                    .last()
                    .map(|i| matches!(i, Instr::Return { .. }))
                    .unwrap_or(false);
                if !else_ends_with_return {
                    self.emit_line(&format!(
                        "    cf.br ^if_after_{lbl}(%{} : i64)",
                        else_result.0
                    ));
                }

                // Join block: declare the block argument that carries the if-value.
                // Both incoming `cf.br` edges supply an i64; the block argument
                // becomes `%dst`.  If both branches returned, the block has no
                // predecessors and mlir-opt will DCE it — that's fine.
                self.emit_line(&format!("  ^if_after_{lbl}(%{} : i64):", dst.0));
                // Register then_result and else_result as known values for any
                // downstream code that directly references them (e.g. for
                // branch_bindings threading in the fn-body loop).
                self.values.insert(*then_result, ValueKind::ScalarI64);
                self.values.insert(*else_result, ValueKind::ScalarI64);
                self.values.insert(*dst, ValueKind::ScalarI64);
            }
            _ => {
                return Err(MlirLowerError::UnsupportedOp {
                    instr_index,
                    op: format!("{:?}", instr),
                })
            }
        }

        Ok(())
    }

    /// RFC 0006 Track B (increment 1) — emit a native MLIR `vector`-dialect
    /// f32 dot-product reduction over two opaque i64 base addresses and a
    /// runtime length.
    ///
    /// Structure (all in the `vector` / `scf` / `arith` / `llvm` dialects,
    /// no runtime-support C call):
    ///
    /// ```text
    ///   main loop  : scf.for step LANES, vector.load + vector.fma
    ///   horizontal : vector.reduction <add>
    ///   scalar tail: scf.for step 1 for the len % LANES remainder
    ///   pack       : arith.bitcast f32 -> i32 -> zext i64  (Option-C ABI)
    /// ```
    ///
    /// The body uses the same op repertoire as the standalone
    /// `Instr::VecLoad` / `VecFma` / `VecReduceAdd` arms; emitting it as one
    /// fused block keeps the SSA namespace local and the loop-carried
    /// `vector<LANES x f32>` accumulator legal. `convert-vector-to-llvm`
    /// + `convert-scf-to-cf` legalise it with no per-target code.
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_f32(&mut self, dst: ValueId, a_addr: ValueId, b_addr: ValueId, len: ValueId) {
        let d = dst.0;
        let l = VEC_DOT_F32_LANES;
        // Byte stride of one f32 element.
        let elem_bytes = std::mem::size_of::<f32>() as i64;
        self.emit_line(&format!("    %vd_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vd_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vd_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vd_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vd_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vd_nv_{d} = arith.divui %vd_len_{d}, %vd_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vd_ve_{d} = arith.muli %vd_nv_{d}, %vd_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vd_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vd_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vd_z_{d} = arith.constant dense<0.0> : vector<{l}xf32>"
        ));
        // Vectorised main loop: LANES-wide FMA accumulation.
        self.emit_line(&format!(
            "    %vd_vacc_{d} = scf.for %vd_i_{d} = %vd_c0_{d} to %vd_ve_{d} \
             step %vd_cl_{d} iter_args(%vd_acc_{d} = %vd_z_{d}) -> (vector<{l}xf32>) {{"
        ));
        self.emit_line(&format!(
            "      %vd_ii_{d} = arith.index_cast %vd_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vd_bo_{d} = arith.muli %vd_ii_{d}, %vd_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vd_ai_{d} = llvm.getelementptr %vd_ap_{d}[%vd_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_bi_{d} = llvm.getelementptr %vd_bp_{d}[%vd_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_av_{d} = llvm.load %vd_ai_{d} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vd_bv_{d} = llvm.load %vd_bi_{d} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vd_fa_{d} = vector.fma %vd_av_{d}, %vd_bv_{d}, %vd_acc_{d} : \
             vector<{l}xf32>"
        ));
        self.emit_line(&format!("      scf.yield %vd_fa_{d} : vector<{l}xf32>"));
        self.emit_line("    }");
        // Horizontal sum of the lane accumulator.
        self.emit_line(&format!(
            "    %vd_vs_{d} = vector.reduction <add>, %vd_vacc_{d} : \
             vector<{l}xf32> into f32"
        ));
        // Scalar tail for the len % LANES remainder.
        self.emit_line(&format!(
            "    %vd_ts_{d} = scf.for %vd_j_{d} = %vd_ve_{d} to %vd_len_{d} \
             step %vd_c1_{d} iter_args(%vd_s_{d} = %vd_vs_{d}) -> (f32) {{"
        ));
        self.emit_line(&format!(
            "      %vd_jj_{d} = arith.index_cast %vd_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vd_jb_{d} = arith.muli %vd_jj_{d}, %vd_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vd_aj_{d} = llvm.getelementptr %vd_ap_{d}[%vd_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_bj_{d} = llvm.getelementptr %vd_bp_{d}[%vd_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vd_as_{d} = llvm.load %vd_aj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vd_bs_{d} = llvm.load %vd_bj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vd_p_{d} = arith.mulf %vd_as_{d}, %vd_bs_{d} : f32"
        ));
        self.emit_line(&format!(
            "      %vd_ns_{d} = arith.addf %vd_s_{d}, %vd_p_{d} : f32"
        ));
        self.emit_line(&format!("      scf.yield %vd_ns_{d} : f32"));
        self.emit_line("    }");
        // Pack the f32 result into the low 32 bits of an i64 (Option-C ABI,
        // identical contract to Track A's `__mind_blas_dot_f32`).
        self.emit_line(&format!(
            "    %vd_bits_{d} = arith.bitcast %vd_ts_{d} : f32 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extui %vd_bits_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 2) — emit a native MLIR
    /// `vector`-dialect Q16.16 dot-product reduction.
    ///
    /// This path is **byte-identical** to the Track A scalar oracle
    /// `mind_blas_dot_q16_scalar` at every length — the cross-arch
    /// bit-identity gate (task #57) extended to the thesis-pure vector
    /// path. The scalar oracle computes, per element,
    /// `acc += ((i64)a[i] * (i64)b[i]) >> 16` (arithmetic shift) and
    /// finally returns `(i64)(i32)acc`. The vector path performs the
    /// *identical* per-element widen-multiply-arithmetic-shift, then
    /// accumulates into `vector<LANES x i64>` lanes and sums the lanes
    /// with `vector.reduction <add>`. Integer addition is associative,
    /// so the lane re-association does not perturb a single bit — unlike
    /// the f32 path, no tolerance is needed.
    ///
    /// Structure:
    ///
    /// ```text
    ///   main loop  : scf.for step LANES, i32 loads, extsi i64,
    ///                muli, shrsi 16, addi (i64-lane accumulate)
    ///   horizontal : vector.reduction <add> over vector<LANES x i64>
    ///   scalar tail : scf.for step 1 for the len % LANES remainder,
    ///                identical per-element op in scalar i64
    ///   pack       : trunc i64 -> i32 -> sext i64 (Option-C ABI,
    ///                identical to `(i64)(i32)acc` in the C oracle)
    /// ```
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_q16(&mut self, dst: ValueId, a_addr: ValueId, b_addr: ValueId, len: ValueId) {
        let d = dst.0;
        let l = VEC_Q16_LANES;
        // Q16.16 lanes are i32 (4 bytes), same stride as f32.
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        self.emit_line(&format!("    %vq_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vq_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vq_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vq_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!("    %vq_s16_{d} = arith.constant 16 : i64"));
        self.emit_line(&format!(
            "    %vq_s16v_{d} = arith.constant dense<16> : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "    %vq_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vq_nv_{d} = arith.divui %vq_len_{d}, %vq_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vq_ve_{d} = arith.muli %vq_nv_{d}, %vq_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vq_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vq_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vq_z_{d} = arith.constant dense<0> : vector<{l}xi64>"
        ));
        // Vectorised main loop: LANES-wide widening MAC into i64 lanes.
        self.emit_line(&format!(
            "    %vq_vacc_{d} = scf.for %vq_i_{d} = %vq_c0_{d} to %vq_ve_{d} \
             step %vq_cl_{d} iter_args(%vq_acc_{d} = %vq_z_{d}) -> (vector<{l}xi64>) {{"
        ));
        self.emit_line(&format!(
            "      %vq_ii_{d} = arith.index_cast %vq_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vq_bo_{d} = arith.muli %vq_ii_{d}, %vq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_ai_{d} = llvm.getelementptr %vq_ap_{d}[%vq_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_bi_{d} = llvm.getelementptr %vq_bp_{d}[%vq_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_av_{d} = llvm.load %vq_ai_{d} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vq_bv_{d} = llvm.load %vq_bi_{d} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vq_aw_{d} = arith.extsi %vq_av_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_bw_{d} = arith.extsi %vq_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_pr_{d} = arith.muli %vq_aw_{d}, %vq_bw_{d} : vector<{l}xi64>"
        ));
        // Per-element arithmetic right shift by 16 — mirrors the scalar
        // oracle's `prod >> 16` exactly (LLVM `ashr`).
        self.emit_line(&format!(
            "      %vq_sh_{d} = arith.shrsi %vq_pr_{d}, %vq_s16v_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vq_na_{d} = arith.addi %vq_acc_{d}, %vq_sh_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!("      scf.yield %vq_na_{d} : vector<{l}xi64>"));
        self.emit_line("    }");
        // Associative horizontal i64 sum — bit-identical regardless of
        // lane grouping (this is what makes #57 hold for the vector path).
        self.emit_line(&format!(
            "    %vq_vs_{d} = vector.reduction <add>, %vq_vacc_{d} : \
             vector<{l}xi64> into i64"
        ));
        // Scalar tail for the len % LANES remainder — identical per-element
        // op in scalar i64 so the boundary elements match the oracle too.
        self.emit_line(&format!(
            "    %vq_ts_{d} = scf.for %vq_j_{d} = %vq_ve_{d} to %vq_len_{d} \
             step %vq_c1_{d} iter_args(%vq_s_{d} = %vq_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "      %vq_jj_{d} = arith.index_cast %vq_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vq_jb_{d} = arith.muli %vq_jj_{d}, %vq_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_aj_{d} = llvm.getelementptr %vq_ap_{d}[%vq_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_bj_{d} = llvm.getelementptr %vq_bp_{d}[%vq_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vq_as_{d} = llvm.load %vq_aj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vq_bs_{d} = llvm.load %vq_bj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vq_asw_{d} = arith.extsi %vq_as_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vq_bsw_{d} = arith.extsi %vq_bs_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vq_p_{d} = arith.muli %vq_asw_{d}, %vq_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_psh_{d} = arith.shrsi %vq_p_{d}, %vq_s16_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vq_ns_{d} = arith.addi %vq_s_{d}, %vq_psh_{d} : i64"
        ));
        self.emit_line(&format!("      scf.yield %vq_ns_{d} : i64"));
        self.emit_line("    }");
        // Final `(i64)(i32)acc`: truncate to the low 32 Q16.16 bits then
        // sign-extend back into i64 — byte-for-byte the C oracle's return.
        self.emit_line(&format!(
            "    %vq_lo_{d} = arith.trunci %vq_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extsi %vq_lo_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 3) — emit a native MLIR
    /// `vector`-dialect Q16.16 **L1** (Manhattan, sum of `|a-b|`)
    /// reduction.
    ///
    /// **Byte-identical** to the Track A scalar oracle
    /// `mind_blas_dot_l1_q16_scalar` at every length — the cross-arch
    /// bit-identity contract (task #57). The oracle accumulates
    /// `d = (i64)a[i] - (i64)b[i]; if (d < 0) d = -d; acc += d` in i64 and
    /// returns `(i64)(i32)acc`. This kernel replicates exactly that: widen
    /// both i32 lanes to i64, signed-subtract, take the absolute value as
    /// `maxsi(d, 0 - d)` (pure `arith`, no `math` dialect — the same value
    /// as the C `if (d<0) d=-d` for every representable `d`, and the lane
    /// difference of two sign-extended i32 is in `[-(2^32-1), 2^32-1]`, far
    /// from `i64::MIN`, so the `-d` negation never overflows), accumulate
    /// into i64 lanes, then an associative `vector.reduction <add>`
    /// horizontal sum + a scalar tail doing the identical per-element op,
    /// then `trunci i64->i32` + `extsi i32->i64`. Integer add is
    /// associative, so lane grouping is irrelevant — bit-identical to the
    /// sequential scalar oracle on every input. This closes the Q16.16
    /// vector-path metric parity deferred in increment 2 (RFC 0006 §9.3).
    /// Track A's `__mind_blas_dot_l1_q16` extern path is untouched.
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_l1_q16(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        len: ValueId,
    ) {
        let d = dst.0;
        let l = VEC_Q16_LANES;
        // Q16.16 lanes are i32 (4 bytes), same stride as f32.
        let elem_bytes = std::mem::size_of::<i32>() as i64;
        self.emit_line(&format!("    %vl_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vl_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vl_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vl_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!("    %vl_z0_{d} = arith.constant 0 : i64"));
        self.emit_line(&format!(
            "    %vl_zv_{d} = arith.constant dense<0> : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "    %vl_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vl_nv_{d} = arith.divui %vl_len_{d}, %vl_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vl_ve_{d} = arith.muli %vl_nv_{d}, %vl_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vl_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vl_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        // Vectorised main loop: LANES-wide widen -> sub -> abs -> i64 accumulate.
        self.emit_line(&format!(
            "    %vl_vacc_{d} = scf.for %vl_i_{d} = %vl_c0_{d} to %vl_ve_{d} \
             step %vl_cl_{d} iter_args(%vl_acc_{d} = %vl_zv_{d}) -> (vector<{l}xi64>) {{"
        ));
        self.emit_line(&format!(
            "      %vl_ii_{d} = arith.index_cast %vl_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vl_bo_{d} = arith.muli %vl_ii_{d}, %vl_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_ai_{d} = llvm.getelementptr %vl_ap_{d}[%vl_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_bi_{d} = llvm.getelementptr %vl_bp_{d}[%vl_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_av_{d} = llvm.load %vl_ai_{d} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vl_bv_{d} = llvm.load %vl_bi_{d} : !llvm.ptr -> vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vl_aw_{d} = arith.extsi %vl_av_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_bw_{d} = arith.extsi %vl_bv_{d} : vector<{l}xi32> to vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_df_{d} = arith.subi %vl_aw_{d}, %vl_bw_{d} : vector<{l}xi64>"
        ));
        // arith-only absolute value: |d| = max(d, -d). Mirrors the C
        // oracle's `if (d < 0) d = -d` exactly for every representable d.
        self.emit_line(&format!(
            "      %vl_ng_{d} = arith.subi %vl_zv_{d}, %vl_df_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_ab_{d} = arith.maxsi %vl_df_{d}, %vl_ng_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!(
            "      %vl_na_{d} = arith.addi %vl_acc_{d}, %vl_ab_{d} : vector<{l}xi64>"
        ));
        self.emit_line(&format!("      scf.yield %vl_na_{d} : vector<{l}xi64>"));
        self.emit_line("    }");
        // Associative horizontal i64 sum — bit-identical regardless of
        // lane grouping (this is what makes #57 hold for the vector path).
        self.emit_line(&format!(
            "    %vl_vs_{d} = vector.reduction <add>, %vl_vacc_{d} : \
             vector<{l}xi64> into i64"
        ));
        // Scalar tail for the len % LANES remainder — identical per-element
        // op in scalar i64 so the boundary elements match the oracle too.
        self.emit_line(&format!(
            "    %vl_ts_{d} = scf.for %vl_j_{d} = %vl_ve_{d} to %vl_len_{d} \
             step %vl_c1_{d} iter_args(%vl_s_{d} = %vl_vs_{d}) -> (i64) {{"
        ));
        self.emit_line(&format!(
            "      %vl_jj_{d} = arith.index_cast %vl_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vl_jb_{d} = arith.muli %vl_jj_{d}, %vl_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_aj_{d} = llvm.getelementptr %vl_ap_{d}[%vl_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_bj_{d} = llvm.getelementptr %vl_bp_{d}[%vl_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vl_as_{d} = llvm.load %vl_aj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vl_bs_{d} = llvm.load %vl_bj_{d} : !llvm.ptr -> i32"
        ));
        self.emit_line(&format!(
            "      %vl_asw_{d} = arith.extsi %vl_as_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vl_bsw_{d} = arith.extsi %vl_bs_{d} : i32 to i64"
        ));
        self.emit_line(&format!(
            "      %vl_sd_{d} = arith.subi %vl_asw_{d}, %vl_bsw_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_sn_{d} = arith.subi %vl_z0_{d}, %vl_sd_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_sa_{d} = arith.maxsi %vl_sd_{d}, %vl_sn_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vl_ns_{d} = arith.addi %vl_s_{d}, %vl_sa_{d} : i64"
        ));
        self.emit_line(&format!("      scf.yield %vl_ns_{d} : i64"));
        self.emit_line("    }");
        // Final `(i64)(i32)acc`: truncate to the low 32 Q16.16 bits then
        // sign-extend back into i64 — byte-for-byte the C oracle's return.
        self.emit_line(&format!(
            "    %vl_lo_{d} = arith.trunci %vl_ts_{d} : i64 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extsi %vl_lo_{d} : i32 to i64"));
    }

    /// RFC 0006 Track B (increment 2) — emit a native MLIR
    /// `vector`-dialect f32 L1 (sum of `|a-b|`) or L∞ (max of `|a-b|`)
    /// reduction.
    ///
    /// Same i64-packed-f32 Option-C ABI as `dot_f32_v`. The reduction is
    /// `vector.reduction <add>` (L1) or `<maximumf>` (L∞) on the lane
    /// accumulator after a sign-bit-mask absolute value of the lane
    /// difference (bitcast f32->i32, AND 0x7fffffff, bitcast back —
    /// `arith`-only, no `math` dialect, identical to Track A's AVX2
    /// `_mm256_and_ps` abs). The
    /// tree-shaped reduction reorders the f32 summation exactly like Track
    /// A's AVX2 L1/L∞ path, so the numerical contract is the documented
    /// 1e-4 relative bound vs an f64 oracle (L∞ max is associative and is
    /// in fact byte-identical, but the harness asserts the same tolerance
    /// for uniformity).
    #[cfg(feature = "std-surface")]
    fn emit_vec_dot_metric_f32(
        &mut self,
        dst: ValueId,
        a_addr: ValueId,
        b_addr: ValueId,
        len: ValueId,
        metric: VecMetric,
    ) {
        let d = dst.0;
        let l = VEC_DOT_F32_LANES;
        let elem_bytes = std::mem::size_of::<f32>() as i64;
        // Lane / scalar reduction op + identity element per metric.
        let (vred_kind, init_dense, scalar_combine_op): (&str, &str, &str) = match metric {
            VecMetric::L1 => ("<add>", "0.0", "arith.addf"),
            VecMetric::Linf => ("<maximumf>", "0.0", "arith.maximumf"),
        };
        self.emit_line(&format!("    %vm_c0_{d} = arith.constant 0 : index"));
        self.emit_line(&format!("    %vm_c1_{d} = arith.constant 1 : index"));
        self.emit_line(&format!("    %vm_cl_{d} = arith.constant {l} : index"));
        self.emit_line(&format!(
            "    %vm_eb_{d} = arith.constant {elem_bytes} : i64"
        ));
        self.emit_line(&format!(
            "    %vm_len_{d} = arith.index_cast %{} : i64 to index",
            len.0
        ));
        self.emit_line(&format!(
            "    %vm_nv_{d} = arith.divui %vm_len_{d}, %vm_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vm_ve_{d} = arith.muli %vm_nv_{d}, %vm_cl_{d} : index"
        ));
        self.emit_line(&format!(
            "    %vm_ap_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            a_addr.0
        ));
        self.emit_line(&format!(
            "    %vm_bp_{d} = llvm.inttoptr %{} : i64 to !llvm.ptr",
            b_addr.0
        ));
        self.emit_line(&format!(
            "    %vm_z_{d} = arith.constant dense<{init_dense}> : vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "    %vm_vacc_{d} = scf.for %vm_i_{d} = %vm_c0_{d} to %vm_ve_{d} \
             step %vm_cl_{d} iter_args(%vm_acc_{d} = %vm_z_{d}) -> (vector<{l}xf32>) {{"
        ));
        self.emit_line(&format!(
            "      %vm_ii_{d} = arith.index_cast %vm_i_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vm_bo_{d} = arith.muli %vm_ii_{d}, %vm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vm_ai_{d} = llvm.getelementptr %vm_ap_{d}[%vm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_bi_{d} = llvm.getelementptr %vm_bp_{d}[%vm_bo_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_av_{d} = llvm.load %vm_ai_{d} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vm_bv_{d} = llvm.load %vm_bi_{d} : !llvm.ptr -> vector<{l}xf32>"
        ));
        self.emit_line(&format!(
            "      %vm_di_{d} = arith.subf %vm_av_{d}, %vm_bv_{d} : vector<{l}xf32>"
        ));
        // Absolute value via sign-bit mask (bitcast f32->i32, AND
        // 0x7fffffff, bitcast back).  This uses only `arith` ops already
        // in the shared lowering pipeline — `math.absf` would need
        // `convert-math-to-llvm` added to the pipeline, perturbing the
        // bench-gate moat.  It is also exactly Track A's AVX2 abs (an
        // `_mm256_and_ps` with a 0x7fffffff mask), so the vector path is
        // numerically faithful to that reference.
        self.emit_line(&format!(
            "      %vm_am_{d} = arith.constant dense<2147483647> : vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_db_{d} = arith.bitcast %vm_di_{d} : vector<{l}xf32> to vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_abi_{d} = arith.andi %vm_db_{d}, %vm_am_{d} : vector<{l}xi32>"
        ));
        self.emit_line(&format!(
            "      %vm_ab_{d} = arith.bitcast %vm_abi_{d} : vector<{l}xi32> to vector<{l}xf32>"
        ));
        match metric {
            VecMetric::L1 => {
                self.emit_line(&format!(
                    "      %vm_na_{d} = arith.addf %vm_acc_{d}, %vm_ab_{d} : vector<{l}xf32>"
                ));
            }
            VecMetric::Linf => {
                self.emit_line(&format!(
                    "      %vm_na_{d} = arith.maximumf %vm_acc_{d}, %vm_ab_{d} : vector<{l}xf32>"
                ));
            }
        }
        self.emit_line(&format!("      scf.yield %vm_na_{d} : vector<{l}xf32>"));
        self.emit_line("    }");
        self.emit_line(&format!(
            "    %vm_vs_{d} = vector.reduction {vred_kind}, %vm_vacc_{d} : \
             vector<{l}xf32> into f32"
        ));
        // Scalar tail.
        self.emit_line(&format!(
            "    %vm_ts_{d} = scf.for %vm_j_{d} = %vm_ve_{d} to %vm_len_{d} \
             step %vm_c1_{d} iter_args(%vm_s_{d} = %vm_vs_{d}) -> (f32) {{"
        ));
        self.emit_line(&format!(
            "      %vm_jj_{d} = arith.index_cast %vm_j_{d} : index to i64"
        ));
        self.emit_line(&format!(
            "      %vm_jb_{d} = arith.muli %vm_jj_{d}, %vm_eb_{d} : i64"
        ));
        self.emit_line(&format!(
            "      %vm_aj_{d} = llvm.getelementptr %vm_ap_{d}[%vm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_bj_{d} = llvm.getelementptr %vm_bp_{d}[%vm_jb_{d}] : \
             (!llvm.ptr, i64) -> !llvm.ptr, i8"
        ));
        self.emit_line(&format!(
            "      %vm_as_{d} = llvm.load %vm_aj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vm_bs_{d} = llvm.load %vm_bj_{d} : !llvm.ptr -> f32"
        ));
        self.emit_line(&format!(
            "      %vm_ds_{d} = arith.subf %vm_as_{d}, %vm_bs_{d} : f32"
        ));
        self.emit_line(&format!(
            "      %vm_asm_{d} = arith.constant 2147483647 : i32"
        ));
        self.emit_line(&format!(
            "      %vm_dsb_{d} = arith.bitcast %vm_ds_{d} : f32 to i32"
        ));
        self.emit_line(&format!(
            "      %vm_absi_{d} = arith.andi %vm_dsb_{d}, %vm_asm_{d} : i32"
        ));
        self.emit_line(&format!(
            "      %vm_abs_{d} = arith.bitcast %vm_absi_{d} : i32 to f32"
        ));
        self.emit_line(&format!(
            "      %vm_ns_{d} = {scalar_combine_op} %vm_s_{d}, %vm_abs_{d} : f32"
        ));
        self.emit_line(&format!("      scf.yield %vm_ns_{d} : f32"));
        self.emit_line("    }");
        self.emit_line(&format!(
            "    %vm_bits_{d} = arith.bitcast %vm_ts_{d} : f32 to i32"
        ));
        self.emit_line(&format!("    %{d} = arith.extui %vm_bits_{d} : i32 to i64"));
    }

    fn tensor_info(
        &self,
        id: &ValueId,
        context: &'static str,
    ) -> Result<TensorInfo, MlirLowerError> {
        match self.values.get(id) {
            Some(ValueKind::Tensor { dtype, shape }) => Ok(TensorInfo {
                dtype: dtype.clone(),
                shape: shape.clone(),
            }),
            _ => Err(MlirLowerError::MissingTypeInfo {
                value: *id,
                context,
            }),
        }
    }
}

/// Lower a verified and canonicalized [`IRModule`] into MLIR text.
///
/// The lowering does not mutate the input module and is deterministic:
/// the same IR produces identical MLIR text.
pub fn lower_ir_to_mlir(module: &IRModule) -> Result<MlirModule, MlirLowerError> {
    let mut ctx = LoweringContext::new();

    for (idx, instr) in module.instrs.iter().enumerate() {
        ctx.emit_instr(idx, instr)?;
    }

    let mut ret_types = Vec::new();
    if !ctx.outputs.is_empty() {
        let mut value_list = String::new();
        let mut type_list = String::new();
        for (i, id) in ctx.outputs.iter().enumerate() {
            let info = ctx.values.get(id).ok_or(MlirLowerError::MissingTypeInfo {
                value: *id,
                context: "function return",
            })?;
            ret_types.push(mlir_type(info)?);
            if i > 0 {
                value_list.push_str(", ");
                type_list.push_str(", ");
            }
            write!(&mut value_list, "%{}", id.0).unwrap();
            write!(&mut type_list, "{}", mlir_type(info)?).unwrap();
        }
        ctx.emit_line(&format!("    return {} : {}", value_list, type_list));
    } else {
        ctx.emit_line("    return");
    }

    let mut out = String::new();
    out.push_str("module {\n");

    // RFC 0005 Phase 0: one `func.func private @callee(i64...) -> i64`
    // declaration per distinct callee, before `@main`, so the
    // `func.call`s emitted above resolve. Sorted (BTreeSet) for
    // deterministic MLIR text / model_hash. Gated; default build
    // emits none of this. P0d: skip names that have a local
    // `func.func` definition emitted below — declaring a private
    // forward decl AND a definition for the same symbol is invalid.
    #[cfg(feature = "std-surface")]
    for (name, arity) in &ctx.extern_calls {
        if ctx.defined_fns.contains(name) {
            continue;
        }
        let params = vec!["i64"; *arity].join(", ");
        out.push_str(&format!("  func.func private @{name}({params}) -> i64\n"));
    }
    // RFC 0005 P0d: user-defined `func.func @name(...) -> i64 { ... }`
    // definitions, in source order. Emitted before `@main` so the
    // `func.call`s inside @main resolve. Gated; default build emits none.
    #[cfg(feature = "std-surface")]
    out.push_str(&ctx.user_fns);

    if ret_types.is_empty() {
        out.push_str("  func.func @main() -> () {\n");
    } else {
        out.push_str(&format!(
            "  func.func @main() -> ({}) {{\n",
            ret_types.join(", ")
        ));
    }
    out.push_str(&ctx.body);
    out.push_str("  }\n");

    // RFC 0002 D2: append `mind_fn_<name>_invoke` C-ABI wrappers as
    // sibling top-level symbols, before the module-closing brace.
    // Module-level feature gate only — the default build never touches
    // this path and emits byte-identical MLIR (compile-speed moat).
    #[cfg(feature = "ffi-c-user")]
    crate::mlir::c_export::emit_c_export_wrappers(&mut out, module)
        .map_err(MlirLowerError::CExportError)?;

    out.push_str("}\n");

    Ok(MlirModule { text: out })
}

/// Convenience helper: verify, canonicalize, and lower into MLIR text.
pub fn compile_ir_to_mlir_text(module: &mut IRModule) -> Result<String, MlirLowerError> {
    crate::ir::verify_module(module)?;
    canonicalize_module(module);
    crate::ir::verify_module(module)?;
    let lowered = lower_ir_to_mlir(module)?;
    Ok(lowered.text)
}

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: DType,
    shape: Vec<ShapeDim>,
}

fn mlir_type(kind: &ValueKind) -> Result<String, MlirLowerError> {
    match kind {
        ValueKind::ScalarI64 => Ok("i64".to_string()),
        ValueKind::Tensor { dtype, shape } => Ok(tensor_type(shape, dtype.as_str())),
        #[cfg(feature = "std-surface")]
        ValueKind::VectorF32 { lanes } => Ok(format!("vector<{lanes}xf32>")),
        #[cfg(feature = "std-surface")]
        ValueKind::VectorI64 { lanes } => Ok(format!("vector<{lanes}xi64>")),
    }
}

fn tensor_type(shape: &[ShapeDim], dtype: &str) -> String {
    if shape.is_empty() {
        return format!("tensor<{}>", dtype);
    }

    let dims = shape
        .iter()
        .map(shape_dim_to_string)
        .collect::<Vec<_>>()
        .join("x");
    format!("tensor<{}x{}>", dims, dtype)
}

fn shape_dim_to_string(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Known(n) => n.to_string(),
        ShapeDim::Sym(sym) => sym.to_string(),
    }
}

fn select_arith_op(op: BinOp, dtype: &DType) -> &'static str {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => match op {
            BinOp::Add => "arith.addf",
            BinOp::Sub => "arith.subf",
            BinOp::Mul => "arith.mulf",
            BinOp::Div => "arith.divf",
            BinOp::Mod => "arith.remf",
            BinOp::Lt => "arith.cmpf \"olt\",",
            BinOp::Le => "arith.cmpf \"ole\",",
            BinOp::Gt => "arith.cmpf \"ogt\",",
            BinOp::Ge => "arith.cmpf \"oge\",",
            BinOp::Eq => "arith.cmpf \"oeq\",",
            BinOp::Ne => "arith.cmpf \"one\",",
            // Bitwise ops on floating-point tensors are not meaningful;
            // emit a placeholder that mlir-opt will reject loudly rather
            // than silently producing wrong code.
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => "arith.andi",
        },
        _ => match op {
            BinOp::Add => "arith.addi",
            BinOp::Sub => "arith.subi",
            BinOp::Mul => "arith.muli",
            BinOp::Div => "arith.divsi",
            BinOp::Mod => "arith.remsi",
            BinOp::Lt => "arith.cmpi \"slt\",",
            BinOp::Le => "arith.cmpi \"sle\",",
            BinOp::Gt => "arith.cmpi \"sgt\",",
            BinOp::Ge => "arith.cmpi \"sge\",",
            BinOp::Eq => "arith.cmpi \"eq\",",
            BinOp::Ne => "arith.cmpi \"ne\",",
            #[cfg(feature = "std-surface")]
            BinOp::BitAnd => "arith.andi",
            #[cfg(feature = "std-surface")]
            BinOp::BitOr => "arith.ori",
            #[cfg(feature = "std-surface")]
            BinOp::BitXor => "arith.xori",
            #[cfg(feature = "std-surface")]
            BinOp::Shl => "arith.shli",
            #[cfg(feature = "std-surface")]
            BinOp::Shr => "arith.shrsi",
        },
    }
}

fn matmul_shapes(
    a_shape: &[ShapeDim],
    b_shape: &[ShapeDim],
    dtype: &str,
) -> Result<(Vec<ShapeDim>, String, String, String), MlirLowerError> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(MlirLowerError::ShapeError(
            "matmul lowering expects rank-2 tensors".to_string(),
        ));
    }
    let m = a_shape[0].clone();
    let k = a_shape[1].clone();
    let k_rhs = b_shape[0].clone();
    if !shapes_compatible(&k, &k_rhs) {
        return Err(MlirLowerError::ShapeError(
            "matmul K dimensions must match".into(),
        ));
    }
    let n = b_shape[1].clone();
    let out_shape = vec![m.clone(), n.clone()];
    let lhs_ty = tensor_type(a_shape, dtype);
    let rhs_ty = tensor_type(b_shape, dtype);
    let out_ty = tensor_type(&out_shape, dtype);
    Ok((out_shape, lhs_ty, rhs_ty, out_ty))
}

fn shapes_compatible(a: &ShapeDim, b: &ShapeDim) -> bool {
    match (a, b) {
        (ShapeDim::Known(x), ShapeDim::Known(y)) => x == y,
        (ShapeDim::Sym(x), ShapeDim::Sym(y)) => x == y,
        _ => true,
    }
}

fn conv2d_shapes(
    input: &TensorInfo,
    filter: &TensorInfo,
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) -> Result<(Vec<ShapeDim>, String, String, String), MlirLowerError> {
    if input.shape.len() != 4 || filter.shape.len() != 4 {
        return Err(MlirLowerError::ShapeError(
            "conv2d lowering expects NHWC input and HWCF filter".into(),
        ));
    }

    let batch = input.shape[0].clone();
    let in_h = &input.shape[1];
    let in_w = &input.shape[2];
    let out_channels = filter.shape[3].clone();
    let kernel_h = &filter.shape[0];
    let kernel_w = &filter.shape[1];

    let in_channels = &input.shape[3];
    let filter_in_channels = &filter.shape[2];
    if !shapes_compatible(in_channels, filter_in_channels) {
        return Err(MlirLowerError::ShapeError(
            "conv2d input channels must match filter input channels".into(),
        ));
    }

    let out_h = conv_output_dim(in_h, kernel_h, stride_h, padding)?;
    let out_w = conv_output_dim(in_w, kernel_w, stride_w, padding)?;

    let out_shape = vec![batch, out_h, out_w, out_channels];
    let input_ty = tensor_type(&input.shape, input.dtype.as_str());
    let filter_ty = tensor_type(&filter.shape, filter.dtype.as_str());
    let out_ty = tensor_type(&out_shape, input.dtype.as_str());
    Ok((out_shape, input_ty, filter_ty, out_ty))
}

fn conv_output_dim(
    input: &ShapeDim,
    kernel: &ShapeDim,
    stride: usize,
    padding: ConvPadding,
) -> Result<ShapeDim, MlirLowerError> {
    let input_known = known_dim(input);
    let kernel_known = known_dim(kernel);
    let result = match padding {
        ConvPadding::Valid => {
            crate::linalg::conv_output_dim_valid(input_known, kernel_known, stride)
                .map_err(MlirLowerError::ShapeError)?
        }
        ConvPadding::Same => crate::linalg::conv_output_dim_same(input_known, stride)
            .map_err(MlirLowerError::ShapeError)?,
    };
    Ok(match result {
        Some(n) => ShapeDim::Known(n),
        None => input.clone(),
    })
}

fn known_dim(dim: &ShapeDim) -> Option<usize> {
    match dim {
        ShapeDim::Known(n) => Some(*n),
        ShapeDim::Sym(_) => None,
    }
}

fn format_fill(fill: Option<f64>, dtype: &DType) -> String {
    match (fill, dtype) {
        (Some(v), DType::F32 | DType::F16 | DType::BF16) => format_number(v),
        (Some(v), _) => format_number(v.trunc()),
        (None, DType::F32 | DType::F16 | DType::BF16) => "0.0".to_string(),
        (None, _) => "0".to_string(),
    }
}

fn format_number(n: f64) -> String {
    if (n.fract()).abs() < f64::EPSILON {
        format!("{:.1}", n)
    } else {
        n.to_string()
    }
}
