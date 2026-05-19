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
    Tensor { dtype: DType, shape: Vec<ShapeDim> },
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
                self.emit_line(&format!(
                    "  // const.array @{label} : [i64; {n}]"
                ));
                self.emit_line(&format!(
                    "  {} = arith.constant dense<[{}]> : tensor<{}xi64>",
                    dst,
                    elems,
                    n
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

                // Determine whether the condition value is already i1 (produced
                // by a comparison BinOp like `arith.cmpi`) or is a plain i64
                // that needs truncation.  MLIR's `cf.cond_br` requires an i1.
                //
                // We inspect the last instruction in `cond_instrs`: if it is a
                // comparison BinOp (Lt/Le/Gt/Ge/Eq/Ne), the result is already
                // i1 and we use it directly.  Otherwise we emit `arith.trunci`.
                let cond_already_i1 = cond_instrs.last().map(|last| {
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
                }).unwrap_or(false);

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
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                "arith.andi"
            }
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
