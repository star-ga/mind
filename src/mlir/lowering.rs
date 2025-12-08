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
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            values: BTreeMap::new(),
            outputs: Vec::new(),
            body: String::new(),
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
    if ret_types.is_empty() {
        out.push_str("  func.func @main() -> () {\n");
    } else {
        out.push_str(&format!("  func.func @main() -> ({}) {{\n", ret_types.join(", ")));
    }
    out.push_str(&ctx.body);
    out.push_str("  }\n");
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
        },
        _ => match op {
            BinOp::Add => "arith.addi",
            BinOp::Sub => "arith.subi",
            BinOp::Mul => "arith.muli",
            BinOp::Div => "arith.divsi",
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
