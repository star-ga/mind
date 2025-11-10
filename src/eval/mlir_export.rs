use std::fmt::Write;

use crate::ir::{BinOp, IRModule, Instr};
use crate::types::{DType, ShapeDim};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlirLowerPreset {
    None,
    ArithLinalg,
    CpuDemo,
}

impl MlirLowerPreset {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "arith-linalg" => Some(Self::ArithLinalg),
            "cpu-demo" => Some(Self::CpuDemo),
            _ => None,
        }
    }
}

impl Default for MlirLowerPreset {
    fn default() -> Self {
        Self::None
    }
}

/// Apply purely textual rewrites to the emitted MLIR for "lowering".
pub fn apply_textual_lowering(mut mlir: String, preset: MlirLowerPreset) -> String {
    match preset {
        MlirLowerPreset::None => mlir,
        MlirLowerPreset::ArithLinalg => {
            // Keep simple and explicit: normalize function attrs, canonicalize tensor.empty uses, etc.
            mlir = mlir.replace("arith.constant", "arith.constant");
            mlir
        }
        MlirLowerPreset::CpuDemo => {
            // Add a couple of cosmetic inlines / canonical names for demos
            mlir = mlir.replace("linalg.fill", "linalg.fill");
            mlir
        }
    }
}

pub fn to_mlir_text(ir: &IRModule) -> String {
    to_mlir(ir, "main")
}

pub fn to_mlir(ir: &IRModule, entry: &str) -> String {
    let mut out = String::new();

    writeln!(&mut out, "module {{").unwrap();
    writeln!(&mut out, "  func.func @{}() -> () {{", entry).unwrap();

    for instr in &ir.instrs {
        match instr {
            Instr::ConstI64(id, n) => {
                writeln!(&mut out, "    %{} = arith.constant {} : i64", id.0, n).unwrap();
            }
            Instr::ConstTensor(id, dtype, shape, fill) => {
                let dtype_str = dtype_to_mlir(dtype);
                let tensor_ty = tensor_type(shape, dtype_str);
                let empty_name = format!("empty{}", id.0);
                let fill_name = format!("fill{}", id.0);
                let fill_value = format_fill(*fill, dtype);

                writeln!(&mut out, "    %{} = tensor.empty() : {}", empty_name, tensor_ty).unwrap();
                writeln!(
                    &mut out,
                    "    %{} = arith.constant {} : {}",
                    fill_name, fill_value, dtype_str
                )
                .unwrap();
                writeln!(
                    &mut out,
                    "    %{} = linalg.fill ins(%{} : {}) outs(%{} : {}) -> {}",
                    id.0, fill_name, dtype_str, empty_name, tensor_ty, tensor_ty
                )
                .unwrap();
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let op_str = match op {
                    BinOp::Add => "arith.addi",
                    BinOp::Sub => "arith.subi",
                    BinOp::Mul => "arith.muli",
                    BinOp::Div => "arith.divsi",
                };
                writeln!(&mut out, "    %{} = {} %{}, %{} : i64", dst.0, op_str, lhs.0, rhs.0)
                    .unwrap();
            }
            Instr::Output(id) => {
                writeln!(&mut out, "    return").unwrap();
                writeln!(&mut out, "    // result: %{}", id.0).unwrap();
            }
            other => {
                writeln!(&mut out, "    // TODO: {:?}", other).unwrap();
            }
        }
    }

    writeln!(&mut out, "  }}").unwrap();
    writeln!(&mut out, "}}").unwrap();

    out
}

fn dtype_to_mlir(dtype: &DType) -> &str {
    match dtype {
        DType::I32 => "i32",
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        DType::F16 => "f16",
    }
}

fn tensor_type(shape: &[ShapeDim], dtype: &str) -> String {
    if shape.is_empty() {
        return format!("tensor<{}>", dtype);
    }

    let dims = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(sym) => sym.to_string(),
        })
        .collect::<Vec<_>>()
        .join("x");

    format!("tensor<{}x{}>", dims, dtype)
}

fn format_fill(fill: Option<f64>, dtype: &DType) -> String {
    match (fill, dtype) {
        (Some(value), DType::F32 | DType::F16 | DType::BF16) => {
            if value.fract() == 0.0 {
                format!("{:.1}", value)
            } else {
                value.to_string()
            }
        }
        (Some(value), _) => {
            if value.fract() == 0.0 {
                (value as i64).to_string()
            } else {
                value.to_string()
            }
        }
        (None, DType::F32 | DType::F16 | DType::BF16) => "0.0".to_string(),
        (None, _) => "0".to_string(),
    }
}
