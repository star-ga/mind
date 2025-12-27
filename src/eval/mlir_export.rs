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

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::BinOp;

use crate::ir::IRModule;

use crate::ir::IndexSpec;

use crate::ir::Instr;

use crate::ir::SliceSpec;

use crate::ir::ValueId;

use crate::types::ConvPadding;
use crate::types::DType;
use crate::types::ShapeDim;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlirEmitMode {
    Plain,
    Executable,
}

#[derive(Debug, Clone)]
pub struct MlirEmitOptions {
    pub lower_preset: Option<String>,
    pub mode: MlirEmitMode,
}

impl Default for MlirEmitOptions {
    fn default() -> Self {
        Self {
            lower_preset: None,
            mode: MlirEmitMode::Plain,
        }
    }
}

#[derive(Clone, Debug)]
struct TensorInfo {
    dtype: DType,
    shape: Vec<ShapeDim>,
}

struct MlirEmitter {
    out: String,
    next_tmp: usize,
    tensors: HashMap<ValueId, TensorInfo>,
    outputs: Vec<ValueId>,
    need_print_i64: bool,
    need_print_newline: bool,
}

impl MlirEmitter {
    fn new() -> Self {
        Self {
            out: String::new(),
            next_tmp: 0,
            tensors: HashMap::new(),
            outputs: Vec::new(),
            need_print_i64: false,
            need_print_newline: false,
        }
    }

    fn finish(self) -> String {
        self.out
    }

    fn write_line(&mut self, line: &str) {
        writeln!(&mut self.out, "{line}").unwrap();
    }

    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) {
        self.out.write_fmt(args).unwrap();
    }

    fn tmp(&mut self) -> String {
        let name = format!("%tmp{}", self.next_tmp);
        self.next_tmp += 1;
        name
    }

    fn record_tensor(&mut self, id: ValueId, dtype: DType, shape: Vec<ShapeDim>) {
        self.tensors.insert(id, TensorInfo { dtype, shape });
    }

    fn tensor_info(&self, id: &ValueId) -> Option<&TensorInfo> {
        self.tensors.get(id)
    }

    fn record_output(&mut self, id: ValueId) {
        self.outputs.push(id);
    }

    fn emit_executable_prints(&mut self) {
        let mut printed_any = false;
        for value in self.outputs.clone() {
            if self.tensors.contains_key(&value) {
                continue;
            }
            self.write_fmt(format_args!(
                "    func.call @printI64(%{}) : (i64) -> ()\n",
                value.0
            ));
            self.need_print_i64 = true;
            printed_any = true;
        }
        if printed_any {
            self.write_line("    func.call @printNewline() : () -> ()");
            self.need_print_newline = true;
        }
    }

    fn emit_executable_helpers(&mut self) {
        if self.need_print_i64 {
            self.write_line("  func.func private @printI64(i64)");
        }
        if self.need_print_newline {
            self.write_line("  func.func private @printNewline()");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MlirLowerPreset {
    #[default]
    None,
    Core,
    ArithLinalg,
    CpuDemo,
    JitCpu,
    GpuDefault,
}

impl MlirLowerPreset {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "core" => Some(Self::Core),
            "arith-linalg" => Some(Self::ArithLinalg),
            "cpu-demo" => Some(Self::CpuDemo),
            "jit-cpu" => Some(Self::JitCpu),
            "gpu-default" => Some(Self::GpuDefault),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Core => "core",
            Self::ArithLinalg => "arith-linalg",
            Self::CpuDemo => "cpu-demo",
            Self::JitCpu => "jit-cpu",
            Self::GpuDefault => "gpu-default",
        }
    }
}

impl std::str::FromStr for MlirLowerPreset {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MlirLowerPreset::parse(s).ok_or(())
    }
}

/// Apply purely textual rewrites to the emitted MLIR for "lowering".
pub fn apply_textual_lowering(mlir: String, preset: MlirLowerPreset) -> String {
    match preset {
        MlirLowerPreset::None
        | MlirLowerPreset::Core
        | MlirLowerPreset::ArithLinalg
        | MlirLowerPreset::CpuDemo
        | MlirLowerPreset::JitCpu
        | MlirLowerPreset::GpuDefault => mlir,
    }
}

pub fn apply_lowering(mlir: &str, preset: &str) -> Result<String, String> {
    if preset.is_empty() {
        return Ok(mlir.to_string());
    }
    let lowered = if let Some(preset) = MlirLowerPreset::parse(preset) {
        apply_textual_lowering(mlir.to_string(), preset)
    } else {
        return Err(format!("unknown MLIR lowering preset '{preset}'"));
    };
    Ok(lowered)
}

pub fn emit_mlir_with_opts(ir: &IRModule, opts: &MlirEmitOptions) -> String {
    let mut text = to_mlir_with_mode(ir, "main", opts.mode);
    if let Some(preset_name) = opts.lower_preset.as_deref() {
        if let Some(preset) = MlirLowerPreset::parse(preset_name) {
            text = apply_textual_lowering(text, preset);
        }
    }
    text
}

pub fn to_mlir_text(ir: &IRModule) -> String {
    to_mlir_with_mode(ir, "main", MlirEmitMode::Plain)
}

pub fn to_mlir(ir: &IRModule, entry: &str) -> String {
    to_mlir_with_mode(ir, entry, MlirEmitMode::Plain)
}

pub fn to_mlir_with_mode(ir: &IRModule, entry: &str, mode: MlirEmitMode) -> String {
    let mut emitter = MlirEmitter::new();

    emitter.write_line("module {");
    emitter.write_fmt(format_args!("  func.func @{}() -> () {{", entry));
    emitter.write_line("");

    for instr in &ir.instrs {
        emit_instr(&mut emitter, instr);
    }

    if matches!(mode, MlirEmitMode::Executable) {
        emitter.emit_executable_prints();
    }

    emitter.write_line("    return");
    emitter.write_line("  }");
    if matches!(mode, MlirEmitMode::Executable) {
        emitter.emit_executable_helpers();
    }
    emitter.write_line("}");

    emitter.finish()
}

fn emit_instr(emitter: &mut MlirEmitter, instr: &Instr) {
    match instr {
        Instr::ConstI64(id, n) => {
            emitter.write_fmt(format_args!("    %{} = arith.constant {} : i64\n", id.0, n));
        }
        Instr::ConstTensor(id, dtype, shape, fill) => {
            emit_const_tensor(emitter, *id, dtype, shape, *fill)
        }
        Instr::BinOp { dst, op, lhs, rhs } => emit_int_binop(emitter, *dst, *op, *lhs, *rhs),
        Instr::Sum {
            dst,
            src,
            axes,
            keepdims,
        } => emit_tensor_reduce(emitter, *dst, *src, axes, *keepdims, ReduceKind::Sum),
        Instr::Mean {
            dst,
            src,
            axes,
            keepdims,
        } => emit_tensor_reduce(emitter, *dst, *src, axes, *keepdims, ReduceKind::Mean),
        Instr::Reshape {
            dst,
            src,
            new_shape,
        } => emit_tensor_reshape(emitter, *dst, *src, new_shape),
        Instr::ExpandDims { dst, src, axis } => emit_expand_dims(emitter, *dst, *src, *axis),
        Instr::Squeeze { dst, src, axes } => emit_squeeze(emitter, *dst, *src, axes),
        Instr::Transpose { dst, src, perm } => emit_transpose(emitter, *dst, *src, perm),
        Instr::Dot { dst, a, b } => emit_dot(emitter, *dst, *a, *b),
        Instr::MatMul { dst, a, b } => emit_matmul(emitter, *dst, *a, *b),
        Instr::Conv2d { .. } => {
            emitter.write_line("    // conv2d lowering is handled by the public MLIR pipeline");
        }
        Instr::Conv2dGradInput {
            dst,
            dy,
            filter,
            input_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            emit_conv2d_grad_input(
                emitter,
                *dst,
                *dy,
                *filter,
                *input_shape,
                *stride_h,
                *stride_w,
                *padding,
            );
        }
        Instr::Conv2dGradFilter {
            dst,
            input,
            dy,
            filter_shape,
            stride_h,
            stride_w,
            padding,
        } => {
            emit_conv2d_grad_filter(
                emitter,
                *dst,
                *input,
                *dy,
                *filter_shape,
                *stride_h,
                *stride_w,
                *padding,
            );
        }
        Instr::Index { dst, src, indices } => emit_index(emitter, *dst, *src, indices),
        Instr::Slice { dst, src, dims } => emit_slice(emitter, *dst, *src, dims),
        Instr::Gather {
            dst,
            src,
            indices,
            axis,
        } => emit_gather(emitter, *dst, *src, *indices, *axis),
        Instr::Output(id) => {
            emitter.record_output(*id);
            emitter.write_fmt(format_args!("    // result: %{}\n", id.0));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReduceKind {
    Sum,
    Mean,
}

fn emit_const_tensor(
    emitter: &mut MlirEmitter,
    id: ValueId,
    dtype: &DType,
    shape: &[ShapeDim],
    fill: Option<f64>,
) {
    let dtype_str = dtype_to_mlir(dtype);
    let tensor_ty = tensor_type(shape, dtype_str);
    let empty_name = emitter.tmp();
    let fill_name = emitter.tmp();
    let fill_value = format_fill(fill, dtype);

    emitter.write_fmt(format_args!(
        "    {} = tensor.empty() : {}\n",
        empty_name, tensor_ty
    ));
    emitter.write_fmt(format_args!(
        "    {} = arith.constant {} : {}\n",
        fill_name, fill_value, dtype_str
    ));
    emitter.write_fmt(format_args!(
        "    %{} = linalg.fill ins({} : {}) outs({} : {}) -> {}\n",
        id.0, fill_name, dtype_str, empty_name, tensor_ty, tensor_ty
    ));

    emitter.record_tensor(id, dtype.clone(), shape.to_vec());
}

fn emit_int_binop(emitter: &mut MlirEmitter, dst: ValueId, op: BinOp, lhs: ValueId, rhs: ValueId) {
    let op_str = match op {
        BinOp::Add => "arith.addi",
        BinOp::Sub => "arith.subi",
        BinOp::Mul => "arith.muli",
        BinOp::Div => "arith.divsi",
    };
    emitter.write_fmt(format_args!(
        "    %{} = {} %{}, %{} : i64\n",
        dst.0, op_str, lhs.0, rhs.0
    ));
}

fn emit_tensor_reduce(
    emitter: &mut MlirEmitter,
    dst: ValueId,
    src: ValueId,
    axes: &[i64],
    keepdims: bool,
    kind: ReduceKind,
) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let dtype = src_info.dtype.clone();
    let dtype_str = dtype_to_mlir(&dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let axes_norm = normalize_axes(axes, src_info.shape.len());
    let out_shape = reduce_shape(&src_info.shape, &axes_norm, keepdims);
    let out_ty = tensor_type(&out_shape, dtype_str);
    let dims_attr = format_dimensions(&axes_norm);
    let zero = match dtype {
        DType::F32 | DType::F16 | DType::BF16 => "0.0".to_string(),
        _ => "0".to_string(),
    };
    let add_op = match dtype {
        DType::F32 | DType::F16 | DType::BF16 => "arith.addf",
        _ => "arith.addi",
    };
    let init_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "    {} = arith.constant {} : {}\n",
        init_name, zero, dtype_str
    ));

    let reduce_result_name = if matches!(kind, ReduceKind::Mean) {
        emitter.tmp()
    } else {
        format!("%{}", dst.0)
    };

    emitter.write_fmt(format_args!(
        "    {} = tensor.reduce %{} init {} {{\n",
        reduce_result_name, src.0, init_name
    ));
    emitter.write_fmt(format_args!(
        "      ^bb0(%elem: {0}, %acc: {0}):\n",
        dtype_str
    ));
    let sum_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "        {} = {} %acc, %elem : {}\n",
        sum_name, add_op, dtype_str
    ));
    emitter.write_fmt(format_args!(
        "        tensor.yield {} : {}\n",
        sum_name, dtype_str
    ));
    emitter.write_fmt(format_args!(
        "    }} {{dimensions = [{dims_attr}]}} : {src_ty} -> {out_ty}\n",
        dims_attr = dims_attr,
        src_ty = src_ty,
        out_ty = out_ty
    ));

    match kind {
        ReduceKind::Sum => {
            emitter.record_tensor(dst, dtype, out_shape);
        }
        ReduceKind::Mean => {
            let count = element_count(&src_info.shape, &axes_norm).max(1);
            let count_literal = match dtype {
                DType::F32 | DType::F16 | DType::BF16 => format!("{:.1}", count as f64),
                _ => count.to_string(),
            };
            let extract = emitter.tmp();
            let indices: Vec<String> = out_shape.iter().map(|_| "0".to_string()).collect();
            emitter.write_fmt(format_args!(
                "    {extract} = tensor.extract {result}[{indices}] : {out_ty}\n",
                extract = extract,
                result = reduce_result_name,
                indices = indices.join(", "),
                out_ty = out_ty
            ));

            let divisor = emitter.tmp();
            emitter.write_fmt(format_args!(
                "    {divisor} = arith.constant {count_literal} : {dtype}\n",
                divisor = divisor,
                count_literal = count_literal,
                dtype = dtype_str
            ));
            let div_op = match dtype {
                DType::F32 | DType::F16 | DType::BF16 => "arith.divf",
                _ => "arith.divsi",
            };
            let div_name = emitter.tmp();
            emitter.write_fmt(format_args!(
                "    {div_name} = {div_op} {extract}, {divisor} : {dtype}\n",
                div_name = div_name,
                div_op = div_op,
                extract = extract,
                divisor = divisor,
                dtype = dtype_str
            ));
            emitter.write_fmt(format_args!(
                "    %{} = tensor.from_elements {div_name} : {}\n",
                dst.0,
                out_ty,
                div_name = div_name
            ));
            emitter.record_tensor(dst, dtype, out_shape);
        }
    }
}

fn emit_tensor_reshape(
    emitter: &mut MlirEmitter,
    dst: ValueId,
    src: ValueId,
    new_shape: &[ShapeDim],
) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let dtype_str = dtype_to_mlir(&src_info.dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let dst_ty = tensor_type(new_shape, dtype_str);
    emitter.write_fmt(format_args!(
        "    %{} = tensor.reshape %{} : {} -> {}\n",
        dst.0, src.0, src_ty, dst_ty
    ));
    emitter.record_tensor(dst, src_info.dtype, new_shape.to_vec());
}

fn emit_expand_dims(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, axis: i64) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let mut shape = src_info.shape.clone();
    let axis_norm = axis.clamp(0, shape.len() as i64) as usize;
    shape.insert(axis_norm, ShapeDim::Known(1));
    emit_tensor_reshape(emitter, dst, src, &shape);
}

fn emit_squeeze(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, axes: &[i64]) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let norm_axes = normalize_axes(axes, src_info.shape.len());
    let mut shape = Vec::new();
    for (i, dim) in src_info.shape.iter().enumerate() {
        if norm_axes.contains(&i) {
            continue;
        }
        shape.push(dim.clone());
    }
    emit_tensor_reshape(emitter, dst, src, &shape);
}

fn emit_transpose(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, perm: &[i64]) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let rank = src_info.shape.len();
    let perm_norm = if perm.is_empty() {
        (0..rank).rev().collect::<Vec<_>>()
    } else {
        normalize_axes(perm, rank)
    };
    let dtype_str = dtype_to_mlir(&src_info.dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let mut dst_shape = vec![ShapeDim::Known(1); rank];
    for (i, &axis) in perm_norm.iter().enumerate() {
        if let Some(dim) = src_info.shape.get(axis) {
            dst_shape[i] = dim.clone();
        }
    }
    let dst_ty = tensor_type(&dst_shape, dtype_str);
    let perm_str = perm_norm
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    emitter.write_fmt(format_args!(
        "    %{} = linalg.transpose %{} [ {} ] : {} -> {}\n",
        dst.0, src.0, perm_str, src_ty, dst_ty
    ));
    emitter.record_tensor(dst, src_info.dtype, dst_shape);
}

fn emit_dot(emitter: &mut MlirEmitter, dst: ValueId, a: ValueId, b: ValueId) {
    let dtype = emitter
        .tensor_info(&a)
        .map(|info| info.dtype.clone())
        .or_else(|| emitter.tensor_info(&b).map(|info| info.dtype.clone()))
        .unwrap_or(DType::F32);
    let dtype_str = dtype_to_mlir(&dtype);
    let vec_ty = format!("tensor<?x{}>", dtype_str);
    let out_ty = tensor_type(&[], dtype_str);
    emitter.write_fmt(format_args!(
        "    %{} = linalg.dot ins(%{} : {vec_ty}, %{} : {vec_ty}) -> {out_ty}\n",
        dst.0,
        a.0,
        b.0,
        vec_ty = vec_ty,
        out_ty = out_ty
    ));
    emitter.record_tensor(dst, dtype, vec![]);
}

fn emit_matmul(emitter: &mut MlirEmitter, dst: ValueId, a: ValueId, b: ValueId) {
    let a_info = emitter
        .tensor_info(&a)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let b_info = emitter
        .tensor_info(&b)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: a_info.dtype.clone(),
            shape: vec![],
        });
    let dtype = a_info.dtype.clone();
    let dtype_str = dtype_to_mlir(&dtype);
    let a_ty = if a_info.shape.is_empty() {
        format!("tensor<?x{}>", dtype_str)
    } else {
        tensor_type(&a_info.shape, dtype_str)
    };
    let b_ty = if b_info.shape.is_empty() {
        format!("tensor<?x{}>", dtype_str)
    } else {
        tensor_type(&b_info.shape, dtype_str)
    };
    let mut out_shape = Vec::new();
    if a_info.shape.len() >= 2 && b_info.shape.len() >= 2 {
        for dim in &a_info.shape[..a_info.shape.len() - 1] {
            out_shape.push(dim.clone());
        }
        out_shape.push(b_info.shape.last().cloned().unwrap_or(ShapeDim::Known(1)));
    }
    let out_ty = if out_shape.is_empty() {
        format!("tensor<?x{}>", dtype_str)
    } else {
        tensor_type(&out_shape, dtype_str)
    };
    emitter.write_fmt(format_args!(
        "    %{} = linalg.matmul ins(%{} : {}, %{} : {}) -> {}\n",
        dst.0, a.0, a_ty, b.0, b_ty, out_ty
    ));
    emitter.record_tensor(dst, dtype, out_shape);
}

fn emit_index(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, indices: &[IndexSpec]) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let dtype_str = dtype_to_mlir(&src_info.dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let ordered: Vec<String> = if src_info.shape.is_empty() {
        Vec::new()
    } else {
        let mut coords = vec!["0".to_string(); src_info.shape.len()];
        for spec in indices {
            let axis = spec.axis.clamp(0, src_info.shape.len() as i64 - 1) as usize;
            coords[axis] = spec.index.to_string();
        }
        coords
    };
    let indices_str = ordered.join(", ");
    emitter.write_fmt(format_args!(
        "    %{} = tensor.extract %{}[{}] : {}\n",
        dst.0, src.0, indices_str, src_ty
    ));
    emitter.record_tensor(dst, src_info.dtype, vec![]);
}

fn emit_slice(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, dims: &[SliceSpec]) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let dtype_str = dtype_to_mlir(&src_info.dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let rank = src_info.shape.len();
    let mut offsets = if rank == 0 {
        Vec::new()
    } else {
        vec!["0".to_string(); rank]
    };
    let mut sizes = Vec::new();
    let mut strides = if rank == 0 {
        Vec::new()
    } else {
        vec!["1".to_string(); rank]
    };
    for spec in dims {
        if rank == 0 {
            continue;
        }
        let axis = spec.axis.clamp(0, rank as i64 - 1) as usize;
        offsets[axis] = spec.start.to_string();
        if let Some(end) = spec.end {
            let size = if let Some(ShapeDim::Known(_dim)) = src_info.shape.get(axis) {
                let start = spec.start.max(0) as usize;
                let end = end.max(spec.start) as usize;
                (end - start).max(1)
            } else {
                1
            };
            sizes.push((axis, size.to_string()));
        }
        strides[axis] = spec.stride.max(1).to_string();
    }
    sizes.sort_by_key(|(axis, _)| *axis);
    let mut final_sizes = vec![];
    for i in 0..rank {
        if let Some((_, size)) = sizes.iter().find(|(axis, _)| *axis == i) {
            final_sizes.push(size.clone());
        } else if let Some(dim) = src_info.shape.get(i) {
            match dim {
                ShapeDim::Known(n) => final_sizes.push(n.to_string()),
                ShapeDim::Sym(sym) => final_sizes.push(sym.to_string()),
            }
        } else {
            final_sizes.push("1".to_string());
        }
    }
    let dst_shape = dims.iter().fold(src_info.shape.clone(), |mut acc, spec| {
        if acc.is_empty() {
            return acc;
        }
        let axis = spec.axis.clamp(0, acc.len() as i64 - 1) as usize;
        if let Some(end) = spec.end {
            let start = spec.start.max(0) as usize;
            let end = end.max(spec.start) as usize;
            let stride = spec.stride.max(1) as usize;
            let len = (end - start).div_ceil(stride);
            acc[axis] = ShapeDim::Known(len);
        }
        acc
    });
    let dst_ty = tensor_type(&dst_shape, dtype_str);
    let offsets_str = if offsets.is_empty() {
        String::new()
    } else {
        offsets.join(", ")
    };
    let sizes_str = if final_sizes.is_empty() {
        String::new()
    } else {
        final_sizes.join(", ")
    };
    let strides_str = if strides.is_empty() {
        String::new()
    } else {
        strides.join(", ")
    };
    emitter.write_fmt(format_args!(
        "    %{} = tensor.extract_slice %{}[{}] [{}] [{}] : {} to {}\n",
        dst.0, src.0, offsets_str, sizes_str, strides_str, src_ty, dst_ty
    ));
    emitter.record_tensor(dst, src_info.dtype, dst_shape);
}

fn emit_gather(emitter: &mut MlirEmitter, dst: ValueId, src: ValueId, indices: ValueId, axis: i64) {
    let src_info = emitter
        .tensor_info(&src)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::F32,
            shape: vec![],
        });
    let idx_info = emitter
        .tensor_info(&indices)
        .cloned()
        .unwrap_or_else(|| TensorInfo {
            dtype: DType::I32,
            shape: vec![],
        });
    let dtype_str = dtype_to_mlir(&src_info.dtype);
    let src_ty = tensor_type(&src_info.shape, dtype_str);
    let idx_ty = tensor_type(&idx_info.shape, "i32");
    let axis = axis.clamp(0, src_info.shape.len() as i64 - 1) as usize;
    let mut result_shape = src_info.shape.clone();
    if let Some(dim) = idx_info.shape.first() {
        result_shape[axis] = dim.clone();
    }
    let result_ty = tensor_type(&result_shape, dtype_str);
    let empty_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "    {empty} = tensor.empty() : {result_ty}\n",
        empty = empty_name,
        result_ty = result_ty
    ));
    let upper_bound = idx_info
        .shape
        .first()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(s) => s.to_string(),
        })
        .unwrap_or_else(|| "0".to_string());
    let loop_result = format!("%{}", dst.0);
    emitter.write_fmt(format_args!(
        "    {loop_result} = scf.for %i0 = 0 to {upper} step 1 iter_args(%acc = {empty}) -> {result_ty} {{\n",
        loop_result = loop_result,
        upper = upper_bound,
        empty = empty_name,
        result_ty = result_ty
    ));
    let idx_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "      {idx_name} = tensor.extract %{}[%i0] : {}\n",
        indices.0,
        idx_ty,
        idx_name = idx_name
    ));
    let val_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "      {val_name} = tensor.extract %{}[{idx_name}] : {}\n",
        src.0,
        src_ty,
        val_name = val_name,
        idx_name = idx_name
    ));
    let updated_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "      {updated} = tensor.insert {val_name} into %acc[%i0] : {result_ty}\n",
        updated = updated_name,
        val_name = val_name,
        result_ty = result_ty
    ));
    emitter.write_fmt(format_args!(
        "      scf.yield {updated} : {result_ty}\n",
        updated = updated_name,
        result_ty = result_ty
    ));
    emitter.write_line("    }");
    emitter.record_tensor(dst, src_info.dtype, result_shape);
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

fn normalize_axes(axes: &[i64], rank: usize) -> Vec<usize> {
    axes.iter()
        .map(|axis| {
            let mut idx = *axis;
            if idx < 0 {
                idx += rank as i64;
            }
            idx.clamp(0, rank.saturating_sub(1) as i64) as usize
        })
        .collect()
}

fn reduce_shape(shape: &[ShapeDim], axes: &[usize], keepdims: bool) -> Vec<ShapeDim> {
    if keepdims {
        let mut out = shape.to_vec();
        for &axis in axes {
            if axis < out.len() {
                out[axis] = ShapeDim::Known(1);
            }
        }
        return out;
    }

    shape
        .iter()
        .enumerate()
        .filter_map(|(i, dim)| {
            if axes.contains(&i) {
                None
            } else {
                Some(dim.clone())
            }
        })
        .collect()
}

fn format_dimensions(axes: &[usize]) -> String {
    axes.iter()
        .map(|axis| axis.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn element_count(shape: &[ShapeDim], axes: &[usize]) -> usize {
    axes.iter()
        .map(|&axis| match shape.get(axis) {
            Some(ShapeDim::Known(n)) => *n,
            _ => 1,
        })
        .product()
}

/// Emit MLIR for Conv2dGradInput (backward pass w.r.t. input).
///
/// Emits a stub that allocates a zero-filled output tensor of the correct shape.
/// Full computation is deferred to runtime or a specialized backend lowering pass.
#[allow(clippy::too_many_arguments)]
fn emit_conv2d_grad_input(
    emitter: &mut MlirEmitter,
    dst: ValueId,
    dy: ValueId,
    filter: ValueId,
    input_shape: [usize; 4],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) {
    let [n, h, w, c] = input_shape;
    let dtype_str = "f32";
    let out_ty = format!("tensor<{}x{}x{}x{}x{}>", n, h, w, c, dtype_str);

    // Allocate output tensor initialized to zero
    let empty_name = emitter.tmp();
    let zero_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "    {} = tensor.empty() : {}\n",
        empty_name, out_ty
    ));
    emitter.write_fmt(format_args!(
        "    {} = arith.constant 0.0 : {}\n",
        zero_name, dtype_str
    ));
    emitter.write_fmt(format_args!(
        "    %{} = linalg.fill ins({} : {}) outs({} : {}) -> {}\n",
        dst.0, zero_name, dtype_str, empty_name, out_ty, out_ty
    ));

    // Record padding and stride info as comments for debugging
    let padding_str = match padding {
        ConvPadding::Valid => "valid",
        ConvPadding::Same => "same",
    };
    emitter.write_fmt(format_args!(
        "    // conv2d_grad_input: dy=%{}, filter=%{}, strides=({},{}), padding={}\n",
        dy.0, filter.0, stride_h, stride_w, padding_str
    ));
    emitter.write_line("    // Full computation deferred to runtime or specialized lowering");

    emitter.record_tensor(
        dst,
        DType::F32,
        input_shape.iter().map(|d| ShapeDim::Known(*d)).collect(),
    );
}

/// Emit MLIR for Conv2dGradFilter (backward pass w.r.t. filter).
///
/// Emits a stub that allocates a zero-filled output tensor of the correct shape.
/// Full computation is deferred to runtime or a specialized backend lowering pass.
#[allow(clippy::too_many_arguments)]
fn emit_conv2d_grad_filter(
    emitter: &mut MlirEmitter,
    dst: ValueId,
    input: ValueId,
    dy: ValueId,
    filter_shape: [usize; 4],
    stride_h: usize,
    stride_w: usize,
    padding: ConvPadding,
) {
    let [kh, kw, c, o] = filter_shape;
    let dtype_str = "f32";
    let out_ty = format!("tensor<{}x{}x{}x{}x{}>", kh, kw, c, o, dtype_str);

    // Allocate output tensor initialized to zero
    let empty_name = emitter.tmp();
    let zero_name = emitter.tmp();
    emitter.write_fmt(format_args!(
        "    {} = tensor.empty() : {}\n",
        empty_name, out_ty
    ));
    emitter.write_fmt(format_args!(
        "    {} = arith.constant 0.0 : {}\n",
        zero_name, dtype_str
    ));
    emitter.write_fmt(format_args!(
        "    %{} = linalg.fill ins({} : {}) outs({} : {}) -> {}\n",
        dst.0, zero_name, dtype_str, empty_name, out_ty, out_ty
    ));

    // Record padding and stride info as comments for debugging
    let padding_str = match padding {
        ConvPadding::Valid => "valid",
        ConvPadding::Same => "same",
    };
    emitter.write_fmt(format_args!(
        "    // conv2d_grad_filter: input=%{}, dy=%{}, strides=({},{}), padding={}\n",
        input.0, dy.0, stride_h, stride_w, padding_str
    ));
    emitter.write_line("    // Full computation deferred to runtime or specialized lowering");

    emitter.record_tensor(
        dst,
        DType::F32,
        filter_shape.iter().map(|d| ShapeDim::Known(*d)).collect(),
    );
}
