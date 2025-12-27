// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! MIC format emitter - Serializes IRModule to MindIR Compact format.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ir::{BinOp, IRModule, Instr};
use crate::types::{DType, ShapeDim};

use super::MIC_HEADER;

/// Emits an IRModule as MIC format string.
///
/// The output is canonical: same input always produces same output.
pub fn emit_mic(module: &IRModule) -> String {
    let mut emitter = MicEmitter::new();
    emitter.emit(module)
}

/// MIC format emitter with interning support.
pub struct MicEmitter {
    /// Symbol table: name -> id
    symbols: HashMap<String, usize>,
    /// Type table: type_key -> id
    types: HashMap<String, usize>,
    /// Output buffer
    output: String,
}

impl MicEmitter {
    /// Create a new emitter.
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            types: HashMap::new(),
            output: String::with_capacity(4096),
        }
    }

    /// Emit an IRModule to MIC format.
    pub fn emit(&mut self, module: &IRModule) -> String {
        self.output.clear();
        self.symbols.clear();
        self.types.clear();

        // Phase 1: Collect all symbols and types
        self.collect_symbols_and_types(module);

        // Phase 2: Emit header
        writeln!(&mut self.output, "{}", MIC_HEADER).unwrap();

        // Phase 3: Emit symbol table
        let mut sorted_symbols: Vec<_> = self.symbols.iter().collect();
        sorted_symbols.sort_by_key(|(_, id)| *id);
        for (name, id) in sorted_symbols {
            writeln!(&mut self.output, "S{} \"{}\"", id, escape_string(name)).unwrap();
        }

        // Phase 4: Emit type table
        let mut sorted_types: Vec<_> = self.types.iter().collect();
        sorted_types.sort_by_key(|(_, id)| *id);
        for (type_key, id) in sorted_types {
            writeln!(&mut self.output, "T{} {}", id, type_key).unwrap();
        }

        // Phase 5: Emit nodes
        for instr in &module.instrs {
            self.emit_instr(instr);
        }

        self.output.clone()
    }

    fn collect_symbols_and_types(&mut self, module: &IRModule) {
        for instr in &module.instrs {
            match instr {
                Instr::ConstTensor(_, dtype, shape, _) => {
                    self.intern_type(dtype, shape);
                }
                Instr::ConstI64(_, _) => {
                    self.intern_scalar_type(&DType::I32);
                }
                Instr::BinOp { .. } => {
                    // Type inferred from operands, use f32 as default
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Sum { .. } | Instr::Mean { .. } => {
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Reshape { new_shape, .. } => {
                    self.intern_type(&DType::F32, new_shape);
                }
                Instr::Transpose { perm, .. } => {
                    // Emit inferred type placeholder
                    let shape: Vec<ShapeDim> = perm.iter().map(|_| ShapeDim::Sym("?")).collect();
                    self.intern_type(&DType::F32, &shape);
                }
                Instr::ExpandDims { .. } | Instr::Squeeze { .. } => {
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Dot { .. } | Instr::MatMul { .. } => {
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Conv2d { .. }
                | Instr::Conv2dGradInput { .. }
                | Instr::Conv2dGradFilter { .. } => {
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Index { .. } | Instr::Slice { .. } | Instr::Gather { .. } => {
                    self.intern_scalar_type(&DType::F32);
                }
                Instr::Output(_) => {}
            }
        }
    }

    #[allow(dead_code)]
    fn intern_symbol(&mut self, name: &str) -> usize {
        let next_id = self.symbols.len();
        *self.symbols.entry(name.to_string()).or_insert(next_id)
    }

    fn intern_scalar_type(&mut self, dtype: &DType) -> usize {
        let key = dtype.as_str().to_string();
        let next_id = self.types.len();
        *self.types.entry(key).or_insert(next_id)
    }

    fn intern_type(&mut self, dtype: &DType, shape: &[ShapeDim]) -> usize {
        let key = format_type(dtype, shape);
        let next_id = self.types.len();
        *self.types.entry(key).or_insert(next_id)
    }

    fn get_type_id(&self, dtype: &DType, shape: &[ShapeDim]) -> usize {
        let key = format_type(dtype, shape);
        *self.types.get(&key).unwrap_or(&0)
    }

    fn get_scalar_type_id(&self, dtype: &DType) -> usize {
        let key = dtype.as_str().to_string();
        *self.types.get(&key).unwrap_or(&0)
    }

    fn emit_instr(&mut self, instr: &Instr) {
        match instr {
            Instr::ConstI64(id, value) => {
                let tid = self.get_scalar_type_id(&DType::I32);
                writeln!(&mut self.output, "N{} const.i64 {} T{}", id.0, value, tid).unwrap();
            }
            Instr::ConstTensor(id, dtype, shape, fill) => {
                let tid = self.get_type_id(dtype, shape);
                let fill_str = fill.map(|f| format!(" fill={}", f)).unwrap_or_default();
                writeln!(
                    &mut self.output,
                    "N{} const.tensor{} T{}",
                    id.0, fill_str, tid
                )
                .unwrap();
            }
            Instr::BinOp { dst, op, lhs, rhs } => {
                let op_str = binop_str(*op);
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} {} N{} N{} T{}",
                    dst.0, op_str, lhs.0, rhs.0, tid
                )
                .unwrap();
            }
            Instr::Sum {
                dst,
                src,
                axes,
                keepdims,
            } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let axes_str = format_axes(axes);
                let kd = if *keepdims { 1 } else { 0 };
                writeln!(
                    &mut self.output,
                    "N{} sum N{} {} kd={} T{}",
                    dst.0, src.0, axes_str, kd, tid
                )
                .unwrap();
            }
            Instr::Mean {
                dst,
                src,
                axes,
                keepdims,
            } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let axes_str = format_axes(axes);
                let kd = if *keepdims { 1 } else { 0 };
                writeln!(
                    &mut self.output,
                    "N{} mean N{} {} kd={} T{}",
                    dst.0, src.0, axes_str, kd, tid
                )
                .unwrap();
            }
            Instr::Reshape {
                dst,
                src,
                new_shape,
            } => {
                let tid = self.get_type_id(&DType::F32, new_shape);
                let shape_str = format_shape_dims(new_shape);
                writeln!(
                    &mut self.output,
                    "N{} reshape N{} {} T{}",
                    dst.0, src.0, shape_str, tid
                )
                .unwrap();
            }
            Instr::ExpandDims { dst, src, axis } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} expand N{} [{}] T{}",
                    dst.0, src.0, axis, tid
                )
                .unwrap();
            }
            Instr::Squeeze { dst, src, axes } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let axes_str = format_axes(axes);
                writeln!(
                    &mut self.output,
                    "N{} squeeze N{} {} T{}",
                    dst.0, src.0, axes_str, tid
                )
                .unwrap();
            }
            Instr::Transpose { dst, src, perm } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let perm_str = format_axes(perm);
                writeln!(
                    &mut self.output,
                    "N{} transpose N{} {} T{}",
                    dst.0, src.0, perm_str, tid
                )
                .unwrap();
            }
            Instr::Dot { dst, a, b } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} dot N{} N{} T{}",
                    dst.0, a.0, b.0, tid
                )
                .unwrap();
            }
            Instr::MatMul { dst, a, b } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} matmul N{} N{} T{}",
                    dst.0, a.0, b.0, tid
                )
                .unwrap();
            }
            Instr::Conv2d {
                dst,
                input,
                filter,
                stride_h,
                stride_w,
                padding,
            } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} conv2d N{} N{} s=[{},{}] p={} T{}",
                    dst.0,
                    input.0,
                    filter.0,
                    stride_h,
                    stride_w,
                    padding.as_str(),
                    tid
                )
                .unwrap();
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
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} conv2d.grad.input N{} N{} is=[{},{},{},{}] s=[{},{}] p={} T{}",
                    dst.0,
                    dy.0,
                    filter.0,
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    input_shape[3],
                    stride_h,
                    stride_w,
                    padding.as_str(),
                    tid
                )
                .unwrap();
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
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} conv2d.grad.filter N{} N{} fs=[{},{},{},{}] s=[{},{}] p={} T{}",
                    dst.0,
                    input.0,
                    dy.0,
                    filter_shape[0],
                    filter_shape[1],
                    filter_shape[2],
                    filter_shape[3],
                    stride_h,
                    stride_w,
                    padding.as_str(),
                    tid
                )
                .unwrap();
            }
            Instr::Index { dst, src, indices } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let indices_str: Vec<String> = indices
                    .iter()
                    .map(|idx| format!("{}:{}", idx.axis, idx.index))
                    .collect();
                writeln!(
                    &mut self.output,
                    "N{} index N{} [{}] T{}",
                    dst.0,
                    src.0,
                    indices_str.join(","),
                    tid
                )
                .unwrap();
            }
            Instr::Slice { dst, src, dims } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                let dims_str: Vec<String> = dims
                    .iter()
                    .map(|d| {
                        let end_str = d.end.map(|e| e.to_string()).unwrap_or_default();
                        format!("{}:{}:{}", d.start, end_str, d.stride)
                    })
                    .collect();
                writeln!(
                    &mut self.output,
                    "N{} slice N{} {} T{}",
                    dst.0,
                    src.0,
                    dims_str.join(","),
                    tid
                )
                .unwrap();
            }
            Instr::Gather {
                dst,
                src,
                indices,
                axis,
            } => {
                let tid = self.get_scalar_type_id(&DType::F32);
                writeln!(
                    &mut self.output,
                    "N{} gather N{} N{} ax={} T{}",
                    dst.0, src.0, indices.0, axis, tid
                )
                .unwrap();
            }
            Instr::Output(id) => {
                writeln!(&mut self.output, "O N{}", id.0).unwrap();
            }
        }
    }
}

impl Default for MicEmitter {
    fn default() -> Self {
        Self::new()
    }
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
    }
}

fn format_type(dtype: &DType, shape: &[ShapeDim]) -> String {
    if shape.is_empty() {
        dtype.as_str().to_string()
    } else {
        let shape_str = format_shape_dims(shape);
        format!(
            "[{};{}]",
            dtype.as_str(),
            &shape_str[1..shape_str.len() - 1]
        )
    }
}

fn format_shape_dims(shape: &[ShapeDim]) -> String {
    let dims: Vec<String> = shape
        .iter()
        .map(|d| match d {
            ShapeDim::Known(n) => n.to_string(),
            ShapeDim::Sym(s) => s.to_string(),
        })
        .collect();
    format!("[{}]", dims.join(","))
}

fn format_axes(axes: &[i64]) -> String {
    let strs: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
    format!("[{}]", strs.join(","))
}

fn escape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\t' => result.push_str("\\t"),
            '\r' => result.push_str("\\r"),
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("a\"b"), "a\\\"b");
        assert_eq!(escape_string("a\\b"), "a\\\\b");
        assert_eq!(escape_string("a\nb"), "a\\nb");
    }

    #[test]
    fn test_format_type() {
        assert_eq!(format_type(&DType::F32, &[]), "f32");
        assert_eq!(
            format_type(&DType::F32, &[ShapeDim::Known(3), ShapeDim::Known(4)]),
            "[f32;3,4]"
        );
    }

    #[test]
    fn test_emit_empty_module() {
        let module = IRModule::new();
        let mic = emit_mic(&module);
        assert_eq!(mic.trim(), "mic@1");
    }
}
