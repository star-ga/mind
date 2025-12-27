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

//! MIC format parser - Deserializes MindIR Compact format to IRModule.

use std::collections::HashMap;

use crate::ir::{BinOp, IRModule, IndexSpec, Instr, SliceSpec, ValueId};
use crate::types::intern::try_intern_str;
use crate::types::{ConvPadding, DType, ShapeDim};

use super::{MIC_HEADER, MIC_VERSION};

/// Maximum input size in bytes (10 MB).
pub const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of lines to parse.
pub const MAX_LINE_COUNT: usize = 1_000_000;

/// Maximum number of nodes in a single module.
pub const MAX_NODE_COUNT: usize = 100_000;

/// Maximum shape dimension count per tensor.
pub const MAX_SHAPE_DIMS: usize = 32;

/// Error type for MIC parsing.
#[derive(Debug, Clone)]
pub struct MicParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for MicParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mic:{}: error: {}", self.line, self.message)
    }
}

impl std::error::Error for MicParseError {}

/// Parse a MIC format string into an IRModule.
///
/// # Security Limits
///
/// This function enforces the following limits to prevent DoS attacks:
/// - Maximum input size: 10 MB
/// - Maximum line count: 1,000,000
/// - Maximum node count: 100,000
/// - Maximum shape dimensions: 32
pub fn parse_mic(input: &str) -> Result<IRModule, MicParseError> {
    // Security: Check input size
    if input.len() > MAX_INPUT_SIZE {
        return Err(MicParseError {
            line: 0,
            message: format!(
                "input too large: {} bytes (max {} bytes)",
                input.len(),
                MAX_INPUT_SIZE
            ),
        });
    }

    let mut parser = MicParser::new(input);

    // Security: Check line count
    if parser.lines.len() > MAX_LINE_COUNT {
        return Err(MicParseError {
            line: 0,
            message: format!(
                "too many lines: {} (max {})",
                parser.lines.len(),
                MAX_LINE_COUNT
            ),
        });
    }

    parser.parse()
}

/// MIC format parser.
struct MicParser<'a> {
    lines: Vec<&'a str>,
    current_line: usize,
    symbols: HashMap<usize, String>,
    types: HashMap<usize, TypeInfo>,
    nodes: HashMap<usize, bool>,
    module: IRModule,
}

#[derive(Debug, Clone)]
struct TypeInfo {
    dtype: DType,
    shape: Vec<ShapeDim>,
}

impl<'a> MicParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines().collect(),
            current_line: 0,
            symbols: HashMap::new(),
            types: HashMap::new(),
            nodes: HashMap::new(),
            module: IRModule::new(),
        }
    }

    fn parse(&mut self) -> Result<IRModule, MicParseError> {
        // Parse version header
        self.parse_header()?;

        // Parse remaining lines
        while self.current_line < self.lines.len() {
            let line = self.lines[self.current_line].trim();
            self.current_line += 1;

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Dispatch based on entry type
            if line.starts_with('S') {
                self.parse_symbol(line)?;
            } else if line.starts_with('T') {
                self.parse_type(line)?;
            } else if line.starts_with('N') {
                self.parse_node(line)?;
            } else if line.starts_with('O') {
                self.parse_output(line)?;
            } else {
                return Err(self.error(format!("unknown entry type: {}", line)));
            }
        }

        Ok(std::mem::take(&mut self.module))
    }

    fn parse_header(&mut self) -> Result<(), MicParseError> {
        // Find first non-empty, non-comment line
        while self.current_line < self.lines.len() {
            let line = self.lines[self.current_line].trim();
            self.current_line += 1;

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Check version header
            if line == MIC_HEADER {
                return Ok(());
            } else if let Some(version_str) = line.strip_prefix("mic@") {
                if let Ok(version) = version_str.parse::<u32>() {
                    if version != MIC_VERSION {
                        return Err(self.error(format!(
                            "unsupported version mic@{}, expected mic@{}",
                            version, MIC_VERSION
                        )));
                    }
                }
                return Err(self.error(format!("invalid version header: {}", line)));
            } else {
                return Err(self.error("missing version header, expected mic@1".to_string()));
            }
        }

        Err(self.error("empty input".to_string()))
    }

    fn parse_symbol(&mut self, line: &str) -> Result<(), MicParseError> {
        // Format: S<id> "<name>"
        let rest = &line[1..];
        let (id_str, name_part) = rest
            .split_once(' ')
            .ok_or_else(|| self.error("invalid symbol format".to_string()))?;

        let id: usize = id_str
            .parse()
            .map_err(|_| self.error(format!("invalid symbol id: {}", id_str)))?;

        // Parse quoted string
        let name = self.parse_quoted_string(name_part.trim())?;
        self.symbols.insert(id, name);
        Ok(())
    }

    fn parse_type(&mut self, line: &str) -> Result<(), MicParseError> {
        // Format: T<id> <typespec>
        let rest = &line[1..];
        let (id_str, type_part) = rest
            .split_once(' ')
            .ok_or_else(|| self.error("invalid type format".to_string()))?;

        let id: usize = id_str
            .parse()
            .map_err(|_| self.error(format!("invalid type id: {}", id_str)))?;

        let type_info = self.parse_type_spec(type_part.trim())?;
        self.types.insert(id, type_info);
        Ok(())
    }

    fn parse_type_spec(&self, spec: &str) -> Result<TypeInfo, MicParseError> {
        if spec.starts_with('[') && spec.ends_with(']') {
            // Tensor type: [dtype;shape]
            let inner = &spec[1..spec.len() - 1];
            let (dtype_str, shape_str) = inner
                .split_once(';')
                .ok_or_else(|| self.error(format!("invalid tensor type: {}", spec)))?;

            let dtype = DType::parse(dtype_str)
                .ok_or_else(|| self.error(format!("invalid dtype: {}", dtype_str)))?;

            let shape = if shape_str == "?" {
                vec![ShapeDim::Sym("?")]
            } else {
                self.parse_shape(shape_str)?
            };

            Ok(TypeInfo { dtype, shape })
        } else {
            // Scalar type
            let dtype =
                DType::parse(spec).ok_or_else(|| self.error(format!("invalid dtype: {}", spec)))?;
            Ok(TypeInfo {
                dtype,
                shape: vec![],
            })
        }
    }

    fn parse_shape(&self, shape_str: &str) -> Result<Vec<ShapeDim>, MicParseError> {
        if shape_str.is_empty() {
            return Ok(vec![]);
        }

        let parts: Vec<&str> = shape_str.split(',').collect();

        // Security: Check shape dimension count
        if parts.len() > MAX_SHAPE_DIMS {
            return Err(self.error(format!(
                "too many shape dimensions: {} (max {})",
                parts.len(),
                MAX_SHAPE_DIMS
            )));
        }

        parts
            .into_iter()
            .map(|s| {
                let s = s.trim();
                if let Ok(n) = s.parse::<usize>() {
                    Ok(ShapeDim::Known(n))
                } else {
                    // Symbolic dimension - use thread-safe string interning with fail-fast
                    match try_intern_str(s) {
                        Some(interned) => Ok(ShapeDim::Sym(interned)),
                        None => Err(self.error(format!(
                            "string interner capacity exceeded for symbol '{}' - too many unique symbolic dimensions",
                            s
                        ))),
                    }
                }
            })
            .collect()
    }

    fn parse_node(&mut self, line: &str) -> Result<(), MicParseError> {
        // Security: Check node count limit
        if self.nodes.len() >= MAX_NODE_COUNT {
            return Err(self.error(format!(
                "too many nodes: {} (max {})",
                self.nodes.len(),
                MAX_NODE_COUNT
            )));
        }

        // Format: N<id> <kind> <args...>
        let rest = &line[1..];
        let parts: Vec<&str> = rest.split_whitespace().collect();

        if parts.is_empty() {
            return Err(self.error("empty node definition".to_string()));
        }

        let id: usize = parts[0]
            .parse()
            .map_err(|_| self.error(format!("invalid node id: {}", parts[0])))?;

        if parts.len() < 2 {
            return Err(self.error("missing node kind".to_string()));
        }

        let kind = parts[1];
        let args = &parts[2..];

        // Allocate ValueId if needed
        while self.module.next_id <= id {
            self.module.fresh();
        }
        let value_id = ValueId(id);

        let instr = match kind {
            "const.i64" => self.parse_const_i64(value_id, args)?,
            "const.tensor" => self.parse_const_tensor(value_id, args)?,
            "add" => self.parse_binop(value_id, BinOp::Add, args)?,
            "sub" => self.parse_binop(value_id, BinOp::Sub, args)?,
            "mul" => self.parse_binop(value_id, BinOp::Mul, args)?,
            "div" => self.parse_binop(value_id, BinOp::Div, args)?,
            "sum" => self.parse_reduction(value_id, true, args)?,
            "mean" => self.parse_reduction(value_id, false, args)?,
            "reshape" => self.parse_reshape(value_id, args)?,
            "expand" => self.parse_expand(value_id, args)?,
            "squeeze" => self.parse_squeeze(value_id, args)?,
            "transpose" => self.parse_transpose(value_id, args)?,
            "dot" => self.parse_dot(value_id, args)?,
            "matmul" => self.parse_matmul(value_id, args)?,
            "conv2d" => self.parse_conv2d(value_id, args)?,
            "conv2d.grad.input" => self.parse_conv2d_grad_input(value_id, args)?,
            "conv2d.grad.filter" => self.parse_conv2d_grad_filter(value_id, args)?,
            "index" => self.parse_index(value_id, args)?,
            "slice" => self.parse_slice(value_id, args)?,
            "gather" => self.parse_gather(value_id, args)?,
            _ => return Err(self.error(format!("unknown node kind: {}", kind))),
        };

        self.nodes.insert(id, true);
        self.module.instrs.push(instr);
        Ok(())
    }

    fn parse_const_i64(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("const.i64 requires value".to_string()));
        }
        let value: i64 = args[0]
            .parse()
            .map_err(|_| self.error(format!("invalid i64 value: {}", args[0])))?;
        Ok(Instr::ConstI64(id, value))
    }

    fn parse_const_tensor(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        // Parse fill=<value> and T<id>
        let mut fill: Option<f64> = None;
        let mut type_id: Option<usize> = None;

        for arg in args {
            if let Some(fill_val) = arg.strip_prefix("fill=") {
                fill = Some(
                    fill_val
                        .parse()
                        .map_err(|_| self.error(format!("invalid fill value: {}", arg)))?,
                );
            } else if let Some(type_str) = arg.strip_prefix('T') {
                type_id = Some(
                    type_str
                        .parse()
                        .map_err(|_| self.error(format!("invalid type ref: {}", arg)))?,
                );
            }
        }

        let tid = type_id.ok_or_else(|| self.error("const.tensor requires type".to_string()))?;
        let type_info = self
            .types
            .get(&tid)
            .ok_or_else(|| self.error(format!("undefined type T{}", tid)))?;

        Ok(Instr::ConstTensor(
            id,
            type_info.dtype.clone(),
            type_info.shape.clone(),
            fill,
        ))
    }

    fn parse_binop(&self, id: ValueId, op: BinOp, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error(format!("{:?} requires 2 operands", op)));
        }

        let lhs = self.parse_node_ref(args[0])?;
        let rhs = self.parse_node_ref(args[1])?;

        self.check_node_defined(lhs.0)?;
        self.check_node_defined(rhs.0)?;

        Ok(Instr::BinOp {
            dst: id,
            op,
            lhs,
            rhs,
        })
    }

    fn parse_reduction(
        &self,
        id: ValueId,
        is_sum: bool,
        args: &[&str],
    ) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("reduction requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut axes = vec![];
        let mut keepdims = false;

        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                axes = self.parse_i64_list(&arg[1..arg.len() - 1])?;
            } else if arg.starts_with("kd=") {
                keepdims = &arg[3..] == "1";
            }
        }

        if is_sum {
            Ok(Instr::Sum {
                dst: id,
                src,
                axes,
                keepdims,
            })
        } else {
            Ok(Instr::Mean {
                dst: id,
                src,
                axes,
                keepdims,
            })
        }
    }

    fn parse_reshape(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("reshape requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut new_shape = vec![];
        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                let shape_str = &arg[1..arg.len() - 1];
                new_shape = self.parse_shape(shape_str)?;
            }
        }

        Ok(Instr::Reshape {
            dst: id,
            src,
            new_shape,
        })
    }

    fn parse_expand(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("expand requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut axis = 0i64;
        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                let axes = self.parse_i64_list(&arg[1..arg.len() - 1])?;
                if !axes.is_empty() {
                    axis = axes[0];
                }
            }
        }

        Ok(Instr::ExpandDims { dst: id, src, axis })
    }

    fn parse_squeeze(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("squeeze requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut axes = vec![];
        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                axes = self.parse_i64_list(&arg[1..arg.len() - 1])?;
            }
        }

        Ok(Instr::Squeeze { dst: id, src, axes })
    }

    fn parse_transpose(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("transpose requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut perm = vec![];
        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                perm = self.parse_i64_list(&arg[1..arg.len() - 1])?;
            }
        }

        Ok(Instr::Transpose { dst: id, src, perm })
    }

    fn parse_dot(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("dot requires 2 operands".to_string()));
        }

        let a = self.parse_node_ref(args[0])?;
        let b = self.parse_node_ref(args[1])?;
        self.check_node_defined(a.0)?;
        self.check_node_defined(b.0)?;

        Ok(Instr::Dot { dst: id, a, b })
    }

    fn parse_matmul(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("matmul requires 2 operands".to_string()));
        }

        let a = self.parse_node_ref(args[0])?;
        let b = self.parse_node_ref(args[1])?;
        self.check_node_defined(a.0)?;
        self.check_node_defined(b.0)?;

        Ok(Instr::MatMul { dst: id, a, b })
    }

    fn parse_conv2d(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("conv2d requires input and filter".to_string()));
        }

        let input = self.parse_node_ref(args[0])?;
        let filter = self.parse_node_ref(args[1])?;
        self.check_node_defined(input.0)?;
        self.check_node_defined(filter.0)?;

        let mut stride_h = 1usize;
        let mut stride_w = 1usize;
        let mut padding = ConvPadding::Valid;

        for arg in &args[2..] {
            if arg.starts_with("s=[") {
                let strides_str = &arg[3..arg.len() - 1];
                let strides = self.parse_usize_list(strides_str)?;
                if strides.len() >= 2 {
                    stride_h = strides[0];
                    stride_w = strides[1];
                }
            } else if let Some(pad_str) = arg.strip_prefix("p=") {
                padding = ConvPadding::parse(pad_str)
                    .ok_or_else(|| self.error(format!("invalid padding: {}", arg)))?;
            }
        }

        Ok(Instr::Conv2d {
            dst: id,
            input,
            filter,
            stride_h,
            stride_w,
            padding,
        })
    }

    fn parse_conv2d_grad_input(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("conv2d.grad.input requires dy and filter".to_string()));
        }

        let dy = self.parse_node_ref(args[0])?;
        let filter = self.parse_node_ref(args[1])?;
        self.check_node_defined(dy.0)?;
        self.check_node_defined(filter.0)?;

        let mut input_shape = [0usize; 4];
        let mut stride_h = 1usize;
        let mut stride_w = 1usize;
        let mut padding = ConvPadding::Valid;

        for arg in &args[2..] {
            if arg.starts_with("is=[") {
                let shape_str = &arg[4..arg.len() - 1];
                let shape = self.parse_usize_list(shape_str)?;
                if shape.len() >= 4 {
                    input_shape = [shape[0], shape[1], shape[2], shape[3]];
                }
            } else if arg.starts_with("s=[") {
                let strides_str = &arg[3..arg.len() - 1];
                let strides = self.parse_usize_list(strides_str)?;
                if strides.len() >= 2 {
                    stride_h = strides[0];
                    stride_w = strides[1];
                }
            } else if let Some(pad_str) = arg.strip_prefix("p=") {
                padding = ConvPadding::parse(pad_str)
                    .ok_or_else(|| self.error(format!("invalid padding: {}", arg)))?;
            }
        }

        Ok(Instr::Conv2dGradInput {
            dst: id,
            dy,
            filter,
            input_shape,
            stride_h,
            stride_w,
            padding,
        })
    }

    fn parse_conv2d_grad_filter(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("conv2d.grad.filter requires input and dy".to_string()));
        }

        let input = self.parse_node_ref(args[0])?;
        let dy = self.parse_node_ref(args[1])?;
        self.check_node_defined(input.0)?;
        self.check_node_defined(dy.0)?;

        let mut filter_shape = [0usize; 4];
        let mut stride_h = 1usize;
        let mut stride_w = 1usize;
        let mut padding = ConvPadding::Valid;

        for arg in &args[2..] {
            if arg.starts_with("fs=[") {
                let shape_str = &arg[4..arg.len() - 1];
                let shape = self.parse_usize_list(shape_str)?;
                if shape.len() >= 4 {
                    filter_shape = [shape[0], shape[1], shape[2], shape[3]];
                }
            } else if arg.starts_with("s=[") {
                let strides_str = &arg[3..arg.len() - 1];
                let strides = self.parse_usize_list(strides_str)?;
                if strides.len() >= 2 {
                    stride_h = strides[0];
                    stride_w = strides[1];
                }
            } else if let Some(pad_str) = arg.strip_prefix("p=") {
                padding = ConvPadding::parse(pad_str)
                    .ok_or_else(|| self.error(format!("invalid padding: {}", arg)))?;
            }
        }

        Ok(Instr::Conv2dGradFilter {
            dst: id,
            input,
            dy,
            filter_shape,
            stride_h,
            stride_w,
            padding,
        })
    }

    fn parse_index(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("index requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut indices = vec![];
        for arg in &args[1..] {
            if arg.starts_with('[') && arg.ends_with(']') {
                let inner = &arg[1..arg.len() - 1];
                for pair in inner.split(',') {
                    if let Some((axis_str, idx_str)) = pair.split_once(':') {
                        let axis: i64 = axis_str.trim().parse().unwrap_or(0);
                        let index: i64 = idx_str.trim().parse().unwrap_or(0);
                        indices.push(IndexSpec { axis, index });
                    }
                }
            }
        }

        Ok(Instr::Index {
            dst: id,
            src,
            indices,
        })
    }

    fn parse_slice(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.is_empty() {
            return Err(self.error("slice requires source".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        self.check_node_defined(src.0)?;

        let mut dims = vec![];
        for arg in &args[1..] {
            if !arg.starts_with('T') && !arg.starts_with('[') {
                // Parse slice specs: start:end:stride,...
                for (axis, spec) in arg.split(',').enumerate() {
                    let parts: Vec<&str> = spec.split(':').collect();
                    if parts.len() >= 2 {
                        let start: i64 = parts[0].parse().unwrap_or(0);
                        let end: Option<i64> = if parts.len() > 1 && !parts[1].is_empty() {
                            parts[1].parse().ok()
                        } else {
                            None
                        };
                        let stride: i64 = if parts.len() > 2 {
                            parts[2].parse().unwrap_or(1)
                        } else {
                            1
                        };
                        dims.push(SliceSpec {
                            axis: axis as i64,
                            start,
                            end,
                            stride,
                        });
                    }
                }
            }
        }

        Ok(Instr::Slice { dst: id, src, dims })
    }

    fn parse_gather(&self, id: ValueId, args: &[&str]) -> Result<Instr, MicParseError> {
        if args.len() < 2 {
            return Err(self.error("gather requires source and indices".to_string()));
        }

        let src = self.parse_node_ref(args[0])?;
        let indices = self.parse_node_ref(args[1])?;
        self.check_node_defined(src.0)?;
        self.check_node_defined(indices.0)?;

        let mut axis = 0i64;
        for arg in &args[2..] {
            if let Some(ax_str) = arg.strip_prefix("ax=") {
                axis = ax_str.parse().unwrap_or(0);
            }
        }

        Ok(Instr::Gather {
            dst: id,
            src,
            indices,
            axis,
        })
    }

    fn parse_output(&mut self, line: &str) -> Result<(), MicParseError> {
        // Format: O N<id>
        let rest = line[1..].trim();
        let node_ref = self.parse_node_ref(rest)?;
        self.check_node_defined(node_ref.0)?;
        self.module.instrs.push(Instr::Output(node_ref));
        Ok(())
    }

    fn parse_node_ref(&self, s: &str) -> Result<ValueId, MicParseError> {
        let s = s.trim();
        if !s.starts_with('N') {
            return Err(self.error(format!("expected node reference, got: {}", s)));
        }
        let id: usize = s[1..]
            .parse()
            .map_err(|_| self.error(format!("invalid node id: {}", s)))?;
        Ok(ValueId(id))
    }

    fn check_node_defined(&self, id: usize) -> Result<(), MicParseError> {
        if !self.nodes.contains_key(&id) {
            return Err(self.error(format!("undefined reference N{}", id)));
        }
        Ok(())
    }

    fn parse_quoted_string(&self, s: &str) -> Result<String, MicParseError> {
        if !s.starts_with('"') || !s.ends_with('"') {
            return Err(self.error(format!("expected quoted string: {}", s)));
        }

        let inner = &s[1..s.len() - 1];
        let mut result = String::new();
        let mut chars = inner.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some(other) => {
                        result.push('\\');
                        result.push(other);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(c);
            }
        }

        Ok(result)
    }

    fn parse_i64_list(&self, s: &str) -> Result<Vec<i64>, MicParseError> {
        if s.is_empty() {
            return Ok(vec![]);
        }
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .map_err(|_| self.error(format!("invalid i64: {}", x)))
            })
            .collect()
    }

    fn parse_usize_list(&self, s: &str) -> Result<Vec<usize>, MicParseError> {
        if s.is_empty() {
            return Ok(vec![]);
        }
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .map_err(|_| self.error(format!("invalid usize: {}", x)))
            })
            .collect()
    }

    fn error(&self, message: String) -> MicParseError {
        MicParseError {
            line: self.current_line,
            message,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        let result = parse_mic("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_only() {
        let result = parse_mic("mic@1\n");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_simple_module() {
        let mic = r#"mic@1
T0 i32
N0 const.i64 42 T0
N1 const.i64 10 T0
N2 add N0 N1 T0
O N2
"#;
        let result = parse_mic(mic);
        assert!(result.is_ok());
        let module = result.unwrap();
        assert_eq!(module.instrs.len(), 4);
    }

    #[test]
    fn test_parse_with_comments() {
        let mic = r#"mic@1
# This is a comment
T0 f32
N0 const.i64 42 T0
"#;
        let result = parse_mic(mic);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_quoted_string() {
        let parser = MicParser::new("");
        assert_eq!(parser.parse_quoted_string("\"hello\"").unwrap(), "hello");
        assert_eq!(parser.parse_quoted_string("\"a\\\"b\"").unwrap(), "a\"b");
        assert_eq!(parser.parse_quoted_string("\"a\\nb\"").unwrap(), "a\nb");
    }
}
