// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! mic@2 text format parser with implicit value IDs.

use super::types::{DType, Graph, Opcode, TensorType, Value};
use super::MIC2_HEADER;

/// Maximum input size in bytes (10 MB).
pub const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of lines.
pub const MAX_LINE_COUNT: usize = 1_000_000;

/// Maximum number of values.
pub const MAX_VALUE_COUNT: usize = 100_000;

/// Maximum shape dimensions.
pub const MAX_SHAPE_DIMS: usize = 32;

/// Parse error with line information.
#[derive(Debug, Clone)]
pub struct Mic2ParseError {
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for Mic2ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mic@2:{}: error: {}", self.line, self.message)
    }
}

impl std::error::Error for Mic2ParseError {}

/// Parse mic@2 text format into a Graph.
///
/// # Grammar
///
/// ```text
/// header: "mic@2"
/// symbol: S <name>
/// type:   T<idx> <dtype> <dim0> <dim1>...
/// arg:    a <name> T<typeidx>
/// param:  p <name> T<typeidx>
/// node:   <op> <in0> <in1>...     (implicit value ID)
/// output: O <value_id>
/// ```
pub fn parse_mic2(input: &str) -> Result<Graph, Mic2ParseError> {
    // Security: size limit
    if input.len() > MAX_INPUT_SIZE {
        return Err(Mic2ParseError {
            line: 0,
            message: format!(
                "input too large: {} bytes (max {})",
                input.len(),
                MAX_INPUT_SIZE
            ),
        });
    }

    let lines: Vec<&str> = input.lines().collect();

    // Security: line count limit
    if lines.len() > MAX_LINE_COUNT {
        return Err(Mic2ParseError {
            line: 0,
            message: format!(
                "too many lines: {} (max {})",
                lines.len(),
                MAX_LINE_COUNT
            ),
        });
    }

    let mut parser = Mic2Parser::new(&lines);
    parser.parse()
}

struct Mic2Parser<'a> {
    lines: &'a [&'a str],
    line_num: usize,
    graph: Graph,
    current_value_id: usize,
    has_output: bool,
}

impl<'a> Mic2Parser<'a> {
    fn new(lines: &'a [&'a str]) -> Self {
        Self {
            lines,
            line_num: 0,
            graph: Graph::new(),
            current_value_id: 0,
            has_output: false,
        }
    }

    fn error(&self, msg: impl Into<String>) -> Mic2ParseError {
        Mic2ParseError {
            line: self.line_num,
            message: msg.into(),
        }
    }

    fn parse(&mut self) -> Result<Graph, Mic2ParseError> {
        // Find and validate header
        self.parse_header()?;

        // Parse remaining lines
        while self.line_num < self.lines.len() {
            let line = self.lines[self.line_num].trim();
            self.line_num += 1;

            // Skip empty and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let first = tokens[0];

            // Dispatch by first token
            match first {
                "S" => self.parse_symbol(&tokens)?,
                "O" => self.parse_output(&tokens)?,
                "a" => self.parse_arg(&tokens)?,
                "p" => self.parse_param(&tokens)?,
                _ if first.starts_with('T') => self.parse_type(first, &tokens[1..])?,
                _ => self.parse_node(first, &tokens[1..])?,
            }
        }

        // Validate - require output only if values exist
        if !self.has_output && !self.graph.values.is_empty() {
            return Err(self.error("missing output line"));
        }

        // Skip full validation for empty graphs
        if !self.graph.values.is_empty() {
            self.graph.validate().map_err(|e| self.error(e))?;
        }

        Ok(std::mem::take(&mut self.graph))
    }

    fn parse_header(&mut self) -> Result<(), Mic2ParseError> {
        while self.line_num < self.lines.len() {
            let line = self.lines[self.line_num].trim();
            self.line_num += 1;

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line == MIC2_HEADER {
                return Ok(());
            }

            if line.starts_with("mic@") {
                return Err(self.error(format!(
                    "unsupported version '{}', expected '{}'",
                    line, MIC2_HEADER
                )));
            }

            return Err(self.error(format!(
                "expected '{}' header, got '{}'",
                MIC2_HEADER, line
            )));
        }

        Err(self.error("empty input or missing header"))
    }

    fn parse_symbol(&mut self, tokens: &[&str]) -> Result<(), Mic2ParseError> {
        // S <name>
        if tokens.len() != 2 {
            return Err(self.error("invalid symbol: expected 'S <name>'"));
        }
        self.graph.add_symbol(tokens[1]);
        Ok(())
    }

    fn parse_type(&mut self, first: &str, rest: &[&str]) -> Result<(), Mic2ParseError> {
        // T<idx> <dtype> <dim0> <dim1>...
        let idx_str = &first[1..];
        let idx: usize = idx_str
            .parse()
            .map_err(|_| self.error(format!("invalid type index: {}", idx_str)))?;

        if idx != self.graph.types.len() {
            return Err(self.error(format!(
                "type indices must be sequential: expected T{}, got T{}",
                self.graph.types.len(),
                idx
            )));
        }

        if rest.is_empty() {
            return Err(self.error("type requires dtype"));
        }

        let dtype = DType::parse(rest[0])
            .ok_or_else(|| self.error(format!("unknown dtype: {}", rest[0])))?;

        // Security: shape dim limit
        if rest.len() - 1 > MAX_SHAPE_DIMS {
            return Err(self.error(format!(
                "too many shape dimensions: {} (max {})",
                rest.len() - 1,
                MAX_SHAPE_DIMS
            )));
        }

        let shape: Vec<String> = rest[1..].iter().map(|s| s.to_string()).collect();
        self.graph.add_type(TensorType::new(dtype, shape));
        Ok(())
    }

    fn parse_arg(&mut self, tokens: &[&str]) -> Result<(), Mic2ParseError> {
        // a <name> T<typeidx>
        if tokens.len() != 3 {
            return Err(self.error("invalid arg: expected 'a <name> T<typeidx>'"));
        }

        let name = tokens[1];
        let type_ref = self.parse_type_ref(tokens[2])?;

        self.add_value(Value::arg(name, type_ref))
    }

    fn parse_param(&mut self, tokens: &[&str]) -> Result<(), Mic2ParseError> {
        // p <name> T<typeidx>
        if tokens.len() != 3 {
            return Err(self.error("invalid param: expected 'p <name> T<typeidx>'"));
        }

        let name = tokens[1];
        let type_ref = self.parse_type_ref(tokens[2])?;

        self.add_value(Value::param(name, type_ref))
    }

    fn parse_node(&mut self, opcode_tok: &str, rest: &[&str]) -> Result<(), Mic2ParseError> {
        // <op> <in0> <in1>... [params]
        // Parse inputs (numbers) and params (non-numbers) separately
        let mut inputs: Vec<usize> = Vec::new();
        let mut params: Vec<&str> = Vec::new();

        for tok in rest {
            if let Ok(id) = tok.parse::<usize>() {
                inputs.push(id);
            } else {
                params.push(tok);
            }
        }

        // Validate inputs
        for &inp in &inputs {
            if inp >= self.current_value_id {
                return Err(self.error(format!(
                    "forward reference: input {} not yet defined (current max: {})",
                    inp,
                    self.current_value_id.saturating_sub(1)
                )));
            }
        }

        // Parse opcode
        let opcode = Opcode::parse(opcode_tok, &params)
            .ok_or_else(|| self.error(format!("unknown opcode: {}", opcode_tok)))?;

        // Validate arity
        if let Some(expected) = opcode.arity() {
            if inputs.len() != expected {
                return Err(self.error(format!(
                    "opcode '{}' requires {} inputs, got {}",
                    opcode_tok, expected, inputs.len()
                )));
            }
        }

        self.add_value(Value::node(opcode, inputs))
    }

    fn parse_output(&mut self, tokens: &[&str]) -> Result<(), Mic2ParseError> {
        // O <value_id>
        if tokens.len() != 2 {
            return Err(self.error("invalid output: expected 'O <value_id>'"));
        }

        let id: usize = tokens[1]
            .parse()
            .map_err(|_| self.error(format!("invalid output id: {}", tokens[1])))?;

        if id >= self.current_value_id {
            return Err(self.error(format!(
                "output references undefined value {} (max: {})",
                id,
                self.current_value_id.saturating_sub(1)
            )));
        }

        self.graph.set_output(id);
        self.has_output = true;
        Ok(())
    }

    fn parse_type_ref(&self, tok: &str) -> Result<usize, Mic2ParseError> {
        if !tok.starts_with('T') {
            return Err(self.error(format!("expected type ref T<idx>, got '{}'", tok)));
        }

        let idx: usize = tok[1..]
            .parse()
            .map_err(|_| self.error(format!("invalid type ref: {}", tok)))?;

        if idx >= self.graph.types.len() {
            return Err(self.error(format!(
                "type T{} not defined (max: T{})",
                idx,
                self.graph.types.len().saturating_sub(1)
            )));
        }

        Ok(idx)
    }

    fn add_value(&mut self, value: Value) -> Result<(), Mic2ParseError> {
        // Security: value count limit
        if self.current_value_id >= MAX_VALUE_COUNT {
            return Err(self.error(format!(
                "too many values: {} (max {})",
                self.current_value_id,
                MAX_VALUE_COUNT
            )));
        }

        self.graph.add_value(value);
        self.current_value_id += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESIDUAL_BLOCK: &str = r#"mic@2
T0 f16 128 128
T1 f16 128
a X T0
p W T0
p b T1
m 0 1
+ 3 2
r 4
+ 5 0
O 6
"#;

    #[test]
    fn test_parse_residual_block() {
        let graph = parse_mic2(RESIDUAL_BLOCK).expect("parse failed");
        assert_eq!(graph.types.len(), 2);
        assert_eq!(graph.values.len(), 7);
        assert_eq!(graph.output, 6);
    }

    #[test]
    fn test_parse_empty() {
        let result = parse_mic2("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_only() {
        let result = parse_mic2("mic@2\n");
        assert!(result.is_ok());
        let g = result.unwrap();
        assert!(g.values.is_empty());
    }

    #[test]
    fn test_parse_with_comments() {
        let input = r#"mic@2
# This is a comment
T0 f32 10
# Another comment
a x T0
O 0
"#;
        let result = parse_mic2(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_forward_ref_error() {
        let input = "mic@2\nT0 f32 10\nm 0 1\n";
        let result = parse_mic2(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("forward reference"));
    }

    #[test]
    fn test_parse_invalid_type_ref() {
        let input = "mic@2\nT0 f32 10\na x T99\n";
        let result = parse_mic2(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("not defined"));
    }

    #[test]
    fn test_parse_wrong_arity() {
        let input = "mic@2\nT0 f32 10\na x T0\na y T0\nm 0\nO 2\n";
        let result = parse_mic2(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("requires 2 inputs"));
    }

    #[test]
    fn test_parse_symbols() {
        let input = "mic@2\nS B\nS seq\nT0 f32 B seq\na x T0\nO 0\n";
        let result = parse_mic2(input);
        assert!(result.is_ok());
        let g = result.unwrap();
        assert_eq!(g.symbols, vec!["B", "seq"]);
    }

    #[test]
    fn test_parse_sequential_types() {
        // Types must be T0, T1, T2...
        let input = "mic@2\nT1 f32 10\n";
        let result = parse_mic2(input);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("sequential"));
    }
}
