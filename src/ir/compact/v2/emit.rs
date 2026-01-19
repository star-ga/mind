// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! mic@2 canonical text serializer.
//!
//! Canonicalization rules:
//! - Lines use exactly one space between tokens
//! - No trailing spaces
//! - No trailing newline after final output line
//! - Sections in order: header, symbols, types, values, output

use super::types::{Graph, Opcode, Value};
use super::MIC2_HEADER;

/// Emit a Graph as canonical mic@2 text.
pub fn emit_mic2(graph: &Graph) -> String {
    let mut emitter = Mic2Emitter::new();
    emitter.emit(graph)
}

/// mic@2 emitter with canonical output.
pub struct Mic2Emitter {
    output: String,
}

impl Mic2Emitter {
    /// Create a new emitter.
    pub fn new() -> Self {
        Self {
            output: String::with_capacity(4096),
        }
    }

    /// Emit a Graph as canonical mic@2 text.
    pub fn emit(&mut self, graph: &Graph) -> String {
        self.output.clear();

        // Header
        self.output.push_str(MIC2_HEADER);
        self.output.push('\n');

        // Symbols (optional)
        for sym in &graph.symbols {
            self.output.push_str("S ");
            self.output.push_str(sym);
            self.output.push('\n');
        }

        // Types
        for (i, t) in graph.types.iter().enumerate() {
            self.output.push('T');
            self.output.push_str(&i.to_string());
            self.output.push(' ');
            self.output.push_str(t.dtype.as_str());
            for dim in &t.shape {
                self.output.push(' ');
                self.output.push_str(dim);
            }
            self.output.push('\n');
        }

        // Values (implicit IDs by order)
        for value in &graph.values {
            self.emit_value(value);
            self.output.push('\n');
        }

        // Output (no trailing newline - canonical)
        self.output.push('O');
        self.output.push(' ');
        self.output.push_str(&graph.output.to_string());

        self.output.clone()
    }

    fn emit_value(&mut self, value: &Value) {
        match value {
            Value::Arg(name, type_idx) => {
                self.output.push_str("a ");
                self.output.push_str(name);
                self.output.push_str(" T");
                self.output.push_str(&type_idx.to_string());
            }
            Value::Param(name, type_idx) => {
                self.output.push_str("p ");
                self.output.push_str(name);
                self.output.push_str(" T");
                self.output.push_str(&type_idx.to_string());
            }
            Value::Node(opcode, inputs) => {
                self.emit_opcode(opcode);
                for inp in inputs {
                    self.output.push(' ');
                    self.output.push_str(&inp.to_string());
                }
            }
        }
    }

    fn emit_opcode(&mut self, opcode: &Opcode) {
        self.output.push_str(opcode.as_token());

        // Emit opcode parameters
        match opcode {
            Opcode::Softmax(axis) if *axis != -1 => {
                self.output.push(' ');
                self.output.push_str(&axis.to_string());
            }
            Opcode::Transpose(perm) if !perm.is_empty() => {
                for p in perm {
                    self.output.push(' ');
                    self.output.push_str(&p.to_string());
                }
            }
            Opcode::Sum(axes) | Opcode::Mean(axes) | Opcode::Max(axes) if !axes.is_empty() => {
                for a in axes {
                    self.output.push(' ');
                    self.output.push_str(&a.to_string());
                }
            }
            Opcode::Concat(axis) => {
                self.output.push(' ');
                self.output.push_str(&axis.to_string());
            }
            Opcode::Split(axis, n) => {
                self.output.push(' ');
                self.output.push_str(&axis.to_string());
                self.output.push(' ');
                self.output.push_str(&n.to_string());
            }
            Opcode::Gather(axis) => {
                self.output.push(' ');
                self.output.push_str(&axis.to_string());
            }
            _ => {}
        }
    }
}

impl Default for Mic2Emitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::v2::parse::parse_mic2;
    use crate::ir::compact::v2::types::GraphEq;

    #[test]
    fn test_emit_residual_block() {
        let graph = Graph::residual_block();
        let text = emit_mic2(&graph);

        // Check canonical format
        assert!(text.starts_with("mic@2\n"));
        assert!(text.ends_with("O 6"));
        assert!(!text.ends_with("\n")); // No trailing newline

        // Check roundtrip
        let parsed = parse_mic2(&text).expect("parse failed");
        assert!(graph.eq(&parsed));
    }

    #[test]
    fn test_emit_determinism() {
        let graph = Graph::residual_block();

        let text1 = emit_mic2(&graph);
        let text2 = emit_mic2(&graph);
        let text3 = emit_mic2(&graph);

        assert_eq!(text1, text2);
        assert_eq!(text2, text3);
    }

    #[test]
    fn test_emit_single_space() {
        let graph = Graph::residual_block();
        let text = emit_mic2(&graph);

        // No double spaces
        assert!(!text.contains("  "));

        // No trailing spaces on lines
        for line in text.lines() {
            assert!(!line.ends_with(' '));
        }
    }

    #[test]
    fn test_emit_empty_graph() {
        let graph = Graph::new();
        let text = emit_mic2(&graph);
        assert_eq!(text, "mic@2\nO 0");
    }

    #[test]
    fn test_emit_with_symbols() {
        let mut graph = Graph::new();
        graph.add_symbol("B");
        graph.add_symbol("seq");
        let text = emit_mic2(&graph);
        assert!(text.contains("S B\n"));
        assert!(text.contains("S seq\n"));
    }

    #[test]
    fn test_canonical_residual_bytes() {
        let graph = Graph::residual_block();
        let text = emit_mic2(&graph);

        // Expected canonical output
        let expected = "mic@2
T0 f16 128 128
T1 f16 128
a X T0
p W T0
p b T1
m 0 1
+ 3 2
r 4
+ 5 0
O 6";

        assert_eq!(text, expected);
    }
}
