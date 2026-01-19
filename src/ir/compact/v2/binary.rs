// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! MIC-B v2 binary format encoder and decoder.
//!
//! Wire format:
//! ```text
//! [0..4)   : magic "MICB"
//! [4]      : version 0x02
//! [5..]    : tables (ULEB128 encoded)
//!
//! Tables:
//!   1. String table: interned strings
//!   2. Symbol table: symbolic dimension names
//!   3. Type table: tensor types
//!   4. Value table: args, params, nodes
//!   5. Output: single value ID
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};

use super::types::{DType, Graph, Opcode, TensorType, Value};
use super::varint::{sleb128_read, sleb128_write, uleb128_read, uleb128_write};
use super::{MICB_MAGIC, MICB_VERSION};

/// Error type for MIC-B operations.
#[derive(Debug, Clone)]
pub struct MicbError {
    pub message: String,
}

impl std::fmt::Display for MicbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MICB error: {}", self.message)
    }
}

impl std::error::Error for MicbError {}

impl From<std::io::Error> for MicbError {
    fn from(e: std::io::Error) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

/// Serialize a Graph to MIC-B v2 binary format.
///
/// The output is deterministic: same Graph always produces same bytes.
pub fn emit_micb<W: Write>(graph: &Graph, w: &mut W) -> Result<(), MicbError> {
    let mut encoder = MicbEncoder::new(graph);
    encoder.encode(w)
}

/// Parse MIC-B v2 binary format into a Graph.
pub fn parse_micb<R: Read>(r: &mut R) -> Result<Graph, MicbError> {
    let mut decoder = MicbDecoder::new();
    decoder.decode(r)
}

/// MIC-B encoder with string table interning.
struct MicbEncoder<'a> {
    graph: &'a Graph,
    strings: Vec<String>,
    string_map: HashMap<String, usize>,
}

impl<'a> MicbEncoder<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            strings: Vec::new(),
            string_map: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.string_map.get(s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.string_map.insert(s.to_string(), idx);
        idx
    }

    fn build_string_table(&mut self) {
        // Intern strings in deterministic order:
        // 1. Symbols
        // 2. Type dimension tokens
        // 3. Value names

        for sym in &self.graph.symbols {
            self.intern(sym);
        }

        for t in &self.graph.types {
            for dim in &t.shape {
                self.intern(dim);
            }
        }

        for v in &self.graph.values {
            match v {
                Value::Arg(name, _) | Value::Param(name, _) => {
                    self.intern(name);
                }
                Value::Node(Opcode::Custom(name), _) => {
                    self.intern(name);
                }
                _ => {}
            }
        }
    }

    fn encode<W: Write>(&mut self, w: &mut W) -> Result<(), MicbError> {
        // Build string table
        self.build_string_table();

        // Magic + version
        w.write_all(&MICB_MAGIC)?;
        w.write_all(&[MICB_VERSION])?;

        // String table
        uleb128_write(w, self.strings.len() as u64)?;
        for s in &self.strings {
            let bytes = s.as_bytes();
            uleb128_write(w, bytes.len() as u64)?;
            w.write_all(bytes)?;
        }

        // Symbol table
        uleb128_write(w, self.graph.symbols.len() as u64)?;
        for sym in &self.graph.symbols {
            let idx = self.string_map[sym];
            uleb128_write(w, idx as u64)?;
        }

        // Type table
        uleb128_write(w, self.graph.types.len() as u64)?;
        for t in &self.graph.types {
            w.write_all(&[t.dtype.to_byte()])?;
            uleb128_write(w, t.shape.len() as u64)?;
            for dim in &t.shape {
                let idx = self.string_map[dim];
                uleb128_write(w, idx as u64)?;
            }
        }

        // Value table
        uleb128_write(w, self.graph.values.len() as u64)?;
        for v in &self.graph.values {
            self.encode_value(w, v)?;
        }

        // Output
        uleb128_write(w, self.graph.output as u64)?;

        Ok(())
    }

    fn encode_value<W: Write>(&self, w: &mut W, value: &Value) -> Result<(), MicbError> {
        match value {
            Value::Arg(name, type_idx) => {
                w.write_all(&[0])?; // tag
                let name_idx = self.string_map[name];
                uleb128_write(w, name_idx as u64)?;
                uleb128_write(w, *type_idx as u64)?;
            }
            Value::Param(name, type_idx) => {
                w.write_all(&[1])?; // tag
                let name_idx = self.string_map[name];
                uleb128_write(w, name_idx as u64)?;
                uleb128_write(w, *type_idx as u64)?;
            }
            Value::Node(opcode, inputs) => {
                w.write_all(&[2])?; // tag
                self.encode_opcode(w, opcode)?;
                uleb128_write(w, inputs.len() as u64)?;
                for inp in inputs {
                    uleb128_write(w, *inp as u64)?;
                }
            }
        }
        Ok(())
    }

    fn encode_opcode<W: Write>(&self, w: &mut W, opcode: &Opcode) -> Result<(), MicbError> {
        w.write_all(&[opcode.to_byte()])?;

        // Encode opcode parameters
        match opcode {
            Opcode::Softmax(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Transpose(perm) => {
                uleb128_write(w, perm.len() as u64)?;
                for p in perm {
                    sleb128_write(w, *p)?;
                }
            }
            Opcode::Sum(axes) | Opcode::Mean(axes) | Opcode::Max(axes) => {
                uleb128_write(w, axes.len() as u64)?;
                for a in axes {
                    sleb128_write(w, *a)?;
                }
            }
            Opcode::Concat(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Split(axis, n) => {
                sleb128_write(w, *axis)?;
                uleb128_write(w, *n as u64)?;
            }
            Opcode::Gather(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Custom(name) => {
                let idx = self.string_map[name];
                uleb128_write(w, idx as u64)?;
            }
            _ => {} // No additional parameters
        }

        Ok(())
    }
}

/// MIC-B decoder.
struct MicbDecoder {
    strings: Vec<String>,
}

impl MicbDecoder {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
        }
    }

    fn decode<R: Read>(&mut self, r: &mut R) -> Result<Graph, MicbError> {
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MICB_MAGIC {
            return Err(MicbError {
                message: format!(
                    "invalid magic: expected {:?}, got {:?}",
                    MICB_MAGIC, magic
                ),
            });
        }

        // Version
        let mut version = [0u8; 1];
        r.read_exact(&mut version)?;
        if version[0] != MICB_VERSION {
            return Err(MicbError {
                message: format!(
                    "unsupported version: expected {}, got {}",
                    MICB_VERSION, version[0]
                ),
            });
        }

        // String table
        let n_strings = uleb128_read(r)? as usize;
        self.strings = Vec::with_capacity(n_strings);
        for _ in 0..n_strings {
            let len = uleb128_read(r)? as usize;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            let s = String::from_utf8(buf).map_err(|_| MicbError {
                message: "invalid UTF-8 in string table".into(),
            })?;
            self.strings.push(s);
        }

        // Symbol table
        let n_symbols = uleb128_read(r)? as usize;
        let mut symbols = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            let idx = uleb128_read(r)? as usize;
            if idx >= self.strings.len() {
                return Err(MicbError {
                    message: format!("symbol string index {} out of bounds", idx),
                });
            }
            symbols.push(self.strings[idx].clone());
        }

        // Type table
        let n_types = uleb128_read(r)? as usize;
        let mut types = Vec::with_capacity(n_types);
        for _ in 0..n_types {
            let mut dtype_byte = [0u8; 1];
            r.read_exact(&mut dtype_byte)?;
            let dtype = DType::from_byte(dtype_byte[0]).ok_or_else(|| MicbError {
                message: format!("unknown dtype byte: {}", dtype_byte[0]),
            })?;

            let rank = uleb128_read(r)? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let idx = uleb128_read(r)? as usize;
                if idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("type dim string index {} out of bounds", idx),
                    });
                }
                shape.push(self.strings[idx].clone());
            }

            types.push(TensorType::new(dtype, shape));
        }

        // Value table
        let n_values = uleb128_read(r)? as usize;
        let mut values = Vec::with_capacity(n_values);
        for vid in 0..n_values {
            let value = self.decode_value(r, vid, types.len())?;
            values.push(value);
        }

        // Output
        let output = uleb128_read(r)? as usize;
        if output >= values.len() && !values.is_empty() {
            return Err(MicbError {
                message: format!("output {} out of bounds (max {})", output, values.len() - 1),
            });
        }

        Ok(Graph {
            symbols,
            types,
            values,
            output,
        })
    }

    fn decode_value<R: Read>(
        &self,
        r: &mut R,
        current_id: usize,
        n_types: usize,
    ) -> Result<Value, MicbError> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;

        match tag[0] {
            0 | 1 => {
                // Arg or Param
                let name_idx = uleb128_read(r)? as usize;
                let type_idx = uleb128_read(r)? as usize;

                if name_idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("value name string index {} out of bounds", name_idx),
                    });
                }
                if type_idx >= n_types {
                    return Err(MicbError {
                        message: format!("value type index {} out of bounds", type_idx),
                    });
                }

                let name = self.strings[name_idx].clone();
                if tag[0] == 0 {
                    Ok(Value::Arg(name, type_idx))
                } else {
                    Ok(Value::Param(name, type_idx))
                }
            }
            2 => {
                // Node
                let opcode = self.decode_opcode(r)?;
                let n_inputs = uleb128_read(r)? as usize;
                let mut inputs = Vec::with_capacity(n_inputs);
                for _ in 0..n_inputs {
                    let inp = uleb128_read(r)? as usize;
                    if inp >= current_id {
                        return Err(MicbError {
                            message: format!(
                                "forward reference: input {} >= current id {}",
                                inp, current_id
                            ),
                        });
                    }
                    inputs.push(inp);
                }
                Ok(Value::Node(opcode, inputs))
            }
            _ => Err(MicbError {
                message: format!("unknown value tag: {}", tag[0]),
            }),
        }
    }

    fn decode_opcode<R: Read>(&self, r: &mut R) -> Result<Opcode, MicbError> {
        let mut opcode_byte = [0u8; 1];
        r.read_exact(&mut opcode_byte)?;

        match opcode_byte[0] {
            0 => Ok(Opcode::Matmul),
            1 => Ok(Opcode::Add),
            2 => Ok(Opcode::Sub),
            3 => Ok(Opcode::Mul),
            4 => Ok(Opcode::Div),
            5 => Ok(Opcode::Relu),
            6 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Softmax(axis))
            }
            7 => Ok(Opcode::Sigmoid),
            8 => Ok(Opcode::Tanh),
            9 => Ok(Opcode::Gelu),
            10 => Ok(Opcode::LayerNorm),
            11 => {
                let n = uleb128_read(r)? as usize;
                let mut perm = Vec::with_capacity(n);
                for _ in 0..n {
                    perm.push(sleb128_read(r)?);
                }
                Ok(Opcode::Transpose(perm))
            }
            12 => Ok(Opcode::Reshape),
            13 => {
                let n = uleb128_read(r)? as usize;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Sum(axes))
            }
            14 => {
                let n = uleb128_read(r)? as usize;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Mean(axes))
            }
            15 => {
                let n = uleb128_read(r)? as usize;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Max(axes))
            }
            16 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Concat(axis))
            }
            17 => {
                let axis = sleb128_read(r)?;
                let n = uleb128_read(r)? as usize;
                Ok(Opcode::Split(axis, n))
            }
            18 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Gather(axis))
            }
            255 => {
                let idx = uleb128_read(r)? as usize;
                if idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("custom opcode name index {} out of bounds", idx),
                    });
                }
                Ok(Opcode::Custom(self.strings[idx].clone()))
            }
            _ => Err(MicbError {
                message: format!("unknown opcode byte: {}", opcode_byte[0]),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::v2::types::GraphEq;
    use std::io::Cursor;

    #[test]
    fn test_roundtrip_residual() {
        let graph = Graph::residual_block();

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).expect("encode failed");

        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).expect("decode failed");

        assert!(graph.eq(&parsed));
    }

    #[test]
    fn test_determinism() {
        let graph = Graph::residual_block();

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();

        emit_micb(&graph, &mut buf1).unwrap();
        emit_micb(&graph, &mut buf2).unwrap();

        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_magic_check() {
        let bad_magic = vec![0x00, 0x00, 0x00, 0x00, 0x02];
        let mut cursor = Cursor::new(&bad_magic);
        assert!(parse_micb(&mut cursor).is_err());
    }

    #[test]
    fn test_version_check() {
        let bad_version = vec![0x4D, 0x49, 0x43, 0x42, 0x99];
        let mut cursor = Cursor::new(&bad_version);
        assert!(parse_micb(&mut cursor).is_err());
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::new();

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).expect("encode failed");

        // Should be small: magic(4) + version(1) + counts
        assert!(buf.len() < 20);

        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).expect("decode failed");
        assert!(graph.eq(&parsed));
    }

    #[test]
    fn test_size_comparison() {
        use crate::ir::compact::v2::emit::emit_mic2;

        let graph = Graph::residual_block();

        // Text format
        let text = emit_mic2(&graph);
        let text_size = text.len();

        // Binary format
        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).unwrap();
        let binary_size = buf.len();

        // Binary should be smaller
        assert!(
            binary_size < text_size,
            "binary ({}) should be smaller than text ({})",
            binary_size,
            text_size
        );

        println!(
            "Residual block: text={} bytes, binary={} bytes, ratio={:.2}x",
            text_size,
            binary_size,
            text_size as f64 / binary_size as f64
        );
    }

    #[test]
    fn test_string_table_dedup() {
        use crate::ir::compact::v2::types::Value;

        let mut graph = Graph::new();

        // Same dim used multiple times
        graph.add_type(TensorType::new(DType::F32, vec!["128".into(), "128".into()]));
        graph.add_type(TensorType::new(DType::F32, vec!["128".into()]));
        graph.add_value(Value::arg("x", 0));

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).unwrap();

        // String "128" should only appear once in table
        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).unwrap();
        assert!(graph.eq(&parsed));
    }
}
