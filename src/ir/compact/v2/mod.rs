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

//! MIC v2 format - Next-generation compact IR serialization.
//!
//! MIC v2 includes two complementary formats:
//! - **mic@2**: Text format with implicit value IDs for minimal token usage
//! - **MIC-B v2**: Binary format with ULEB128 varints for maximum compactness
//!
//! # Key Improvements over v1
//!
//! - **Implicit IDs**: Values are numbered sequentially (0, 1, 2...) by order of appearance
//! - **Compact Opcodes**: Single-character ops (`m`, `+`, `r`, `s`) vs verbose names
//! - **Space-separated Dims**: `T0 f16 128 128` vs `[f16;128,128]`
//! - **Binary Format**: ~10x smaller than JSON, ~4x smaller than text
//!
//! # mic@2 Example (Residual Block)
//!
//! ```text
//! mic@2
//! T0 f16 128 128
//! T1 f16 128
//! a X T0
//! p W T0
//! p b T1
//! m 0 1
//! + 3 2
//! r 4
//! + 5 0
//! O 6
//! ```

mod binary;
mod emit;
mod parse;
mod types;
mod varint;

pub use binary::{emit_micb, parse_micb, MicbError};
pub use emit::{emit_mic2, Mic2Emitter};
pub use parse::{parse_mic2, Mic2ParseError};
pub use types::{DType, Graph, GraphEq, Opcode, TensorType, Value};
pub use varint::{uleb128_read, uleb128_write, zigzag_decode, zigzag_encode};

/// MIC v2 text format version header.
pub const MIC2_HEADER: &str = "mic@2";

/// MIC-B v2 binary magic bytes.
pub const MICB_MAGIC: [u8; 4] = [0x4D, 0x49, 0x43, 0x42]; // "MICB"

/// MIC-B v2 version byte.
pub const MICB_VERSION: u8 = 0x02;

/// Detects the format of input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicFormat {
    /// mic@1 text format (legacy)
    Mic1,
    /// mic@2 text format
    Mic2,
    /// MIC-B v2 binary format
    MicB,
    /// Unknown format
    Unknown,
}

/// Detect the format of input bytes.
pub fn detect_format(data: &[u8]) -> MicFormat {
    // Check for binary magic first
    if data.len() >= 5 && data[0..4] == MICB_MAGIC {
        return MicFormat::MicB;
    }

    // Check text formats
    if let Ok(text) = std::str::from_utf8(data) {
        let trimmed = text.trim_start();
        if trimmed.starts_with("mic@2") {
            return MicFormat::Mic2;
        }
        if trimmed.starts_with("mic@1") {
            return MicFormat::Mic1;
        }
    }

    MicFormat::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_mic2() {
        assert_eq!(detect_format(b"mic@2\nT0 f16 128"), MicFormat::Mic2);
        assert_eq!(detect_format(b"  mic@2\n"), MicFormat::Mic2);
    }

    #[test]
    fn test_detect_mic1() {
        assert_eq!(detect_format(b"mic@1\nN0 const.i64 42"), MicFormat::Mic1);
    }

    #[test]
    fn test_detect_micb() {
        assert_eq!(
            detect_format(&[0x4D, 0x49, 0x43, 0x42, 0x02]),
            MicFormat::MicB
        );
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(detect_format(b"garbage"), MicFormat::Unknown);
        assert_eq!(detect_format(b""), MicFormat::Unknown);
    }

    #[test]
    fn test_roundtrip_residual() {
        use std::io::Cursor;

        let graph = Graph::residual_block();

        // Text roundtrip
        let text = emit_mic2(&graph);
        let parsed = parse_mic2(&text).expect("parse failed");
        assert!(graph.eq(&parsed), "Text roundtrip failed");

        // Binary roundtrip
        let mut binary = Vec::new();
        emit_micb(&graph, &mut binary).expect("emit failed");
        let mut cursor = Cursor::new(&binary);
        let parsed_bin = parse_micb(&mut cursor).expect("parse failed");
        assert!(graph.eq(&parsed_bin), "Binary roundtrip failed");

        // Cross-format roundtrip
        assert!(parsed.eq(&parsed_bin), "Cross-format equality failed");
    }

    #[test]
    fn test_determinism() {
        let graph = Graph::residual_block();

        // Text determinism
        let text1 = emit_mic2(&graph);
        let text2 = emit_mic2(&graph);
        let text3 = emit_mic2(&graph);
        assert_eq!(text1, text2);
        assert_eq!(text2, text3);

        // Binary determinism
        let mut bin1 = Vec::new();
        let mut bin2 = Vec::new();
        emit_micb(&graph, &mut bin1).unwrap();
        emit_micb(&graph, &mut bin2).unwrap();
        assert_eq!(bin1, bin2);
    }
}
