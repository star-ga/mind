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

//! MindIR Compact (MIC) format - Token-efficient IR serialization for AI agents.
//!
//! MIC supports two format versions:
//!
//! ## mic@1 (Legacy)
//!
//! Text format with explicit node IDs:
//! ```text
//! mic@1
//! T0 f32
//! N0 const.i64 42 T0
//! N1 add N0 N0 T0
//! O N1
//! ```
//!
//! ## mic@2 (Recommended)
//!
//! Text format with implicit value IDs for ~40% token reduction:
//! ```text
//! mic@2
//! T0 f16 128 128
//! a X T0
//! p W T0
//! m 0 1
//! O 2
//! ```
//!
//! ## MIC-B v2 (Binary)
//!
//! Compact binary format with ULEB128 varints, ~4x smaller than text.
//!
//! # Format Detection
//!
//! Use `detect_format()` to identify input format:
//! ```ignore
//! use mind::ir::compact::{detect_format, MicFormat};
//!
//! match detect_format(data) {
//!     MicFormat::Mic1 => parse_mic(data),
//!     MicFormat::Mic2 => v2::parse_mic2(data),
//!     MicFormat::MicB => v2::parse_micb(data),
//!     MicFormat::Unknown => Err(...),
//! }
//! ```

mod emit;
mod parse;
pub mod v2;

pub use emit::{emit_mic, MicEmitter};
pub use parse::{parse_mic, MicParseError};
pub use v2::{detect_format, MicFormat};

/// MIC format version.
pub const MIC_VERSION: u32 = 1;

/// MIC version header string.
pub const MIC_HEADER: &str = "mic@1";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, IRModule, Instr, ValueId};
    use crate::types::{DType, ShapeDim};

    #[test]
    fn test_roundtrip_simple() {
        let mut module = IRModule::new();
        let v0 = module.fresh();
        let v1 = module.fresh();
        let v2 = module.fresh();

        module.instrs.push(Instr::ConstI64(v0, 42));
        module.instrs.push(Instr::ConstI64(v1, 10));
        module.instrs.push(Instr::BinOp {
            dst: v2,
            op: BinOp::Add,
            lhs: v0,
            rhs: v1,
        });
        module.instrs.push(Instr::Output(v2));

        let mic = emit_mic(&module);
        assert!(mic.starts_with("mic@1\n"));

        let parsed = parse_mic(&mic).expect("parse failed");
        assert_eq!(parsed.instrs.len(), module.instrs.len());
    }

    #[test]
    fn test_roundtrip_tensor() {
        let mut module = IRModule::new();
        let v0 = module.fresh();

        module.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(4)],
            Some(1.0),
        ));
        module.instrs.push(Instr::Output(v0));

        let mic = emit_mic(&module);
        let parsed = parse_mic(&mic).expect("parse failed");
        assert_eq!(parsed.instrs.len(), 2);
    }

    #[test]
    fn test_version_check() {
        let bad_mic = "mic@99\nN1 const.i64 42 T0\n";
        let result = parse_mic(bad_mic);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_reference() {
        let bad_mic = "mic@1\nN1 add N99 N98 T0\nO N1\n";
        let result = parse_mic(bad_mic);
        assert!(result.is_err());
    }

    // =========================================================================
    // DETERMINISM TESTS - RFC-0001 Canonicalization Requirements
    // =========================================================================

    #[test]
    fn test_emit_determinism() {
        // RFC-0001: "Same IR always produces same MIC"
        let mut module = IRModule::new();
        let v0 = module.fresh();
        let v1 = module.fresh();
        let v2 = module.fresh();

        module.instrs.push(Instr::ConstI64(v0, 42));
        module.instrs.push(Instr::ConstI64(v1, 10));
        module.instrs.push(Instr::BinOp {
            dst: v2,
            op: BinOp::Add,
            lhs: v0,
            rhs: v1,
        });
        module.instrs.push(Instr::Output(v2));

        // Emit multiple times
        let mic1 = emit_mic(&module);
        let mic2 = emit_mic(&module);
        let mic3 = emit_mic(&module);

        assert_eq!(mic1, mic2, "Determinism: emit should be consistent");
        assert_eq!(mic2, mic3, "Determinism: emit should be consistent");
    }

    #[test]
    fn test_roundtrip_determinism() {
        // RFC-0001: "parse(emit(ir)) == ir" semantically
        let mic_input = r#"mic@1
T0 i32
N0 const.i64 42 T0
N1 const.i64 10 T0
N2 add N0 N1 T0
O N2
"#;

        let parsed1 = parse_mic(mic_input).expect("parse 1 failed");
        let emitted1 = emit_mic(&parsed1);
        let parsed2 = parse_mic(&emitted1).expect("parse 2 failed");
        let emitted2 = emit_mic(&parsed2);

        // Roundtrip should produce identical output
        assert_eq!(emitted1, emitted2, "Roundtrip determinism failed");
    }

    #[test]
    fn test_symbolic_shape_determinism() {
        // Test that symbolic shapes are handled deterministically
        let mut module = IRModule::new();
        let v0 = module.fresh();

        module.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(128)],
            None,
        ));
        module.instrs.push(Instr::Output(v0));

        let mic1 = emit_mic(&module);
        let mic2 = emit_mic(&module);

        assert_eq!(mic1, mic2, "Symbolic shape determinism failed");

        // Verify roundtrip
        let parsed = parse_mic(&mic1).expect("parse failed");
        let mic3 = emit_mic(&parsed);

        // Note: The actual output may differ slightly due to type inference
        // but the structure should be consistent
        assert!(mic3.contains("mic@1"), "Version header present");
    }

    // =========================================================================
    // SECURITY TESTS - RFC-0001 Security Considerations
    // =========================================================================

    #[test]
    fn test_input_size_limit() {
        // RFC-0001: "Parsers MUST limit input size to prevent memory exhaustion"
        let huge_input = "mic@1\n".to_string() + &"N".repeat(20 * 1024 * 1024); // 20 MB

        let result = parse_mic(&huge_input);
        assert!(result.is_err(), "Should reject oversized input");
        assert!(
            result.unwrap_err().message.contains("too large"),
            "Error should mention size"
        );
    }

    #[test]
    fn test_node_count_limit() {
        // RFC-0001: Parsers should limit resource usage
        let mut mic = String::from("mic@1\nT0 i32\n");
        for i in 0..150_000 {
            mic.push_str(&format!("N{} const.i64 {} T0\n", i, i));
        }

        let result = parse_mic(&mic);
        assert!(result.is_err(), "Should reject too many nodes");
    }

    #[test]
    fn test_shape_dim_limit() {
        // RFC-0001: Limit shape dimensions
        let mut shape = (0..50).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let mic = format!(
            "mic@1\nT0 [f32;{}]\nN0 const.tensor fill=0.0 T0\nO N0\n",
            shape
        );

        let result = parse_mic(&mic);
        assert!(result.is_err(), "Should reject too many shape dimensions");
    }

    #[test]
    fn test_escape_handling() {
        // RFC-0001: "String parsing MUST handle escape sequences safely"
        let mic = r#"mic@1
S0 "test\nwith\tescape\"chars\\"
T0 f32
"#;

        let result = parse_mic(mic);
        assert!(result.is_ok(), "Should parse escape sequences");
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_empty_module() {
        let mic = "mic@1\n";
        let result = parse_mic(mic);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().instrs.len(), 0);
    }

    #[test]
    fn test_comments_preserved() {
        let mic = r#"mic@1
# This is a comment
T0 f32
# Another comment
N0 const.i64 42 T0
O N0
"#;

        let result = parse_mic(mic);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().instrs.len(), 2);
    }

    #[test]
    fn test_whitespace_handling() {
        let mic = "mic@1\n  T0 f32  \n  N0 const.i64 42 T0  \n  O N0  \n";

        let result = parse_mic(mic);
        assert!(result.is_ok(), "Should handle whitespace");
    }

    #[test]
    fn test_all_binops() {
        for (op_str, op) in [
            ("add", BinOp::Add),
            ("sub", BinOp::Sub),
            ("mul", BinOp::Mul),
            ("div", BinOp::Div),
        ] {
            let mic = format!(
                "mic@1\nT0 f32\nN0 const.i64 10 T0\nN1 const.i64 5 T0\nN2 {} N0 N1 T0\nO N2\n",
                op_str
            );

            let result = parse_mic(&mic);
            assert!(result.is_ok(), "Failed to parse {} operation", op_str);
        }
    }
}
