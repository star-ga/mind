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

//! MIC@3 binary format — compact, deterministic binary encoding of [`IRModule`](crate::ir::IRModule).
//!
//! # Wire layout (IR body)
//!
//! ```text
//! [0..4)  magic  "MIC3"
//! [4]     version 0x01
//! varint  string-table count N
//! N × (varint byte-len, utf-8 bytes)  — string table entries (first-seen order)
//! varint  next_id
//! varint  exports count  (sorted lexicographically)
//! N × varint  exports[i] — index into string table
//! varint  instr count
//! N × encoded-instr
//! #[cfg(std-surface)]
//! varint  struct_defs count
//! N × (varint name-idx, varint field-count, M × varint field-name-idx)
//! varint  const_array_defs count
//! N × (varint name-idx, varint elem-count, M × zigzag-i64)
//! varint  repr_c_structs count
//! N × (varint name-idx, varint field-count, M × type-ann bytes)
//! ```
//!
//! # Optional MAP epilogue (RFC 0021 §4.2)
//!
//! When an artifact is emitted with evidence, a MAP epilogue immediately follows
//! the last IR body byte.  Its format is documented in [`evidence`].  A reader
//! that does not understand the epilogue can still parse the IR body by passing
//! `bytes[..body_end]` to [`parse_mic3`].
//!
//! ```text
//! 0x4D                      -- MAP sentinel ('M')
//! ULEB128 entry_count
//! For each entry (lexicographic key order):
//!   ULEB128 key_len + key_bytes
//!   value_tag (0=String, 1=Int, 2=Bytes) + encoded value
//! ```
//!
//! # Opcode table
//!
//! | byte | variant |
//! |------|---------|
//! | 0x01 | ConstI64 |
//! | 0x02 | ConstF64 |
//! | 0x03 | ConstTensor |
//! | 0x04 | BinOp |
//! | 0x05 | Sum |
//! | 0x06 | Mean |
//! | 0x07 | Reshape |
//! | 0x08 | ExpandDims |
//! | 0x09 | Squeeze |
//! | 0x0A | Transpose |
//! | 0x0B | Dot |
//! | 0x0C | MatMul |
//! | 0x0D | Conv2d |
//! | 0x0E | Conv2dGradInput |
//! | 0x0F | Conv2dGradFilter |
//! | 0x10 | Index |
//! | 0x11 | Slice |
//! | 0x12 | Gather |
//! | 0x13 | Output |
//! | 0x14 | SparseAttr |
//! | 0x15 | FnDef |
//! | 0x16 | Call |
//! | 0x17 | Return |
//! | 0x18 | Param |
//! | 0x19 | ConstArray (std-surface) |
//! | 0x1A | ArrayLoad (std-surface) |
//! | 0x1B | While (std-surface) |
//! | 0x1C | If (std-surface) |
//! | 0x1D | VecLoad (std-surface) |
//! | 0x1E | VecFma (std-surface) |
//! | 0x1F | VecReduceAdd (std-surface) |
//! | 0x20 | VecStore (std-surface) |
//! | 0x21 | VecLoadI32 (std-surface) |
//! | 0x22 | VecMulAddQ16 (std-surface) |
//! | 0x23 | VecReduceAddI64 (std-surface) |
//! | 0x24 | Region (std-surface) |
//! | 0x25 | ExternFnDecl (std-surface) |

pub mod evidence;
mod emit;
mod parse;

pub use emit::emit_mic3;
pub use parse::{parse_mic3, Mic3Error};
pub use evidence::{emit_mic3_with_evidence, mic3_evidence_report};
// Re-export the evidence vocabulary at the v3 level for convenience.
pub use crate::ir::compact::v2::{Determinism, EvidenceError, EvidenceReport};

/// Magic header bytes for MIC@3 binary format.
pub const MIC3_MAGIC: [u8; 4] = [b'M', b'I', b'C', b'3'];

/// MIC@3 format version byte.
pub const MIC3_VERSION: u8 = 0x01;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, IRModule, IndexSpec, Instr, SliceSpec};
    use crate::types::{ConvPadding, DType, ShapeDim};

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Re-emit via mic@1 to compare modules structurally without PartialEq on IRModule.
    fn mic1_canonical(m: &IRModule) -> String {
        crate::ir::compact::emit_mic(m)
    }

    fn roundtrip(m: &IRModule) -> IRModule {
        let bytes = emit_mic3(m);
        parse_mic3(&bytes).expect("parse_mic3 failed")
    }

    // -------------------------------------------------------------------------
    // 1. Per-category round-trip tests
    // -------------------------------------------------------------------------

    #[test]
    fn roundtrip_consts() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 42));
        m.instrs.push(Instr::ConstI64(v1, -1));
        m.instrs.push(Instr::ConstF64(v2, std::f64::consts::PI));
        m.instrs.push(Instr::Output(v2));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_const_tensor() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(4)],
            Some(1.0),
        ));
        m.instrs.push(Instr::ConstTensor(
            v1,
            DType::F16,
            vec![ShapeDim::Sym("B"), ShapeDim::Known(128)],
            None,
        ));
        m.instrs.push(Instr::Output(v0));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_binop() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 10));
        m.instrs.push(Instr::ConstI64(v1, 5));
        m.instrs.push(Instr::BinOp {
            dst: v2,
            op: BinOp::Add,
            lhs: v0,
            rhs: v1,
        });
        m.instrs.push(Instr::Output(v2));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_all_binops() {
        let base_ops = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Mod,
            BinOp::Lt,
            BinOp::Le,
            BinOp::Gt,
            BinOp::Ge,
            BinOp::Eq,
            BinOp::Ne,
        ];
        for op in base_ops {
            let mut m = IRModule::new();
            let v0 = m.fresh();
            let v1 = m.fresh();
            let v2 = m.fresh();
            m.instrs.push(Instr::ConstI64(v0, 1));
            m.instrs.push(Instr::ConstI64(v1, 2));
            m.instrs.push(Instr::BinOp { dst: v2, op, lhs: v0, rhs: v1 });
            m.instrs.push(Instr::Output(v2));
            let parsed = roundtrip(&m);
            // Verify instrs count preserved
            assert_eq!(parsed.instrs.len(), m.instrs.len());
        }
    }

    #[test]
    fn roundtrip_reductions() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(8)],
            Some(0.5),
        ));
        m.instrs.push(Instr::Sum {
            dst: v1,
            src: v0,
            axes: vec![0, 1],
            keepdims: true,
        });
        m.instrs.push(Instr::Mean {
            dst: v2,
            src: v0,
            axes: vec![1],
            keepdims: false,
        });
        m.instrs.push(Instr::Output(v1));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_tensor_ops() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        let v3 = m.fresh();
        let v4 = m.fresh();
        let v5 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(4)],
            None,
        ));
        m.instrs.push(Instr::Reshape {
            dst: v1,
            src: v0,
            new_shape: vec![ShapeDim::Known(12)],
        });
        m.instrs.push(Instr::ExpandDims { dst: v2, src: v1, axis: 0 });
        m.instrs.push(Instr::Squeeze { dst: v3, src: v2, axes: vec![0] });
        m.instrs.push(Instr::Transpose {
            dst: v4,
            src: v0,
            perm: vec![1, 0],
        });
        m.instrs.push(Instr::Gather {
            dst: v5,
            src: v0,
            indices: v1,
            axis: 1,
        });
        m.instrs.push(Instr::Output(v3));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_dot_matmul() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        let v3 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(4)],
            None,
        ));
        m.instrs.push(Instr::ConstTensor(
            v1,
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(4)],
            None,
        ));
        m.instrs.push(Instr::Dot { dst: v2, a: v0, b: v1 });
        m.instrs.push(Instr::MatMul { dst: v3, a: v0, b: v1 });
        m.instrs.push(Instr::Output(v3));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_conv2d() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        let v3 = m.fresh();
        let v4 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(1), ShapeDim::Known(8), ShapeDim::Known(8), ShapeDim::Known(3)],
            None,
        ));
        m.instrs.push(Instr::ConstTensor(
            v1,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(3), ShapeDim::Known(3), ShapeDim::Known(16)],
            None,
        ));
        m.instrs.push(Instr::Conv2d {
            dst: v2,
            input: v0,
            filter: v1,
            stride_h: 1,
            stride_w: 1,
            padding: ConvPadding::Same,
        });
        m.instrs.push(Instr::Conv2dGradInput {
            dst: v3,
            dy: v2,
            filter: v1,
            input_shape: [1, 8, 8, 3],
            stride_h: 2,
            stride_w: 2,
            padding: ConvPadding::Valid,
        });
        m.instrs.push(Instr::Conv2dGradFilter {
            dst: v4,
            input: v0,
            dy: v2,
            filter_shape: [3, 3, 3, 16],
            stride_h: 1,
            stride_w: 1,
            padding: ConvPadding::Same,
        });
        m.instrs.push(Instr::Output(v2));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_index_slice() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(8)],
            None,
        ));
        m.instrs.push(Instr::Index {
            dst: v1,
            src: v0,
            indices: vec![IndexSpec { axis: 0, index: 2 }],
        });
        m.instrs.push(Instr::Slice {
            dst: v2,
            src: v0,
            dims: vec![SliceSpec { axis: 0, start: 1, end: Some(3), stride: 1 }],
        });
        m.instrs.push(Instr::Output(v1));

        let parsed = roundtrip(&m);
        assert_eq!(mic1_canonical(&parsed), mic1_canonical(&m));
    }

    #[test]
    fn roundtrip_output_and_sparse_attr() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 99));
        m.instrs.push(Instr::SparseAttr {
            src: v0,
            dst: v1,
            layout: crate::ast::SparseLayout::Csr,
        });
        m.instrs.push(Instr::Output(v1));

        let parsed = roundtrip(&m);
        assert_eq!(parsed.instrs.len(), m.instrs.len());
    }

    #[test]
    fn roundtrip_fndef_call_return_param() {
        let mut m = IRModule::new();
        let p0 = m.fresh();
        let v0 = m.fresh();
        let ret = m.fresh();
        let call_dst = m.fresh();

        m.instrs.push(Instr::FnDef {
            name: "my_fn".into(),
            params: vec![("x".into(), p0)],
            ret_id: Some(ret),
            body: vec![
                Instr::Param { dst: p0, name: "x".into(), index: 0 },
                Instr::ConstI64(v0, 7),
                Instr::BinOp { dst: ret, op: BinOp::Add, lhs: p0, rhs: v0 },
                Instr::Return { value: Some(ret) },
            ],
            reap_threshold: Some(0.5),
        });
        m.instrs.push(Instr::ConstI64(v0, 3));
        m.instrs.push(Instr::Call {
            dst: call_dst,
            name: "my_fn".into(),
            args: vec![v0],
        });
        m.instrs.push(Instr::Output(call_dst));

        let parsed = roundtrip(&m);
        assert_eq!(parsed.instrs.len(), m.instrs.len());
        // Verify FnDef body is preserved
        if let Instr::FnDef { body, name, reap_threshold, .. } = &parsed.instrs[0] {
            assert_eq!(name, "my_fn");
            assert_eq!(body.len(), 4);
            assert_eq!(*reap_threshold, Some(0.5));
        } else {
            panic!("expected FnDef");
        }
    }

    #[test]
    fn roundtrip_nested_fndef() {
        let mut m = IRModule::new();
        let inner_p = m.fresh();
        let inner_ret = m.fresh();
        let outer_p = m.fresh();
        let outer_ret = m.fresh();

        m.instrs.push(Instr::FnDef {
            name: "outer".into(),
            params: vec![("a".into(), outer_p)],
            ret_id: Some(outer_ret),
            body: vec![
                Instr::Param { dst: outer_p, name: "a".into(), index: 0 },
                Instr::FnDef {
                    name: "inner".into(),
                    params: vec![("b".into(), inner_p)],
                    ret_id: Some(inner_ret),
                    body: vec![
                        Instr::Param { dst: inner_p, name: "b".into(), index: 0 },
                        Instr::Return { value: Some(inner_p) },
                    ],
                    reap_threshold: None,
                },
                Instr::Return { value: Some(outer_p) },
            ],
            reap_threshold: None,
        });
        m.instrs.push(Instr::Output(outer_p));

        let parsed = roundtrip(&m);
        assert_eq!(parsed.instrs.len(), m.instrs.len());
        if let Instr::FnDef { name, body, .. } = &parsed.instrs[0] {
            assert_eq!(name, "outer");
            assert_eq!(body.len(), 3);
            // Nested FnDef
            if let Instr::FnDef { name: inner_name, body: inner_body, .. } = &body[1] {
                assert_eq!(inner_name, "inner");
                assert_eq!(inner_body.len(), 2);
            } else {
                panic!("expected nested FnDef");
            }
        } else {
            panic!("expected FnDef");
        }
    }

    #[test]
    fn roundtrip_exports() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 1));
        m.instrs.push(Instr::Output(v0));
        m.exports.insert("foo".into());
        m.exports.insert("bar".into());

        let parsed = roundtrip(&m);
        assert_eq!(parsed.exports, m.exports);
    }

    #[test]
    fn roundtrip_empty_module() {
        let m = IRModule::new();
        let parsed = roundtrip(&m);
        assert!(parsed.instrs.is_empty());
        assert!(parsed.exports.is_empty());
        assert_eq!(parsed.next_id, 0);
    }

    #[test]
    fn roundtrip_return_none() {
        let mut m = IRModule::new();
        m.instrs.push(Instr::Return { value: None });
        let parsed = roundtrip(&m);
        assert_eq!(parsed.instrs.len(), 1);
        assert!(matches!(parsed.instrs[0], Instr::Return { value: None }));
    }

    #[test]
    fn roundtrip_const_f64_special_values() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstF64(v0, f64::NEG_INFINITY));
        m.instrs.push(Instr::ConstF64(v1, 0.0));
        m.instrs.push(Instr::ConstF64(v2, f64::MAX));
        m.instrs.push(Instr::Output(v0));

        let parsed = roundtrip(&m);
        assert_eq!(parsed.instrs.len(), 4);
        if let Instr::ConstF64(_, v) = parsed.instrs[0] {
            assert_eq!(v.to_bits(), f64::NEG_INFINITY.to_bits());
        }
    }

    // -------------------------------------------------------------------------
    // 2. Fixed-point test
    // -------------------------------------------------------------------------

    #[test]
    fn fixed_point() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 42));
        m.instrs.push(Instr::ConstI64(v1, 10));
        m.instrs.push(Instr::BinOp { dst: v2, op: BinOp::Mul, lhs: v0, rhs: v1 });
        m.instrs.push(Instr::Output(v2));

        let bytes1 = emit_mic3(&m);
        let parsed = parse_mic3(&bytes1).unwrap();
        let bytes2 = emit_mic3(&parsed);
        assert_eq!(bytes1, bytes2, "fixed-point: emit(parse(emit(m))) != emit(m)");
    }

    // -------------------------------------------------------------------------
    // 3. Determinism tests
    // -------------------------------------------------------------------------

    #[test]
    fn determinism_same_module_same_bytes() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(4)],
            Some(1.0),
        ));
        m.instrs.push(Instr::Output(v0));
        m.exports.insert("out".into());
        m.instrs.push(Instr::ConstF64(v1, 1.234));

        let b1 = emit_mic3(&m);
        let b2 = emit_mic3(&m);
        let b3 = emit_mic3(&m);
        assert_eq!(b1, b2);
        assert_eq!(b2, b3);
    }

    #[test]
    fn determinism_exports_sorted() {
        // Two modules with the same exports in different insertion order must produce
        // identical bytes (exports are serialized sorted).
        let mut m1 = IRModule::new();
        m1.exports.insert("zzz".into());
        m1.exports.insert("aaa".into());
        m1.exports.insert("mmm".into());

        let mut m2 = IRModule::new();
        m2.exports.insert("mmm".into());
        m2.exports.insert("zzz".into());
        m2.exports.insert("aaa".into());

        assert_eq!(emit_mic3(&m1), emit_mic3(&m2));
    }

    // -------------------------------------------------------------------------
    // 4. Cross-check vs mic@1
    // -------------------------------------------------------------------------

    #[test]
    fn cross_check_simple() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 10));
        m.instrs.push(Instr::ConstI64(v1, 20));
        m.instrs.push(Instr::BinOp { dst: v2, op: BinOp::Add, lhs: v0, rhs: v1 });
        m.instrs.push(Instr::Output(v2));

        let from_mic3 = parse_mic3(&emit_mic3(&m)).unwrap();
        let from_mic1 = crate::ir::compact::parse_mic(&crate::ir::compact::emit_mic(&m)).unwrap();

        assert_eq!(mic1_canonical(&from_mic3), mic1_canonical(&from_mic1),
            "mic@3 and mic@1 round-trips produce different modules");
    }

    #[test]
    fn cross_check_tensor_ops() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::BF16,
            vec![ShapeDim::Known(2), ShapeDim::Known(3)],
            Some(0.0),
        ));
        m.instrs.push(Instr::Sum { dst: v1, src: v0, axes: vec![0], keepdims: false });
        m.instrs.push(Instr::Transpose { dst: v2, src: v0, perm: vec![1, 0] });
        m.instrs.push(Instr::Output(v1));

        let from_mic3 = parse_mic3(&emit_mic3(&m)).unwrap();
        let from_mic1 = crate::ir::compact::parse_mic(&crate::ir::compact::emit_mic(&m)).unwrap();

        assert_eq!(mic1_canonical(&from_mic3), mic1_canonical(&from_mic1));
    }

    #[test]
    fn cross_check_conv2d() {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            v0,
            DType::F32,
            vec![ShapeDim::Known(1), ShapeDim::Known(4), ShapeDim::Known(4), ShapeDim::Known(1)],
            None,
        ));
        m.instrs.push(Instr::ConstTensor(
            v1,
            DType::F32,
            vec![ShapeDim::Known(3), ShapeDim::Known(3), ShapeDim::Known(1), ShapeDim::Known(8)],
            None,
        ));
        m.instrs.push(Instr::Conv2d {
            dst: v2,
            input: v0,
            filter: v1,
            stride_h: 1,
            stride_w: 1,
            padding: ConvPadding::Valid,
        });
        m.instrs.push(Instr::Output(v2));

        let from_mic3 = parse_mic3(&emit_mic3(&m)).unwrap();
        let from_mic1 = crate::ir::compact::parse_mic(&crate::ir::compact::emit_mic(&m)).unwrap();
        assert_eq!(mic1_canonical(&from_mic3), mic1_canonical(&from_mic1));
    }

    // -------------------------------------------------------------------------
    // 5. std-surface gated tests
    // -------------------------------------------------------------------------

    #[cfg(feature = "std-surface")]
    mod std_surface_tests {
        use super::*;
        use crate::ast::{CallConv, TypeAnn};

        #[test]
        fn roundtrip_while() {
            let mut m = IRModule::new();
            let cond = m.fresh();
            let body_v = m.fresh();
            let live_v = m.fresh();
            let init_v = m.fresh();

            m.instrs.push(Instr::While {
                cond_id: cond,
                cond_instrs: vec![Instr::ConstI64(cond, 1)],
                body: vec![Instr::ConstI64(body_v, 42)],
                live_vars: vec![("i".into(), live_v)],
                init_ids: vec![init_v],
            });
            m.instrs.push(Instr::Output(cond));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
            if let Instr::While { cond_instrs, body, live_vars, init_ids, .. } = &parsed.instrs[0] {
                assert_eq!(cond_instrs.len(), 1);
                assert_eq!(body.len(), 1);
                assert_eq!(live_vars.len(), 1);
                assert_eq!(live_vars[0].0, "i");
                assert_eq!(init_ids.len(), 1);
            } else {
                panic!("expected While");
            }
        }

        #[test]
        fn roundtrip_if() {
            let mut m = IRModule::new();
            let cond = m.fresh();
            let then_r = m.fresh();
            let else_r = m.fresh();
            let dst = m.fresh();

            m.instrs.push(Instr::If {
                cond_id: cond,
                cond_instrs: vec![Instr::ConstI64(cond, 1)],
                then_instrs: vec![Instr::ConstI64(then_r, 10)],
                then_result: then_r,
                else_instrs: vec![Instr::ConstI64(else_r, 20)],
                else_result: else_r,
                dst,
                branch_bindings: vec![("x".into(), dst)],
            });
            m.instrs.push(Instr::Output(dst));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
            if let Instr::If {
                then_instrs, else_instrs, branch_bindings, ..
            } = &parsed.instrs[0]
            {
                assert_eq!(then_instrs.len(), 1);
                assert_eq!(else_instrs.len(), 1);
                assert_eq!(branch_bindings.len(), 1);
                assert_eq!(branch_bindings[0].0, "x");
            } else {
                panic!("expected If");
            }
        }

        #[test]
        fn roundtrip_region() {
            let mut m = IRModule::new();
            let result = m.fresh();
            let enter_id = m.fresh();
            let exit_id = m.fresh();
            let alloc = m.fresh();

            m.instrs.push(Instr::Region {
                body: vec![Instr::ConstI64(result, 7)],
                result,
                enter_id,
                exit_id,
                alloc_ids: vec![alloc],
            });
            m.instrs.push(Instr::Output(result));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
            if let Instr::Region { body, alloc_ids, result: r, enter_id: ei, exit_id: xi } =
                &parsed.instrs[0]
            {
                assert_eq!(body.len(), 1);
                assert_eq!(alloc_ids.len(), 1);
                assert_eq!(*r, result);
                assert_eq!(*ei, enter_id);
                assert_eq!(*xi, exit_id);
            } else {
                panic!("expected Region");
            }
        }

        #[test]
        fn roundtrip_vec_ops() {
            let mut m = IRModule::new();
            let base = m.fresh();
            let off = m.fresh();
            let dst_load = m.fresh();
            let _b = m.fresh();
            let _acc = m.fresh();
            let fma_dst = m.fresh();
            let red_dst = m.fresh();

            m.instrs.push(Instr::ConstI64(base, 0));
            m.instrs.push(Instr::ConstI64(off, 0));
            m.instrs.push(Instr::VecLoad { dst: dst_load, base, offset: off, lanes: 8 });
            m.instrs.push(Instr::VecFma { dst: fma_dst, a: dst_load, b: dst_load, acc: dst_load, lanes: 8 });
            m.instrs.push(Instr::VecReduceAdd { dst: red_dst, src: fma_dst, lanes: 8 });
            m.instrs.push(Instr::VecStore { src: dst_load, base, offset: off, lanes: 8 });
            m.instrs.push(Instr::Output(red_dst));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
            if let Instr::VecLoad { lanes, .. } = parsed.instrs[2] {
                assert_eq!(lanes, 8);
            }
        }

        #[test]
        fn roundtrip_vec_i32_q16_ops() {
            let mut m = IRModule::new();
            let base = m.fresh();
            let off = m.fresh();
            let vi32 = m.fresh();
            let _acc = m.fresh();
            let muladd = m.fresh();
            let red = m.fresh();

            m.instrs.push(Instr::ConstI64(base, 0));
            m.instrs.push(Instr::ConstI64(off, 0));
            m.instrs.push(Instr::VecLoadI32 { dst: vi32, base, offset: off, lanes: 8 });
            m.instrs.push(Instr::VecMulAddQ16 { dst: muladd, a: vi32, b: vi32, acc: vi32, lanes: 8 });
            m.instrs.push(Instr::VecReduceAddI64 { dst: red, src: muladd, lanes: 8 });
            m.instrs.push(Instr::Output(red));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
        }

        #[test]
        fn roundtrip_const_array_and_array_load() {
            let mut m = IRModule::new();
            let arr = m.fresh();
            let idx = m.fresh();
            let elem = m.fresh();

            m.instrs.push(Instr::ConstArray {
                dst: arr,
                name: Some("MY_TABLE".into()),
                values: vec![1, 2, 3, 4],
            });
            m.instrs.push(Instr::ConstI64(idx, 2));
            m.instrs.push(Instr::ArrayLoad { dst: elem, base: arr, index: idx });
            m.instrs.push(Instr::Output(elem));

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), m.instrs.len());
            if let Instr::ConstArray { name, values, .. } = &parsed.instrs[0] {
                assert_eq!(name.as_deref(), Some("MY_TABLE"));
                assert_eq!(values, &[1i64, 2, 3, 4]);
            } else {
                panic!("expected ConstArray");
            }
        }

        #[test]
        fn roundtrip_extern_fn_decl() {
            let mut m = IRModule::new();
            m.instrs.push(Instr::ExternFnDecl {
                name: "printf".into(),
                param_types: vec!["!llvm.ptr".into()],
                ret_type: Some("i64".into()),
                is_varargs: true,
                vararg_hints: vec!["i64".into(), "!llvm.ptr".into()],
                callconv: CallConv::C,
            });

            let parsed = roundtrip(&m);
            assert_eq!(parsed.instrs.len(), 1);
            if let Instr::ExternFnDecl {
                name, param_types, ret_type, is_varargs, vararg_hints, callconv,
            } = &parsed.instrs[0]
            {
                assert_eq!(name, "printf");
                assert_eq!(param_types, &["!llvm.ptr"]);
                assert_eq!(ret_type.as_deref(), Some("i64"));
                assert!(*is_varargs);
                assert_eq!(vararg_hints, &["i64", "!llvm.ptr"]);
                assert_eq!(*callconv, CallConv::C);
            } else {
                panic!("expected ExternFnDecl");
            }
        }

        #[test]
        fn roundtrip_struct_defs() {
            let mut m = IRModule::new();
            m.struct_defs.insert("Point".into(), vec!["x".into(), "y".into()]);
            m.struct_defs.insert("Vec3".into(), vec!["x".into(), "y".into(), "z".into()]);

            let parsed = roundtrip(&m);
            assert_eq!(parsed.struct_defs, m.struct_defs);
        }

        #[test]
        fn roundtrip_const_array_defs() {
            let mut m = IRModule::new();
            m.const_array_defs.insert("TABLE".into(), vec![10, 20, 30]);

            let parsed = roundtrip(&m);
            assert_eq!(parsed.const_array_defs, m.const_array_defs);
        }

        #[test]
        fn roundtrip_repr_c_structs() {
            let mut m = IRModule::new();
            m.repr_c_structs.insert(
                "CPoint".into(),
                vec![TypeAnn::ScalarI64, TypeAnn::ScalarF64],
            );

            let parsed = roundtrip(&m);
            assert_eq!(parsed.repr_c_structs.len(), m.repr_c_structs.len());
            let fields = parsed.repr_c_structs.get("CPoint").unwrap();
            assert_eq!(fields.len(), 2);
        }

        #[test]
        fn fixed_point_std_surface() {
            let mut m = IRModule::new();
            let cond = m.fresh();
            m.instrs.push(Instr::While {
                cond_id: cond,
                cond_instrs: vec![Instr::ConstI64(cond, 1)],
                body: vec![],
                live_vars: vec![],
                init_ids: vec![],
            });
            m.struct_defs.insert("S".into(), vec!["a".into()]);
            m.const_array_defs.insert("T".into(), vec![1, 2]);

            let b1 = emit_mic3(&m);
            let parsed = parse_mic3(&b1).unwrap();
            let b2 = emit_mic3(&parsed);
            assert_eq!(b1, b2);
        }
    }
}
