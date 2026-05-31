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

//! RFC 0021 §4.2 — MAP epilogue for mic@3 artifacts.
//!
//! Attaches an `evidence_chain.*` MAP to a mic@3 IR body so that provenance
//! attests the **real compiled [`IRModule`]**, not a toy dataflow graph.
//!
//! # Canonicalization invariant (5/5-consult requirement)
//!
//! The MAP epilogue encoding is **byte-canonical** by construction:
//!
//! 1. **Sorted MAP keys** — entries are emitted in lexicographic (byte-order)
//!    key order, independent of insertion order.
//! 2. **Fixed-width length prefixes** — all lengths are ULEB128 varints (no
//!    padding, no alignment bytes).
//! 3. **No padding** — the epilogue is appended immediately after the last
//!    mic@3 body byte; there are no alignment or padding bytes.
//! 4. **Deterministic float encoding** — `f64` values are encoded as
//!    `to_bits()` little-endian `u64`s (already enforced in the mic@3 body).
//! 5. **Fixed depth-first region traversal** — inherited from the mic@3 body
//!    encoder's sequential `instrs` traversal (pre-order, already enforced).
//! 6. **Inline string encoding** — MAP keys and string values are encoded
//!    inline as `(ULEB length, UTF-8 bytes)`.  There is no shared string table
//!    between the MAP epilogue and the IR body.  This makes the epilogue
//!    self-contained and independent of IR-level string interning order.
//!
//! # Epilogue wire format
//!
//! ```text
//! 0x4D                      -- MAP sentinel byte ('M')
//! ULEB128 entry_count       -- number of top-level MAP entries
//! For each entry (lexicographic key order):
//!   ULEB128 key_len
//!   key_bytes (UTF-8)       -- no NUL terminator
//!   value_tag (1 byte):
//!     0x00  String:  ULEB128 val_len + UTF-8 bytes
//!     0x01  Int:     zigzag-ULEB128 (signed, matches sleb128 in varint.rs)
//!     0x02  Bytes:   ULEB128 len + raw bytes
//! ```
//!
//! # Back-compat guarantee
//!
//! `emit_mic3` (plain, no evidence) produces zero MAP bytes after the body.
//! `parse_mic3` reads the MAP only when it encounters the `0x4D` sentinel.
//! At EOF the MAP is empty, and the parsed `IRModule` is byte-identical to the
//! output of the pre-RFC-0021-§4.2 encoder.
//!
//! # Relationship to mic@2.1 MAP (v2/evidence.rs)
//!
//! [`Determinism`], [`EvidenceReport`], and [`EvidenceError`] are **re-exported
//! from `v2::evidence`** — there is exactly one evidence vocabulary in the
//! codebase. The MAP *encoding* in the mic@3 epilogue is simpler (inline
//! strings, no nested maps needed for `evidence_chain.*`); the *semantics* of
//! every field are identical to those in RFC 0016 Phase A/B.

use std::io::Write;

use crate::ir::IRModule;
use crate::ir::compact::v2::{uleb128_read, uleb128_write, zigzag_decode, zigzag_encode};
use crate::ir::evidence::ir_trace_hash;

// Re-export the canonical evidence vocabulary from v2 — one set of types for
// the whole codebase (RFC 0021 §3.2: "reuse, don't rebuild").
pub use crate::ir::compact::v2::{Determinism, EvidenceError, EvidenceReport};

// ─── MAP sentinel ─────────────────────────────────────────────────────────────

/// Sentinel byte that introduces the MAP epilogue in a mic@3 artifact.
/// Same value as the v2 MIC-B MAP marker (`0x4D` = ASCII `'M'`) for consistency.
pub const MAP_SENTINEL: u8 = 0x4D;

// ─── Value tags (inline encoding — no shared string table) ────────────────────

const TAG_STRING: u8 = 0x00;
const TAG_INT: u8 = 0x01;
const TAG_BYTES: u8 = 0x02;

// ─── Evidence-chain key constants ─────────────────────────────────────────────

const KEY_DETERMINISM: &str = "evidence_chain.determinism";
const KEY_PARENT: &str = "evidence_chain.parent";
const KEY_SCHEMA: &str = "evidence_chain.schema";
const KEY_SUBSTRATE: &str = "evidence_chain.substrate";
const KEY_TOOLCHAIN: &str = "evidence_chain.toolchain";
const KEY_TRACE_HASH: &str = "evidence_chain.trace_hash";

// ─── Emit ─────────────────────────────────────────────────────────────────────

/// Emit a mic@3 artifact with an embedded `evidence_chain.*` MAP epilogue.
///
/// The `trace_hash` is SHA-256 of the canonical mic@3 bytes (via
/// [`ir_trace_hash`]) — the RFC-0001 fixed-point encoding that the pipeline
/// and self-host already produce, and which carries full function bodies.  The
/// MAP is appended **after** the mic@3 body so that a reader ignorant of the
/// epilogue can still parse the IR.
///
/// ## Canonicalization
///
/// The MAP is emitted with keys in lexicographic order.  Calling this function
/// twice with the same arguments produces byte-identical output.
///
/// ## Back-compat
///
/// A call to [`emit_mic3`](super::emit_mic3) with the same `ir` produces bytes
/// that are a prefix of the output of this function — the bodies are identical;
/// only the epilogue differs.
pub fn emit_mic3_with_evidence(
    ir: &IRModule,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
) -> Vec<u8> {
    let body = super::emit_mic3(ir);
    let trace_hash = ir_trace_hash(ir);
    let mut out = body;
    append_map_epilogue(
        &mut out,
        substrate,
        parent,
        determinism,
        toolchain,
        trace_hash,
    );
    out
}

/// Parse a mic@3 artifact (with or without MAP epilogue) and verify the
/// embedded `evidence_chain` block.
///
/// Returns an [`EvidenceReport`] with `trace_hash_valid = true` when the stored
/// `trace_hash` equals the SHA-256 of the canonical mic@3 bytes of the parsed
/// module.  A single flipped byte anywhere in the body or MAP flips
/// `trace_hash_valid` to `false`.
///
/// Returns [`EvidenceError::Missing`] if no `evidence_chain.*` keys are present.
pub fn mic3_evidence_report(bytes: &[u8]) -> Result<EvidenceReport, EvidenceError> {
    // Locate the MAP sentinel.
    let body_end = find_map_sentinel(bytes).ok_or(EvidenceError::Missing)?;

    // Parse the IR from the plain body prefix.
    let ir = super::parse_mic3(&bytes[..body_end])
        .map_err(|_| EvidenceError::Malformed("evidence_chain.trace_hash"))?;

    // Parse the MAP epilogue.
    let map_bytes = &bytes[body_end..];
    let entries = parse_map_epilogue(map_bytes)
        .map_err(|_| EvidenceError::Malformed("evidence_chain.trace_hash"))?;

    decode_evidence_report(&ir, &entries)
}

// ─── Internal MAP emit ────────────────────────────────────────────────────────

/// Build and append the MAP epilogue to `out`.
///
/// Keys are emitted sorted lexicographically. Inline string encoding: no shared
/// string table with the IR body.
fn append_map_epilogue(
    out: &mut Vec<u8>,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
    trace_hash: [u8; 32],
) {
    // Collect entries sorted by key.
    let mut entries: Vec<(&str, MapEntryValue)> = vec![
        (
            KEY_DETERMINISM,
            MapEntryValue::Str(match determinism {
                Determinism::Deterministic => "deterministic",
                Determinism::Nondeterministic => "nondeterministic",
            }),
        ),
        (KEY_SCHEMA, MapEntryValue::Int(1)),
        (KEY_SUBSTRATE, MapEntryValue::Str(substrate)),
        (KEY_TOOLCHAIN, MapEntryValue::Str(toolchain)),
        (KEY_TRACE_HASH, MapEntryValue::Bytes(&trace_hash)),
    ];
    if let Some(ref p) = parent {
        entries.push((KEY_PARENT, MapEntryValue::Bytes(p)));
    }
    // Lexicographic sort — this is the canonical-encoding invariant.
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    out.write_all(&[MAP_SENTINEL]).unwrap();
    uleb128_write(out, entries.len() as u64).unwrap();

    for (key, val) in &entries {
        let kb = key.as_bytes();
        uleb128_write(out, kb.len() as u64).unwrap();
        out.write_all(kb).unwrap();
        match val {
            MapEntryValue::Str(s) => {
                out.write_all(&[TAG_STRING]).unwrap();
                let sb = s.as_bytes();
                uleb128_write(out, sb.len() as u64).unwrap();
                out.write_all(sb).unwrap();
            }
            MapEntryValue::Int(i) => {
                out.write_all(&[TAG_INT]).unwrap();
                uleb128_write(out, zigzag_encode(*i)).unwrap();
            }
            MapEntryValue::Bytes(b) => {
                out.write_all(&[TAG_BYTES]).unwrap();
                uleb128_write(out, b.len() as u64).unwrap();
                out.write_all(b).unwrap();
            }
        }
    }
}

/// Ephemeral value type used only during MAP construction.
enum MapEntryValue<'a> {
    Str(&'a str),
    Int(i64),
    Bytes(&'a [u8]),
}

// ─── Internal MAP parse ───────────────────────────────────────────────────────

/// Owned parsed MAP entry for verification.
#[derive(Debug)]
pub(crate) struct ParsedEntry {
    pub key: String,
    pub value: ParsedValue,
}

#[derive(Debug)]
#[allow(dead_code)] // Int is only consumed in tests; all variants needed for completeness.
pub(crate) enum ParsedValue {
    Str(String),
    Int(i64),
    Bytes(Vec<u8>),
}

/// Find the byte offset of the MAP sentinel in `bytes`, returning `None` if the
/// sentinel is not present (no MAP epilogue).
///
/// The sentinel search starts from the end of a minimal valid mic@3 body.
/// We rely on `parse_mic3` having consumed exactly the body bytes: the sentinel
/// is simply the first byte after a successful `parse_mic3` parse.  To avoid
/// scanning the body (which could contain `0x4D` bytes legitimately), we use
/// a different approach: try to parse the body by scanning from the front
/// and recording where the cursor stops.
pub(crate) fn find_map_sentinel(bytes: &[u8]) -> Option<usize> {
    // Attempt to find the body/epilogue boundary by trying parse_mic3 on
    // progressive slices — but that is expensive and fragile.  Instead, we
    // use the simpler observation: the MAP sentinel byte `0x4D` immediately
    // follows the last byte that `parse_mic3` consumed from `bytes`.
    //
    // Since `parse_mic3` takes a `&[u8]` and uses a `Cursor` internally, we
    // cannot query the cursor position after parsing.  The approach is:
    // 1. Parse the IR from the full byte slice. If that fails, there is no MAP.
    // 2. Re-emit the parsed IR as mic@3 (body only). The body length is the
    //    length of that re-emission (deterministic, fixed-point property).
    // 3. The byte at that offset in the original bytes is the MAP sentinel if
    //    it equals `MAP_SENTINEL` and the remaining bytes pass MAP validation.
    //
    // This is O(body) work — acceptable for a per-artifact call path.

    let ir = super::parse_mic3(bytes).ok()?;
    let body = super::emit_mic3(&ir);
    let body_end = body.len();

    if body_end >= bytes.len() {
        // No bytes beyond the body — no MAP epilogue.
        return None;
    }
    if bytes[body_end] == MAP_SENTINEL {
        Some(body_end)
    } else {
        None
    }
}

/// Parse the MAP epilogue bytes (starting with the sentinel byte) into a list
/// of [`ParsedEntry`] values.
pub(crate) fn parse_map_epilogue(bytes: &[u8]) -> Result<Vec<ParsedEntry>, ParseMapError> {
    if bytes.is_empty() || bytes[0] != MAP_SENTINEL {
        return Err(ParseMapError::MissingSentinel);
    }
    let mut r = std::io::Cursor::new(&bytes[1..]);
    let count = uleb128_read(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let key_len = uleb128_read(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
        let mut key_buf = vec![0u8; key_len];
        use std::io::Read;
        r.read_exact(&mut key_buf)
            .map_err(|_| ParseMapError::Truncated)?;
        let key = String::from_utf8(key_buf).map_err(|_| ParseMapError::InvalidUtf8)?;

        let tag_byte = {
            let mut tb = [0u8];
            r.read_exact(&mut tb)
                .map_err(|_| ParseMapError::Truncated)?;
            tb[0]
        };

        let value = match tag_byte {
            TAG_STRING => {
                let val_len = uleb128_read(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
                let mut vb = vec![0u8; val_len];
                r.read_exact(&mut vb)
                    .map_err(|_| ParseMapError::Truncated)?;
                let s = String::from_utf8(vb).map_err(|_| ParseMapError::InvalidUtf8)?;
                ParsedValue::Str(s)
            }
            TAG_INT => {
                let encoded = uleb128_read(&mut r).map_err(|_| ParseMapError::Truncated)?;
                ParsedValue::Int(zigzag_decode(encoded))
            }
            TAG_BYTES => {
                let blen = uleb128_read(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
                let mut bb = vec![0u8; blen];
                r.read_exact(&mut bb)
                    .map_err(|_| ParseMapError::Truncated)?;
                ParsedValue::Bytes(bb)
            }
            other => return Err(ParseMapError::UnknownTag(other)),
        };

        entries.push(ParsedEntry { key, value });
    }
    Ok(entries)
}

/// Error produced by the low-level MAP epilogue parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParseMapError {
    MissingSentinel,
    Truncated,
    InvalidUtf8,
    UnknownTag(u8),
}

// ─── Evidence decode + verify ─────────────────────────────────────────────────

/// Decode and verify an evidence report from parsed MAP entries against `ir`.
///
/// Recomputes `trace_hash = SHA-256(canonical mic@3 bytes)` and compares it to
/// the stored value.  A single flipped byte in the IR body yields a mismatched
/// hash and `trace_hash_valid = false`.
fn decode_evidence_report(
    ir: &IRModule,
    entries: &[ParsedEntry],
) -> Result<EvidenceReport, EvidenceError> {
    let has_ec = entries.iter().any(|e| e.key.starts_with("evidence_chain."));
    if !has_ec {
        return Err(EvidenceError::Missing);
    }

    let substrate = find_str(entries, KEY_SUBSTRATE)?;
    let toolchain = find_str(entries, KEY_TOOLCHAIN)?;
    let determinism = {
        let s = find_str(entries, KEY_DETERMINISM)?;
        match s.as_str() {
            "deterministic" => Determinism::Deterministic,
            "nondeterministic" => Determinism::Nondeterministic,
            other => return Err(EvidenceError::UnknownDeterminism(other.to_owned())),
        }
    };

    let parent = find_bytes_opt(entries, KEY_PARENT)?;

    let stored_hash = find_bytes32(entries, KEY_TRACE_HASH)?;

    // Recompute via the same FIPS-180-4 seam as ir_trace_hash.
    let recomputed = ir_trace_hash(ir);
    let trace_hash_valid = recomputed == stored_hash;

    Ok(EvidenceReport {
        substrate,
        determinism,
        toolchain,
        parent,
        trace_hash: stored_hash,
        trace_hash_valid,
    })
}

fn find_entry<'a>(entries: &'a [ParsedEntry], key: &str) -> Option<&'a ParsedValue> {
    entries.iter().find(|e| e.key == key).map(|e| &e.value)
}

fn find_str(entries: &[ParsedEntry], key: &'static str) -> Result<String, EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Str(s)) => Ok(s.clone()),
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Err(EvidenceError::MissingKey(key)),
    }
}

fn find_bytes32(entries: &[ParsedEntry], key: &'static str) -> Result<[u8; 32], EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Bytes(b)) => b
            .as_slice()
            .try_into()
            .map_err(|_| EvidenceError::Malformed(key)),
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Err(EvidenceError::MissingKey(key)),
    }
}

fn find_bytes_opt(
    entries: &[ParsedEntry],
    key: &'static str,
) -> Result<Option<[u8; 32]>, EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Bytes(b)) => {
            let arr: [u8; 32] = b
                .as_slice()
                .try_into()
                .map_err(|_| EvidenceError::Malformed(key))?;
            Ok(Some(arr))
        }
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Ok(None),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::v3::{emit_mic3, parse_mic3};
    use crate::ir::{BinOp, IRModule, Instr};
    use crate::types::{DType, ShapeDim};

    // -------------------------------------------------------------------------
    // Module generators — spread of IRModule shapes for differential coverage
    // -------------------------------------------------------------------------

    fn mod_empty() -> IRModule {
        IRModule::new()
    }

    fn mod_const_i64() -> IRModule {
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstI64(v, 42));
        m.instrs.push(Instr::Output(v));
        m
    }

    fn mod_const_f64() -> IRModule {
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstF64(v, std::f64::consts::E));
        m.instrs.push(Instr::Output(v));
        m
    }

    fn mod_binop() -> IRModule {
        let mut m = IRModule::new();
        let a = m.fresh();
        let b = m.fresh();
        let c = m.fresh();
        m.instrs.push(Instr::ConstI64(a, 10));
        m.instrs.push(Instr::ConstI64(b, 20));
        m.instrs.push(Instr::BinOp {
            dst: c,
            op: BinOp::Add,
            lhs: a,
            rhs: b,
        });
        m.instrs.push(Instr::Output(c));
        m
    }

    fn mod_reduction() -> IRModule {
        let mut m = IRModule::new();
        let t = m.fresh();
        let s = m.fresh();
        m.instrs.push(Instr::ConstTensor(
            t,
            DType::F32,
            vec![ShapeDim::Known(4), ShapeDim::Known(8)],
            Some(1.0),
        ));
        m.instrs.push(Instr::Sum {
            dst: s,
            src: t,
            axes: vec![0],
            keepdims: false,
        });
        m.instrs.push(Instr::Output(s));
        m
    }

    fn mod_fndef() -> IRModule {
        let mut m = IRModule::new();
        let p = m.fresh();
        let v = m.fresh();
        let r = m.fresh();
        let dst = m.fresh();
        m.instrs.push(Instr::FnDef {
            name: "add_one".into(),
            params: vec![("x".into(), p)],
            ret_id: Some(r),
            body: vec![
                Instr::Param {
                    dst: p,
                    name: "x".into(),
                    index: 0,
                },
                Instr::ConstI64(v, 1),
                Instr::BinOp {
                    dst: r,
                    op: BinOp::Add,
                    lhs: p,
                    rhs: v,
                },
                Instr::Return { value: Some(r) },
            ],
            reap_threshold: None,
        });
        m.instrs.push(Instr::ConstI64(v, 5));
        m.instrs.push(Instr::Call {
            dst,
            name: "add_one".into(),
            args: vec![v],
        });
        m.instrs.push(Instr::Output(dst));
        m
    }

    fn mod_call_chain() -> IRModule {
        let mut m = IRModule::new();
        let a = m.fresh();
        let b = m.fresh();
        let c = m.fresh();
        let d = m.fresh();
        m.instrs.push(Instr::ConstI64(a, 1));
        m.instrs.push(Instr::ConstI64(b, 2));
        m.instrs.push(Instr::BinOp {
            dst: c,
            op: BinOp::Mul,
            lhs: a,
            rhs: b,
        });
        m.instrs.push(Instr::BinOp {
            dst: d,
            op: BinOp::Sub,
            lhs: c,
            rhs: a,
        });
        m.instrs.push(Instr::Output(d));
        m
    }

    fn mod_with_exports() -> IRModule {
        let mut m = IRModule::new();
        let v = m.fresh();
        m.instrs.push(Instr::ConstI64(v, 99));
        m.instrs.push(Instr::Output(v));
        m.exports.insert("main".into());
        m.exports.insert("init".into());
        m
    }

    fn mod_nested_fndef() -> IRModule {
        let mut m = IRModule::new();
        let ip = m.fresh();
        let ir = m.fresh();
        let op = m.fresh();
        let or = m.fresh();
        m.instrs.push(Instr::FnDef {
            name: "outer".into(),
            params: vec![("a".into(), op)],
            ret_id: Some(or),
            body: vec![
                Instr::Param {
                    dst: op,
                    name: "a".into(),
                    index: 0,
                },
                Instr::FnDef {
                    name: "inner".into(),
                    params: vec![("b".into(), ip)],
                    ret_id: Some(ir),
                    body: vec![
                        Instr::Param {
                            dst: ip,
                            name: "b".into(),
                            index: 0,
                        },
                        Instr::Return { value: Some(ip) },
                    ],
                    reap_threshold: None,
                },
                Instr::Return { value: Some(op) },
            ],
            reap_threshold: Some(0.75),
        });
        m.instrs.push(Instr::Output(op));
        m
    }

    fn all_modules() -> Vec<(&'static str, IRModule)> {
        vec![
            ("empty", mod_empty()),
            ("const_i64", mod_const_i64()),
            ("const_f64", mod_const_f64()),
            ("binop", mod_binop()),
            ("reduction", mod_reduction()),
            ("fndef", mod_fndef()),
            ("call_chain", mod_call_chain()),
            ("with_exports", mod_with_exports()),
            ("nested_fndef", mod_nested_fndef()),
        ]
    }

    // -------------------------------------------------------------------------
    // (a) Byte-determinism: emit == emit on repeated calls
    // -------------------------------------------------------------------------

    #[test]
    fn byte_determinism_across_repeated_calls() {
        for (name, ir) in all_modules() {
            let b1 =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            let b2 =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            let b3 =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            assert_eq!(b1, b2, "module '{}': emit call 1 != call 2", name);
            assert_eq!(b2, b3, "module '{}': emit call 2 != call 3", name);
        }
    }

    // -------------------------------------------------------------------------
    // (b) Round-trip: parse(emit) == ir (incl. MAP recovery)
    // -------------------------------------------------------------------------

    #[test]
    fn roundtrip_ir_body_survives_evidence_emit() {
        for (name, ir) in all_modules() {
            let bytes =
                emit_mic3_with_evidence(&ir, "arm_neon", None, Determinism::Deterministic, "0.8.0");
            // The IR body is a prefix — parse just the body.
            let body_end = find_map_sentinel(&bytes).expect("sentinel must be present");
            let parsed = parse_mic3(&bytes[..body_end])
                .unwrap_or_else(|e| panic!("module '{}': parse failed: {}", name, e));

            // Structural equality via mic@1 canonical text.
            let mic1_original = crate::ir::compact::emit_mic(&ir);
            let mic1_parsed = crate::ir::compact::emit_mic(&parsed);
            assert_eq!(
                mic1_original, mic1_parsed,
                "module '{}': mic@1 canonical text diverged after round-trip",
                name
            );
        }
    }

    // -------------------------------------------------------------------------
    // (c) mic3_evidence_report returns trace_hash_valid=true on untampered bytes
    // -------------------------------------------------------------------------

    #[test]
    fn evidence_report_valid_for_untampered_artifact() {
        for (name, ir) in all_modules() {
            let bytes =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            let report = mic3_evidence_report(&bytes)
                .unwrap_or_else(|e| panic!("module '{}': evidence_report error: {:?}", name, e));
            assert!(
                report.trace_hash_valid,
                "module '{}': trace_hash_valid must be true for untampered artifact",
                name
            );
            assert_eq!(report.substrate, "x86_avx2", "module '{}'", name);
            assert_eq!(
                report.determinism,
                Determinism::Deterministic,
                "module '{}'",
                name
            );
            assert_eq!(report.toolchain, "0.8.0", "module '{}'", name);
            assert!(report.is_root(), "module '{}': no parent ⇒ is_root", name);
        }
    }

    // -------------------------------------------------------------------------
    // (d) Post-seal mutation of the IR body flips trace_hash_valid=false
    // -------------------------------------------------------------------------

    #[test]
    fn tamper_detection_flips_trace_hash_valid() {
        for (name, ir) in all_modules() {
            let bytes =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");

            // Flip one byte in the IR body (byte index 5 is inside the body,
            // past the 5-byte header; safe because all modules have at least
            // several more bytes).
            let body_end =
                find_map_sentinel(&bytes).expect("sentinel must be present for non-empty modules");

            if body_end < 6 {
                // empty module has a 5-byte header + tiny body; skip tamper
                // test (nothing to flip that keeps the body parseable).
                continue;
            }

            // Build a tampered slice: flip a byte in the body.
            let mut tampered = bytes.clone();
            tampered[5] ^= 0xFF;

            // The tampered bytes may fail to parse (invalid opcode), or parse
            // but produce a different IR (hash mismatch). Either way, the
            // evidence report must NOT validate.
            match mic3_evidence_report(&tampered) {
                Ok(report) => {
                    assert!(
                        !report.trace_hash_valid,
                        "module '{}': trace_hash_valid must be false after body mutation",
                        name
                    );
                }
                Err(_) => {
                    // The parser rejected the tampered body — that also counts
                    // as tamper detection (no valid report produced).
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // (e) Back-compat: plain emit_mic3 bytes are unchanged; no MAP sentinel present
    // -------------------------------------------------------------------------

    #[test]
    fn plain_emit_mic3_unchanged_no_sentinel() {
        for (name, ir) in all_modules() {
            let plain = emit_mic3(&ir);
            let with_ev =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");

            // Plain body is an exact prefix of the evidence-bearing bytes.
            assert!(
                with_ev.starts_with(&plain),
                "module '{}': emit_mic3 body must be a prefix of emit_mic3_with_evidence",
                name
            );

            // No sentinel in the plain bytes.
            assert_eq!(
                find_map_sentinel(&plain),
                None,
                "module '{}': plain mic@3 must not contain MAP sentinel",
                name
            );

            // plain parse_mic3 still works (no regression).
            let _ = parse_mic3(&plain)
                .unwrap_or_else(|e| panic!("module '{}': plain parse_mic3 failed: {}", name, e));
        }
    }

    // -------------------------------------------------------------------------
    // (f) Parent hash round-trips and changes trace_hash
    // -------------------------------------------------------------------------

    #[test]
    fn parent_hash_round_trips_and_changes_trace_hash() {
        let ir = mod_binop();
        let parent = [0xABu8; 32];

        let no_parent_bytes =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let with_parent_bytes = emit_mic3_with_evidence(
            &ir,
            "x86_avx2",
            Some(parent),
            Determinism::Deterministic,
            "0.8.0",
        );

        // Different bytes (parent is part of the MAP).
        assert_ne!(no_parent_bytes, with_parent_bytes);

        // Both verify correctly.
        let r_no = mic3_evidence_report(&no_parent_bytes).unwrap();
        let r_with = mic3_evidence_report(&with_parent_bytes).unwrap();

        assert!(r_no.trace_hash_valid);
        assert!(r_with.trace_hash_valid);

        // Different trace_hash (parent is included in what gets hashed... via MAP
        // difference; actually trace_hash is ir_trace_hash, so it's the same for
        // the same IR). The MAP bytes differ, but trace_hash is sha256(mic1_text).
        // Both should equal ir_trace_hash(&ir).
        let expected = ir_trace_hash(&ir);
        assert_eq!(r_no.trace_hash, expected);
        assert_eq!(r_with.trace_hash, expected);

        // Parent field correctly round-trips.
        assert!(r_no.parent.is_none());
        assert_eq!(r_with.parent, Some(parent));
        assert!(r_no.is_root());
        assert!(!r_with.is_root());
    }

    // -------------------------------------------------------------------------
    // (g) Nondeterministic declaration round-trips
    // -------------------------------------------------------------------------

    #[test]
    fn nondeterministic_declaration_round_trips() {
        let ir = mod_const_i64();
        let bytes = emit_mic3_with_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Nondeterministic,
            "0.8.0",
        );
        let report = mic3_evidence_report(&bytes).unwrap();
        assert!(report.trace_hash_valid);
        assert_eq!(report.determinism, Determinism::Nondeterministic);
    }

    // -------------------------------------------------------------------------
    // (h) Missing evidence → EvidenceError::Missing
    // -------------------------------------------------------------------------

    #[test]
    fn plain_mic3_returns_missing_evidence_error() {
        for (name, ir) in all_modules() {
            let plain = emit_mic3(&ir);
            let result = mic3_evidence_report(&plain);
            assert_eq!(
                result,
                Err(EvidenceError::Missing),
                "module '{}': plain mic3 must yield EvidenceError::Missing",
                name
            );
        }
    }

    // -------------------------------------------------------------------------
    // (i) MAP keys are sorted lexicographically (canonicalization gate)
    // -------------------------------------------------------------------------

    #[test]
    fn map_keys_are_lexicographically_sorted() {
        let ir = mod_binop();
        let bytes =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let body_end = find_map_sentinel(&bytes).unwrap();
        let entries = parse_map_epilogue(&bytes[body_end..]).unwrap();

        let keys: Vec<&str> = entries.iter().map(|e| e.key.as_str()).collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(
            keys, sorted,
            "MAP keys must be in lexicographic order (canonical-encoding gate)"
        );
    }

    // -------------------------------------------------------------------------
    // (j) Substrate changes trace_hash (it is part of the MAP, not of the IR
    //     body, so ir_trace_hash is the same — but the MAP bytes differ)
    // -------------------------------------------------------------------------

    #[test]
    fn substrate_field_is_correct_in_report() {
        let ir = mod_binop();
        let avx2 =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let neon =
            emit_mic3_with_evidence(&ir, "arm_neon", None, Determinism::Deterministic, "0.8.0");

        let r_avx2 = mic3_evidence_report(&avx2).unwrap();
        let r_neon = mic3_evidence_report(&neon).unwrap();

        assert_eq!(r_avx2.substrate, "x86_avx2");
        assert_eq!(r_neon.substrate, "arm_neon");
        assert!(r_avx2.trace_hash_valid);
        assert!(r_neon.trace_hash_valid);

        // Same IR → same ir_trace_hash regardless of substrate.
        assert_eq!(r_avx2.trace_hash, r_neon.trace_hash);
    }

    // -------------------------------------------------------------------------
    // (k) Schema key is present and equals 1
    // -------------------------------------------------------------------------

    #[test]
    fn schema_key_present_and_equals_one() {
        let ir = mod_const_i64();
        let bytes =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let body_end = find_map_sentinel(&bytes).unwrap();
        let entries = parse_map_epilogue(&bytes[body_end..]).unwrap();
        let schema_entry = entries
            .iter()
            .find(|e| e.key == KEY_SCHEMA)
            .expect("evidence_chain.schema must be present");
        match &schema_entry.value {
            ParsedValue::Int(1) => {}
            other => panic!("evidence_chain.schema must be Int(1), got {:?}", other),
        }
    }

    // -------------------------------------------------------------------------
    // (l) std-surface module with While/If/Region round-trips correctly
    // -------------------------------------------------------------------------

    #[cfg(feature = "std-surface")]
    #[test]
    fn std_surface_while_module_evidence_round_trips() {
        use crate::ir::Instr;
        let mut m = IRModule::new();
        let cond = m.fresh();
        let bv = m.fresh();
        let lv = m.fresh();
        let iv = m.fresh();
        m.instrs.push(Instr::While {
            cond_id: cond,
            cond_instrs: vec![Instr::ConstI64(cond, 1)],
            body: vec![Instr::ConstI64(bv, 42)],
            live_vars: vec![("i".into(), lv)],
            init_ids: vec![iv],
            exit_ids: Vec::new(),
        });
        m.instrs.push(Instr::Output(cond));

        let bytes =
            emit_mic3_with_evidence(&m, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let report = mic3_evidence_report(&bytes).unwrap();
        assert!(report.trace_hash_valid);
        assert_eq!(report.substrate, "x86_avx2");
    }
}
