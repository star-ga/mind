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

//! RFC 0016 Phase A — compile-time evidence-chain emission.
//!
//! Populates the `evidence_chain.*` reserved namespace in the mic@2.1 MAP
//! section of a [`Graph`].  Phase A is schema + emit, inert, unsigned: no
//! signing, no default-path behavior change.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mind::ir::compact::v2::evidence::{attach_evidence_chain, Determinism};
//!
//! attach_evidence_chain(
//!     &mut graph,
//!     "x86_avx2",
//!     None,            // no parent (root link)
//!     Determinism::Deterministic,
//!     env!("CARGO_PKG_VERSION"),
//! );
//! ```
//!
//! The builder is **opt-in**: it is never called from [`emit_mic2`](super::emit::emit_mic2) or
//! [`emit_micb`] — a graph without evidence attached emits byte-identically
//! to pre-RFC-0016 output (the MAP epilogue is omitted when empty).

use super::binary::emit_micb;
use super::types::{Graph, Map, MapValue};

/// Thin, swappable seam over the bootstrap compiler's FIPS-180-4 SHA-256
/// (`crate::deps::mini_sha256`). Per RFC 0016 §5.4, `trace_hash` is SHA-256 of
/// the canonical mic@2.1 binary BYTES — so the bootstrap's Rust hash and the
/// pure-MIND `std.sha256` (which runs the same FIPS algorithm over the same
/// bytes) are bit-identical. This one-line seam is the swap point for the
/// self-host endpoint (Phase A.5). One Rust SHA-256 is reused — no duplicate.
fn trace_sha256(bytes: &[u8]) -> [u8; 32] {
    crate::deps::mini_sha256(bytes)
}

/// Determinism declaration for an evidence-chain link (RFC 0016 §3).
///
/// `Deterministic` is the default per §5.1.  `Nondeterministic` emits a
/// refusal-to-attest-identity marker rather than silently omitting evidence.
///
/// # TODO(#289)
///
/// Source the value from the type-checker determinism attribute once the
/// default-flip lands (#289 / RFC 0012 Phase D).  For Phase A it is always
/// supplied by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Determinism {
    /// Graph is deterministic (default).
    #[default]
    Deterministic,
    /// Graph explicitly opts out of determinism (`#[nondeterministic]`).
    Nondeterministic,
}

impl Determinism {
    fn as_str(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Nondeterministic => "nondeterministic",
        }
    }
}

/// Attach an `evidence_chain.*` block to `graph.map` (RFC 0016 Phase A).
///
/// Populates the following keys per §3:
///
/// | Key | Type | Notes |
/// |-----|------|-------|
/// | `evidence_chain.substrate`    | string  | RFC 0014 canonical tier id |
/// | `evidence_chain.determinism`  | string  | `"deterministic"` or `"nondeterministic"` |
/// | `evidence_chain.toolchain`    | string  | `mindc` version string |
/// | `evidence_chain.parent`       | bytes32 | only when `parent` is `Some` |
/// | `evidence_chain.trace_hash`   | bytes32 | §3.2 self-reference hash (last) |
///
/// ## §3.2 self-reference rule
///
/// `trace_hash` = SHA-256 of the canonical MIC-B binary of the graph whose MAP
/// contains `substrate`, `parent` (if any), `determinism`, and `toolchain`,
/// but **excludes** both `signature.*` and `evidence_chain.trace_hash` itself.
///
/// Algorithm:
/// 1. Insert substrate / parent / determinism / toolchain into `graph.map`.
/// 2. Build a temporary graph with a stripped MAP (remove `signature.*` and
///    `evidence_chain.trace_hash`).
/// 3. Emit MIC-B of the stripped graph.
/// 4. SHA-256 the bytes → 32-byte hash.
/// 5. Insert `evidence_chain.trace_hash = bytes(hash)` into the real map.
///
/// A verifier recomputes steps 2-4 and compares; a single flipped byte fails.
///
/// ## Opt-in guarantee
///
/// This function is **never called** from [`emit_mic2`](super::emit::emit_mic2) or [`emit_micb`].
/// Default emit paths are unmodified; an unannotated graph emits byte-identically
/// to pre-Phase-A output.
///
/// ## Panics
///
/// Panics if `graph.map` already contains any `evidence_chain.*` key (would
/// indicate a double-call; use [`remove_evidence_chain`] first if re-attaching).
pub fn attach_evidence_chain(
    graph: &mut Graph,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
) {
    // Guard against double-attach.
    assert!(
        !graph
            .map
            .iter()
            .any(|(k, _)| k.starts_with("evidence_chain.")),
        "evidence_chain.* keys already present; call remove_evidence_chain first"
    );

    // Step 1: insert all keys except trace_hash.
    graph
        .map
        .insert("evidence_chain.substrate", MapValue::String(substrate.to_owned()));
    if let Some(p) = parent {
        graph
            .map
            .insert("evidence_chain.parent", MapValue::Bytes(p.to_vec()));
    }
    graph.map.insert(
        "evidence_chain.determinism",
        MapValue::String(determinism.as_str().to_owned()),
    );
    graph
        .map
        .insert("evidence_chain.toolchain", MapValue::String(toolchain.to_owned()));

    // Step 2-4: compute trace_hash via §3.2.
    let hash = compute_trace_hash(graph);

    // Step 5: insert trace_hash.
    graph
        .map
        .insert("evidence_chain.trace_hash", MapValue::Bytes(hash.to_vec()));
}

/// Remove all `evidence_chain.*` keys from `graph.map`.
///
/// Useful before re-attaching evidence (e.g., after a transformation step that
/// changes the graph).
pub fn remove_evidence_chain(graph: &mut Graph) {
    graph.map = strip_prefixes(&graph.map, &["evidence_chain.", "signature."]);
}

// ---------------------------------------------------------------------------
// §3.2 implementation detail
// ---------------------------------------------------------------------------

/// Compute the §3.2 `trace_hash`: SHA-256 of the MIC-B encoding of `graph`
/// with a MAP stripped of `signature.*` and `evidence_chain.trace_hash`.
///
/// The graph must already have all other `evidence_chain.*` keys populated.
///
/// # Panics
///
/// Panics if MIC-B serialization fails. Serialization here writes into an
/// in-memory `Vec<u8>`, which is infallible, so this cannot fire on a valid
/// graph — but the panic is documented because this is a `pub` entry point.
pub fn compute_trace_hash(graph: &Graph) -> [u8; 32] {
    let stripped_map = strip_keys_for_hash(&graph.map);
    let temp_graph = Graph {
        symbols: graph.symbols.clone(),
        types: graph.types.clone(),
        values: graph.values.clone(),
        output: graph.output,
        map: stripped_map,
    };
    let mut bytes: Vec<u8> = Vec::new();
    emit_micb(&temp_graph, &mut bytes).expect("emit_micb failed during trace_hash computation");
    trace_sha256(&bytes)
}

/// Build the MAP used for hashing: current MAP minus `signature.*` and
/// minus `evidence_chain.trace_hash`.  All other `evidence_chain.*` keys
/// (substrate, parent, determinism, toolchain) are retained — they are part
/// of what is attested.
fn strip_keys_for_hash(map: &Map) -> Map {
    let mut out = Map::new();
    for (k, v) in map.iter() {
        if k.starts_with("signature.") || k == "evidence_chain.trace_hash" {
            continue;
        }
        out.insert(k, v.clone());
    }
    out
}

/// Build a new MAP retaining only keys that do NOT start with any of `prefixes`.
fn strip_prefixes(map: &Map, prefixes: &[&str]) -> Map {
    let mut out = Map::new();
    for (k, v) in map.iter() {
        if prefixes.iter().any(|p| k.starts_with(p)) {
            continue;
        }
        out.insert(k, v.clone());
    }
    out
}

// ---------------------------------------------------------------------------
// §3.2 / §4 verification (Phase B core)
// ---------------------------------------------------------------------------

/// The decoded, validated contents of an artifact's `evidence_chain` block —
/// the reusable core a `mindc verify --evidence` CLI (RFC 0016 Phase B) wraps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReport {
    /// RFC 0014 substrate tier id (`evidence_chain.substrate`).
    pub substrate: String,
    /// Declared determinism (`evidence_chain.determinism`).
    pub determinism: Determinism,
    /// Toolchain version string (`evidence_chain.toolchain`).
    pub toolchain: String,
    /// Predecessor `trace_hash`, or `None` for a root artifact (§4).
    pub parent: Option<[u8; 32]>,
    /// The stored `trace_hash` exactly as carried in the MAP.
    pub trace_hash: [u8; 32],
    /// `true` iff a §3.2 recomputation reproduces `trace_hash` (i.e. the
    /// artifact has not been tampered with). The whole point of the chain.
    pub trace_hash_valid: bool,
}

impl EvidenceReport {
    /// A root artifact has no parent (§4).
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }
}

/// Why an `evidence_chain` block failed to decode for verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceError {
    /// No `evidence_chain.*` keys are present — the artifact is unattested.
    Missing,
    /// A required key is absent (substrate / determinism / toolchain / trace_hash).
    MissingKey(&'static str),
    /// A key has the wrong MAP value type, or a bytes field is not 32 bytes.
    Malformed(&'static str),
    /// `determinism` is neither `"deterministic"` nor `"nondeterministic"`.
    UnknownDeterminism(String),
}

fn map_get<'a>(map: &'a Map, key: &str) -> Option<&'a MapValue> {
    map.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

/// Verify the §3.2 self-reference of `graph`'s `evidence_chain` block and
/// decode it into an [`EvidenceReport`]. Recomputes `trace_hash` over the
/// canonical MIC-B bytes (MAP minus `signature.*` and the stored `trace_hash`)
/// and records whether it matches — a single flipped byte anywhere in the
/// attested graph flips `trace_hash_valid` to `false`.
///
/// This is the verifier core; the chain *walk* (following `parent` across
/// artifacts) and signature checks (§6) are layered on top by the CLI.
pub fn verify_evidence_chain(graph: &Graph) -> Result<EvidenceReport, EvidenceError> {
    // An artifact with no evidence_chain.* keys is unattested, not invalid.
    let has_any = graph
        .map
        .iter()
        .any(|(k, _)| k.starts_with("evidence_chain."));
    if !has_any {
        return Err(EvidenceError::Missing);
    }

    let substrate = match map_get(&graph.map, "evidence_chain.substrate") {
        Some(MapValue::String(s)) => s.clone(),
        Some(_) => return Err(EvidenceError::Malformed("evidence_chain.substrate")),
        None => return Err(EvidenceError::MissingKey("evidence_chain.substrate")),
    };

    let toolchain = match map_get(&graph.map, "evidence_chain.toolchain") {
        Some(MapValue::String(s)) => s.clone(),
        Some(_) => return Err(EvidenceError::Malformed("evidence_chain.toolchain")),
        None => return Err(EvidenceError::MissingKey("evidence_chain.toolchain")),
    };

    let determinism = match map_get(&graph.map, "evidence_chain.determinism") {
        Some(MapValue::String(s)) => match s.as_str() {
            "deterministic" => Determinism::Deterministic,
            "nondeterministic" => Determinism::Nondeterministic,
            other => return Err(EvidenceError::UnknownDeterminism(other.to_owned())),
        },
        Some(_) => return Err(EvidenceError::Malformed("evidence_chain.determinism")),
        None => return Err(EvidenceError::MissingKey("evidence_chain.determinism")),
    };

    // parent is optional (root has none).
    let parent = match map_get(&graph.map, "evidence_chain.parent") {
        Some(MapValue::Bytes(b)) => {
            let arr: [u8; 32] = b
                .clone()
                .try_into()
                .map_err(|_| EvidenceError::Malformed("evidence_chain.parent"))?;
            Some(arr)
        }
        Some(_) => return Err(EvidenceError::Malformed("evidence_chain.parent")),
        None => None,
    };

    let trace_hash: [u8; 32] = match map_get(&graph.map, "evidence_chain.trace_hash") {
        Some(MapValue::Bytes(b)) => b
            .clone()
            .try_into()
            .map_err(|_| EvidenceError::Malformed("evidence_chain.trace_hash"))?,
        Some(_) => return Err(EvidenceError::Malformed("evidence_chain.trace_hash")),
        None => return Err(EvidenceError::MissingKey("evidence_chain.trace_hash")),
    };

    // §3.2: recompute over the MAP-stripped canonical bytes and compare.
    let recomputed = compute_trace_hash(graph);
    let trace_hash_valid = recomputed == trace_hash;

    Ok(EvidenceReport {
        substrate,
        determinism,
        toolchain,
        parent,
        trace_hash,
        trace_hash_valid,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::v2::binary::{emit_micb, parse_micb};
    use crate::ir::compact::v2::emit::emit_mic2;
    use crate::ir::compact::v2::parse::parse_mic2;
    use crate::ir::compact::v2::types::GraphEq;
    use std::io::Cursor;

    fn base_graph() -> Graph {
        Graph::residual_block()
    }

    fn attach(
        g: &mut Graph,
        substrate: &str,
        parent: Option<[u8; 32]>,
        det: Determinism,
    ) {
        attach_evidence_chain(g, substrate, parent, det, "0.7.0");
    }

    // -----------------------------------------------------------------------
    // (a) Self-reference correctness: independently recompute trace_hash
    // -----------------------------------------------------------------------

    #[test]
    fn self_reference_hash_matches_independent_recomputation() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);

        // Extract the stored trace_hash.
        let stored = g
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| match v {
                MapValue::Bytes(b) => b.clone(),
                other => panic!("expected Bytes, got {:?}", other),
            })
            .expect("trace_hash missing");

        // Independently recompute by the §3.2 algorithm.
        let recomputed = compute_trace_hash(&g);

        assert_eq!(
            stored,
            recomputed.to_vec(),
            "stored trace_hash must equal §3.2 recomputation"
        );
        assert_eq!(stored.len(), 32, "trace_hash must be 32 bytes");
    }

    // -----------------------------------------------------------------------
    // (b) Determinism: same graph + same params → identical trace_hash
    // -----------------------------------------------------------------------

    #[test]
    fn same_graph_same_params_identical_trace_hash() {
        let mut g1 = base_graph();
        let mut g2 = base_graph();

        attach(&mut g1, "x86_avx2", None, Determinism::Deterministic);
        attach(&mut g2, "x86_avx2", None, Determinism::Deterministic);

        let h1 = g1
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| match v {
                MapValue::Bytes(b) => b.clone(),
                _ => panic!(),
            })
            .unwrap();
        let h2 = g2
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| match v {
                MapValue::Bytes(b) => b.clone(),
                _ => panic!(),
            })
            .unwrap();

        assert_eq!(h1, h2, "identical inputs must yield identical trace_hash");
    }

    // -----------------------------------------------------------------------
    // (c) Text and binary round-trip byte-identity
    // -----------------------------------------------------------------------

    #[test]
    fn evidence_text_roundtrip_byte_identical() {
        let mut g = base_graph();
        attach(&mut g, "arm_neon", None, Determinism::Deterministic);

        let text1 = emit_mic2(&g);
        let parsed = parse_mic2(&text1).expect("parse failed");
        let text2 = emit_mic2(&parsed);

        assert_eq!(text1, text2, "emit(parse(emit(G))) must equal emit(G) for text");
    }

    #[test]
    fn evidence_binary_roundtrip_byte_identical() {
        let mut g = base_graph();
        attach(&mut g, "arm_neon", None, Determinism::Deterministic);

        let mut bin1: Vec<u8> = Vec::new();
        emit_micb(&g, &mut bin1).expect("emit_micb failed");

        let parsed = parse_micb(&mut Cursor::new(&bin1)).expect("parse_micb failed");

        let mut bin2: Vec<u8> = Vec::new();
        emit_micb(&parsed, &mut bin2).expect("emit_micb failed");

        assert_eq!(
            bin1, bin2,
            "emit_micb(parse_micb(emit_micb(G))) must equal emit_micb(G)"
        );
    }

    // -----------------------------------------------------------------------
    // (d) Two artifacts, same graph, different substrate → different trace_hash
    // -----------------------------------------------------------------------

    #[test]
    fn different_substrate_different_trace_hash() {
        let mut g_avx2 = base_graph();
        let mut g_neon = base_graph();

        attach(&mut g_avx2, "x86_avx2", None, Determinism::Deterministic);
        attach(&mut g_neon, "arm_neon", None, Determinism::Deterministic);

        // Substrate field differs.
        let sub_avx2 = g_avx2
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.substrate")
            .map(|(_, v)| v.clone())
            .unwrap();
        let sub_neon = g_neon
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.substrate")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_ne!(sub_avx2, sub_neon, "substrate must differ");

        // trace_hash must differ (substrate is included in the hashed bytes).
        let hash_avx2 = g_avx2
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| v.clone())
            .unwrap();
        let hash_neon = g_neon
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_ne!(
            hash_avx2, hash_neon,
            "different substrate must yield different trace_hash (substrate is hashed, §3.2)"
        );
    }

    // -----------------------------------------------------------------------
    // (e) Nondeterministic: all required keys present, value is "nondeterministic"
    // -----------------------------------------------------------------------

    #[test]
    fn nondeterministic_graph_emits_correct_marker() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Nondeterministic);

        let det = g
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.determinism")
            .map(|(_, v)| v.clone())
            .expect("determinism key must be present");

        assert_eq!(
            det,
            MapValue::String("nondeterministic".to_owned()),
            "nondeterministic graph must emit determinism = \"nondeterministic\""
        );

        // All other required keys must also be present.
        let keys: Vec<&str> = g
            .map
            .iter()
            .filter(|(k, _)| k.starts_with("evidence_chain."))
            .map(|(k, _)| k)
            .collect();
        assert!(keys.contains(&"evidence_chain.substrate"));
        assert!(keys.contains(&"evidence_chain.toolchain"));
        assert!(keys.contains(&"evidence_chain.trace_hash"));
    }

    // -----------------------------------------------------------------------
    // (f) Parent present vs absent
    // -----------------------------------------------------------------------

    #[test]
    fn parent_absent_omits_parent_key() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);

        let has_parent = g
            .map
            .iter()
            .any(|(k, _)| k == "evidence_chain.parent");
        assert!(!has_parent, "absent parent must not produce evidence_chain.parent key");
    }

    #[test]
    fn parent_present_includes_parent_key_with_correct_bytes() {
        let parent_hash: [u8; 32] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
        ];

        let mut g = base_graph();
        attach(&mut g, "x86_avx2", Some(parent_hash), Determinism::Deterministic);

        let stored = g
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.parent")
            .map(|(_, v)| match v {
                MapValue::Bytes(b) => b.clone(),
                other => panic!("expected Bytes, got {:?}", other),
            })
            .expect("evidence_chain.parent must be present when parent supplied");

        assert_eq!(stored, parent_hash.to_vec(), "stored parent must match supplied hash");
        assert_eq!(stored.len(), 32);
    }

    #[test]
    fn parent_present_differs_from_parent_absent_in_trace_hash() {
        let parent_hash = [0xabu8; 32];

        let mut g_no_parent = base_graph();
        let mut g_with_parent = base_graph();

        attach(&mut g_no_parent, "x86_avx2", None, Determinism::Deterministic);
        attach(&mut g_with_parent, "x86_avx2", Some(parent_hash), Determinism::Deterministic);

        let h_no = g_no_parent
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| v.clone())
            .unwrap();
        let h_with = g_with_parent
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| v.clone())
            .unwrap();

        assert_ne!(h_no, h_with, "presence of parent must change trace_hash");
    }

    // -----------------------------------------------------------------------
    // Default emit path unchanged: unannotated graph emits byte-identically
    // -----------------------------------------------------------------------

    #[test]
    fn unannotated_graph_text_unchanged() {
        let g = base_graph();
        let text = emit_mic2(&g);
        // Must not contain evidence_chain keys.
        assert!(
            !text.contains("evidence_chain"),
            "unannotated graph must not contain evidence_chain in text output"
        );
        // Must end with canonical output line, no MAP block.
        assert!(text.ends_with("O 6"), "canonical residual block must end with 'O 6'");
    }

    #[test]
    fn unannotated_graph_binary_unchanged() {
        let g = base_graph();
        let mut bin: Vec<u8> = Vec::new();
        emit_micb(&g, &mut bin).expect("emit_micb failed");
        // Binary must not contain MAP marker 0x4D after position 4 in a
        // way that signals a MAP section.  We verify by re-parsing: it must
        // have an empty map.
        let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse_micb failed");
        assert!(
            parsed.map.is_empty(),
            "unannotated graph binary must parse as empty MAP"
        );
    }

    // -----------------------------------------------------------------------
    // Graph structural equality includes MAP (regression guard)
    // -----------------------------------------------------------------------

    #[test]
    fn graph_eq_distinguishes_evidence_presence() {
        let g_bare = base_graph();
        let mut g_with = base_graph();
        attach(&mut g_with, "x86_avx2", None, Determinism::Deterministic);

        assert!(
            !g_bare.eq(&g_with),
            "graph with evidence must not equal graph without evidence"
        );
    }

    // -----------------------------------------------------------------------
    // Remove helper clears evidence keys
    // -----------------------------------------------------------------------

    #[test]
    fn remove_evidence_chain_clears_all_ec_keys() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);

        remove_evidence_chain(&mut g);

        let has_ec = g.map.iter().any(|(k, _)| k.starts_with("evidence_chain."));
        assert!(!has_ec, "remove_evidence_chain must clear all evidence_chain.* keys");
        assert!(g.map.is_empty(), "map must be empty after removing only evidence keys");
    }

    /// Double-attach without an intervening `remove_evidence_chain` must trip
    /// the guard (the §3.2 self-reference would otherwise hash a graph that
    /// already carries a stale `trace_hash` from the prior attach).
    #[test]
    #[should_panic(expected = "already present")]
    fn double_attach_without_remove_panics() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);
        // Second attach on the same graph — must panic, not silently re-hash.
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);
    }

    /// Helper: read the stored `trace_hash` from a graph's evidence block.
    fn stored_trace_hash(g: &Graph) -> [u8; 32] {
        let v = g
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.trace_hash")
            .map(|(_, v)| v.clone())
            .expect("trace_hash missing");
        match v {
            MapValue::Bytes(b) => b.try_into().expect("trace_hash must be 32 bytes"),
            other => panic!("expected Bytes, got {other:?}"),
        }
    }

    /// §4 Merkle-DAG: a chained artifact (parse → IR) carries the prior link's
    /// `trace_hash` as its `parent`, and the whole chain survives a MIC-B
    /// round-trip — the load-bearing claim a `mindc verify` chain walk relies
    /// on (Phase B). This exercises the chain end-to-end through serialization,
    /// not just a single link in memory.
    #[test]
    fn evidence_chain_links_survive_micb_roundtrip() {
        // Link 1 — root (no parent). A distinct graph stands in for the
        // "source-parse" stage artifact.
        let mut g1 = base_graph();
        attach(&mut g1, "x86_avx2", None, Determinism::Deterministic);
        let h1 = stored_trace_hash(&g1);

        // Link 2 — the "IR" stage. A different graph, with parent = h1.
        let mut g2 = Graph::residual_block();
        // Perturb g2's MAP so it is a genuinely different artifact from g1
        // (otherwise the only difference would be the parent key).
        g2.map.insert("stage", MapValue::String("ir".to_owned()));
        attach(&mut g2, "x86_avx2", Some(h1), Determinism::Deterministic);
        let h2 = stored_trace_hash(&g2);
        assert_ne!(h1, h2, "distinct artifacts must have distinct trace_hash");

        // Round-trip link 2 through MIC-B and re-parse.
        let mut bin: Vec<u8> = Vec::new();
        emit_micb(&g2, &mut bin).expect("emit_micb failed");
        let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse_micb failed");

        // (a) The parent pointer survived and equals link 1's trace_hash.
        let parent = parsed
            .map
            .iter()
            .find(|(k, _)| *k == "evidence_chain.parent")
            .map(|(_, v)| match v {
                MapValue::Bytes(b) => b.clone(),
                other => panic!("expected Bytes, got {other:?}"),
            })
            .expect("parent missing after round-trip");
        assert_eq!(parent, h1.to_vec(), "round-tripped parent must equal link 1 trace_hash");

        // (b) A verifier recomputing §3.2 over the parsed graph reproduces the
        //     stored trace_hash — the chain link is independently checkable.
        assert_eq!(
            compute_trace_hash(&parsed).to_vec(),
            h2.to_vec(),
            "recomputed trace_hash of the round-tripped link must equal the stored one"
        );
    }

    // -----------------------------------------------------------------------
    // verify_evidence_chain (Phase B core)
    // -----------------------------------------------------------------------

    #[test]
    fn verify_accepts_a_valid_attested_graph() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);

        let report = verify_evidence_chain(&g).expect("valid evidence must verify");
        assert!(report.trace_hash_valid, "untampered graph must verify");
        assert_eq!(report.substrate, "x86_avx2");
        assert_eq!(report.determinism, Determinism::Deterministic);
        assert_eq!(report.toolchain, "0.7.0");
        assert!(report.is_root(), "no parent ⇒ root artifact");
        assert_eq!(report.trace_hash, compute_trace_hash(&g));
    }

    #[test]
    fn verify_detects_tampering() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);
        // Mutate the attested graph AFTER the hash was sealed, without
        // re-attaching. The §3.2 recompute now covers the new key, so the
        // stored trace_hash no longer matches.
        g.map.insert("stage", MapValue::String("tampered".to_owned()));

        let report = verify_evidence_chain(&g).expect("decodes, but must report invalid");
        assert!(
            !report.trace_hash_valid,
            "post-seal mutation must flip trace_hash_valid to false"
        );
    }

    #[test]
    fn verify_reports_missing_evidence() {
        let g = base_graph(); // no evidence attached
        assert_eq!(verify_evidence_chain(&g), Err(EvidenceError::Missing));
    }

    #[test]
    fn verify_reports_parent_and_nonroot() {
        let parent_hash = [0x7eu8; 32];
        let mut g = base_graph();
        attach(&mut g, "arm_neon", Some(parent_hash), Determinism::Nondeterministic);

        let report = verify_evidence_chain(&g).expect("valid evidence must verify");
        assert!(report.trace_hash_valid);
        assert_eq!(report.parent, Some(parent_hash));
        assert!(!report.is_root(), "an artifact with a parent is not a root");
        assert_eq!(report.determinism, Determinism::Nondeterministic);
    }

    #[test]
    fn verify_survives_micb_roundtrip() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);

        let mut bin: Vec<u8> = Vec::new();
        emit_micb(&g, &mut bin).expect("emit_micb failed");
        let parsed = parse_micb(&mut Cursor::new(&bin)).expect("parse_micb failed");

        let report = verify_evidence_chain(&parsed).expect("round-tripped evidence must verify");
        assert!(report.trace_hash_valid, "evidence must verify after a MIC-B round-trip");
        assert_eq!(report.substrate, "x86_avx2");
    }

    // -----------------------------------------------------------------------
    // (g) FIPS 180-4 conformance — the Rust↔MIND SHA-256 duality is frozen here.
    //
    // `trace_hash` is SHA-256 of canonical mic@2.1 binary bytes (§3.2). The
    // Rust bootstrap hashes those bytes with `crate::deps::mini_sha256`; the
    // self-host endpoint (Phase A.5) will hash the *same* bytes with the pure-
    // MIND `std.sha256`. Both are FIPS 180-4, so they MUST agree byte-for-byte.
    //
    // These vectors are the frozen conformance target: if a future MIND emitter
    // produces a different digest for "abc" or the empty string, it is not
    // FIPS-conformant and the duality (hence the whole substrate-portable
    // trace_hash claim) is broken. The constants are the canonical FIPS 180-4
    // appendix test vectors — they never change.
    // -----------------------------------------------------------------------

    #[test]
    fn trace_sha256_matches_fips_180_4_vectors() {
        // FIPS 180-4 §B.1: SHA-256("abc")
        let abc = trace_sha256(b"abc");
        assert_eq!(
            abc,
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
            ],
            "SHA-256(\"abc\") must equal the FIPS 180-4 vector — the frozen \
             conformance target the pure-MIND std.sha256 endpoint must also hit"
        );

        // FIPS 180-4: SHA-256("") (empty message)
        let empty = trace_sha256(b"");
        assert_eq!(
            empty,
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
            ],
            "SHA-256(\"\") must equal the FIPS 180-4 empty-string vector"
        );

        // FIPS 180-4 §B.2: the two-block (56-byte) vector exercises padding
        // that spills into a second compression block.
        let two_block = trace_sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        );
        assert_eq!(
            two_block,
            [
                0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8,
                0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
                0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67,
                0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
            ],
            "SHA-256 of the FIPS 180-4 two-block vector must match"
        );
    }

    /// Pin the trace_hash of the canonical residual-block graph to a recorded
    /// constant. This is the cross-version regression guard: the §3.2 hash of a
    /// fixed graph must never drift silently (a change here means either the
    /// mic@2.1 binary encoding changed, the strip rules changed, or the hash
    /// changed — all of which need an explicit, reviewed decision).
    #[test]
    fn residual_block_trace_hash_is_pinned() {
        let mut g = base_graph();
        attach(&mut g, "x86_avx2", None, Determinism::Deterministic);
        let got = compute_trace_hash(&g);
        // Recorded on first green run; see the FIPS conformance test above for
        // why this is stable across Rust and the future MIND emitter.
        let expected = EXPECTED_RESIDUAL_TRACE_HASH;
        assert_eq!(
            got, expected,
            "residual-block trace_hash drifted from the pinned constant; if the \
             mic@2.1 encoding or §3.2 strip rules changed intentionally, update \
             EXPECTED_RESIDUAL_TRACE_HASH with the new value and note why"
        );
    }

    /// The pinned §3.2 trace_hash of `Graph::residual_block()` with substrate
    /// `x86_avx2`, no parent, deterministic, toolchain `0.7.0`.
    const EXPECTED_RESIDUAL_TRACE_HASH: [u8; 32] = [
        0x5a, 0x62, 0x9d, 0x55, 0x9b, 0x25, 0x17, 0xe0,
        0x44, 0xd7, 0x21, 0x61, 0x41, 0x29, 0x0f, 0x57,
        0xb9, 0x1b, 0x5d, 0x9b, 0x4c, 0xf5, 0x18, 0x85,
        0x31, 0xee, 0x8e, 0x6f, 0x1d, 0x72, 0x29, 0x82,
    ];
}
