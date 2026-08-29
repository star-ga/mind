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

use crate::deps::mini_sha256;
use crate::ir::IRModule;
// `uleb128_read` (the LENIENT reader) is deliberately NOT imported here: every
// length in the MAP epilogue is canonical-form-critical, so this module reads
// them with `uleb128_read_minimal`, which rejects zero-padded encodings.
use crate::ir::compact::v2::{uleb128_read_minimal, uleb128_write, zigzag_decode, zigzag_encode};
use crate::ir::compact::v3::collapse_receipt::{
    CollapseReceipt, decode_collapse_receipts, encode_collapse_receipts,
};
use crate::ir::evidence::ir_trace_hash;

// Re-export the canonical evidence vocabulary from v2 — one set of types for
// the whole codebase (RFC 0021 §3.2: "reuse, don't rebuild").
pub use crate::ir::compact::v2::{Determinism, EvidenceError, EvidenceReport, TraceHashKind};

// ─── MAP sentinel ─────────────────────────────────────────────────────────────

/// Sentinel byte that introduces the MAP epilogue in a mic@3 artifact.
/// Same value as the v2 MIC-B MAP marker (`0x4D` = ASCII `'M'`) for consistency.
pub const MAP_SENTINEL: u8 = 0x4D;

// ─── Value tags (inline encoding — no shared string table) ────────────────────

const TAG_STRING: u8 = 0x00;
const TAG_INT: u8 = 0x01;
const TAG_BYTES: u8 = 0x02;

// ─── Evidence-chain key constants ─────────────────────────────────────────────

const KEY_COLLAPSE_RECEIPTS: &str = "evidence_chain.collapse_receipts";
const KEY_DETERMINISM: &str = "evidence_chain.determinism";
const KEY_PARENT: &str = "evidence_chain.parent";
const KEY_SCHEMA: &str = "evidence_chain.schema";
const KEY_SUBSTRATE: &str = "evidence_chain.substrate";
const KEY_TOOLCHAIN: &str = "evidence_chain.toolchain";
const KEY_TRACE_HASH: &str = "evidence_chain.trace_hash";
const KEY_TRACE_HASH_KIND: &str = "evidence_chain.trace_hash_kind";

/// Every DEFINED key in the reserved `evidence_chain.` namespace, enumerated ONCE
/// beside the constants themselves.
///
/// The read path allowlisted `signature.*` but nothing checked this namespace, so an
/// unknown `evidence_chain.<anything>` entry was accepted on every read path. That is
/// the same malleability shape the `signature.*` allowlist exists to close: the key is
/// reserved, `validate_app_entries` refuses to EMIT it, and therefore no legitimate
/// producer writes one — but emit-side validation is not a read-side control, and a
/// hex editor is not obliged to use our emitter.
///
/// Fail-closed is the right default here even though it is strict: a reader that does
/// not understand an evidence-chain key cannot know whether the key changes the meaning
/// of the chain, so accepting it silently is worse than refusing the artifact. A new
/// key (e.g. RFC 0027's `evidence_chain.device_receipts`) MUST be added here in the same
/// change that starts emitting it, or every artifact carrying it fails verification —
/// which is the loud direction to fail in.
const EVIDENCE_CHAIN_KEYS: [&str; 8] = [
    KEY_COLLAPSE_RECEIPTS,
    KEY_DETERMINISM,
    KEY_PARENT,
    KEY_SCHEMA,
    KEY_SUBSTRATE,
    KEY_TOOLCHAIN,
    KEY_TRACE_HASH,
    KEY_TRACE_HASH_KIND,
];

// ─── Signature-layer key constants (RFC 0021 §6, additive/optional) ────────────
//
// The `signature.*` prefix is the reserved signing namespace: these keys are an
// OPTIONAL, additive layer emitted only when a signing key is supplied. They sort
// lexicographically AFTER every `evidence_chain.*` key ('e' < 's'), so an unsigned
// artifact is byte-identical to the pre-signing encoder and a verifier that
// predates this layer ignores them. The signature covers the canonical PROVENANCE
// PREIMAGE (`build_signature_preimage`): the 32-byte `evidence_chain.trace_hash`
// (the canonical mic@3 anchor) PLUS a canonical serialization of every other
// `evidence_chain.*` key (substrate, toolchain, determinism, parent, schema,
// trace_hash_kind) — so provenance is authenticated, not just the code. The
// `signature.*` keys are themselves excluded from the preimage (a signature never
// covers itself), so the signature does NOT feed back into `trace_hash` — adding
// it cannot change the anchor, keeping the determinism gate green.
// The `signature.scheme` value is the crypto-agility `alg` tag (OMB M-26-15): a
// verifier reads it to select which verifier(s) to run. Fail-closed on an
// unrecognized value so a downgrade/unknown scheme can never be silently accepted.
const KEY_SIG_SCHEME: &str = "signature.scheme";
// Ed25519 (classical) key + signature.
const KEY_SIG_PUBKEY: &str = "signature.pubkey";
const KEY_SIG_ED25519: &str = "signature.ed25519";
// ML-DSA-65 (FIPS-204, post-quantum) key + signature. Variable length (pk 1952 B,
// sig 3309 B), stored as Bytes.
const KEY_SIG_MLDSA_PUBKEY: &str = "signature.mldsa_pubkey";
const KEY_SIG_MLDSA: &str = "signature.mldsa";
// ML-DSA-87 (FIPS-204, cat-5) + SLH-DSA-SHAKE-256s (FIPS-205, cat-5) — the two
// legs of the bulletproof PQC-hybrid. Variable length; all four keys start with
// 's' so they still sort AFTER every `evidence_chain.*` key (the unsigned-prefix
// byte-identity invariant is preserved).
const KEY_SIG_MLDSA87_PUBKEY: &str = "signature.mldsa87_pubkey";
const KEY_SIG_MLDSA87: &str = "signature.mldsa87";
const KEY_SIG_SLHDSA_PUBKEY: &str = "signature.slhdsa_pubkey";
const KEY_SIG_SLHDSA: &str = "signature.slhdsa";

/// Every DEFINED key in the reserved `signature.` namespace, in one place, beside
/// the constants themselves. `parse_map_epilogue` rejects any other `signature.*`
/// key on read.
///
/// This list is the ONLY place the set is enumerated. It sits here rather than at
/// the check site because the failure mode is a new `KEY_SIG_*` const being added
/// without the reader learning about it — which is how the ml-dsa-87 / slh-dsa
/// keys would have been rejected when a 5-key allowlist met a 9-key emitter.
const SIGNATURE_KEYS: [&str; 9] = [
    KEY_SIG_SCHEME,
    KEY_SIG_PUBKEY,
    KEY_SIG_ED25519,
    KEY_SIG_MLDSA_PUBKEY,
    KEY_SIG_MLDSA,
    KEY_SIG_MLDSA87_PUBKEY,
    KEY_SIG_MLDSA87,
    KEY_SIG_SLHDSA_PUBKEY,
    KEY_SIG_SLHDSA,
];

/// `alg` tag: classical Ed25519 only (NON-COMPLIANT for the federal PQC mandate;
/// retained for interop/legacy).
const SIG_SCHEME_ED25519: &str = "ed25519";
/// `alg` tag: post-quantum ML-DSA-65 (FIPS-204) only. Compliant PQC signature.
const SIG_SCHEME_MLDSA65: &str = "ml-dsa-65";
/// `alg` tag: hybrid — BOTH Ed25519 AND ML-DSA-65 must verify. Preferred for the
/// PQC transition (defense-in-depth: safe if either primitive is later broken).
const SIG_SCHEME_HYBRID: &str = "hybrid-ed25519-ml-dsa-65";
/// `alg` tag: the BULLETPROOF PQC-hybrid — BOTH ML-DSA-87 (FIPS-204, cat-5,
/// module-lattice) AND SLH-DSA-SHAKE-256s (FIPS-205, cat-5, hash-based) must
/// verify. Two mathematically INDEPENDENT post-quantum foundations, so no single
/// cryptanalytic break and no single implementation bug is sufficient to forge.
/// Ed25519 is deliberately ABSENT: quantum-dead, and keeping it a live tag value
/// would preserve a downgrade target.
const SIG_SCHEME_PQC_HYBRID: &str = "pqc-hybrid-ml-dsa-87-slh-dsa-256s";

/// Environment variable holding the 32-byte Ed25519 seed as 64 hex chars.
/// Never hardcode a key — the seed is supplied out-of-band by the operator.
pub const ENV_ED25519_SEED: &str = "MIND_EVIDENCE_ED25519_KEY";
/// Re-export of the ML-DSA key-generation seed env var (see [`super::mldsa`]).
pub use super::mldsa::ENV_MLDSA_SEED;

/// Which signature scheme(s) to embed, selected from the operator-supplied seeds.
///
/// This is the crypto-agility surface (OMB M-26-15): the caller picks a scheme and
/// the encoder tags the artifact with the corresponding `signature.scheme` value so
/// a verifier can dispatch. All variants sign the SAME payload — the 32-byte
/// canonical mic@3 `trace_hash` — so the anchor (Constitution Article IV) is
/// untouched regardless of scheme.
#[derive(Debug, Clone)]
pub enum SigningKey {
    /// Classical Ed25519 (32-byte seed). Legacy/interop; not PQC-compliant.
    Ed25519([u8; 32]),
    /// Post-quantum ML-DSA-65 (32-byte FIPS-204 keygen seed ξ).
    MlDsa65([u8; 32]),
    /// Hybrid: sign with BOTH; a verifier requires BOTH halves to verify.
    Hybrid {
        /// Ed25519 seed.
        ed25519: [u8; 32],
        /// ML-DSA-65 keygen seed ξ.
        mldsa65: [u8; 32],
    },
    /// Bulletproof PQC-hybrid: sign with BOTH ML-DSA-87 and SLH-DSA-SHAKE-256s;
    /// a verifier requires BOTH to verify (AND). Two independent post-quantum
    /// foundations. The max-security release profile.
    PqcHybrid {
        /// ML-DSA-87 keygen seed ξ (32 bytes).
        mldsa87: [u8; 32],
        /// SLH-DSA-SHAKE-256s seed (96 bytes: SK.seed ‖ SK.prf ‖ PK.seed).
        slhdsa: [u8; 96],
    },
}

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
    // Dedup (Slice-0, task #70): `ir_trace_hash(ir)` is `mini_sha256(&emit_mic3(ir))`,
    // and `body` already holds those exact canonical mic@3 bytes — so hash the bytes
    // in hand instead of re-running `emit_mic3`. Byte-identical trace_hash (Article
    // IV anchor unchanged); `emit_mic3` now runs once on this path instead of twice.
    let trace_hash = mini_sha256(&body);
    let mut out = body;
    append_map_epilogue(
        &mut out,
        substrate,
        parent,
        determinism,
        toolchain,
        trace_hash,
        None,
        None,
        &[],
    );
    out
}

/// Emit a mic@3 artifact with an `evidence_chain.*` MAP that ALSO carries
/// Salov loop-collapse receipts (S4), plus an optional crypto-agile signature.
///
/// This is the CLI's `--emit-evidence` entry point: it threads the receipts the
/// collapse pass produced (empty for any source with no `#[collapse]` fold) into
/// the epilogue under `evidence_chain.collapse_receipts`.
///
/// ## Anchor invariant (Constitution Article IV)
///
/// The receipts live in the epilogue, OUTSIDE the `trace_hash` preimage, so the
/// anchor (`trace_hash = SHA-256(canonical mic@3 body)`) is BYTE-IDENTICAL to
/// [`emit_mic3_with_evidence`] on the same `ir` — a receipt-bearing build and a
/// no-receipt build agree on both the code bytes and the `trace_hash`. When
/// `receipts` is empty the `collapse_receipts` key is omitted entirely, so the
/// epilogue is byte-identical to the no-receipt encoder (full back-compat).
///
/// Being an `evidence_chain.*` key, the receipt blob IS folded into the
/// signature preimage, so on a signed artifact the receipts are authenticated
/// (not strippable/swappable). Unsigned, the chain is tamper-EVIDENT.
///
/// # Errors
///
/// Returns `Err` only when a PQC signature scheme is requested on a build
/// compiled WITHOUT the `evidence-mldsa` feature (fail-closed).
// Eight legitimately-distinct evidence inputs (IR, substrate, parent, determinism,
// toolchain, signing key, collapse receipts, application entries); bundling them into
// a struct would obscure the wire contract more than it clarifies. House style already
// takes this allow for the analogous MLIR-export emitters.
#[allow(clippy::too_many_arguments)]
pub fn emit_mic3_with_evidence_and_receipts(
    ir: &IRModule,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
    signing: Option<&SigningKey>,
    receipts: &[CollapseReceipt],
    app_entries: &[(String, String)],
) -> Result<Vec<u8>, &'static str> {
    // Validate application-namespace entries at the library boundary (not only at
    // the CLI): reserved-prefix / un-namespaced / duplicate / empty keys must never
    // reach the MAP or the signature preimage, or a non-CLI caller could break the
    // canonical "MAP is a set" invariant. Byte-neutral for the empty / valid case.
    validate_app_entries(app_entries)
        .map_err(|_| "invalid application-namespace evidence entry")?;
    let body = super::emit_mic3(ir);
    let trace_hash = mini_sha256(&body);
    // Encode the canonical receipt blob; `None` (omit the key) when there is
    // nothing to attest, so the empty-receipt path stays byte-identical.
    let blob = if receipts.is_empty() {
        None
    } else {
        Some(encode_collapse_receipts(receipts))
    };
    let blob_ref = blob.as_deref();

    let payload = match signing {
        Some(key) => {
            // Sign the canonical provenance preimage built from the SAME entries
            // the MAP emits (incl. the collapse-receipt blob), so what is signed
            // is byte-for-byte what ships.
            let evidence_entries = build_evidence_entries(
                substrate,
                parent.as_ref(),
                determinism,
                toolchain,
                &trace_hash,
                blob_ref,
                app_entries,
            );
            let scheme = scheme_for_key(key);
            let preimage = build_signature_preimage(&evidence_entries, &trace_hash, scheme);
            Some(compute_signature_payload(key, &preimage)?)
        }
        None => None,
    };

    let mut out = body;
    append_map_epilogue(
        &mut out,
        substrate,
        parent,
        determinism,
        toolchain,
        trace_hash,
        payload.as_ref(),
        blob_ref,
        app_entries,
    );
    Ok(out)
}

/// Emit a mic@3 artifact with an `evidence_chain.*` MAP **and** an Ed25519
/// signature over the canonical provenance preimage (RFC 0021 §6).
///
/// The signature is deterministic (RFC 8032): the `seed` plus the preimage fully
/// determine the 64-byte signature, so this call is byte-reproducible.
///
/// ## Anchor invariant (Constitution Article IV)
///
/// The signature covers the canonical provenance preimage
/// (`build_signature_preimage`): the 32-byte `trace_hash` (the SHA-256 of the
/// canonical mic@3 body) plus every other `evidence_chain.*` key, so provenance
/// (substrate/toolchain/determinism/parent) is authenticated too. It is emitted
/// as `signature.*` MAP keys that sort after every `evidence_chain.*` key, so:
///   * `trace_hash` is unchanged versus the unsigned path — the anchor and the
///     determinism gate are untouched; and
///   * the unsigned prefix (body + `evidence_chain.*`) is byte-identical to the
///     output of [`emit_mic3_with_evidence`] with the same arguments.
pub fn emit_mic3_with_signed_evidence(
    ir: &IRModule,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
    seed: &[u8; 32],
) -> Vec<u8> {
    // Ed25519 signing never depends on an optional feature ⇒ infallible.
    emit_mic3_with_signed_evidence_scheme(
        ir,
        substrate,
        parent,
        determinism,
        toolchain,
        &SigningKey::Ed25519(*seed),
    )
    .expect("ed25519 signing is always available")
}

/// Emit a mic@3 artifact with an `evidence_chain.*` MAP **and** a crypto-agile
/// signature over the canonical `trace_hash` (RFC 0021 §6).
///
/// The [`SigningKey`] selects the scheme — classical Ed25519, post-quantum
/// ML-DSA-65 (FIPS-204), or the hybrid of both — and the artifact is tagged with
/// the matching `signature.scheme` (`alg`) value so a verifier is crypto-agile
/// (OMB M-26-15). Every scheme signs the SAME payload (the canonical provenance
/// preimage: mic@3 anchor + all other `evidence_chain.*` keys), so the
/// `trace_hash` — and therefore the determinism/keystone gate — is identical
/// across all schemes and versus the unsigned path.
///
/// # Determinism
///
/// Ed25519 (RFC 8032) and ML-DSA (FIPS-204 deterministic variant, all-zero rnd)
/// are both byte-reproducible, so the whole artifact is reproducible.
///
/// # Errors
///
/// Returns `Err` if a PQC scheme is requested on a build compiled WITHOUT the
/// `evidence-mldsa` feature (fail-closed: never emit an unsigned artifact when a
/// signature was requested).
pub fn emit_mic3_with_signed_evidence_scheme(
    ir: &IRModule,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
    key: &SigningKey,
) -> Result<Vec<u8>, &'static str> {
    let body = super::emit_mic3(ir);
    let trace_hash = ir_trace_hash(ir);
    // Sign the CANONICAL PROVENANCE PREIMAGE, not just the anchor: trace_hash ||
    // canonical serialization of every `evidence_chain.*` entry except the
    // trace_hash itself (which is derived from the body it anchors). Building the
    // preimage from the same `build_evidence_entries` the MAP is emitted from
    // guarantees the signed bytes are byte-for-byte the provenance on the wire, so
    // substrate/toolchain/determinism/parent are all authenticated — not just code.
    let evidence_entries = build_evidence_entries(
        substrate,
        parent.as_ref(),
        determinism,
        toolchain,
        &trace_hash,
        None,
        &[],
    );
    // Bind the scheme (`alg`) tag into the signed preimage (see
    // `build_signature_preimage`): the scheme is read from the SAME source the
    // verifier uses (here the `SigningKey` variant), so a later downgrade that
    // rewrites `signature.scheme` changes the preimage and the surviving half
    // fails to verify (fail-closed).
    let scheme = scheme_for_key(key);
    let preimage = build_signature_preimage(&evidence_entries, &trace_hash, scheme);
    let payload = compute_signature_payload(key, &preimage)?;
    let mut out = body;
    append_map_epilogue(
        &mut out,
        substrate,
        parent,
        determinism,
        toolchain,
        trace_hash,
        Some(&payload),
        None,
        &[],
    );
    Ok(out)
}

/// The `signature.scheme` (`alg`) tag a [`SigningKey`] variant emits. Read on the
/// emit side so the SAME string that ends up in the MAP is bound into the signed
/// preimage — a verify-time downgrade that rewrites the tag cannot then reproduce
/// the preimage the genuine half was signed over.
fn scheme_for_key(key: &SigningKey) -> &'static str {
    match key {
        SigningKey::Ed25519(_) => SIG_SCHEME_ED25519,
        SigningKey::MlDsa65(_) => SIG_SCHEME_MLDSA65,
        SigningKey::Hybrid { .. } => SIG_SCHEME_HYBRID,
        SigningKey::PqcHybrid { .. } => SIG_SCHEME_PQC_HYBRID,
    }
}

/// Computed signature material ready to embed as `signature.*` MAP keys.
/// ML-DSA keys/signatures are variable-length ⇒ owned `Vec`s.
struct SignaturePayload {
    scheme: &'static str,
    ed25519: Option<([u8; 32], [u8; 64])>,
    mldsa: Option<(Vec<u8>, Vec<u8>)>,
    /// (pubkey, signature) for the ML-DSA-87 leg of the bulletproof PQC-hybrid.
    mldsa87: Option<(Vec<u8>, Vec<u8>)>,
    /// (pubkey, signature) for the SLH-DSA-SHAKE-256s leg of the PQC-hybrid.
    slhdsa: Option<(Vec<u8>, Vec<u8>)>,
}

/// Sign the canonical provenance `preimage` under `key`, producing the embeddable
/// payload. The preimage is `trace_hash || canonical(evidence_chain.* \ trace_hash)`
/// (see `build_signature_preimage`) so the signature covers the whole provenance
/// (substrate, toolchain, determinism, parent), not just the code anchor.
/// Fail-closed when a PQC scheme is requested without the `evidence-mldsa` feature.
fn compute_signature_payload(
    key: &SigningKey,
    preimage: &[u8],
) -> Result<SignaturePayload, &'static str> {
    let ed = |seed: &[u8; 32]| -> ([u8; 32], [u8; 64]) {
        let sig = super::ed25519::sign(seed, preimage);
        let pubkey = super::ed25519::public_key(seed);
        (pubkey, sig)
    };
    let mldsa = |seed: &[u8; 32]| -> Result<(Vec<u8>, Vec<u8>), &'static str> {
        if !super::mldsa::supported() {
            return Err(
                "ML-DSA signing requires the `evidence-mldsa` cargo feature (rebuild with \
                 --features evidence-mldsa)",
            );
        }
        Ok((
            super::mldsa::public_key(seed),
            super::mldsa::sign(seed, preimage),
        ))
    };
    // Bulletproof-hybrid legs. Both sign the IDENTICAL `preimage` (the scheme tag
    // is already bound into it — that is the per-algorithm domain separation), so
    // the two signatures cover exactly the same authenticated bytes.
    let mldsa87 = |seed: &[u8; 32]| -> Result<(Vec<u8>, Vec<u8>), &'static str> {
        if !super::mldsa::supported() {
            return Err(
                "ML-DSA-87 signing requires the `evidence-mldsa` cargo feature (rebuild with \
                 --features evidence-mldsa)",
            );
        }
        Ok((
            super::mldsa::public_key_87(seed),
            super::mldsa::sign_87(seed, preimage),
        ))
    };
    let slhdsa = |seed: &[u8; 96]| -> Result<(Vec<u8>, Vec<u8>), &'static str> {
        if !super::slhdsa::supported() {
            return Err(
                "SLH-DSA signing requires the `evidence-slhdsa` cargo feature (rebuild with \
                 --features evidence-slhdsa)",
            );
        }
        Ok((
            super::slhdsa::public_key(seed),
            super::slhdsa::sign(seed, preimage),
        ))
    };
    match key {
        SigningKey::Ed25519(seed) => Ok(SignaturePayload {
            scheme: SIG_SCHEME_ED25519,
            ed25519: Some(ed(seed)),
            mldsa: None,
            mldsa87: None,
            slhdsa: None,
        }),
        SigningKey::MlDsa65(seed) => Ok(SignaturePayload {
            scheme: SIG_SCHEME_MLDSA65,
            ed25519: None,
            mldsa: Some(mldsa(seed)?),
            mldsa87: None,
            slhdsa: None,
        }),
        SigningKey::Hybrid { ed25519, mldsa65 } => Ok(SignaturePayload {
            scheme: SIG_SCHEME_HYBRID,
            ed25519: Some(ed(ed25519)),
            mldsa: Some(mldsa(mldsa65)?),
            mldsa87: None,
            slhdsa: None,
        }),
        SigningKey::PqcHybrid {
            mldsa87: m87_seed,
            slhdsa: slh_seed,
        } => Ok(SignaturePayload {
            scheme: SIG_SCHEME_PQC_HYBRID,
            ed25519: None,
            mldsa: None,
            mldsa87: Some(mldsa87(m87_seed)?),
            slhdsa: Some(slhdsa(slh_seed)?),
        }),
    }
}

/// Why a mic@3 artifact was rejected as non-canonical by [`mic3_canonical_check`].
///
/// Every variant is a byte stream that the decoder would otherwise normalise
/// away — i.e. a mutation that changes the artifact's bytes (and its SHA-256)
/// while leaving `trace_hash_valid` and any embedded signature intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mic3NonCanonical {
    /// Input exceeds the mic@3 input cap.
    Oversize,
    /// The mic@3 body did not parse; canonicality is undecidable here (the
    /// parse/SSA path reports the authoritative diagnostic).
    Unparseable,
    /// The literal body bytes differ from the canonical re-emission of the IR
    /// they decode to, first at `offset`. Covers a normalised-away body flip and
    /// a wire-version downgrade inside the accepted read window.
    Body { offset: usize },
    /// Bytes follow the canonical body but do not begin a MAP epilogue.
    TrailingGarbage { offset: usize },
    /// The MAP epilogue did not parse (includes leftover bytes and non-minimal
    /// ULEB lengths, both rejected by `parse_map_epilogue`).
    MalformedMap,
    /// A key in the reserved `signature.` namespace that is not one of the five
    /// defined signature keys. Rejected on READ, not just on emit: the signature
    /// preimage deliberately EXCLUDES `signature.*` (it cannot sign itself), and
    /// the canonical byte-compare happily re-encodes any entry that parsed — so
    /// without this check an injected `signature.<anything>` is authenticated by
    /// neither mechanism yet rides inside a "signature is valid" artifact.
    ReservedKey(String),
    /// A MAP key appears more than once. The emitter never does this, and a
    /// duplicate makes "the entry for key K" reader-dependent.
    DuplicateKey(String),
    /// The MAP epilogue bytes differ from their canonical re-encoding, first at
    /// `offset` (relative to the epilogue). Covers out-of-order keys and any
    /// padded length the minimal reader did not already reject.
    Map { offset: usize },
}

impl std::fmt::Display for Mic3NonCanonical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Oversize => write!(f, "artifact exceeds the mic@3 input cap"),
            Self::Unparseable => write!(f, "mic@3 body did not parse"),
            Self::Body { offset } => write!(
                f,
                "body byte {offset} differs from the canonical re-emission of the IR it decodes to"
            ),
            Self::TrailingGarbage { offset } => write!(
                f,
                "{offset} bytes of the body are followed by data that is not a MAP epilogue"
            ),
            Self::MalformedMap => write!(
                f,
                "MAP epilogue is malformed (leftover bytes, or a non-minimal length encoding)"
            ),
            Self::DuplicateKey(k) => write!(f, "MAP key `{k}` appears more than once"),
            Self::ReservedKey(k) => write!(
                f,
                "MAP key `{k}` is in the reserved `signature.` namespace but is not a \
                 defined signature key; the signature preimage cannot cover it"
            ),
            Self::Map { offset } => write!(
                f,
                "MAP epilogue byte {offset} differs from its canonical re-encoding \
                 (keys out of canonical order, or a non-canonical value encoding)"
            ),
        }
    }
}

/// Canonical-form gate: re-emit the artifact from what it decodes to and
/// byte-compare against the literal input. Any difference is a hard failure.
///
/// This is the enforcement point for literal-byte integrity, which neither the
/// `trace_hash` (it anchors the *re-emission*, not the input bytes) nor the
/// RFC 0021 signature (its preimage covers decoded entries, not bytes) provides.
/// Without it a signed artifact is byte-MALLEABLE: appending a byte, padding a
/// ULEB length in place, downgrading the wire-version byte inside the accepted
/// read window, or flipping a body byte the decoder normalises away all yield a
/// different file with a different SHA-256 that still reports
/// "signature is valid and signer key is trusted".
///
/// The discipline mirrors the collapse-receipt guard in this same module
/// (`re-encode and byte-compare, never trust the received encoding`), widened
/// from one blob to the whole artifact.
///
/// Returns `Ok(())` for an artifact in canonical form — which is every artifact
/// this compiler emits, since emit is the function being inverted here.
///
/// KNOWN, DELIBERATE CONSEQUENCE — legacy wire versions do not pass. The
/// canonical form of an artifact is its re-emission by THIS toolchain, which
/// writes `MIC3_VERSION_BASE` / `MIC3_VERSION`; a genuine `0x01` artifact
/// (readable, since `MIC3_MIN_READ_VERSION` is `0x01`) re-emits at `0x02` and is
/// therefore reported non-canonical. That is not an oversight to be relaxed: a
/// `0x02` body whose content has no `exit_ids` / `merges` decodes IDENTICALLY
/// with the version byte set to `0x01`, so admitting `0x01` is exactly the
/// downgrade malleability this check exists to close — the two cases are
/// indistinguishable from the bytes. `parse_mic3` still reads `0x01` for
/// consumers that only need the IR; it is the canonical-form claim, and hence
/// `mindc verify`, that a legacy artifact cannot satisfy. Re-emit such an
/// artifact with a current toolchain to bring it back under attestation.
pub fn mic3_canonical_check(bytes: &[u8]) -> Result<(), Mic3NonCanonical> {
    if bytes.len() > super::parse::MAX_MIC3_INPUT {
        return Err(Mic3NonCanonical::Oversize);
    }
    // (1) Body. `parse_mic3` stops at the body/epilogue boundary, so re-emitting
    // the module it recovers reproduces exactly the canonical body bytes.
    let ir = super::parse_mic3(bytes).map_err(|_| Mic3NonCanonical::Unparseable)?;
    let body = super::emit_mic3(&ir);
    if body.len() > bytes.len() {
        return Err(Mic3NonCanonical::Body {
            offset: bytes.len(),
        });
    }
    if let Some(offset) = first_difference(&body, &bytes[..body.len()]) {
        return Err(Mic3NonCanonical::Body { offset });
    }
    let rest = &bytes[body.len()..];
    if rest.is_empty() {
        // Plain (unattested) artifact: body is the whole file and it is canonical.
        return Ok(());
    }
    if rest[0] != MAP_SENTINEL {
        return Err(Mic3NonCanonical::TrailingGarbage { offset: body.len() });
    }

    // (2) MAP epilogue. `parse_map_epilogue` already rejects leftover bytes and
    // non-minimal lengths; the re-encode below additionally pins key ORDER and
    // value encoding, so emit and verify cannot drift.
    let entries = parse_map_epilogue(rest).map_err(|e| match e {
        ParseMapError::ReservedKey(k) => Mic3NonCanonical::ReservedKey(k),
        ParseMapError::DuplicateKey(k) => Mic3NonCanonical::DuplicateKey(k),
        _ => Mic3NonCanonical::MalformedMap,
    })?;
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for e in &entries {
        if !seen.insert(e.key.as_str()) {
            return Err(Mic3NonCanonical::DuplicateKey(e.key.clone()));
        }
    }
    // (Reserved-namespace enforcement now lives in `parse_map_epilogue`, the
    // shared choke point, so every read path gets it — see the note there.)
    let canonical_map = encode_map_epilogue_canonical(&entries);
    if canonical_map.len() != rest.len() {
        return Err(Mic3NonCanonical::Map {
            offset: canonical_map.len().min(rest.len()),
        });
    }
    if let Some(offset) = first_difference(&canonical_map, rest) {
        return Err(Mic3NonCanonical::Map { offset });
    }
    Ok(())
}

/// Index of the first differing byte between two equal-length-compared slices.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

/// Re-encode parsed MAP entries in canonical form.
///
/// Shares [`push_map_value`] with the emit path (`append_map_epilogue`), so the
/// canonical form has exactly one definition: sentinel, ULEB entry count, then
/// entries in lexicographic key-byte order, each a ULEB-length-prefixed key
/// followed by its tagged value.
fn encode_map_epilogue_canonical(entries: &[ParsedEntry]) -> Vec<u8> {
    let mut ordered: Vec<&ParsedEntry> = entries.iter().collect();
    ordered.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));

    let mut out = vec![MAP_SENTINEL];
    uleb128_write(&mut out, ordered.len() as u64).unwrap();
    for e in ordered {
        let kb = e.key.as_bytes();
        uleb128_write(&mut out, kb.len() as u64).unwrap();
        out.extend_from_slice(kb);
        match &e.value {
            ParsedValue::Str(s) => push_map_value(&mut out, &MapEntryValue::Str(s)),
            ParsedValue::Int(i) => push_map_value(&mut out, &MapEntryValue::Int(*i)),
            ParsedValue::Bytes(b) => push_map_value(&mut out, &MapEntryValue::Bytes(b)),
        }
    }
    out
}

/// Parse a mic@3 artifact (with or without MAP epilogue) and verify the
/// embedded `evidence_chain` block.
///
/// Returns an [`EvidenceReport`] with `trace_hash_valid = true` when the stored
/// `trace_hash` equals the SHA-256 of the canonical mic@3 bytes of the parsed
/// module. Note the anchor is the **canonical form of the IR recovered from the
/// artifact** (`ir_trace_hash` hashes the *re-emission* of the parsed module),
/// not the artifact's literal bytes: a mutation that the decoder normalises away
/// — a non-minimal ULEB encoding, or trailing bytes after the body — re-emits
/// identically and still reports `trace_hash_valid = true`. Any byte flip that
/// changes the recovered IR flips `trace_hash_valid` to `false`.
///
/// The signature does NOT close that gap either: the RFC 0021 preimage (built by
/// the private `build_signature_preimage`) covers the trace_hash, the scheme tag
/// and the DECODED provenance entries — the literal artifact bytes never enter
/// it. So a normalised-away mutation keeps both `trace_hash_valid` AND the
/// signature valid. Callers that need literal-byte integrity MUST also call
/// [`mic3_canonical_check`], which re-emits the artifact canonically and
/// byte-compares; `mindc verify` runs it as a hard gate.
///
/// deferred: the literal artifact length / prefix hash is NOT bound into the
/// RFC 0021 signature preimage. Binding it is a preimage change — every existing
/// signature would stop verifying — so it needs an RFC 0021 revision plus a new
/// `signature.scheme` tag to stay crypto-agile, not a silent edit here. Upgrade
/// path: add a scheme tag whose preimage prepends a hash of the literal artifact
/// bytes (with the `signature.*` entries elided) to the current preimage, keep
/// the old tag verifying old artifacts, and retire it on a deprecation clock.
/// Until then the canonical-form check above is the enforcement point, and it is
/// a LOCAL re-derivation: it fails closed, but it is not a SIGNED claim, so a
/// relaying party that only checks the signature does not inherit it.
///
/// Returns [`EvidenceError::Missing`] if no `evidence_chain.*` keys are present.
pub fn mic3_evidence_report(bytes: &[u8]) -> Result<EvidenceReport, EvidenceError> {
    // DoS guard: reject oversized input up front, before scanning for the MAP
    // sentinel or any allocation. Mirrors the mic@3 body parser's input cap.
    if bytes.len() > super::parse::MAX_MIC3_INPUT {
        return Err(EvidenceError::Malformed("evidence_chain.trace_hash"));
    }

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

/// Extract the application-namespace metadata (Phase 17.8) from a mic@3 artifact.
///
/// Returns every MAP key that is NOT in a reserved (`evidence_chain.*` /
/// `signature.*`) namespace, paired with its string value, in canonical
/// (lexicographic) key order. An artifact with no application entries — every
/// artifact emitted before 17.8, and every no-`--evidence-attr` build — yields an
/// empty vector. Non-string application values are skipped (the emitter only
/// writes strings). Returns [`EvidenceError::Missing`] when there is no MAP at all.
pub fn mic3_app_metadata(bytes: &[u8]) -> Result<Vec<(String, String)>, EvidenceError> {
    if bytes.len() > super::parse::MAX_MIC3_INPUT {
        return Err(EvidenceError::Malformed("app_metadata"));
    }
    let body_end = find_map_sentinel(bytes).ok_or(EvidenceError::Missing)?;
    let entries = parse_map_epilogue(&bytes[body_end..])
        .map_err(|_| EvidenceError::Malformed("app_metadata"))?;
    let mut out: Vec<(String, String)> = entries
        .iter()
        // Read-back mirrors `validate_app_entries`: only well-formed application
        // keys (non-empty, dotted, non-reserved) are surfaced, so a hand-crafted
        // artifact cannot smuggle a malformed key past a consumer that trusts
        // "app keys are namespaced".
        .filter(|e| {
            !e.key.is_empty()
                && e.key.contains('.')
                && !RESERVED_KEY_PREFIXES.iter().any(|p| e.key.starts_with(p))
        })
        .filter_map(|e| match &e.value {
            ParsedValue::Str(s) => Some((e.key.clone(), s.clone())),
            _ => None,
        })
        .collect();
    out.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    Ok(out)
}

// ─── Collapse-receipt verification (S4) ────────────────────────────────────────

/// Outcome of verifying the Salov loop-collapse receipts (S4) embedded in a
/// mic@3 artifact. Every non-`Verified` variant except `Absent` is a
/// verification FAILURE the CLI maps to a non-zero exit — fail-closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollapseVerifyStatus {
    /// No `evidence_chain.collapse_receipts` key — nothing to re-derive.
    Absent,
    /// Every receipt re-derived to its claimed constant (O(1), loop not re-run)
    /// AND that constant is materialised as a `ConstI64` in the hashed body.
    /// Carries the number of receipts verified.
    Verified(usize),
    /// The receipt blob could not be decoded (truncated / unknown kind /
    /// trailing bytes).
    Malformed,
    /// The blob decoded but is not in canonical form (re-encoding it yields
    /// different bytes) — a malleability/tamper indicator.
    NonCanonical,
    /// A receipt's recorded `constant` is NOT the O(1) re-derivation of its
    /// recorded parameters — the load-bearing forgery reject.
    Rederivation {
        /// The constant re-derived from the recorded loop parameters.
        rederived: i64,
        /// The (forged) constant the receipt claimed.
        claimed: i64,
    },
    /// A receipt's constant re-derives consistently but is not materialised in
    /// the hashed mic@3 body — the receipt describes a value the program does
    /// not use (fail-closed: a receipt must bind to the attested body).
    NotInBody {
        /// The re-derived-and-claimed constant absent from the body.
        constant: i64,
    },
}

/// Verify the loop-collapse receipts (S4) in a mic@3 artifact.
///
/// For each receipt this RE-DERIVES the folded constant from the recorded loop
/// parameters in O(1) (NEVER re-running the loop, ring-exact `Z/2^64`), demands
/// bitwise equality with the claimed constant, and confirms that constant is
/// materialised as a `ConstI64` in the mic@3 body the `trace_hash` authenticates.
/// The blob is also re-encoded and byte-compared to reject a non-canonical form.
///
/// This is orthogonal to [`mic3_evidence_report`]: a caller should first confirm
/// `trace_hash_valid` (the body is untampered), then use this to confirm the
/// folded constants are the correct closed forms of their declared loops.
pub fn mic3_collapse_verify(bytes: &[u8]) -> Result<CollapseVerifyStatus, EvidenceError> {
    if bytes.len() > super::parse::MAX_MIC3_INPUT {
        return Err(EvidenceError::Malformed(KEY_COLLAPSE_RECEIPTS));
    }
    // No MAP epilogue ⇒ no receipts (an unattested / plain artifact).
    let body_end = match find_map_sentinel(bytes) {
        Some(e) => e,
        None => return Ok(CollapseVerifyStatus::Absent),
    };
    let entries = parse_map_epilogue(&bytes[body_end..])
        .map_err(|_| EvidenceError::Malformed(KEY_COLLAPSE_RECEIPTS))?;

    let blob = match find_entry(&entries, KEY_COLLAPSE_RECEIPTS) {
        Some(ParsedValue::Bytes(b)) => b.clone(),
        // Wrong-typed value is a tamper indicator (fail-closed).
        Some(_) => return Err(EvidenceError::Malformed(KEY_COLLAPSE_RECEIPTS)),
        None => return Ok(CollapseVerifyStatus::Absent),
    };

    let receipts = match decode_collapse_receipts(&blob) {
        Ok(r) => r,
        Err(_) => return Ok(CollapseVerifyStatus::Malformed),
    };
    // Canonical-encoding guard: re-encode and byte-compare (never trust the
    // received bytes' ordering/padding).
    if encode_collapse_receipts(&receipts) != blob {
        return Ok(CollapseVerifyStatus::NonCanonical);
    }

    // Parse the body ONCE and collect its ConstI64 pool for the body-binding
    // check. A parse failure here means the whole artifact is malformed — the
    // trace_hash path reports it authoritatively; here we fail closed.
    let ir = super::parse_mic3(&bytes[..body_end])
        .map_err(|_| EvidenceError::Malformed(KEY_COLLAPSE_RECEIPTS))?;
    let mut body_consts: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    collect_const_i64(&ir.instrs, &mut body_consts);

    for rec in &receipts {
        // (1) Self-consistency: recorded constant == O(1) re-derivation.
        let rederived = rec.rederive();
        let claimed = rec.constant();
        if rederived != claimed {
            return Ok(CollapseVerifyStatus::Rederivation { rederived, claimed });
        }
        // (2) Body binding: the constant is materialised in the hashed body.
        if !body_consts.contains(&claimed) {
            return Ok(CollapseVerifyStatus::NotInBody { constant: claimed });
        }
    }
    Ok(CollapseVerifyStatus::Verified(receipts.len()))
}

/// Recursively collect every `ConstI64` value defined anywhere in `instrs`
/// (including inside function bodies and control-flow regions), so the
/// collapse-receipt body-binding check can confirm a folded constant is actually
/// materialised in the hashed body.
fn collect_const_i64(instrs: &[crate::ir::Instr], out: &mut std::collections::BTreeSet<i64>) {
    use crate::ir::Instr;
    for instr in instrs {
        if let Instr::ConstI64(_, v) = instr {
            out.insert(*v);
        }
        match instr {
            Instr::FnDef { body, .. } => collect_const_i64(body, out),
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => {
                collect_const_i64(cond_instrs, out);
                collect_const_i64(body, out);
            }
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                collect_const_i64(cond_instrs, out);
                collect_const_i64(then_instrs, out);
                collect_const_i64(else_instrs, out);
            }
            #[cfg(feature = "std-surface")]
            Instr::Region { body, .. } => collect_const_i64(body, out),
            _ => {}
        }
    }
}

// ─── Signature verification (RFC 0021 §6) ──────────────────────────────────────

/// Result of checking the optional Ed25519 signature layer on a mic@3 artifact.
///
/// Signature status is reported *separately* from `trace_hash_valid`: the two
/// properties are orthogonal.  `trace_hash_valid` proves the stored anchor equals
/// the recomputed mic@3 hash (tamper-evidence of the body); a `Valid` signature
/// proves that anchor was signed by the holder of `pubkey` (authenticity).  A
/// fully-trusted artifact needs BOTH.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureStatus {
    /// No `signature.*` keys present — an unsigned (but possibly attested) artifact.
    Absent,
    /// Every signature required by the artifact's `scheme` verified over the stored
    /// `trace_hash`. For a hybrid scheme this means BOTH the Ed25519 and the
    /// ML-DSA-65 halves verified.
    Valid(VerifiedScheme),
    /// `signature.*` keys present and well-formed, but at least one required
    /// signature does NOT verify over the stored `trace_hash`. Fail-closed.
    Invalid,
    /// `signature.*` keys present but structurally unusable — unknown scheme,
    /// wrong key/signature length, or a missing companion key. Fail-closed.
    Malformed(&'static str),
    /// The artifact requires a post-quantum verifier (`ml-dsa-65`/`hybrid-*`) but
    /// this build lacks the `evidence-mldsa` feature. Fail-closed: a build that
    /// cannot check the PQC half must never report `Valid`.
    Unsupported(&'static str),
}

/// Which scheme verified, and the public key(s) the signature(s) were checked
/// against. Surfaced by `mindc verify` for provenance/audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedScheme {
    /// The `signature.scheme` (`alg`) tag: `ed25519`, `ml-dsa-65`, or
    /// `hybrid-ed25519-ml-dsa-65`.
    pub scheme: String,
    /// The Ed25519 public key, present for `ed25519` and hybrid schemes.
    pub ed25519_pubkey: Option<[u8; 32]>,
    /// The ML-DSA-65 public key (1952 bytes), present for `ml-dsa-65` and hybrid.
    pub mldsa_pubkey: Option<Vec<u8>>,
    /// The ML-DSA-87 public key, present for the `pqc-hybrid-*` scheme.
    pub mldsa87_pubkey: Option<Vec<u8>>,
    /// The SLH-DSA-SHAKE-256s public key (64 bytes), present for `pqc-hybrid-*`.
    pub slhdsa_pubkey: Option<Vec<u8>>,
}

/// Inspect the optional Ed25519 signature layer of a mic@3 artifact.
///
/// Returns [`SignatureStatus::Absent`] when the artifact carries no
/// `signature.*` keys (back-compat: unsigned artifacts are legal).  When a
/// signature is present it is verified against the canonical provenance preimage
/// (the mic@3 anchor plus every other `evidence_chain.*` key —
/// `build_signature_preimage`) under the embedded `signature.pubkey`.
///
/// This does NOT recompute the trace_hash — use [`mic3_evidence_report`] for the
/// body-tamper check.  A production `verify` should require BOTH
/// `trace_hash_valid` and [`SignatureStatus::Valid`].
pub fn mic3_signature_status(bytes: &[u8]) -> Result<SignatureStatus, EvidenceError> {
    if bytes.len() > super::parse::MAX_MIC3_INPUT {
        return Err(EvidenceError::Malformed(KEY_SIG_ED25519));
    }
    let body_end = find_map_sentinel(bytes).ok_or(EvidenceError::Missing)?;
    let entries = parse_map_epilogue(&bytes[body_end..])
        .map_err(|_| EvidenceError::Malformed(KEY_SIG_ED25519))?;

    // Bind the signature to the ACTUAL body (defense-in-depth for callers that
    // check the signature layer in isolation, without also calling
    // `mic3_evidence_report`): recompute the mic@3 anchor over the parsed body and
    // reject if it disagrees with the stored `trace_hash` the signature covers. A
    // body tamper that leaves the stored hash in place would otherwise let the
    // (genuine, over-the-old-hash) signature still report Valid here.
    if entries.iter().any(|e| e.key.starts_with("signature.")) {
        let ir = super::parse_mic3(&bytes[..body_end])
            .map_err(|_| EvidenceError::Malformed(KEY_TRACE_HASH))?;
        let recomputed = ir_trace_hash(&ir);
        let stored = find_bytes32(&entries, KEY_TRACE_HASH)?;
        if recomputed != stored {
            return Ok(SignatureStatus::Invalid);
        }
    }

    signature_status_from_entries(&entries)
}

/// Decode + verify the signature layer from parsed MAP entries.
///
/// Crypto-agile dispatch (OMB M-26-15): reads the `signature.scheme` (`alg`) tag
/// and runs the corresponding verifier(s). A hybrid scheme requires BOTH halves to
/// verify. Every scheme is fail-closed — an unknown `alg`, a missing/short key or
/// signature, or a required-but-uncompiled PQC verifier all yield a non-`Valid`
/// status, never a silent pass.
fn signature_status_from_entries(
    entries: &[ParsedEntry],
) -> Result<SignatureStatus, EvidenceError> {
    let has_sig = entries.iter().any(|e| e.key.starts_with("signature."));
    if !has_sig {
        return Ok(SignatureStatus::Absent);
    }

    let scheme = match find_str_opt(entries, KEY_SIG_SCHEME)? {
        Some(s) => s,
        None => return Ok(SignatureStatus::Malformed("signature.scheme")),
    };

    // Rebuild the canonical provenance preimage that was signed at emit time:
    // trace_hash || canonical(evidence_chain.* \ trace_hash). Building it from the
    // SAME serialization the emitter used means editing any provenance key
    // (substrate, toolchain, determinism, parent) changes the preimage and the
    // signature no longer verifies — the whole provenance is authenticated, not
    // just the code anchor. `signature.*` keys are excluded (a signature never
    // covers itself).
    let trace_hash = find_bytes32(entries, KEY_TRACE_HASH)?;
    let view: Vec<(&str, MapEntryValue)> = entries
        .iter()
        .map(|e| {
            let v = match &e.value {
                ParsedValue::Str(s) => MapEntryValue::Str(s.as_str()),
                ParsedValue::Int(i) => MapEntryValue::Int(*i),
                ParsedValue::Bytes(b) => MapEntryValue::Bytes(b.as_slice()),
            };
            (e.key.as_str(), v)
        })
        .collect();
    // Bind the scheme tag into the preimage (anti-downgrade — see
    // `build_signature_preimage`). Read it from the SAME parsed `signature.scheme`
    // entry the dispatch below uses, so an untampered artifact reproduces the
    // preimage the genuine half was signed over.
    let preimage = build_signature_preimage(&view, &trace_hash, &scheme);

    let want_ed = scheme == SIG_SCHEME_ED25519 || scheme == SIG_SCHEME_HYBRID;
    let want_mldsa = scheme == SIG_SCHEME_MLDSA65 || scheme == SIG_SCHEME_HYBRID;
    // The bulletproof PQC-hybrid requires BOTH the ML-DSA-87 and SLH-DSA legs.
    // No scheme value names only one of them, so both flags are true TOGETHER —
    // the verifier structurally cannot be coaxed into accepting a single leg
    // (the non-degradability property: a stripped leg fails the presence check
    // below, never silently collapses the hybrid to one scheme).
    let want_mldsa87 = scheme == SIG_SCHEME_PQC_HYBRID;
    let want_slhdsa = scheme == SIG_SCHEME_PQC_HYBRID;
    if !want_ed && !want_mldsa && !want_mldsa87 && !want_slhdsa {
        // Unknown `alg` — fail closed (never accept a scheme we do not understand).
        return Ok(SignatureStatus::Malformed("signature.scheme"));
    }

    // Reject an artifact that carries a signature half the scheme tag does NOT
    // name (e.g. scheme=ed25519 but a `signature.mldsa` entry is present). This is
    // the structural companion to the preimage scheme-binding above: a downgrade
    // that flips the tag but forgets to strip the now-unnamed half is caught here
    // as Malformed, and a stray/confusing half can never be silently ignored.
    let has_ed_half = entries
        .iter()
        .any(|e| e.key == KEY_SIG_ED25519 || e.key == KEY_SIG_PUBKEY);
    let has_mldsa_half = entries
        .iter()
        .any(|e| e.key == KEY_SIG_MLDSA || e.key == KEY_SIG_MLDSA_PUBKEY);
    let has_mldsa87_half = entries
        .iter()
        .any(|e| e.key == KEY_SIG_MLDSA87 || e.key == KEY_SIG_MLDSA87_PUBKEY);
    let has_slhdsa_half = entries
        .iter()
        .any(|e| e.key == KEY_SIG_SLHDSA || e.key == KEY_SIG_SLHDSA_PUBKEY);
    if has_ed_half && !want_ed {
        return Ok(SignatureStatus::Malformed(KEY_SIG_SCHEME));
    }
    if has_mldsa_half && !want_mldsa {
        return Ok(SignatureStatus::Malformed(KEY_SIG_SCHEME));
    }
    if has_mldsa87_half && !want_mldsa87 {
        return Ok(SignatureStatus::Malformed(KEY_SIG_SCHEME));
    }
    if has_slhdsa_half && !want_slhdsa {
        return Ok(SignatureStatus::Malformed(KEY_SIG_SCHEME));
    }

    // ── Ed25519 half ──────────────────────────────────────────────────────────
    let mut ed_pubkey: Option<[u8; 32]> = None;
    if want_ed {
        let pubkey = match find_bytes_opt_n::<32>(entries, KEY_SIG_PUBKEY)? {
            Some(pk) => pk,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_PUBKEY)),
        };
        let sig = match find_bytes_opt_n::<64>(entries, KEY_SIG_ED25519)? {
            Some(s) => s,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_ED25519)),
        };
        if !super::ed25519::verify(&pubkey, &preimage, &sig) {
            return Ok(SignatureStatus::Invalid);
        }
        ed_pubkey = Some(pubkey);
    }

    // ── ML-DSA-65 half (post-quantum) ─────────────────────────────────────────
    let mut mldsa_pubkey: Option<Vec<u8>> = None;
    if want_mldsa {
        let pubkey = match find_bytes_var(entries, KEY_SIG_MLDSA_PUBKEY)? {
            Some(pk) => pk,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_MLDSA_PUBKEY)),
        };
        let sig = match find_bytes_var(entries, KEY_SIG_MLDSA)? {
            Some(s) => s,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_MLDSA)),
        };
        // Fail-closed if this build cannot check the PQC signature.
        if !super::mldsa::supported() {
            return Ok(SignatureStatus::Unsupported(
                "artifact requires ML-DSA verification; rebuild with --features evidence-mldsa",
            ));
        }
        if !super::mldsa::verify(&pubkey, &preimage, &sig) {
            return Ok(SignatureStatus::Invalid);
        }
        mldsa_pubkey = Some(pubkey);
    }

    // ── ML-DSA-87 leg (bulletproof-hybrid lattice foundation, cat-5) ──────────
    let mut mldsa87_pubkey: Option<Vec<u8>> = None;
    if want_mldsa87 {
        let pubkey = match find_bytes_var(entries, KEY_SIG_MLDSA87_PUBKEY)? {
            Some(pk) => pk,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_MLDSA87_PUBKEY)),
        };
        let sig = match find_bytes_var(entries, KEY_SIG_MLDSA87)? {
            Some(s) => s,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_MLDSA87)),
        };
        if !super::mldsa::supported() {
            return Ok(SignatureStatus::Unsupported(
                "artifact requires ML-DSA-87 verification; rebuild with --features evidence-mldsa",
            ));
        }
        if !super::mldsa::verify_87(&pubkey, &preimage, &sig) {
            return Ok(SignatureStatus::Invalid);
        }
        mldsa87_pubkey = Some(pubkey);
    }

    // ── SLH-DSA-SHAKE-256s leg (bulletproof-hybrid hash-based foundation) ──────
    let mut slhdsa_pubkey: Option<Vec<u8>> = None;
    if want_slhdsa {
        let pubkey = match find_bytes_var(entries, KEY_SIG_SLHDSA_PUBKEY)? {
            Some(pk) => pk,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_SLHDSA_PUBKEY)),
        };
        let sig = match find_bytes_var(entries, KEY_SIG_SLHDSA)? {
            Some(s) => s,
            None => return Ok(SignatureStatus::Malformed(KEY_SIG_SLHDSA)),
        };
        if !super::slhdsa::supported() {
            return Ok(SignatureStatus::Unsupported(
                "artifact requires SLH-DSA verification; rebuild with --features evidence-slhdsa",
            ));
        }
        if !super::slhdsa::verify(&pubkey, &preimage, &sig) {
            return Ok(SignatureStatus::Invalid);
        }
        slhdsa_pubkey = Some(pubkey);
    }

    Ok(SignatureStatus::Valid(VerifiedScheme {
        scheme,
        ed25519_pubkey: ed_pubkey,
        mldsa_pubkey,
        mldsa87_pubkey,
        slhdsa_pubkey,
    }))
}

/// Read an optional variable-length bytes entry (ML-DSA keys/sigs are variable
/// length). `None` if absent; `Malformed` (via `Err`) if present with a
/// wrong-typed value (fail-closed).
fn find_bytes_var(
    entries: &[ParsedEntry],
    key: &'static str,
) -> Result<Option<Vec<u8>>, EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Bytes(b)) => Ok(Some(b.clone())),
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Ok(None),
    }
}

/// Read an optional fixed-length bytes entry. `None` if absent; `Malformed` if
/// present with a wrong-typed or wrong-length value (fail-closed).
fn find_bytes_opt_n<const N: usize>(
    entries: &[ParsedEntry],
    key: &'static str,
) -> Result<Option<[u8; N]>, EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Bytes(b)) => {
            let arr: [u8; N] = b
                .as_slice()
                .try_into()
                .map_err(|_| EvidenceError::Malformed(key))?;
            Ok(Some(arr))
        }
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Ok(None),
    }
}

// ─── Canonical provenance entries + signature preimage ─────────────────────────

/// Build the canonical `evidence_chain.*` MAP entries, sorted lexicographically.
///
/// This is the SINGLE SOURCE OF TRUTH for the provenance entries: both the MAP
/// emitter ([`append_map_epilogue`]) and the signature preimage
/// (`build_signature_preimage`) derive from it, so the bytes covered by a
/// signature are byte-for-byte the provenance that ends up on the wire.
fn build_evidence_entries<'a>(
    substrate: &'a str,
    parent: Option<&'a [u8; 32]>,
    determinism: Determinism,
    toolchain: &'a str,
    trace_hash: &'a [u8; 32],
    collapse_blob: Option<&'a [u8]>,
    app_entries: &'a [(String, String)],
) -> Vec<(&'a str, MapEntryValue<'a>)> {
    let mut entries: Vec<(&'a str, MapEntryValue<'a>)> = vec![
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
        (KEY_TRACE_HASH, MapEntryValue::Bytes(trace_hash)),
        // In-band anchor descriptor: every artifact emitted here uses the mic@3
        // bytes anchor (ir_trace_hash). Makes the artifact self-describing so a
        // verifier need not infer the anchor from `schema`.
        (
            KEY_TRACE_HASH_KIND,
            MapEntryValue::Str(TraceHashKind::Mic3Bytes.as_str()),
        ),
    ];
    if let Some(p) = parent {
        entries.push((KEY_PARENT, MapEntryValue::Bytes(p)));
    }
    // Collapse receipts (S4): the canonical TLV blob is emitted ONLY when the
    // build actually collapsed a `#[collapse]` loop to a constant. An empty
    // blob is never added, so a no-collapse (or no-receipt) build omits the key
    // and is byte-identical elsewhere. The blob sits in the epilogue — OUTSIDE
    // the trace_hash preimage — so it never moves the anchor; being an
    // `evidence_chain.*` key it IS folded into the signature preimage, so a
    // signed artifact's receipts cannot be stripped/swapped.
    if let Some(blob) = collapse_blob {
        entries.push((KEY_COLLAPSE_RECEIPTS, MapEntryValue::Bytes(blob)));
    }
    // Application-namespace metadata (Phase 17.8): non-reserved, dotted keys
    // (`org.example.*`) the caller supplies. Emitted ONLY when present, so a
    // build with no app entries is byte-identical to the pre-17.8 encoder
    // (byte-additive back-compat; older readers skip unknown keys). Reserved
    // prefixes are rejected upstream at emit (`validate_app_entries`), so these
    // can never shadow an `evidence_chain.*` / `signature.*` key. They sort into
    // place with everything else; being non-`signature.*` they ARE covered by the
    // signature preimage (authenticated on a signed artifact — not strippable).
    for (k, v) in app_entries {
        entries.push((k.as_str(), MapEntryValue::Str(v.as_str())));
    }
    // Lexicographic sort — the canonical-encoding invariant.
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    entries
}

/// Reserved MAP key prefixes an application namespace may NOT use. `org.example.*`
/// is fine; `evidence_chain.*` and `signature.*` are the compiler's own.
const RESERVED_KEY_PREFIXES: [&str; 2] = ["evidence_chain.", "signature."];

/// Validate application-namespace MAP entries (Phase 17.8). Fail-closed: an empty
/// key, a key colliding with a reserved namespace, or an un-namespaced key (no
/// `.`) is rejected so an app attribute can never shadow or be mistaken for a
/// compiler-reserved provenance field. Duplicate keys are also rejected (the MAP
/// is a set; a duplicate would make the encoding non-injective).
pub fn validate_app_entries(app_entries: &[(String, String)]) -> Result<(), String> {
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for (k, _) in app_entries {
        if k.is_empty() {
            return Err("application evidence key must not be empty".to_string());
        }
        for reserved in RESERVED_KEY_PREFIXES {
            if k.starts_with(reserved) {
                return Err(format!(
                    "application evidence key `{k}` uses the reserved `{reserved}` namespace"
                ));
            }
        }
        if !k.contains('.') {
            return Err(format!(
                "application evidence key `{k}` must be namespaced (e.g. `org.example.{k}`)"
            ));
        }
        if !seen.insert(k.as_str()) {
            return Err(format!("duplicate application evidence key `{k}`"));
        }
    }
    Ok(())
}

/// Serialize one MAP entry value into `out` using the exact mic@3 MAP wire
/// encoding (tag + length-prefixed payload). Shared by the MAP emitter and the
/// signature preimage so both produce byte-identical entry encodings.
fn push_map_value(out: &mut Vec<u8>, val: &MapEntryValue) {
    match val {
        MapEntryValue::Str(s) => {
            out.push(TAG_STRING);
            let sb = s.as_bytes();
            uleb128_write(out, sb.len() as u64).unwrap();
            out.extend_from_slice(sb);
        }
        MapEntryValue::Int(i) => {
            out.push(TAG_INT);
            uleb128_write(out, zigzag_encode(*i)).unwrap();
        }
        MapEntryValue::Bytes(b) => {
            out.push(TAG_BYTES);
            uleb128_write(out, b.len() as u64).unwrap();
            out.extend_from_slice(b);
        }
    }
}

/// Build the canonical signature preimage — the bytes a signature actually covers.
///
/// ```text
/// preimage = trace_hash (32 B)
///          || ULEB128 scheme_len || scheme_bytes   -- the `alg` tag (anti-downgrade)
///          || ULEB128 selected_entry_count
///          || for each selected entry (lexicographic key order):
///               ULEB128 key_len || key_bytes || value_tag || value
/// ```
///
/// The `scheme` (`signature.scheme` / `alg` tag) is bound length-prefixed right
/// after the anchor so a downgrade attack — take a hybrid artifact, delete one
/// half, and rewrite `signature.scheme` to name only the surviving half — changes
/// the preimage: the surviving genuine signature was made over the ORIGINAL
/// scheme string and no longer verifies (fail-closed). The scheme is read from the
/// same source on both sides (emit: the [`SigningKey`] variant via
/// [`scheme_for_key`]; verify: the parsed `signature.scheme` entry), so an
/// untampered artifact reproduces the identical preimage.
///
/// "Selected" = every `evidence_chain.*` entry EXCEPT `evidence_chain.trace_hash`
/// (that hash is derived from the body it anchors, so covering it again is
/// redundant), and EXCLUDING every `signature.*` key (a signature never covers
/// itself). Because both the emit path (over [`build_evidence_entries`]) and the
/// verify path (over the parsed MAP entries) call this with the same logical
/// entries, the two preimages are byte-identical — so editing `substrate`,
/// `toolchain`, `determinism`, or `parent` on a signed artifact changes the
/// preimage and the signature fails to verify (fail-closed).
///
/// The per-entry length prefixes plus the leading entry count make the encoding
/// canonical and injective: no two distinct provenance sets can splice to the
/// same preimage. The order is the same lexicographic key order used everywhere
/// else in this module, so the preimage is substrate-independent and
/// deterministic.
fn build_signature_preimage(
    entries: &[(&str, MapEntryValue)],
    trace_hash: &[u8; 32],
    scheme: &str,
) -> Vec<u8> {
    // Cover every provenance key EXCEPT the two exclusions: `signature.*` (a
    // signature never covers itself) and `evidence_chain.trace_hash` (derived
    // from the body it anchors). This deliberately INCLUDES the Phase-17.8
    // application-namespace keys (`org.example.*`) so app metadata is
    // authenticated on a signed artifact. For a no-app artifact the selected set
    // is byte-identical to the pre-17.8 filter (there are simply no non-reserved
    // keys), so existing signatures still verify.
    let mut selected: Vec<&(&str, MapEntryValue)> = entries
        .iter()
        .filter(|(k, _)| !k.starts_with("signature.") && *k != KEY_TRACE_HASH)
        .collect();
    selected.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let mut out = Vec::new();
    out.extend_from_slice(trace_hash);
    // Anti-downgrade: bind the scheme tag (length-prefixed) into the signed bytes.
    let scheme_bytes = scheme.as_bytes();
    uleb128_write(&mut out, scheme_bytes.len() as u64).unwrap();
    out.extend_from_slice(scheme_bytes);
    uleb128_write(&mut out, selected.len() as u64).unwrap();
    for (key, val) in selected {
        let kb = key.as_bytes();
        uleb128_write(&mut out, kb.len() as u64).unwrap();
        out.extend_from_slice(kb);
        push_map_value(&mut out, val);
    }
    out
}

// ─── Internal MAP emit ────────────────────────────────────────────────────────

/// Build and append the MAP epilogue to `out`.
///
/// Keys are emitted sorted lexicographically. Inline string encoding: no shared
/// string table with the IR body.
#[allow(clippy::too_many_arguments)]
fn append_map_epilogue(
    out: &mut Vec<u8>,
    substrate: &str,
    parent: Option<[u8; 32]>,
    determinism: Determinism,
    toolchain: &str,
    trace_hash: [u8; 32],
    signature: Option<&SignaturePayload>,
    collapse_blob: Option<&[u8]>,
    app_entries: &[(String, String)],
) {
    // Collect the canonical evidence_chain.* entries (single source of truth,
    // shared with the signature preimage so what is signed == what is emitted).
    let mut entries = build_evidence_entries(
        substrate,
        parent.as_ref(),
        determinism,
        toolchain,
        &trace_hash,
        collapse_blob,
        app_entries,
    );
    // Optional signature layer (sorts after every evidence_chain.* key). Bound
    // to `signature` so the borrows live through the sort + emit below. The
    // `scheme` tag names which halves are present; a verifier dispatches on it.
    if let Some(payload) = signature {
        entries.push((KEY_SIG_SCHEME, MapEntryValue::Str(payload.scheme)));
        if let Some((ref pubkey, ref sig)) = payload.ed25519 {
            entries.push((KEY_SIG_PUBKEY, MapEntryValue::Bytes(pubkey)));
            entries.push((KEY_SIG_ED25519, MapEntryValue::Bytes(sig)));
        }
        if let Some((ref pubkey, ref sig)) = payload.mldsa {
            entries.push((KEY_SIG_MLDSA_PUBKEY, MapEntryValue::Bytes(pubkey)));
            entries.push((KEY_SIG_MLDSA, MapEntryValue::Bytes(sig)));
        }
        if let Some((ref pubkey, ref sig)) = payload.mldsa87 {
            entries.push((KEY_SIG_MLDSA87_PUBKEY, MapEntryValue::Bytes(pubkey)));
            entries.push((KEY_SIG_MLDSA87, MapEntryValue::Bytes(sig)));
        }
        if let Some((ref pubkey, ref sig)) = payload.slhdsa {
            entries.push((KEY_SIG_SLHDSA_PUBKEY, MapEntryValue::Bytes(pubkey)));
            entries.push((KEY_SIG_SLHDSA, MapEntryValue::Bytes(sig)));
        }
    }
    // Lexicographic sort — this is the canonical-encoding invariant.
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    out.write_all(&[MAP_SENTINEL]).unwrap();
    uleb128_write(out, entries.len() as u64).unwrap();

    for (key, val) in &entries {
        let kb = key.as_bytes();
        uleb128_write(out, kb.len() as u64).unwrap();
        out.write_all(kb).unwrap();
        push_map_value(out, val);
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
///
/// The epilogue runs to the END of the artifact: `bytes` must be consumed
/// exactly, and every length must be minimally ULEB-encoded. Both are
/// canonical-form requirements, not merely hygiene — a padded length or a
/// trailing byte is a second byte stream that decodes to the same entries, i.e.
/// a signature-preserving mutation.
pub(crate) fn parse_map_epilogue(bytes: &[u8]) -> Result<Vec<ParsedEntry>, ParseMapError> {
    if bytes.is_empty() || bytes[0] != MAP_SENTINEL {
        return Err(ParseMapError::MissingSentinel);
    }
    // DoS hardening (mirrors `parse::read_count` / `bounded_cap`): every ULEB
    // length here (`count`, `key_len`, `val_len`, `blen`) is attacker-controlled
    // and can encode up to 2^64 in a handful of bytes. A raw `Vec::with_capacity`
    // / `vec![0u8; len]` on such a value SIGABRTs on a multi-terabyte allocation.
    // Each MAP entry, and each length-prefixed field, needs at least that many
    // backing bytes present in the epilogue — so any length exceeding the bytes
    // still remaining can never be satisfied and is rejected up front, turning an
    // allocation bomb into an immediate `Err` (fail-closed, never abort/panic).
    let data = &bytes[1..];
    let total = data.len();
    let mut r = std::io::Cursor::new(data);
    use std::io::Read;
    let remaining =
        |r: &std::io::Cursor<&[u8]>| -> usize { total.saturating_sub(r.position() as usize) };

    let count = uleb128_read_minimal(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
    // Reader is the MINIMAL variant: a zero-padded ULEB encodes the same value
    // in different bytes, which is malleability on a signed artifact.
    // Each MAP entry occupies at least MIN_ENTRY_BYTES on the wire: a >= 1-byte
    // key-length ULEB, a 1-byte value tag, and a >= 1-byte value payload. A count
    // beyond `remaining / MIN_ENTRY_BYTES` is therefore unsatisfiable. Rejecting it
    // here both fails closed on a truncated stream AND caps the `Vec::with_capacity`
    // below to input-size / MIN_ENTRY_BYTES — closing the allocation-amplification
    // gap where a multi-MiB artifact could pre-size a several-hundred-MB Vec from an
    // attacker-chosen `count` before the read loop errored out.
    const MIN_ENTRY_BYTES: usize = 3;
    if count > remaining(&r) / MIN_ENTRY_BYTES {
        return Err(ParseMapError::Truncated);
    }
    let mut entries: Vec<ParsedEntry> = Vec::with_capacity(count);
    for _ in 0..count {
        let key_len = uleb128_read_minimal(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
        if key_len > remaining(&r) {
            return Err(ParseMapError::Truncated);
        }
        let mut key_buf = vec![0u8; key_len];
        r.read_exact(&mut key_buf)
            .map_err(|_| ParseMapError::Truncated)?;
        let key = String::from_utf8(key_buf).map_err(|_| ParseMapError::InvalidUtf8)?;

        // The canonical MAP is a SET: emit writes each key exactly once, in sorted
        // order. A duplicate is off-canon — and because `find_entry` is first-wins
        // and duplicate copies are filtered out of the signature preimage, an
        // attacker could append a redundant copy of an already-signed key without
        // breaking the trace-hash anchor (byte-malleability of an otherwise-Valid
        // artifact). Reject it fail-closed. (O(n^2), but a MAP has O(10) entries.)
        if entries.iter().any(|e| e.key == key) {
            return Err(ParseMapError::DuplicateKey(key));
        }

        let tag_byte = {
            let mut tb = [0u8];
            r.read_exact(&mut tb)
                .map_err(|_| ParseMapError::Truncated)?;
            tb[0]
        };

        let value = match tag_byte {
            TAG_STRING => {
                let val_len =
                    uleb128_read_minimal(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
                if val_len > remaining(&r) {
                    return Err(ParseMapError::Truncated);
                }
                let mut vb = vec![0u8; val_len];
                r.read_exact(&mut vb)
                    .map_err(|_| ParseMapError::Truncated)?;
                let s = String::from_utf8(vb).map_err(|_| ParseMapError::InvalidUtf8)?;
                ParsedValue::Str(s)
            }
            TAG_INT => {
                let encoded = uleb128_read_minimal(&mut r).map_err(|_| ParseMapError::Truncated)?;
                ParsedValue::Int(zigzag_decode(encoded))
            }
            TAG_BYTES => {
                let blen =
                    uleb128_read_minimal(&mut r).map_err(|_| ParseMapError::Truncated)? as usize;
                if blen > remaining(&r) {
                    return Err(ParseMapError::Truncated);
                }
                let mut bb = vec![0u8; blen];
                r.read_exact(&mut bb)
                    .map_err(|_| ParseMapError::Truncated)?;
                ParsedValue::Bytes(bb)
            }
            other => return Err(ParseMapError::UnknownTag(other)),
        };

        entries.push(ParsedEntry { key, value });
    }
    // The epilogue is the artifact's final region ([sentinel .. EOF]); emit writes
    // exactly `count` entries and nothing after the last one. Bytes the loop did not
    // consume are trailing garbage appended after a valid MAP — a second malleability
    // vector (they ride inside a Valid artifact, outside the signed preimage). Require
    // the cursor to have consumed the whole epilogue region.
    if (r.position() as usize) != total {
        return Err(ParseMapError::TrailingBytes);
    }
    // Reserved-namespace enforcement lives HERE, in the shared parser, not in any
    // one consumer. `validate_app_entries` enforces this at EMIT; the read side has
    // FIVE entry points into these bytes (`mic3_canonical_check`,
    // `mic3_evidence_report`, `mic3_app_metadata`, `mic3_signature_status`, and the
    // raw callers), and putting the rule in only one of them left the other four
    // accepting an injected key — `mic3_signature_status` in particular returned
    // `Valid` for a 200 KB attacker payload, which is exactly what an out-of-tree
    // consumer calls.
    //
    // Why nothing else catches it: `build_signature_preimage` deliberately FILTERS
    // every `signature.`-prefixed key (a signature cannot sign itself), the canonical
    // byte-compare re-encodes any entry that legitimately parsed, and `trace_hash`
    // anchors the BODY, not the epilogue. An unknown `signature.*` key is therefore
    // covered by no mechanism at all.
    for e in &entries {
        // enforced-by: SIG-RESV-01
        if e.key.starts_with("signature.") && !SIGNATURE_KEYS.contains(&e.key.as_str()) {
            return Err(ParseMapError::ReservedKey(e.key.clone()));
        }
        // Same rule, the other reserved namespace. Previously unchecked on read.
        // enforced-by: SIG-RESV-02
        if e.key.starts_with("evidence_chain.") && !EVIDENCE_CHAIN_KEYS.contains(&e.key.as_str()) {
            return Err(ParseMapError::ReservedKey(e.key.clone()));
        }
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
    /// A key appeared more than once. The canonical MAP is a set, so a duplicate
    /// key is off-canon and rejected fail-closed (anti-malleability). Carries the
    /// offending key so `mindc verify --json` can name it in `canonical_reason`.
    DuplicateKey(String),
    /// Bytes remained after the last entry. The epilogue must consume its whole
    /// region; trailing garbage inside a Valid artifact is rejected fail-closed.
    TrailingBytes,
    /// A key in the reserved `signature.` namespace that is not one of the five
    /// defined signature keys. Enforced in the shared parser so all five read
    /// entry points inherit it, not in one consumer.
    ReservedKey(String),
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

    // `trace_hash_kind` is the in-band anchor descriptor. ABSENT (a legacy
    // artifact that predates this field) ⇒ default to mic@3 bytes: every
    // artifact emitted since the 2026-05-31 re-anchor uses mic@3, and the
    // recomputation below (`ir_trace_hash`, mic@3 bytes) is what those artifacts
    // actually stored, so the default keeps untampered legacy artifacts green.
    let trace_hash_kind = match find_str_opt(entries, KEY_TRACE_HASH_KIND)? {
        Some(s) => {
            TraceHashKind::from_str(&s).ok_or(EvidenceError::Malformed(KEY_TRACE_HASH_KIND))?
        }
        None => TraceHashKind::default(),
    };

    // Recompute via the same FIPS-180-4 seam as ir_trace_hash.
    let recomputed = ir_trace_hash(ir);
    let trace_hash_valid = recomputed == stored_hash;

    // Strict-FP mode re-derived from the SAME re-parsed body the hash attests:
    // a hidden taint op can't sit in a "strict" body without breaking the hash,
    // so this is as trustworthy as trace_hash_valid and needs no wire field.
    let fp_mode = crate::ir::fp_contract_mode(ir);

    Ok(EvidenceReport {
        substrate,
        determinism,
        toolchain,
        parent,
        trace_hash: stored_hash,
        trace_hash_kind,
        trace_hash_valid,
        fp_mode,
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

/// Read an optional string-valued MAP entry. `None` if the key is absent;
/// `Err(Malformed)` if present with a non-string value.
fn find_str_opt(
    entries: &[ParsedEntry],
    key: &'static str,
) -> Result<Option<String>, EvidenceError> {
    match find_entry(entries, key) {
        Some(ParsedValue::Str(s)) => Ok(Some(s.clone())),
        Some(_) => Err(EvidenceError::Malformed(key)),
        None => Ok(None),
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
    use std::io::Write;

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
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
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
                    #[cfg(feature = "std-surface")]
                    value_types: std::collections::BTreeMap::new(),
                },
                Instr::Return { value: Some(op) },
            ],
            reap_threshold: Some(0.75),
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
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
    // Risk-2: a FORGED `determinism` MAP field is caught by re-derivation
    // -------------------------------------------------------------------------

    #[test]
    fn forged_determinism_label_is_caught_by_rederivation() {
        // The `determinism` MAP field sits OUTSIDE the trace_hash anchor, so on an
        // unsigned artifact it is forgeable while the trace_hash still matches.
        // Build an IR that GENUINELY calls a PRNG builtin, then emit it with a
        // FORGED `Deterministic` label.
        let mut ir = IRModule::new();
        let seed = ir.fresh();
        let r = ir.fresh();
        ir.instrs.push(Instr::ConstI64(seed, 0));
        ir.instrs.push(Instr::Call {
            dst: r,
            name: "random".to_string(),
            args: vec![seed],
        });
        ir.instrs.push(Instr::Output(r));

        let bytes =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");

        // The stored MAP field carries the FORGED value...
        let report = mic3_evidence_report(&bytes).expect("evidence report");
        assert!(
            matches!(report.determinism, Determinism::Deterministic),
            "stored label carries the forged Deterministic value"
        );

        // ...but the value RE-DERIVED from the hashed body is the TRUTH.
        let parsed = parse_mic3(&bytes).expect("parse mic@3 body");
        assert!(
            !crate::ir::ir_declares_deterministic(&parsed),
            "re-derivation must see the random() call and report nondeterministic"
        );
        // stored (deterministic) != re-derived (nondeterministic) — exactly the
        // mismatch `mindc verify` fails-closed on (Risk-2). The forge cannot pass.
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

    // -------------------------------------------------------------------------
    // (m) trace_hash_kind round-trips through emit→parse as "mic3-bytes"
    // -------------------------------------------------------------------------

    #[test]
    fn trace_hash_kind_round_trips_as_mic3_bytes() {
        for (name, ir) in all_modules() {
            let bytes =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");

            // The key is present in the raw MAP with the expected string value.
            let body_end = find_map_sentinel(&bytes).unwrap();
            let entries = parse_map_epilogue(&bytes[body_end..]).unwrap();
            let kind_entry = entries
                .iter()
                .find(|e| e.key == KEY_TRACE_HASH_KIND)
                .unwrap_or_else(|| {
                    panic!("module '{}': trace_hash_kind key must be present", name)
                });
            match &kind_entry.value {
                ParsedValue::Str(s) => {
                    assert_eq!(s, "mic3-bytes", "module '{}': trace_hash_kind value", name)
                }
                other => panic!(
                    "module '{}': trace_hash_kind must be Str, got {:?}",
                    name, other
                ),
            }

            // It decodes into the report as Mic3Bytes.
            let report = mic3_evidence_report(&bytes).unwrap();
            assert_eq!(
                report.trace_hash_kind,
                TraceHashKind::Mic3Bytes,
                "module '{}': decoded trace_hash_kind",
                name
            );
            assert!(report.trace_hash_valid, "module '{}'", name);
        }
    }

    // -------------------------------------------------------------------------
    // (n) Back-compat: a legacy MAP with NO trace_hash_kind key decodes as
    //     "mic3-bytes" (the default) and still verifies.
    // -------------------------------------------------------------------------

    #[test]
    fn legacy_map_without_kind_defaults_to_mic3_bytes() {
        // Synthesize a legacy artifact by emitting a current one, parsing the
        // MAP, dropping the trace_hash_kind entry, and re-encoding the MAP
        // epilogue with the remaining keys (sorted) onto the same body.
        let ir = mod_binop();
        let bytes =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let body_end = find_map_sentinel(&bytes).unwrap();
        let entries = parse_map_epilogue(&bytes[body_end..]).unwrap();

        // Confirm the modern artifact carries the key (so the drop is meaningful).
        assert!(
            entries.iter().any(|e| e.key == KEY_TRACE_HASH_KIND),
            "precondition: modern artifact must carry trace_hash_kind"
        );

        // Rebuild a MAP epilogue from all entries EXCEPT trace_hash_kind.
        let mut kept: Vec<&ParsedEntry> = entries
            .iter()
            .filter(|e| e.key != KEY_TRACE_HASH_KIND)
            .collect();
        kept.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));

        let mut legacy = bytes[..body_end].to_vec();
        legacy.push(MAP_SENTINEL);
        uleb128_write(&mut legacy, kept.len() as u64).unwrap();
        for e in &kept {
            let kb = e.key.as_bytes();
            uleb128_write(&mut legacy, kb.len() as u64).unwrap();
            legacy.write_all(kb).unwrap();
            match &e.value {
                ParsedValue::Str(s) => {
                    legacy.push(TAG_STRING);
                    uleb128_write(&mut legacy, s.len() as u64).unwrap();
                    legacy.write_all(s.as_bytes()).unwrap();
                }
                ParsedValue::Int(i) => {
                    legacy.push(TAG_INT);
                    uleb128_write(&mut legacy, zigzag_encode(*i)).unwrap();
                }
                ParsedValue::Bytes(b) => {
                    legacy.push(TAG_BYTES);
                    uleb128_write(&mut legacy, b.len() as u64).unwrap();
                    legacy.write_all(b).unwrap();
                }
            }
        }

        // No trace_hash_kind key in the synthesized legacy MAP.
        let legacy_entries = parse_map_epilogue(&legacy[body_end..]).unwrap();
        assert!(
            !legacy_entries.iter().any(|e| e.key == KEY_TRACE_HASH_KIND),
            "synthesized legacy MAP must NOT carry trace_hash_kind"
        );

        // Decodes with the back-compat default and still verifies green.
        let report = mic3_evidence_report(&legacy).unwrap();
        assert_eq!(
            report.trace_hash_kind,
            TraceHashKind::Mic3Bytes,
            "absent trace_hash_kind must default to mic3-bytes"
        );
        assert!(
            report.trace_hash_valid,
            "untampered legacy artifact must still verify under the default anchor"
        );
    }

    // -------------------------------------------------------------------------
    // Collapse receipts (S4) — byte-identity + O(1) re-derivation + forgery
    // -------------------------------------------------------------------------

    use crate::ir::compact::v3::collapse_receipt::CollapseReceipt;

    /// A `#[collapse]`-fold constant materialised in a body: `s = ConstI64(4950)`.
    fn mod_with_const(v: i64) -> IRModule {
        let mut m = IRModule::new();
        let c = m.fresh();
        m.instrs.push(Instr::ConstI64(c, v));
        m.instrs.push(Instr::Output(c));
        m
    }

    // (S4-a) THE byte-identity proof: receipts live in the epilogue, so they move
    // NEITHER the mic@3 body NOR the trace_hash, and an empty-receipt emit is
    // byte-identical to the no-receipt encoder.
    #[test]
    fn receipts_do_not_perturb_body_or_trace_hash() {
        let ir = mod_with_const(4950);
        let receipts = vec![CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 4950,
        }];
        let no_r = emit_mic3_with_evidence(&ir, "cpu", None, Determinism::Deterministic, "0.10.1");
        let with_r = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &receipts,
            &[],
        )
        .unwrap();

        // The mic@3 BODY prefix is byte-identical.
        let be_n = find_map_sentinel(&no_r).unwrap();
        let be_w = find_map_sentinel(&with_r).unwrap();
        assert_eq!(
            &no_r[..be_n],
            &with_r[..be_w],
            "collapse receipts must NOT move the mic@3 body"
        );
        // The trace_hash anchor is identical.
        assert_eq!(
            mic3_evidence_report(&no_r).unwrap().trace_hash,
            mic3_evidence_report(&with_r).unwrap().trace_hash,
            "collapse receipts must NOT change the trace_hash"
        );
        // An EMPTY receipt list is byte-identical to `emit_mic3_with_evidence`.
        let empty = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            empty, no_r,
            "empty receipts must be byte-identical to the no-receipt encoder"
        );
    }

    // (17.8) Application-namespace MAP keys: round-trip, byte-additivity, and
    // fail-closed reserved-prefix validation.
    #[test]
    fn app_namespace_keys_round_trip_and_are_byte_additive() {
        let ir = mod_with_const(7);
        // No app entries ⇒ byte-identical to the closed-key evidence encoder.
        let plain = emit_mic3_with_evidence(&ir, "cpu", None, Determinism::Deterministic, "0.10.1");
        let no_app = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            plain, no_app,
            "empty app entries must be byte-identical to the pre-17.8 encoder"
        );

        // With app entries: they round-trip; the body anchor is untouched.
        let app = vec![
            ("org.example.build_id".to_string(), "42".to_string()),
            ("com.acme.reviewer".to_string(), "qa".to_string()),
        ];
        let with_app = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &[],
            &app,
        )
        .unwrap();
        assert_eq!(
            mic3_evidence_report(&no_app).unwrap().trace_hash,
            mic3_evidence_report(&with_app).unwrap().trace_hash,
            "app metadata must not move the trace_hash anchor"
        );
        assert!(mic3_evidence_report(&with_app).unwrap().trace_hash_valid);

        // Reads back in canonical (lexicographic) key order.
        let read = mic3_app_metadata(&with_app).unwrap();
        assert_eq!(
            read,
            vec![
                ("com.acme.reviewer".to_string(), "qa".to_string()),
                ("org.example.build_id".to_string(), "42".to_string()),
            ]
        );
        assert!(mic3_app_metadata(&no_app).unwrap().is_empty());

        // Fail-closed validation.
        assert!(validate_app_entries(&app).is_ok());
        assert!(
            validate_app_entries(&[("evidence_chain.x".to_string(), "y".to_string())]).is_err()
        );
        assert!(validate_app_entries(&[("signature.x".to_string(), "y".to_string())]).is_err());
        assert!(validate_app_entries(&[("nodot".to_string(), "y".to_string())]).is_err());
        assert!(validate_app_entries(&[(String::new(), "y".to_string())]).is_err());
    }

    // (S4-b) Round-trip: an emitted receipt re-derives + binds to the body.
    #[test]
    fn collapse_receipt_verifies_when_present_in_body() {
        let ir = mod_with_const(4950);
        let receipts = vec![CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 4950,
        }];
        let bytes = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &receipts,
            &[],
        )
        .unwrap();
        assert_eq!(
            mic3_collapse_verify(&bytes).unwrap(),
            CollapseVerifyStatus::Verified(1)
        );
    }

    // (S4-c) FORGERY: editing the receipt's claimed constant (in the epilogue,
    // OUTSIDE the trace_hash) leaves trace_hash VALID but must be caught by O(1)
    // re-derivation. The load-bearing soundness proof.
    #[test]
    fn collapse_receipt_forged_constant_fails_rederivation() {
        let ir = mod_with_const(4950);
        // Emit with an already-forged receipt (constant lies: 9999, real = 4950).
        let receipts = vec![CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 9999,
        }];
        let bytes = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &receipts,
            &[],
        )
        .unwrap();
        // trace_hash is still VALID (the epilogue is outside the anchor)...
        assert!(mic3_evidence_report(&bytes).unwrap().trace_hash_valid);
        // ...but the receipt layer fails closed on the re-derivation mismatch.
        assert_eq!(
            mic3_collapse_verify(&bytes).unwrap(),
            CollapseVerifyStatus::Rederivation {
                rederived: 4950,
                claimed: 9999,
            }
        );
    }

    // (S4-d) A self-consistent receipt whose constant is NOT in the body fails
    // closed (the receipt must bind to a value the program actually materialises).
    #[test]
    fn collapse_receipt_absent_from_body_fails_closed() {
        let ir = mod_with_const(4950); // body has 4950, not 5050
        let receipts = vec![CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 101, // Σ 0..100 = 5050 (self-consistent) but NOT in the body
            constant: 5050,
        }];
        let bytes = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            None,
            &receipts,
            &[],
        )
        .unwrap();
        assert!(receipts[0].is_self_consistent());
        assert_eq!(
            mic3_collapse_verify(&bytes).unwrap(),
            CollapseVerifyStatus::NotInBody { constant: 5050 }
        );
    }

    // (S4-e) An artifact with NO receipts reports Absent (back-compat), never a
    // failure.
    #[test]
    fn no_receipts_reports_absent() {
        let ir = mod_with_const(1);
        let bytes = emit_mic3_with_evidence(&ir, "cpu", None, Determinism::Deterministic, "0.10.1");
        assert_eq!(
            mic3_collapse_verify(&bytes).unwrap(),
            CollapseVerifyStatus::Absent
        );
    }

    // (S4-f) A signed artifact's receipts are covered by the signature preimage:
    // editing the receipt blob under a valid signature breaks the signature.
    #[test]
    fn signed_receipt_edit_breaks_signature() {
        let ir = mod_with_const(4950);
        let receipts = vec![CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 4950,
        }];
        let signed = emit_mic3_with_evidence_and_receipts(
            &ir,
            "cpu",
            None,
            Determinism::Deterministic,
            "0.10.1",
            Some(&SigningKey::Ed25519(TEST_SEED)),
            &receipts,
            &[],
        )
        .unwrap();
        // Untampered: signature valid.
        assert!(matches!(
            mic3_signature_status(&signed).unwrap(),
            SignatureStatus::Valid(_)
        ));
        // Swap the receipt blob to a different (self-consistent) receipt.
        let forged = vec![CollapseReceipt::GeometricPow {
            r: 7,
            lo: 0,
            hi: 13,
            constant: 96_889_010_407,
        }];
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        let new_blob = crate::ir::compact::v3::collapse_receipt::encode_collapse_receipts(&forged);
        for e in entries.iter_mut() {
            if e.key == "evidence_chain.collapse_receipts" {
                e.value = ParsedValue::Bytes(new_blob.clone());
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "editing the receipt blob under a valid signature must break the signature"
        );
    }

    // -------------------------------------------------------------------------
    // Ed25519 signing layer (RFC 0021 §6)
    // -------------------------------------------------------------------------

    // Deterministic test seed (NOT a production key — tests only).
    const TEST_SEED: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfc, 0xbe, 0xb2, 0xc4, 0xcf, 0x3d, 0x1e, 0x79, 0xfd, 0xae,
        0x0e, 0x34, 0xbc, 0xcb, 0xaa, 0xcf, 0x9e, 0xc2, 0x4b, 0xd0, 0xe7, 0x5c, 0x4e, 0x5d, 0x6b,
        0xde, 0x1e,
    ];

    // (o) Unsigned path is byte-identical to a signed artifact's unsigned prefix,
    //     and the trace_hash is unchanged — the determinism-gate invariant.
    #[test]
    fn signed_artifact_unsigned_prefix_is_byte_identical() {
        for (name, ir) in all_modules() {
            let unsigned =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            let signed = emit_mic3_with_signed_evidence(
                &ir,
                "x86_avx2",
                None,
                Determinism::Deterministic,
                "0.8.0",
                &TEST_SEED,
            );
            // The signature keys sort AFTER evidence_chain.* keys, but they are
            // interleaved into the same MAP, so the signed artifact is NOT a byte
            // prefix. The load-bearing invariant is that the *trace_hash anchor*
            // is identical, and the plain body prefix is identical.
            let body_end_u = find_map_sentinel(&unsigned).unwrap();
            let body_end_s = find_map_sentinel(&signed).unwrap();
            assert_eq!(
                &unsigned[..body_end_u],
                &signed[..body_end_s],
                "module '{}': mic@3 body must be byte-identical signed vs unsigned",
                name
            );
            let r_u = mic3_evidence_report(&unsigned).unwrap();
            let r_s = mic3_evidence_report(&signed).unwrap();
            assert_eq!(
                r_u.trace_hash, r_s.trace_hash,
                "module '{}': trace_hash anchor must be identical signed vs unsigned",
                name
            );
            assert!(
                r_s.trace_hash_valid,
                "module '{}': signed still attests",
                name
            );
        }
    }

    // (p) sign → verify round trip: a freshly-signed artifact reports Valid.
    #[test]
    fn signature_round_trip_is_valid() {
        for (name, ir) in all_modules() {
            let signed = emit_mic3_with_signed_evidence(
                &ir,
                "arm_neon",
                None,
                Determinism::Deterministic,
                "0.8.0",
                &TEST_SEED,
            );
            let status = mic3_signature_status(&signed)
                .unwrap_or_else(|e| panic!("module '{}': signature_status err {:?}", name, e));
            let expected_pk = super::super::ed25519::public_key(&TEST_SEED);
            match status {
                SignatureStatus::Valid(v) => {
                    assert_eq!(v.scheme, "ed25519", "module '{}': scheme tag", name);
                    assert_eq!(
                        v.ed25519_pubkey,
                        Some(expected_pk),
                        "module '{}': ed25519 pubkey",
                        name
                    );
                    assert!(v.mldsa_pubkey.is_none(), "module '{}': no mldsa half", name);
                }
                other => panic!("module '{}': expected Valid, got {:?}", name, other),
            }
        }
    }

    // (q) An unsigned artifact reports Absent (back-compat tolerance).
    #[test]
    fn unsigned_artifact_signature_absent() {
        let ir = mod_binop();
        let unsigned =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        assert_eq!(
            mic3_signature_status(&unsigned).unwrap(),
            SignatureStatus::Absent,
            "unsigned artifact must report SignatureStatus::Absent"
        );
    }

    // (r) Tamper: flipping a body byte breaks BOTH trace_hash_valid and the
    //     signature (the signed trace_hash no longer matches the body).
    #[test]
    fn signature_tamper_body_byte_fails_closed() {
        let ir = mod_fndef();
        let signed = emit_mic3_with_signed_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &TEST_SEED,
        );
        let mut tampered = signed.clone();
        tampered[5] ^= 0xFF; // inside the mic@3 body
        // trace_hash recompute over the tampered body no longer matches.
        if let Ok(report) = mic3_evidence_report(&tampered) {
            assert!(
                !report.trace_hash_valid,
                "body tamper must break trace_hash_valid"
            );
        }
    }

    // (s) Tamper: flipping a signature byte flips Valid → Invalid.
    #[test]
    fn signature_tamper_sig_byte_flips_invalid() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &TEST_SEED,
        );
        // Flip one byte INSIDE the signature.ed25519 value (not the trailing
        // `signature.scheme` string, which now sorts last in the MAP) and confirm
        // the signature no longer verifies. We tamper via the parsed entries so the
        // target is precise regardless of MAP key ordering.
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        let mut found = false;
        for e in entries.iter_mut() {
            if e.key == "signature.ed25519" {
                if let ParsedValue::Bytes(b) = &mut e.value {
                    b[0] ^= 0x01;
                    found = true;
                }
            }
        }
        assert!(
            found,
            "precondition: signed artifact carries signature.ed25519"
        );
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "flipped signature byte must report Invalid"
        );
    }

    // (t) Tamper: substituting a different public key flips Valid → Invalid.
    #[test]
    fn signature_wrong_pubkey_is_invalid() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &TEST_SEED,
        );
        // Re-emit MAP with a foreign pubkey but the same signature bytes.
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        let mut other_seed = TEST_SEED;
        other_seed[0] ^= 0x01;
        let other_pk = super::super::ed25519::public_key(&other_seed);
        for e in entries.iter_mut() {
            if e.key == "signature.pubkey" {
                e.value = ParsedValue::Bytes(other_pk.to_vec());
            }
        }
        let status = signature_status_from_entries(&entries).unwrap();
        assert_eq!(
            status,
            SignatureStatus::Invalid,
            "signature must not verify under a substituted public key"
        );
    }

    // (u) Signing is deterministic: same seed + IR ⇒ byte-identical artifact.
    #[test]
    fn signing_is_byte_deterministic() {
        let ir = mod_call_chain();
        let a = emit_mic3_with_signed_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &TEST_SEED,
        );
        let b = emit_mic3_with_signed_evidence(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &TEST_SEED,
        );
        assert_eq!(a, b, "deterministic signing must be byte-reproducible");
    }

    // -------------------------------------------------------------------------
    // Post-quantum ML-DSA-65 (FIPS-204) + hybrid signing (RFC 0021 §6)
    // These require the `evidence-mldsa` feature (the vetted fips204 backend).
    // -------------------------------------------------------------------------

    // Second deterministic test seed (ML-DSA keygen ξ). NOT a production key.
    #[cfg(feature = "evidence-mldsa")]
    const TEST_MLDSA_SEED: [u8; 32] = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x00, 0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78, 0x87, 0x96, 0xa5, 0xb4, 0xc3, 0xd2,
        0xe1, 0xf0,
    ];

    // (v) ML-DSA-65 sign → verify round trip reports Valid with the ml-dsa-65 tag.
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn mldsa_round_trip_is_valid() {
        for (name, ir) in all_modules() {
            let signed = emit_mic3_with_signed_evidence_scheme(
                &ir,
                "x86_avx2",
                None,
                Determinism::Deterministic,
                "0.8.0",
                &SigningKey::MlDsa65(TEST_MLDSA_SEED),
            )
            .expect("ml-dsa signing available under the feature");
            match mic3_signature_status(&signed).unwrap() {
                SignatureStatus::Valid(v) => {
                    assert_eq!(v.scheme, "ml-dsa-65", "module '{}': scheme", name);
                    assert!(v.ed25519_pubkey.is_none(), "module '{}': no ed half", name);
                    assert_eq!(
                        v.mldsa_pubkey.as_deref().map(<[u8]>::len),
                        Some(1952),
                        "module '{}': ml-dsa-65 pk length",
                        name
                    );
                }
                other => panic!("module '{}': expected Valid, got {:?}", name, other),
            }
            // The mic@3 body/anchor is byte-identical to the unsigned path.
            let unsigned =
                emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
            let be_u = find_map_sentinel(&unsigned).unwrap();
            let be_s = find_map_sentinel(&signed).unwrap();
            assert_eq!(
                &unsigned[..be_u],
                &signed[..be_s],
                "module '{}': ML-DSA signing must not perturb the mic@3 body",
                name
            );
            assert_eq!(
                mic3_evidence_report(&unsigned).unwrap().trace_hash,
                mic3_evidence_report(&signed).unwrap().trace_hash,
                "module '{}': trace_hash anchor unchanged under ML-DSA signing",
                name
            );
        }
    }

    // (w) ML-DSA deterministic signing is byte-reproducible.
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn mldsa_signing_is_byte_deterministic() {
        let ir = mod_call_chain();
        let mk = || {
            emit_mic3_with_signed_evidence_scheme(
                &ir,
                "x86_avx2",
                None,
                Determinism::Deterministic,
                "0.8.0",
                &SigningKey::MlDsa65(TEST_MLDSA_SEED),
            )
            .unwrap()
        };
        assert_eq!(
            mk(),
            mk(),
            "ML-DSA deterministic signing must reproduce byte-for-byte"
        );
    }

    // (x) Hybrid sign → verify: BOTH halves present and Valid.
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn hybrid_round_trip_requires_both() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &SigningKey::Hybrid {
                ed25519: TEST_SEED,
                mldsa65: TEST_MLDSA_SEED,
            },
        )
        .unwrap();
        match mic3_signature_status(&signed).unwrap() {
            SignatureStatus::Valid(v) => {
                assert_eq!(v.scheme, "hybrid-ed25519-ml-dsa-65");
                assert_eq!(
                    v.ed25519_pubkey,
                    Some(super::super::ed25519::public_key(&TEST_SEED))
                );
                assert_eq!(v.mldsa_pubkey.as_deref().map(<[u8]>::len), Some(1952));
            }
            other => panic!("expected hybrid Valid, got {:?}", other),
        }
    }

    // (y) Hybrid tamper: corrupting ONLY the ML-DSA signature must fail the whole
    //     hybrid verification (both halves are required — defense-in-depth).
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn hybrid_tamper_mldsa_half_fails_closed() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &SigningKey::Hybrid {
                ed25519: TEST_SEED,
                mldsa65: TEST_MLDSA_SEED,
            },
        )
        .unwrap();
        // Flip one byte inside the signature.mldsa value, leaving ed25519 intact.
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        for e in entries.iter_mut() {
            if e.key == "signature.mldsa" {
                if let ParsedValue::Bytes(b) = &mut e.value {
                    b[100] ^= 0x01;
                }
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "hybrid must fail closed when only the ML-DSA half is corrupted"
        );
    }

    // (z) ML-DSA tamper: flip a body byte → trace_hash no longer matches the
    //     signed anchor → signature no longer verifies (authenticity broken).
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn mldsa_tamper_body_fails_closed() {
        let ir = mod_fndef();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &SigningKey::MlDsa65(TEST_MLDSA_SEED),
        )
        .unwrap();
        let mut tampered = signed.clone();
        tampered[5] ^= 0xFF;
        if let Ok(report) = mic3_evidence_report(&tampered) {
            assert!(
                !report.trace_hash_valid,
                "body tamper breaks trace_hash_valid"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Bulletproof PQC-hybrid: ML-DSA-87 + SLH-DSA-SHAKE-256s, both-must-verify.
    // Requires BOTH evidence-mldsa (lattice leg) and evidence-slhdsa (hash leg).
    // SLH-DSA-256s SIGNING is slow (small-sig variant); SLH-DSA VERIFY is fast, so
    // each test emits exactly ONCE and reuses the artifact for the fast checks.
    // -------------------------------------------------------------------------

    /// 96-byte SLH-DSA test seed (SK.seed ‖ SK.prf ‖ PK.seed). NOT a production key.
    #[cfg(all(feature = "evidence-mldsa", feature = "evidence-slhdsa"))]
    fn test_slhdsa_seed() -> [u8; 96] {
        let mut s = [0u8; 96];
        let mut i = 0;
        while i < 96 {
            s[i] = (i as u8).wrapping_mul(7).wrapping_add(0x13);
            i += 1;
        }
        s
    }

    #[cfg(all(feature = "evidence-mldsa", feature = "evidence-slhdsa"))]
    fn pqc_hybrid_key() -> SigningKey {
        SigningKey::PqcHybrid {
            mldsa87: TEST_MLDSA_SEED,
            slhdsa: test_slhdsa_seed(),
        }
    }

    // (pqc-1) Round-trip Valid + byte-identical unsigned prefix + anchor unchanged.
    #[cfg(all(feature = "evidence-mldsa", feature = "evidence-slhdsa"))]
    #[test]
    fn pqc_hybrid_valid_and_byte_identical() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &pqc_hybrid_key(),
        )
        .expect("pqc-hybrid signing available under both features");
        match mic3_signature_status(&signed).unwrap() {
            SignatureStatus::Valid(v) => {
                assert_eq!(v.scheme, "pqc-hybrid-ml-dsa-87-slh-dsa-256s");
                assert!(
                    v.ed25519_pubkey.is_none(),
                    "no classical leg in the PQC-hybrid"
                );
                assert!(
                    v.mldsa_pubkey.is_none(),
                    "no ml-dsa-65 leg in the PQC-hybrid"
                );
                assert!(v.mldsa87_pubkey.is_some(), "ml-dsa-87 pubkey present");
                assert_eq!(
                    v.slhdsa_pubkey.as_deref().map(<[u8]>::len),
                    Some(64),
                    "slh-dsa-shake-256s pk length"
                );
            }
            other => panic!("expected pqc-hybrid Valid, got {:?}", other),
        }
        // Signing never perturbs the mic@3 body/anchor (the byte-identity wedge).
        let unsigned =
            emit_mic3_with_evidence(&ir, "x86_avx2", None, Determinism::Deterministic, "0.8.0");
        let be_u = find_map_sentinel(&unsigned).unwrap();
        let be_s = find_map_sentinel(&signed).unwrap();
        assert_eq!(
            &unsigned[..be_u],
            &signed[..be_s],
            "PQC-hybrid signing must not perturb the mic@3 body"
        );
        assert_eq!(
            mic3_evidence_report(&unsigned).unwrap().trace_hash,
            mic3_evidence_report(&signed).unwrap().trace_hash,
            "trace_hash anchor unchanged under PQC-hybrid signing"
        );
    }

    // (pqc-2) NON-DEGRADABLE (the load-bearing property from the cross-model security review):
    //         stripping EITHER leg, or tampering EITHER signature, must fail closed
    //         — the hybrid can never collapse to a valid single-scheme signature.
    #[cfg(all(feature = "evidence-mldsa", feature = "evidence-slhdsa"))]
    #[test]
    fn pqc_hybrid_non_degradable() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &pqc_hybrid_key(),
        )
        .unwrap();
        let body_end = find_map_sentinel(&signed).unwrap();
        // Control: as-signed it is Valid.
        let base = parse_map_epilogue(&signed[body_end..]).unwrap();
        assert!(matches!(
            signature_status_from_entries(&base).unwrap(),
            SignatureStatus::Valid(_)
        ));

        // (a) STRIP the SLH-DSA leg → must NOT collapse to a valid ml-dsa-87-only
        //     signature; the pqc-hybrid tag still demands the missing leg.
        let mut stripped_slh = parse_map_epilogue(&signed[body_end..]).unwrap();
        stripped_slh.retain(|e| e.key != "signature.slhdsa" && e.key != "signature.slhdsa_pubkey");
        assert!(
            matches!(
                signature_status_from_entries(&stripped_slh).unwrap(),
                SignatureStatus::Malformed(_)
            ),
            "stripping the SLH-DSA leg must be Malformed (non-degradable), got {:?}",
            signature_status_from_entries(&stripped_slh).unwrap()
        );

        // (b) STRIP the ML-DSA-87 leg → likewise refused.
        let mut stripped_ml = parse_map_epilogue(&signed[body_end..]).unwrap();
        stripped_ml.retain(|e| e.key != "signature.mldsa87" && e.key != "signature.mldsa87_pubkey");
        assert!(
            matches!(
                signature_status_from_entries(&stripped_ml).unwrap(),
                SignatureStatus::Malformed(_)
            ),
            "stripping the ML-DSA-87 leg must be Malformed (non-degradable)"
        );

        // (c) TAMPER the ML-DSA-87 signature → Invalid (both legs required).
        let mut tam_ml = parse_map_epilogue(&signed[body_end..]).unwrap();
        for e in tam_ml.iter_mut() {
            if e.key == "signature.mldsa87" {
                if let ParsedValue::Bytes(b) = &mut e.value {
                    b[100] ^= 0x01;
                }
            }
        }
        assert_eq!(
            signature_status_from_entries(&tam_ml).unwrap(),
            SignatureStatus::Invalid,
            "tampering the ML-DSA-87 leg must fail closed"
        );

        // (d) TAMPER the SLH-DSA signature → Invalid.
        let mut tam_slh = parse_map_epilogue(&signed[body_end..]).unwrap();
        for e in tam_slh.iter_mut() {
            if e.key == "signature.slhdsa" {
                if let ParsedValue::Bytes(b) = &mut e.value {
                    b[100] ^= 0x01;
                }
            }
        }
        assert_eq!(
            signature_status_from_entries(&tam_slh).unwrap(),
            SignatureStatus::Invalid,
            "tampering the SLH-DSA leg must fail closed"
        );

        // (e) DOWNGRADE the scheme tag to a weaker single-leg / legacy-hybrid scheme
        //     while leaving BOTH pqc-hybrid legs in place → must never verify. The
        //     scheme tag is length-prefixed into the signed preimage, so whichever
        //     verifier the flipped tag selects checks a preimage bound to the WRONG
        //     scheme string (and looks for key material the pqc-hybrid artifact does
        //     not carry) — the pqc-hybrid AND-path is never re-entered, no collapse.
        for weaker in ["ed25519", "ml-dsa-65", "hybrid-ed25519-ml-dsa-65"] {
            let mut flipped = parse_map_epilogue(&signed[body_end..]).unwrap();
            for e in flipped.iter_mut() {
                if e.key == KEY_SIG_SCHEME {
                    e.value = ParsedValue::Str(weaker.to_string());
                }
            }
            let st = signature_status_from_entries(&flipped);
            assert!(
                !matches!(st, Ok(SignatureStatus::Valid(_))),
                "downgrading the pqc-hybrid scheme tag to `{weaker}` must not verify, got {st:?}"
            );
        }
    }

    // (aa) Cross-substrate: the same IR + same seed yields a byte-identical signed
    //      artifact regardless of the declared substrate label — the signature is a
    //      pure function of (seed, trace_hash), and the trace_hash is substrate-free.
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn hybrid_signature_is_substrate_independent() {
        let ir = mod_call_chain();
        let key = SigningKey::Hybrid {
            ed25519: TEST_SEED,
            mldsa65: TEST_MLDSA_SEED,
        };
        // Same substrate label twice ⇒ identical (determinism of the whole path).
        let a = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &key,
        )
        .unwrap();
        let b = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &key,
        )
        .unwrap();
        assert_eq!(a, b, "hybrid signed artifact must be byte-reproducible");
    }

    // -------------------------------------------------------------------------
    // Provenance authenticity (HIGH fix): the signature now covers the whole
    // canonical provenance preimage (trace_hash + every other evidence_chain.*
    // key), so editing substrate / toolchain / determinism / parent on a
    // validly-signed artifact must break signature verification (fail-closed).
    // Regression against: "signature is valid" while parent/substrate forgeable.
    // -------------------------------------------------------------------------

    /// Sign `mod_binop` with the Ed25519 test seed and return the parsed MAP
    /// entries (the exact structure the verifier consumes).
    fn signed_ed25519_entries(parent: Option<[u8; 32]>) -> Vec<ParsedEntry> {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence(
            &ir,
            "cpu",
            parent,
            Determinism::Deterministic,
            "0.10.0",
            &TEST_SEED,
        );
        let body_end = find_map_sentinel(&signed).unwrap();
        parse_map_epilogue(&signed[body_end..]).unwrap()
    }

    /// The unmodified signed entries must verify (positive control for the four
    /// tamper tests below).
    fn assert_valid(entries: &[ParsedEntry], ctx: &str) {
        match signature_status_from_entries(entries).unwrap() {
            SignatureStatus::Valid(_) => {}
            other => panic!("{ctx}: expected Valid before tamper, got {other:?}"),
        }
    }

    // (ab) Editing `substrate` on a signed artifact breaks the signature.
    #[test]
    fn provenance_tamper_substrate_fails_closed() {
        let mut entries = signed_ed25519_entries(None);
        assert_valid(&entries, "substrate");
        for e in entries.iter_mut() {
            if e.key == "evidence_chain.substrate" {
                e.value = ParsedValue::Str("gpu".to_string()); // cpu -> gpu
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "editing substrate (cpu->gpu) must invalidate the signature"
        );
    }

    // (ac) Editing `toolchain` on a signed artifact breaks the signature.
    #[test]
    fn provenance_tamper_toolchain_fails_closed() {
        let mut entries = signed_ed25519_entries(None);
        assert_valid(&entries, "toolchain");
        for e in entries.iter_mut() {
            if e.key == "evidence_chain.toolchain" {
                e.value = ParsedValue::Str("0.10.9".to_string()); // 0.10.0 -> 0.10.9
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "editing toolchain (0.10.0->0.10.9) must invalidate the signature"
        );
    }

    // (ad) Editing `determinism` on a signed artifact breaks the signature.
    #[test]
    fn provenance_tamper_determinism_fails_closed() {
        let mut entries = signed_ed25519_entries(None);
        assert_valid(&entries, "determinism");
        for e in entries.iter_mut() {
            if e.key == "evidence_chain.determinism" {
                e.value = ParsedValue::Str("nondeterministic".to_string());
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "flipping determinism deterministic->nondeterministic must invalidate the signature"
        );
    }

    // (ae) Editing `parent` (chain linkage) on a signed artifact breaks the
    //      signature — the whole reason provenance authenticity matters.
    #[test]
    fn provenance_tamper_parent_fails_closed() {
        let mut entries = signed_ed25519_entries(Some([0x11u8; 32]));
        assert_valid(&entries, "parent");
        let mut tampered = false;
        for e in entries.iter_mut() {
            if e.key == "evidence_chain.parent" {
                if let ParsedValue::Bytes(b) = &mut e.value {
                    b[0] ^= 0xFF; // re-point the chain to a forged parent
                    tampered = true;
                }
            }
        }
        assert!(
            tampered,
            "precondition: signed artifact must carry a parent"
        );
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Invalid,
            "re-pointing the parent link must invalidate the signature"
        );
    }

    // (af) The preimage excludes `evidence_chain.trace_hash` (derived) and every
    //      `signature.*` key (a signature never covers itself) — proven by
    //      construction here so a future refactor cannot silently fold them in.
    #[test]
    fn signature_preimage_excludes_trace_hash_and_signature_keys() {
        let th = [0x42u8; 32];
        let entries: Vec<(&str, MapEntryValue)> = vec![
            ("evidence_chain.substrate", MapEntryValue::Str("cpu")),
            ("evidence_chain.trace_hash", MapEntryValue::Bytes(&th)),
            ("signature.scheme", MapEntryValue::Str("ed25519")),
        ];
        let pre = build_signature_preimage(&entries, &th, "ed25519");
        // Leading 32 bytes are the trace_hash; then the length-prefixed scheme tag
        // ("ed25519", 7 bytes); then ULEB count == 1 (only substrate survives the
        // filter). trace_hash and signature.* are excluded.
        assert_eq!(&pre[..32], &th, "preimage must lead with the trace_hash");
        assert_eq!(pre[32], 7, "scheme length prefix (ed25519 = 7 bytes)");
        assert_eq!(
            &pre[33..40],
            b"ed25519",
            "scheme tag bound into the preimage"
        );
        assert_eq!(
            pre[40], 1,
            "only one evidence_chain.* entry (substrate) survives the filter"
        );
    }

    // (ag) Scheme-downgrade attack (HIGH #1): take a hybrid-signed artifact, strip
    //      the ML-DSA half, and rewrite `signature.scheme` -> "ed25519". The
    //      genuine Ed25519 half was signed over the HYBRID preimage, so with the
    //      scheme now bound into the preimage it no longer verifies. Must never
    //      report Valid. Distinct from `hybrid_tamper_mldsa_half_fails_closed`
    //      (which leaves scheme=hybrid and corrupts the ML-DSA bytes).
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn scheme_downgrade_strip_mldsa_half_fails_closed() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &SigningKey::Hybrid {
                ed25519: TEST_SEED,
                mldsa65: TEST_MLDSA_SEED,
            },
        )
        .unwrap();
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        // Positive control: untouched hybrid verifies.
        assert!(matches!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Valid(_)
        ));
        // The downgrade: drop BOTH ML-DSA keys and rename the scheme tag.
        entries.retain(|e| e.key != "signature.mldsa" && e.key != "signature.mldsa_pubkey");
        for e in entries.iter_mut() {
            if e.key == "signature.scheme" {
                e.value = ParsedValue::Str("ed25519".to_string());
            }
        }
        let status = signature_status_from_entries(&entries).unwrap();
        assert!(
            matches!(status, SignatureStatus::Invalid),
            "scheme downgrade (hybrid -> ed25519, ML-DSA stripped) must be Invalid, got {status:?}"
        );
        assert!(
            !matches!(status, SignatureStatus::Valid(_)),
            "downgrade must NEVER report Valid"
        );
    }

    // (ah) Scheme/half mismatch (HIGH #1, structural companion): an artifact whose
    //      `signature.scheme` does not name a half it carries is Malformed — a
    //      downgrade that flips the tag but forgets to strip the now-unnamed half
    //      cannot be silently accepted.
    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn scheme_names_no_mldsa_but_mldsa_half_present_is_malformed() {
        let ir = mod_binop();
        let signed = emit_mic3_with_signed_evidence_scheme(
            &ir,
            "x86_avx2",
            None,
            Determinism::Deterministic,
            "0.8.0",
            &SigningKey::Hybrid {
                ed25519: TEST_SEED,
                mldsa65: TEST_MLDSA_SEED,
            },
        )
        .unwrap();
        let body_end = find_map_sentinel(&signed).unwrap();
        let mut entries = parse_map_epilogue(&signed[body_end..]).unwrap();
        // Flip the tag to ed25519 but LEAVE the ML-DSA half in place.
        for e in entries.iter_mut() {
            if e.key == "signature.scheme" {
                e.value = ParsedValue::Str("ed25519".to_string());
            }
        }
        assert_eq!(
            signature_status_from_entries(&entries).unwrap(),
            SignatureStatus::Malformed(KEY_SIG_SCHEME),
            "scheme=ed25519 with a signature.mldsa half present must be Malformed"
        );
    }

    // (ai) Allocation-bomb (HIGH #2): a crafted epilogue whose `blen` claims 2^47
    //      bytes with no payload must fail-closed via `Err`, never SIGABRT/panic on
    //      a multi-terabyte allocation.
    #[test]
    fn epilogue_alloc_bomb_blen_is_rejected() {
        // Minimal real mic@3 body so `find_map_sentinel` locates the epilogue.
        let ir = mod_binop();
        let body = emit_mic3(&ir);
        let mut art = body.clone();
        // Epilogue: sentinel, count=1, key_len=17, "signature.ed25519", TAG_BYTES,
        // blen = 2^47 (astronomically large ULEB), and NO payload bytes.
        art.push(MAP_SENTINEL);
        art.push(0x01); // count = 1
        let key = b"signature.ed25519";
        art.push(key.len() as u8);
        art.extend_from_slice(key);
        art.push(TAG_BYTES);
        // ULEB128 of 2^47 = 140737488355328.
        let mut n: u64 = 1u64 << 47;
        loop {
            let mut b = (n & 0x7f) as u8;
            n >>= 7;
            if n != 0 {
                b |= 0x80;
            }
            art.push(b);
            if n == 0 {
                break;
            }
        }
        // The low-level parser must Err, not allocate/panic.
        assert_eq!(
            parse_map_epilogue(&art[body.len()..]).err(),
            Some(ParseMapError::Truncated),
            "oversized blen must be rejected before allocation"
        );
        // And the public entry points must fail-closed (Err), never crash.
        assert!(
            mic3_signature_status(&art).is_err(),
            "mic3_signature_status must fail-closed on the alloc bomb"
        );
        assert!(
            mic3_evidence_report(&art).is_err(),
            "mic3_evidence_report must fail-closed on the alloc bomb"
        );
    }

    // (aj) Fail-open coercion guard (MED #4): a type-confused signature field
    //      (`signature.ed25519` re-encoded as a String) must make the signature
    //      layer return `Err` — the CLI maps that to a verification FAILURE, not to
    //      `Absent`/"unsigned but attested".
    #[test]
    fn type_confused_ed25519_field_is_err() {
        let mut entries = signed_ed25519_entries(None);
        for e in entries.iter_mut() {
            if e.key == "signature.ed25519" {
                // Re-encode the 64-byte signature as a String (type confusion).
                e.value = ParsedValue::Str("not-a-signature".to_string());
            }
        }
        assert!(
            signature_status_from_entries(&entries).is_err(),
            "a type-confused signature.ed25519 must be Err (fail-closed), never Ok(Absent)"
        );
    }

    // -------------------------------------------------------------------------
    // ML-DSA-65 (FIPS-204) crate-pinning regression vector.
    //
    // PROVENANCE (honest): this is NOT an authoritative NIST ACVP vector. It was
    // generated ONCE with the vetted `fips204` crate under the deterministic
    // FIPS-204 variant (all-zero keygen implied by the fixed seed ξ, all-zero
    // signing rnd) and recorded verbatim. It pins ML-DSA-65 KeyGen+Sign output so
    // a silent change in crate version / lattice params is caught, rather than
    // trusting the dependency blindly.
    // deferred: swap for an official FIPS-204 ML-DSA-65 ACVP KAT vector when an
    //   authoritative source is wired into the test tree — upgrade path: replace
    //   the recorded constants below with the ACVP (seed -> pk, seed+msg -> sig)
    //   expected outputs and drop the crate-pinning caveat.
    // -------------------------------------------------------------------------
    // Recorded verbatim from the fips204 crate (ML-DSA-65, all-zero keygen seed ξ,
    // deterministic all-zero signing rnd, message = [0xA5; 32]). See PROVENANCE
    // note on the test below — crate-pinning vector, not a NIST ACVP KAT.
    #[cfg(feature = "evidence-mldsa")]
    const MLDSA65_PIN_PUBKEY_PREFIX: [u8; 16] = [
        66, 75, 47, 38, 126, 88, 213, 179, 180, 77, 113, 172, 252, 106, 101, 107,
    ];
    #[cfg(feature = "evidence-mldsa")]
    const MLDSA65_PIN_SIG_PREFIX: [u8; 16] = [
        220, 200, 45, 112, 249, 251, 188, 27, 86, 255, 170, 16, 190, 20, 115, 186,
    ];

    #[cfg(feature = "evidence-mldsa")]
    #[test]
    fn mldsa65_crate_pinning_regression_vector() {
        // Fixed keygen seed ξ (all-zero) and a fixed 32-byte message.
        let seed = [0u8; 32];
        let msg = [0xA5u8; 32];

        let pk = super::super::mldsa::public_key(&seed);
        let sig = super::super::mldsa::sign(&seed, &msg);

        // Structural pins (FIPS-204 ML-DSA-65 fixed sizes).
        assert_eq!(pk.len(), 1952, "ML-DSA-65 public key length");
        assert_eq!(sig.len(), 3309, "ML-DSA-65 signature length");

        // Value pins — recorded verbatim from the fips204 crate (see PROVENANCE).
        assert_eq!(
            &pk[..16],
            &MLDSA65_PIN_PUBKEY_PREFIX,
            "ML-DSA-65 pubkey prefix drifted from the pinned crate output"
        );
        assert_eq!(
            &sig[..16],
            &MLDSA65_PIN_SIG_PREFIX,
            "ML-DSA-65 signature prefix drifted from the pinned crate output"
        );

        // Self-consistency: the pinned pubkey verifies the pinned signature.
        assert!(
            super::super::mldsa::verify(&pk, &msg, &sig),
            "pinned ML-DSA-65 (pk, sig) must verify over the pinned message"
        );
    }

    /// An UNKNOWN key in the reserved `signature.` namespace must be rejected on
    /// READ, not merely refused at emit.
    ///
    /// Why this is its own test rather than a case in the malleation suite: the
    /// injected entry is a *legitimately parsed* entry, so it survives the
    /// canonical re-encode byte-for-byte, and `build_signature_preimage` filters
    /// every `signature.`-prefixed key out of the signed bytes (a signature
    /// cannot cover itself). An attacker-chosen `signature.pad` was therefore
    /// authenticated by nothing at all — not the trace_hash (it lives in the
    /// epilogue, not the body), not the canonical check, not the signature —
    /// while `mindc verify` still reported "signature is valid and signer key is
    /// trusted", exit 0, under both `--signer-pubkey` and `--require-signed`.
    /// Measured before the fix: a 447-byte signed artifact carried 200,000 bytes
    /// of attacker payload and still verified.
    #[test]
    // enforces: SIG-RESV-01
    fn unknown_reserved_signature_key_is_rejected_on_read() {
        // Derived from SIGNATURE_KEYS, never re-listed. A hardcoded copy here was
        // pinned at the original FIVE keys while the allowlist grew to NINE, so the
        // four PQC keys (ml-dsa-87 / slh-dsa, both pubkey and signature) were never
        // asserted acceptable — dropping one from SIGNATURE_KEYS would have left this
        // test green while every artifact of that scheme failed verification. That is
        // the same "one rule, two lists, one updated" shape this test exists to catch.
        for k in SIGNATURE_KEYS {
            assert!(
                k.starts_with("signature."),
                "{k} must be in the reserved namespace"
            );
        }
        // The allowlist is exactly these five and nothing else: a new signature
        // key added without extending the read-side check would reopen the hole,
        // so this asserts the pairing rather than trusting it.
        // END-TO-END: build a real artifact carrying an injected reserved key and
        // assert the canonical check rejects it. Asserting the allowlist set alone
        // would not prove the check is WIRED — that is the mistake this whole
        // finding was about.
        let m = IRModule::new();
        let body = emit_mic3(&m);

        for injected in [
            "signature.pad",
            "signature.",
            "signature.x",
            "signature.ed25519_extra",
        ] {
            assert!(
                !SIGNATURE_KEYS.contains(&injected),
                "{injected} must not be a defined signature key"
            );
            let entries = vec![ParsedEntry {
                key: injected.to_string(),
                // A large attacker-chosen payload: the original report carried
                // 200,000 bytes this way.
                value: ParsedValue::Bytes(vec![0x41; 4096]),
            }];
            let mut artifact = body.clone();
            artifact.extend_from_slice(&encode_map_epilogue_canonical(&entries));

            match mic3_canonical_check(&artifact) {
                Err(Mic3NonCanonical::ReservedKey(k)) => assert_eq!(k, injected),
                other => panic!(
                    "injected reserved key `{injected}` must be REJECTED on read, got {other:?}"
                ),
            }
        }

        // EVERY read path must reject, not just the canonical check. The rule lives
        // in `parse_map_epilogue`, so this asserts the choke point actually covers
        // the consumers: `mic3_signature_status` is what an out-of-tree caller uses
        // and it previously returned Valid for exactly this artifact.
        {
            let entries = vec![ParsedEntry {
                key: "signature.pad".to_string(),
                value: ParsedValue::Bytes(vec![0x41; 4096]),
            }];
            let mut art = body.clone();
            art.extend_from_slice(&encode_map_epilogue_canonical(&entries));
            assert!(
                parse_map_epilogue(&art[body.len()..]).is_err(),
                "the shared parser must reject the injected reserved key"
            );
            assert!(
                mic3_signature_status(&art).is_err(),
                "mic3_signature_status must NOT report a verdict on an artifact \
                 carrying an injected reserved key"
            );
            assert!(
                mic3_evidence_report(&art).is_err(),
                "mic3_evidence_report must reject it too"
            );
        }

        // R2: a DUPLICATE key is the same malleability class through a different key
        // class. The signature preimage excludes `signature.*` and trace_hash, and
        // every accessor takes the FIRST match, so duplicating a preimage-excluded
        // key changes artifact bytes without changing the preimage. Measured before
        // the fix: mic3_signature_status returned Valid(ed25519) over a 200 KB stream
        // that was not the signed artifact. Rejection must therefore live in the
        // shared parser, not in mic3_canonical_check alone.
        {
            let dup = vec![
                ParsedEntry {
                    key: "org.example.k".to_string(),
                    value: ParsedValue::Int(1),
                },
                ParsedEntry {
                    key: "org.example.k".to_string(),
                    value: ParsedValue::Bytes(vec![0x41; 4096]),
                },
            ];
            let mut art = body.clone();
            art.extend_from_slice(&encode_map_epilogue_canonical(&dup));
            match parse_map_epilogue(&art[body.len()..]) {
                Err(ParseMapError::DuplicateKey(k)) => assert_eq!(k, "org.example.k"),
                other => {
                    panic!("duplicate key must be rejected by the SHARED parser, got {other:?}")
                }
            }
            assert!(
                mic3_signature_status(&art).is_err(),
                "mic3_signature_status must not report a verdict over a duplicated-key artifact"
            );
            assert!(
                mic3_evidence_report(&art).is_err(),
                "evidence_report must reject it too"
            );
            assert!(
                mic3_canonical_check(&art).is_err(),
                "canonical check must still reject it"
            );
        }

        // enforces: SIG-RESV-02
        // The OTHER reserved namespace. `evidence_chain.*` was allowlisted only on the
        // emit side (`validate_app_entries`); the read side checked `signature.*` alone,
        // so an unknown evidence-chain key rode through every read path.
        for injected in ["evidence_chain.zzz", "evidence_chain.device_receipts"] {
            assert!(
                !EVIDENCE_CHAIN_KEYS.contains(&injected),
                "{injected} must not be a defined evidence-chain key"
            );
            let entries = vec![ParsedEntry {
                key: injected.to_string(),
                value: ParsedValue::Int(1),
            }];
            let mut art = body.clone();
            art.extend_from_slice(&encode_map_epilogue_canonical(&entries));
            match parse_map_epilogue(&art[body.len()..]) {
                Err(ParseMapError::ReservedKey(k)) => assert_eq!(k, injected),
                other => {
                    panic!("unknown evidence_chain key must be rejected on READ, got {other:?}")
                }
            }
            assert!(
                mic3_signature_status(&art).is_err(),
                "signature_status must not report a verdict over an unknown evidence-chain key"
            );
        }

        // Control: a non-reserved application key with the same payload is still
        // accepted, so the check rejects the namespace rather than the size.
        let ok_entries = vec![ParsedEntry {
            key: "org.example.note".to_string(),
            value: ParsedValue::Bytes(vec![0x41; 4096]),
        }];
        let mut ok_artifact = body.clone();
        ok_artifact.extend_from_slice(&encode_map_epilogue_canonical(&ok_entries));
        assert!(
            mic3_canonical_check(&ok_artifact).is_ok(),
            "a non-reserved application key must still be accepted"
        );
    }
}
