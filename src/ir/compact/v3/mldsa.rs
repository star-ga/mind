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

//! ML-DSA-65 (FIPS-204) — post-quantum signing for the evidence chain (RFC 0021 §6).
//!
//! # Why ML-DSA (compliance, not preference)
//!
//! Classical Ed25519 is NON-COMPLIANT for the federal PQC migration mandate
//! (EO 14412: ML-DSA by 2031; OMB M-26-15: crypto-agility required; the FAR PQC
//! rule pushes NIST PQC onto covered contractors by 2030). The evidence chain's
//! authenticity signature must therefore be a NIST PQC standard: **ML-DSA
//! (Dilithium), FIPS-204**. ML-DSA-65 is NIST security-strength category 3.
//!
//! # We do NOT reimplement the lattice math
//!
//! The FIPS-204 core (NTT over Z_q, rejection sampling, hint packing) is provided
//! by the vetted, pure-Rust `fips204` crate (integritychain/fips204), gated behind
//! the OPTIONAL `evidence-mldsa` cargo feature. When the feature is off, this
//! module compiles to fail-closed stubs so the evidence layer still builds with
//! its minimal (Ed25519-only) dependency surface, and the determinism/keystone
//! gate never compiles a PQC crate.
//!
//! # Determinism (the load-bearing MIND invariant)
//!
//! Both key derivation and signing are RNG-free and byte-reproducible:
//!   * `public_key`/keypair: `keygen_from_seed(ξ)` — the 32-byte operator seed ξ
//!     deterministically expands to the (pk, sk) pair (FIPS-204 §5.1 `ML-DSA.KeyGen_internal`).
//!   * `sign`: `try_sign_with_seed(&[0u8; 32], …)` — the all-zero rnd is the
//!     FIPS-204 *deterministic* signing variant (§5.2), so the same seed + message
//!     always yield byte-identical signature bytes on every substrate.
//!
//! The signature is metadata appended AFTER the mic@3 MAP sentinel; it never feeds
//! back into `trace_hash`, so unsigned artifacts stay byte-identical and the
//! keystone gate is untouched.

/// The `signature.scheme` tag value for a pure ML-DSA-65 signature.
pub const SCHEME: &str = "ml-dsa-65";

/// The `signature.scheme` tag value for a pure ML-DSA-87 signature (NIST security
/// category 5). This is the lattice leg of the bulletproof PQC-hybrid; a bare
/// ML-DSA-87 artifact uses this tag directly.
pub const SCHEME_87: &str = "ml-dsa-87";

/// Environment variable holding the 32-byte ML-DSA key-generation seed ξ as 64
/// hex chars. Never hardcode a key — the seed is supplied out-of-band by the
/// operator (compliance: OMB M-26-15 key hygiene).
pub const ENV_MLDSA_SEED: &str = "MIND_EVIDENCE_MLDSA_KEY";

/// Environment variable holding the 32-byte ML-DSA-87 key-generation seed ξ (64
/// hex chars) — the lattice leg of the bulletproof PQC-hybrid.
pub const ENV_MLDSA87_SEED: &str = "MIND_EVIDENCE_MLDSA87_KEY";

/// Is post-quantum ML-DSA signing compiled into this build?
///
/// `true` iff the `evidence-mldsa` feature is enabled. A verifier that reaches a
/// `ml-dsa-65`/`hybrid-*` artifact on a build where this is `false` must fail
/// closed (it cannot check the PQC half), never report `valid`.
pub const fn supported() -> bool {
    cfg!(feature = "evidence-mldsa")
}

#[cfg(feature = "evidence-mldsa")]
mod imp {
    use fips204::ml_dsa_65;
    use fips204::traits::{KeyGen, SerDes, Signer, Verifier};

    /// Public key length in bytes (FIPS-204 ML-DSA-65). Stable, exported so the
    /// verifier can size the byte-array without a magic literal.
    pub const PK_LEN: usize = ml_dsa_65::PK_LEN;
    /// Signature length in bytes (FIPS-204 ML-DSA-65).
    pub const SIG_LEN: usize = ml_dsa_65::SIG_LEN;

    /// Deterministically derive the ML-DSA-65 public key from the 32-byte seed ξ.
    pub fn public_key(seed: &[u8; 32]) -> Vec<u8> {
        let (pk, _sk) = ml_dsa_65::KG::keygen_from_seed(seed);
        pk.into_bytes().to_vec()
    }

    /// Deterministically sign `msg` (the 32-byte trace_hash) under the key derived
    /// from `seed`. Uses the FIPS-204 deterministic variant (all-zero rnd), empty
    /// context string. Byte-reproducible across substrates.
    pub fn sign(seed: &[u8; 32], msg: &[u8]) -> Vec<u8> {
        let (_pk, sk) = ml_dsa_65::KG::keygen_from_seed(seed);
        // try_sign_with_seed is infallible here: empty ctx (< 255 bytes) and a
        // fixed rnd seed — the only documented error causes cannot occur.
        let sig = sk
            .try_sign_with_seed(&[0u8; 32], msg, b"")
            .expect("ml-dsa-65 deterministic sign over 32-byte hash cannot fail");
        sig.to_vec()
    }

    /// Verify an ML-DSA-65 signature over `msg` under `pk_bytes`. Fail-closed on
    /// any length/parse error (returns `false`, never panics on attacker input).
    pub fn verify(pk_bytes: &[u8], msg: &[u8], sig_bytes: &[u8]) -> bool {
        let pk_arr: [u8; PK_LEN] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let pk = match ml_dsa_65::PublicKey::try_from_bytes(pk_arr) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let sig_arr: [u8; SIG_LEN] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        pk.verify(msg, &sig_arr, b"")
    }

    // ── ML-DSA-87 (NIST category 5) — the bulletproof-hybrid lattice leg ──────
    use fips204::ml_dsa_87;

    /// Public key length in bytes (FIPS-204 ML-DSA-87).
    pub const PK_LEN_87: usize = ml_dsa_87::PK_LEN;
    /// Signature length in bytes (FIPS-204 ML-DSA-87).
    pub const SIG_LEN_87: usize = ml_dsa_87::SIG_LEN;

    /// Deterministically derive the ML-DSA-87 public key from the 32-byte seed ξ.
    pub fn public_key_87(seed: &[u8; 32]) -> Vec<u8> {
        let (pk, _sk) = ml_dsa_87::KG::keygen_from_seed(seed);
        pk.into_bytes().to_vec()
    }

    /// Deterministically sign `msg` under the ML-DSA-87 key derived from `seed`.
    /// FIPS-204 deterministic variant (all-zero rnd), empty context — byte-
    /// reproducible across substrates.
    pub fn sign_87(seed: &[u8; 32], msg: &[u8]) -> Vec<u8> {
        let (_pk, sk) = ml_dsa_87::KG::keygen_from_seed(seed);
        let sig = sk
            .try_sign_with_seed(&[0u8; 32], msg, b"")
            .expect("ml-dsa-87 deterministic sign over a fixed message cannot fail");
        sig.to_vec()
    }

    /// Verify an ML-DSA-87 signature over `msg` under `pk_bytes`. Fail-closed on
    /// any length/parse error (returns `false`, never panics on attacker input).
    pub fn verify_87(pk_bytes: &[u8], msg: &[u8], sig_bytes: &[u8]) -> bool {
        let pk_arr: [u8; PK_LEN_87] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let pk = match ml_dsa_87::PublicKey::try_from_bytes(pk_arr) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let sig_arr: [u8; SIG_LEN_87] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        pk.verify(msg, &sig_arr, b"")
    }
}

#[cfg(not(feature = "evidence-mldsa"))]
mod imp {
    // Fail-closed stubs when ML-DSA support is not compiled in. `sign`/`public_key`
    // are never reached (the emit path checks `supported()` first and errors), and
    // `verify` always returns false so a verifier without PQC support can never
    // report a `ml-dsa-65`/`hybrid-*` artifact as valid.
    pub fn public_key(_seed: &[u8; 32]) -> Vec<u8> {
        unreachable!("ml-dsa public_key called without the evidence-mldsa feature")
    }
    pub fn sign(_seed: &[u8; 32], _msg: &[u8]) -> Vec<u8> {
        unreachable!("ml-dsa sign called without the evidence-mldsa feature")
    }
    pub fn verify(_pk_bytes: &[u8], _msg: &[u8], _sig_bytes: &[u8]) -> bool {
        false
    }
    // ML-DSA-87 stubs (same fail-closed contract as -65 above).
    pub fn public_key_87(_seed: &[u8; 32]) -> Vec<u8> {
        unreachable!("ml-dsa-87 public_key called without the evidence-mldsa feature")
    }
    pub fn sign_87(_seed: &[u8; 32], _msg: &[u8]) -> Vec<u8> {
        unreachable!("ml-dsa-87 sign called without the evidence-mldsa feature")
    }
    pub fn verify_87(_pk_bytes: &[u8], _msg: &[u8], _sig_bytes: &[u8]) -> bool {
        false
    }
}

pub use imp::{public_key, public_key_87, sign, sign_87, verify, verify_87};

#[cfg(all(test, feature = "evidence-mldsa"))]
mod tests_87 {
    use super::{public_key_87, sign_87, verify_87};

    fn seed() -> [u8; 32] {
        let mut s = [0u8; 32];
        for (i, b) in s.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(11).wrapping_add(5);
        }
        s
    }

    #[test]
    fn ml_dsa_87_round_trip() {
        let s = seed();
        let pk = public_key_87(&s);
        let msg = b"canonical provenance preimage";
        let sig = sign_87(&s, msg);
        assert!(
            verify_87(&pk, msg, &sig),
            "genuine ML-DSA-87 signature must verify"
        );
    }

    #[test]
    fn ml_dsa_87_deterministic() {
        let s = seed();
        let msg = b"determinism is the wedge";
        assert_eq!(
            sign_87(&s, msg),
            sign_87(&s, msg),
            "ML-DSA-87 sign must be byte-reproducible"
        );
        assert_eq!(public_key_87(&s), public_key_87(&s));
    }

    #[test]
    fn ml_dsa_87_tamper_and_wrong_key_fail() {
        let s = seed();
        let pk = public_key_87(&s);
        let msg = b"authenticity";
        let mut sig = sign_87(&s, msg);
        sig[0] ^= 0x01;
        assert!(
            !verify_87(&pk, msg, &sig),
            "tampered ML-DSA-87 signature must fail closed"
        );
        let good = sign_87(&s, msg);
        assert!(!verify_87(&pk, b"other", &good), "wrong message must fail");
        let mut s2 = seed();
        s2[0] ^= 0xFF;
        assert!(
            !verify_87(&public_key_87(&s2), msg, &good),
            "wrong key must fail closed"
        );
    }

    #[test]
    fn ml_dsa_87_malformed_fail_closed() {
        let s = seed();
        let pk = public_key_87(&s);
        let sig = sign_87(&s, b"m");
        assert!(!verify_87(&pk[..pk.len() - 1], b"m", &sig));
        assert!(!verify_87(&pk, b"m", &sig[..sig.len() - 1]));
        assert!(!verify_87(&[], b"m", &sig));
    }
}
