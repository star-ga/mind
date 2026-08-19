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

//! SLH-DSA-SHAKE-256s (FIPS-205) — the HASH-BASED leg of the bulletproof PQC-hybrid
//! evidence-chain signature (RFC 0016 Phase C, max-security profile).
//!
//! # Why a second, independent post-quantum foundation
//!
//! The lattice leg ([`super::mldsa`], ML-DSA-87 / FIPS-204) is quantum-safe and
//! standards-compliant on its own. SLH-DSA is added *alongside* it — never
//! instead — because its security reduces to nothing more than the security of
//! the underlying hash function (SHAKE-256), the single most conservative and
//! longest-studied assumption in cryptography. In an all-must-verify AND
//! composition, forging a STARGA release then requires breaking BOTH a
//! module-lattice scheme AND a hash-based scheme — two mathematically independent
//! foundations, and two independent codebases (`fips204` + `fips205`). No single
//! cryptanalytic breakthrough, and no single implementation bug, is sufficient to
//! forge. That is the property that turns "strong today" into "survives the
//! failure of either pillar."
//!
//! Parameter set: **SLH-DSA-SHAKE-256s** — NIST security category 5, the
//! *small-signature* (`s`) variant. Its slow signing / large-but-bounded 29 792 B
//! signature is exactly right here: release signing is offline, air-gapped, and
//! low-cadence, and the CI side only ever *verifies*, so neither cost matters.
//!
//! # We do NOT reimplement the hash-tree math
//!
//! FIPS-205 (FORS + Merkle hypertree + WOTS+) is provided by the vetted, pure-Rust
//! `fips205` crate (integritychain/fips205), gated behind the OPTIONAL
//! `evidence-slhdsa` cargo feature. When the feature is off, this module compiles
//! to fail-closed stubs, so the default compiler build keeps its minimal
//! dependency surface and the determinism/keystone gate never compiles a PQC crate.
//!
//! # Determinism (the load-bearing MIND invariant)
//!
//! Both key derivation and signing are RNG-free and byte-reproducible:
//!   * key generation: `KG::keygen_with_seeds(sk_seed, sk_prf, pk_seed)` — the
//!     96-byte operator seed splits into three 32-byte halves (FIPS-205 `SK.seed`,
//!     `SK.prf`, `PK.seed`); the crate feeds them through a deterministic dummy RNG.
//!   * signing: `try_sign(msg, ctx, hedged = false)` — the *deterministic* variant
//!     (`opt_rand = PK.seed`, the OS RNG is never consulted), so the same seed +
//!     message always yield byte-identical signature bytes on every substrate.
//!
//! Like the ML-DSA leg, the signature is metadata appended AFTER the mic@3 MAP
//! sentinel; it never feeds back into `trace_hash`, so unsigned artifacts stay
//! byte-identical and the keystone gate is untouched.

/// The `signature.scheme` tag value for a pure SLH-DSA-SHAKE-256s signature.
/// (The bulletproof profile embeds this leg inside the `pqc-hybrid-*` scheme; a
/// bare SLH-DSA artifact uses this tag directly.)
pub const SCHEME: &str = "slh-dsa-shake-256s";

/// Environment variable holding the 96-byte SLH-DSA key-generation seed as 192
/// hex chars: `SK.seed || SK.prf || PK.seed`, 32 bytes each. Never hardcode a key
/// — the seed is supplied out-of-band from the offline / air-gapped signer
/// (compliance: OMB M-26-15 key hygiene).
pub const ENV_SLHDSA_SEED: &str = "MIND_EVIDENCE_SLHDSA_KEY";

/// Length in bytes of the raw seed the operator supplies (three 32-byte FIPS-205
/// key components concatenated).
pub const SEED_LEN: usize = 96;

/// Is post-quantum SLH-DSA signing compiled into this build?
///
/// `true` iff the `evidence-slhdsa` feature is enabled. A verifier that reaches a
/// `slh-dsa-*`/`pqc-hybrid-*` artifact on a build where this is `false` must fail
/// closed (it cannot check the hash-based half), never report `valid`.
pub const fn supported() -> bool {
    cfg!(feature = "evidence-slhdsa")
}

#[cfg(feature = "evidence-slhdsa")]
mod imp {
    use fips205::slh_dsa_shake_256s;
    use fips205::traits::{KeyGen, SerDes, Signer, Verifier};

    /// A zero-filling RNG. `try_sign_with_rng(.., hedged = false)` is the FIPS-205
    /// *deterministic* variant — it sets `opt_rand = PK.seed` and never consults
    /// the RNG — so this is only present to satisfy the `CryptoRngCore` bound and
    /// is never actually read. It fills zeros deterministically so that even a
    /// future code path that DID read it stays byte-reproducible (never
    /// substrate- or run-dependent). fips205's own deterministic keygen path uses
    /// the same private-`DummyRng` trick; this is its public equivalent.
    struct ZeroRng;
    impl rand_core::RngCore for ZeroRng {
        fn next_u32(&mut self) -> u32 {
            0
        }
        fn next_u64(&mut self) -> u64 {
            0
        }
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for b in dest.iter_mut() {
                *b = 0;
            }
        }
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dest);
            Ok(())
        }
    }
    // A deterministic all-zero source is trivially not a real CSPRNG, but the
    // deterministic-signing contract (hedged = false) never draws from it, so the
    // marker is sound for this single, audited use.
    impl rand_core::CryptoRng for ZeroRng {}

    /// Public key length in bytes (FIPS-205 SLH-DSA-SHAKE-256s = 2·N = 64).
    pub const PK_LEN: usize = slh_dsa_shake_256s::PK_LEN;
    /// Signature length in bytes (FIPS-205 SLH-DSA-SHAKE-256s = 29 792).
    pub const SIG_LEN: usize = slh_dsa_shake_256s::SIG_LEN;

    /// Split the 96-byte operator seed into the three FIPS-205 key components.
    /// Returns `None` on a wrong-length seed (fail-closed — never pad or truncate
    /// key material).
    fn split_seed(seed: &[u8]) -> Option<([u8; 32], [u8; 32], [u8; 32])> {
        if seed.len() != super::SEED_LEN {
            return None;
        }
        let mut sk_seed = [0u8; 32];
        let mut sk_prf = [0u8; 32];
        let mut pk_seed = [0u8; 32];
        sk_seed.copy_from_slice(&seed[0..32]);
        sk_prf.copy_from_slice(&seed[32..64]);
        pk_seed.copy_from_slice(&seed[64..96]);
        Some((sk_seed, sk_prf, pk_seed))
    }

    /// Deterministically derive the SLH-DSA-SHAKE-256s public key from the 96-byte
    /// seed. Panics only on a mis-sized seed (a caller bug, never attacker input —
    /// the emit path validates the env seed before signing).
    pub fn public_key(seed: &[u8]) -> Vec<u8> {
        let (sk_seed, sk_prf, pk_seed) =
            split_seed(seed).expect("slh-dsa seed must be exactly 96 bytes");
        let (pk, _sk) = slh_dsa_shake_256s::KG::keygen_with_seeds(&sk_seed, &sk_prf, &pk_seed);
        pk.into_bytes().to_vec()
    }

    /// Deterministically sign `msg` (the shared signature preimage) under the key
    /// derived from `seed`. Uses the FIPS-205 *deterministic* variant
    /// (`hedged = false` ⇒ `opt_rand = PK.seed`, the OS RNG is never consulted),
    /// empty context string. Byte-reproducible across substrates.
    pub fn sign(seed: &[u8], msg: &[u8]) -> Vec<u8> {
        let (sk_seed, sk_prf, pk_seed) =
            split_seed(seed).expect("slh-dsa seed must be exactly 96 bytes");
        let (_pk, sk) = slh_dsa_shake_256s::KG::keygen_with_seeds(&sk_seed, &sk_prf, &pk_seed);
        // hedged = false is the deterministic branch (opt_rand = PK.seed); the RNG
        // is never read, empty ctx (< 256 bytes) — the only documented error causes
        // cannot occur here.
        let sig = sk
            .try_sign_with_rng(&mut ZeroRng, msg, b"", false)
            .expect("slh-dsa-shake-256s deterministic sign cannot fail");
        sig.to_vec()
    }

    /// Verify an SLH-DSA-SHAKE-256s signature over `msg` under `pk_bytes`.
    /// Fail-closed on any length/parse error (returns `false`, never panics on
    /// attacker input).
    pub fn verify(pk_bytes: &[u8], msg: &[u8], sig_bytes: &[u8]) -> bool {
        let pk_arr: [u8; PK_LEN] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let pk = match slh_dsa_shake_256s::PublicKey::try_from_bytes(&pk_arr) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let sig_arr: [u8; SIG_LEN] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        pk.verify(msg, &sig_arr, b"")
    }
}

#[cfg(not(feature = "evidence-slhdsa"))]
mod imp {
    // Fail-closed stubs when SLH-DSA support is not compiled in. `sign`/`public_key`
    // are never reached (the emit path checks `supported()` first and errors), and
    // `verify` always returns false so a verifier without hash-based PQC support can
    // never report a `slh-dsa-*`/`pqc-hybrid-*` artifact as valid.
    pub fn public_key(_seed: &[u8]) -> Vec<u8> {
        unreachable!("slh-dsa public_key called without the evidence-slhdsa feature")
    }
    pub fn sign(_seed: &[u8], _msg: &[u8]) -> Vec<u8> {
        unreachable!("slh-dsa sign called without the evidence-slhdsa feature")
    }
    pub fn verify(_pk_bytes: &[u8], _msg: &[u8], _sig_bytes: &[u8]) -> bool {
        false
    }
}

pub use imp::{public_key, sign, verify};

#[cfg(all(test, feature = "evidence-slhdsa"))]
mod tests {
    use super::{SEED_LEN, public_key, sign, verify};

    // A fixed, arbitrary 96-byte test seed (NOT a production key — tests only).
    fn test_seed() -> Vec<u8> {
        (0u8..96)
            .map(|i| i.wrapping_mul(7).wrapping_add(3))
            .collect()
    }

    #[test]
    fn seed_len_is_96() {
        assert_eq!(SEED_LEN, 96);
        assert_eq!(test_seed().len(), 96);
    }

    #[test]
    fn round_trip_verifies() {
        let seed = test_seed();
        let pk = public_key(&seed);
        let msg = b"the canonical provenance preimage bytes";
        let sig = sign(&seed, msg);
        assert!(
            verify(&pk, msg, &sig),
            "genuine SLH-DSA signature must verify"
        );
    }

    #[test]
    fn signing_is_deterministic() {
        // The load-bearing MIND invariant: same seed + same message => byte-identical
        // signature, every call (hedged = false, opt_rand = PK.seed, no RNG draw).
        let seed = test_seed();
        let msg = b"determinism is the wedge";
        let a = sign(&seed, msg);
        let b = sign(&seed, msg);
        assert_eq!(a, b, "SLH-DSA deterministic sign must be byte-reproducible");
        // Public key derivation is deterministic too.
        assert_eq!(public_key(&seed), public_key(&seed));
    }

    #[test]
    fn tampered_signature_fails() {
        let seed = test_seed();
        let msg = b"authenticity";
        let mut sig = sign(&seed, msg);
        sig[0] ^= 0x01; // flip one bit
        let pk = public_key(&seed);
        assert!(
            !verify(&pk, msg, &sig),
            "a tampered signature must fail closed"
        );
    }

    #[test]
    fn tampered_message_fails() {
        let seed = test_seed();
        let sig = sign(&seed, b"original message");
        let pk = public_key(&seed);
        assert!(
            !verify(&pk, b"different message", &sig),
            "a signature must not verify over a different message"
        );
    }

    #[test]
    fn wrong_pubkey_fails() {
        let seed = test_seed();
        let msg = b"who signed this";
        let sig = sign(&seed, msg);
        // A different seed => a different key => must not verify.
        let mut other = test_seed();
        other[95] ^= 0xFF;
        let wrong_pk = public_key(&other);
        assert!(
            !verify(&wrong_pk, msg, &sig),
            "verification under the wrong key must fail closed"
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let seed = test_seed();
        let pk = public_key(&seed);
        let sig = sign(&seed, b"m");
        // Truncated / oversized pubkey and signature must all return false, never panic.
        assert!(!verify(&pk[..pk.len() - 1], b"m", &sig));
        assert!(!verify(&pk, b"m", &sig[..sig.len() - 1]));
        assert!(!verify(&[], b"m", &sig));
        assert!(!verify(&pk, b"m", &[]));
    }
}
