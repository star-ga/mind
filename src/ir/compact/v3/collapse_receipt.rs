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

//! Salov loop-collapse EVIDENCE RECEIPTS (S4) — the O(1) re-derivation proof.
//!
//! Design partner: Valerii Salov. The loop-collapse primitive (recognising an
//! accumulation over a counted loop and replacing it with its exact closed form)
//! is his contribution to the MIND roadmap. S1 collapses an affine SUM, S2 a
//! geometric PRODUCT; S4 (this module) records a **collapse receipt** in the
//! evidence chain so a consumer can INDEPENDENTLY RE-DERIVE the folded constant
//! in O(1) — re-checking the closed-form proof WITHOUT re-running the O(n) loop.
//!
//! # Placement (does NOT perturb `trace_hash`)
//!
//! A receipt lives in the appended MAP epilogue under the single key
//! `evidence_chain.collapse_receipts` (a `Bytes` value holding the TLV blob
//! encoded here). The epilogue is OUTSIDE the `trace_hash` preimage
//! (`trace_hash = SHA-256(canonical mic@3 body)` only), so a build that emits
//! receipts and one that does not produce the SAME `trace_hash` AND the same
//! code bytes: the no-receipt build simply omits the key, byte-identical
//! elsewhere. The receipt IS covered by the optional signature layer (it is an
//! `evidence_chain.*` key folded into the signature preimage), so on a SIGNED
//! artifact the receipts cannot be stripped/swapped; unsigned, the chain is
//! tamper-EVIDENT (never claim "signed").
//!
//! # Soundness (what the receipt proves — and what it does not)
//!
//! Verify RE-DERIVES, never trusts: it recomputes the folded constant from the
//! recorded loop parameters with the SAME ring-exact `Z/2^64` closed form the
//! collapse pass used (an INDEPENDENT re-implementation of
//! [`crate::opt::scev::closed_form_i64`] / [`crate::opt::scev::geometric_pow_i64`],
//! cross-checked in the tests) and demands bitwise equality with the constant
//! the receipt claims. The re-derivation is **O(1)** — it NEVER re-runs the loop
//! — and is EXACT: the affine `/2` is applied to the guaranteed-even factor
//! (exactly one of `n`, `p` is even), and the geometric factor is a fixed
//! 64-step square-and-multiply (NO division), so there is no non-divisible /
//! modular-wrap-assumption hazard. `Z/2^64` is MIND's defined integer width, not
//! an assumed one, and the collapse pass rejects (E2201/E2202) any const span
//! whose trip count overflows `i64`, so a receipt is only ever emitted for a
//! feasible range whose closed form equals the true trip-count sum.
//!
//! The receipt attests: *the constant materialised in the hashed body is the
//! arithmetically-correct closed form of the loop parameters recorded in the
//! receipt.* It does NOT (and cannot, from the artifact alone) prove the recorded
//! parameters were the original source loop — that binding is what SIGNING
//! provides (the compiler vouches for the whole artifact incl. the receipt).
//!
//! # Canonical encoding (fixed-width big-endian TLV)
//!
//! One encoding per receipt by construction — no varint / shortest-int
//! malleability, no floats/pointers/native-endian/map-iteration-order:
//!
//! ```text
//! [u32 BE receipt_count]
//! for each receipt (sorted canonically by (kind, params, constant)):
//!   [u8 kind]                     -- 0 = affine_sum, 1 = geometric_pow
//!   affine  (kind 0): i64 BE a | i64 BE b | i64 BE lo | i64 BE hi | i64 BE constant
//!   geometric(kind 1): i64 BE r | i64 BE lo | i64 BE hi | i64 BE constant
//! ```
//!
//! Every `i64` is 8-byte big-endian two's complement. Verify RE-ENCODES the
//! decoded receipts and byte-compares against the stored blob, so a
//! non-canonical (re-ordered / re-padded) encoding is rejected.
//!
//! deferred: `fixed_point` (S3 Q16.16) receipts — the period-1 re-derivation
//!   `f(claimed) == claimed` needs the user's map body, which is only present as
//!   mic@3 IR at verify time (not the AST the S3 comptime evaluator consumes).
//!   S3 collapse is already SOUND at compile time (it iterates the user's REAL
//!   body); it simply does not yet emit a verify-time receipt. Upgrade path: an
//!   IR-level comptime evaluator (or an AST-carrying side channel) that lets
//!   verify apply `f` once to the claimed constant.

/// TLV kind tag: affine sum `Σ_{i=lo}^{hi-1} (a·i + b)` (Slice S1).
const KIND_AFFINE: u8 = 0;
/// TLV kind tag: geometric power `acc · R^(hi-lo)` (Slice S2) — the recorded
/// `constant` is the multiplicative factor `R^n`.
const KIND_GEOMETRIC: u8 = 1;

/// Fixed on-wire size of one encoded receipt (kind byte + fixed-width fields).
const AFFINE_LEN: usize = 1 + 5 * 8; // kind + a,b,lo,hi,constant
const GEOMETRIC_LEN: usize = 1 + 4 * 8; // kind + r,lo,hi,constant
/// Smallest possible receipt — used to bound the decoded count against the input
/// length up front (DoS guard: an unsatisfiable count can never allocate).
const MIN_RECEIPT_LEN: usize = GEOMETRIC_LEN;

/// A single loop-collapse receipt: the recorded closed-form parameters plus the
/// folded `constant` the collapse pass emitted into the mic@3 body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollapseReceipt {
    /// Affine sum (S1): `Σ_{i=lo}^{hi-1} (a·i + b)` folded to `constant`.
    AffineSum {
        /// Coefficient `A` in `A·i + B`.
        a: i64,
        /// Constant term `B` in `A·i + B`.
        b: i64,
        /// Inclusive lower bound `LO`.
        lo: i64,
        /// Exclusive upper bound `HI`.
        hi: i64,
        /// The folded constant the collapse pass emitted (`closed_form_i64`).
        constant: i64,
    },
    /// Geometric power (S2): `acc · R^(HI-LO)` folded to the factor `constant`.
    GeometricPow {
        /// Loop-invariant multiplier `R`.
        r: i64,
        /// Inclusive lower bound `LO`.
        lo: i64,
        /// Exclusive upper bound `HI`.
        hi: i64,
        /// The folded factor `R^(HI-LO)` (`geometric_pow_i64`).
        constant: i64,
    },
}

impl CollapseReceipt {
    /// The folded constant the collapse pass emitted for this loop.
    pub fn constant(&self) -> i64 {
        match self {
            CollapseReceipt::AffineSum { constant, .. } => *constant,
            CollapseReceipt::GeometricPow { constant, .. } => *constant,
        }
    }

    /// INDEPENDENTLY re-derive the folded constant from the recorded loop
    /// parameters, in O(1). NEVER re-runs the loop. Ring-exact in `Z/2^64`
    /// (MIND's defined integer semantics) — this is a separate implementation of
    /// the collapse formula, cross-checked against [`crate::opt::scev`] in the
    /// tests, so it detects a tampered `constant` or tampered parameters.
    pub fn rederive(&self) -> i64 {
        match self {
            CollapseReceipt::AffineSum { a, b, lo, hi, .. } => rederive_affine(*a, *b, *lo, *hi),
            CollapseReceipt::GeometricPow { r, lo, hi, .. } => rederive_geometric(*r, *lo, *hi),
        }
    }

    /// `true` iff the recorded `constant` is the exact O(1) re-derivation of the
    /// recorded parameters — the load-bearing self-consistency check.
    pub fn is_self_consistent(&self) -> bool {
        self.rederive() == self.constant()
    }

    /// Canonical sort key: `(kind, params…, constant)` as fixed-width fields, so
    /// the encoded array order is deterministic and substrate-independent.
    fn sort_key(&self) -> (u8, [i64; 5]) {
        match self {
            CollapseReceipt::AffineSum {
                a,
                b,
                lo,
                hi,
                constant,
            } => (KIND_AFFINE, [*a, *b, *lo, *hi, *constant]),
            CollapseReceipt::GeometricPow {
                r,
                lo,
                hi,
                constant,
            } => (KIND_GEOMETRIC, [*r, *lo, *hi, *constant, 0]),
        }
    }
}

/// Ring-exact `Σ_{i=lo}^{hi-1} (a·i + b) mod 2^64` — an independent re-derivation
/// mirroring [`crate::opt::scev::closed_form_i64`]. Reversed/empty range → 0. The
/// `/2` is applied to the even factor (exactly one of `n`, `p` is even), so it is
/// EXACT — no non-divisible hazard.
fn rederive_affine(a: i64, b: i64, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
        return 0;
    }
    let n = hi.wrapping_sub(lo);
    let p = lo.wrapping_add(hi.wrapping_sub(1));
    let sum_i = n
        .wrapping_div(2)
        .wrapping_mul(p)
        .wrapping_add(n.wrapping_rem(2).wrapping_mul(p.wrapping_div(2)));
    a.wrapping_mul(sum_i).wrapping_add(b.wrapping_mul(n))
}

/// Ring-exact `R^(hi-lo) mod 2^64` via a FIXED 64-step square-and-multiply
/// ladder — an independent re-derivation mirroring
/// [`crate::opt::scev::geometric_pow_i64`]. Reversed/empty range → `R^0 = 1`. No
/// division ⇒ no non-divisible / modular-wrap-assumption hazard.
fn rederive_geometric(r: i64, lo: i64, hi: i64) -> i64 {
    if hi <= lo {
        return 1;
    }
    let n = hi.wrapping_sub(lo);
    let mut result: i64 = 1;
    let mut base: i64 = r;
    let mut e: i64 = n;
    let mut k = 0;
    while k < 64 {
        let bit = e.wrapping_rem(2);
        result = result.wrapping_mul(1i64.wrapping_add(bit.wrapping_mul(base.wrapping_sub(1))));
        base = base.wrapping_mul(base);
        e = e.wrapping_div(2);
        k += 1;
    }
    result
}

/// Error decoding a collapse-receipt TLV blob (fail-closed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollapseReceiptError {
    /// The blob ended mid-field (or claimed more receipts than the bytes allow).
    Truncated,
    /// An unrecognised `kind` tag (a downstream verifier fails closed).
    UnknownKind(u8),
    /// Bytes remained after the declared receipt count was consumed.
    TrailingBytes,
}

/// Encode receipts as the canonical fixed-width big-endian TLV blob. Receipts are
/// sorted by their canonical key so the output is deterministic regardless of
/// discovery order (substrate-independent, byte-stable).
pub fn encode_collapse_receipts(receipts: &[CollapseReceipt]) -> Vec<u8> {
    let mut sorted: Vec<&CollapseReceipt> = receipts.iter().collect();
    sorted.sort_by_key(|r| r.sort_key());

    let mut out = Vec::new();
    out.extend_from_slice(&(sorted.len() as u32).to_be_bytes());
    for rec in sorted {
        match rec {
            CollapseReceipt::AffineSum {
                a,
                b,
                lo,
                hi,
                constant,
            } => {
                out.push(KIND_AFFINE);
                for v in [a, b, lo, hi, constant] {
                    out.extend_from_slice(&v.to_be_bytes());
                }
            }
            CollapseReceipt::GeometricPow {
                r,
                lo,
                hi,
                constant,
            } => {
                out.push(KIND_GEOMETRIC);
                for v in [r, lo, hi, constant] {
                    out.extend_from_slice(&v.to_be_bytes());
                }
            }
        }
    }
    out
}

/// Decode a canonical TLV blob into receipts. Fail-closed on truncation, an
/// unknown kind, or trailing bytes.
pub fn decode_collapse_receipts(
    bytes: &[u8],
) -> Result<Vec<CollapseReceipt>, CollapseReceiptError> {
    if bytes.len() < 4 {
        return Err(CollapseReceiptError::Truncated);
    }
    let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut cur = 4usize;
    let avail = bytes.len() - cur;
    // DoS guard: each receipt needs >= MIN_RECEIPT_LEN bytes, so a count beyond
    // what the remaining input can hold is unsatisfiable — reject before the
    // `Vec::with_capacity` allocation.
    if count > avail / MIN_RECEIPT_LEN {
        return Err(CollapseReceiptError::Truncated);
    }
    let read_i64 = |b: &[u8], at: usize| -> i64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&b[at..at + 8]);
        i64::from_be_bytes(buf)
    };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if cur >= bytes.len() {
            return Err(CollapseReceiptError::Truncated);
        }
        let kind = bytes[cur];
        cur += 1;
        match kind {
            KIND_AFFINE => {
                if bytes.len() - cur < AFFINE_LEN - 1 {
                    return Err(CollapseReceiptError::Truncated);
                }
                let a = read_i64(bytes, cur);
                let b = read_i64(bytes, cur + 8);
                let lo = read_i64(bytes, cur + 16);
                let hi = read_i64(bytes, cur + 24);
                let constant = read_i64(bytes, cur + 32);
                cur += 5 * 8;
                out.push(CollapseReceipt::AffineSum {
                    a,
                    b,
                    lo,
                    hi,
                    constant,
                });
            }
            KIND_GEOMETRIC => {
                if bytes.len() - cur < GEOMETRIC_LEN - 1 {
                    return Err(CollapseReceiptError::Truncated);
                }
                let r = read_i64(bytes, cur);
                let lo = read_i64(bytes, cur + 8);
                let hi = read_i64(bytes, cur + 16);
                let constant = read_i64(bytes, cur + 24);
                cur += 4 * 8;
                out.push(CollapseReceipt::GeometricPow {
                    r,
                    lo,
                    hi,
                    constant,
                });
            }
            other => return Err(CollapseReceiptError::UnknownKind(other)),
        }
    }
    if cur != bytes.len() {
        return Err(CollapseReceiptError::TrailingBytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::scev::{closed_form_i64, geometric_pow_i64};

    // --- Re-derivation is byte-faithful to the collapse authority -----------

    #[test]
    fn affine_rederivation_matches_scev_authority() {
        let big = 1i64 << 30;
        let cases = [
            (1i64, 0i64, 0i64, 100i64),
            (3, 5, 0, 17),
            (-2, 7, -4, 6),
            (i64::MAX / 2, 0, 0, 10),
            (i64::MIN, 3, 0, 9),
            (1, 0, 0, big),
            (7, -3, 10, 3), // reversed → 0
            (5, 9, 5, 5),   // empty → 0
        ];
        for (a, b, lo, hi) in cases {
            let rec = CollapseReceipt::AffineSum {
                a,
                b,
                lo,
                hi,
                constant: closed_form_i64(a, b, lo, hi),
            };
            assert_eq!(
                rec.rederive(),
                closed_form_i64(a, b, lo, hi),
                "affine rederive != scev for a={a} b={b} lo={lo} hi={hi}"
            );
            assert!(rec.is_self_consistent());
        }
    }

    #[test]
    fn geometric_rederivation_matches_scev_authority() {
        let cases = [
            (7i64, 0i64, 13i64),
            (2, 0, 10),
            (-9, 0, 5),
            (3, 4, 4),  // empty → 1
            (5, 10, 3), // reversed → 1
            (i64::MAX, 0, 3),
        ];
        for (r, lo, hi) in cases {
            let rec = CollapseReceipt::GeometricPow {
                r,
                lo,
                hi,
                constant: geometric_pow_i64(r, lo, hi),
            };
            assert_eq!(
                rec.rederive(),
                geometric_pow_i64(r, lo, hi),
                "geometric rederive != scev for r={r} lo={lo} hi={hi}"
            );
            assert!(rec.is_self_consistent());
        }
    }

    // --- Encode/decode round-trip + canonical order -------------------------

    #[test]
    fn encode_decode_round_trip() {
        let receipts = vec![
            CollapseReceipt::AffineSum {
                a: 1,
                b: 0,
                lo: 0,
                hi: 100,
                constant: 4950,
            },
            CollapseReceipt::GeometricPow {
                r: 7,
                lo: 0,
                hi: 13,
                constant: 96_889_010_407,
            },
        ];
        let blob = encode_collapse_receipts(&receipts);
        let back = decode_collapse_receipts(&blob).expect("decode");
        // Sorted canonically (affine kind 0 before geometric kind 1).
        assert_eq!(back.len(), 2);
        assert!(matches!(back[0], CollapseReceipt::AffineSum { .. }));
        assert!(matches!(back[1], CollapseReceipt::GeometricPow { .. }));
        // Re-encoding the decoded receipts is byte-identical (canonical).
        assert_eq!(encode_collapse_receipts(&back), blob);
    }

    #[test]
    fn encoding_is_order_independent_canonical() {
        let a = CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 4950,
        };
        let g = CollapseReceipt::GeometricPow {
            r: 7,
            lo: 0,
            hi: 13,
            constant: 96_889_010_407,
        };
        // Both discovery orders encode to the SAME canonical bytes.
        assert_eq!(
            encode_collapse_receipts(&[a.clone(), g.clone()]),
            encode_collapse_receipts(&[g, a])
        );
    }

    #[test]
    fn empty_receipts_encode_to_four_zero_bytes() {
        assert_eq!(encode_collapse_receipts(&[]), vec![0, 0, 0, 0]);
        assert_eq!(decode_collapse_receipts(&[0, 0, 0, 0]).unwrap(), vec![]);
    }

    // --- Fail-closed decode -------------------------------------------------

    #[test]
    fn decode_rejects_truncated() {
        assert_eq!(
            decode_collapse_receipts(&[0, 0, 0]),
            Err(CollapseReceiptError::Truncated)
        );
        // count=1 but no receipt bytes.
        assert_eq!(
            decode_collapse_receipts(&[0, 0, 0, 1]),
            Err(CollapseReceiptError::Truncated)
        );
    }

    #[test]
    fn decode_rejects_unknown_kind() {
        // count=1, kind=9 (unknown), then 33 filler bytes so the count guard passes.
        let mut blob = vec![0, 0, 0, 1, 9];
        blob.extend_from_slice(&[0u8; 40]);
        assert_eq!(
            decode_collapse_receipts(&blob),
            Err(CollapseReceiptError::UnknownKind(9))
        );
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let mut blob = encode_collapse_receipts(&[CollapseReceipt::GeometricPow {
            r: 2,
            lo: 0,
            hi: 4,
            constant: 16,
        }]);
        blob.push(0xAB); // one stray byte
        assert_eq!(
            decode_collapse_receipts(&blob),
            Err(CollapseReceiptError::TrailingBytes)
        );
    }

    #[test]
    fn decode_rejects_alloc_bomb_count() {
        // count = u32::MAX with only 4 header bytes — unsatisfiable, must Err
        // before any allocation.
        let blob = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(
            decode_collapse_receipts(&blob),
            Err(CollapseReceiptError::Truncated)
        );
    }

    // --- Forgery: a tampered constant is not self-consistent ----------------

    #[test]
    fn tampered_constant_is_not_self_consistent() {
        let honest = CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 4950,
        };
        assert!(honest.is_self_consistent());
        let forged = CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 0,
            hi: 100,
            constant: 9999, // lie
        };
        assert!(!forged.is_self_consistent());
        assert_eq!(forged.rederive(), 4950);
    }

    #[test]
    fn tampered_params_break_self_consistency() {
        // Same claimed constant, but the recorded bounds were shifted — the
        // re-derivation no longer matches.
        let forged = CollapseReceipt::AffineSum {
            a: 1,
            b: 0,
            lo: 50, // was 0
            hi: 100,
            constant: 4950,
        };
        assert!(!forged.is_self_consistent());
    }
}
