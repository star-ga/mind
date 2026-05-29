// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! RFC 0016 evidence anchored on the canonical mic@1 IR — the artifact `mindc`
//! actually produces and ships.
//!
//! ## Why this exists (GAP-1, architecture audit 2026-05-26)
//!
//! RFC 0016 Phase A/B (`compact::v2::evidence`) computes `trace_hash` over the
//! v2 compact [`Graph`](crate::ir::compact::v2::Graph) (mic@2.1) — a
//! representation the compile pipeline **never produces for a real program**
//! (it carries 22 pure-dataflow opcodes, no control flow / functions / SIMD,
//! and the only non-test `Graph` is a hardcoded fixture). So that `trace_hash`
//! attested a toy graph, not the program a user builds.
//!
//! The compiler's canonical, stable, deterministic, platform-portable, and
//! *loadable* artifact is **mic@1 IR text** — [`save`]`(&`[`IRModule`]`)`, whose
//! `save → load → save` round-trip is an RFC-0001 fixed point, byte-identical
//! across runs and platforms (see `docs/ir-stability.md`). That is the
//! substrate-independent **IR link** of the RFC 0016 §4 Merkle DAG, and exactly
//! the object whose cross-substrate `trace_hash` equality RFC 0015 asserts (one
//! IR fans out to N per-substrate binaries; the IR itself is identical).
//!
//! Therefore the compiled-artifact evidence anchor lives here:
//! `trace_hash = SHA-256(canonical mic@1 text)`, computed through the same
//! [`mini_sha256`](crate::deps) FIPS-180-4 seam as the rest of RFC 0016 — so it
//! is bit-identical to a future pure-MIND `std.sha256` over the same canonical
//! bytes (RFC 0016 §5.4). The mic@2.1 MAP remains the evidence *container*
//! (it carries the `evidence_chain.*` keys and Ed25519 signing, §6); this module
//! supplies the value those keys must hold for a compiled artifact.

use crate::deps::mini_sha256;
use crate::ir::{IRModule, save};

/// The RFC 0016 §3.2/§3.3 `trace_hash` for a compiled artifact's **IR link**:
/// SHA-256 of the canonical mic@1 IR text of `ir`.
///
/// `ir::save` is RFC-0001 deterministic — byte-identical across runs and
/// platforms, with `save → load → save` a fixed point — so this hash is a
/// stable, substrate-independent identity for the compiled program. It is the
/// value RFC 0015 asserts equal across substrates for a Q16.16 graph, and the
/// `parent` a per-substrate binary link points at (§4).
///
/// Reuses the FIPS-180-4 `mini_sha256` seam, so the digest is bit-identical
/// whether computed by the Rust bootstrap or a future pure-MIND `std.sha256`
/// over the same canonical bytes.
pub fn ir_trace_hash(ir: &IRModule) -> [u8; 32] {
    mini_sha256(save(ir).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::parse_mic;
    use crate::ir::{BinOp, Instr};

    /// A small but non-trivial deterministic IR: `(42 + 10)` output.
    fn sample() -> IRModule {
        let mut m = IRModule::new();
        let v0 = m.fresh();
        let v1 = m.fresh();
        let v2 = m.fresh();
        m.instrs.push(Instr::ConstI64(v0, 42));
        m.instrs.push(Instr::ConstI64(v1, 10));
        m.instrs.push(Instr::BinOp {
            dst: v2,
            op: BinOp::Add,
            lhs: v0,
            rhs: v1,
        });
        m.instrs.push(Instr::Output(v2));
        m
    }

    #[test]
    fn trace_hash_is_sha256_of_canonical_mic1_text() {
        let m = sample();
        assert_eq!(
            ir_trace_hash(&m),
            mini_sha256(save(&m).as_bytes()),
            "ir_trace_hash must be SHA-256 of the canonical mic@1 text"
        );
        assert_ne!(ir_trace_hash(&m), [0u8; 32]);
    }

    #[test]
    fn trace_hash_stable_across_save_load_save() {
        // RFC-0001 fixed point ⇒ the IR link's trace_hash survives a round trip.
        let m = sample();
        let h1 = ir_trace_hash(&m);
        let reloaded = parse_mic(&save(&m)).expect("mic@1 must re-parse");
        let h2 = ir_trace_hash(&reloaded);
        assert_eq!(
            h1, h2,
            "save→load→save fixed point ⇒ identical IR trace_hash"
        );
    }

    #[test]
    fn distinct_irs_have_distinct_trace_hash() {
        let a = sample();
        let mut b = IRModule::new();
        let v0 = b.fresh();
        b.instrs.push(Instr::ConstI64(v0, 7));
        b.instrs.push(Instr::Output(v0));
        assert_ne!(
            ir_trace_hash(&a),
            ir_trace_hash(&b),
            "different programs must have different IR trace_hash"
        );
    }
}
