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

//! RFC 0016 evidence anchored on the canonical mic@3 IR — the full-fidelity
//! serialization of the artifact `mindc` actually produces and ships.
//!
//! ## Why this exists (GAP-1, architecture audit 2026-05-26)
//!
//! RFC 0016 Phase A/B (`compact::v2::evidence`) computed `trace_hash` over the
//! v2 compact [`Graph`](crate::ir::compact::v2::Graph) (mic@2.1) — a
//! representation the compile pipeline **never produces for a real program**
//! (it carries 22 pure-dataflow opcodes, no control flow / functions / SIMD,
//! and the only non-test `Graph` is a hardcoded fixture). So that `trace_hash`
//! attested a toy graph, not the program a user builds.
//!
//! ## Why mic@3, not mic@1 text (collision audit 2026-05-31)
//!
//! The obvious candidate is mic@1 IR *text* ([`save`](crate::ir::save)), which
//! has an RFC-0001 `save → load → save` fixed point. But mic@1 text is **lossy
//! for function bodies**: [`ir::print`](crate::ir::print) emits an
//! [`Instr::FnDef`](crate::ir::Instr::FnDef) as a bare `// fn <name>` comment
//! and drops its `params`, `ret_id`, and `body` entirely. Two exported
//! functions that differ only in body — e.g. `f(x) { x + 2 }` versus
//! `f(x) { x * 999 }` — therefore serialize to byte-identical mic@1 text and
//! collide to the same SHA-256. A `trace_hash` over mic@1 text would attest the
//! signature surface, not the computation, defeating the evidence chain.
//!
//! The compiler's full-fidelity, deterministic, platform-portable canonical
//! artifact is **mic@3** — [`emit_mic3`], the
//! binary IR that carries complete function bodies and is itself the object the
//! cross-substrate byte-identity gate compares. It is the substrate-independent
//! **IR link** of the RFC 0016 §4 Merkle DAG, and exactly the object whose
//! cross-substrate `trace_hash` equality RFC 0015 asserts (one IR fans out to N
//! per-substrate binaries; the IR itself is identical).
//!
//! Therefore the compiled-artifact evidence anchor lives here:
//! `trace_hash = SHA-256(canonical mic@3 bytes)`, computed through the same
//! [`mini_sha256`](crate::deps) FIPS-180-4 seam as the rest of RFC 0016 — so it
//! is bit-identical to a future pure-MIND `std.sha256` over the same canonical
//! bytes (RFC 0016 §5.4). The mic@2.1 MAP remains the evidence *container*
//! (it carries the `evidence_chain.*` keys and Ed25519 signing, §6); this module
//! supplies the value those keys must hold for a compiled artifact.

use crate::deps::mini_sha256;
use crate::ir::IRModule;
use crate::ir::compact::emit_mic3;

/// The RFC 0016 §3.2/§3.3 `trace_hash` for a compiled artifact's **IR link**:
/// SHA-256 of the canonical mic@3 bytes of `ir`.
///
/// [`emit_mic3`] is RFC-0001 deterministic —
/// byte-identical across runs and platforms, with `emit → parse → emit` a fixed
/// point — and, unlike mic@1 text, it carries complete function bodies, so the
/// hash depends on what each exported function computes, not just its signature.
/// This is the value RFC 0015 asserts equal across substrates for a Q16.16
/// graph, and the `parent` a per-substrate binary link points at (§4).
///
/// Reuses the FIPS-180-4 `mini_sha256` seam, so the digest is bit-identical
/// whether computed by the Rust bootstrap or a future pure-MIND `std.sha256`
/// over the same canonical bytes.
pub fn ir_trace_hash(ir: &IRModule) -> [u8; 32] {
    mini_sha256(&emit_mic3(ir))
}

/// Bare non-deterministic builtins — PRNG draws that read hidden generator state
/// and wall-clock / stdin reads. A compiled module that calls one is genuinely
/// non-deterministic (its output is not a pure function of its inputs), so its
/// evidence chain MUST honestly declare `nondeterministic` rather than forge
/// `deterministic` (the claim `mind verify` reports). This is the determinism
/// wedge's honesty invariant: the attestation can never lie.
///
/// Kept in sync with `type_checker::mod::NONDETERMINISTIC_BUILTINS` (the AST-side
/// classifier used by the `#[deterministic]` call-graph check); the two lists
/// describe the same set from two layers. The legitimate, DETERMINISTIC randomness
/// API is the SEEDED counter-based form (`randn(shape, seed)`, `Random(seed=…)`,
/// Philox/Threefry) — those are pure functions of `(seed, index)` and are NOT in
/// this list. `randn` appears here as the BARE/unseeded draw; a seeded call is
/// resolved to its explicit generator, not this implicit builtin.
const NONDETERMINISTIC_BUILTINS: &[&str] = &[
    "monotonic_now",
    "now",
    "rand",
    "rand_bytes",
    "rand_int",
    "rand_normal",
    "rand_range",
    "rand_uniform",
    "randn",
    "random",
    "read_input",
    "read_line",
    "shuffle",
    "system_time",
    "time_now",
];

/// Whether `callee` names a bare non-deterministic builtin (matched on the bare
/// name or the last dotted/`::`-qualified path segment, so `std.rand.random` and
/// `rng::rand_uniform` are caught too).
fn callee_is_nondeterministic(callee: &str) -> bool {
    let tail = callee.rsplit(['.', ':']).next().unwrap_or(callee);
    NONDETERMINISTIC_BUILTINS.contains(&tail) || NONDETERMINISTIC_BUILTINS.contains(&callee)
}

/// Recursively true iff any `Instr::Call` anywhere in `instrs` (including nested
/// function bodies, loop bodies, and if-branches) targets a non-deterministic
/// builtin.
fn stream_has_nondeterministic_call(instrs: &[crate::ir::Instr]) -> bool {
    use crate::ir::Instr;
    instrs.iter().any(|instr| match instr {
        Instr::Call { name, .. } => callee_is_nondeterministic(name),
        Instr::FnDef { body, .. } => stream_has_nondeterministic_call(body),
        #[cfg(feature = "std-surface")]
        Instr::While {
            cond_instrs, body, ..
        } => {
            stream_has_nondeterministic_call(cond_instrs) || stream_has_nondeterministic_call(body)
        }
        #[cfg(feature = "std-surface")]
        Instr::If {
            cond_instrs,
            then_instrs,
            else_instrs,
            ..
        } => {
            stream_has_nondeterministic_call(cond_instrs)
                || stream_has_nondeterministic_call(then_instrs)
                || stream_has_nondeterministic_call(else_instrs)
        }
        _ => false,
    })
}

/// The evidence-chain determinism declaration for a compiled module: `true`
/// (deterministic) UNLESS the module calls a PRNG / wall-clock / stdin builtin,
/// in which case `false` (non-deterministic). This makes the
/// `evidence_chain.determinism` field HONEST-BY-DERIVATION instead of a hardcoded
/// optimistic default — a `random()` / `now()` program can no longer forge a
/// `deterministic` attestation. Deterministic programs (the overwhelmingly common
/// case, including seeded `randn(shape, seed)`) are unaffected.
pub fn ir_declares_deterministic(module: &IRModule) -> bool {
    !stream_has_nondeterministic_call(&module.instrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::parse_mic3;
    use crate::ir::{BinOp, Instr};

    #[test]
    fn determinism_declaration_is_honest_by_derivation() {
        // A pure-arithmetic module is deterministic.
        let mut det = IRModule::new();
        let a = det.fresh();
        let b = det.fresh();
        let s = det.fresh();
        det.instrs.push(Instr::ConstI64(a, 42));
        det.instrs.push(Instr::ConstI64(b, 10));
        det.instrs.push(Instr::BinOp {
            dst: s,
            op: BinOp::Add,
            lhs: a,
            rhs: b,
        });
        det.instrs.push(Instr::Output(s));
        assert!(
            ir_declares_deterministic(&det),
            "pure arithmetic module must declare deterministic"
        );

        // A module that calls a PRNG builtin is NON-deterministic — the evidence
        // chain must not forge `deterministic` (the honesty invariant).
        for nondet_call in ["random", "rand_uniform", "now", "std.rand.rand_uniform"] {
            let mut nd = IRModule::new();
            let seed = nd.fresh();
            let r = nd.fresh();
            nd.instrs.push(Instr::ConstI64(seed, 0));
            nd.instrs.push(Instr::Call {
                dst: r,
                name: nondet_call.to_string(),
                args: vec![seed],
            });
            nd.instrs.push(Instr::Output(r));
            assert!(
                !ir_declares_deterministic(&nd),
                "a module calling `{nondet_call}` must declare NON-deterministic \
                 (evidence chain may not forge `deterministic`)"
            );
        }

        // A nested (inside a fn body) PRNG call is still detected.
        let mut nested = IRModule::new();
        let p = nested.fresh();
        let rr = nested.fresh();
        nested.instrs.push(Instr::FnDef {
            name: "g".to_string(),
            params: vec![],
            ret_id: Some(rr),
            body: vec![
                Instr::ConstI64(p, 0),
                Instr::Call {
                    dst: rr,
                    name: "random".to_string(),
                    args: vec![p],
                },
                Instr::Return { value: Some(rr) },
            ],
            reap_threshold: None,
        });
        assert!(
            !ir_declares_deterministic(&nested),
            "a PRNG call nested in a fn body must be detected"
        );
    }

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

    /// A single exported function `f(x) -> x + k`, used to prove the trace_hash
    /// is sensitive to function *bodies*, not just signatures. The two modules
    /// produced for different `k` share an identical mic@1-text rendering
    /// (`// fn f` + dropped body) yet differ in mic@3.
    fn fn_module(k: i64) -> IRModule {
        let mut m = IRModule::new();
        let p = m.fresh();
        let c = m.fresh();
        let r = m.fresh();
        m.exports.insert("f".to_string());
        m.instrs.push(Instr::FnDef {
            name: "f".to_string(),
            params: vec![("x".to_string(), p)],
            ret_id: Some(r),
            body: vec![
                Instr::Param {
                    dst: p,
                    name: "x".to_string(),
                    index: 0,
                },
                Instr::ConstI64(c, k),
                Instr::BinOp {
                    dst: r,
                    op: BinOp::Add,
                    lhs: p,
                    rhs: c,
                },
                Instr::Return { value: Some(r) },
            ],
            reap_threshold: None,
        });
        m
    }

    #[test]
    fn trace_hash_is_sha256_of_canonical_mic3_bytes() {
        let m = sample();
        assert_eq!(
            ir_trace_hash(&m),
            mini_sha256(&emit_mic3(&m)),
            "ir_trace_hash must be SHA-256 of the canonical mic@3 bytes"
        );
        assert_ne!(ir_trace_hash(&m), [0u8; 32]);
    }

    #[test]
    fn trace_hash_stable_across_emit_parse_emit() {
        // mic@3 is the full-fidelity fixed point ⇒ trace_hash survives a round trip.
        let m = sample();
        let h1 = ir_trace_hash(&m);
        let reloaded = parse_mic3(&emit_mic3(&m)).expect("mic@3 must re-parse");
        let h2 = ir_trace_hash(&reloaded);
        assert_eq!(
            h1, h2,
            "emit→parse→emit fixed point ⇒ identical IR trace_hash"
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

    #[test]
    fn trace_hash_distinguishes_function_bodies() {
        // Regression guard for the mic@1-text collision (2026-05-31): two
        // exported functions with identical signatures but different bodies
        // (`f(x) = x + 2` vs `f(x) = x + 999`) MUST hash differently. Anchoring
        // on mic@1 text dropped the body and collided them; mic@3 carries it.
        assert_ne!(
            ir_trace_hash(&fn_module(2)),
            ir_trace_hash(&fn_module(999)),
            "trace_hash must depend on function bodies, not just signatures"
        );
    }
}
