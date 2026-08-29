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

// The determinism classifier: whether `callee` can make the module's result
// vary between runs on the same declared inputs. A compiled module that reaches
// one is genuinely non-deterministic, so its evidence chain MUST honestly
// declare `nondeterministic` rather than forge `deterministic` (the claim
// `mindc verify` reports). This is the determinism wedge's honesty invariant:
// the attestation can never lie.
//
// It lives in `crate::intrinsics`, on the same row as the intrinsic's name
// and arity, and the AST-side `#[deterministic]` call-graph check calls the
// SAME function. Until 2026-08-28 the two layers each kept their own array with
// a comment asking for them to be kept in sync; they drifted identically, and
// the registered `__mind_nerve_rt_*` clock / entropy / stdin / getenv surface
// was classified by neither — so `fn main() -> i64 { __mind_nerve_rt_monotonic_ns() }`
// emitted an artifact attesting `determinism: deterministic` that PASSED
// `mindc verify --require-deterministic`. Re-deriving the label from the hashed
// mic@3 body (below) is no defence when the classifier being re-run is itself
// incomplete, so there is now exactly one classifier and the registry forces a
// verdict on every new intrinsic. The `deferred:` blocks in that module record
// the two deliberate non-taints (raw memory intrinsics; file reads).
//
// One verdict cannot be reached by NAME at all. `__mind_read` is the same symbol
// for a file read and for a stdin read, so its registry row (`Det::Pure`, for the
// file case) let `__mind_read(0, buf, 1, -1)` pipe stdin into the program's result
// under a `determinism: deterministic` attestation that passed
// `mindc verify --require-deterministic`. The descriptor is an IR operand, so that
// verdict is taken at the CALL SITE — `ScopeConsts` + `call_reads_world_stream`
// below, against the policy predicates in `crate::intrinsics`. It FAILS CLOSED: a
// descriptor this module cannot prove constant can be `0`, so it is world-reading.
use crate::intrinsics::callee_is_nondeterministic;

/// The NAME of the first non-deterministic builtin an instruction stream calls
/// (searching nested function bodies, loop bodies, and if-branches in
/// deterministic instruction order), or `None` if the stream is deterministic.
/// Returning the offender's name — not just a bool — lets the build gate NAME it
/// in a fail-loud diagnostic and lets `verify` re-derive the label from the
/// hashed body.
fn find_nondeterministic_call(instrs: &[crate::ir::Instr]) -> Option<String> {
    find_nondeterministic_call_ext(instrs, &std::collections::BTreeSet::new())
}

/// Scope ENTRY: classify one SSA namespace's instruction stream.
///
/// A namespace is the module top level or ONE `FnDef` body — value ids are
/// numbered per function (`src/ir/verify.rs`: "a body `%0` and an
/// enclosing/top-level `%0` are distinct values"), so the constant environment
/// the call-site descriptor check consults is built per namespace, never shared
/// across a `FnDef` boundary.
fn find_nondeterministic_call_ext(
    instrs: &[crate::ir::Instr],
    externs: &std::collections::BTreeSet<String>,
) -> Option<String> {
    scan_scope(instrs, externs, &ScopeConsts::for_scope(instrs, &[]))
}

/// Classify one instruction stream WITHIN an already-built namespace. `If` /
/// `While` / `Region` sub-streams are the SAME namespace and reuse `consts`;
/// a `FnDef` body opens a fresh one.
fn scan_scope(
    instrs: &[crate::ir::Instr],
    externs: &std::collections::BTreeSet<String>,
    consts: &ScopeConsts,
) -> Option<String> {
    use crate::ir::Instr;
    for instr in instrs {
        let hit = match instr {
            Instr::Call { name, args, .. }
                if callee_is_nondeterministic(name)
                    || extern_call_is_unclassified(name, externs)
                    || call_reads_world_stream(name, args, consts) =>
            {
                Some(name.clone())
            }
            Instr::Call { .. } => None,
            Instr::FnDef { params, body, .. } => {
                let param_ids: Vec<crate::ir::ValueId> =
                    params.iter().map(|(_name, id)| *id).collect();
                scan_scope(body, externs, &ScopeConsts::for_scope(body, &param_ids))
            }
            #[cfg(feature = "std-surface")]
            Instr::While {
                cond_instrs, body, ..
            } => scan_scope(cond_instrs, externs, consts)
                .or_else(|| scan_scope(body, externs, consts)),
            #[cfg(feature = "std-surface")]
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => scan_scope(cond_instrs, externs, consts)
                .or_else(|| scan_scope(then_instrs, externs, consts))
                .or_else(|| scan_scope(else_instrs, externs, consts)),
            // RFC 0010 Phase J-A region body carries a FULL nested instruction
            // stream (`src/ir/mod.rs`). A nondeterministic `now()`/`rand()` call
            // inside `region { }` must NOT be invisible to the attestation
            // classifier, or a genuinely nondeterministic module could forge a
            // `deterministic` label (the honesty invariant). Mirrors the
            // `Instr::Region { body, .. }` recursion already in `verify.rs` (SSA)
            // and `fp_mode.rs` (strict-FP taint).
            #[cfg(feature = "std-surface")]
            Instr::Region { body, .. } => scan_scope(body, externs, consts),
            // Remaining instructions carry NO nested instruction stream and are
            // not themselves a builtin call, so they cannot introduce a
            // nondeterministic callee. Enumerated EXHAUSTIVELY (no blanket `_`)
            // so that a FUTURE variant carrying a nested body becomes a COMPILE
            // ERROR here — a forced review point — rather than a silent
            // attestation escape (the exact class of bug this arm replaces).
            Instr::ConstI64(..)
            | Instr::ConstF64(..)
            | Instr::ConstTensor(..)
            | Instr::ConstDenseTensor { .. }
            | Instr::BinOp { .. }
            | Instr::Sum { .. }
            | Instr::Mean { .. }
            | Instr::Relu { .. }
            | Instr::ReluGrad { .. }
            | Instr::Reshape { .. }
            | Instr::ExpandDims { .. }
            | Instr::Squeeze { .. }
            | Instr::Transpose { .. }
            | Instr::Dot { .. }
            | Instr::MatMul { .. }
            | Instr::Conv2d { .. }
            | Instr::Conv2dGradInput { .. }
            | Instr::Conv2dGradFilter { .. }
            | Instr::Index { .. }
            | Instr::Slice { .. }
            | Instr::Gather { .. }
            | Instr::Output(..)
            | Instr::SparseAttr { .. }
            | Instr::Return { .. }
            | Instr::Param { .. } => None,
            #[cfg(feature = "std-surface")]
            Instr::ConstArray { .. }
            | Instr::ArrayLoad { .. }
            | Instr::ArrayStore { .. }
            | Instr::Break { .. }
            | Instr::Continue { .. }
            | Instr::VecLoad { .. }
            | Instr::VecFma { .. }
            | Instr::VecReduceAdd { .. }
            | Instr::VecStore { .. }
            | Instr::VecLoadI32 { .. }
            | Instr::VecMulAddQ16 { .. }
            | Instr::VecReduceAddI64 { .. }
            | Instr::ExternFnDecl { .. } => None,
        };
        if hit.is_some() {
            return hit;
        }
    }
    None
}

/// The first non-deterministic builtin (`random` / `now` / …) a compiled module
/// calls, or `None` when the module is deterministic. THE single derivation used
/// by three consumers so they can never disagree: the emit-side evidence label
/// (`mindc.rs`), the build-time fail-loud gate (which names this offender), and
/// the verify-side re-derivation that re-computes the label from the hashed mic@3
/// body — so the `evidence_chain.determinism` MAP field cannot be forged even on
/// an unsigned artifact.
pub fn ir_first_nondeterministic_call(module: &IRModule) -> Option<String> {
    let externs = collect_extern_symbols(&module.instrs);
    find_nondeterministic_call_ext(&module.instrs, &externs)
}

/// Every `extern "C"` symbol declared anywhere in the module.
///
/// Needed because the determinism classifier is a REGISTRY lookup: a name it does
/// not recognise returns "deterministic". For a user function that is correct —
/// its body is in the module and is classified on its own merits. For an
/// `extern "C"` symbol it is default-ADMIT on an unknown, which is backwards for
/// an ATTESTATION: the callee's body is outside the artifact entirely, so nothing
/// in the module can witness what it does. Measured before this: a program
/// declaring libc `time()` or `getenv()` and calling it emitted evidence attesting
/// `determinism: deterministic`.
fn collect_extern_symbols(instrs: &[crate::ir::Instr]) -> std::collections::BTreeSet<String> {
    use crate::ir::Instr;
    let mut out = std::collections::BTreeSet::new();
    fn walk(instrs: &[Instr], out: &mut std::collections::BTreeSet<String>) {
        for instr in instrs {
            match instr {
                #[cfg(feature = "std-surface")]
                Instr::ExternFnDecl { name, .. } => {
                    out.insert(name.clone());
                }
                Instr::FnDef { body, .. } => walk(body, out),
                #[cfg(feature = "std-surface")]
                Instr::While {
                    cond_instrs, body, ..
                } => {
                    walk(cond_instrs, out);
                    walk(body, out);
                }
                #[cfg(feature = "std-surface")]
                Instr::If {
                    cond_instrs,
                    then_instrs,
                    else_instrs,
                    ..
                } => {
                    walk(cond_instrs, out);
                    walk(then_instrs, out);
                    walk(else_instrs, out);
                }
                _ => {}
            }
        }
    }
    walk(instrs, &mut out);
    out
}

/// True when a call to `name` must be treated as world-touching because it
/// crosses the artifact boundary: an `extern "C"` symbol that the intrinsic
/// registry does not explicitly classify as pure.
///
/// `deferred:` the honest end state is RFC 0019 §3.3 decline-to-attest — an
/// artifact calling an unclassified extern should refuse to make ANY determinism
/// claim rather than claim nondeterminism. Reporting nondeterministic is the
/// conservative direction (it can never forge a `deterministic` attestation), so
/// it is the safe interim; upgrade path is a third `Unknown` verdict threaded
/// through `ir_declares_deterministic` and the verify surface.
fn extern_call_is_unclassified(name: &str, externs: &std::collections::BTreeSet<String>) -> bool {
    externs.contains(name)
        && crate::intrinsics::intrinsic_determinism(name) != Some(crate::intrinsics::Det::Pure)
}

/// The constants provable at a single SSA namespace's call sites.
///
/// ## Why the classifier needs this
///
/// Some intrinsics are not classifiable by NAME. `__mind_read(fd, buf, n, off)`
/// is the same symbol whether it reads a file the program opened or drains the
/// stdin stream, and the registry row can only say one thing — it said `Pure`, so
/// `fn main() -> i64 { __mind_read(0, buf, 1, -1) }` piped a byte of stdin into
/// its exit code while its evidence chain attested `determinism: deterministic`
/// and `mindc verify --require-deterministic` exited 0. The descriptor is right
/// there in the IR as `args[0]`, so the verdict belongs at the call site.
///
/// ## What "provable" means here, and why it FAILS CLOSED
///
/// A value id is provably `v` only when EVERY definition of it in this namespace
/// is `ConstI64(id, v)` for the same `v`. Anything else — a parameter, a struct
/// field load (`std.io`'s `file_read(f, …)` reads `f.fd`), an `If` merge, a
/// `While` exit id, an arithmetic result, a second `ConstI64` with a different
/// value — POISONS the id, and a poisoned or unknown descriptor is treated as
/// world-reading. That direction is forced: an unprovable descriptor CAN be `0`,
/// and `file_read(stdin(), buf, n, -1)` — a call the shipped `std/io.mind`
/// surface makes trivial — is exactly how a descriptor-value-only rule would be
/// walked past. An attestation may over-report nondeterminism; it may never
/// under-report it.
///
/// Poisoning reuses [`crate::ir::verify::expose_region_definitions`] — the SAME
/// block-arg/result exposure the SSA verifier uses — so this never invents a
/// second, divergent notion of "defined".
#[derive(Default)]
struct ScopeConsts {
    known: std::collections::BTreeMap<crate::ir::ValueId, i64>,
    poisoned: std::collections::BTreeSet<crate::ir::ValueId>,
}

impl ScopeConsts {
    /// Build the environment for ONE namespace: `instrs` plus, for a function
    /// body, its parameter ids (which are inputs, never provable constants).
    /// Descends same-namespace `If`/`While`/`Region` sub-streams and STOPS at
    /// every `FnDef` (a nested function is a separate namespace).
    fn for_scope(instrs: &[crate::ir::Instr], params: &[crate::ir::ValueId]) -> Self {
        let mut out = Self::default();
        for p in params {
            out.poison(*p);
        }
        out.collect(instrs);
        out
    }

    fn poison(&mut self, id: crate::ir::ValueId) {
        self.known.remove(&id);
        self.poisoned.insert(id);
    }

    fn define_const(&mut self, id: crate::ir::ValueId, value: i64) {
        if self.poisoned.contains(&id) {
            return;
        }
        match self.known.get(&id) {
            // Re-stating the same constant is not a conflict (a `ConstI64` inside
            // a loop body is re-executed, not redefined).
            Some(prev) if *prev == value => {}
            // Two different constants for one id: malformed or shadowed IR. Fail
            // closed — the call site cannot know which one it sees.
            Some(_) => self.poison(id),
            None => {
                self.known.insert(id, value);
            }
        }
    }

    fn collect(&mut self, instrs: &[crate::ir::Instr]) {
        use crate::ir::Instr;
        for instr in instrs {
            if let Instr::ConstI64(dst, value) = instr {
                self.define_const(*dst, *value);
            } else {
                // Every id this instruction defines — including an `If`'s merge
                // ids and a `While`'s exit ids, which are definitions no
                // `instruction_dst` reports — is unprovable.
                let mut defs = std::collections::BTreeSet::new();
                crate::ir::verify::expose_region_definitions(instr, &mut defs);
                for id in defs {
                    self.poison(id);
                }
            }
            match instr {
                #[cfg(feature = "std-surface")]
                Instr::If {
                    cond_instrs,
                    then_instrs,
                    else_instrs,
                    ..
                } => {
                    self.collect(cond_instrs);
                    self.collect(then_instrs);
                    self.collect(else_instrs);
                }
                #[cfg(feature = "std-surface")]
                Instr::While {
                    cond_instrs, body, ..
                } => {
                    self.collect(cond_instrs);
                    self.collect(body);
                }
                #[cfg(feature = "std-surface")]
                Instr::Region { body, .. } => self.collect(body),
                // `FnDef` is deliberately NOT descended: its body is a separate
                // SSA namespace and gets its own `ScopeConsts`.
                _ => {}
            }
        }
    }

    /// The constant this id provably holds in this namespace, or `None` when the
    /// call site cannot prove one (the fail-closed case).
    fn provable(&self, id: crate::ir::ValueId) -> Option<i64> {
        if self.poisoned.contains(&id) {
            return None;
        }
        self.known.get(&id).copied()
    }
}

/// True when THIS call site reads a world channel because of the descriptor it
/// passes — the call-site half of the determinism classifier.
///
/// `crate::intrinsics` owns the policy (which intrinsic, which argument position,
/// which descriptors are world); this function owns the proof. A descriptor that
/// cannot be proven constant is world-reading: it can be `0`.
fn call_reads_world_stream(name: &str, args: &[crate::ir::ValueId], consts: &ScopeConsts) -> bool {
    let Some(pos) = crate::intrinsics::fd_dependent_read_arg(name) else {
        return false;
    };
    match args.get(pos).and_then(|id| consts.provable(*id)) {
        Some(fd) => crate::intrinsics::read_fd_is_world(fd),
        // Unprovable descriptor (a parameter, a `f.fd` field load, an arithmetic
        // result) — or a call whose arity does not even reach the descriptor
        // position, i.e. malformed IR. Both fail closed.
        None => true,
    }
}

/// The evidence-chain determinism declaration for a compiled module: `true`
/// (deterministic) UNLESS the module calls a PRNG / wall-clock builtin, or reads
/// a descriptor it cannot prove is not an inherited standard stream, in which
/// case `false` (non-deterministic). Honest-by-derivation, not a
/// hardcoded default — a `random()` / `now()` / `__mind_read(0, …)` program
/// cannot forge a `deterministic` attestation. Deterministic programs (including seeded
/// `randn(shape, seed)`) are unaffected.
pub fn ir_declares_deterministic(module: &IRModule) -> bool {
    ir_first_nondeterministic_call(module).is_none()
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
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
        });
        assert!(
            !ir_declares_deterministic(&nested),
            "a PRNG call nested in a fn body must be detected"
        );
    }

    /// A module whose ONLY nondeterministic call lives inside a `region { }`
    /// body. Exercises the RFC 0010 `Instr::Region` recursion path of the
    /// determinism classifier.
    #[cfg(feature = "std-surface")]
    fn region_module_with_call(callee: &str) -> IRModule {
        let mut m = IRModule::new();
        let c = m.fresh();
        let r = m.fresh();
        let enter = m.fresh();
        let exit = m.fresh();
        m.instrs.push(Instr::Region {
            body: vec![
                Instr::ConstI64(c, 0),
                Instr::Call {
                    dst: r,
                    name: callee.to_string(),
                    args: vec![c],
                },
            ],
            result: r,
            enter_id: enter,
            exit_id: exit,
            alloc_ids: vec![],
        });
        m.instrs.push(Instr::Output(r));
        m
    }

    /// Regression (attestation-honesty): a nondeterministic `now()`/`rand()`
    /// call buried in a `region { }` body MUST be seen by the classifier — it
    /// cannot forge a `deterministic` attestation. This is the fix for the
    /// `Instr::Region`-blind-spot: before the fix the blanket `_ => None`
    /// swallowed the region body and this module verified as deterministic.
    #[cfg(feature = "std-surface")]
    #[test]
    fn region_nested_nondeterministic_call_is_detected() {
        for callee in ["now", "rand"] {
            let m = region_module_with_call(callee);
            assert_eq!(
                ir_first_nondeterministic_call(&m).as_deref(),
                Some(callee),
                "region-nested `{callee}()` must be named by the classifier"
            );
            assert!(
                !ir_declares_deterministic(&m),
                "a module calling `{callee}()` inside a region must declare NON-deterministic"
            );
        }
    }

    /// Positive control: a Region whose body is PURE arithmetic stays
    /// deterministic — the fix does not over-taint region-carrying modules.
    #[cfg(feature = "std-surface")]
    #[test]
    fn deterministic_region_body_stays_deterministic() {
        let mut m = IRModule::new();
        let a = m.fresh();
        let b = m.fresh();
        let s = m.fresh();
        let enter = m.fresh();
        let exit = m.fresh();
        m.instrs.push(Instr::Region {
            body: vec![
                Instr::ConstI64(a, 42),
                Instr::ConstI64(b, 10),
                Instr::BinOp {
                    dst: s,
                    op: BinOp::Add,
                    lhs: a,
                    rhs: b,
                },
            ],
            result: s,
            enter_id: enter,
            exit_id: exit,
            alloc_ids: vec![],
        });
        m.instrs.push(Instr::Output(s));
        assert!(
            ir_declares_deterministic(&m),
            "a region with a pure-arithmetic body must stay deterministic"
        );
        assert_eq!(ir_first_nondeterministic_call(&m), None);
    }

    /// Control (unchanged behavior): a top-level `now()` and an FnDef-nested
    /// `now()` were already detected before the fix and must still be.
    #[test]
    fn top_level_and_fndef_nested_nondeterministic_still_detected() {
        // Top-level.
        let mut top = IRModule::new();
        let s = top.fresh();
        let r = top.fresh();
        top.instrs.push(Instr::ConstI64(s, 0));
        top.instrs.push(Instr::Call {
            dst: r,
            name: "now".to_string(),
            args: vec![s],
        });
        top.instrs.push(Instr::Output(r));
        assert_eq!(ir_first_nondeterministic_call(&top).as_deref(), Some("now"));
        assert!(!ir_declares_deterministic(&top));

        // FnDef-nested.
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
                    name: "now".to_string(),
                    args: vec![p],
                },
                Instr::Return { value: Some(rr) },
            ],
            reap_threshold: None,
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
        });
        assert!(!ir_declares_deterministic(&nested));
    }

    // ---------------------------------------------------------------
    // Call-site descriptor classification for `__mind_read`
    // ---------------------------------------------------------------

    /// A module whose `main` reads `count` bytes from the descriptor produced by
    /// `fd_instr`, exactly as `__mind_read(fd, buf, 1, -1)` lowers.
    fn read_module(fd_instrs: Vec<Instr>, fd: crate::ir::ValueId) -> IRModule {
        let mut m = IRModule::new();
        let size = crate::ir::ValueId(90);
        let buf = crate::ir::ValueId(91);
        let off = crate::ir::ValueId(92);
        let n = crate::ir::ValueId(93);
        m.instrs.extend(fd_instrs);
        m.instrs.push(Instr::ConstI64(size, 8));
        m.instrs.push(Instr::Call {
            dst: buf,
            name: "__mind_alloc".to_string(),
            args: vec![size],
        });
        m.instrs.push(Instr::ConstI64(off, -1));
        m.instrs.push(Instr::Call {
            dst: n,
            name: "__mind_read".to_string(),
            args: vec![fd, buf, size, off],
        });
        m.instrs.push(Instr::Output(n));
        m
    }

    /// THE regression this call-site rule exists for: reading stdin was attested
    /// `determinism: deterministic` because the classifier keyed on the NAME
    /// `__mind_read`, which is `Det::Pure` in the registry (it is also how a file
    /// is read). A byte of stdin reaching the program's result is world input.
    #[test]
    fn stdin_read_is_nondeterministic() {
        let fd = crate::ir::ValueId(80);
        let m = read_module(vec![Instr::ConstI64(fd, 0)], fd);
        assert_eq!(
            ir_first_nondeterministic_call(&m).as_deref(),
            Some("__mind_read"),
            "`__mind_read(0, …)` reads stdin and must be named by the classifier"
        );
        assert!(
            !ir_declares_deterministic(&m),
            "a module that reads stdin must NOT attest `deterministic`"
        );
    }

    /// The other two inherited standard streams are the same channel class — a
    /// caller can point either at any file or pipe — so neither is a spelling
    /// that walks past the stdin check.
    #[test]
    fn reading_stdout_or_stderr_descriptors_is_nondeterministic() {
        for stream_fd in [1i64, 2] {
            let fd = crate::ir::ValueId(80);
            let m = read_module(vec![Instr::ConstI64(fd, stream_fd)], fd);
            assert!(
                !ir_declares_deterministic(&m),
                "`__mind_read({stream_fd}, …)` reads an inherited standard stream"
            );
        }
    }

    /// CONTROL — the check must reject the BEHAVIOUR, not every read. A read
    /// from a proven non-standard descriptor keeps the registry's `Det::Pure`
    /// verdict (the file-read line argued in `crate::intrinsics`), so
    /// file-processing programs are not blanket-tainted.
    #[test]
    fn proven_non_stream_descriptor_read_stays_deterministic() {
        let fd = crate::ir::ValueId(80);
        let m = read_module(vec![Instr::ConstI64(fd, 7)], fd);
        assert_eq!(
            ir_first_nondeterministic_call(&m),
            None,
            "a proven non-standard descriptor must not be tainted (over-taint is \
             as dishonest as under-taint)"
        );
        assert!(ir_declares_deterministic(&m));
    }

    /// FAIL CLOSED — a descriptor the call site cannot prove constant may BE
    /// stdin, so it may not be attested deterministic. This is the case
    /// `std/io.mind` makes trivial: `file_read(f, …)` passes `f.fd`, a struct
    /// field load, and `file_read(stdin(), …)` is then a stdin read wearing an
    /// unprovable descriptor.
    #[test]
    fn unprovable_descriptor_fails_closed() {
        // Descriptor from a call result (the shape a `__mind_load_i64(f + 0)`
        // field read, or an `__mind_open(path)`, lowers to).
        let addr = crate::ir::ValueId(80);
        let fd = crate::ir::ValueId(81);
        let m = read_module(
            vec![
                Instr::ConstI64(addr, 4096),
                Instr::Call {
                    dst: fd,
                    name: "__mind_load_i64".to_string(),
                    args: vec![addr],
                },
            ],
            fd,
        );
        assert_eq!(
            ir_first_nondeterministic_call(&m).as_deref(),
            Some("__mind_read"),
            "an unprovable descriptor must fail closed — it can be 0"
        );

        // Descriptor never defined at all (hand-forged mic@3).
        let dangling = crate::ir::ValueId(77);
        let m = read_module(vec![], dangling);
        assert!(
            !ir_declares_deterministic(&m),
            "an undefined descriptor operand must fail closed"
        );
    }

    /// Two different constants bound to one id (malformed or shadowed IR) is not
    /// a proof — the call site cannot know which it sees, so it fails closed.
    #[test]
    fn conflicting_constant_descriptor_fails_closed() {
        let fd = crate::ir::ValueId(80);
        let m = read_module(vec![Instr::ConstI64(fd, 7), Instr::ConstI64(fd, 0)], fd);
        assert!(
            !ir_declares_deterministic(&m),
            "an id with two different constant definitions must not be provable"
        );
    }

    /// Value ids are numbered PER FUNCTION (`src/ir/verify.rs`), so a top-level
    /// `%0 = const.i64 7` must not make a function body's `%0` — here the `fd`
    /// PARAMETER — provably 7. A shared constant environment would let a caller
    /// launder a stdin read through any function that takes a descriptor.
    #[test]
    fn function_body_does_not_inherit_enclosing_constants() {
        let fd = crate::ir::ValueId(0);
        let size = crate::ir::ValueId(1);
        let buf = crate::ir::ValueId(2);
        let n = crate::ir::ValueId(3);

        let mut m = IRModule::new();
        // Top-level namespace: %0 is the constant 7.
        m.instrs.push(Instr::ConstI64(fd, 7));
        m.instrs.push(Instr::Output(fd));
        // Function namespace: %0 is the `fd` parameter — unprovable.
        m.instrs.push(Instr::FnDef {
            name: "slurp".to_string(),
            params: vec![("fd".to_string(), fd)],
            ret_id: Some(n),
            body: vec![
                Instr::Param {
                    dst: fd,
                    name: "fd".to_string(),
                    index: 0,
                },
                Instr::ConstI64(size, 8),
                Instr::Call {
                    dst: buf,
                    name: "__mind_alloc".to_string(),
                    args: vec![size],
                },
                Instr::Call {
                    dst: n,
                    name: "__mind_read".to_string(),
                    args: vec![fd, buf, size, size],
                },
                Instr::Return { value: Some(n) },
            ],
            reap_threshold: None,
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
        });

        assert_eq!(
            ir_first_nondeterministic_call(&m).as_deref(),
            Some("__mind_read"),
            "a function parameter is not a provable descriptor, whatever the \
             enclosing scope binds to the same numeric id"
        );
    }

    /// The descriptor proof must survive the mic@3 round trip, or the verify-side
    /// re-derivation (`mindc verify`, which re-parses the hashed body) would
    /// disagree with the emit-side label and the attestation could be forged by
    /// shipping the artifact instead of the source.
    #[test]
    fn descriptor_verdict_survives_mic3_round_trip() {
        for (fd_value, want_deterministic) in [(0i64, false), (7, true)] {
            let fd = crate::ir::ValueId(80);
            let m = read_module(vec![Instr::ConstI64(fd, fd_value)], fd);
            let reparsed = parse_mic3(&emit_mic3(&m)).expect("mic@3 must re-parse");
            assert_eq!(
                ir_declares_deterministic(&reparsed),
                want_deterministic,
                "re-derived verdict for `__mind_read({fd_value}, …)` must match the \
                 emit-side verdict"
            );
        }
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
            #[cfg(feature = "std-surface")]
            value_types: std::collections::BTreeMap::new(),
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
