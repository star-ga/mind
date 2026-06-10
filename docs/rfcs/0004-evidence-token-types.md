# RFC 0004: Compile-Time Evidence Token Types

- **Start Date**: 2026-05-18
- **RFC PR**: TBD
- **Status**: Draft
- **Target Release**: v0.4.0 (post self-hosting prerequisites)
- **Normative reference**: `mind-spec` RFC-0007 (FFI/ABI) for symbol
  interaction; `512-mind` governance model for the invariant identifier
  namespace.

## Summary

Introduce `Evidence[K]` — a zero-size type carrying a compile-time
governance-invariant key `K`. A function's signature declares the
attestations it *produces* and *consumes*. The type-checker runs a DAG
reachability pass that rejects any program where a required
`Evidence[K]` is not produced on every control-flow path reaching its
consumer. A governance-critical MIND program with a missing or
reordered attestation step **does not compile**.

This is the type-level lift of the runtime evidence chain MIND already
emits. It is the one architectural direction no mainstream systems
language (Rust, Zig, Mojo, Pony, F*/Low*) has, and it sits exactly on
MIND's existing invariant + evidence + determinism core.

## Motivation

### Today

- Evidence is emitted at *runtime*. A bypassed or reordered governance
  step produces a broken/short evidence chain that is only detected
  when a verifier replays the chain — after the fact.
- `[invariant(n)]` annotations are opaque identifiers with no
  type-system consequence; nothing forces a caller to have actually
  run the verification a decision depends on.
- The largest residual governance risk is *silent omission*: a
  refactor that drops an attestation emit compiles cleanly and ships.

### After this RFC

- `fn verify(txn: Transaction) -> (Decision, Evidence[Invariant_5])`
  statically declares it produces the attestation for invariant 5.
- `fn audit(d: Decision, e: Evidence[Invariant_5]) -> Report`
  statically *requires* that attestation to exist.
- If the verification path is bypassed, the missing `Evidence[K]`
  value is a **type error**, not a runtime chain break.
- `[invariant(5)]` on a function becomes sugar for adding
  `Evidence[Invariant_5]` to its return type — existing annotations
  gain teeth with zero source churn.

## Guide-level explanation

```mind
// An invariant key lives in the governance namespace (512-mind owns
// the registry; the compiler only needs the nominal identifier).
type Invariant_5 = invariant_key;

pub fn verify(txn: Transaction) -> (Decision, Evidence[Invariant_5]) {
    // ... checks ...
    let ev = emit_evidence(Invariant_5);   // produces the token
    (decision, ev)
}

pub fn audit(d: Decision, e: Evidence[Invariant_5]) -> Report {
    // `e` can only exist if `verify` (or another producer of
    // Evidence[Invariant_5]) ran on every path reaching here.
    build_report(d, e)
}
```

Sugar form — identical semantics, no new syntax to learn:

```mind
[invariant(5)]
pub fn verify(txn: Transaction) -> Decision { ... }
//        desugars to  -> (Decision, Evidence[Invariant_5])
```

`Evidence[K]` is zero-size: it carries no runtime bytes, only the
compile-time obligation. It is `Copy`-equivalent and propagates
through generics automatically — a generic `fn pipeline<E>(.., e: E)`
threads the obligation without special-casing.

## Reference-level explanation

### The type

`Evidence[K]` where `K` is a nominal compile-time key drawn from the
governance invariant namespace. No value-level payload; the type
*is* the proposition "invariant K was attested on this path."

### The check (one pass, after type-checking)

1. Build the per-module **evidence dependency DAG**: nodes are
   functions; an edge `A → B` exists when `B` consumes an
   `Evidence[K]` that `A` produces.
2. For every function that consumes `Evidence[K]`, assert that on
   *every* control-flow path reaching the consumption point a value
   of `Evidence[K]` is in scope (produced locally, received as a
   parameter, or returned by a called producer).
3. Reject with a precise diagnostic naming the missing `K` and the
   offending path if not.

Complexity: one topological reachability pass, `O(V + E)` in
functions and evidence edges. For a typical governance module (tens
of functions) this is sub-microsecond.

### Distinction from proof-carrying code (Necula 1997)

1. PCC attaches proofs to *binaries* for a *host verifier*; evidence
   tokens are checked *at compile time by the same compiler* — no
   external verifier.
2. PCC proofs are general logical formulae; evidence tokens are
   nominal keys tied to declared invariants — SMT-free, `O(1)` to
   check.
3. PCC is post-compilation; evidence tokens are compositional in the
   type system and propagate through generics automatically.

## Compile-speed invariant (the moat)

- Entire feature is module-level `#[cfg(feature = "evidence-types")]`.
  Default build compiles none of it and is byte-identical.
- Tensor-only modules with no `Evidence[K]` in any signature pay
  **zero** cost: the DAG pass early-outs when the module's evidence
  edge set is empty.
- The existing `bench-gate.yml` one-sided +10% regression threshold applies to the three
  headline benches; a dedicated `bench_evidence_dag` sub-benchmark
  observes per-edge cost in isolation, never as a delta to the
  headline numbers.
- No statement-level `cfg`, no runtime dispatch on the hot path.

## Drawbacks

- A third feature gate to maintain alongside `ffi-c-user` and
  `mlir-build`.
- The invariant-key namespace must be agreed with `512-mind`;
  divergence there is a cross-repo coordination cost.
- Over-use could push users toward "evidence-type soup" in
  non-governance code; the sugar form is deliberately the only
  ergonomic entry point so the feature stays scoped to governed
  surfaces.

## Rationale and alternatives

- **Runtime-only (today).** Rejected: silent omission ships.
- **`#[must_use]`-style lint.** Rejected: a lint is advisory and
  bypassable; a type error is not.
- **Full refinement types (Liquid/F\*).** Rejected for this purpose:
  SMT per constraint breaks the µs frontend. Index-refinements
  (a separate future RFC, per the autoresearch) are the
  compile-speed-safe refinement subset; evidence tokens are nominal
  and need no solver.
- **Effect rows carrying evidence.** Considered; deferred. Effect
  rows (a Koka-style two-tier capability system) are a larger
  separate RFC. Evidence tokens are the minimal viable slice that
  delivers the governance guarantee without the effect-system
  surface area.

## Adoption plan

1. Land `Evidence[K]` as a zero-size type + parser support, gated,
   no checker change. Sub-bench added.
2. Land the evidence-DAG reachability pass, gated.
3. Wire `[invariant(n)]` desugaring to `Evidence[Invariant_n]` in
   the return type.
4. 512-mind publishes the canonical invariant-key namespace; mindc
   consumes it as nominal identifiers only.
5. mind-runtime / mind-agents / 512-mind governed paths adopt the
   typed form; runtime emit stays for the evidence *chain*, now
   provably non-omittable.

## Open questions

1. Should `Evidence[K]` be consumable-once (linear) or freely
   copyable? Lean: copyable — the obligation is "was attested," not
   "attested exactly once"; linearity is a separate concern.
2. Cross-module evidence: does a producer in module A satisfy a
   consumer in module B automatically, or must the obligation be
   re-declared at the module boundary? Lean: re-declared at the
   boundary (consistent with RFC 0002's explicit-export discipline).
3. Interaction with the self-hosting bootstrap: the evidence-DAG
   pass must itself be expressible in MIND for stage-1. It is —
   it is a topological sort over integer-keyed nodes, no SMT, no
   floating point.
