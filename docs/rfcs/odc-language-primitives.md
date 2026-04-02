# RFC: Observer-Dependent Cognition — Language Primitives

**Status:** Proposed
**Author:** STARGA, Inc.
**Date:** 2026-04-01
**Roadmap:** Phase 13.5

## Summary

Extend MIND's type system with observation-axis annotations for cognitive and governance computations. MIND already has `axis` for tensor operations — this RFC generalizes it to cognitive contexts where computation results depend on the chosen observation basis.

## Motivation

In quantum mechanics, measurement outcomes depend on the observer's choice of basis. The same principle applies to AI cognition: retrieval results depend on the search axis, consensus depends on which models you ask, governance outcomes depend on which invariants you check.

MIND should make this explicit at the language level.

## Design

### `@axis` annotation
```mind
@axis("semantic")
fn recall(query: Tensor[1, 768]) -> Tensor[K, 768] { ... }

@axis("governance", invariants=[1,2,5,7])
fn verify(txn: Transaction) -> Evidence { ... }
```

### `observe` keyword
```mind
let candidates = speculate(draft_model, context)  // superposition
let result = observe(candidates, axis="verifier_a")  // collapse
```

### Determinism certificates
The compiler emits proof that a code path is:
- `axis_independent` — fully deterministic regardless of observation basis
- `axis_dependent` — result varies with declared axis (must be explicit)

### Axis contracts
Compile-time verification that composed computations use compatible observation bases.

## References
- [ODC Specification v1.0](specs/observer-dependent-cognition.md)
- QBism: Stanford Encyclopedia of Philosophy
- Wang et al., "Decoupling Metacognition from Cognition" (AAAI 2025)
