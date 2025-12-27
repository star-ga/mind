# MIND IR core

MIND lowers parsed programs into a lightweight SSA-style IR that is intentionally
deterministic. The IR is used by static autodiff, the interpreter, and the MLIR
export path. This document tracks the hardening work that keeps the IR stable
for downstream consumers.

## Canonicalization

The `canonicalize_module` pass lives in `src/opt/ir_canonical.rs` and performs
deterministic cleanups:

- Orders operands for commutative arithmetic to remove incidental variance.
- Removes dead instructions that do not contribute to outputs.
- Performs simple constant folding on integer binary ops.
- Resets `next_id` so SSA identifiers remain dense and reproducible.

The pass is pure and idempotent—running it multiple times produces identical
results.

## Verifier

The IR verifier (`src/ir/verify.rs`) enforces SSA discipline and basic operand
sanity:

- Every SSA value is defined exactly once and before it is used.
- An `Output` instruction must be present.
- `next_id` must be in sync with the highest defined value.
- Numeric operands (axes, etc.) are checked for obviously invalid values.

The verifier never panics on malformed IR; it returns structured
`IrVerifyError` values for deterministic diagnostics.

## Stable pretty-printer

`format_ir_module` (`src/ir/print.rs`) renders IR in a stable, human-readable
form that is suitable for snapshot tests and debugging. Instructions are emitted
in deterministic order with consistent SSA identifiers.

## Backend preparation

Use `prepare_ir_for_backend` to combine verification and canonicalization before
handing IR to downstream consumers:

```rust
let mut ir = lower_to_ir(&parsed_module);
prepare_ir_for_backend(&mut ir)?;
```

This helper ensures the IR is valid and canonical before autodiff, MLIR
lowering, or interpretation.
