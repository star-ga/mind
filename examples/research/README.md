# Scientific-computing research examples (not certified theorems)

## Kakeya-inspired directional diagnostic
`kakeya_field_diagnostic.mind` is a **minimal integer/control-flow compiler smoke**, not an implementation of the Kakeya construction, the restriction conjecture, or local smoothing. It counts eight symbolic directions over three steps and returns 24. It is intentionally independent of floats/tensors, since the native-ELF self-hosted subset and full MLIR tensor backend have different readiness.

Suggested validation (review compiler CLI syntax at the checked-out commit):

```sh
cargo run --bin mindc -- examples/research/kakeya_field_diagnostic.mind --emit-ir
```

Future scientifically useful example: finite 2D Q16.16 transport/diffusion comparison with a provable discrete energy bound, explicit boundary conditions, canonical rounding, known-answer fixtures and cross-target parity. Do **not** advertise this until compiled and measured.

## October 26 discoveries research intake
- Local smoothing / Kakeya / Stein restriction: relevant as mathematical inspiration for **oscillatory** propagation; a plain diffusion update does not satisfy the theorem hypotheses automatically.
- ETH: investigate bounded exact-search encodings and compare to SAT/SMT, not blanket inference speedups.
- Yang–Mills: no transfer without a gauge-theoretic model and verified discretization.

**Promotion gate:** independent paper review, precise assumptions, reproducible executable fixture, deterministic parity, performance comparison, and no unverified theorem claims.
