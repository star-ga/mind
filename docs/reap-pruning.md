# REAP Expert Pruning

The MIND compiler implements compile-time dead-expert elimination for
Mixture-of-Experts (MoE) architectures.  This is based on the REAP
approach (arXiv:2510.13999) — Redundancy-Aware Expert Pruning.

## The Problem

In MoE models, each input is routed to a small subset of expert functions.
At compile time it is often statically known that certain experts will never
be selected (e.g. experts whose activation statistics fall below a routing
threshold across the training distribution).  Keeping dead experts in the
compiled module wastes memory bandwidth, I-cache space, and compute on
wafer-scale hardware where expert dispatch overhead is amplified by the
network-on-chip cost.

## Usage

Annotate expert functions with `[reap_threshold(t)]` where `t ∈ [0.0, 1.0)`:

```
[reap_threshold(0.5)]
fn expert_7(x: tensor<f32[1024]>) -> tensor<f32[1024]> {
    return matmul(x, W7);
}
```

The threshold value is stored on the function definition and propagated to
the IR.  At canonicalization time, the compiler scans for experts with zero
call sites and replaces their bodies with no-op tombstones.

## Threshold Semantics

The `reap_threshold` attribute is a declaration that the expert is a
candidate for pruning.  The compiler applies the conservative baseline:

- **Zero call sites → tombstone.** An expert that is never called (i.e.,
  has no `Call` instruction referencing it and no other function definitions
  that could dispatch to it) is considered dead and its body is eliminated.

- **Any potential callers → preserved.** If the module contains any non-REAP
  function definitions alongside the expert, the compiler conservatively
  assumes they could dispatch to the expert and does not prune.

Threshold-guided pruning (retain top-k experts by activation weight; prune
those whose weight is below `t`) requires routing-function analysis and is
a planned follow-on.

## Valid Threshold Range

Thresholds must be in `[0.0, 1.0)`.  A value of `1.0` or above is silently
treated as absent (no annotation).

```
[reap_threshold(0.0)]   // valid — always a REAP candidate
[reap_threshold(0.5)]   // valid — typical 50% sparsity target
[reap_threshold(0.9)]   // valid — aggressive pruning threshold
[reap_threshold(1.0)]   // ignored — treated as no annotation
```

## Feature Gating

The dead-expert DCE sub-pass is feature-gated: it only runs when at least
one function in the module carries a `reap_threshold` attribute.  Non-MoE
modules pay zero extra analysis cost.

This preserves the compile-speed moat for standard feed-forward and
convolutional networks.

## SSA Integrity

Dead expert bodies are not removed; they are replaced with a single
`ConstI64(ret_id, 0)` tombstone.  This preserves the SSA value ID that the
expert's return value occupied, avoiding renumbering and keeping the module
well-formed.

## Wafer-Scale Accelerator Impact

Compile-time expert pruning is particularly valuable on wafer-scale
accelerators because:

1. Expert dispatch is a network-on-chip operation — unused experts still
   occupy tile addresses and consume routing bandwidth.
2. Static pruning lets the place-and-route pass pack live experts into
   contiguous tile regions, maximizing cache locality.
3. Dead expert elimination reduces binary size, which matters when fitting
   a large MoE within the on-wafer SRAM budget.

## References

- arXiv:2510.13999 — REAP: Redundancy-Aware Expert Pruning for MoE models
- arXiv:2202.04305 — MLIR sparse_tensor dialect (layout taxonomy)
- docs/sparse-tensor-types.md — type surface for sparse tensors in MIND
