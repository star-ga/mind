# Compiler research consumer of an offline hypothesis service

Status: proposed / review required / default OFF. No implementation or deployment is claimed.

The offline research service is optional. It may generate source-grounded candidate hypotheses for parsing, IR, MLIR lowering, static tensors, native backends, reproducibility and kernel algorithms, but MIND remains the sole authority for compiler semantics, correctness and releases.

- [ ] Pilot fixed-corpus hypothesis selection from the research pipeline and the orchestration layer versus the current proposer on bounded compiler tasks.
- [ ] Require independent reference oracles, frozen tests, artifact hashes, cross-substrate determinism, complete correctness and performance regression checks before considering changes.
- [ ] Reject any candidate that alters semantics, compiler pins, reproducible outputs or the evidence contract without explicit review.
- [ ] Adopt domain-specific model retraining only if held-out matched-budget comparisons demonstrate independently verified wins.
- [ ] Keep Python/ML exploration outside the required pure-MIND compiler implementation and document replacement boundaries.

Research scores do not establish correctness, novelty, source rights or release readiness.
