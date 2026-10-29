# Compiler research consumer of MIND-Lab

Status: proposed / review required / default OFF. No implementation or deployment is claimed.
Sources: https://arxiv.org/abs/2603.01092 ; https://github.com/alejandrohdez00/alien-space-of-science ; https://github.com/tiptreesystems/lacuna-research-mcp

MIND-Lab is an optional offline research/optimization service for the compiler ecosystem. It may generate source-grounded candidate hypotheses for parsing, IR, MLIR lowering, static tensors, native backends, reproducibility and kernel algorithms, but MIND remains the sole authority for compiler semantics, correctness and releases.

- [ ] Pilot fixed-corpus Alien/Lacuna hypothesis selection versus current MIND-Lab alginv proposer on bounded compiler tasks.
- [ ] Require independent reference oracles, frozen tests, artifact hashes, cross-substrate determinism, complete correctness and performance regression checks before considering changes.
- [ ] Reject any candidate that alters semantics, compiler pins, reproducible outputs or the evidence contract without explicit review.
- [ ] Adopt domain-specific model retraining only if held-out matched-budget comparisons demonstrate independently verified wins.
- [ ] Keep Python/ML exploration outside the required pure-MIND compiler implementation and document replacement boundaries.

Research scores do not establish correctness, novelty, source rights or release readiness.
