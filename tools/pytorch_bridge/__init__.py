# Copyright 2025-2026 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
# Part of the MIND project (Machine Intelligence Native Design).

"""PyTorch / JAX → MIND bridge tooling.

This package implements the migration path documented in
`docs/migration.md` and tracked under "Python Bridge Tooling" in
`mind-spec/spec/v1.0/future-extensions.md`.

Two entry points:

* :func:`pytorch_to_mind`  — accepts a `torch.nn.Module` (or an exported
  ONNX file) and produces a `.mind` source string.
* :func:`jax_to_mind`      — accepts a JAX-compiled function (or an
  emitted XLA HLO text) and produces a `.mind` source string.

The transpiler operates at the graph level — it walks the upstream IR
once, classifies each op into MIND's supported set, and emits MIND
text. Unsupported ops produce a structured ``UnsupportedOp`` event so
the caller can choose between (a) failing the build, (b) emitting a
host-callback shim, or (c) collecting the op for an AI-assisted proof
pass.

The translator does **not** import torch / jax at module load time —
it shells out via the ONNX / HLO text format so this package can run
on a CI machine that doesn't have the heavy ML frameworks installed.
"""

from .ir import MindModule, MindOp, OpKind
from .pytorch import pytorch_to_mind
from .jax import jax_to_mind
from .ai_proof import build_unsat_prompt, UnsatContext

__all__ = [
    "MindModule",
    "MindOp",
    "OpKind",
    "UnsatContext",
    "build_unsat_prompt",
    "pytorch_to_mind",
    "jax_to_mind",
]
