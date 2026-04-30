# Copyright 2025-2026 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
# Part of the MIND project (Machine Intelligence Native Design).

"""JAX → MIND transpiler.

JAX programs are lowered through XLA HLO before reaching the bridge.
The function below accepts an HLO text dump (the format JAX writes
when you call ``jax.jit(f).lower(args).compile().runtime_executable
.hlo_modules()[0].to_string()``) and walks the instructions linearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .ir import MindModule, MindOp, OpKind, kind_from_str


_HLO_TO_MIND = {
    "dot": OpKind.MATMUL,
    "convolution": OpKind.CONV2D,
    "add": OpKind.ADD,
    "subtract": OpKind.SUB,
    "multiply": OpKind.MUL,
    "divide": OpKind.DIV,
    "maximum": OpKind.RELU,  # only when one side is zero — caller must filter
    "reshape": OpKind.RESHAPE,
    "transpose": OpKind.TRANSPOSE,
    "reduce-add": OpKind.REDUCE_SUM,
    "reduce": OpKind.REDUCE_SUM,
}


@dataclass
class JaxUnsupported:
    hlo_op: str
    output: str


@dataclass
class JaxResult:
    module: MindModule
    unsupported: List[JaxUnsupported]


def jax_to_mind(
    hlo_path: str | Path | None = None,
    *,
    hlo_text: str | None = None,
    module_name: str = "transpiled_jax",
) -> JaxResult:
    """Lower XLA HLO text to MIND.

    Either pass an ``hlo_path`` to a file emitted by JAX or supply
    ``hlo_text`` directly. Exactly one of the two is required.
    """
    if hlo_path is not None and hlo_text is not None:
        raise ValueError("pass either hlo_path or hlo_text, not both")
    if hlo_path is None and hlo_text is None:
        raise ValueError("hlo_path or hlo_text is required")

    text = hlo_text if hlo_text is not None else Path(hlo_path).read_text()  # type: ignore[arg-type]
    module = MindModule(name=module_name)
    unsupported: List[JaxUnsupported] = []
    last_output: str = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("HloModule"):
            continue
        if line.startswith("ENTRY") or line.endswith("{") or line.startswith("}"):
            continue

        parsed = _parse_hlo_line(line)
        if parsed is None:
            continue
        out, op_name, in_names, shape = parsed

        kind = _HLO_TO_MIND.get(op_name) or kind_from_str(op_name)
        if kind is None:
            unsupported.append(JaxUnsupported(hlo_op=op_name, output=out))
            continue

        module.add_op(
            MindOp(
                name=out,
                kind=kind,
                inputs=tuple(in_names),
                output=out,
                shape=tuple(shape),
                dtype="f32",
            )
        )
        last_output = out

    if last_output:
        module.set_output(last_output)

    return JaxResult(module=module, unsupported=unsupported)


def _parse_hlo_line(
    line: str,
) -> Tuple[str, str, List[str], List[int]] | None:
    """Parse a single HLO instruction line, e.g.
       %c = f32[16,16] dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    """
    if "=" not in line:
        return None
    lhs, _, rhs = line.partition("=")
    out = lhs.strip()
    rhs = rhs.strip()
    # f32[16,16] dot(%a, %b), ...
    bracket = rhs.find("[")
    if bracket == -1:
        return None
    rb = rhs.find("]", bracket)
    if rb == -1:
        return None
    shape_str = rhs[bracket + 1 : rb]
    rest = rhs[rb + 1 :].strip()
    op_part, _, _ = rest.partition(",")
    paren = op_part.find("(")
    if paren == -1:
        return None
    op_name = op_part[:paren].strip()
    args_str = op_part[paren + 1 :].rstrip(")")
    in_names = [a.strip() for a in args_str.split(",") if a.strip()]
    try:
        shape = [int(s) for s in shape_str.split(",") if s.strip()]
    except ValueError:
        shape = []
    return out, op_name, in_names, shape
