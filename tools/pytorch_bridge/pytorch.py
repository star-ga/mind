# Copyright 2025-2026 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
# Part of the MIND project (Machine Intelligence Native Design).

"""PyTorch → MIND transpiler.

Strategy: take an ONNX-exported model (``torch.onnx.export``) and walk
its node list. Every recognised op kind becomes a ``MindOp``; everything
else is collected as an ``UnsupportedOp`` so the AI-assist pass can pick
it up.

The function does not import torch — it only touches ONNX through
:mod:`onnx` (a small pure-Python package). This keeps the bridge usable
in CI without the multi-GB torch install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from .ir import MindModule, MindOp, OpKind, kind_from_str


@dataclass
class UnsupportedOp:
    pytorch_kind: str
    onnx_op_type: str
    inputs: Tuple[str, ...]
    output: str


@dataclass
class TranspileResult:
    module: MindModule
    unsupported: List[UnsupportedOp]


def pytorch_to_mind(
    onnx_path: str | Path,
    *,
    module_name: str = "transpiled",
) -> TranspileResult:
    """Lower an ONNX file to a MIND module.

    The signature accepts a path because it is the lowest-friction way
    to handle very large models (>500 MB) without holding both the
    torch state dict and the bridge IR in memory at once.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")

    nodes, inputs, output_name = _read_onnx_nodes(path)

    module = MindModule(name=module_name)
    for n, shape, dtype in inputs:
        module.add_input(n, shape, dtype)

    unsupported: List[UnsupportedOp] = []
    for op_type, in_names, out_name, shape, dtype in nodes:
        kind = kind_from_str(op_type)
        if kind is None:
            unsupported.append(
                UnsupportedOp(
                    pytorch_kind=op_type,
                    onnx_op_type=op_type,
                    inputs=tuple(in_names),
                    output=out_name,
                )
            )
            continue
        module.add_op(
            MindOp(
                name=out_name,
                kind=kind,
                inputs=tuple(in_names),
                output=out_name,
                shape=tuple(shape),
                dtype=dtype,
            )
        )

    if output_name:
        module.set_output(output_name)

    return TranspileResult(module=module, unsupported=unsupported)


def _read_onnx_nodes(
    path: Path,
) -> Tuple[
    List[Tuple[str, List[str], str, List[int], str]],
    List[Tuple[str, Tuple[int, ...], str]],
    str,
]:
    """Minimal ONNX reader that tolerates a missing onnx package.

    Detection rule: paths ending in ``.onnxtxt`` always use the textual
    path. Anything else tries the proto reader first; if onnx isn't
    installed (or the proto can't be parsed) we still fall back.
    """
    if path.suffix.lower() == ".onnxtxt":
        return _read_textual(path)
    try:
        import onnx  # type: ignore[import-not-found]

        proto = onnx.load(str(path))
        graph = proto.graph
        inputs = [
            (i.name, _shape_from_proto(i), _dtype_from_proto(i))
            for i in graph.input
        ]
        nodes = []
        for n in graph.node:
            shape = [1]  # ONNX shape inference happens upstream
            dtype = "f32"
            nodes.append(
                (n.op_type, list(n.input), (n.output[0] if n.output else ""), shape, dtype)
            )
        out_name = graph.output[0].name if graph.output else ""
        return nodes, inputs, out_name
    except (ImportError, Exception):
        return _read_textual(path)


def _shape_from_proto(value_info) -> Tuple[int, ...]:  # type: ignore[no-untyped-def]
    try:
        dims = value_info.type.tensor_type.shape.dim
        return tuple(int(d.dim_value) for d in dims if d.dim_value > 0)
    except Exception:
        return tuple()


def _dtype_from_proto(value_info) -> str:  # type: ignore[no-untyped-def]
    try:
        elem = value_info.type.tensor_type.elem_type
        return _ONNX_ELEM_TYPE.get(int(elem), "f32")
    except Exception:
        return "f32"


_ONNX_ELEM_TYPE = {
    1: "f32",   # FLOAT
    2: "u8",    # UINT8
    3: "i8",    # INT8
    4: "u16",   # UINT16
    5: "i16",   # INT16
    6: "i32",   # INT32
    7: "i64",   # INT64
    10: "f16",  # FLOAT16
    11: "f64",  # DOUBLE
    16: "bf16", # BFLOAT16
}


def _read_textual(
    path: Path,
) -> Tuple[
    List[Tuple[str, List[str], str, List[int], str]],
    List[Tuple[str, Tuple[int, ...], str]],
    str,
]:
    """Read an .onnxtxt-style file used by the bridge tests.

    Format (one entry per line):

        INPUT name shape=1x16 dtype=f32
        OP    matmul %a %b -> %c shape=1x16
        OUT   %c
    """
    inputs: List[Tuple[str, Tuple[int, ...], str]] = []
    nodes: List[Tuple[str, List[str], str, List[int], str]] = []
    out_name = ""

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        kind, _, rest = line.partition(" ")
        kind = kind.upper()
        if kind == "INPUT":
            tokens = rest.split()
            name = tokens[0]
            shape: Tuple[int, ...] = tuple()
            dtype = "f32"
            for kv in tokens[1:]:
                if kv.startswith("shape="):
                    shape = tuple(int(s) for s in kv[len("shape=") :].split("x") if s)
                elif kv.startswith("dtype="):
                    dtype = kv[len("dtype=") :]
            inputs.append((name, shape, dtype))
        elif kind == "OP":
            tokens = rest.split()
            op = tokens[0]
            arrow_idx = tokens.index("->")
            in_names = tokens[1:arrow_idx]
            output = tokens[arrow_idx + 1]
            shape_list: List[int] = []
            dtype = "f32"
            for kv in tokens[arrow_idx + 2 :]:
                if kv.startswith("shape="):
                    shape_list = [int(s) for s in kv[len("shape=") :].split("x") if s]
                elif kv.startswith("dtype="):
                    dtype = kv[len("dtype=") :]
            nodes.append((op, in_names, output, shape_list, dtype))
        elif kind == "OUT":
            out_name = rest.strip()

    return nodes, inputs, out_name


def emit_unsupported_summary(unsupported: Iterable[UnsupportedOp]) -> str:
    """Human-readable summary suitable for build-log output."""
    items = list(unsupported)
    if not items:
        return "✓ All ops lowered natively."
    lines = ["⚠ Unsupported ops (forwarded to AI-assist proof pass):"]
    for u in items:
        lines.append(f"  - {u.pytorch_kind} ({', '.join(u.inputs)}) → {u.output}")
    return "\n".join(lines)
