# Copyright 2025-2026 STARGA Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
# Part of the MIND project (Machine Intelligence Native Design).

"""Unit tests for tools/pytorch_bridge."""

import sys
import textwrap
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[2]))

from pytorch_bridge import (  # noqa: E402
    UnsatContext,
    build_unsat_prompt,
    jax_to_mind,
    pytorch_to_mind,
)
from pytorch_bridge.ir import MindModule, MindOp, OpKind, kind_from_str  # noqa: E402
from pytorch_bridge.pytorch import emit_unsupported_summary  # noqa: E402


class IrTests(unittest.TestCase):
    def test_kind_from_str_matmul_aliases(self):
        for name in ("matmul", "MatMul", "mm", "linear", "GEMM"):
            self.assertEqual(kind_from_str(name), OpKind.MATMUL, name)

    def test_kind_from_str_unknown(self):
        self.assertIsNone(kind_from_str("does_not_exist"))

    def test_module_emits_canonical_text(self):
        m = MindModule(name="m")
        m.add_input("a", (1, 16))
        m.add_input("b", (16, 16))
        m.add_op(
            MindOp(
                name="c",
                kind=OpKind.MATMUL,
                inputs=("a", "b"),
                output="c",
                shape=(1, 16),
            )
        )
        m.set_output("c")
        out = m.emit()
        self.assertIn("fn m(", out)
        self.assertIn("matmul(a, b)", out)
        self.assertIn("return c", out)


class PytorchBridgeTests(unittest.TestCase):
    def _write_textual(self, body: str) -> Path:
        import tempfile
        fd, name = tempfile.mkstemp(suffix=".onnxtxt")
        import os
        os.close(fd)
        p = Path(name)
        p.write_text(textwrap.dedent(body))
        return p

    def test_lowers_supported_ops(self):
        path = self._write_textual(
            """
            INPUT a shape=1x16 dtype=f32
            INPUT b shape=16x16 dtype=f32
            OP matmul a b -> c shape=1x16
            OP relu c -> d shape=1x16
            OUT d
            """
        )
        result = pytorch_to_mind(path, module_name="net")
        self.assertEqual(result.unsupported, [])
        self.assertEqual(result.module.output, "d")
        self.assertEqual(len(result.module.ops), 2)
        self.assertEqual(result.module.ops[0].kind, OpKind.MATMUL)
        self.assertEqual(result.module.ops[1].kind, OpKind.RELU)

    def test_collects_unsupported(self):
        path = self._write_textual(
            """
            INPUT a shape=4x4 dtype=f32
            OP exotic_op a -> b shape=4x4
            OUT b
            """
        )
        result = pytorch_to_mind(path)
        self.assertEqual(len(result.unsupported), 1)
        self.assertEqual(result.unsupported[0].onnx_op_type, "exotic_op")
        # The op is not lowered into the module.
        self.assertEqual(len(result.module.ops), 0)

    def test_unsupported_summary_renders(self):
        path = self._write_textual(
            """
            INPUT a shape=4x4
            OP exotic a -> b shape=4x4
            """
        )
        r = pytorch_to_mind(path)
        msg = emit_unsupported_summary(r.unsupported)
        self.assertIn("exotic", msg)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            pytorch_to_mind("/tmp/__no_such_file__.onnxtxt")


class JaxBridgeTests(unittest.TestCase):
    def test_lowers_dot_to_matmul(self):
        hlo = textwrap.dedent(
            """
            HloModule test
            ENTRY main {
              %a = f32[16,16] parameter(0)
              %b = f32[16,16] parameter(1)
              %c = f32[16,16] dot(%a, %b)
            }
            """
        )
        result = jax_to_mind(hlo_text=hlo)
        kinds = [op.kind for op in result.module.ops]
        self.assertIn(OpKind.MATMUL, kinds)

    def test_collects_unsupported(self):
        hlo = "%c = f32[16,16] mystery(%a, %b)"
        result = jax_to_mind(hlo_text=hlo)
        self.assertEqual(len(result.unsupported), 1)
        self.assertEqual(result.unsupported[0].hlo_op, "mystery")

    def test_either_path_or_text(self):
        with self.assertRaises(ValueError):
            jax_to_mind()
        with self.assertRaises(ValueError):
            jax_to_mind(hlo_path="x", hlo_text="y")


class AiProofTests(unittest.TestCase):
    def test_prompt_is_deterministic(self):
        ctx = UnsatContext(
            constraint="shape(a) == shape(b)",
            location="src/model.mind:42",
            expected="tensor<1x16xf32>",
            actual="tensor<1x32xf32>",
            notes=["one possible fix: insert reshape"],
        )
        a = build_unsat_prompt(ctx)
        b = build_unsat_prompt(ctx)
        self.assertEqual(a, b)
        self.assertIn("UNSAT", a)
        self.assertIn("tensor<1x16xf32>", a)
        self.assertIn("tensor<1x32xf32>", a)


if __name__ == "__main__":
    unittest.main()
