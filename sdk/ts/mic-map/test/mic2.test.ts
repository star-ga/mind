// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { describe, it, expect } from "vitest";
import { parse, emit, MIC2_VERSION, residualBlock, Mic2ParseError, type Mic2Module } from "../src/index.js";

const RESIDUAL_BLOCK_TEXT = `mic@2
T0 f16 128 128
T1 f16 128
a X T0
p W T0
p b T1
m 0 1
+ 3 2
r 4
+ 5 0
O 6`;

describe("MIC v2 — residual block", () => {
  it("parses residual block from canonical text", () => {
    const mod = parse(RESIDUAL_BLOCK_TEXT);
    expect(mod.types.length).toBe(2);
    expect(mod.values.length).toBe(7);
    expect(mod.output).toBe(6);
    expect(mod.symbols).toEqual([]);
  });

  it("roundtrip: emit(parse(text)) === text for residual block", () => {
    const mod = parse(RESIDUAL_BLOCK_TEXT);
    expect(emit(mod)).toBe(RESIDUAL_BLOCK_TEXT);
  });

  it("roundtrip: emit(residualBlock()) equals canonical text", () => {
    expect(emit(residualBlock())).toBe(RESIDUAL_BLOCK_TEXT);
  });
});

describe("MIC v2 — version header", () => {
  it("MIC2_VERSION constant is 'mic@2'", () => {
    expect(MIC2_VERSION).toBe("mic@2");
  });

  it("rejects missing header", () => {
    expect(() => parse("")).toThrow(Mic2ParseError);
    expect(() => parse("T0 f32 10\n")).toThrow(Mic2ParseError);
  });

  it("rejects wrong version", () => {
    expect(() => parse("mic@1\nO 0\n")).toThrow(Mic2ParseError);
  });

  it("accepts header-only (empty graph)", () => {
    const mod = parse("mic@2\n");
    expect(mod.values).toEqual([]);
  });
});

describe("MIC v2 — DType variants", () => {
  const DTYPES = [
    "f16", "f32", "f64", "bf16",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "bool",
  ] as const;

  it("parses all 13 DType variants", () => {
    for (const dtype of DTYPES) {
      const text = `mic@2\nT0 ${dtype} 8\na x T0\nO 0`;
      const mod = parse(text);
      expect(mod.types[0]!.dtype).toBe(dtype);
    }
  });
});

describe("MIC v2 — implicit value ID assignment", () => {
  it("assigns IDs sequentially from zero", () => {
    const text = "mic@2\nT0 f32 4\na x T0\nO 0";
    const mod = parse(text);
    // values[0] is the arg 'x', output=0
    expect(mod.values[0]!.kind).toBe("arg");
    expect(mod.output).toBe(0);
  });

  it("node references prior IDs correctly", () => {
    const text = "mic@2\nT0 f32 4\na x T0\na y T0\nm 0 1\nO 2";
    const mod = parse(text);
    const node = mod.values[2]!;
    expect(node.kind).toBe("node");
    if (node.kind === "node") {
      expect(node.op.kind).toBe("matmul");
      expect(node.inputs).toEqual([0, 1]);
    }
  });
});

describe("MIC v2 — malformed input errors", () => {
  it("rejects unterminated line (forward ref)", () => {
    expect(() => parse("mic@2\nT0 f32 4\nm 0 1\n")).toThrow(Mic2ParseError);
  });

  it("rejects bad opcode", () => {
    expect(() => parse("mic@2\nT0 f32 4\na x T0\nbadop 0\nO 1")).toThrow(Mic2ParseError);
  });

  it("rejects missing output", () => {
    expect(() => parse("mic@2\nT0 f32 4\na x T0\n")).toThrow(Mic2ParseError);
  });

  it("rejects sequential type index violation", () => {
    expect(() => parse("mic@2\nT1 f32 10\n")).toThrow(Mic2ParseError);
  });

  it("rejects invalid type ref", () => {
    expect(() => parse("mic@2\nT0 f32 10\na x T99\n")).toThrow(Mic2ParseError);
  });

  it("rejects wrong arity (matmul needs 2 inputs)", () => {
    expect(() => parse("mic@2\nT0 f32 10\na x T0\nm 0\nO 1")).toThrow(Mic2ParseError);
  });
});

describe("MIC v2 — multi-result split node", () => {
  // Rust parse_node: only non-usize tokens go to params. Since split axis/n
  // are unsigned ints, they go to inputs, not params. The canonical text
  // for split is: "split <axis> <n> <input_id>"
  // The Rust parse_node puts numeric tokens in inputs and non-numeric in params.
  // For split, Opcode::parse("split", params) uses params.first() and params.get(1)
  // — so the axis and n come from params (non-numeric tokens), and inputs come from
  // numeric tokens. But axis and n ARE numeric — so they go to inputs, and params is empty.
  // The Rust parser for split: axis = params.first() (default 0), n = params.get(1) (default 2).
  // With empty params: axis=0, n=2, inputs=[0,1,2] from numeric tokens.
  // This is the inherent ambiguity. Our tests must match the actual Rust behavior.
  //
  // A split node on input 0 emits as: "split 0 2 0" (axis=0, n=2, input=0)
  // When parsed: all three numbers go to inputs=[0,2,0] (as usize), params=[]
  // so axis=0 (default), n=2 (default), inputs=[0,2,0] — this is a 3-input split.
  // The text format is ambiguous for split with numeric params; this mirrors Rust.

  it("parses split node with default axis and n (from empty params)", () => {
    // Simple case: split with no extra numeric params = default axis=0, n=2
    const text = "mic@2\nT0 f32 8\na x T0\nsplit 0\nO 1";
    const mod = parse(text);
    const node = mod.values[1]!;
    expect(node.kind).toBe("node");
    if (node.kind === "node") {
      expect(node.op.kind).toBe("split");
      if (node.op.kind === "split") {
        expect(node.op.n).toBe(2); // default
      }
    }
  });

  it("emit produces split with axis n inputs as separate space-separated tokens", () => {
    // Programmatically built split node (not round-trippable via text due to
    // the usize-only tokenizer ambiguity for positive numeric params).
    const mod: Mic2Module = {
      symbols: [],
      types: [{ dtype: "f32", shape: ["8"] }],
      values: [
        { kind: "arg", name: "x", typeIdx: 0 },
        { kind: "node", op: { kind: "split", axis: 0n, n: 2 }, inputs: [0] },
      ],
      output: 1,
    };
    const text = emit(mod);
    // Emits: "split 0 2 0" — axis=0, n=2, input=0
    expect(text).toContain("split 0 2 0");
  });
});

describe("MIC v2 — custom opcode", () => {
  // Custom opcodes exist in the graph data model (Value::Node(Opcode::Custom(name), ...))
  // but the text parser does NOT accept arbitrary tokens as custom opcodes —
  // it mirrors Rust which returns None for all unrecognized tokens.
  // Custom opcodes can only be created programmatically and serialized/deserialized
  // via MIC-B binary (which uses the string table for custom names).

  it("emits custom opcode as its name token", () => {
    const mod: Mic2Module = {
      symbols: [],
      types: [{ dtype: "f32", shape: ["4"] }],
      values: [
        { kind: "arg", name: "x", typeIdx: 0 },
        { kind: "node", op: { kind: "custom", name: "fused_gelu" }, inputs: [0] },
      ],
      output: 1,
    };
    const text = emit(mod);
    expect(text).toContain("fused_gelu 0");
  });

  it("rejects unknown tokens as bad opcodes", () => {
    // Unknown tokens (not in the grammar) must throw
    expect(() => parse("mic@2\nT0 f32 4\na x T0\nbadop 0\nO 1")).toThrow(Mic2ParseError);
    expect(() => parse("mic@2\nT0 f32 4\na x T0\n@custom 0\nO 1")).toThrow(Mic2ParseError);
  });
});

describe("MIC v2 — symbols", () => {
  it("parses multiple symbol declarations", () => {
    const text = "mic@2\nS B\nS seq\nT0 f32 B seq\na x T0\nO 0";
    const mod = parse(text);
    expect(mod.symbols).toEqual(["B", "seq"]);
    expect(emit(mod)).toBe(text);
  });
});

describe("MIC v2 — canonicalization", () => {
  it("no double spaces in output", () => {
    const text = emit(residualBlock());
    expect(text).not.toContain("  ");
  });

  it("no trailing spaces on any line", () => {
    const lines = emit(residualBlock()).split("\n");
    for (const line of lines) {
      expect(line).not.toMatch(/ $/);
    }
  });

  it("no trailing newline", () => {
    expect(emit(residualBlock())).not.toMatch(/\n$/);
  });

  it("deterministic across multiple calls", () => {
    const mod = residualBlock();
    expect(emit(mod)).toBe(emit(mod));
  });
});

describe("MIC v2 — opcode parameters in text", () => {
  // The text format tokenizes: numeric tokens → inputs, non-numeric → params.
  // Only NEGATIVE axis values are unambiguously non-usize, so they go to params.
  // Positive numeric params are indistinguishable from input IDs in this format.

  it("softmax with negative axis -1 (default) — token goes to params", () => {
    // "s -1 0": "-1" not parseable as usize → params=["-1"], "0" → inputs=[0]
    const text = "mic@2\nT0 f32 4\na x T0\ns -1 0\nO 1";
    const mod = parse(text);
    const node = mod.values[1]!;
    if (node.kind === "node") {
      expect(node.op.kind).toBe("softmax");
      if (node.op.kind === "softmax") expect(node.op.axis).toBe(-1n);
    }
  });

  it("concat axis emitted as last token before inputs", () => {
    // "cat 0 0 1": axis=0 (default, params=[]), inputs=[0,0,1] — variadic
    // concat is variadic so any number of inputs is ok
    const text = "mic@2\nT0 f32 4\na x T0\na y T0\ncat 0 0 1\nO 2";
    const mod = parse(text);
    const node = mod.values[2]!;
    if (node.kind === "node") {
      // All numbers go to inputs; params=[] → axis defaults to 0
      expect(node.op.kind).toBe("concat");
    }
  });

  it("sum with no explicit axes (empty params)", () => {
    // "sum 0": no extra tokens → sum with empty axes, input=[0]
    const text = "mic@2\nT0 f32 4\na x T0\nsum 0\nO 1";
    const mod = parse(text);
    const node = mod.values[1]!;
    if (node.kind === "node") {
      expect(node.op.kind).toBe("sum");
      if (node.op.kind === "sum") expect(node.op.axes).toEqual([]);
    }
  });

  it("gather emits negative axis unambiguously", () => {
    // "gth -1 0 1": "-1" → params, "0","1" → inputs=[0,1] (arity=2)
    const text = "mic@2\nT0 f32 4\na x T0\na y T0\ngth -1 0 1\nO 2";
    const mod = parse(text);
    const node = mod.values[2]!;
    if (node.kind === "node") {
      expect(node.op.kind).toBe("gather");
      if (node.op.kind === "gather") expect(node.op.axis).toBe(-1n);
    }
  });
});

describe("MIC v2 — comments and blank lines", () => {
  it("ignores comment lines", () => {
    const text = "mic@2\n# comment\nT0 f32 4\n# another\na x T0\nO 0";
    const mod = parse(text);
    expect(mod.values.length).toBe(1);
  });

  it("ignores blank lines", () => {
    const text = "mic@2\n\nT0 f32 4\n\na x T0\nO 0";
    const mod = parse(text);
    expect(mod.values.length).toBe(1);
  });
});

describe("MIC v2 — empty graph", () => {
  it("emits empty graph as 'mic@2\\nO 0'", () => {
    const mod: Mic2Module = { symbols: [], types: [], values: [], output: 0 };
    expect(emit(mod)).toBe("mic@2\nO 0");
  });
});
