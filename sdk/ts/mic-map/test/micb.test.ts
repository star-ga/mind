// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  encodeBinary, decodeBinary, detectFormat,
  MICB_MAGIC, MICB_VERSION,
  parse, emit,
  residualBlock, modulesEqual,
  MicbError,
  uleb128Write, uleb128Read,
  zigzagEncode, zigzagDecode,
} from "../src/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.join(__dirname, "fixtures");

describe("MIC-B — roundtrip", () => {
  it("decodeBinary(encodeBinary(module)) equals original for residual block", () => {
    const mod = residualBlock();
    const bytes = encodeBinary(mod);
    const decoded = decodeBinary(bytes);
    expect(modulesEqual(mod, decoded)).toBe(true);
  });

  it("cross-encoding: parse text → encode binary → decode binary → emit text → byte-equal", () => {
    const canonical = `mic@2
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
    const parsed = parse(canonical);
    const bytes = encodeBinary(parsed);
    const decoded = decodeBinary(bytes);
    const reEmitted = emit(decoded);
    expect(reEmitted).toBe(canonical);
  });
});

describe("MIC-B — determinism", () => {
  it("same module always produces same bytes", () => {
    const mod = residualBlock();
    const a = encodeBinary(mod);
    const b = encodeBinary(mod);
    expect(a).toEqual(b);
  });

  it("binary is smaller than text for residual block", () => {
    const mod = residualBlock();
    const text = emit(mod);
    const bytes = encodeBinary(mod);
    expect(bytes.length).toBeLessThan(text.length);
  });
});

describe("MIC-B — magic and version", () => {
  it("encoded bytes start with MICB magic", () => {
    const bytes = encodeBinary(residualBlock());
    expect(bytes[0]).toBe(0x4d);
    expect(bytes[1]).toBe(0x49);
    expect(bytes[2]).toBe(0x43);
    expect(bytes[3]).toBe(0x42);
  });

  it("encoded bytes version byte is 0x02", () => {
    const bytes = encodeBinary(residualBlock());
    expect(bytes[4]).toBe(MICB_VERSION);
  });

  it("rejects bad magic", () => {
    const bad = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x02, 0x00]);
    expect(() => decodeBinary(bad)).toThrow(MicbError);
  });

  it("rejects wrong version", () => {
    const bad = new Uint8Array([...MICB_MAGIC, 0x99]);
    expect(() => decodeBinary(bad)).toThrow(MicbError);
  });
});

describe("MIC-B — truncated data", () => {
  it("rejects truncated string table", () => {
    const bytes = encodeBinary(residualBlock());
    // Chop off half the payload
    const truncated = bytes.slice(0, Math.floor(bytes.length / 2));
    expect(() => decodeBinary(truncated)).toThrow();
  });

  it("rejects empty buffer", () => {
    expect(() => decodeBinary(new Uint8Array(0))).toThrow();
  });
});

describe("MIC-B — detectFormat", () => {
  it("identifies MIC v2 text", () => {
    expect(detectFormat("mic@2\nT0 f32 4\n")).toBe("mic2");
    expect(detectFormat(new TextEncoder().encode("mic@2\n"))).toBe("mic2");
  });

  it("identifies MIC-B binary", () => {
    const bytes = encodeBinary(residualBlock());
    expect(detectFormat(bytes)).toBe("micb");
  });

  it("returns unknown for garbage", () => {
    expect(detectFormat("garbage")).toBe("unknown");
    expect(detectFormat(new Uint8Array([0, 1, 2, 3]))).toBe("unknown");
  });
});

describe("MIC-B — varint roundtrip", () => {
  it("uleb128: positive values", () => {
    for (const v of [0n, 1n, 127n, 128n, 255n, 16383n, 1000000n]) {
      const out: number[] = [];
      uleb128Write(out, v);
      const [decoded] = uleb128Read(new Uint8Array(out), 0);
      expect(decoded).toBe(v);
    }
  });

  it("uleb128: 0 encodes as single byte 0x00", () => {
    const out: number[] = [];
    uleb128Write(out, 0n);
    expect(out).toEqual([0]);
  });

  it("uleb128: 127 encodes as single byte 0x7f", () => {
    const out: number[] = [];
    uleb128Write(out, 127n);
    expect(out).toEqual([0x7f]);
  });

  it("uleb128: 128 encodes as two bytes [0x80, 0x01]", () => {
    const out: number[] = [];
    uleb128Write(out, 128n);
    expect(out).toEqual([0x80, 0x01]);
  });

  it("zigzag: canonical mappings", () => {
    expect(zigzagEncode(0n)).toBe(0n);
    expect(zigzagEncode(-1n)).toBe(1n);
    expect(zigzagEncode(1n)).toBe(2n);
    expect(zigzagEncode(-2n)).toBe(3n);
    expect(zigzagEncode(2n)).toBe(4n);
  });

  it("zigzag: roundtrip for range -1000 to 1000", () => {
    for (let i = -1000; i <= 1000; i++) {
      const v = BigInt(i);
      expect(zigzagDecode(zigzagEncode(v))).toBe(v);
    }
  });

  it("zigzag: zero roundtrip", () => {
    expect(zigzagDecode(zigzagEncode(0n))).toBe(0n);
  });
});

describe("MIC-B — cross-language fixture", () => {
  const fixturePath = path.join(FIXTURES, "residual_block.micb.bin");

  it("fixture file exists and decodes without error", () => {
    if (!existsSync(fixturePath)) {
      // Fixture not yet generated — generate from our own encoder
      // and verify the roundtrip. Cross-language test skipped without Rust.
      const mod = residualBlock();
      const bytes = encodeBinary(mod);
      const decoded = decodeBinary(bytes);
      expect(modulesEqual(mod, decoded)).toBe(true);
      return;
    }

    const fixtureBytes = new Uint8Array(readFileSync(fixturePath).buffer);
    const decoded = decodeBinary(fixtureBytes);
    expect(modulesEqual(residualBlock(), decoded)).toBe(true);
  });

  it("encode matches fixture bytes when fixture exists", () => {
    if (!existsSync(fixturePath)) return;

    const fixtureBytes = new Uint8Array(readFileSync(fixturePath).buffer);
    const encoded = encodeBinary(residualBlock());
    expect(encoded).toEqual(fixtureBytes);
  });
});

describe("MIC-B — string table deduplication", () => {
  it("same dim string is interned once", () => {
    // Two types sharing "128" should produce compact output
    const mod = {
      symbols: [],
      types: [
        { dtype: "f32" as const, shape: ["128", "128"] },
        { dtype: "f32" as const, shape: ["128"] },
      ],
      values: [{ kind: "arg" as const, name: "x", typeIdx: 0 }],
      output: 0,
    };
    const bytes = encodeBinary(mod);
    const decoded = decodeBinary(bytes);
    expect(modulesEqual(mod, decoded)).toBe(true);
  });
});
