// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { describe, it, expect } from "vitest";
import { encodeMap, decodeMap, MapDecodeError, type MapFrame, type MapValue } from "../src/index.js";

// ─── Roundtrip helper ───────────────────────────────────────────────────────

function rt(frame: MapFrame): MapFrame {
  return decodeMap(encodeMap(frame));
}

function fieldsRoundtrip(frame: MapFrame): boolean {
  const f2 = rt(frame);
  return encodeMap(frame) === encodeMap(f2);
}

// ─── Basic frame kinds ──────────────────────────────────────────────────────

describe("MAP — request frames", () => {
  it("roundtrips a request with no fields", () => {
    const f: MapFrame = { kind: "req", op: "compile", fields: {} };
    expect(rt(f)).toMatchObject({ kind: "req", op: "compile", fields: {} });
  });

  it("roundtrips a request with mixed scalar fields", () => {
    const f: MapFrame = {
      kind: "req", op: "load",
      fields: { path: "/tmp/model.mic2", flags: 3n, verbose: true },
    };
    expect(fieldsRoundtrip(f)).toBe(true);
    const decoded = rt(f);
    expect(decoded.kind).toBe("req");
    if (decoded.kind === "req") {
      expect(decoded.fields["flags"]).toBe(3n);
      expect(decoded.fields["verbose"]).toBe(true);
    }
  });

  it("encodes as '@<op> ...'", () => {
    const line = encodeMap({ kind: "req", op: "compile", fields: { n: 1n } });
    expect(line.startsWith("@compile ")).toBe(true);
  });
});

describe("MAP — ok frames", () => {
  it("roundtrips ok with fields", () => {
    const f: MapFrame = {
      kind: "ok",
      fields: { status: "success", count: 42n },
    };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("encodes as '=ok ...'", () => {
    const line = encodeMap({ kind: "ok", fields: { x: 1n } });
    expect(line.startsWith("=ok ")).toBe(true);
  });
});

describe("MAP — error frames", () => {
  it("roundtrips err with code and fields", () => {
    const f: MapFrame = {
      kind: "err", code: "E001",
      fields: { msg: "type mismatch", line: 7n },
    };
    expect(fieldsRoundtrip(f)).toBe(true);
    const decoded = rt(f);
    if (decoded.kind === "err") {
      expect(decoded.code).toBe("E001");
    }
  });

  it("encodes as '=err <code> ...'", () => {
    const line = encodeMap({ kind: "err", code: "E404", fields: {} });
    expect(line.startsWith("=err E404")).toBe(true);
  });
});

describe("MAP — event frames", () => {
  it("roundtrips event with fields", () => {
    const f: MapFrame = {
      kind: "event", event: "progress",
      fields: { pct: 50n, stage: "opt" },
    };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("encodes as '!<event> ...'", () => {
    const line = encodeMap({ kind: "event", event: "done", fields: {} });
    expect(line.startsWith("!done")).toBe(true);
  });
});

// ─── String escaping ────────────────────────────────────────────────────────

describe("MAP — string escape handling", () => {
  it("roundtrips string with backslash", () => {
    const f: MapFrame = { kind: "ok", fields: { path: "C:\\Users\\foo" } };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("roundtrips string with double quote", () => {
    const f: MapFrame = { kind: "ok", fields: { msg: 'say "hello"' } };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("roundtrips string with both backslash and quote", () => {
    const f: MapFrame = { kind: "ok", fields: { v: 'a\\b"c' } };
    expect(fieldsRoundtrip(f)).toBe(true);
    const decoded = rt(f);
    if (decoded.kind === "ok") expect(decoded.fields["v"]).toBe('a\\b"c');
  });

  it("bareword strings are not quoted", () => {
    const line = encodeMap({ kind: "ok", fields: { status: "ok" } });
    expect(line).toContain("status=ok");
    expect(line).not.toContain('"ok"');
  });

  it("strings with spaces are quoted", () => {
    const line = encodeMap({ kind: "ok", fields: { msg: "hello world" } });
    expect(line).toContain('"hello world"');
  });
});

// ─── Field edge cases ───────────────────────────────────────────────────────

describe("MAP — field count extremes", () => {
  it("roundtrips frame with zero fields", () => {
    const f: MapFrame = { kind: "ok", fields: {} };
    const decoded = rt(f);
    expect(decoded.fields).toEqual({});
  });

  it("roundtrips frame with single field", () => {
    const f: MapFrame = { kind: "req", op: "ping", fields: { x: 1n } };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("roundtrips frame with 50 integer fields", () => {
    const fields: Record<string, MapValue> = {};
    for (let i = 0; i < 50; i++) fields[`f${i}`] = BigInt(i);
    const f: MapFrame = { kind: "ok", fields };
    expect(fieldsRoundtrip(f)).toBe(true);
  });
});

// ─── Bytes field ────────────────────────────────────────────────────────────

describe("MAP — bytes field (hex)", () => {
  it("roundtrips Uint8Array field", () => {
    const bytes = new Uint8Array([0x1a, 0x2b, 0x3c, 0xff]);
    const f: MapFrame = { kind: "ok", fields: { data: bytes } };
    const decoded = rt(f);
    if (decoded.kind === "ok") {
      const v = decoded.fields["data"];
      expect(v instanceof Uint8Array).toBe(true);
      if (v instanceof Uint8Array) expect(v).toEqual(bytes);
    }
  });

  it("encodes bytes as 0x hex", () => {
    const bytes = new Uint8Array([0xde, 0xad, 0xbe, 0xef]);
    const line = encodeMap({ kind: "ok", fields: { hash: bytes } });
    expect(line).toContain("0xdeadbeef");
  });

  it("roundtrips empty byte array", () => {
    const f: MapFrame = { kind: "ok", fields: { data: new Uint8Array(0) } };
    const decoded = rt(f);
    if (decoded.kind === "ok") {
      const v = decoded.fields["data"];
      expect(v instanceof Uint8Array).toBe(true);
      if (v instanceof Uint8Array) expect(v.length).toBe(0);
    }
  });
});

// ─── Type discrimination ─────────────────────────────────────────────────────

describe("MAP — type discrimination after decode", () => {
  it("boolean true decodes as boolean", () => {
    const f: MapFrame = { kind: "ok", fields: { v: true } };
    const decoded = rt(f);
    if (decoded.kind === "ok") expect(typeof decoded.fields["v"]).toBe("boolean");
  });

  it("boolean false decodes as boolean", () => {
    const f: MapFrame = { kind: "ok", fields: { v: false } };
    const decoded = rt(f);
    if (decoded.kind === "ok") expect(decoded.fields["v"]).toBe(false);
  });

  it("negative integer decodes as bigint", () => {
    const f: MapFrame = { kind: "ok", fields: { n: -42n } };
    const decoded = rt(f);
    if (decoded.kind === "ok") {
      expect(typeof decoded.fields["n"]).toBe("bigint");
      expect(decoded.fields["n"]).toBe(-42n);
    }
  });

  it("string decodes as string", () => {
    const f: MapFrame = { kind: "ok", fields: { s: "hello" } };
    const decoded = rt(f);
    if (decoded.kind === "ok") expect(typeof decoded.fields["s"]).toBe("string");
  });
});

// ─── List field ─────────────────────────────────────────────────────────────

describe("MAP — list fields", () => {
  it("roundtrips integer list", () => {
    const f: MapFrame = { kind: "ok", fields: { ids: [1n, 2n, 3n] } };
    expect(fieldsRoundtrip(f)).toBe(true);
    const decoded = rt(f);
    if (decoded.kind === "ok") {
      expect(decoded.fields["ids"]).toEqual([1n, 2n, 3n]);
    }
  });

  it("roundtrips string list", () => {
    const f: MapFrame = { kind: "ok", fields: { names: ["a", "b"] } };
    expect(fieldsRoundtrip(f)).toBe(true);
  });

  it("roundtrips empty list", () => {
    const f: MapFrame = { kind: "ok", fields: { items: [] } };
    const decoded = rt(f);
    if (decoded.kind === "ok") {
      expect(decoded.fields["items"]).toEqual([]);
    }
  });
});

// ─── Error cases ─────────────────────────────────────────────────────────────

describe("MAP — malformed input errors", () => {
  it("rejects empty string", () => {
    expect(() => decodeMap("")).toThrow(MapDecodeError);
  });

  it("rejects unknown prefix", () => {
    expect(() => decodeMap("?foo bar=1")).toThrow(MapDecodeError);
  });

  it("rejects = without ok or err", () => {
    expect(() => decodeMap("=unknown")).toThrow(MapDecodeError);
  });

  it("rejects err without code", () => {
    expect(() => decodeMap("=err")).toThrow(MapDecodeError);
  });
});
