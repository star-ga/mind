// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { describe, it, expect } from "vitest";
import {
  framePayload, readFrames, MAX_FRAME_SIZE,
  FrameTooLarge, FrameDecodeError,
} from "../src/index.js";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Build a ReadableStream from a flat Uint8Array. */
function streamOf(...chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(ctrl) {
      for (const c of chunks) ctrl.enqueue(c);
      ctrl.close();
    },
  });
}

/** Collect all frames from a stream into an array. */
async function collectFrames(s: ReadableStream<Uint8Array>): Promise<Uint8Array[]> {
  const out: Uint8Array[] = [];
  for await (const frame of readFrames(s)) out.push(frame);
  return out;
}

// ─── framePayload ────────────────────────────────────────────────────────────

describe("framing — framePayload", () => {
  it("prepends 4-byte big-endian length prefix", () => {
    const payload = new Uint8Array([0x01, 0x02, 0x03]);
    const frame = framePayload(payload);

    const view = new DataView(frame.buffer);
    expect(view.getUint32(0, false)).toBe(3);
    expect(frame[4]).toBe(0x01);
    expect(frame[5]).toBe(0x02);
    expect(frame[6]).toBe(0x03);
  });

  it("total frame length is header(4) + payload", () => {
    const payload = new Uint8Array(100);
    expect(framePayload(payload).length).toBe(104);
  });

  it("encodes string payload as UTF-8", () => {
    const frame = framePayload("hello");
    const view = new DataView(frame.buffer);
    expect(view.getUint32(0, false)).toBe(5);
    expect(frame.slice(4)).toEqual(new TextEncoder().encode("hello"));
  });

  it("handles multi-byte UTF-8 correctly", () => {
    const msg = "éà"; // 2-byte UTF-8 chars
    const frame = framePayload(msg);
    const bytes = new TextEncoder().encode(msg);
    const view = new DataView(frame.buffer);
    expect(view.getUint32(0, false)).toBe(bytes.length);
  });

  it("rejects payload larger than MAX_FRAME_SIZE", () => {
    const tooBig = new Uint8Array(MAX_FRAME_SIZE + 1);
    expect(() => framePayload(tooBig)).toThrow(FrameTooLarge);
  });

  it("accepts payload exactly at MAX_FRAME_SIZE", () => {
    const exact = new Uint8Array(MAX_FRAME_SIZE);
    expect(() => framePayload(exact)).not.toThrow();
  });
});

// ─── readFrames ──────────────────────────────────────────────────────────────

describe("framing — readFrames", () => {
  it("reads a single frame", async () => {
    const payload = new TextEncoder().encode("hello world");
    const framed = framePayload(payload);
    const frames = await collectFrames(streamOf(framed));
    expect(frames.length).toBe(1);
    expect(frames[0]).toEqual(payload);
  });

  it("reads N frames from a stream", async () => {
    const payloads = ["one", "two", "three"].map(s => new TextEncoder().encode(s));
    const framed = new Uint8Array(
      payloads.reduce((sum, p) => sum + 4 + p.length, 0),
    );
    let offset = 0;
    for (const p of payloads) {
      const f = framePayload(p);
      framed.set(f, offset);
      offset += f.length;
    }

    const frames = await collectFrames(streamOf(framed));
    expect(frames.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(frames[i]).toEqual(payloads[i]);
    }
  });

  it("handles frames split across multiple stream chunks", async () => {
    const payload = new TextEncoder().encode("split across chunks");
    const framed = framePayload(payload);
    // Split the frame into three chunks
    const c1 = framed.slice(0, 2);
    const c2 = framed.slice(2, 5);
    const c3 = framed.slice(5);

    const frames = await collectFrames(streamOf(c1, c2, c3));
    expect(frames.length).toBe(1);
    expect(frames[0]).toEqual(payload);
  });

  it("rejects frame with length > MAX_FRAME_SIZE", async () => {
    const buf = new Uint8Array(4);
    const view = new DataView(buf.buffer);
    view.setUint32(0, MAX_FRAME_SIZE + 1, false);
    await expect(collectFrames(streamOf(buf))).rejects.toThrow(FrameTooLarge);
  });

  it("rejects zero-length frame header", async () => {
    const buf = new Uint8Array(4); // all zeros = length 0
    await expect(collectFrames(streamOf(buf))).rejects.toThrow(FrameDecodeError);
  });

  it("clean end with no frames when stream is immediately closed", async () => {
    const empty = new ReadableStream<Uint8Array>({
      start(ctrl) { ctrl.close(); },
    });
    const frames = await collectFrames(empty);
    expect(frames.length).toBe(0);
  });
});
