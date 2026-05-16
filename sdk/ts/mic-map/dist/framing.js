// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.
import { FrameTooLarge, FrameDecodeError } from "./errors.js";
/** Maximum frame payload size: 16 MiB. */
export const MAX_FRAME_SIZE = 16 * 1024 * 1024;
const HEADER_SIZE = 4; // 4-byte BE u32 length prefix
/**
 * Wrap a payload with a 4-byte big-endian u32 length prefix.
 * Throws FrameTooLarge if payload exceeds MAX_FRAME_SIZE.
 */
export function framePayload(payload) {
    const bytes = typeof payload === "string"
        ? new TextEncoder().encode(payload)
        : payload;
    if (bytes.length > MAX_FRAME_SIZE) {
        throw new FrameTooLarge(bytes.length, MAX_FRAME_SIZE);
    }
    const frame = new Uint8Array(HEADER_SIZE + bytes.length);
    const view = new DataView(frame.buffer);
    view.setUint32(0, bytes.length, false); // big-endian
    frame.set(bytes, HEADER_SIZE);
    return frame;
}
/**
 * Async generator that reads length-prefixed frames from a ReadableStream<Uint8Array>.
 * Each yielded Uint8Array is one frame payload (no length prefix).
 *
 * Throws FrameTooLarge if a frame length header exceeds MAX_FRAME_SIZE.
 * Throws FrameDecodeError if the stream ends mid-frame or with a zero-length prefix.
 */
export async function* readFrames(stream) {
    const reader = stream.getReader();
    // Buffer accumulates data across chunks (plain ArrayBuffer-backed)
    let buf = new Uint8Array(0);
    try {
        while (true) {
            // Read until we have at least the header
            while (buf.length < HEADER_SIZE) {
                const { done, value } = await reader.read();
                if (done) {
                    if (buf.length === 0)
                        return; // clean end
                    throw new FrameDecodeError("stream ended mid-header");
                }
                buf = concat(buf, value);
            }
            // Parse length from 4-byte BE u32
            const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
            const frameLen = view.getUint32(0, false);
            if (frameLen === 0)
                throw new FrameDecodeError("zero-length frame");
            if (frameLen > MAX_FRAME_SIZE)
                throw new FrameTooLarge(frameLen, MAX_FRAME_SIZE);
            const totalNeeded = HEADER_SIZE + frameLen;
            // Read until we have the full frame
            while (buf.length < totalNeeded) {
                const { done, value } = await reader.read();
                if (done)
                    throw new FrameDecodeError("stream ended mid-frame");
                buf = concat(buf, value);
            }
            yield buf.slice(HEADER_SIZE, totalNeeded);
            buf = buf.slice(totalNeeded);
        }
    }
    finally {
        reader.releaseLock();
    }
}
function concat(a, b) {
    const out = new Uint8Array(a.length + b.length);
    out.set(a);
    out.set(b, a.length);
    return out;
}
//# sourceMappingURL=framing.js.map