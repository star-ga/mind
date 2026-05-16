// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.
/** Maximum bytes for u64 ULEB128 encoding (ceil(64/7) = 10). */
const MAX_ULEB128_BYTES = 10;
/** Maximum frame size constant for bounds checks. */
export const MAX_FRAME_BYTES = 16 * 1024 * 1024;
/**
 * Write an unsigned integer as ULEB128 into a byte array.
 * Mirrors Rust uleb128_write().
 */
export function uleb128Write(out, value) {
    let v = value < 0n ? 0n : value; // unsigned guard
    do {
        let byte = Number(v & 0x7fn);
        v >>= 7n;
        if (v !== 0n)
            byte |= 0x80;
        out.push(byte);
    } while (v !== 0n);
}
/**
 * Read an unsigned integer from ULEB128 at pos in buf.
 * Returns [value, newPos]. Mirrors Rust uleb128_read().
 */
export function uleb128Read(buf, pos) {
    let result = 0n;
    let shift = 0n;
    let i = pos;
    for (let k = 0; k < MAX_ULEB128_BYTES; k++) {
        if (i >= buf.length) {
            throw new Error("ULEB128: unexpected end of buffer");
        }
        const byte = buf[i++];
        const lo = BigInt(byte & 0x7f);
        if (shift >= 64n || (shift === 63n && lo > 1n)) {
            throw new Error("ULEB128 overflow");
        }
        result |= lo << shift;
        shift += 7n;
        if ((byte & 0x80) === 0)
            return [result, i];
    }
    throw new Error("ULEB128 too long (> 10 bytes)");
}
/**
 * Zigzag-encode a signed integer to unsigned.
 * Mirrors Rust zigzag_encode().
 */
export function zigzagEncode(value) {
    return (value << 1n) ^ (value >> 63n);
}
/**
 * Zigzag-decode an unsigned integer to signed.
 * Mirrors Rust zigzag_decode().
 */
export function zigzagDecode(encoded) {
    return (encoded >> 1n) ^ -(encoded & 1n);
}
/**
 * Write a signed integer as zigzag-encoded ULEB128.
 * Mirrors Rust sleb128_write().
 */
export function sleb128Write(out, value) {
    uleb128Write(out, zigzagEncode(value));
}
/**
 * Read a signed integer from zigzag-encoded ULEB128.
 * Mirrors Rust sleb128_read().
 */
export function sleb128Read(buf, pos) {
    const [encoded, newPos] = uleb128Read(buf, pos);
    return [zigzagDecode(encoded), newPos];
}
/** Calculate encoded size in ULEB128 bytes. */
export function uleb128Size(value) {
    let v = value;
    let size = 0;
    do {
        size++;
        v >>= 7n;
    } while (v !== 0n);
    return size;
}
//# sourceMappingURL=varint.js.map