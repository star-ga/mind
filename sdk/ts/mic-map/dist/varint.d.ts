/** Maximum frame size constant for bounds checks. */
export declare const MAX_FRAME_BYTES: number;
/**
 * Write an unsigned integer as ULEB128 into a byte array.
 * Mirrors Rust uleb128_write().
 */
export declare function uleb128Write(out: number[], value: bigint): void;
/**
 * Read an unsigned integer from ULEB128 at pos in buf.
 * Returns [value, newPos]. Mirrors Rust uleb128_read().
 */
export declare function uleb128Read(buf: Uint8Array, pos: number): [bigint, number];
/**
 * Zigzag-encode a signed integer to unsigned.
 * Mirrors Rust zigzag_encode().
 */
export declare function zigzagEncode(value: bigint): bigint;
/**
 * Zigzag-decode an unsigned integer to signed.
 * Mirrors Rust zigzag_decode().
 */
export declare function zigzagDecode(encoded: bigint): bigint;
/**
 * Write a signed integer as zigzag-encoded ULEB128.
 * Mirrors Rust sleb128_write().
 */
export declare function sleb128Write(out: number[], value: bigint): void;
/**
 * Read a signed integer from zigzag-encoded ULEB128.
 * Mirrors Rust sleb128_read().
 */
export declare function sleb128Read(buf: Uint8Array, pos: number): [bigint, number];
/** Calculate encoded size in ULEB128 bytes. */
export declare function uleb128Size(value: bigint): number;
//# sourceMappingURL=varint.d.ts.map