import { type Mic2Module } from "./types.js";
/** MIC-B v2 magic bytes: ASCII "MICB". Mirrors Rust MICB_MAGIC. */
export declare const MICB_MAGIC: Uint8Array<ArrayBuffer>;
/** MIC-B v2 version byte. Mirrors Rust MICB_VERSION. */
export declare const MICB_VERSION = 2;
/**
 * Encode a Mic2Module to MIC-B binary format.
 * Output is deterministic: same module → same bytes.
 * Faithful port of Rust emit_micb().
 */
export declare function encodeBinary(module: Mic2Module): Uint8Array;
/**
 * Decode a MIC-B binary buffer to Mic2Module.
 * Faithful port of Rust parse_micb().
 */
export declare function decodeBinary(bytes: Uint8Array): Mic2Module;
//# sourceMappingURL=micb.d.ts.map