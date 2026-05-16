import { type Mic2Module } from "./types.js";
/**
 * Emit a Mic2Module as canonical mic@2 text.
 * Faithful port of Rust emit_mic2().
 *
 * Canonicalization rules (mirrors Rust):
 * - Lines use exactly one space between tokens
 * - No trailing spaces
 * - No trailing newline after the final output line
 * - Sections in order: header, symbols, types, values, output
 */
export declare function emit(module: Mic2Module): string;
//# sourceMappingURL=mic2_emit.d.ts.map