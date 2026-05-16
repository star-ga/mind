import { type Mic2Module } from "./types.js";
export declare const MIC2_HEADER = "mic@2";
/**
 * Parse mic@2 text format into a Mic2Module.
 * Faithful port of Rust parse_mic2().
 */
export declare function parse(text: string): Mic2Module;
//# sourceMappingURL=mic2_parse.d.ts.map