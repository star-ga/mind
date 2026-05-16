export type { DType, TensorType, Opcode, GraphValue, Mic2Module, } from "./types.js";
export type { MapValue, MapFrame } from "./map.js";
export { parseDType, dtypeToByte, dtypeFromByte, opcodeToken, opcodeToByte, opcodeArity, valueTag, modulesEqual, residualBlock, ALL_DTYPES, } from "./types.js";
export { parse, MIC2_HEADER } from "./mic2_parse.js";
export { emit } from "./mic2_emit.js";
export declare const MIC2_VERSION = "mic@2";
export { encodeBinary, decodeBinary, MICB_MAGIC, MICB_VERSION, } from "./micb.js";
export { encode as encodeMap, decode as decodeMap } from "./map.js";
export { framePayload, readFrames, MAX_FRAME_SIZE } from "./framing.js";
export { uleb128Write, uleb128Read, sleb128Write, sleb128Read, zigzagEncode, zigzagDecode, } from "./varint.js";
export { Mic2ParseError, MicbError, MapDecodeError, FrameTooLarge, FrameDecodeError, } from "./errors.js";
/**
 * Detect the format of input data.
 * Mirrors Rust detect_format().
 */
export declare function detectFormat(data: Uint8Array | string): "mic2" | "micb" | "unknown";
//# sourceMappingURL=index.d.ts.map