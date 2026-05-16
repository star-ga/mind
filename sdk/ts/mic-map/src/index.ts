// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

// Public re-exports for all layers.

export type {
  DType, TensorType, Opcode, GraphValue, Mic2Module,
} from "./types.js";
export type { MapValue, MapFrame } from "./map.js";
export {
  parseDType, dtypeToByte, dtypeFromByte,
  opcodeToken, opcodeToByte, opcodeArity,
  valueTag, modulesEqual, residualBlock,
  ALL_DTYPES,
} from "./types.js";

// Layer A: MIC v2 text codec
export { parse, MIC2_HEADER } from "./mic2_parse.js";
export { emit } from "./mic2_emit.js";
export const MIC2_VERSION = "mic@2";

// Layer B: MIC-B binary codec
export {
  encodeBinary, decodeBinary, MICB_MAGIC, MICB_VERSION,
} from "./micb.js";

// Layer C: MAP protocol frames
export { encode as encodeMap, decode as decodeMap } from "./map.js";

// Layer D: Length-prefixed framing
export { framePayload, readFrames, MAX_FRAME_SIZE } from "./framing.js";

// Varint utilities (exposed for cross-language testing)
export {
  uleb128Write, uleb128Read, sleb128Write, sleb128Read,
  zigzagEncode, zigzagDecode,
} from "./varint.js";

// Errors
export {
  Mic2ParseError, MicbError, MapDecodeError, FrameTooLarge, FrameDecodeError,
} from "./errors.js";

/**
 * Detect the format of input data.
 * Mirrors Rust detect_format().
 */
export function detectFormat(data: Uint8Array | string): "mic2" | "micb" | "unknown" {
  if (typeof data === "string") {
    const trimmed = data.trimStart();
    if (trimmed.startsWith("mic@2")) return "mic2";
    return "unknown";
  }

  // Binary: check for MICB magic
  if (
    data.length >= 5 &&
    data[0] === 0x4d && data[1] === 0x49 && data[2] === 0x43 && data[3] === 0x42
  ) {
    return "micb";
  }

  // Try UTF-8 decode for text detection
  try {
    const text = new TextDecoder().decode(data);
    if (text.trimStart().startsWith("mic@2")) return "mic2";
  } catch {
    // not valid UTF-8
  }

  return "unknown";
}
