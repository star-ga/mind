// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

/** Typed error for MIC v2 parse failures. */
export class Mic2ParseError extends Error {
  readonly line: number;

  constructor(line: number, message: string) {
    super(`mic@2:${line}: error: ${message}`);
    this.name = "Mic2ParseError";
    this.line = line;
  }
}

/** Typed error for MIC-B binary codec failures. */
export class MicbError extends Error {
  constructor(message: string) {
    super(`MICB error: ${message}`);
    this.name = "MicbError";
  }
}

/** Typed error for MAP frame codec failures. */
export class MapDecodeError extends Error {
  constructor(message: string) {
    super(`MAP decode error: ${message}`);
    this.name = "MapDecodeError";
  }
}

/** Typed error for length-prefixed framing failures. */
export class FrameTooLarge extends Error {
  readonly size: number;
  readonly maxSize: number;

  constructor(size: number, maxSize: number) {
    super(`Frame too large: ${size} bytes exceeds limit of ${maxSize} bytes`);
    this.name = "FrameTooLarge";
    this.size = size;
    this.maxSize = maxSize;
  }
}

/** Typed error for invalid frame format. */
export class FrameDecodeError extends Error {
  constructor(message: string) {
    super(`Frame decode error: ${message}`);
    this.name = "FrameDecodeError";
  }
}
