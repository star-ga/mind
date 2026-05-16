/** Maximum frame payload size: 16 MiB. */
export declare const MAX_FRAME_SIZE: number;
/**
 * Wrap a payload with a 4-byte big-endian u32 length prefix.
 * Throws FrameTooLarge if payload exceeds MAX_FRAME_SIZE.
 */
export declare function framePayload(payload: string | Uint8Array): Uint8Array;
/**
 * Async generator that reads length-prefixed frames from a ReadableStream<Uint8Array>.
 * Each yielded Uint8Array is one frame payload (no length prefix).
 *
 * Throws FrameTooLarge if a frame length header exceeds MAX_FRAME_SIZE.
 * Throws FrameDecodeError if the stream ends mid-frame or with a zero-length prefix.
 */
export declare function readFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<Uint8Array>;
//# sourceMappingURL=framing.d.ts.map