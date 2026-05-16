/** Typed error for MIC v2 parse failures. */
export declare class Mic2ParseError extends Error {
    readonly line: number;
    constructor(line: number, message: string);
}
/** Typed error for MIC-B binary codec failures. */
export declare class MicbError extends Error {
    constructor(message: string);
}
/** Typed error for MAP frame codec failures. */
export declare class MapDecodeError extends Error {
    constructor(message: string);
}
/** Typed error for length-prefixed framing failures. */
export declare class FrameTooLarge extends Error {
    readonly size: number;
    readonly maxSize: number;
    constructor(size: number, maxSize: number);
}
/** Typed error for invalid frame format. */
export declare class FrameDecodeError extends Error {
    constructor(message: string);
}
//# sourceMappingURL=errors.d.ts.map