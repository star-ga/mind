/**
 * MAP field value type. Discriminated by runtime shape:
 * - bigint → integer
 * - Uint8Array → bytes (hex in wire)
 * - boolean → true/false
 * - string → quoted or bareword
 * - array → list
 */
export type MapValue = string | bigint | Uint8Array | boolean | MapValue[];
/**
 * MAP protocol frame. Four kinds mirror the four prefix chars:
 * - req   → @<op> <field>=<value> ...
 * - ok    → =ok <field>=<value> ...
 * - err   → =err <code> <field>=<value> ...
 * - event → !<event> <field>=<value> ...
 */
export type MapFrame = {
    readonly kind: "req";
    readonly op: string;
    readonly fields: Record<string, MapValue>;
} | {
    readonly kind: "ok";
    readonly fields: Record<string, MapValue>;
} | {
    readonly kind: "err";
    readonly code: string;
    readonly fields: Record<string, MapValue>;
} | {
    readonly kind: "event";
    readonly event: string;
    readonly fields: Record<string, MapValue>;
};
/**
 * Encode a MapFrame to a single wire line (no trailing newline).
 * Grammar:
 *   req   → @<op> <k>=<v>...
 *   ok    → =ok <k>=<v>...
 *   err   → =err <code> <k>=<v>...
 *   event → !<event> <k>=<v>...
 */
export declare function encode(frame: MapFrame): string;
/**
 * Decode a single MAP wire line into a MapFrame.
 * Throws MapDecodeError on malformed input.
 */
export declare function decode(line: string): MapFrame;
//# sourceMappingURL=map.d.ts.map