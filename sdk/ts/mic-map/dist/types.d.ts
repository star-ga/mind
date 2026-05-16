/** Data type for tensor elements. Mirrors Rust DType enum. */
export type DType = "f16" | "f32" | "f64" | "bf16" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool";
export declare const ALL_DTYPES: readonly DType[];
/** Parse dtype from string token. Returns undefined for unknown tokens. */
export declare function parseDType(s: string): DType | undefined;
/** Convert DType to binary encoding byte. Mirrors Rust DType::to_byte(). */
export declare function dtypeToByte(d: DType): number;
/** Parse DType from binary encoding byte. Mirrors Rust DType::from_byte(). */
export declare function dtypeFromByte(b: number): DType | undefined;
/** Tensor type with dtype and shape dimensions. */
export interface TensorType {
    readonly dtype: DType;
    /** Shape dims: fixed ("128"), symbolic ("B"), wildcard ("?"). */
    readonly shape: readonly string[];
}
/**
 * Operation code. Discriminated union mirroring Rust Opcode enum.
 * Params are embedded in the variant's data fields.
 */
export type Opcode = {
    readonly kind: "matmul";
} | {
    readonly kind: "add";
} | {
    readonly kind: "sub";
} | {
    readonly kind: "mul";
} | {
    readonly kind: "div";
} | {
    readonly kind: "relu";
} | {
    readonly kind: "softmax";
    readonly axis: bigint;
} | {
    readonly kind: "sigmoid";
} | {
    readonly kind: "tanh";
} | {
    readonly kind: "gelu";
} | {
    readonly kind: "layernorm";
} | {
    readonly kind: "transpose";
    readonly perm: readonly bigint[];
} | {
    readonly kind: "reshape";
} | {
    readonly kind: "sum";
    readonly axes: readonly bigint[];
} | {
    readonly kind: "mean";
    readonly axes: readonly bigint[];
} | {
    readonly kind: "max";
    readonly axes: readonly bigint[];
} | {
    readonly kind: "concat";
    readonly axis: bigint;
} | {
    readonly kind: "split";
    readonly axis: bigint;
    readonly n: number;
} | {
    readonly kind: "gather";
    readonly axis: bigint;
} | {
    readonly kind: "custom";
    readonly name: string;
};
/** Map opcode kind to MIC v2 text token. Mirrors Rust Opcode::as_token(). */
export declare function opcodeToken(op: Opcode): string;
/** Convert opcode to binary byte. Mirrors Rust Opcode::to_byte(). */
export declare function opcodeToByte(op: Opcode): number;
/** Optional arity (undefined = variadic). Mirrors Rust Opcode::arity(). */
export declare function opcodeArity(op: Opcode): number | undefined;
/**
 * Value in the computation graph. Discriminated union mirroring Rust Value enum.
 * Values have implicit sequential IDs by their index in Graph.values.
 */
export type GraphValue = {
    readonly kind: "arg";
    readonly name: string;
    readonly typeIdx: number;
} | {
    readonly kind: "param";
    readonly name: string;
    readonly typeIdx: number;
} | {
    readonly kind: "node";
    readonly op: Opcode;
    readonly inputs: readonly number[];
};
/** Binary tag for value kind. Mirrors Rust Value::tag(). */
export declare function valueTag(v: GraphValue): number;
/** Computation graph — the top-level MIC v2 data structure. */
export interface Mic2Module {
    /** Optional symbol declarations (e.g. "B", "seq"). */
    readonly symbols: readonly string[];
    /** Type table — TensorType indexed sequentially. */
    readonly types: readonly TensorType[];
    /** Value table — implicit IDs by index. */
    readonly values: readonly GraphValue[];
    /** ID of the output value. */
    readonly output: number;
}
/** Deep structural equality for Mic2Module. */
export declare function modulesEqual(a: Mic2Module, b: Mic2Module): boolean;
/** Canonical residual block fixture: Y = relu(XW + b) + X. Mirrors Rust Graph::residual_block(). */
export declare function residualBlock(): Mic2Module;
//# sourceMappingURL=types.d.ts.map