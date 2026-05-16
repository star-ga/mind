// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.
export const ALL_DTYPES = [
    "f16", "f32", "f64", "bf16",
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "bool",
];
/** Parse dtype from string token. Returns undefined for unknown tokens. */
export function parseDType(s) {
    return ALL_DTYPES.includes(s)
        ? s
        : undefined;
}
/** Convert DType to binary encoding byte. Mirrors Rust DType::to_byte(). */
export function dtypeToByte(d) {
    const map = {
        f16: 0, f32: 1, f64: 2, bf16: 3,
        i8: 4, i16: 5, i32: 6, i64: 7,
        u8: 8, u16: 9, u32: 10, u64: 11,
        bool: 12,
    };
    return map[d];
}
/** Parse DType from binary encoding byte. Mirrors Rust DType::from_byte(). */
export function dtypeFromByte(b) {
    const map = {
        0: "f16", 1: "f32", 2: "f64", 3: "bf16",
        4: "i8", 5: "i16", 6: "i32", 7: "i64",
        8: "u8", 9: "u16", 10: "u32", 11: "u64",
        12: "bool",
    };
    return map[b];
}
/** Map opcode kind to MIC v2 text token. Mirrors Rust Opcode::as_token(). */
export function opcodeToken(op) {
    const map = {
        matmul: "m", add: "+", sub: "-", mul: "*", div: "/",
        relu: "r", softmax: "s", sigmoid: "sig", tanh: "th", gelu: "gelu",
        layernorm: "ln", transpose: "t", reshape: "rshp",
        sum: "sum", mean: "mean", max: "max",
        concat: "cat", split: "split", gather: "gth",
    };
    if (op.kind === "custom")
        return op.name;
    return map[op.kind] ?? op.kind;
}
/** Convert opcode to binary byte. Mirrors Rust Opcode::to_byte(). */
export function opcodeToByte(op) {
    const map = {
        matmul: 0, add: 1, sub: 2, mul: 3, div: 4,
        relu: 5, softmax: 6, sigmoid: 7, tanh: 8, gelu: 9,
        layernorm: 10, transpose: 11, reshape: 12,
        sum: 13, mean: 14, max: 15,
        concat: 16, split: 17, gather: 18,
        custom: 255,
    };
    return map[op.kind] ?? 255;
}
/** Optional arity (undefined = variadic). Mirrors Rust Opcode::arity(). */
export function opcodeArity(op) {
    switch (op.kind) {
        case "matmul":
        case "add":
        case "sub":
        case "mul":
        case "div": return 2;
        case "relu":
        case "sigmoid":
        case "tanh":
        case "gelu": return 1;
        case "softmax":
        case "layernorm":
        case "transpose":
        case "reshape": return 1;
        case "sum":
        case "mean":
        case "max": return 1;
        case "split": return 1;
        case "gather": return 2;
        case "concat":
        case "custom": return undefined; // variadic
    }
}
/** Binary tag for value kind. Mirrors Rust Value::tag(). */
export function valueTag(v) {
    switch (v.kind) {
        case "arg": return 0;
        case "param": return 1;
        case "node": return 2;
    }
}
/** Deep structural equality for Mic2Module. */
export function modulesEqual(a, b) {
    if (a.symbols.length !== b.symbols.length)
        return false;
    if (a.symbols.some((s, i) => s !== b.symbols[i]))
        return false;
    if (a.types.length !== b.types.length)
        return false;
    for (let i = 0; i < a.types.length; i++) {
        const ta = a.types[i], tb = b.types[i];
        if (ta.dtype !== tb.dtype)
            return false;
        if (ta.shape.length !== tb.shape.length)
            return false;
        if (ta.shape.some((d, j) => d !== tb.shape[j]))
            return false;
    }
    if (a.values.length !== b.values.length)
        return false;
    for (let i = 0; i < a.values.length; i++) {
        if (!valuesEqual(a.values[i], b.values[i]))
            return false;
    }
    return a.output === b.output;
}
function valuesEqual(a, b) {
    if (a.kind !== b.kind)
        return false;
    if (a.kind === "arg" || a.kind === "param") {
        const bv = b;
        return a.name === bv.name && a.typeIdx === bv.typeIdx;
    }
    if (a.kind === "node") {
        const bv = b;
        if (!opcodesEqual(a.op, bv.op))
            return false;
        if (a.inputs.length !== bv.inputs.length)
            return false;
        return a.inputs.every((inp, i) => inp === bv.inputs[i]);
    }
    return false;
}
function opcodesEqual(a, b) {
    if (a.kind !== b.kind)
        return false;
    switch (a.kind) {
        case "softmax": return a.axis === b.axis;
        case "transpose": {
            const bv = b;
            return a.perm.length === bv.perm.length && a.perm.every((p, i) => p === bv.perm[i]);
        }
        case "sum":
        case "mean":
        case "max": {
            const bv = b;
            return a.axes.length === bv.axes.length && a.axes.every((x, i) => x === bv.axes[i]);
        }
        case "concat": return a.axis === b.axis;
        case "split": return a.axis === b.axis && a.n === b.n;
        case "gather": return a.axis === b.axis;
        case "custom": return a.name === b.name;
        default: return true;
    }
}
/** Canonical residual block fixture: Y = relu(XW + b) + X. Mirrors Rust Graph::residual_block(). */
export function residualBlock() {
    return {
        symbols: [],
        types: [
            { dtype: "f16", shape: ["128", "128"] },
            { dtype: "f16", shape: ["128"] },
        ],
        values: [
            { kind: "arg", name: "X", typeIdx: 0 },
            { kind: "param", name: "W", typeIdx: 0 },
            { kind: "param", name: "b", typeIdx: 1 },
            { kind: "node", op: { kind: "matmul" }, inputs: [0, 1] },
            { kind: "node", op: { kind: "add" }, inputs: [3, 2] },
            { kind: "node", op: { kind: "relu" }, inputs: [4] },
            { kind: "node", op: { kind: "add" }, inputs: [5, 0] },
        ],
        output: 6,
    };
}
//# sourceMappingURL=types.js.map