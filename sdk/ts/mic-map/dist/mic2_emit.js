// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.
import { opcodeToken } from "./types.js";
import { MIC2_HEADER } from "./mic2_parse.js";
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
export function emit(module) {
    const parts = [];
    // Header
    parts.push(MIC2_HEADER + "\n");
    // Symbols
    for (const sym of module.symbols) {
        parts.push(`S ${sym}\n`);
    }
    // Types
    for (let i = 0; i < module.types.length; i++) {
        const t = module.types[i];
        const shapePart = t.shape.length > 0 ? " " + t.shape.join(" ") : "";
        parts.push(`T${i} ${t.dtype}${shapePart}\n`);
    }
    // Values
    for (const v of module.values) {
        parts.push(emitValue(v) + "\n");
    }
    // Output — no trailing newline (canonical)
    parts.push(`O ${module.output}`);
    return parts.join("");
}
function emitValue(v) {
    switch (v.kind) {
        case "arg": return `a ${v.name} T${v.typeIdx}`;
        case "param": return `p ${v.name} T${v.typeIdx}`;
        case "node": return emitNode(v.op, v.inputs);
    }
}
function emitNode(op, inputs) {
    const opStr = emitOpcode(op);
    const inputStr = inputs.length > 0 ? " " + inputs.join(" ") : "";
    return opStr + inputStr;
}
/** Emit opcode token + any inline parameters. Mirrors Rust Mic2Emitter::emit_opcode(). */
function emitOpcode(op) {
    const tok = opcodeToken(op);
    switch (op.kind) {
        case "softmax":
            return op.axis !== -1n ? `${tok} ${op.axis}` : tok;
        case "transpose":
            return op.perm.length > 0 ? `${tok} ${op.perm.join(" ")}` : tok;
        case "sum":
        case "mean":
        case "max":
            return op.axes.length > 0 ? `${tok} ${op.axes.join(" ")}` : tok;
        case "concat":
            return `${tok} ${op.axis}`;
        case "split":
            return `${tok} ${op.axis} ${op.n}`;
        case "gather":
            return `${tok} ${op.axis}`;
        default:
            return tok;
    }
}
//# sourceMappingURL=mic2_emit.js.map