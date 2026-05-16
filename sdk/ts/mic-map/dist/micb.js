// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.
import { MicbError } from "./errors.js";
import { dtypeToByte, dtypeFromByte, opcodeToByte, } from "./types.js";
import { uleb128Write, uleb128Read, sleb128Write, sleb128Read, } from "./varint.js";
/** MIC-B v2 magic bytes: ASCII "MICB". Mirrors Rust MICB_MAGIC. */
export const MICB_MAGIC = new Uint8Array([0x4d, 0x49, 0x43, 0x42]);
/** MIC-B v2 version byte. Mirrors Rust MICB_VERSION. */
export const MICB_VERSION = 0x02;
/**
 * Encode a Mic2Module to MIC-B binary format.
 * Output is deterministic: same module → same bytes.
 * Faithful port of Rust emit_micb().
 */
export function encodeBinary(module) {
    const out = [];
    const encoder = new MicbEncoder(module);
    encoder.encode(out);
    return new Uint8Array(out);
}
/**
 * Decode a MIC-B binary buffer to Mic2Module.
 * Faithful port of Rust parse_micb().
 */
export function decodeBinary(bytes) {
    return new MicbDecoder(bytes).decode();
}
// ─── Encoder ────────────────────────────────────────────────────────────────
class MicbEncoder {
    module;
    strings = [];
    stringMap = new Map();
    constructor(module) {
        this.module = module;
    }
    encode(out) {
        this.buildStringTable();
        // Magic + version
        for (const b of MICB_MAGIC)
            out.push(b);
        out.push(MICB_VERSION);
        this.encodeStringTable(out);
        this.encodeSymbolTable(out);
        this.encodeTypeTable(out);
        this.encodeValueTable(out);
        // Output
        uleb128Write(out, BigInt(this.module.output));
    }
    intern(s) {
        const existing = this.stringMap.get(s);
        if (existing !== undefined)
            return existing;
        const idx = this.strings.length;
        this.strings.push(s);
        this.stringMap.set(s, idx);
        return idx;
    }
    buildStringTable() {
        // Deterministic order: symbols, type dims, value names / custom opcodes
        for (const sym of this.module.symbols)
            this.intern(sym);
        for (const t of this.module.types) {
            for (const dim of t.shape)
                this.intern(dim);
        }
        for (const v of this.module.values) {
            if (v.kind === "arg" || v.kind === "param") {
                this.intern(v.name);
            }
            else if (v.kind === "node" && v.op.kind === "custom") {
                this.intern(v.op.name);
            }
        }
    }
    encodeStringTable(out) {
        uleb128Write(out, BigInt(this.strings.length));
        for (const s of this.strings) {
            const bytes = new TextEncoder().encode(s);
            uleb128Write(out, BigInt(bytes.length));
            for (const b of bytes)
                out.push(b);
        }
    }
    encodeSymbolTable(out) {
        uleb128Write(out, BigInt(this.module.symbols.length));
        for (const sym of this.module.symbols) {
            uleb128Write(out, BigInt(this.stringMap.get(sym)));
        }
    }
    encodeTypeTable(out) {
        uleb128Write(out, BigInt(this.module.types.length));
        for (const t of this.module.types) {
            out.push(dtypeToByte(t.dtype));
            uleb128Write(out, BigInt(t.shape.length));
            for (const dim of t.shape) {
                uleb128Write(out, BigInt(this.stringMap.get(dim)));
            }
        }
    }
    encodeValueTable(out) {
        uleb128Write(out, BigInt(this.module.values.length));
        for (const v of this.module.values) {
            this.encodeValue(out, v);
        }
    }
    encodeValue(out, v) {
        if (v.kind === "arg" || v.kind === "param") {
            out.push(v.kind === "arg" ? 0 : 1);
            uleb128Write(out, BigInt(this.stringMap.get(v.name)));
            uleb128Write(out, BigInt(v.typeIdx));
        }
        else {
            out.push(2); // node tag
            this.encodeOpcode(out, v.op);
            uleb128Write(out, BigInt(v.inputs.length));
            for (const inp of v.inputs) {
                uleb128Write(out, BigInt(inp));
            }
        }
    }
    encodeOpcode(out, op) {
        out.push(opcodeToByte(op));
        switch (op.kind) {
            case "softmax":
                sleb128Write(out, op.axis);
                break;
            case "transpose":
                uleb128Write(out, BigInt(op.perm.length));
                for (const p of op.perm)
                    sleb128Write(out, p);
                break;
            case "sum":
            case "mean":
            case "max":
                uleb128Write(out, BigInt(op.axes.length));
                for (const a of op.axes)
                    sleb128Write(out, a);
                break;
            case "concat":
                sleb128Write(out, op.axis);
                break;
            case "split":
                sleb128Write(out, op.axis);
                uleb128Write(out, BigInt(op.n));
                break;
            case "gather":
                sleb128Write(out, op.axis);
                break;
            case "custom":
                uleb128Write(out, BigInt(this.stringMap.get(op.name)));
                break;
            default: break; // no params
        }
    }
}
// ─── Decoder ────────────────────────────────────────────────────────────────
class MicbDecoder {
    buf;
    strings = [];
    pos = 0;
    constructor(buf) {
        this.buf = buf;
    }
    decode() {
        this.checkMagic();
        this.checkVersion();
        this.strings = this.decodeStringTable();
        const symbols = this.decodeSymbolTable();
        const types = this.decodeTypeTable();
        const values = this.decodeValueTable(types.length);
        const output = this.readUleb();
        if (values.length > 0 && output >= values.length) {
            throw new MicbError(`output ${output} out of bounds (max ${values.length - 1})`);
        }
        return { symbols, types, values, output };
    }
    checkMagic() {
        if (this.buf.length < 4)
            throw new MicbError("truncated: missing magic");
        for (let i = 0; i < 4; i++) {
            if (this.buf[i] !== MICB_MAGIC[i]) {
                throw new MicbError(`invalid magic: expected MICB, got ${String.fromCharCode(...this.buf.slice(0, 4))}`);
            }
        }
        this.pos = 4;
    }
    checkVersion() {
        if (this.pos >= this.buf.length)
            throw new MicbError("truncated: missing version");
        const version = this.buf[this.pos++];
        if (version !== MICB_VERSION) {
            throw new MicbError(`unsupported version: expected ${MICB_VERSION}, got ${version}`);
        }
    }
    decodeStringTable() {
        const n = this.readUleb();
        const result = [];
        const decoder = new TextDecoder();
        for (let i = 0; i < n; i++) {
            const len = this.readUleb();
            if (this.pos + len > this.buf.length)
                throw new MicbError("truncated string table");
            const bytes = this.buf.slice(this.pos, this.pos + len);
            result.push(decoder.decode(bytes));
            this.pos += len;
        }
        return result;
    }
    decodeSymbolTable() {
        const n = this.readUleb();
        const result = [];
        for (let i = 0; i < n; i++) {
            const idx = this.readUleb();
            if (idx >= this.strings.length)
                throw new MicbError(`symbol string index ${idx} out of bounds`);
            result.push(this.strings[idx]);
        }
        return result;
    }
    decodeTypeTable() {
        const n = this.readUleb();
        const result = [];
        for (let i = 0; i < n; i++) {
            if (this.pos >= this.buf.length)
                throw new MicbError("truncated type table");
            const dtypeByte = this.buf[this.pos++];
            const dtype = dtypeFromByte(dtypeByte);
            if (!dtype)
                throw new MicbError(`unknown dtype byte: ${dtypeByte}`);
            const rank = this.readUleb();
            const shape = [];
            for (let j = 0; j < rank; j++) {
                const idx = this.readUleb();
                if (idx >= this.strings.length)
                    throw new MicbError(`type dim string index ${idx} out of bounds`);
                shape.push(this.strings[idx]);
            }
            result.push({ dtype, shape });
        }
        return result;
    }
    decodeValueTable(nTypes) {
        const n = this.readUleb();
        const result = [];
        for (let vid = 0; vid < n; vid++) {
            result.push(this.decodeValue(vid, nTypes));
        }
        return result;
    }
    decodeValue(currentId, nTypes) {
        if (this.pos >= this.buf.length)
            throw new MicbError("truncated value table");
        const tag = this.buf[this.pos++];
        if (tag === 0 || tag === 1) {
            const nameIdx = this.readUleb();
            const typeIdx = this.readUleb();
            if (nameIdx >= this.strings.length)
                throw new MicbError(`value name index ${nameIdx} out of bounds`);
            if (typeIdx >= nTypes)
                throw new MicbError(`value type index ${typeIdx} out of bounds`);
            const name = this.strings[nameIdx];
            return tag === 0
                ? { kind: "arg", name, typeIdx }
                : { kind: "param", name, typeIdx };
        }
        if (tag === 2) {
            const op = this.decodeOpcode();
            const nInputs = this.readUleb();
            const inputs = [];
            for (let i = 0; i < nInputs; i++) {
                const inp = this.readUleb();
                if (inp >= currentId)
                    throw new MicbError(`forward reference: input ${inp} >= current id ${currentId}`);
                inputs.push(inp);
            }
            return { kind: "node", op, inputs };
        }
        throw new MicbError(`unknown value tag: ${tag}`);
    }
    decodeOpcode() {
        if (this.pos >= this.buf.length)
            throw new MicbError("truncated opcode");
        const b = this.buf[this.pos++];
        switch (b) {
            case 0: return { kind: "matmul" };
            case 1: return { kind: "add" };
            case 2: return { kind: "sub" };
            case 3: return { kind: "mul" };
            case 4: return { kind: "div" };
            case 5: return { kind: "relu" };
            case 6: return { kind: "softmax", axis: this.readSleb() };
            case 7: return { kind: "sigmoid" };
            case 8: return { kind: "tanh" };
            case 9: return { kind: "gelu" };
            case 10: return { kind: "layernorm" };
            case 11: {
                const n = this.readUleb();
                const perm = [];
                for (let i = 0; i < n; i++)
                    perm.push(this.readSleb());
                return { kind: "transpose", perm };
            }
            case 12: return { kind: "reshape" };
            case 13: {
                const n = this.readUleb();
                const axes = [];
                for (let i = 0; i < n; i++)
                    axes.push(this.readSleb());
                return { kind: "sum", axes };
            }
            case 14: {
                const n = this.readUleb();
                const axes = [];
                for (let i = 0; i < n; i++)
                    axes.push(this.readSleb());
                return { kind: "mean", axes };
            }
            case 15: {
                const n = this.readUleb();
                const axes = [];
                for (let i = 0; i < n; i++)
                    axes.push(this.readSleb());
                return { kind: "max", axes };
            }
            case 16: return { kind: "concat", axis: this.readSleb() };
            case 17: {
                const axis = this.readSleb();
                const n = this.readUleb();
                return { kind: "split", axis, n };
            }
            case 18: return { kind: "gather", axis: this.readSleb() };
            case 255: {
                const idx = this.readUleb();
                if (idx >= this.strings.length)
                    throw new MicbError(`custom opcode name index ${idx} out of bounds`);
                return { kind: "custom", name: this.strings[idx] };
            }
            default: throw new MicbError(`unknown opcode byte: ${b}`);
        }
    }
    readUleb() {
        const [val, newPos] = uleb128Read(this.buf, this.pos);
        this.pos = newPos;
        return Number(val);
    }
    readSleb() {
        const [val, newPos] = sleb128Read(this.buf, this.pos);
        this.pos = newPos;
        return val;
    }
}
//# sourceMappingURL=micb.js.map