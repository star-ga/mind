// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { Mic2ParseError } from "./errors.js";
import {
  type DType, type Mic2Module, type Opcode, type TensorType, type GraphValue,
  parseDType, opcodeArity,
} from "./types.js";

export const MIC2_HEADER = "mic@2";

const MAX_INPUT_SIZE = 10 * 1024 * 1024;
const MAX_LINE_COUNT = 1_000_000;
const MAX_VALUE_COUNT = 100_000;
const MAX_SHAPE_DIMS = 32;

/**
 * Parse mic@2 text format into a Mic2Module.
 * Faithful port of Rust parse_mic2().
 */
export function parse(text: string): Mic2Module {
  if (text.length > MAX_INPUT_SIZE) {
    throw new Mic2ParseError(0, `input too large: ${text.length} bytes (max ${MAX_INPUT_SIZE})`);
  }

  const lines = text.split("\n");
  if (lines.length > MAX_LINE_COUNT) {
    throw new Mic2ParseError(0, `too many lines: ${lines.length} (max ${MAX_LINE_COUNT})`);
  }

  return new Mic2Parser(lines).parse();
}

class Mic2Parser {
  private lineNum = 0;
  private symbols: string[] = [];
  private types: TensorType[] = [];
  private values: GraphValue[] = [];
  private output = 0;
  private hasOutput = false;

  constructor(private readonly lines: string[]) {}

  private err(msg: string): Mic2ParseError {
    return new Mic2ParseError(this.lineNum, msg);
  }

  parse(): Mic2Module {
    this.parseHeader();

    while (this.lineNum < this.lines.length) {
      const line = (this.lines[this.lineNum] ?? "").trim();
      this.lineNum++;

      if (line === "" || line.startsWith("#")) continue;

      const tokens = line.split(/\s+/).filter(t => t.length > 0);
      if (tokens.length === 0) continue;

      const first = tokens[0]!;
      if (first === "S") this.parseSymbol(tokens);
      else if (first === "O") this.parseOutput(tokens);
      else if (first === "a") this.parseArg(tokens);
      else if (first === "p") this.parseParam(tokens);
      else if (first.startsWith("T") && first.length > 1 && /^\d+$/.test(first.slice(1))) {
        this.parseType(first, tokens.slice(1));
      } else {
        this.parseNode(first, tokens.slice(1));
      }
    }

    if (!this.hasOutput && this.values.length > 0) {
      throw this.err("missing output line");
    }

    if (this.values.length > 0) this.validate();

    return {
      symbols: this.symbols,
      types: this.types,
      values: this.values,
      output: this.output,
    };
  }

  private parseHeader(): void {
    while (this.lineNum < this.lines.length) {
      const line = (this.lines[this.lineNum] ?? "").trim();
      this.lineNum++;

      if (line === "" || line.startsWith("#")) continue;

      if (line === MIC2_HEADER) return;

      if (line.startsWith("mic@")) {
        throw this.err(`unsupported version '${line}', expected '${MIC2_HEADER}'`);
      }

      throw this.err(`expected '${MIC2_HEADER}' header, got '${line}'`);
    }
    throw this.err("empty input or missing header");
  }

  private parseSymbol(tokens: string[]): void {
    if (tokens.length !== 2) throw this.err("invalid symbol: expected 'S <name>'");
    this.symbols.push(tokens[1]!);
  }

  private parseType(first: string, rest: string[]): void {
    const idxStr = first.slice(1);
    const idx = parseInt(idxStr, 10);
    if (isNaN(idx)) throw this.err(`invalid type index: ${idxStr}`);

    if (idx !== this.types.length) {
      throw this.err(`type indices must be sequential: expected T${this.types.length}, got T${idx}`);
    }
    if (rest.length === 0) throw this.err("type requires dtype");

    const dtype = parseDType(rest[0]!) as DType | undefined;
    if (!dtype) throw this.err(`unknown dtype: ${rest[0]!}`);

    const shapeDims = rest.slice(1);
    if (shapeDims.length > MAX_SHAPE_DIMS) {
      throw this.err(`too many shape dimensions: ${shapeDims.length} (max ${MAX_SHAPE_DIMS})`);
    }

    this.types.push({ dtype, shape: shapeDims });
  }

  private parseArg(tokens: string[]): void {
    if (tokens.length !== 3) throw this.err("invalid arg: expected 'a <name> T<typeidx>'");
    const typeIdx = this.parseTypeRef(tokens[2]!);
    this.addValue({ kind: "arg", name: tokens[1]!, typeIdx });
  }

  private parseParam(tokens: string[]): void {
    if (tokens.length !== 3) throw this.err("invalid param: expected 'p <name> T<typeidx>'");
    const typeIdx = this.parseTypeRef(tokens[2]!);
    this.addValue({ kind: "param", name: tokens[1]!, typeIdx });
  }

  private parseNode(opcodeTok: string, rest: string[]): void {
    const inputs: number[] = [];
    const params: string[] = [];

    // Mirrors Rust: only non-negative usize parses go to inputs; negative
    // numbers and non-numeric tokens go to params (used as opcode parameters).
    for (const tok of rest) {
      const n = parseInt(tok, 10);
      if (!isNaN(n) && n >= 0 && String(n) === tok) {
        inputs.push(n);
      } else {
        params.push(tok);
      }
    }

    for (const inp of inputs) {
      if (inp >= this.values.length) {
        throw this.err(
          `forward reference: input ${inp} not yet defined (current max: ${this.values.length - 1})`,
        );
      }
    }

    const op = parseOpcode(opcodeTok, params);
    if (!op) throw this.err(`unknown opcode: ${opcodeTok}`);

    const arity = opcodeArity(op);
    if (arity !== undefined && inputs.length !== arity) {
      throw this.err(`opcode '${opcodeTok}' requires ${arity} inputs, got ${inputs.length}`);
    }

    this.addValue({ kind: "node", op, inputs });
  }

  private parseOutput(tokens: string[]): void {
    if (tokens.length !== 2) throw this.err("invalid output: expected 'O <value_id>'");
    const id = parseInt(tokens[1]!, 10);
    if (isNaN(id)) throw this.err(`invalid output id: ${tokens[1]!}`);
    if (id >= this.values.length) {
      throw this.err(`output references undefined value ${id} (max: ${this.values.length - 1})`);
    }
    this.output = id;
    this.hasOutput = true;
  }

  private parseTypeRef(tok: string): number {
    if (!tok.startsWith("T")) throw this.err(`expected type ref T<idx>, got '${tok}'`);
    const idx = parseInt(tok.slice(1), 10);
    if (isNaN(idx)) throw this.err(`invalid type ref: ${tok}`);
    if (idx >= this.types.length) {
      throw this.err(`type T${idx} not defined (max: T${this.types.length - 1})`);
    }
    return idx;
  }

  private addValue(v: GraphValue): void {
    if (this.values.length >= MAX_VALUE_COUNT) {
      throw this.err(`too many values: ${this.values.length} (max ${MAX_VALUE_COUNT})`);
    }
    this.values.push(v);
  }

  private validate(): void {
    for (let vid = 0; vid < this.values.length; vid++) {
      const v = this.values[vid]!;
      if (v.kind === "arg" || v.kind === "param") {
        if (v.typeIdx >= this.types.length) {
          throw this.err(`Value ${vid} references invalid type ${v.typeIdx}`);
        }
      } else if (v.kind === "node") {
        for (const inp of v.inputs) {
          if (inp >= vid) {
            throw this.err(`Value ${vid} has forward reference to ${inp}`);
          }
        }
      }
    }
    if (this.output >= this.values.length) {
      throw this.err(`Output ${this.output} references invalid value`);
    }
  }
}

/** Parse opcode from mic@2 text token. Mirrors Rust Opcode::parse(). */
function parseOpcode(tok: string, params: string[]): Opcode | undefined {
  const firstParam = params[0];
  const parseBigInt = (s: string | undefined, def: bigint): bigint => {
    if (s === undefined) return def;
    try { return BigInt(s); } catch { return def; }
  };
  const parseBigIntArr = (ps: string[]): bigint[] =>
    ps.flatMap(s => { try { return [BigInt(s)]; } catch { return []; } });

  switch (tok) {
    case "m": return { kind: "matmul" };
    case "+": return { kind: "add" };
    case "-": return { kind: "sub" };
    case "*": return { kind: "mul" };
    case "/": return { kind: "div" };
    case "r": return { kind: "relu" };
    case "s": return { kind: "softmax", axis: parseBigInt(firstParam, -1n) };
    case "sig": return { kind: "sigmoid" };
    case "th": return { kind: "tanh" };
    case "gelu": return { kind: "gelu" };
    case "ln": return { kind: "layernorm" };
    case "t": return { kind: "transpose", perm: parseBigIntArr(params) };
    case "rshp": return { kind: "reshape" };
    case "sum": return { kind: "sum", axes: parseBigIntArr(params) };
    case "mean": return { kind: "mean", axes: parseBigIntArr(params) };
    case "max": return { kind: "max", axes: parseBigIntArr(params) };
    case "cat": return { kind: "concat", axis: parseBigInt(firstParam, 0n) };
    case "split": {
      const axis = parseBigInt(firstParam, 0n);
      const n = params[1] !== undefined ? parseInt(params[1], 10) : 2;
      return { kind: "split", axis, n: isNaN(n) ? 2 : n };
    }
    case "gth": return { kind: "gather", axis: parseBigInt(firstParam, 0n) };
    default:
      return undefined; // unknown — mirrors Rust Opcode::parse returning None
  }
}
