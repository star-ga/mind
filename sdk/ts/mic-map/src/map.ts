// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").
// MIC/MAP TypeScript SDK — STARGA Inc.

import { MapDecodeError } from "./errors.js";

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
export type MapFrame =
  | { readonly kind: "req"; readonly op: string; readonly fields: Record<string, MapValue> }
  | { readonly kind: "ok"; readonly fields: Record<string, MapValue> }
  | { readonly kind: "err"; readonly code: string; readonly fields: Record<string, MapValue> }
  | { readonly kind: "event"; readonly event: string; readonly fields: Record<string, MapValue> };

// ─── Encoder ────────────────────────────────────────────────────────────────

/**
 * Encode a MapFrame to a single wire line (no trailing newline).
 * Grammar:
 *   req   → @<op> <k>=<v>...
 *   ok    → =ok <k>=<v>...
 *   err   → =err <code> <k>=<v>...
 *   event → !<event> <k>=<v>...
 */
export function encode(frame: MapFrame): string {
  switch (frame.kind) {
    case "req":
      return buildLine(`@${frame.op}`, frame.fields);
    case "ok":
      return buildLine("=ok", frame.fields);
    case "err":
      return buildLine(`=err ${frame.code}`, frame.fields);
    case "event":
      return buildLine(`!${frame.event}`, frame.fields);
  }
}

function buildLine(prefix: string, fields: Record<string, MapValue>): string {
  const entries = Object.entries(fields);
  if (entries.length === 0) return prefix;
  const fieldStr = entries.map(([k, v]) => `${k}=${encodeValue(v)}`).join(" ");
  return `${prefix} ${fieldStr}`;
}

function encodeValue(v: MapValue): string {
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "string") return encodeString(v);
  if (v instanceof Uint8Array) return "0x" + bufToHex(v);
  if (Array.isArray(v)) return "[" + v.map(encodeValue).join(",") + "]";
  // Unreachable but satisfies exhaustiveness
  return String(v);
}

/** Encode a string: bareword if safe, otherwise quoted with \\ and \" escapes. */
function encodeString(s: string): string {
  if (isBareword(s)) return s;
  return '"' + s.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"';
}

function isBareword(s: string): boolean {
  if (s.length === 0) return false;
  // Bareword: no whitespace, no =, no ", no \, no [ or ], no , and not a
  // number/bool/hex literal (which would be ambiguous on decode).
  if (/[\s="\\[\],]/.test(s)) return false;
  // Avoid ambiguity with bool/int/hex
  if (s === "true" || s === "false") return false;
  if (/^-?\d/.test(s)) return false;
  if (/^0x/i.test(s)) return false;
  return true;
}

function bufToHex(buf: Uint8Array): string {
  return Array.from(buf).map(b => b.toString(16).padStart(2, "0")).join("");
}

// ─── Decoder ────────────────────────────────────────────────────────────────

/**
 * Decode a single MAP wire line into a MapFrame.
 * Throws MapDecodeError on malformed input.
 */
export function decode(line: string): MapFrame {
  const trimmed = line.trimEnd();
  if (trimmed.length === 0) throw new MapDecodeError("empty line");

  const prefix = trimmed[0]!;

  if (prefix === "@") {
    const { head, rest } = splitFirstWord(trimmed.slice(1));
    if (!head) throw new MapDecodeError("missing op after @");
    return { kind: "req", op: head, fields: parseFields(rest) };
  }

  if (prefix === "=") {
    const body = trimmed.slice(1);
    const { head: kw, rest } = splitFirstWord(body);
    if (kw === "ok") {
      return { kind: "ok", fields: parseFields(rest) };
    }
    if (kw === "err") {
      const { head: code, rest: fieldStr } = splitFirstWord(rest);
      if (!code) throw new MapDecodeError("missing error code after =err");
      return { kind: "err", code, fields: parseFields(fieldStr) };
    }
    throw new MapDecodeError(`unknown = frame keyword: '${kw ?? ""}'`);
  }

  if (prefix === "!") {
    const { head, rest } = splitFirstWord(trimmed.slice(1));
    if (!head) throw new MapDecodeError("missing event name after !");
    return { kind: "event", event: head, fields: parseFields(rest) };
  }

  throw new MapDecodeError(`unknown frame prefix: '${prefix}'`);
}

function splitFirstWord(s: string): { head: string | undefined; rest: string } {
  const trimmed = s.trimStart();
  const spaceIdx = trimmed.search(/\s/);
  if (spaceIdx === -1) return { head: trimmed || undefined, rest: "" };
  return { head: trimmed.slice(0, spaceIdx), rest: trimmed.slice(spaceIdx + 1).trimStart() };
}

function parseFields(s: string): Record<string, MapValue> {
  const result: Record<string, MapValue> = {};
  let rest = s.trimStart();

  while (rest.length > 0) {
    const eqIdx = rest.indexOf("=");
    if (eqIdx <= 0) throw new MapDecodeError(`expected key=value, got: '${rest}'`);

    const key = rest.slice(0, eqIdx);
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      throw new MapDecodeError(`invalid field key: '${key}'`);
    }

    rest = rest.slice(eqIdx + 1);
    const [value, consumed] = parseValue(rest);
    result[key] = value;
    rest = rest.slice(consumed).trimStart();
  }

  return result;
}

/**
 * Parse a single MapValue from the front of s. Returns [value, charsConsumed].
 * When inList=true, stops at ',' and ']' in addition to whitespace.
 */
function parseValue(s: string, inList = false): [MapValue, number] {
  if (s.length === 0) throw new MapDecodeError("expected value, got empty string");

  // List: [...]
  if (s[0] === "[") return parseList(s);

  // Quoted string: "..."
  if (s[0] === '"') return parseQuotedString(s);

  // Hex bytes: 0x...
  if (s.startsWith("0x") || s.startsWith("0X")) return parseHex(s, inList);

  // Boolean or number or bareword — read until whitespace (and list delimiters when in list)
  const stopRe = inList ? /[\s,\]]/ : /[\s]/;
  const end = s.search(stopRe);
  const tok = end === -1 ? s : s.slice(0, end);
  const consumed = tok.length;

  if (tok === "true") return [true, consumed];
  if (tok === "false") return [false, consumed];

  // Integer (incl. negative)
  if (/^-?\d+$/.test(tok)) {
    return [BigInt(tok), consumed];
  }

  // Bareword string
  return [tok, consumed];
}

function parseList(s: string): [MapValue[], number] {
  // s starts with '['
  let i = 1;
  const items: MapValue[] = [];

  while (i < s.length) {
    // Skip whitespace
    while (i < s.length && /\s/.test(s[i]!)) i++;
    if (s[i] === "]") return [items, i + 1];
    if (items.length > 0) {
      if (s[i] !== ",") throw new MapDecodeError(`expected ',' in list, got '${s[i] ?? "EOF"}'`);
      i++;
      while (i < s.length && /\s/.test(s[i]!)) i++;
    }
    const [val, consumed] = parseValue(s.slice(i), true);
    items.push(val);
    i += consumed;
  }
  throw new MapDecodeError("unterminated list");
}

function parseQuotedString(s: string): [string, number] {
  // s starts with '"'
  let i = 1;
  let result = "";

  while (i < s.length) {
    const ch = s[i]!;
    if (ch === '"') return [result, i + 1];
    if (ch === "\\") {
      i++;
      if (i >= s.length) throw new MapDecodeError("unterminated escape sequence");
      const esc = s[i]!;
      if (esc === '"') result += '"';
      else if (esc === "\\") result += "\\";
      else throw new MapDecodeError(`unknown escape: \\${esc}`);
    } else {
      result += ch;
    }
    i++;
  }
  throw new MapDecodeError("unterminated string literal");
}

function parseHex(s: string, inList = false): [Uint8Array, number] {
  const stopRe = inList ? /[\s,\]]/ : /[\s]/;
  const end = s.search(stopRe);
  const tok = end === -1 ? s : s.slice(0, end);
  const hex = tok.slice(2); // strip 0x
  if (hex.length % 2 !== 0) throw new MapDecodeError(`odd-length hex: '${tok}'`);
  if (hex.length > 0 && !/^[0-9a-fA-F]+$/.test(hex)) {
    throw new MapDecodeError(`invalid hex: '${tok}'`);
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return [bytes, tok.length];
}
