# Phase 6.1 — Expected Token Stream

> Smoke gate for `examples/lexer/main.mind` when run against
> `examples/lexer/fixture.mind`. The MIND lexer must emit *exactly*
> this stream. Phase 6.2 promotes this from a documented fixture to
> an integration test that compiles `main.mind` to a `.so` and diffs
> token-by-token against mindc-Rust's own tokeniser.

## Fixture

`examples/lexer/fixture.mind` (94 bytes, byte offsets shown):

```
   0  // Phase 6.1 lexer smoke fixture.
  35  //
  39  // Token-stream contract for this file is documented in
  93  // examples/lexer/EXPECTED.md.  Don't reformat without bumping the
 157  // expected stream.
 176
 177  use std.vec;
 190
 191  pub fn add(x: i64, y: i64) -> i64 {
 226      let z: i64 = x + y;
 247      z
 252  }
 254
```

(Total file length: 254 bytes when counted byte-exact; lo/hi values
below assume byte-offsets into that buffer.)

## Expected stream

Each row is `(kind_tag, lo, hi)`. Comment ranges are *consumed*, not
emitted — the comment scanner advances `i` past the `\n` without
pushing a token. Trailing newline produces no token.

| # | kind | name | lo | hi | source |
|---|------|------|----|----|--------|
| 1 | 20 | `tk_kw_use` | 177 | 180 | `use` |
| 2 | 1  | `tk_ident` | 181 | 184 | `std` |
| 3 | 15 | `tk_slash` | 184 | 185 | `.` — *see note* |
| 4 | 1  | `tk_ident` | 185 | 188 | `vec` |
| 5 | 8  | `tk_semi` | 188 | 189 | `;` |
| 6 | 21 | `tk_kw_pub` | 191 | 194 | `pub` |
| 7 | 18 | `tk_kw_fn` | 195 | 197 | `fn` |
| 8 | 1  | `tk_ident` | 198 | 201 | `add` |
| 9 | 3  | `tk_lparen` | 201 | 202 | `(` |
| 10 | 1 | `tk_ident` | 202 | 203 | `x` |
| 11 | 9 | `tk_colon` | 203 | 204 | `:` |
| 12 | 1 | `tk_ident` | 205 | 208 | `i64` |
| 13 | 7 | `tk_comma` | 208 | 209 | `,` |
| 14 | 1 | `tk_ident` | 210 | 211 | `y` |
| 15 | 9 | `tk_colon` | 211 | 212 | `:` |
| 16 | 1 | `tk_ident` | 213 | 216 | `i64` |
| 17 | 4 | `tk_rparen` | 216 | 217 | `)` |
| 18 | 10 | `tk_arrow` | 218 | 220 | `->` |
| 19 | 1 | `tk_ident` | 221 | 224 | `i64` |
| 20 | 5 | `tk_lbrace` | 225 | 226 | `{` |
| 21 | 19 | `tk_kw_let` | 231 | 234 | `let` |
| 22 | 1 | `tk_ident` | 235 | 236 | `z` |
| 23 | 9 | `tk_colon` | 236 | 237 | `:` |
| 24 | 1 | `tk_ident` | 238 | 241 | `i64` |
| 25 | 11 | `tk_eq` | 242 | 243 | `=` |
| 26 | 1 | `tk_ident` | 244 | 245 | `x` |
| 27 | 12 | `tk_plus` | 246 | 247 | `+` |
| 28 | 1 | `tk_ident` | 248 | 249 | `y` |
| 29 | 8 | `tk_semi` | 249 | 250 | `;` |
| 30 | 1 | `tk_ident` | 255 | 256 | `z` |
| 31 | 6 | `tk_rbrace` | 257 | 258 | `}` |
| 32 | 0 | `tk_eof` | 259 | 259 | — |

**Total flat-Vec length:** 32 tokens × 3 i64 = **96 entries**.

## Known Phase 6.1 limitations (resolved by Phase 6.2+)

Row 3 above shows that `.` (byte `46`) currently falls through to
`tk_slash` because `match_punct` doesn't handle dot. Phase 6.2 will
add `tk_dot()` (and `tk_eq_eq()`, `tk_neq()`, `tk_le()`, `tk_ge()`,
`tk_amp_amp()`, `tk_pipe_pipe()`, `tk_colon_colon()`, `tk_dot_dot()`,
plus `tk_str()` for string literals). The fixture's expected stream
locks the *current* behaviour so the gate stays stable across
intentional Phase 6.2 additions.

## Note on byte offsets

The fixture file uses LF line endings. Recomputing byte offsets on
CRLF would shift every value after the first newline by 1; do not
edit fixture.mind with a CRLF-rewriting editor. The expected stream
assumes the byte-exact LF layout shipped in the repo.

## How to verify (manual, Phase 6.1)

1. `cargo run --features "std-surface cross-module-imports" --bin mindc -- examples/lexer/main.mind --emit-ir`
   Expected: clean exit, IR dump ends with `next_id = 36`.
2. Once mindc gains `--emit-shared` integration tests against an
   in-process Vec<i64> reader, this fixture becomes a Cargo
   integration test asserting the table above byte-by-byte.

## Phase 6.2 follow-up — `while`-statement support

mindc v0.4.4's parser does not accept `while` as a statement (see
`main.mind`'s comment block). The lexer currently expresses loops as
tail recursion. Phase 6.2 will either:

- Add `while`-statement parsing to mindc (small grammar addition,
  module-level gated to preserve the compile-speed moat), or
- Keep recursion as the canonical loop primitive and document it as
  the MIND idiom.

Either choice keeps the existing recursion-based lexer working
unchanged; the decision affects the *parser* sub-stage of Phase 6.2,
not the lexer.
