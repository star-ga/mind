# RFC 0005 Phase 6.1 — Self-Host Lexer Seed

> First step of the self-host ladder. A pure-MIND tokeniser that
> walks an in-memory source buffer byte-by-byte and emits a flat
> `Vec<i64>` token stream. Smoke contract: byte-identical output
> to a fixed reference table in [EXPECTED.md](./EXPECTED.md).
>
> **Position in RFC 0005 landing table:** Phase 6 row (`open` →
> `in flight` as of this commit).

## Files

| File | Purpose |
|---|---|
| `main.mind` | The lexer itself (~250 LOC pure MIND, tail-recursive) |
| `fixture.mind` | A 254-byte source file used as the smoke gate |
| `EXPECTED.md` | The exact token stream `main.mind` must produce on `fixture.mind` |

## What works

- Identifiers and integer literals
- Single-char punctuation: `( ) { } , ; : = + - * / < >`
- Two-char lookahead: `->`
- Line comments `//`
- Whitespace skipping (space, tab, LF, CR)
- Keywords: `fn`, `let`, `use`, `pub`
- EOF token at end-of-input

## What's deferred to Phase 6.2

- Two-char operators: `== != <= >= && || :: ..`
- String literals
- Float literals
- More keywords: `if`, `else`, `return`, `pub`, `struct`, `enum`, `match`, `mut`
- The `.` and `!` punctuation (currently fall through to `tk_slash` / `tk_ident` respectively — documented in EXPECTED.md as a known limitation)
- An integration test that compiles `main.mind` to a `.so` and diffs against mindc-Rust's own tokeniser

## Build

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/lexer/main.mind --emit-ir
```

Expected output: clean exit, IR dump ending with `next_id = 36`.

The `std-surface` + `cross-module-imports` features are required
because the lexer's `use std.vec;` resolves the bundled pure-MIND
Vec implementation. mindc v0.4.4 ships the bundled stdlib via
`include_str!` (RFC 0005 Phase C, committed 2026-05-18) so there's
no external dependency.

## Loop strategy: tail recursion

mindc v0.4.4's parser does not accept `while` as a statement. (The
`while` references in `src/parser/mod.rs` are in the Rust
implementation's internal scanner, not in the surface grammar.)
Every loop in `main.mind` is therefore expressed as tail recursion.

This is *not* a workaround — it's a deliberate self-discipline
exercise. The recursive shape forces every loop to be in
explicit fixed-state form (no hidden mutation, no implicit
condition). If Phase 6.2 adds `while`, these recursions port back
cleanly because the iteration state is already explicit.

## Smoke gate

Phase 6.1 contract:

1. `main.mind` parses cleanly under mindc v0.4.4 (`--emit-ir` produces a
   `module { ... }` block without errors).
2. The lexer's output on `fixture.mind` matches the table in
   `EXPECTED.md` byte-for-byte.

Item 1 is enforced by this commit. Item 2 is currently a manual
fixture; Phase 6.2 promotes it to a Cargo integration test.

## Why this matters

Self-hosting is the credibility milestone for any new language
(Mojo, Carbon, Zig, Rust all used it as their "we're serious" moment).
The lexer is the easiest end of the ladder — no recursive types, no
symbol tables, no codegen. If MIND can express its own lexer in pure
MIND on top of the seven RFC 0005 intrinsics, the next stages
(parser, type-checker, IR emitter) follow mechanically.

The deeper thesis is that **the same Q16.16 / heap-record / cross-arch
deterministic primitives that mind-nerve's Phase 2 native inference
needs are *also* what a self-hosted compiler needs**. Both stress-test
the same surface. Phase 6.1 + mind-nerve A1 share substrate work — one
proves the language is mature, the other proves the language is
useful.

## Companion docs

- [`docs/rfcs/0005-pure-mind-std-surface.md`](../../docs/rfcs/0005-pure-mind-std-surface.md) — RFC 0005 landing table
- [`docs/rfcs/0005-phase-d2b-design-note.md`](../../docs/rfcs/0005-phase-d2b-design-note.md) — sibling next-phase design note
- `mind-internal/plans/2026-05-18-thesis-apex-90day-plan.md` — §B1 strategy
