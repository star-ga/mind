# Mindcraft Phase 2 — Implementation Plan

> Working document for the `mindc fmt` scaffolding rung of RFC 0007.
> Phase 1 (`MindcraftConfig` manifest surface) landed at `mind@6526029`.

## Status

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | `MindcraftConfig` types in `Mind.toml` | ✅ shipped (`6526029`) |
| 2A | `mindc fmt` scaffolding, no line-wrap | planned (this doc) |
| 2B | Soft line-wrap at `max_line_length` | deferred |
| 2C | Migrate formatter from Rust to pure-MIND | gated on stable AST ABI |
| 3 | Rule infrastructure | blocked on 2A |
| 4 | First 5 lint rules | blocked on 3 |
| 5 | `mindc check` project driver | blocked on 4 |
| 6 | `--fix` flag + CI integration | blocked on 5 |

## Summary

A new `mindc fmt` subcommand canonicalises `*.mind` source files in
place, with `--check` (CI gate), `--diff` (unified-diff preview) and
`--stdin` (editor integration) modes. It loads `[mindcraft.format]`
from `Mind.toml`, falls back to the canonical defaults already in
`MindcraftFormatConfig`, walks files/directories of `*.mind`, parses
through the existing pure-MIND front-end (extended to preserve
comments as a CST trivia layer), and emits a token-faithful
pretty-print.

Phase 2A ships the scaffolding, indentation, trailing-comma
normalisation, whitespace normalisation, and comment re-attachment.
**Soft line-wrapping at `max_line_length` is deferred to Phase 2B.**

## Critical finding from planning

**The existing parser drops comments before parsing.**

`/home/n/mind/src/parser/mod.rs:3159-3186` runs `strip_comments` before
parsing; `:95-115` discards `//`-to-EOL during `skip_ws_and_newlines`.
The AST `Node` enum (`/home/n/mind/src/ast/mod.rs:200`) has no
comment/trivia variant; `Module.items` is `Vec<Node>` with no
leading-trivia field.

The formatter cannot achieve "token-faithful round-trip" without
addressing this first. Step 1 of the implementation sequence does so
via a parallel CST/trivia layer (Option B in the planner's analysis)
— no AST change, optional `TriviaCollector` parameter that captures
`(byte_offset_in_original, TriviaKind, text)` records.

## Architecture decision: Rust scaffolding now, pure-MIND in Phase 2C

RFC 0007 §3 makes "pure MIND" non-negotiable in the long run. §7 has
an explicit escape hatch for transitional bridges that drive the
compiler's existing front-end.

**Decision: Phase 2 ships as Rust scaffolding consuming the same AST
the self-hosted parser produces.** Rationale:

1. The pure-MIND front-end (`libmindc_mind.so`) does not yet expose
   a public AST API for dlopen consumers — wiring an ABI for that is
   itself a multi-week task.
2. The formatter is a mechanical AST walk; doing it in Rust
   short-term doesn't regress §7 as long as it consumes the same AST.
3. Migrating to pure MIND becomes a contained refactor once the
   formatter has full test coverage (Phase 2C).

Tracked separately as Phase 2C, gated on a stable in-process AST ABI
from `libmindc_mind.so`.

## CLI surface

```
mindc fmt [PATHS...] [--check] [--diff] [--stdin]
```

Flags:
- `--check` — exit non-zero if any file would change; no write.
- `--diff` — print unified diff to stdout; no write.
- `--stdin` — read source from stdin, write formatted output to stdout.
- Positional paths — files or directories; empty defaults to current dir.

Exit codes: `0` clean, `1` drift / error, `2` invalid usage.

## Crate layout

```
src/fmt/
├── mod.rs         — public entry: format_source(src, cfg) -> Result<String>
├── cst.rs         — CST + trivia layer
├── printer.rs     — AST walker, no line-wrap in 2A
└── diff.rs        — unified-diff producer for --diff (use similar crate)
src/bin/mindc_fmt.rs — CLI dispatcher (keeps mindc.rs slim)
```

## Formatter rules (Phase 2A)

- Indent: `cfg.indent_width` spaces (default 4) per nesting. Never tabs.
- Top-level spacing: single blank line between items; no leading/trailing
  blank line in file; exactly one `\n` at EOF.
- Trailing comma when `cfg.trailing_comma == true` AND multi-line, on
  struct literals, struct decl fields, fn params, fn args, match arm
  bodies, array literals. Single-line constructs never get one.
- Whitespace normalisation: collapse internal multi-space; one space
  after `,`; one space around binary operators (`+ - * / % == != < > <=
  >= && ||`); `**`/`..`/`..=` stay tight; no space inside `()`/`[]`/`{}`
  for single-line constructs.
- String-literal contents passed through bytewise — never modified.
- Blank-line collapse inside `{ ... }` fn bodies; max 1 consecutive;
  leading/trailing blank lines stripped.
- Comment attachment: own-line comment before item attaches to item;
  trailing on same line stays with item; `///` doc-comments attach to
  next following item; copyright/license header preserved at top.

Idempotence: every emit rule is a fixed point.

## Stability tests (the acceptance gate)

Three orthogonal tests:

1. `tests/fmt_stdlib_stability.rs` — for every `std/*.mind`,
   `format_source(read(path), default()) == read(path)`. The killer
   test: the std style IS the canon.
2. `tests/fmt_idempotence.rs` — `format(format(src)) == format(src)`
   for std + examples + fixtures.
3. `tests/fmt_ir_preservation.rs` — `emit_mic(parse(src)) ==
   emit_mic(parse(format(src)))` byte-exact. Formatting must never
   change MIC IR.

If stdlib stability finds drift, the resolution rule: **either patch
the formatter to match the file's style, or patch the file to match
canon — file changes require explicit review.** Default: formatter
conforms to existing std.

## Test fixtures (`tests/mindcraft/fmt/`)

| # | Name | Exercises |
|---|------|-----------|
| 01 | `indent_if_else` | nested if/else indentation |
| 02 | `struct_literal_multiline` | short-form vs multi-line, trailing comma |
| 03 | `fn_args_multiline` | wrapped fn signature |
| 04 | `trailing_comma_toggle` | with sidecar `Mind.toml` setting false |
| 05 | `internal_whitespace` | `1   +    2 *   3` → `1 + 2 * 3` |
| 06 | `comment_attachment` | leading/trailing/doc + copyright header |
| 07 | `string_literal_passthrough` | double-space + `//` lookalike in string |
| 08 | `long_call_no_wrap` | >100 char line stays as-is in 2A; wraps in 2B |
| 09 | `self_test` / `mindc_fmt_self_test` | every Node variant the formatter handles |

## PR sequence (one per shippable step)

| # | PR | Branch | Risk |
|---|-----|--------|------|
| 1 | `feat(parser): trivia layer for comment-preserving CST` | `mindcraft/cst-trivia` | M |
| 2 | `feat(fmt): scaffolding + canonical walker (Phase 2A, no wrap)` | `mindcraft/fmt-walker` | M |
| 3 | `test(fmt): stdlib stability + idempotence + IR-preservation` | `mindcraft/fmt-stability` | L-M |
| 4 | `feat(mindc): fmt subcommand (--check/--diff/--stdin)` | `mindcraft/fmt-cli` | L |
| 5 | `bench(fmt): full-repo budget benchmark` | `mindcraft/fmt-bench` | L |
| 6 | `docs(mindcraft): fmt rules reference + RFC 0007 status` | `mindcraft/fmt-docs` | L |
| 7 | `feat(fmt): Phase 2B — soft line wrap at max_line_length` | `mindcraft/fmt-wrap` | H |
| 8 | `docs(site): Mindcraft Phase-2 card on mindlang.dev` | `mindlang.dev/mindcraft-phase-2` | L |

PRs 1–6 = Phase 2A (~1 week). PR 7 = Phase 2B (~2 weeks, doesn't block
Phase 3). PR 8 = public credibility update after 2A lands.

## Bench gate

`mindc fmt` on the full mind repo (~117 `.mind` files) must complete in
<2.0s on the dev box (i7-5930K). Per-file cost: parser + walker +
string emit, all O(n) — ~5ms typical → ~600ms expected for full repo,
~1.5s worst case. Comfortably under budget.

Add criterion bench `bench/mindcraft_fmt.rs` measuring per-file format
time on three representative files (`std/vec.mind`, the largest
`examples/mindc_mind/main.mind`, a synthetic 1000-LOC stress file).
Tie into the +7% bench-gate ceiling.

## Out of scope for Phase 2

- `mindc lint`, `mindc check` (Phase 3, 4, 5)
- Lint-rule engine, rule-as-`.mind` files (Phase 3)
- Project walker, VCS-aware filtering, `[mindcraft.vcs]` (Phase 5)
- `[[mindcraft.overrides]]` glob layering (Phase 5)
- `[mindcraft.cpu]` per-target sections (Phase 3+, not fmt-relevant)
- `--reporter json` (out of scope; `--check` exit code + `--diff` is
  the machine surface)
- Full LSP / editor integration (`--stdin` is the hook)
- `--fix` (fmt itself is the fix)
- Soft line wrapping (Phase 2B)

## Relevant files

- `docs/rfcs/0007-mindcraft.md` (spec)
- `src/project/mod.rs:51-222` (Phase 1 manifest types — already shipped)
- `src/bin/mindc.rs:53-124` (where the `Fmt` variant goes)
- `src/parser/mod.rs:95-115, 3156-3228` (comment-stripping that Step 1
  must convert to a trivia layer)
- `src/ast/mod.rs:200, 666` (AST that stays unchanged)
- `std/vec.mind` (canonical style reference)
- `examples/mindc_mind/main.mind` (largest extant `.mind` source; bench
  reference)
