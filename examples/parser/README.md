# RFC 0005 Phase 6.2 — Self-Host Parser Seed

> Second step of the self-host ladder.  A pure-MIND Pratt parser
> that consumes the Phase 6.1 lexer's stride-3 `Vec<i64>` token
> stream and produces an AST built from RFC 0005 Option-C heap-
> record nodes.  Smoke contract: the AST shape on a fixed reference
> fixture matches [EXPECTED.md](./EXPECTED.md) node-for-node.
>
> **Position in RFC 0005 landing table:** Phase 6 row, sub-row 6.2
> (`open` → `in flight` as of this commit).
>
> **Predecessor:** Phase 6.1 self-host lexer (`examples/lexer/`,
> commit `29bd08b`, mindc v0.4.4).

## Files

| File | Purpose |
|---|---|
| `main.mind` | The parser itself (~620 LOC pure MIND, tail-recursive Pratt loop) |
| `fixture.mind` | A short `.mind` source used as the smoke gate (extends the lexer fixture with calls + a precedence-mixed binop chain + multi-arg signatures) |
| `EXPECTED.md` | The exact AST tree `main.mind` must produce on `fixture.mind` |

## Grammar supported

```
program     := item*
item        := use_stmt | fn_def
use_stmt    := "use" IDENT "." IDENT ";"
fn_def      := "pub"? "fn" IDENT "(" params ")" "->" IDENT block
params      := (param ("," param)*)?
param       := IDENT ":" IDENT
block       := "{" stmt* "}"
stmt        := let_stmt | return_stmt | expr_stmt
let_stmt    := "let" IDENT ":" IDENT "=" expr ";"
return_stmt := IDENT("return") expr ";"
expr_stmt   := expr ";"?
expr        := pratt(0)
pratt(p)    := prefix (infix_op pratt(prec+1))*
prefix      := INT_LIT | IDENT | IDENT "(" args ")" | "(" expr ")"
args        := (expr ("," expr)*)?
infix_op    := "+" | "-" | "*" | "/" | "<" | ">" | "="
```

## Pratt precedence table

| ops          | precedence |
|--------------|-----------:|
| `=`          | 1          |
| `<` `>`      | 2          |
| `+` `-`      | 3          |
| `*` `/`      | 4          |

All operators are left-associative.

## AST node shape

Every AST node is a 7×i64 heap record (56 bytes), allocated via
`__mind_alloc(56)` and accessed via `__mind_load_i64` /
`__mind_store_i64` at fixed offsets.  See
[EXPECTED.md](./EXPECTED.md) for the field-by-field schema.

12 AST kinds:

```
ast_int_lit  ast_ident   ast_binop   ast_call
ast_fn_def   ast_let     ast_use     ast_return
ast_param    ast_block   ast_program ast_paren
```

Variable-length child lists (block statements, fn parameters, call
arguments, program items) live in a `Vec<i64>` of node addresses;
the Vec's base address is stored in the parent node's `child0` slot
and its logical length in `aux`.  This means **no** AST kind needs a
non-`i64` field — the whole tree is opaque-address-only, which is
the RFC 0005 P0a discipline.

## ParseResult plumbing

Every recursive parse helper returns

```mind
struct ParseResult {
    next_pos: i64,
    node: i64,
}
```

where `node` is the i64 base-address of the constructed AST.  This
keeps the threading explicit (no implicit parser state), which is
load-bearing for the tail-recursion-only constraint.

## Build

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/parser/main.mind --emit-ir
```

Expected output: clean exit, IR dump ending with `next_id = 90`.

The `std-surface` + `cross-module-imports` features are required
because the parser's `use std.vec;` resolves the bundled pure-MIND
Vec implementation.  mindc v0.4.4 ships the bundled stdlib via
`include_str!` so there's no external dependency.

The IR dump shows every fn lowering to `const.i64 0` and a series of
`[WARN] lower_expr: ...` lines — that's the same behaviour the Phase
6.1 lexer prints.  v0.4.4's parser is fully functional but its IR
lowering is intentionally incomplete (Phase D2b ships the full
lowering).  The Phase 6.2 contract is **parse-clean**, not run-clean.

## Loop strategy: still tail recursion

mindc v0.4.4's parser does not accept `while` as a statement (see
[`docs/rfcs/0005-phase-6-2-mindc-gaps.md`](../../docs/rfcs/0005-phase-6-2-mindc-gaps.md)
Gap 1).  Every loop in `main.mind` is therefore expressed as tail
recursion:

- `parse_pratt(toks, pos, buf, lhs, min_prec) -> ParseResult` — the
  Pratt fold loop, eats one infix op + rhs per call, terminates when
  the next token's precedence drops below `min_prec`.
- `parse_args_rest(toks, pos, buf, acc) -> ParseResult` — comma-
  separated call argument list, terminates on `)`.
- `parse_params_rest(toks, pos, buf, acc) -> ParseResult` — comma-
  separated fn parameter list, terminates on `)`.
- `parse_block_stmts(toks, pos, buf, acc) -> ParseResult` — `{ ... }`
  body, terminates on `}` or EOF.
- `parse_items_rest(toks, pos, buf, acc) -> ParseResult` — top-level
  item loop, terminates on EOF.

If Phase 6.3 adds `while` to mindc, these recursions port back
cleanly because the iteration state is already explicit.

## Smoke gate

Phase 6.2 contract:

1. `main.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces a `module { ... }` block with `next_id = 90` and no
   parse errors).
2. `fixture.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces `next_id = 3`).
3. The parser's output on `fixture.mind` matches the tree in
   [EXPECTED.md](./EXPECTED.md) node-for-node.

Items 1 + 2 are enforced by this commit.  Item 3 is currently a
manual fixture; Phase 6.3 promotes it to a Cargo integration test
that compiles `main.mind` to a `.so`, feeds `fixture.mind`'s bytes
through it, and walks the heap-record tree against the expected
table.

## What works in Phase 6.2

- Use statements: `use std.vec;` (head + tail ident, dot ignored)
- Function definitions with `pub` prefix, multi-param signatures,
  arrow return-type syntax, and block bodies
- Function parameters: `IDENT : IDENT` pairs, comma-separated
- Let bindings: `let NAME : TYPE = EXPR ;`
- Return statements (detected via byte-compare on `tk_ident "return"`)
- Expression statements (with optional trailing `;`)
- Integer literals, identifiers
- Function calls `IDENT(arg, arg, ...)` with arbitrary arity
- Parenthesised expressions `(EXPR)` with span propagation
- Pratt-style binary operators across four precedence levels
- Block bodies `{ stmt* }` with proper EOF / `}` termination
- Top-level item driver loop

## Deferred to Phase 6.3 (type-checker)

- A real `Result`-shaped node carrying diagnostics (instead of soft-
  fallback ident leaves on unexpected tokens)
- Function return-type storage on the `ast_fn_def` node (requires
  widening the 7×i64 heap record or a side-table)
- Promotion of `return` from byte-compare to a dedicated `tk_kw_return`
  (requires Phase 6.1 lexer keyword-table growth)
- A symbol table walking the AST and resolving identifier references
- Type inference for `let` bindings (drops the explicit `: TYPE`
  annotation when unambiguous)

## Deferred to Phase 6.4 (MLIR emit)

- Walking the AST and emitting MLIR (currently mindc-Rust does this
  for `.mind` inputs; Phase 6.4 makes the MIND-side parser drive the
  same emission path through a `lower_program(ast_addr) -> i64`
  fn in MIND)

## Why this matters

The lexer proved MIND can chew through bytes and produce a flat
typed stream.  The parser proves MIND can build **recursive
typed structures** entirely on top of the seven RFC 0005
intrinsics and the bundled stdlib — no built-in pointer type, no
ADT-with-payload syntactic sugar, no allocator beyond the heap-
record discipline.  The same Option-C ABI that lowered `Vec` and
`String` in Phase 2 also lowers a full AST in Phase 6.2; the proof
is structural, not by reflection.

The next stage (Phase 6.3 type-checker) is the credibility wall.
If MIND can walk its own AST, build a symbol table, and check types
in pure MIND, the language is self-evidently mature enough to bootstrap.

## Companion docs

- [`docs/rfcs/0005-pure-mind-std-surface.md`](../../docs/rfcs/0005-pure-mind-std-surface.md) — RFC 0005 landing table
- [`docs/rfcs/0005-phase-6-2-mindc-gaps.md`](../../docs/rfcs/0005-phase-6-2-mindc-gaps.md) — mindc feature-gap design note (no new gaps surfaced during Phase 6.2; the parser ships entirely within the v0.4.4 surface grammar)
- [`examples/lexer/`](../lexer/) — Phase 6.1 predecessor (tokeniser)
- [`examples/lexer/EXPECTED.md`](../lexer/EXPECTED.md) — input contract for this parser

## Phase 6.3 hand-off

Phase 6.3 owner picks up `examples/typeck/main.mind` (new directory)
and writes:

```mind
use std.vec;
use std.map;     // symbol table

pub fn typeck(ast_root: i64) -> i64 {
    // walk the ast_program node, build a symbol table, check types,
    // return 0 on success or an error-node addr on failure.
}
```

The Phase 6.2 parser's exported entry is `parse(toks: Vec, buf: i64)
-> i64`, returning the heap-record base address of the `ast_program`
root.  Phase 6.3's `typeck` takes that root and walks it via the
`ast_kind` / `ast_child0` / `ast_child1` / `ast_child2` / `ast_aux`
accessors documented in `main.mind`.  No additional intrinsics or
mindc features are required; std.map (Phase 5b) is already in the
bundled stdlib for the symbol table.
