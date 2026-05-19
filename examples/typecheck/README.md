# RFC 0005 Phase 6.3 — Self-Host Type-Checker Seed

> Third step of the self-host ladder.  A pure-MIND walker over the
> AST that the Phase 6.2 parser produces.  Emits a `String` report
> documenting every binding it sees and every return-type check it
> performs.  Smoke contract: the report on a fixed reference fixture
> matches [EXPECTED.md](./EXPECTED.md) byte-for-byte.
>
> **Position in RFC 0005 landing table:** Phase 6 row, sub-row 6.3
> (`open` → `in flight` as of this commit).
>
> **Predecessors:** Phase 6.1 self-host lexer (`examples/lexer/`),
> Phase 6.2 self-host parser (`examples/parser/`), both shipped at
> mindc v0.4.4.

## Files

| File | Purpose |
|---|---|
| `main.mind` | The type-checker itself (~580 LOC pure MIND, tail-recursive walkers) |
| `fixture.mind` | A short `.mind` source used as the smoke gate (extends the parser fixture with a third fn that exercises the `<` → `ty_bool` branch) |
| `EXPECTED.md` | The exact 6-line type-check report `main.mind` must produce on `fixture.mind` |

## What gets type-checked

Phase 6.3 covers the surface the Phase 6.2 parser already produces:

```
ast_program     → walk items
ast_use         → no-op (no name binding)
ast_fn_def      → walk body in fresh env; emit "fn NAME : (P0, ...) -> R" line
ast_param       → bind { name → declared type } in fn env
ast_block       → walk stmts under env
ast_let         → bind { name → T }; emit "let NAME : <T>" line
ast_return      → assert expr type == fn's declared return; emit ERROR on mismatch
ast_int_lit     → ty_i64
ast_ident       → env lookup; ty_unknown if not bound
ast_binop       → ty_bool for `<`, `>`, `=`; type of operand 0 for `+`, `-`, `*`, `/`
ast_paren       → type of inner expr (transparent)
ast_call        → ty_unknown (full sig matching deferred to Phase 6.3b)
```

## Type tags

Seven `ty_*` constants (i64), frozen for Phase 6.3:

| tag | name         | source spelling |
|-----|--------------|-----------------|
| 0   | `ty_unknown` | `?`             |
| 1   | `ty_i64`     | `i64`           |
| 2   | `ty_f64`     | `f64`           |
| 3   | `ty_bool`    | `bool`          |
| 4   | `ty_vec`     | `Vec`           |
| 5   | `ty_string`  | `String`        |
| 6   | `ty_unit`    | `()`            |

`resolve_type_ident(buf, lo, hi)` is the byte-range → tag mapper.
Phase 6.3b grows it to consume a `Map` of program-declared
struct/enum names once the parser learns `struct` / `enum` grammar.

## Symbol-table key encoding

`std.map` keys are i64.  Identifier names live in the source byte
buffer as `(lo, hi)` spans, so we fold each name down to a single
i64 via a DJB2-style hash:

```mind
pub fn name_hash_rest(buf, lo, hi, acc) -> i64 {
    if lo >= hi { return acc; }
    let b = load_byte(buf, lo);
    let mixed = (acc * 33) + b;
    name_hash_rest(buf, lo + 1, hi, mixed)
}
```

DJB2 was chosen over FNV-1a 64-bit because mindc v0.4.4's parser
treats integer literals as signed i64 and rejects FNV-1a's
`0xCBF29CE484222325` offset basis as integer overflow.  DJB2's
seed (`5381`) and prime (`33`) stay inside the signed-i64 range
across the short identifiers Phase 6.3 fixtures contain.  Both
hashes are deterministic — same input → same key on every run,
which is what `std.map`'s insertion-order contract requires for
evidence-chain reproducibility.

Collision probability across the Phase 6.3 fixture's 11 distinct
identifiers (`add`, `x`, `y`, `z`, `compute`, `r`, `cmp`, `b`,
`i64`, `bool`, `Vec`) is 0 — verified by hand.  Phase 6.3b grows
a secondary disambiguation check (byte-range compare on hash hits)
if a real-world fixture surfaces a collision.

## TcResult plumbing

Mirrors the parser's `ParseResult` shape — every walker returns

```mind
struct TcResult {
    env: Map,
    report: String,
}
```

so the env and report-so-far thread explicitly through each
recursive call.  Same load-bearing reason: tail-recursion-only
forbids implicit state.

## Build

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/typecheck/main.mind --emit-ir
```

Expected output: clean exit, IR dump ending with `next_id = 90`.

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/typecheck/fixture.mind --emit-ir
```

Expected: clean exit, IR dump ending with `next_id = 4`.

The `std-surface` + `cross-module-imports` features carry the
bundled stdlib (`std.vec`, `std.map`, `std.string`).  v0.4.4 ships
all three via `include_str!` so there's no external dependency.

The IR dump shows every fn lowering to `const.i64 0` for the same
reason the lexer + parser dumps do — v0.4.4's parser is fully
functional but its IR lowering is intentionally incomplete (Phase
D2b ships the full lowering).  Phase 6.3's contract is **parse-clean**,
not run-clean.

## Loop strategy: still tail recursion

mindc v0.4.4's parser does not accept `while` as a statement
(`docs/rfcs/0005-phase-6-2-mindc-gaps.md` Gap 1).  Every walk in
`main.mind` is therefore expressed as tail recursion:

- `tc_items_rest`        — top-level item loop
- `tc_block_stmts_rest`  — block statement loop
- `tc_params_rest`       — fn parameter loop (also emits header bytes)
- `env_lookup_rest`      — linear scan over the insertion-ordered map
- `name_hash_rest`       — byte-by-byte DJB2 fold
- `bytes_eq_rest`        — byte-by-byte range compare
- `push_bytes`           — byte-by-byte append into a String
- `push_i64`             — left-fold of decimal digits

If Phase 6.3 / 6.4 promotes `while` into mindc's surface grammar,
every walker here ports back cleanly because the iteration state
is already in fixed-state explicit form.

## Smoke gate

Phase 6.3 contract:

1. `main.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces a `module { ... }` block with `next_id = 90` and no
   parse errors).
2. `fixture.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces `next_id = 4`).
3. The type-checker's report on `fixture.mind` matches the text
   in [EXPECTED.md](./EXPECTED.md) byte-for-byte (124 bytes total
   across 6 LF-terminated lines).

Items 1 + 2 are enforced by this commit.  Item 3 is currently a
manual fixture; Phase 6.3b promotes it to a Cargo integration test
that links the type-checker's compiled `.so` and diffs the report
bytes against the documented table.

## What works in Phase 6.3

- Top-level `use` statements skip cleanly (no env effect)
- Top-level `pub? fn NAME (params) -> RET { body }` definitions —
  header line emitted, body walked under fresh env
- Multi-parameter signatures with comma-separated types in the
  header line
- `let NAME : TYPE = INIT ;` — env binding + report line
- `return EXPR ;` — return-type check (ERROR line on mismatch,
  silent on match)
- Identifier env lookups via DJB2-hashed name keys in `std.map`
- Binop result-type rules: `<` `>` `=` → `ty_bool`; `+` `-` `*` `/`
  → type of LHS operand
- Parenthesised expressions are transparent (type of inner)
- Type-name resolution for the 5 built-in type names + `ty_unknown`
  fallback for everything else
- Report bytes assembled into a `std.string.String` with pure
  byte-level appends — no UTF-8 normalisation, byte-identical
  output across hosts

## Deferred to Phase 6.3b

- Full call-signature matching (`ast_call` types to `ty_unknown`
  today; needs the parser to first stash the fn's declared return-
  type ident on `ast_fn_def`)
- Numeric promotion + mismatch diagnostics on let bindings
- Cross-fn name resolution (top-level fns get hoisted into program
  env so `add(...)` from inside `compute` resolves to the real sig)
- `return` keyword promotion in the Phase 6.1 lexer (Gap 3 in the
  mindc-gaps doc)
- Struct / enum / user-defined type-name resolution
- A real `Result`-shaped node carrying diagnostics (instead of an
  inline ERROR line in the report)

## Deferred to Phase 6.4 (MLIR emit)

- Walking the type-annotated AST + emitting MLIR directly from MIND
  (currently mindc-Rust does this for `.mind` inputs; Phase 6.4
  makes the MIND-side type-checker + emitter drive the same
  emission path through a `lower_program(typed_ast) -> i64` fn in
  MIND)

## Why this matters

The lexer proved MIND can chew through bytes.  The parser proved
MIND can build recursive typed structures.  The type-checker proves
MIND can **maintain a name-keyed environment, walk that environment
recursively across a recursive AST, emit a byte-precise report, and
match its own declared return-type discipline** — all on top of the
seven RFC 0005 intrinsics, three stdlib modules (`std.vec`,
`std.map`, `std.string`), and the AST helpers the Phase 6.2 parser
already published.  No new compiler features required.  No new
intrinsics.  No host calls.

The next stage (Phase 6.4 MLIR emit) closes the bootstrap loop:
once `typecheck → emit` ships entirely in MIND, the compiler can
walk its own input through its own type-checker into its own
backend, and the `.mind` source becomes its own canonical reference.

## Companion docs

- [`docs/rfcs/0005-pure-mind-std-surface.md`](../../docs/rfcs/0005-pure-mind-std-surface.md) — RFC 0005 landing table
- [`docs/rfcs/0005-phase-6-2-mindc-gaps.md`](../../docs/rfcs/0005-phase-6-2-mindc-gaps.md) — mindc feature-gap design note (Phase 6.3 surfaces a new minor gap, appended below the existing two)
- [`examples/parser/`](../parser/) — Phase 6.2 predecessor (AST source)
- [`examples/parser/EXPECTED.md`](../parser/EXPECTED.md) — input contract for this type-checker
- [`std/map.mind`](../../std/map.mind) — insertion-order Map used as the symbol table
- [`std/string.mind`](../../std/string.mind) — byte buffer used for the report

## Phase 6.3b / 6.4 hand-off

Phase 6.4 (MLIR emit) owner picks up `examples/emit_ir/main.mind`
(new directory, parallel to this one).  Phase 6.3's exported entry is

```mind
pub fn typecheck(ast_root: i64, buf: i64) -> String
```

returning the byte buffer of the report.  Phase 6.4's `emit_ir`
takes the same `(ast_root, buf)` pair plus the type-checker's
verdict (currently the report `String`; Phase 6.3b will return a
richer `TypedAst` handle) and walks the AST a second time to emit
MLIR.  No additional intrinsics or mindc features are required for
Phase 6.3 itself; Phase 6.3b needs the parser to first widen
`ast_fn_def` to carry the declared return-type ident — see
[EXPECTED.md](./EXPECTED.md) "What's deferred to Phase 6.3b".
