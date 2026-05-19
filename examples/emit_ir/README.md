# RFC 0005 Phase 6.4 — Self-Host MLIR Text Emitter

> Third step of the self-host ladder.  A pure-MIND IR emitter that
> walks the Phase 6.2a parser's AST and produces an MLIR-style text
> dump byte-identical to mindc-Rust's `--emit-ir` output on the same
> fixture.  Smoke contract: byte-for-byte parity with the canonical
> table in [EXPECTED.md](./EXPECTED.md).
>
> **Position in RFC 0005 landing table:** Phase 6 row, sub-row 6.4
> (`open` → `in flight` as of this commit).
>
> **Predecessors:**
> - Phase 6.1 self-host lexer (`examples/lexer/`, mindc v0.4.4)
> - Phase 6.2a self-host parser (`examples/parser/`, mindc v0.4.4)

## Files

| File | Purpose |
|---|---|
| `main.mind` | The emitter itself (~530 LOC pure MIND, tail-recursive walkers) |
| `fixture.mind` | The 3-decl smoke fixture (verbatim from `examples/parser/fixture.mind`) |
| `EXPECTED.md` | The exact 148-byte IR text the emitter must produce |

## What works

- `module { … }  // next_id = N` framing (matches `format_ir_module`)
- `// fn NAME` comments — name byte range copied from source span
- `%K = const.i64 V` instruction format with decimal value writer
- `%K = OP %i, %j` binop format (7 ops: add/sub/mul/div/lt/gt/eq)
- `output %K` instruction format
- Tail-recursive walkers for: program items, block stmts, call args
- Decimal-from-bytes integer-literal parser (`parse_int_span`)
- Decimal-to-bytes SSA-id and constant-value writer (`emit_i64`)
- Heap-record `EmitState { buf: String, next_id: i64, last_id: i64 }`
  threaded through every recursive call (RFC 0005 P0e Option-C ABI)
- `flush_to_stdout` wrapper that pushes the buffer through
  `std.io.__mind_write` in one syscall

## What's deferred to Phase 6.5

- The end-to-end driver: `cdylib` `main()` that wires lexer → parser
  → emitter → stdout.  Gated on mindc v0.5.0 const-blob linkage.
- Real fn-body lowering once mindc-Rust's `lower_expr` populates a
  symbol environment (Phase D2b).  Until then the emitter and
  mindc-Rust both produce a single `const.i64 0 + output` pair per
  fn — the byte-for-byte gate locks both into the same shape.
- A populated symbol table on the emitter side (so identifier
  references resolve to the SSA id of their let-binding's init
  expression instead of fallback-to-zero).
- An `__mind_memcpy` intrinsic so `string_push_byte` doesn't pay
  O(N²) on the per-push reallocation path.

## Build

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/emit_ir/main.mind --emit-ir
```

Expected output: clean exit, IR dump ending with `next_id = 64`.

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/emit_ir/fixture.mind --emit-ir
```

Expected output: clean exit, IR dump matching
[EXPECTED.md](./EXPECTED.md) byte-for-byte (`next_id = 3`).

The `std-surface` + `cross-module-imports` features are required
because the emitter's `use std.vec; use std.string; use std.io;`
resolves the bundled pure-MIND stdlib.  mindc v0.4.4 ships the
bundled stdlib via `include_str!` (RFC 0005 Phase C) so there's no
external dependency.

## Output format reference

The emitter is keyed against `src/ir/print.rs::format_ir_module` in
mindc-Rust.  Per-instruction format strings:

| Instr             | Format                              |
|-------------------|-------------------------------------|
| `Instr::ConstI64` | `  {val_name} = const.i64 {value}`  |
| `Instr::BinOp`    | `  {dst} = {op} {lhs}, {rhs}`       |
| `Instr::Output`   | `  output {val_name}`               |
| `Instr::FnDef`    | `  // fn {name}`                    |
| Module header     | `module {`                          |
| Module trailer    | `}  // next_id = {n}`               |

`{val_name}` is `%K` for `ValueId(K)`.  Binop mnemonics: `add`,
`sub`, `mul`, `div`, `mod`, `lt`, `le`, `gt`, `ge`, `eq`, `ne`.  The
Phase 6.4 emitter implements the 7-op subset the parser produces
(`add` / `sub` / `mul` / `div` / `lt` / `gt` / `eq`); the other four
land when the lexer learns the corresponding two-char operators.

## Heap-record discipline

Every reference threaded through the emitter is an `i64`:

- AST nodes: 7×i64 records allocated by the parser
- `Vec<i64>` of child node addresses: stride-8 heap blocks
- `String` byte buffer: `addr/len/cap` record from `std.string`
- `EmitState`: `buf/next_id/last_id` record (Option-C struct ABI)

No built-in pointer types, no ADT-with-payload syntactic sugar, no
allocator beyond `__mind_alloc` and the heap-record convention.  The
same Option-C ABI that lowered `Vec`, `String`, `ParseResult`, and
the AST nodes lowers `EmitState` here; the proof is structural.

## Loop strategy: still tail recursion

mindc v0.4.4 does not accept `while` as a statement (gap 1 in
[`../../docs/rfcs/0005-phase-6-2-mindc-gaps.md`](../../docs/rfcs/0005-phase-6-2-mindc-gaps.md)).
Every loop in `main.mind` is therefore a tail recursion:

- `emit_digits_collect(n, acc) -> Vec` — decimal-digit peel-off for
  `emit_i64`.
- `emit_digits_flush(s, digits, i) -> EmitState` — reverse-order
  digit flush.
- `emit_byte_range(s, buf, lo, hi) -> EmitState` — fn-name byte copy.
- `emit_call_args(s, buf, addr, len, i) -> EmitState` — call-arg
  lowering loop.
- `emit_block_stmts(s, buf, addr, len, i) -> EmitState` — block-stmt
  lowering loop.
- `emit_program_items(s, buf, addr, len, i) -> EmitState` — top-level
  item loop.

If Phase 6.5 adds `while` to mindc, these recursions port back
cleanly because the iteration state is already explicit.

## Smoke gate

Phase 6.4 contract:

1. `main.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces a `module { ... }` block with `next_id = 64` and no
   parse errors).
2. `fixture.mind` parses cleanly under mindc v0.4.4 (`--emit-ir`
   produces `next_id = 3`).
3. The emitter's output on `fixture.mind`'s AST matches the table
   in [EXPECTED.md](./EXPECTED.md) byte-for-byte.

Items 1 + 2 are enforced by this commit.  Item 3 is currently a
manual fixture; Phase 6.5 promotes it to a Cargo integration test
that compiles `main.mind` to a `.so`, drives it through the parser's
AST output, and diffs the resulting buffer byte-for-byte against
EXPECTED.md.

## Why this matters

The lexer proved MIND can chew through bytes and produce a flat
typed stream.  The parser proved MIND can build recursive typed
structures.  This emitter closes the trio: MIND can walk a typed
recursive structure and produce **textual output that's
indistinguishable from the host compiler's**.  Three stages, three
proofs:

1. bytes-in → tokens-out (Phase 6.1, 254 bytes → 32 tokens)
2. tokens-in → AST-out (Phase 6.2a, 32 tokens → 42 AST nodes)
3. AST-in → text-out (Phase 6.4, 42 AST nodes → 148 bytes)

The next stage (Phase 6.5 bootstrap driver) wires them together
into a `cdylib` `main()` and runs the round trip end-to-end.  If the
output of the MIND-side pipeline byte-for-byte matches mindc-Rust's
own `--emit-ir` output on the same source, MIND has demonstrated a
**non-trivial self-host fragment**: not a full bootstrap (no MLIR
lowering on the MIND side yet), but enough of the front-end + IR
text emit to prove the language is mature enough to bootstrap the
rest.

## Companion docs

- [`../../docs/rfcs/0005-pure-mind-std-surface.md`](../../docs/rfcs/0005-pure-mind-std-surface.md) — RFC 0005 landing table
- [`../../docs/rfcs/0005-phase-6-2-mindc-gaps.md`](../../docs/rfcs/0005-phase-6-2-mindc-gaps.md) — mindc feature-gap design note (no *new* gaps surfaced during Phase 6.4; the emitter ships entirely within the v0.4.4 surface grammar)
- [`../lexer/`](../lexer/) — Phase 6.1 predecessor (tokeniser)
- [`../parser/`](../parser/) — Phase 6.2a predecessor (parser)
- [`../parser/EXPECTED.md`](../parser/EXPECTED.md) — input contract for this emitter

## Phase 6.5 hand-off

Phase 6.5 owner picks up `examples/bootstrap/main.mind` (new
directory) and writes the end-to-end driver:

```mind
use std.vec;
use std.string;
use std.io;

pub fn main() -> i64 {
    // 1. Read source bytes into a heap buffer.
    let src_buf: i64 = read_source_into_heap();
    // 2. Lex.
    let toks: Vec = lex(src_buf);
    // 3. Parse.
    let ast: i64 = parse(toks, src_buf);
    // 4. Emit IR text.
    let ir: EmitState = lower_program(ast, src_buf);
    // 5. Flush.
    flush_to_stdout(ir)
}
```

The Phase 6.4 emitter's exported entry is
`lower_program(ast_addr: i64, buf: i64) -> EmitState`, returning a
heap-record carrying the byte buffer, terminal `next_id`, and
`last_id`.  Phase 6.5 wires this into a `cdylib` once mindc v0.5.0
ships const-blob linkage (the gap that currently blocks compiling a
`.mind` file with multiple module-level `use` statements into a
self-contained dylib).

No new mindc feature gaps surfaced during Phase 6.4 — the emitter
ships entirely within the v0.4.4 surface grammar.
