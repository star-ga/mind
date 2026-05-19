# Phase 6.4 — Expected IR Text

> Smoke gate for `examples/emit_ir/main.mind` when run against
> `examples/emit_ir/fixture.mind`.  The MIND-side emitter must emit
> *exactly* this byte stream when consuming the AST that
> `examples/parser/main.mind` produces on the same fixture.  Phase 6.5
> promotes this from a documented fixture to a Cargo integration test
> that compiles `main.mind` to a `.so`, drives the parser + emitter
> end-to-end, and diffs the resulting buffer byte-for-byte against
> the table below.

## Fixture

[`examples/emit_ir/fixture.mind`](./fixture.mind), 3 top-level items:

1. `use std.vec;`
2. `pub fn add(x: i64, y: i64) -> i64 { let z: i64 = x + y; return z; }`
3. `pub fn compute(x: i64, y: i64, z: i64) -> i64 { let r: i64 = x + y * z; add(r, x) }`

The Phase 6.1 lexer + Phase 6.2a parser walk that source and produce
the AST documented in [`../parser/EXPECTED.md`](../parser/EXPECTED.md)
(42 nodes total: 1 program, 1 use, 2 fn_def, 5 param, 2 block, 2 let,
1 return, 3 binop, 1 call, 24 ident).

## Expected IR text

The emitter must produce **exactly** the following 148 bytes (no trailing
newline beyond the one after `next_id = 3`):

```
module {
  %0 = const.i64 0
  output %0
  // fn add
  %1 = const.i64 0
  output %1
  // fn compute
  %2 = const.i64 0
  output %2
}  // next_id = 3
```

This is byte-for-byte the output of `cargo run --features "std-surface
cross-module-imports" --bin mindc -- examples/emit_ir/fixture.mind --emit-ir`
on the same fixture (Phase 6.4 contract: the MIND-side emitter and
mindc-Rust's `format_ir_module` are textually indistinguishable on
the fixture surface).

### Byte-level layout

| offset | bytes (ASCII)                | meaning                              |
|-------:|------------------------------|--------------------------------------|
|      0 | `module {\n`                 | module open                          |
|      9 | `  %0 = const.i64 0\n`       | implicit zero-result const           |
|     28 | `  output %0\n`              | implicit zero-result output          |
|     40 | `  // fn add\n`              | fn-def comment for `add`             |
|     53 | `  %1 = const.i64 0\n`       | per-fn const placeholder             |
|     72 | `  output %1\n`              | per-fn output                        |
|     84 | `  // fn compute\n`          | fn-def comment for `compute`         |
|    101 | `  %2 = const.i64 0\n`       | per-fn const placeholder             |
|    120 | `  output %2\n`              | per-fn output                        |
|    132 | `}  // next_id = 3\n`        | module close + terminal next_id      |
|    149 | (EOF)                        |                                      |

`next_id = 3` matches the parser fixture's documented exit value
(see `../parser/EXPECTED.md` step 2: "Expected: clean exit, IR dump
ending with `next_id = 3`").

## Output-format reference

The MIND-side emitter is keyed against `src/ir/print.rs::format_ir_module`
in mindc-Rust.  Per-instruction format strings the emitter must match:

| Instr             | Format                              |
|-------------------|-------------------------------------|
| `Instr::ConstI64` | `  {val_name} = const.i64 {value}`  |
| `Instr::BinOp`    | `  {dst} = {op} {lhs}, {rhs}`       |
| `Instr::Output`   | `  output {val_name}`               |
| `Instr::FnDef`    | `  // fn {name}`                    |
| module header     | `module {`                          |
| module trailer    | `}  // next_id = {n}`               |

Where `{val_name}` is `%K` for `ValueId(K)` and `{op}` is one of
`add` / `sub` / `mul` / `div` / `mod` / `lt` / `le` / `gt` / `ge` /
`eq` / `ne`.  The Phase 6.4 emitter implements the **7-op subset**
the parser produces (`add` / `sub` / `mul` / `div` / `lt` / `gt` /
`eq`) — the other four lower from comparison sugar the Phase 6.1
lexer does not yet recognise.

## Why the fixture has only zero-valued consts

mindc v0.4.4's `lower_expr` does not yet have a populated symbol
environment for fn bodies — it walks the AST and emits a single
`const.i64 0` + `output` pair per fn while logging
`[WARN] lower_expr: unhandled AST node kind, defaulting to 0` once
(per module).  Phase 6.4's emitter matches that behaviour exactly so
the byte-for-byte gate holds today.  Once mindc-Rust grows real per-
fn body lowering (Phase D2b on the RFC 0005 landing table), Phase 6.5
re-runs the gate on the same fixture and bumps EXPECTED.md to track.

## How to verify (manual, Phase 6.4)

1. **Parse-clean check for the emitter itself:**

   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/emit_ir/main.mind --emit-ir
   ```

   Expected: clean exit, IR dump ending with `next_id = 64` (one
   const+output pair per `pub fn` in the emitter, plus the implicit
   leading zero pair).

2. **Parse-clean check for the fixture:**

   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/emit_ir/fixture.mind --emit-ir
   ```

   Expected: clean exit, IR dump matching the table above byte-for-
   byte (`next_id = 3`).

3. **End-to-end gate (Phase 6.5 only):** once mindc v0.5.0 ships
   const-blob linkage, compile `main.mind` to a `.so`, feed the AST
   that `examples/parser/main.mind` produces on the fixture, and diff
   `stdout` against the 148-byte table above.

## Phase 6.5 hand-off

Phase 6.5 owner picks up `examples/bootstrap/main.mind` (new
directory) and writes the end-to-end driver:

```mind
use std.vec;
use std.string;
use std.io;

pub fn main() -> i64 {
    let src_buf: i64 = read_source_into_heap();
    let toks: Vec = lex(src_buf);
    let ast: i64 = parse(toks, src_buf);
    let ir: EmitState = lower_program(ast, src_buf);
    flush_to_stdout(ir)
}
```

The Phase 6.4 emitter's exported entry is
`lower_program(ast_addr: i64, buf: i64) -> EmitState`, returning a
heap-record carrying the byte buffer (`String`), terminal `next_id`,
and `last_id` slot.  Phase 6.5's driver wires the four phases together
into a `cdylib` that mindc v0.5.0's const-blob linkage compiles.  No
new mindc features or RFC 0005 intrinsics are required from this side
of the gate.

## Known Phase 6.4 limitations (resolved by Phase 6.5+)

1. The emitter still produces the "single const.i64 0 + output" per
   fn shape, mirroring mindc-Rust's current per-fn lowering.  When
   Phase D2b widens that lowering (real symbol-table-aware body
   walk), the EXPECTED.md tables in both this directory and
   `../parser/` update in lockstep.

2. Integer-literal lowering is implemented (`parse_int_span` +
   `emit_i64`) but not exercised by this fixture (the fixture's `let
   z: i64 = x + y` has no literal constants).  Phase 6.5 extends the
   fixture with literal-bearing expressions to exercise the digit-
   writer paths.

3. Identifier resolution is the same "default to 0" fallback
   mindc-Rust emits.  Phase 6.5 wires a pure-MIND symbol table
   (`std.map` is already in the bundled stdlib) and switches the
   identifier arm of `emit_expr` to look up the SSA id of the
   binding's init expression.

4. No error recovery.  Unknown AST kinds default to `const.i64 0`.
   Phase 6.5 introduces a Result-shaped emit state once the type-
   checker is wired and can route diagnostics.

5. Heap allocation per byte push (each `string_push_byte` may
   re-allocate the backing store).  Phase 6.5 lands an
   `__mind_memcpy` intrinsic so the amortised copy stays sub-linear;
   the byte stream the emitter produces is identical either way.

6. The emitter is single-pass and deterministic in walk order.
   That's required for byte-for-byte parity with mindc-Rust's
   format_ir_module; it's not a property the emitter can relax
   without breaking the gate.
