# Phase 6.2 — Expected AST Tree

> Smoke gate for `examples/parser/main.mind` when run against
> `examples/parser/fixture.mind`.  The MIND parser must emit *exactly*
> this AST when consuming the token stream produced by
> `examples/lexer/main.mind` on the fixture.  Phase 6.3 promotes this
> from a documented fixture to a Cargo integration test that links
> the parser's compiled `.so` and walks the heap-record tree against
> the table below.

## Fixture

[`examples/parser/fixture.mind`](./fixture.mind), 8 declarations:

1. `use std.vec;`
2. `pub fn add(x: i64, y: i64) -> i64 { let z: i64 = x + y; return z; }`
3. `pub fn compute(x: i64, y: i64, z: i64) -> i64 { let r: i64 = x + y * z; add(r, x) }`

The Phase 6.1 lexer tokenises this into a single flat `Vec<i64>`
of stride-3 triples terminated by a `tk_eof()` triple.  The parser
walks that stream and produces the AST shown below.

## AST node tags

Frozen for Phase 6.2 (`main.mind` `ast_*` constants):

| tag | name           | child0           | child1            | child2  | aux                                |
|-----|----------------|------------------|-------------------|---------|------------------------------------|
| 1   | `ast_int_lit`  | —                | —                 | —       | —                                  |
| 2   | `ast_ident`    | —                | —                 | —       | —                                  |
| 3   | `ast_binop`    | lhs              | rhs               | —       | op tag (`op_*`)                    |
| 4   | `ast_call`     | callee ident     | args Vec addr     | —       | args count                         |
| 5   | `ast_fn_def`   | name ident       | params Vec addr   | body    | `params_len * 2 + is_pub`          |
| 6   | `ast_let`      | name ident       | type ident        | init    | —                                  |
| 7   | `ast_use`      | path head ident  | path tail ident   | —       | —                                  |
| 8   | `ast_return`   | value expr       | —                 | —       | —                                  |
| 9   | `ast_param`    | name ident       | type ident        | —       | —                                  |
| 10  | `ast_block`    | stmts Vec addr   | —                 | —       | stmts count                        |
| 11  | `ast_program`  | items Vec addr   | —                 | —       | items count                        |
| 12  | `ast_paren`    | inner            | —                 | —       | —                                  |

Operator tags (subset of token kinds, mirrored into the AST so the
type-checker doesn't need to import lexer constants):

| `op_*`     | value | source token |
|------------|-------|--------------|
| `op_add`   | 1     | `+`          |
| `op_sub`   | 2     | `-`          |
| `op_mul`   | 3     | `*`          |
| `op_div`   | 4     | `/`          |
| `op_lt`    | 5     | `<`          |
| `op_gt`    | 6     | `>`          |
| `op_eq`    | 7     | `=`          |

## Expected tree

```
Program (items_len = 3)
├── Use
│   ├── head: Ident "std"
│   └── tail: Ident "vec"
│
├── FnDef "add"  (is_pub = 1, params_len = 2)
│   ├── name: Ident "add"
│   ├── params:
│   │   ├── Param  name=Ident "x"  ty=Ident "i64"
│   │   └── Param  name=Ident "y"  ty=Ident "i64"
│   └── body: Block (stmts_len = 2)
│       ├── Let
│       │   ├── name: Ident "z"
│       │   ├── ty:   Ident "i64"
│       │   └── init: BinOp op_add
│       │       ├── lhs: Ident "x"
│       │       └── rhs: Ident "y"
│       └── Return
│           └── value: Ident "z"
│
└── FnDef "compute"  (is_pub = 1, params_len = 3)
    ├── name: Ident "compute"
    ├── params:
    │   ├── Param  name=Ident "x"  ty=Ident "i64"
    │   ├── Param  name=Ident "y"  ty=Ident "i64"
    │   └── Param  name=Ident "z"  ty=Ident "i64"
    └── body: Block (stmts_len = 2)
        ├── Let
        │   ├── name: Ident "r"
        │   ├── ty:   Ident "i64"
        │   └── init: BinOp op_add
        │       ├── lhs: Ident "x"
        │       └── rhs: BinOp op_mul          # precedence: * binds tighter than +
        │           ├── lhs: Ident "y"
        │           └── rhs: Ident "z"
        └── Call
            ├── callee: Ident "add"
            └── args (count = 2):
                ├── Ident "r"
                └── Ident "x"
```

## Heap-record byte layout

Every AST node is a 56-byte record (7×i64):

| offset | field      | meaning                                          |
|--------|------------|--------------------------------------------------|
| +0     | `kind`     | `ast_*` tag (1–12)                               |
| +8     | `span_lo`  | source-byte start (inclusive)                    |
| +16    | `span_hi`  | source-byte end (exclusive)                      |
| +24    | `child0`   | i64 base address of subnode (or 0)               |
| +32    | `child1`   | i64 base address of subnode / Vec addr (or 0)    |
| +40    | `child2`   | i64 base address of subnode (or 0)               |
| +48    | `aux`      | op tag / count / packed flags (see table above)  |

Variable-length child lists (block stmts, fn params, call args,
program items) ride in **child0** as the i64 base address of a
`Vec<i64>` whose elements are AST node addresses, with the Vec's
logical length stored in **aux**.  This avoids inventing a generic
"node list" pointer type in Phase 6.2; Phase 6.3 may widen the node
shape once cross-fn reference patterns surface.

## Node counts

| AST kind     | count in fixture |
|--------------|------------------|
| `ast_program`| 1                |
| `ast_use`    | 1                |
| `ast_fn_def` | 2                |
| `ast_param`  | 5                |
| `ast_block`  | 2                |
| `ast_let`    | 2                |
| `ast_return` | 1                |
| `ast_binop`  | 3                |
| `ast_call`   | 1                |
| `ast_ident`  | 24               |
| **Total**    | **42**           |

(Ident nodes appear at every name reference, type reference, and
path component — see the tree above for the full enumeration.)

## How to verify (manual, Phase 6.2)

1. Parse-clean check:
   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/parser/main.mind --emit-ir
   ```
   Expected: clean exit, IR dump ending with `next_id = 90`.

2. Fixture parse-clean check:
   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/parser/fixture.mind --emit-ir
   ```
   Expected: clean exit, IR dump ending with `next_id = 3`.

3. Once mindc gains `--emit-shared` integration tests against an in-
   process heap-record reader, this fixture becomes a Cargo
   integration test asserting the tree above node-by-node.

## Known Phase 6.2 limitations (resolved by Phase 6.3+)

1. The `return` keyword is detected via byte-compare against
   `tk_ident`'s span because the Phase 6.1 lexer doesn't recognise
   `return` as a keyword.  Phase 6.3 lexer growth will promote it.

2. The `.` punctuation in `use std.vec;` is currently tokenised as
   `tk_slash` (see `examples/lexer/EXPECTED.md` row 3 note).  The
   parser's `parse_use` treats whatever lives between the two
   identifiers as the path separator; the `tk_dot` tag (when
   added) requires no parser change.

3. Function return types are parsed but not stored on the
   `ast_fn_def` node — the heap-record shape is fixed at 7×i64 and
   we already use all slots.  Phase 6.3 will widen the record (or
   indirect through a side-table) to carry the return-type ident.

4. No error recovery.  Unknown prefix tokens, missing `)`, and
   missing `;` are *soft-handled* (advance one token, default to an
   empty ident leaf).  Phase 6.3 introduces a Result-shaped node
   carrying diagnostics.

5. Single `=` is treated as a binop (precedence 1) rather than an
   assignment statement.  Phase 6.2's grammar matches the lexer's
   token-kind surface; statement-level assignment is a Phase 6.3 item
   alongside `mut` bindings.

6. No `if`/`else`/`while`/`match` — those keywords aren't in the
   Phase 6.1 lexer either.  Adding them is sequenced after
   Phase 6.3 type-check work because the type-checker informs the
   right ABI for branch targets.
