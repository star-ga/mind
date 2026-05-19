# Phase 6.3 — Expected Type-Check Report

> Smoke gate for `examples/typecheck/main.mind` when run against
> `examples/typecheck/fixture.mind`.  The MIND type-checker must emit
> *exactly* this report (byte-for-byte) when consuming the AST that
> the Phase 6.2 parser produces on the fixture.  Phase 6.3b promotes
> this from a documented fixture to a Cargo integration test that
> links the type-checker's compiled `.so` and diffs the report bytes
> against the table below.

## Fixture

[`examples/typecheck/fixture.mind`](./fixture.mind), 4 top-level items:

1. `use std.vec;`                                              — no report (no name binding at this layer)
2. `pub fn add(x: i64, y: i64) -> i64 { ... }`
3. `pub fn compute(x: i64, y: i64, z: i64) -> i64 { ... }`
4. `pub fn cmp(x: i64, y: i64) -> i64 { ... }`

## Type tags (i64 constants)

Frozen for Phase 6.3 (`main.mind` `ty_*` fns):

| tag | name         | source spelling | notes                                                |
|-----|--------------|-----------------|------------------------------------------------------|
| 0   | `ty_unknown` | `?`             | identifier not in env; ast_call (deferred 6.3b)      |
| 1   | `ty_i64`     | `i64`           |                                                      |
| 2   | `ty_f64`     | `f64`           |                                                      |
| 3   | `ty_bool`    | `bool`          | result of `<`, `>`, `=` binops                       |
| 4   | `ty_vec`     | `Vec`           | std.vec.Vec heap-record                              |
| 5   | `ty_string`  | `String`        | std.string.String heap-record                        |
| 6   | `ty_unit`    | `()`            | reserved for stmt-form returns; unused in fixture    |

## Expected report (byte-for-byte)

```
fn add : (i64, i64) -> i64
let z : i64
fn compute : (i64, i64, i64) -> i64
let r : i64
fn cmp : (i64, i64) -> i64
let b : bool
```

Six lines total, every line LF-terminated (byte 10), no trailing
whitespace.  The opening `use std.vec;` produces nothing — the
Phase 6.3 type-checker has no use-statement handling beyond
"skip cleanly".

### Per-fn breakdown

#### `add` — declared `(i64, i64) -> i64`

Header line:

```
fn add : (i64, i64) -> i64
```

Body statements:

| stmt        | env effect          | report line       |
|-------------|---------------------|-------------------|
| `let z: i64 = x + y;` | bind `z → i64` | `let z : i64`     |
| `return z;` | (return-type check) | (silent: matches) |

The `return z` is silent because `type_of_expr(z, env, buf)` returns
`ty_i64` (looked up via env_lookup on the byte-range "z" hashed by
DJB2), which matches the fn's declared return type
`fn_default_ret() = ty_i64`.  Phase 6.3 hardcodes the expected
return as `ty_i64` for every fn because the Phase 6.2 parser does
not stash the declared return-type ident on the `ast_fn_def` node
(see `examples/parser/EXPECTED.md` "Known Phase 6.2 limitations"
item 3).  Phase 6.3b widens the heap-record to carry the return
ident, at which point this hardcoded default becomes a real lookup.

#### `compute` — declared `(i64, i64, i64) -> i64`

Header line:

```
fn compute : (i64, i64, i64) -> i64
```

Body statements:

| stmt        | env effect          | report line       |
|-------------|---------------------|-------------------|
| `let r: i64 = x + y * z;` | bind `r → i64` | `let r : i64`     |
| `add(r, x)` (tail expr)   | none           | (silent)          |

The tail expression is an `ast_call`, which currently types to
`ty_unknown` (Phase 6.3b deferred work — full fn-signature lookup
across the program env).  Because Phase 6.3's stmt walker does not
emit on tail expressions, no line is added; the silence is the
contract.

#### `cmp` — declared `(i64, i64) -> i64`

Header line:

```
fn cmp : (i64, i64) -> i64
```

Body statements:

| stmt        | env effect          | report line       |
|-------------|---------------------|-------------------|
| `let b: bool = x < y;` | bind `b → bool` | `let b : bool`    |
| `return x;` | (return-type check) | (silent: matches) |

The let binding's declared type (`bool`) is what gets written into
the env; the init expression's type is `ty_bool` too (because
`x < y` is an `ast_binop` with `op_lt` → `ty_bool`), so there's no
mismatch to flag.  The `return x;` is silent for the same reason as
`add`: `x` is an `i64` param, fn returns `i64`.

## Line-by-line byte map

Useful when the integration test diffs report bytes:

| line | bytes (decimal)                                                       | length |
|------|-----------------------------------------------------------------------|-------:|
| 1    | `102 110 32 97 100 32 58 32 40 105 54 52 44 32 105 54 52 41 32 45 62 32 105 54 52 10` | 26 |
| 2    | `108 101 116 32 122 32 58 32 105 54 52 10`                            | 12 |
| 3    | `102 110 32 99 111 109 112 117 116 101 32 58 32 40 105 54 52 44 32 105 54 52 44 32 105 54 52 41 32 45 62 32 105 54 52 10` | 35 |
| 4    | `108 101 116 32 114 32 58 32 105 54 52 10`                            | 12 |
| 5    | `102 110 32 99 109 112 32 58 32 40 105 54 52 44 32 105 54 52 41 32 45 62 32 105 54 52 10` | 26 |
| 6    | `108 101 116 32 98 32 58 32 98 111 111 108 10`                        | 13 |
| total | —                                                                    | 124 |

## What's deferred to Phase 6.3b

1. **Full call-signature matching.**  `ast_call` types to
   `ty_unknown` today.  Phase 6.3b walks the program env to look up
   the callee fn's declared signature and types the call against
   the return type.  Requires the parser to first stash the fn's
   declared return-type ident on the `ast_fn_def` node — see the
   Phase 6.2 parser's "Deferred to Phase 6.3" item 3 (widening the
   7×i64 heap-record).
2. **Numeric promotion / mismatch diagnostics.**  Today `tc_let`
   binds the declared type into env without verifying that the
   init expression's inferred type matches it.  Phase 6.3b grows a
   per-let `actual vs declared` check and emits ERROR lines on
   mismatch.
3. **Closures / cross-fn name resolution.**  Phase 6.3's `tc_fn_def`
   uses a fresh env for every fn body; it does *not* visit the
   outer (program-level) env.  Phase 6.3b lifts fn names into the
   top-level env so that `add(r, x)` in `compute` resolves to the
   declared sig.
4. **`return` as a real keyword.**  Phase 6.1 lexer still tokenises
   `return` as `tk_ident` (byte-compare in parser); Phase 6.3
   inherits that — `tc_return` is invoked via the parser's
   identifier-span detection.  Phase 6.3b lexer promotion (Gap 3
   in `docs/rfcs/0005-phase-6-2-mindc-gaps.md`) replaces it.
5. **Struct / enum / user-defined types.**  Type-name resolution
   today is a hardcoded 5-name switch (`i64` / `f64` / `bool` /
   `Vec` / `String`).  Phase 6.3b grows the type-name env from the
   AST's `struct` / `enum` declarations (which the Phase 6.2 parser
   does not yet produce — they're Phase 6.4 grammar growth).

## How to verify (manual, Phase 6.3)

1. Parse-clean check (type-checker itself):
   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/typecheck/main.mind --emit-ir
   ```
   Expected: clean exit, IR dump ending with `next_id = 90`.

2. Fixture parse-clean check:
   ```bash
   cargo run --features "std-surface cross-module-imports" \
       --bin mindc -- examples/typecheck/fixture.mind --emit-ir
   ```
   Expected: clean exit, IR dump ending with `next_id = 4`.

3. Once mindc gains `--emit-shared` integration tests against an
   in-process heap-record reader, this fixture becomes a Cargo
   integration test that:
   1. Tokenises `fixture.mind` via the Phase 6.1 lexer .so
   2. Parses the token stream via the Phase 6.2 parser .so
   3. Feeds the resulting `ast_program` addr + the original byte
      buffer through the Phase 6.3 type-checker's `typecheck` fn
   4. Reads the returned `String` byte-by-byte and diffs it
      against the 124-byte expected output above.
