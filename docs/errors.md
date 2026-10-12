# MIND Core Error Model

MIND Core normalizes public-facing errors so tooling can parse them reliably.

## Error classes

- **Parse / type errors**: surfaced as structured diagnostics.
- **IR verification errors**: failures of the public IR invariants.
- **Autodiff errors**: failures and validation errors encountered during automatic
  differentiation of Core IR modules.
- **MLIR lowering errors**: failures while translating canonical IR into MLIR
  (behind the `mlir-lowering` feature).
- **Runtime and artifact-production errors**: execution failures, unavailable
  backends, and deterministic compiler-side materialization refusals.

## Diagnostic formats

`mindc` supports structured and human diagnostics:

```
mindc --diagnostic-format human   # default; multi-line with spans and notes
mindc --diagnostic-format short   # single line, grep-friendly
mindc --diagnostic-format json    # one diagnostic per line of JSON
```

JSON diagnostics are line-delimited with a stable shape:

```
{
  "phase": "parse",
  "code": "E1001",
  "severity": "error",
  "message": "unexpected token `)`; expected identifier",
  "span": {
    "file": "simple.mind",
    "line": 3,
    "column": 11,
    "length": 1
  },
  "notes": [
    "while parsing function `main`"
  ],
  "help": "check for an extra trailing comma or remove the unmatched `)`"
}
```

Human output uses consistent phase prefixes (`error[parse]`, `error[type-check]`,
etc.), includes caret highlights when spans are available, and respects
`--color` / `MINDC_COLOR` for ANSI styling.

All error variants propagate non-zero exit codes from the CLI.

## Error codes

Every diagnostic carries a stable code for the Core v1 pipeline phase:

- Parse: `E1xxx`
- Type-check: `E2xxx`
- IR verification: `E3xxx`
- Autodiff: `E4xxx`
- MLIR lowering: `E5xxx`
- Runtime and artifact production: `E6xxx`

Shape validation for Core v1 operators is also surfaced during type checking:

- Broadcast compatibility: `E2101`
- Rank or shape expectation mismatches (including invalid reductions): `E2102`
- Matmul inner-dimension mismatches: `E2103`

Slice call validation with `std-surface` reports:

- `E2032`: unsupported slice ABI shape or an argument without a proven
  compatible dynamic-array layout.
- `E2033`: a borrowed slice capability would be erased, escaped, or used for
  an operation it does not permit.
- `E2034`: a collection-owner field or indexed owner slot would be replaced by
  a status, scalar, or different declared collection type.

These are type-checking refusals before artifact emission. See
[slice calls](type-system.md#slice-calls) for the supported boundary.

`E2035` rejects duplicate struct declarations in one source module. Inline
`module { ... }` blocks are transparent declaration lists and share that
namespace. Separate project files may declare structs with the same name.

`E2036` rejects a proven non-integer or opaque-handle right-hand side assigned
to an `i8`, `u8`, `i16`, or `u16` array element. This prevents an opaque handle
from being silently truncated to the destination cell width.

In a manifest-captured project, a bare cross-module name is resolved against
the current module's exact imports. If multiple imported modules export the
same bare function or type name, the checker refuses it with `E2003` for a
call or `E2002` for a type; qualify the type or remove the conflicting import.
An unimported sibling cannot supply the signature or symbol for an imported
call. Legacy explicit multi-file checks retain their implicit project metadata
when no manifest-captured import scope is available.

Two E6xxx assignments distinguish host capability from compiler lowering:

- `E6002`: the requested backend is unavailable on the current host or build.
- `E6009`: a language-valid aggregate cannot be materialized into the runnable
  artifact representation, including a deterministic materialization limit.
  This includes struct-owned fixed-array record element-field receivers, which
  do not yet have a runnable representation, while a declared fixed-array
  return may be indexed for a record field on the supported Rust/MLIR path and
  fixed-array struct fields with `i64` or `f64` scalar cells do.
- `E6010`: a `Mind.toml [exports] c_abi` entry is invalid. This is a user
  configuration error and does not indicate backend capability.

Both refusals exit nonzero. `E6009` is emitted before a runnable artifact is
published and is not a type error.

## Runtime trap exit codes

A compiled MIND program that hits one of these runtime traps terminates at once
with a fixed exit status. Where both backends (`--backend native` and
`--backend mlir`) can hit the event, they use the same status:

| Exit | Event |
|------|-------|
| 70 | Native arena exhausted (native backend, explicit layout); the program writes `error: native arena exhausted` to stderr first. |
| 77 | Array index out of bounds (see [array semantics](ARRAY_SEMANTICS.md)). |
| 79 | Integer division trap (below). |

### Integer division trap

For `i64` operands, `x / 0`, `x % 0`, `INT_MIN / -1` and `INT_MIN % -1` are
a deterministic trap: the process performs no further action (in particular it
writes nothing more to any file descriptor) and exits with status **79**. For
`i8`/`i16`/`i32` operands a zero divisor traps the same way; the narrow
`MIN / -1` pair does not trap and gives `MIN` (remainder `0`), the result of
dividing the sign-extended values as `i64` and narrowing.

Unsigned operands divide as unsigned values and trap only on a zero divisor:

- MLIR backend: `u8`/`u16`/`u32`/`u64` locals, parameters, return values and
  `u8`/`u16`/`u32` const items and struct fields use unsigned division (`divui`/`remui`), and a
  zero divisor traps with status 79. **Open defect:** a `u64` `const` item and
  a `u64` struct field lose their unsigned type on this path and are divided
  as signed (`const BIG: u64 = 18446744073709551615;` then `BIG / 2` gives
  `-1 / 2 == 0`); `tests/u64_tag_survival_run.rs` tracks the struct-field case.
- Native backend: `u8`/`u16`/`u32` values are kept zero-extended to their
  declared width. A `let`, an assignment and a parameter of such a type
  truncate the value they receive, a function returning one truncates its
  return value, a struct field of such a type truncates an arithmetic value
  stored into it (in a struct literal or a field assignment), and an 8- or
  16-bit result of `+`, `-`, `*`, `/`, `%` or `<<` is truncated before it is
  used. The result is typed by the same narrow rules the MLIR backend uses: a
  call to a function returning an 8- or 16-bit type has that type, a shift has
  the type of its left operand and takes its count modulo that width
  (`a << 9` on `u8` shifts by 1), and `&`, `|` and `^` give a width-neutral
  result, so arithmetic over two such results, or over one and a literal, is
  not truncated. A `/` or `%` therefore sees the wrapped values (with
  `a: u8 = 200` and `b: u8 = 100`, both `let d: u8 = a + b; d / 3` and
  `(a + b) / 3` are `14`, and `((a << 1) + 200) % 11` is `0`) and gives the
  unsigned result; a zero divisor, including one produced by wrapping, traps
  with status 79. Signed `i8`/`i16`/`i32` values are sign-extended the same
  way. Division of 64-bit unsigned values is **not implemented natively**. The
  native backend refuses (builds nothing) when a `/` or `%` sits in a function
  that can hold a `u64`/`usize` value: the function's signature or body spells
  `u64` or `usize` (an annotation, a cast, a generic argument; comments and
  string literals do not count), or names a function whose return type spells
  it, a struct or enum whose declaration spells it (or one of its fields or
  variants), or a type alias, const or static of such a type. Build such a
  program with the MLIR backend.

An `i32`/`u32` result of `+`, `-`, `*` or `<<` is wrapped when it is bound (a
`let`, an assignment, an argument, a return value or a struct field). Before it
is bound the two backends do not yet agree on it: the MLIR backend evaluates it
at 32 bits when its operands are parameters and at 64 bits when they are local
variables (with `a: u32 = 0` and `b: u32 = 1` as locals,
`let q: u32 = (a - b) / 2;` gives `0`; as parameters it gives `2147483647`).
The same holds for an `i32` quotient: `MIN / -1` is `2147483648` at 64 bits and
`MIN` at 32 bits. So the native backend refuses (builds nothing) a `/` or `%`
when:

- an operand is an unbound `i32`/`u32` arithmetic result (`let d: u32 = a - b;
  d / 2` is `2147483647` on both backends), including arithmetic that reaches
  an `i32`/`u32` value through `&`, `|`, `^`, `<<` or `>>` (with `b: u32`,
  `((b >> 1) - 7) % 11` is refused like `(b - 7) % 11`);
- an operand contains an `i32`/`u32` `>>` whose count is not a decimal
  literal from 0 to 31 (MLIR takes that count modulo 32 when the operands are
  parameters and modulo 64 when they are local variables);
- an operand's unbound arithmetic involves a call to a function returning
  `i32`/`u32`, a struct field whose name any struct declares with a narrow
  integer type, a `u8` array element of a struct field, or a method returning a
  narrow integer type (MLIR narrows those leaves, the native lattice does not);
- one operand has a narrow integer type and the other is a wide integer, or a
  narrow integer of the other signedness (an integer literal never counts as
  a mismatch here), or a field name is declared with different types by two
  structs;
- an `i32`/`u32` operand meets an integer literal that is not a decimal of at
  most 9 digits, or the other operand has a different 32-bit type;
- it is a `/` with an `i32` operand when neither the divisor nor the dividend
  is such a small literal, unless the quotient is bound straight to an `i32`
  slot: the initial value of a `let` or the value of an assignment of an `i32`
  name, the operand of a `return` in a function returning `i32`, or the value
  stored to an `i32` struct field. With `a: i32` and `b: i32`, `let x: i32 = a / b;` builds (and
  `MIN / -1` gives `MIN`), while `if a / b == a`, `(a / b) as i64` and a bare
  tail expression `a / b` are refused.

Every other integer division truncates toward zero, and every other remainder
has the sign of the dividend (`-7 / 2 == -3`, `-7 % 2 == -1`, `7 % -2 == 1`).

Shift counts are taken modulo 64 for `i64`: `x << n` and `x >> n` use
`n & 63`, so a count of 64 shifts by 0, 65 by 1, and -1 by 63. `>>` is
arithmetic for signed operands. An 8- or 16-bit left operand takes its count
modulo its width on both backends (`n & 7`, `n & 15`). **Open defects (shifts
outside a division):** the native backend applies that narrow count rule only
when its lattice types the left operand (a variable, parameter, cast, call or
arithmetic result), so a shift of an 8- or 16-bit struct field or array
element still uses `n & 63` there (with `v: u8 = 200`, `let x: u8 = s.v << 9;`
is `0` natively and `144` on MLIR); and an `i32`/`u32` count of 32 or more
differs inside the MLIR backend itself (modulo 32 for parameters, modulo 64 for
local variables), where the native backend always uses modulo 64. A `/` or `%`
whose operand contains either shape is refused natively, as listed above.

The program does not need to do anything to get this behaviour. The MLIR
backend omits the check when the divisor is a constant other than `0` and `-1`.
It exits through libc `_exit`, or through `_Exit` when the program defines its
own function named `_exit`; a program that divides and defines functions named
both `_exit` and `_Exit` is refused by the MLIR backend. When the program itself
declares that libc symbol (`extern` `_exit` or `_Exit`), the MLIR backend calls
the program's declaration, and refuses a program that divides when that
declaration does not take exactly one integer parameter.

Scope: the trap is implemented by `mindc build` with the MLIR and native
backends. Three other evaluators do not implement it yet: the tree-walking
interpreter, the `--emit-mlir` text export (`src/eval/mlir_export.rs`, a bare
`arith.divsi`), and compile-time constant evaluation (`src/opt/comptime.rs`,
which refuses a zero divisor and folds `INT_MIN / -1` to `INT_MIN`). The
pure-MIND compiler's IR-text path (`mindc_compile` in
`examples/mindc_mind/main.mind`, which folds a bare top-level constant
expression) refuses a module whose fold would divide by zero or divide
`INT_MIN` by `-1`, rather than fold it.

## Native backend: `main` parameters

`mindc build --backend native` binds an entry `fn main(argc: i64, argv: i64)`
to the process arguments: `argc` is the argument count and `argv` is the address
of the C `char **` array, so `__mind_load_i64(argv + 8 * i)` is the pointer to
the NUL-terminated `argv[i]` and `argv + 8 * argc` holds the null terminator.
The MLIR build passes the same values. Any other parameter list on the entry
`main` is refused (`native.entry_main_has_params`), as is an artifact from a
compiler seed that does not bind the parameters.

See [`docs/versioning.md`](versioning.md) for how these classes fit the stability
contract.

> **MAP protocol responses use a separate code space.** The Mind AI Protocol
> server (`mind-ai`) returns short transport-layer codes in its
> `=<seq> err code=... msg="..."` responses (e.g. `E005` unknown command;
> `E101`–`E104` resource-budget rejections). These are protocol responses, not
> compiler diagnostics, and do not overlap the four-digit `E1xxx`–`E6xxx`
> pipeline codes above. The MAP resource budgets are documented in
> [`docs/security.md`](security.md#map-protocol-resource-budgets).

Passing the Core v1 conformance suite indicates that the current build produces
the expected diagnostics, IR, autodiff, and MLIR text for the relevant profile
(CPU baseline or the optional GPU profile). Experimental features outside those
profiles may still emit different errors or remain unsupported.
