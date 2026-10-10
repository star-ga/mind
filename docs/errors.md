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

### Native wave-three admission contracts

The Rust fence admits signed i64 division/remainder and shifts and unsigned
operations only with a proven unsigned operand type. On this base, source
admission admits the full-width `u64` form; narrow integer locals and parameters
remain `type.non_i64_integer` refusals until their normalization is proven.
`FnDef.value_types` is an **aggregate** table on this base, not a scalar signedness
map. Admission therefore derives scalar facts from signatures, constants,
conversion tags, arithmetic, control-flow merges, and call/return dataflow.

* `binop.div_operand_type` / `binop.mod_operand_type`: operands must both be
  proven i64 or both proven unsigned. Float, unknown, mixed signedness, and
  inconsistent call-site types refuse. These replace the unconditional
  `binop.div` / `binop.mod` refusals.
* `binop.shl_operand_type` / `binop.shr_operand_type`: both operands must be
  proven integers. These replace `binop.shl` / `binop.shr`. Counts use the low six
  bits (`count & 63`), including negative and oversized counts; signed right
  shift is arithmetic, unsigned right shift is logical.
* `call.conversion_operand_type`: `__mind_conv_i32`, `__mind_conv_i64`,
  `__mind_conv_u64`, and `__mind_conv_f64` require one proven numeric scalar.
  Unknown inputs refuse. Wrong arity remains `call.intrinsic_arity`; other
  conversion names remain `call.undefined_or_builtin` (or an unresolved closure
  edge). An i32 cast result is sign-extended in a native register; this does not
  admit i32 local/parameter signatures.
* `cast.pointer_operand_type`: casts to `*const u8` / `*mut u8` require a proven
  integer source. Other pointees remain outside the source profile. A pointer
  cast does not establish that the address is valid to dereference.
* `native.entry_main_has_params`: only the exact parameterized entry
  `main(argc: i64, argv: i64) -> i64` is admitted, alongside the existing
  parameterless entry. Different names, order, types, return type, arity, or a
  generic parameterized entry refuse.
* `call.alloc_size_not_proven_positive`, `alloc.in_loop`, and
  `alloc.total_over_budget` retain their default-profile meanings. Explicit
  `--native-profile bounded` transfers size/loop/budget enforcement to the bounded
  allocator (deterministic exit 70 under the 2 GiB address-space contract).
  Dynamic sizes no longer fail the static byte sum in that mode. Intrinsic
  arity, operand/result type checks, recursive allocation, ambiguous calls, and
  unrelated I/O rules remain enforced.
* `closure.extern_signature_mismatch`: an extern declaration must use the C
  calling convention, be non-variadic, and exactly match the parameter and
  return types of an available function listed in `std.posix`. Parameter names
  and the informational `safe`/`unsafe` tag are not signature components.
  A same-named function in another std module is insufficient. Linked POSIX
  bodies must pass all the ordinary source, IR, and whole-program fences.

`std/posix.mind` is absent at base `387cf368`. The admission contract is exercised
with a small parsed/lowered fixture module in `tests/native_wave3_admission.rs`.
The parallel seed batch must add POSIX to its manifest and source image, provide
matching wrappers (including positioned I/O), implement the conversion and
unsigned paths, bind process arguments, and provide the bounded allocator.
Admission-only sentinel tests do not certify those emitter implementations.
The parallel division batch owns its documented deterministic trap exit code
for zero divisors and signed MIN/-1; this Rust-only change does not select or
change that exit code or update the committed seed executables.
