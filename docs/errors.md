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

The E6xxx assignments distinguish host capability from compiler lowering:

- `E6002`: the requested backend is unavailable on the current host or build.
- `E6009`: a language-valid aggregate cannot be materialized into the runnable
  artifact representation, including a deterministic materialization limit.
  This includes struct-owned fixed-array record element-field receivers, which
  do not yet have a runnable representation, while a declared fixed-array
  return may be indexed for a record field on the supported Rust/MLIR path and
  fixed-array struct fields with `i64` or `f64` scalar cells do.
- `E6010`: a `Mind.toml [exports] c_abi` entry is invalid. This is a user
  configuration error and does not indicate backend capability.
- `E6011`: the native backend's compiler process failed
  (`error[backend-native][E6011]: ...`). `mindc build --backend native` streams
  the source image into a child compiler. The code is used when that child
  could not be started (the OS error is named, for example `Permission denied`
  or `Cannot allocate memory`), when a pipe to it failed (for example
  `Broken pipe` because the child died before reading its input), when it was
  terminated by a signal, or when it could not be reaped. A loader failure
  under an address-space cap (`RLIMIT_AS`) usually appears as termination by
  `SIGSEGV` after a broken pipe. The bridge always reaps the child first. The
  report then names the outcome on a `compiler process:` line: the exit
  status, or the signal number and name. A child that keeps running after a
  stream failure is killed with `SIGKILL` after 5 seconds and reported as
  killed. The report also shows the child's stderr: the first 16 KiB and the
  last 48 KiB, with the number of bytes left out in between. The exit status
  is the child's own nonzero status, `128 + signal` for a signal, or 2. No
  artifact is written. A child that read the whole image and exited nonzero has
  rejected the program, not failed as a process. That keeps its own uncoded
  message (`pure-MIND compiler rejected the program (exit N)`) and exit status.

Every refusal above exits nonzero. `E6009` is emitted before a runnable artifact
is published and is not a type error.

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
