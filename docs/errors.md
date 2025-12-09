# MIND Core Error Model

MIND Core normalizes public-facing errors so tooling can parse them reliably.

## Error classes

- **Parse / type errors**: surfaced as structured diagnostics.
- **IR verification errors**: failures of the public IR invariants.
- **Autodiff errors**: failures and validation errors encountered during automatic
  differentiation of Core IR modules.
- **MLIR lowering errors**: failures while translating canonical IR into MLIR
  (behind the `mlir-lowering` feature).

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

See [`docs/versioning.md`](versioning.md) for how these classes fit the stability
contract.

Passing the Core v1 conformance suite indicates that the current build produces
the expected diagnostics, IR, autodiff, and MLIR text for the relevant profile
(CPU baseline or the optional GPU profile). Experimental features outside those
profiles may still emit different errors or remain unsupported.
