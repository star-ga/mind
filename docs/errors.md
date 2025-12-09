# MIND Core Error Model

MIND Core normalizes public-facing errors so tooling can parse them reliably.

## Error classes

- **Parse / type errors**: surfaced as structured diagnostics.
- **IR verification errors**: failures of the public IR invariants.
- **Autodiff errors**: failures and validation errors encountered during automatic
  differentiation of Core IR modules.
- **MLIR lowering errors**: failures while translating canonical IR into MLIR
  (behind the `mlir-lowering` feature).

## CLI formatting

CLI-facing errors are prefixed to identify their source:

```
error[parse]: …
error[type-check]: …
error[ir-verify]: …
error[autodiff]: …
error[mlir]: …
```

All error variants propagate non-zero exit codes from the CLI.

See [`docs/versioning.md`](versioning.md) for how these classes fit the stability
contract.

Passing the Core v1 conformance suite indicates that the current build produces
the expected diagnostics, IR, autodiff, and MLIR text for the relevant profile
(CPU baseline or the optional GPU profile). Experimental features outside those
profiles may still emit different errors or remain unsupported.
