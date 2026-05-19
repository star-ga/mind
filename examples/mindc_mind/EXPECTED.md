# Phase 6.5 Stage 5 — Expected IR Text (APEX)

> Smoke gate for `examples/mindc_mind/main.mind` (combined pipeline) when
> driven against `examples/mindc_mind/fixture.mind` via the `mindc_compile`
> entry point.  The combined pure-MIND pipeline must emit **exactly** this
> byte stream — byte-for-byte identical to mindc-Rust's `--emit-ir` on the
> same fixture.  This identity is the APEX pass criterion for
> Phase 6.5 Stage 5.

## Fixture

[`examples/mindc_mind/fixture.mind`](./fixture.mind), 3 top-level items
(identical to `examples/emit_ir/fixture.mind`):

1. `use std.vec;`
2. `pub fn add(x: i64, y: i64) -> i64 { let z: i64 = x + y; return z; }`
3. `pub fn compute(x: i64, y: i64, z: i64) -> i64 { let r: i64 = x + y * z; add(r, x) }`

## Expected IR text

The combined pipeline (`lex` → `parse` → `typecheck` → `lower_program`)
must produce **exactly** the following 148 bytes:

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

This is byte-for-byte the output of:

```bash
cargo run --features "std-surface cross-module-imports" \
    --bin mindc -- examples/mindc_mind/fixture.mind --emit-ir
```

## How to verify (manual)

1. Build the combined shared library:

   ```bash
   cargo run --features "mlir-build std-surface cross-module-imports" \
       --bin mindc -- \
       examples/mindc_mind/main.mind \
       --emit-shared examples/mindc_mind/libmindc_mind.so
   ```

2. Run the smoke harness:

   ```bash
   python3 examples/mindc_mind/bootstrap_smoke.py
   ```

   Expected output: `RESULT: APEX PASS`

## Byte-level layout

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
|    148 | (EOF)                        |                                      |

## APEX criterion

**APEX PASS** = `libmindc_mind.so`'s `mindc_compile` on `fixture.mind`
produces MLIR text byte-identical to the 148-byte table above.

At that point, **the self-host thesis is proven**: the four pure-MIND
mindc sub-components, in their integrated form inside a single cdylib,
compile MIND programs to byte-identical output as the Rust reference
compiler.
