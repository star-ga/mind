# RFC 0001: Native BitNet Support — `tri` and `q16_16` Types

- **Start Date**: 2026-04-28
- **RFC PR**: TBD
- **Status**: Draft
- **Target Release**: v0.2.6
- **Phase**: 1 of 3 (Phase 2: packed storage + SIMD; Phase 3: release & docs)

## Summary

Add two native `DType` primitives to mindc:

1. **`tri`** — 2-bit packed ternary `{-1, 0, +1}` for BitNet-style 1.58-bit weights
2. **`q16_16`** — 32-bit signed fixed-point with 16 integer bits and 16 fractional bits, for deterministic activation arithmetic

Plus parser, type-checker, IR, and codegen support sufficient to compile a forward pass written entirely with these types and observe bit-identical outputs across CPU, ARM, and CUDA targets.

This is the substrate the `bitnet-mind-governance` reference project currently emulates with `i32` and bit-packing tricks at user level. Making them native enables the compiler to emit deterministic integer code paths and downstream consumers to declare BitNet-class invariants statically.

## Motivation

### Today

The `bitnet-mind-governance` project demonstrates a 700M BitNet b1.58 forward pass written in MIND, but every ternary value is currently encoded as an `i32` carrying `-1 | 0 | 1`, every fixed-point activation is encoded as an `i32` with manual scaling, and every BitLinear is a hand-rolled multiply-and-shift loop. None of this requires compiler features — but it leaves three real gaps:

1. **No type-level guarantee** that a tensor declared as ternary actually contains only `{-1, 0, +1}`. A bug introducing `2` would compile and silently corrupt downstream evidence chains.
2. **No codegen optimization** — the compiler cannot lower `ternary × q16_16 → q16_16` to integer add/sub/zero, because it doesn't know the operands are constrained. It emits the same code as `i32 × i32` with full multipliers.
3. **No invariant interop** — downstream `mic@1` consumers (governance compilers, AOT deployment tools) cannot encode "this tensor is provably ternary" in the IR they consume.

### After this RFC

```mind
type Weight = tri[1024, 4096]    // compile-time-checked: only {-1, 0, +1}
type Act    = q16_16[1024]       // 32-bit fixed-point, deterministic arithmetic

fn bitlinear(w: Weight, x: Act) -> Act {
    matmul(w, x)                  // lowers to integer add/sub/zero, no FP
}
```

`mindc` proves the multiply has no floating-point operations and no reduction-order dependence. The output is bit-identical across substrates by construction.

## Guide-level explanation

### `tri` — ternary type

A `tri` value is one of `-1`, `0`, or `+1`. Literals: `-1tri`, `0tri`, `+1tri`. The compiler stores `tri` values as 2-bit packed (4 values per byte) when they appear in a tensor, but exposes element access as `i8`-equivalent for arithmetic.

```mind
let x: tri = -1tri        // OK
let y: tri = 0tri         // OK
let z: tri = 2tri         // ERROR: out-of-range ternary literal
```

Conversions:
- `tri → i8`, `tri → i16`, `tri → i32`, `tri → i64`: implicit, sign-extended
- `tri → q16_16`: implicit, lifts `-1 → -65536`, `0 → 0`, `+1 → 65536`
- `i32 → tri`: explicit only, runtime-checked at conversion site (panics if out of range)
- `f32 → tri`: explicit, requires explicit rounding mode

### `q16_16` — fixed-point type

A `q16_16` value is a 32-bit signed integer interpreted as a number with 16 integer bits and 16 fractional bits. Range: `[-32768.0, 32767.99998]` with resolution `1/65536`.

Literals: `1.0q16_16`, `-3.14q16_16`. The compiler converts the literal to its integer representation at compile time.

Operations:
- `q16_16 + q16_16 → q16_16` — integer add, no overflow check by default
- `q16_16 - q16_16 → q16_16` — integer sub
- `q16_16 * q16_16 → q16_16` — multiply with right-shift by 16 (truncated, not rounded)
- `q16_16 * tri → q16_16` — sign-mask + zero-mask, no multiplier needed
- `q16_16 / q16_16 → q16_16` — divide with left-shift by 16 (panics on divide-by-zero)

All operations are integer-only at the hardware level. No FP unit involvement. Reduction-order-independent for `+` (associative on integers without overflow).

### BitLinear primitive

```mind
fn bitlinear<I: usize, O: usize>(
    w: tri[O, I],
    x: q16_16[I]
) -> q16_16[O]
```

The compiler lowers this to a kernel that, for each output element, accumulates `+x[k]` when `w[o,k] == +1`, `-x[k]` when `w[o,k] == -1`, and skips when `w[o,k] == 0`. Zero multipliers used.

## Reference-level explanation

### File-by-file changes

#### `src/types/mod.rs`

Add to surface `DType`:

```rust
pub enum DType {
    I32,
    F32,
    BF16,
    F16,
    Tri,        // NEW: 2-bit ternary {-1, 0, +1}
    Q16_16,     // NEW: 32-bit fixed-point Q16.16
}
```

Update `parse_name`, `as_str`, and `FromStr` to handle `"tri"` and `"q16_16"`.

#### `src/ir/compact/v2/types.rs`

Add to IR `DType`:

```rust
pub enum DType {
    F16, F32, F64, BF16,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
    Tri,        // NEW: 2-bit packed ternary
    Q16_16,     // NEW: 32-bit Q16.16 fixed-point
}
```

Update `parse`, `Display`, and `byte_size`:
- `Tri` reports `byte_size = None` for 2-bit-packed (special-case in tensor allocator), exposes `bits_per_element = 2`
- `Q16_16` reports `byte_size = 4`

#### `src/ast/mod.rs`

Add to `TypeAnn`:
```rust
pub enum TypeAnn {
    // ... existing variants ...
    Tri,
    Q16_16,
}
```

Add literal kinds:
- `LiteralKind::Tri(i8)` — value constrained to `-1 | 0 | 1`
- `LiteralKind::Q16_16(i32)` — raw 32-bit fixed-point representation

#### `src/parser/mod.rs`

- Type keyword recognition: `tri`, `q16_16`
- Literal suffix recognition: `Ntri`, `N.NNNq16_16`
- Validation: `Ntri` requires `N ∈ {-1, 0, 1}`, parse error otherwise
- `q16_16` literal: parse decimal, multiply by 65536, store as `i32`

#### `src/types/check.rs` (new module or extension)

Type rules:
```
tri + tri      : NOT ALLOWED (no closure under add)
tri × q16_16   : q16_16  (the BitLinear-friendly op)
q16_16 op q16_16 : q16_16 for op ∈ {+, -, *, /}
q16_16 → f32   : explicit only
f32 → q16_16   : explicit, requires rounding mode
```

#### `src/eval/lower.rs` and `src/eval/mlir_*`

Lowering rules:
- `tri` → MLIR `i8` with assertion that values are in `{-1, 0, +1}` at boundaries
- `q16_16` → MLIR `i32`
- `tri × q16_16`: lower to `select(w == 0, 0, select(w == 1, x, -x))` — sign-mask, no multiplier
- `q16_16 * q16_16`: lower to `(i64(a) * i64(b)) >> 16` truncated to `i32`

Codegen invariant: no FP instruction emitted in any path that touches `tri` or `q16_16` operands.

#### `src/ir/compact/v2/encode.rs` and `decode.rs`

Update `mic@1` IR encoding to include the two new dtypes in the wire format. **Wire-format-stable concern**: this is an additive change. Existing `mic@1` consumers see new dtype tags they don't recognize and should refuse to decode (per IR stability contract). Consumers that opt in should accept the new tags transparently.

#### `tests/bitnet/`

New test directory:
- `test_tri_parser.rs` — round-trip parsing of `tri` literals and types
- `test_q16_16_parser.rs` — `q16_16` parsing including edge cases (overflow, fractional precision)
- `test_tri_q16_16_arithmetic.rs` — `+`, `-`, `*`, `/` on q16_16; `tri × q16_16`
- `test_bitlinear_codegen.rs` — verify no FP instruction emitted for BitLinear lowering
- `test_cross_arch_determinism.rs` — compile same program for x86, ARM, CUDA targets, verify output hashes match

### Build system

`Cargo.toml` versions bump to `0.2.6`. `CHANGELOG.md` adds an entry under `[Unreleased]` initially, promoted to `[0.2.6]` at release time.

## Drawbacks

1. **Two new dtypes mean two new code paths in every backend.** MLIR CPU + MLIR GPU + (future) CUDA + WASM all need lowering rules. Estimate ~600 LoC across backends.
2. **`tri` packing complicates tensor allocator.** Other dtypes are byte-aligned; 2-bit-packed needs special handling for views, slicing, and stride math. Estimate +200 LoC in tensor.
3. **Overflow semantics for `q16_16 × q16_16`** are a real footgun. Without saturation or overflow checks, multiplying two values near the type range overflows silently. Default behavior must be documented; consider `#[overflow_check]` attribute as future work.
4. **Wire-format change.** `mic@1` is supposed to be stable as of v0.2.5. Adding dtypes is additive but downstream tooling (governance compilers, evidence-chain readers) may reject `mic@1` files using the new tags until updated. Migration story needed.

## Rationale and alternatives

### Alternative A: User-level types

Keep `tri` and `q16_16` as user-defined types in MIND code (current approach in `bitnet-mind-governance`). No compiler change needed.

**Why rejected**: Loses type-level checking, codegen optimization, and IR-level invariant declaration. The whole point of native types is to let the compiler prove things.

### Alternative B: Macro-style annotations

`@ternary i32` or `@fixed_point<16,16> i32` instead of new dtypes.

**Why rejected**: Annotations don't propagate cleanly through type inference and IR. Native dtypes are simpler and more orthogonal.

### Alternative C: Library-level intrinsics only

Add only `bitlinear()` as a builtin function operating on `i8` ternary tensors and `i32` Q16.16 tensors, without elevating either to dtype.

**Why rejected**: The user has to remember "this `i32` is actually Q16.16, do not mix with normal `i32`". Type system can't help. We've seen this fail in practice — Q16.16 / I32 mix-ups are common bugs.

## Prior art

- **Stable Diffusion `int8` quantization in CUDA**: similar shape, but no compile-time guarantee, runtime calibration only
- **BitNet b1.58** (Microsoft): the actual model architecture. Original implementation is in PyTorch with `int8` storage and runtime scale tracking
- **Q15/Q31 fixed-point in DSP toolchains** (TI C6x, ARM CMSIS-DSP): mature pattern, native types in compiler intrinsics. We're one step further (more fractional bits, native first-class type)
- **APL/J `int1`** (single-bit): demonstrates packed sub-byte storage in established languages

## Unresolved questions

1. **Saturation semantics for `q16_16 + q16_16` overflow** — default to wrap (current `i32` behavior) or saturate? Saturate is safer for ML; wrap is faster.
2. **Rounding mode for `f32 → q16_16` conversion** — round-to-nearest vs truncate vs banker's rounding?
3. **`tri × tri`?** — closure under multiplication is fine (`{-1, 0, 1} × {-1, 0, 1} ⊆ {-1, 0, 1}`), but does any real workload need it? Could add or defer.
4. **CUDA backend strategy** — emit hand-written PTX for ternary multiply, or rely on MLIR `arith.select` lowering? Performance question, not correctness.
5. **`bitlinear()` as builtin vs library** — if library, where? `mind/std/nn/`? If builtin, codegen path question.

## Future possibilities

- **Phase 2** (separate RFC, target v0.2.7): packed-storage tensor allocator for `tri`, vectorized SIMD codegen for `q16_16` ops, NEON/AVX2/AVX-512 paths
- **Phase 3** (separate RFC, target v0.2.8): `bitnet-attention` builtin (multi-head attention on ternary weights with q16_16 KV cache)
- **`q8_8`, `q24_8`, `q4_12`** sibling types — same fixed-point family, different precision tradeoffs
- **`tri2` (5-state ternary {-2, -1, 0, +1, +2})** for BitNet variants

## Implementation plan

### Step 1: Surface types (1-2 days)
- Add `Tri` and `Q16_16` to `src/types/mod.rs::DType` and `src/ir/compact/v2/types.rs::DType`
- Update parser keyword tables
- Add literal kinds to AST
- Round-trip parser tests

### Step 2: Type checker (2-3 days)
- Type rules table (above) implemented in `src/types/check.rs`
- Coercion rules for explicit casts
- Test mixed-type expressions

### Step 3: IR encoding (1 day)
- Add dtype tags to `mic@1` encoder/decoder
- Wire-format tests
- Migration note in `docs/ir-stability.md`

### Step 4: Lowering (3-4 days)
- MLIR lowering rules
- BitLinear lowering
- Cross-target codegen tests
- Verify no FP instructions in BitNet path

### Step 5: Cross-arch determinism harness (1 day)
- Hook `bitnet-mind-governance/tests/test_determinism.py` into mindc CI
- Verify SHA-256 evidence chain matches across x86 / ARM / CUDA

### Step 6: Release (1 day)
- Update `CHANGELOG.md`
- Bump `Cargo.toml` to `0.2.6`
- Tag `v0.2.6`, push, GitHub Release with notes
- Update `bitnet-mind-governance` docs to point to `mindc >= 0.2.6` for "native BitNet support"

**Total estimate: 9-12 days of focused engineering** (Phase 1 only — Phase 2 + 3 are separate RFCs).

## Status checklist

- [ ] RFC reviewed and accepted
- [ ] Step 1: Surface types
- [ ] Step 2: Type checker
- [ ] Step 3: IR encoding
- [ ] Step 4: Lowering
- [ ] Step 5: Cross-arch determinism harness
- [ ] Step 6: Release v0.2.6
