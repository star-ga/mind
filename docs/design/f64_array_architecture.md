<!-- Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0. -->

# Normative array-semantics + f64-aggregate architecture record

Status: **normative** (frozen contract for the f64-completion / array-convergence
work). Establishes ONE language-level array semantics with MULTIPLE backend
lowering strategies and observable cross-substrate parity. Grounded in the source
as of `3dd11ce8` (Slice A: const `[f64;N]` typed dense lowering).

This record answers the required design questions and fixes the tokens that gate
all subsequent implementation. Backends may differ in *storage*; they may NOT
differ in *observable semantics*.

## Canonical model

Two — and only two — language-level array categories:

| Category | Surface | T | N |
|---|---|---|---|
| **Fixed** | `[T; N]` (conceptually `Array<T,N>`) | part of the type | part of the type (compile-time) |
| **Dynamic** | `array<T>` | part of the type | runtime (`len` in the descriptor) |

`[f64; N]` is **f64-typed**. A raw `i64`/`u64` carrier is permitted only at the
storage/transport boundary; it is NEVER the canonical *semantic* representation.
`[f64;N]` is never reinterpreted as `[i64;N]` merely because both occupy 8-byte
cells.

## The 20 answers (grounded in source)

1. **Semantic type of `[T;N]`** — a fixed aggregate of `N` elements of scalar type
   `T`. Today there is **no first-class `Array<T,N>` IR node**; a `[T;N]` value is
   materialised as `Instr::ConstArray` (i64 const, `src/ir/mod.rs:462`) or
   `Instr::ConstDenseTensor { dtype, shape, data:Vec<u64> }` (typed dense const,
   Slice A) or, for the growable/mutable surface, the dtype-blind `std.vec` path.
   *Target:* fixed arrays carry `T`+`N` in compiler semantics uniformly.
2. **Where `T` is stored** — const: `ConstDenseTensor.dtype` (canonical IR **and**
   mic@3). MLIR backend also carries it per-value in
   `values: BTreeMap<ValueId, ValueKind::Tensor{dtype,shape}>` (`src/mlir/lowering.rs:355`)
   — derived, lowering-local. Dynamic `array<T>`: **nowhere** (the `vec_*` calls are
   opaque i64 — the gap the mandate forbids for f64).
3. **Where `N` is stored** — const: `ConstDenseTensor.shape` / `ConstArray.values.len()`.
   Dynamic: runtime `len` in the `[addr|len|cap]` descriptor. Fixed mutable: TBD
   (step H) — from the declared type.
4. **`array<T>` semantically** — a dynamic, growable sequence of `T`.
5. **Where dynamic `T` is stored** — **nowhere canonical today**; `vec_new/vec_push/
   vec_get/vec_set` are all-`i64` (`ARRAY_VEC_SENTINEL="vec"`, `src/eval/lower.rs:2945`).
   *Target:* `TIER_A_DTYPE_IN_CANONICAL_IR` — a dtype-aware canonical dynamic-array IR
   (typed ops, or a canonical `ValueId → ArrayType{elem_dtype,size_kind}` table).
6. **`array<T>` runtime representation** — `[addr | len | cap]` heap record (`std.vec`).
7. **Fixed-array runtime representation** — const: MLIR `arith.constant dense<…> :
   tensor<Nxf64>` (Tier B) / native-ELF contiguous 8-byte IEEE-754 cells
   (`1d8b2d4b`). Mutable: TBD (step H) — contiguous 8-byte cells.
8. **Fixed-array call ABI** — pass by pointer/reference; `N` is known from the
   callee signature / monomorphised instance. Do **not** add a fat pointer solely to
   carry `N`. Avoid whole-array copies unless value semantics require it.
9. **Dynamic-array call ABI** — the `[addr|len|cap]` descriptor (as an opaque handle
   today); `len` survives the call boundary via the descriptor.
10. **Ownership / aliasing** — audit pending (`ARRAY_ALIASING`); historical self-host
    i64 arrays behave as reference/pointer-copy in some paths. The canonical f64
    implementation MUST match the actual language contract; mutation-through-alias and
    through-function-argument are tested accordingly.
11. **Runtime OOB** — currently a **fork**: MLIR clamps the index to `[0, len-1]`
    (`arith.maxsi`/`minsi`, `src/mlir/lowering.rs:4534-4556`; empty array rejected at
    4529); pure-MIND native-ELF traps `_exit(77)` (`e0a1dcc1`). **DECIDED:
    `ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP`** — the clamp is removed and replaced
    with a deterministic guard/trap (step C). `_exit(77)` is the current native ABI, not
    the eternal language semantics.
12. **Compile-time-provable OOB** — may fail at compile time.
13. **Canonical IR** — `ConstArray`, `ConstDenseTensor`, `ArrayLoad`, `IndexAssign`.
    *Target adds:* dtype-aware dynamic-array ops **or** a canonical typed-value table
    (whichever is the cleaner general invariant).
14. **mic@3** — `ConstDenseTensor` serialises dtype byte + fixed 8-byte-LE bits
    (`src/ir/compact/v3/emit.rs`); `ConstArray` serialises i64 values. Dynamic `vec_*`
    are generic `Call`s (no dtype on the wire). If a typed dynamic array needs canonical
    replay, dtype must be encoded (versioned) — decided by semantic necessity, not fear
    of a byte change.
15. **Derived lowering metadata** — MLIR `ValueKind` table (per-`ValueId` dtype/shape),
    `const_dense_defs` (name → dense blob; **not** serialised — the node carries the bits).
16. **Backend-only** — MLIR `tensor<Nxf64>` typing + `tensor.extract`, native-ELF cell
    layout + `movsd`/native f64 load-store instruction selection.
17. **Exact float bits** — canonical: `ConstDenseTensor.data: Vec<u64>` (raw IEEE-754).
    Render currently reconstructs a host float + decimal text (`render_dense_elem`/
    `format_number`) — **insufficient for NaN payloads**; step I replaces it with one
    bit-exact codec (raw bits → MLIR hex float literal, no host numeric conversion).
18. **Const arrays** — `ConstDenseTensor` (f64/f32) / `ConstArray` (i64). `TIER_B=KEEP`.
19. **Mutable arrays** — step H: typed contiguous 8-byte cells; loads/stores via typed
    IR that lowers to native f64 memory ops (raw-bit bitcast only where a generic carrier
    is genuinely required — never `fptosi`/`sitofp`).
20. **Self-host parity** — the pure-MIND native-ELF emitter already implements f64
    arrays (`1d8b2d4b`: literals/indexed-reads/variable-index/arith/`.len()`/exact cells;
    `96562008`: typed `[T;N]` + mutable writes + fail-closed; `71ecc358`: float tensor,
    strict L→R reduction). Parity is required on overlapping surfaces; gaps reported as
    `DEFERRED_WITH_ISSUE`, never fake `PASS`.

## Storage contract (f64 element)

`WIDTH=64 bits` · `STRIDE=8 bytes` · `FORMAT=IEEE-754 binary64` · `STORAGE_IDENTITY=EXACT`.
Transport through a generic 64-bit carrier uses `bitcast` only. Forbidden for storage:
`fptosi`/`sitofp`/numeric conversion/decimal reparse. Preserved through store→load:
`+0.0`/`-0.0` (`0x8000000000000000`) / normals / max-finite / min-normal / subnormals /
`±inf` / qNaN payloads. Reference values (machine-checked): `2.5 = 4612811918334230528
= 0x4004000000000000`; `-0.0 = 0x8000000000000000`.

## Performance contract

Typed IR stays typed; the backend emits native `load f64`/`store f64`/`movsd` where
equivalent — **zero avoidable representation conversions**, no boxing for statically
known fixed arrays, no runtime dtype lookup for statically typed arrays, no redundant
bitcasts, no unnecessary heap/copies. Fixed `[T;N]` with a compile-time-proven-in-range
index elides the bounds check. `CRITERION_MAX_REGRESSION=10%`, target ~0%.

## SOTA generalization test

Adding `f32`/`f16`/`bf16`/`complex64` later must NOT require another forest of
`if dtype == …` patches across parser/lowering/MLIR/runtime. The aggregate layer is
parameterised by (element dtype, layout), carried once — not "i64 arrays + f64 exceptions".

## Architecture tokens

```
FIXED_ARRAY_SEMANTICS=DOCUMENTED
DYNAMIC_ARRAY_SEMANTICS=DOCUMENTED
ELEMENT_DTYPE_LOCATION=DOCUMENTED   (const: ConstDenseTensor.dtype canonical; dynamic: MISSING -> target typed IR/ValueType)
FIXED_N_LOCATION=DOCUMENTED         (const: shape; target: type)
ARRAY_STORAGE_ABI=DOCUMENTED        (8-byte exact IEEE-754 cells)
ARRAY_FUNCTION_ABI=DOCUMENTED       (fixed: ref + static N; dynamic: [addr|len|cap] handle)
ARRAY_ALIASING=DOCUMENTED           (audit pending; historical pointer-copy)
ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP
IR_MIC3_BOUNDARY=DOCUMENTED
BACKEND_STORAGE_BOUNDARY=DOCUMENTED
```

## Implementation order (from Slice A)

A ✅ const `[f64;N]` Tier B (`3dd11ce8`). B this record. C OOB clamp→deterministic trap
(+ differential low/high/empty controls). D dtype-aware canonical dynamic-array IR (never
the dtype-blind vec path). E early backend differential-execution gate. F mutable/local
f64 on the typed IR. G dynamic indexed load/store. H function ABI / cross-module / return.
I exact float-literal codec (before the NaN/Inf battery; unify `dense_elem_bits` /
`render_dense_elem`). J full bit-exact battery. K self-host + evaluator parity. L tensor
consistency. M regression/keystone/mic3/criterion. N RH flattened-matrix canary.
