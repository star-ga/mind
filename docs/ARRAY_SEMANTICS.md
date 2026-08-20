<!-- Copyright 2025 STARGA Inc. Licensed under the Apache License, Version 2.0. -->

# MIND array semantics — normative architecture record

Status: **normative**. This record fixes ONE language-level array semantics for
MIND with MULTIPLE backend lowering strategies and *tested* (never assumed)
observable cross-substrate parity. It is the authoritative contract that gates
the f64-completion / array-convergence work.

Ground truth: source as of `3dd11ce8` (Slice A — const `[f64;N]` typed dense
lowering) plus the empirical three-engine probe run recorded below. Every
`EVIDENCE=` line is a source citation or a real-binary observation, not a claim
from memory. Where a surface has never been executed, this record says so and
marks the question `OPEN`, rather than guessing.

**Method note.** Behavior was captured with real exit codes — compile via
`mindc build --emit=binary`/`--backend=native` (capturing the *compiler's* exit),
then run the artifact directly (capturing the *program's* exit). No
`… | tail; $?` pipe ever stands in for a producer exit. Evaluator behavior is
from `mindc test` (tree evaluator). Probe harness: `scratchpad/arr_probe.py`.

---

## 0. Four layers — kept strictly separate

The central discipline of this record: an "array" is described at four distinct
layers, and a defect at one layer must not be excused by health at another.

| Layer | What it is | Where it lives today |
|---|---|---|
| **LANGUAGE TYPE** | what the programmer writes and the checker reasons about: `[T; N]` (fixed) and `array<T>` (dynamic) | AST `TypeAnn::Array { element, length }`; type checker |
| **CANONICAL IR TYPE** | the compiler-invariant, backend-independent, *serialized* type of an aggregate ValueId | **FRAGMENTED** — see §2. Const dense carries dtype+shape; everything else carries nothing canonical |
| **PHYSICAL ABI** | how an aggregate is passed across a call / stored in a slot: pointer vs value, descriptor shape, register vs stack | fixed: unspecified/unsupported at the call boundary (§Q8); dynamic: `[addr\|len\|cap]` opaque handle |
| **BACKEND STORAGE** | the concrete bytes a given backend emits: `tensor<Nxf64>` const, native 8-byte cells, heap vec record | MLIR `tensor<…>`; native-ELF contiguous cells; `std.vec` heap |

The **CANONICAL IR TYPE** layer is the load-bearing one for the wedge, and it is
the one that is broken. `ConstArray` and `ConstDenseTensor` are **materialization
instructions**, not the semantic definition of `[T;N]`; the MLIR `ValueKind`
table is **derived, lowering-local** metadata, not a canonical type. Neither is a
canonical IR type for aggregates. This distinction is the spine of the whole
record — see §2 and the ROOT_ARCHITECTURAL_DEFECT.

---

## 1. Canonical model — two array categories, one storage contract

| Category | Surface | `T` | `N` |
|---|---|---|---|
| **Fixed** | `[T; N]` (conceptually `Array<T,N>`) | part of the type | part of the type (compile-time) |
| **Dynamic** | `array<T>` | part of the type | runtime (`len` in the descriptor) |

`[f64; N]` is **f64-typed**. A raw `i64`/`u64` carrier is permitted only at the
storage/transport boundary; it is NEVER the canonical *semantic* representation.
`[f64;N]` is never reinterpreted as `[i64;N]` merely because both occupy 8-byte
cells. Element storage (all element dtypes): `WIDTH=64 bits` (for 64-bit dtypes)
· `STRIDE=8 bytes` · `FORMAT=IEEE-754 binary64` for f64 · `STORAGE_IDENTITY=EXACT`
(store→load preserves `+0.0`/`-0.0`=`0x8000000000000000`/normals/subnormals/
`±inf`/qNaN payloads bit-for-bit). Transport through a generic 64-bit carrier uses
`bitcast` only — never `fptosi`/`sitofp`/decimal reparse. Machine-checked
reference: `2.5 = 4612811918334230528 = 0x4004000000000000`;
`-0.0 = 0x8000000000000000`.

---

## 2. Canonical IR array model — FRAGMENTED (the core finding)

There is **no single first-class aggregate type node** in canonical IR. An
aggregate ValueId's type is represented one of four incompatible ways depending
on how it was produced:

1. **`Instr::ConstArray { values: Vec<i64> }`** (`src/ir/mod.rs:462`) — i64-only
   const. Floats coerced to `0`. Carries `N` (via `values.len()`); carries no
   element dtype (implicitly i64).
2. **`Instr::ConstDenseTensor { dtype: DType, shape: Vec<ShapeDim>, data: Vec<u64> }`**
   (Slice A) — typed dense const. Carries `T` **and** `N` **canonically and on
   the mic@3 wire** (`src/ir/compact/v3/emit.rs`, dtype byte + 8-byte-LE bits).
   This is the ONLY path where an aggregate's element dtype survives
   serialization. **`TIER_B_CONST_DESIGN=KEEP`.**
3. **`std.vec` opaque handle** — the growable/mutable/dynamic surface. `vec_new`/
   `vec_push`/`vec_get`/`vec_set` are all-i64 generic `Instr::Call`s tagged
   `ARRAY_VEC_SENTINEL="vec"` (`src/eval/lower.rs:1460,1490`). The element dtype
   is **absent** from canonical IR — it exists only as an AST dispatch input
   (`is_array_surface_type`, `lower.rs:2982`), consumed at lowering time and
   discarded.
4. **MLIR `values: BTreeMap<ValueId, ValueKind::Tensor{dtype,shape}>`**
   (`src/mlir/lowering.rs`) — a per-ValueId dtype/shape map. This is **derived,
   lowering-local, and never serialized**; it is reconstructed inside one backend
   pass and does not exist in canonical IR or mic@3. It is NOT sufficient
   canonical typing: it cannot survive a wire round-trip, a different backend, or
   a fresh lowering context (which is exactly why `fn_env` inheriting a stale
   module SSA id silently untyped a const-dense load until Slice A's filter fix).

`CURRENT_CANONICAL_IR_ARRAY_MODEL=FRAGMENTED`.
`CANONICAL_DYNAMIC_ELEMENT_DTYPE=LOST`.
`MLIR_VALUE_DTYPE_TRACKING=LOWERING_LOCAL_ONLY`.

**Dtype-loss boundary (traced):** for `array<T>` / any mutable-fixed literal, `T`
is present in the AST `TypeAnn`, is read by `is_array_surface_type` at the
lowering dispatch (`lower.rs:1427`), and is then dropped when
`lower_array_surface_lit` (`lower.rs:3018`) emits opaque `vec_*` Calls. No
parallel canonical type table preserves it (grep for an IR-level
`array_elem`/`elem_dtype`/`ArrayType` table returns nothing).
→ **`DYNAMIC_ARRAY_DTYPE_FIRST_LOST_LAYER=AST_TO_IR_LOWERING`.**

---

## 3. The 20 questions — per engine, decision, gap, evidence

For each: `CURRENT_RUST_MLIR` / `CURRENT_SELFHOST` (pure-MIND native-ELF) /
`CURRENT_EVALUATOR` (tree eval) = observed reality; `CANONICAL_DECISION` = the
normative target; `GAP` = distance; `EVIDENCE` = citation/observation.

### Q1 — Semantic type of `[T; N]`
- CURRENT_RUST_MLIR = materialized as `ConstArray`/`ConstDenseTensor`/`std.vec`; **no first-class `Array<T,N>` IR node**.
- CURRENT_SELFHOST = native cells with a local dtype tag on some paths (`1d8b2d4b`/`96562008`); no canonical node.
- CURRENT_EVALUATOR = an in-memory element list; no canonical node.
- CANONICAL_DECISION = a fixed aggregate of `N` elements of scalar `T`, with `T` and `N` carried as a canonical compiler invariant (not re-derived per instruction/backend).
- GAP = the semantic type exists only implicitly and inconsistently (§2).
- EVIDENCE = `src/ir/mod.rs:462`; `src/eval/lower.rs:2982,3018`.

### Q2 — Where element type `T` is stored
- CURRENT_RUST_MLIR = const dense: `ConstDenseTensor.dtype` (canonical + wire). MLIR also mirrors it lowering-locally in `ValueKind::Tensor`. Dynamic/mutable: **nowhere canonical**.
- CURRENT_SELFHOST = native emitter local tag on supported fixed paths; not canonical.
- CURRENT_EVALUATOR = runtime value tag only.
- CANONICAL_DECISION = `T` lives once in the canonical aggregate type for ALL categories, and survives serialization/replay.
- GAP = present only for const dense; missing for dynamic/mutable.
- EVIDENCE = `src/ir/compact/v3/emit.rs` (dtype byte); §2.4 (ValueKind local).

### Q3 — Where length `N` is stored
- CURRENT_RUST_MLIR = const: `ConstDenseTensor.shape` / `ConstArray.values.len()`. Dynamic: runtime `len` in `[addr|len|cap]`.
- CURRENT_SELFHOST = const cells: static count; dynamic length **not preserved across calls** (`e0a1dcc1` limitation).
- CURRENT_EVALUATOR = element-list length.
- CANONICAL_DECISION = fixed `N` from the type (compile-time); dynamic `N` from the descriptor `len`.
- GAP = fixed mutable `N` has no canonical home yet (§Q19); native dynamic `N` not call-stable.
- EVIDENCE = `e0a1dcc1` note; §2.

### Q4 — `array<T>` semantically
- CURRENT_RUST_MLIR / CURRENT_SELFHOST / CURRENT_EVALUATOR = a dynamic growable sequence, physically `std.vec`.
- CANONICAL_DECISION = dynamic growable sequence of `T`, `T` explicit in canonical IR.
- GAP = `T` is not in canonical IR (§2.3).
- EVIDENCE = `ARRAY_VEC_SENTINEL`, `lower.rs:1460,1490`.

### Q5 — Where dynamic `T` is stored
- CURRENT_RUST_MLIR = **nowhere canonical**; `vec_*` are all-i64.
- CURRENT_SELFHOST = n/a (native bridge fail-closes on typed dynamic f64).
- CURRENT_EVALUATOR = runtime value tag only.
- CANONICAL_DECISION = `TIER_A_DTYPE_IN_CANONICAL_IR` — a dtype-aware canonical dynamic-array representation (typed ops OR a canonical `ValueId → ArrayType{elem_dtype,size_kind}` table). Physical i64 slots are fine; a *semantic* i64 array is not.
- GAP = total for the dynamic path.
- EVIDENCE = §2.3; probe `dyn_array_f64` COMPILE_FAIL (parse), `dyn_array_i64` works.

### Q6 — `array<T>` runtime representation
- ALL = `[addr | len | cap]` heap record (`std.vec`).
- CANONICAL_DECISION = keep the descriptor; add the canonical element dtype above it.
- GAP = dtype only.
- EVIDENCE = std.vec; `lower.rs`.

### Q7 — Fixed-array runtime representation
- CURRENT_RUST_MLIR = const: `arith.constant dense<…> : tensor<Nxf64>` (Tier B). Mutable/local literal: routed to the **untyped std.vec heap** (float non-const → not extractable → surface lit).
- CURRENT_SELFHOST = contiguous 8-byte IEEE-754 cells (`1d8b2d4b`, `96562008`).
- CURRENT_EVALUATOR = element list.
- CANONICAL_DECISION = contiguous typed 8-byte cells; const may stay the dense-tensor constant.
- GAP = mutable/local fixed `[f64;N]` has no typed backing (falls to vec).
- EVIDENCE = probe "missing type information … array load base" (mutable/local `[f64;N]` falls to the untyped std.vec heap); §2.

### Q8 — Fixed-array call ABI — **OPEN (traced: does not lower today)**
- CURRENT_RUST_MLIR = **UNSUPPORTED**. Passing `[i64;3]` or `[f64;3]` to a fn both COMPILE_FAIL "missing type information for value". The per-fn param typing (`fn_param_kinds`, `lower.rs:459-478`) is a scalar/narrow-int ABI (`"f64"`/`"f32"`/`"i64"`); an array param never receives `ValueKind::Tensor`, so the callee's `a[i]` ArrayLoad base is untyped (error #239).
- CURRENT_SELFHOST = **UNSUPPORTED**. `fixed_arg_i64`/`fixed_arg_f64` COMPILE_FAIL "unsupported construct".
- CURRENT_EVALUATOR = OPEN (not probed; likely supported via value list).
- CANONICAL_DECISION = pass by pointer/reference; `N` known from the callee signature / monomorphized instance. Do **not** add a fat pointer solely to carry `N`. **Do NOT assume `N` survives the boundary today — it does not, because the argument does not lower at all.**
- GAP = the entire fixed-array call boundary is unimplemented on both compiled backends. `FIXED_ARRAY_CALL_BOUNDARY_LENGTH=N/A (arg never lowers)`; `BOUNDS=N/A`; `CAUSE=array params get no aggregate ValueKind (MLIR) / unsupported construct (native)`.
- EVIDENCE = probe `fixed_arg_i64`+`fixed_arg_f64` both COMPILE_FAIL on both backends; `lower.rs:459-478`, error site `src/mlir/lowering.rs:239`.

### Q9 — Dynamic-array call ABI
- CURRENT_RUST_MLIR = the `[addr|len|cap]` descriptor passed as an opaque i64 handle; `len` survives via the descriptor.
- CURRENT_SELFHOST = native dynamic length not call-stable (`e0a1dcc1`).
- CURRENT_EVALUATOR = value passed directly.
- CANONICAL_DECISION = descriptor handle carrying `len`; element dtype from the canonical type.
- GAP = dtype absent; native `len` not call-stable.
- EVIDENCE = §Q6; `e0a1dcc1`.

### Q10 — Ownership / aliasing — **empirical**
- CURRENT_RUST_MLIR = **VALUE-COPY** for fixed arrays: `let mut b = a; b[0]=99; return a[0]` returns `1` (unchanged). Assignment copies.
- CURRENT_SELFHOST = `let b = a` on a fixed array is **UNSUPPORTED** ("unsupported construct") — array-to-array bind does not lower.
- CURRENT_EVALUATOR = OPEN (not probed).
- CANONICAL_DECISION = ONE aliasing contract, tested (not assumed) identically on every backend that supports the construct; the f64 implementation MUST match whatever the language contract is fixed to. Copy-semantics is the current Rust/MLIR reality; it must be made a deliberate, uniform decision (and native must then implement it, not refuse it).
- GAP = backends disagree (copy vs unsupported); the contract is undecided/untested.
- EVIDENCE = probe `alias_fixed_i64` RUST_MLIR exit=1 (copy); native COMPILE_FAIL.

### Q11 — Runtime OOB — **THREE-way fork (confirmed)**
- CURRENT_RUST_MLIR = **CLAMP** to `[0, len-1]` (`arith.maxsi`/`minsi`, `src/mlir/lowering.rs:4534-4556`; empty rejected at `4529`). Probe: `a[-1]`→`a[0]`=10; `a[5]`→`a[len-1]`=30.
- CURRENT_SELFHOST = **TRAP** `_exit(77)`. Probe: `a[-1]`→77; `a[5]`→77; `a[2]` in-bounds→30 (`e0a1dcc1`).
- CURRENT_EVALUATOR = **HARD ERROR**: "unsupported: array index -1 out of bounds (len 3)".
- CANONICAL_DECISION = **`ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP`**. Remove the MLIR clamp; every runtime OOB is a deterministic trap. Compile-provable OOB may fail at compile time. `_exit(77)` is the current native *ABI* for the observable trap in differential tests — NOT the eternal language semantics (a future version may surface a typed panic/Result).
- GAP = three different observable behaviors for the same program; the wedge cannot tolerate this.
- EVIDENCE = probes `oob_neg`/`oob_high`/`oob_inbounds`; clamp `lowering.rs:4534-4556`; native `e0a1dcc1`; evaluator error text.

### Q12 — Compile-time-provable OOB
- CURRENT = not systematically diagnosed at check for constant indices.
- CANONICAL_DECISION = may fail at compile time (allowed, not required for v1).
- GAP = optional; low priority.
- EVIDENCE = —.

### Q13 — Canonical IR node inventory
- CURRENT = `ConstArray`, `ConstDenseTensor`, `ArrayLoad`, `IndexAssign`, plus `std.vec` `Call`s.
- CANONICAL_DECISION = add a dtype-aware dynamic-array representation (typed ops OR a canonical typed-value table) — whichever yields the cleaner single invariant. Specify the invariant BEFORE choosing the encoding (§4).
- GAP = no canonical aggregate type; no typed dynamic ops.
- EVIDENCE = §2.

### Q14 — mic@3 wire
- CURRENT = `ConstDenseTensor` serializes dtype byte + fixed 8-byte-LE bits; `ConstArray` serializes i64 values; `vec_*` are generic `Call`s (no dtype on the wire).
- CANONICAL_DECISION = a typed dynamic array that needs canonical replay MUST encode its dtype (versioned, additive). Decided by semantic necessity, not fear of a byte change; any wire change re-proves the emit→parse→emit fixed point + keystone + oracle-parity + a version bump.
- GAP = dynamic dtype unrepresented on the wire.
- EVIDENCE = `src/ir/compact/v3/emit.rs`; §2.3.

### Q15 — Derived (non-canonical) lowering metadata
- CURRENT = MLIR `ValueKind` per-ValueId table; `const_dense_defs` (name→dense blob, **not** serialized — the node carries the bits).
- CANONICAL_DECISION = derived metadata stays derived; it must never be the sole home of element type (that is the §2 defect).
- GAP = today it is the sole home for many paths.
- EVIDENCE = `src/mlir/lowering.rs` ValueKind; `const_dense_defs` (Slice A).

### Q16 — Backend-only concerns
- CURRENT = MLIR `tensor<Nxf64>` typing + `tensor.extract`; native cell layout + `movsd`/native f64 load-store selection.
- CANONICAL_DECISION = legitimately backend-local; must be driven by the canonical aggregate type, not re-inferred.
- GAP = re-inference today (ValueKind reconstruction).
- EVIDENCE = `tensor.extract` `lowering.rs:4557-4564`.

### Q17 — Exact float bits
- CURRENT = canonical `ConstDenseTensor.data: Vec<u64>` (raw IEEE-754). Render reconstructs a host float + decimal text (`render_dense_elem`/`format_number`) — **insufficient for NaN payloads** and `f32::from_bits(bits) as f64` is numeric widening, not bit transport.
- CANONICAL_DECISION = one shared bit-exact codec: raw IEEE bits → MLIR hex float literal, no host numeric conversion (step I).
- GAP = decimal render path on the const surface.
- EVIDENCE = `render_dense_elem`/`format_number`; #305 (sub-EPSILON destroyed).

### Q18 — Const arrays
- CURRENT = `ConstDenseTensor` (f64/f32) / `ConstArray` (i64).
- CANONICAL_DECISION = **`TIER_B_CONST_DESIGN=KEEP`**, frozen at `3dd11ce8`.
- GAP = none for const.
- EVIDENCE = `tests/const_f64_array_run.rs` (dlopen exec, exact bits incl. `-0.0` + dynamic index).

### Q19 — Mutable arrays
- CURRENT_RUST_MLIR = `let mut a:[T;N]; a[i]=v` on a plain fixed `[T;N]` / const-literal `tensor<T[N]>` receiver now **WORKS as a top-level (straight-line) statement**: it lowers to `Instr::ArrayStore` (value-semantic — `tensor.insert` yields a FRESH aggregate incarnation) and the fn-body statement dispatch rebinds the receiver name to that fresh id, so a later read observes the write (verified by `tests/array_store_run.rs`: `a[0]=9;return a[0]`→9, two-write→90, `tensor` literal→9, untouched sibling→2). Inside a LOOP / BRANCH body (or in expression position) it still **fails CLOSED** (loud diagnostic) — the fresh incarnation's rebind is not yet threaded through the F2 region-exit machinery, so a store there would be silently lost (verified: a loop-body store read back the PRE-loop value); the compiler rejects it rather than miscompile. Working mutable receivers (`array<T>` → `vec_set`, `bytes[N]` → `__mind_store_i8`) and all reads are unaffected. The **silent store-drop is eliminated in every case** (straight-line = correct store; loop/branch = fail-closed).
- CURRENT_SELFHOST = typed `[T;N]` + mutable writes on supported fixed paths (`96562008`), fail-closed elsewhere. main.mind self-uses zero surface `a[i]=v` (raw `__mind_store_i64` ABI), so `ArrayStore` is inert during self-compile — keystone byte-identical.
- CURRENT_EVALUATOR = OPEN.
- CANONICAL_DECISION = typed contiguous 8-byte cells; loads/stores via typed IR lowering to native memory ops. Un-drop IndexAssign (fail-closed until a real store path exists) — **DONE, and the real value-semantic store path (`Instr::ArrayStore` → `tensor.insert` + name rebind) is now LANDED for straight-line writes**. The remaining follow-on is **F2 region-exit threading** so `a[i]=v` inside a loop / branch also rebinds correctly across the region boundary (mint the aggregate name into `While.exit_ids` / `region_exit_rebindings`, like a scalar loop-carried var), plus the `value_types` aggregate-type-invariant population for the fresh `dst` (see Step D).
- GAP = straight-line `a[i]=v` works; loop / branch / expression-position `a[i]=v` is fail-closed pending the F2 region-exit rebind. No silent-wrong in any case.
- EVIDENCE = `Instr::ArrayStore` in `src/ir/mod.rs`; MLIR `tensor.insert` lowering in `src/mlir/lowering.rs`; emit + name-rebind at the `IndexAssign` statement dispatch in `src/eval/lower.rs`; fail-closed panic for non-statement positions. Tests: `tests/array_store_run.rs` (straight-line RUNS correct; loop/while fail-closed), `tests/mic3_array_store_roundtrip.rs` (0x2B wire fixed point).
- **`SILENT_STORE_DROP` = IMPOSSIBLE_OR_FAIL_CLOSED** is the required end state for i64/f32/f64 alike — **now enforced in every case** (straight-line stores correctly; loop/branch fails closed).

### Q20 — Self-host / evaluator parity
- CURRENT_SELFHOST = native-ELF implements f64 arrays on several fixed surfaces (`1d8b2d4b` literals/indexed-reads/variable-index/arith/`.len()`/exact cells; `96562008` typed `[T;N]`+mutable+fail-closed; `71ecc358` float tensor strict L→R reduction) but diverges on OOB (trap vs clamp), aliasing (unsupported), and fixed-array args (unsupported).
- CURRENT_EVALUATOR = diverges on OOB (hard error).
- CANONICAL_DECISION = observable parity is REQUIRED on every overlapping supported surface; report the exact boundary — a gap is `DEFERRED_WITH_ISSUE`, never a fake `PASS`. Never infer parity from one backend.
- GAP = OOB + aliasing + call-ABI diverge across all three engines (§Q8/Q10/Q11).
- EVIDENCE = the probe matrix (§Audit F).

---

## 4. Invariant, specified BEFORE any encoding

The canonical-IR repair MUST satisfy this invariant, and the invariant is fixed
here before the encoding is chosen:

> **AGGREGATE-TYPE INVARIANT.** Every canonical ValueId that denotes an aggregate
> carries an `ArrayType { elem_dtype: DType, size: Fixed(N) | Dynamic }` that is
> (a) set at AST→IR lowering, (b) never reconstructed by a backend, (c) preserved
> byte-for-byte across mic@3 emit→parse→emit, and (d) identical across all
> backends. No aggregate ValueId may reach a backend without it.

Encoding is chosen to satisfy the invariant, not vice versa. Two admissible
encodings (pick the one that yields the single cleanest invariant, then commit):

- **(E1) canonical typed-value table** — `IRModule.value_types: BTreeMap<ValueId,
  ArrayType>`, serialized in mic@3 (versioned, additive). Element dtype carried
  ONCE per ValueId, not per op.
- **(E2) typed aggregate ops** — dtype-parameterized array/vec ops
  (`array_new<T>`, `array_get<T>`, …) replacing the sentinel `vec_*` Calls.

The arch test both must pass: adding `f32`/`f16`/`bf16`/`complex64` later must NOT
require another forest of `if dtype == …` patches across parser/lowering/MLIR/
runtime. If it would, the encoding is insufficient. `TIER_A_DTYPE_IN_CANONICAL_IR
=PASS` is a precondition for ANY typed dynamic-array lowering.

---

## 5. Audits A–H

- **A. Fixed vs dynamic.** Two categories, cleanly separated at the LANGUAGE
  layer (`[T;N]` vs `array<T>`); catastrophically merged at the CANONICAL IR
  layer (both collapse toward untyped carriers except const dense). §1, §2.
- **B. Empirical aliasing.** Fixed arrays are VALUE-COPY on Rust/MLIR
  (`alias_fixed_i64` → 1); native refuses the array-to-array bind. Contract is
  undecided and untested cross-backend. §Q10.
- **C. Fixed-array call-boundary bounds.** The boundary does not lower at all —
  both dtypes COMPILE_FAIL on both compiled backends. `N` does not survive because
  the argument never lowers; do not model this as "N is lost", model it as
  "unimplemented". §Q8.
- **D. Dynamic dtype-loss point.** `DYNAMIC_ARRAY_DTYPE_FIRST_LOST_LAYER=
  AST_TO_IR_LOWERING`: `T` is consumed by `is_array_surface_type` at the dispatch
  and never written to canonical IR; no parallel type table preserves it. §2.
- **E. Const/mutable relationship.** Const `[f64;N]` is healthy and frozen (Tier
  B, exact wire bits). Mutable/local `[f64;N]` is unrelated in the code: it falls
  to the untyped std.vec heap and (for the generic receiver) silently drops
  stores. The two share no typed backing today. §Q7, §Q18, §Q19.
- **F. OOB fork.** THREE-way confirmed: CLAMP (Rust/MLIR) / TRAP-77 (native) /
  HARD-ERROR (evaluator). Decision = DETERMINISTIC_BOUNDS_TRAP; kill the clamp.
  §Q11. Probe matrix:

  | case (len 3) | RUST_MLIR | NATIVE | EVALUATOR |
  |---|---|---|---|
  | `a[-1]` | 10 (→a[0], clamp) | exit 77 (trap) | error "index -1 out of bounds" |
  | `a[5]`  | 30 (→a[len-1], clamp) | exit 77 (trap) | — |
  | `a[2]`  | 30 | 30 | — |

- **G. Bit storage.** Const dense stores raw IEEE-754 `Vec<u64>` exactly; the
  render step reconstructs decimal (NaN-payload-lossy) — replace with a bit-exact
  codec. Mutable/dynamic have no bit-exact typed storage yet. §Q17.
- **H. mic@3.** Const dense carries dtype+bits on the wire (no change needed).
  Dynamic dtype is absent from the wire; adding it is additive+versioned when a
  typed dynamic array needs canonical replay. §Q14.

---

## 6. Performance & SOTA-generalization contracts

Typed IR stays typed; the backend emits native `load f64`/`store f64`/`movsd`
where equivalent — zero avoidable representation conversions, no boxing for
statically known fixed arrays, no runtime dtype lookup for statically typed
arrays, no redundant bitcasts. A compile-time-proven-in-range index elides the
bounds check. `CRITERION_MAX_REGRESSION=10%`, target ~0%. Generalization: the
aggregate layer is parameterized by (element dtype, layout), carried once — not
"i64 arrays + f64 exceptions".

---

## 7. Tokens

```
CURRENT_CANONICAL_IR_ARRAY_MODEL=FRAGMENTED
CANONICAL_DYNAMIC_ELEMENT_DTYPE=LOST
DYNAMIC_ARRAY_DTYPE_FIRST_LOST_LAYER=AST_TO_IR_LOWERING
MLIR_VALUE_DTYPE_TRACKING=LOWERING_LOCAL_ONLY
FIXED_ARRAY_CALL_ABI=UNIMPLEMENTED_BOTH_BACKENDS
FIXED_ARRAY_ALIASING=VALUE_COPY_RUST_MLIR / UNSUPPORTED_NATIVE (contract undecided)
ARRAY_OOB_SEMANTIC_FORK=CONFIRMED_THREE_WAY
ARRAY_OOB_CONTRACT=DETERMINISTIC_BOUNDS_TRAP
SILENT_STORE_DROP_TARGET=IMPOSSIBLE_OR_FAIL_CLOSED
TIER_B_CONST_DESIGN=KEEP
TIER_A_DTYPE_IN_CANONICAL_IR=REQUIRED_PRECONDITION
AGGREGATE_TYPE_INVARIANT=SPECIFIED (E1|E2 pending)

STEP_B_ARCHITECTURE_RECORD=PASS
ARCHITECTURE_GENERALIZES_BEYOND_F64=PASS
FOUR_LAYERS_SEPARATED=PASS
EVERY_SURFACE_HAS_EVIDENCE=PASS
NO_IMPLEMENTATION_CHANGE_IN_STEP_B=PASS
```

---

## 8. ROOT_ARCHITECTURAL_DEFECT

**The aggregate element type is represented inconsistently across paths, and the
canonical IR has no single home for it.** Type information survives *too far* in
some paths (const dense carries dtype+shape all the way to mic@3; the MLIR
ValueKind table re-derives dtype inside one backend) and disappears *too early* in
others (dynamic `array<T>` and mutable/local `[T;N]` collapse to an opaque,
untyped carrier — `std.vec` i64 handles — at AST→IR lowering, before canonical IR
even exists). Because there is no canonical aggregate type, every downstream
concern is forced to re-invent or fake one: backends re-infer it (ValueKind), the
call boundary can't type an array param (so fixed-array args don't lower at all),
OOB forks three ways because no shared contract is anchored to a shared type, and
mutable stores silently drop because the generic receiver has no typed store path.
The fragmentation IS the defect; f64 is merely the surface that exposed it.

## 9. TARGET_REPAIR

**Make the aggregate type a canonical compiler invariant, while allowing distinct
physical lowerings per backend.** Introduce `ArrayType{elem_dtype, size:
Fixed(N)|Dynamic}` on every aggregate ValueId (§4 invariant), set once at AST→IR
lowering, preserved through mic@3 (additive/versioned), identical across all
backends, and never reconstructed by a backend. Keep Tier B (const dense) as the
const materialization of that type. Route dynamic `array<T>` and mutable `[T;N]`
through the SAME canonical type with their own physical lowerings (heap descriptor
/ typed cells) — never through an untyped i64 carrier at the semantic layer.
Anchor ONE OOB contract (deterministic trap), ONE aliasing contract, and ONE
call ABI to that shared type, and gate every backend against a merge-blocking
differential-execution test so a fork like the OOB three-way divergence can never
recur. This satisfies the SOTA generalization test: a new dtype adds a `DType`
variant and its physical lowering, not a forest of special cases.

---

## 10. Implementation order (unchanged from Slice A; B complete)

A ✅ const `[f64;N]` Tier B (`3dd11ce8`). **B ✅ this record.** C OOB
clamp→deterministic trap (+ differential low/high/empty controls; remove the MLIR
clamp `lowering.rs:4534-4556`). D dtype-aware canonical dynamic-array IR (the §4
invariant; never the dtype-blind vec path). E early backend differential-execution
gate (merge-blocking). F mutable/local f64 on the typed IR (un-drop IndexAssign).
G dynamic indexed load/store. H function ABI / cross-module / return (implement
the fixed-array call boundary that does not lower today). I exact float-literal
codec (before the NaN/Inf battery). J full bit-exact battery. K self-host +
evaluator parity. L tensor consistency. M regression/keystone/mic3/criterion. N a
flattened-matrix numerical-research canary.
