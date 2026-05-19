# RFC 0005: Pure-MIND Standard Surface

- **Start Date**: 2026-05-18
- **RFC PR**: TBD
- **Status**: **Shipped through Phase C in v0.4.2** (2026-05-18) —
  see "Adoption plan" below for the per-phase landing record.  The
  document itself stays *Draft* for the future Phase D+ items
  (cross-arg type matching across Named structs, env-var stdlib
  root for the single-file mindc CLI, dependency-style Mind.toml
  declaration of external modules).
- **Target Release**: v0.4.x (Phase 2 + Phase B + Phase C all
  landed under the same minor; future phases pin to v0.4.x+).
- **Normative reference**: `mind-spec` v1.0 (type system, slices,
  determinism contract); RFC 0003 (cdylib/`llc` shell-out seam used by
  the I/O layer).

## Summary

Define the minimum pure-MIND standard surface a non-trivial MIND
program — specifically a MIND compiler — needs: a growable `Vec<T>`,
a `String`, an order-deterministic `Map<K, V>`, and deterministic
file I/O. Every type is implementable *in MIND* (so the stage-1
self-hosted compiler can use it) and every operation preserves the
Q16.16 cross-arch bit-identity and evidence-chain contracts.

This is the **long pole of Phase 15**. Cross-module imports
(Phase 10.6 item 9) and the C-ABI/cdylib seam (RFC 0002/0003) are
done; a lexer / parser / symbol-table still cannot be *written*
without these four types.

## Motivation

### Today

- MIND has fixed slices, `.len()`, `let mut`, tensors, structs,
  enums, `match`. It has **no growable container, no string type, no
  map, no file I/O**.
- A compiler's lexer needs a growable token buffer; its parser needs
  a string scanner; its symbol table needs a map; its driver needs to
  read source files. None are expressible today.
- Consequence: the self-hosted `mindc.mind` (Phase 15 deliverable 1)
  cannot be started — the blocker is not Rust, it is the absence of
  these four types.

### After this RFC

- `Vec<T>`, `String`, `Map<K, V>`, and `io` are available as pure-MIND
  library modules. `mindc.mind` becomes expressible. The IR→LLVM-text
  backend (Phase 15) can be written.

## Guide-level explanation

```mind
use std.vec
use std.string
use std.map
use std.io

pub fn count_idents(path: String) -> u32 {
    let src: String = io.read_to_string(path);   // deterministic read
    let mut toks: Vec<String> = vec.new();
    let mut seen: Map<String, u32> = map.new();   // insertion-ordered
    // ... lex `src`, vec.push(&mut toks, tok), map.insert(&mut seen, ...)
    map.len(&seen)
}
```

Types are nominal MIND structs; operations are free functions in the
module (consistent with MIND's existing free-function style — no
trait/method dispatch is introduced by this RFC).

## Prerequisites discovered during implementation research (2026-05-18)

A pre-implementation read of `src/types/value.rs`,
`src/type_checker/mod.rs`, `src/mlir/lowering.rs`, and
`src/opt/ir_canonical.rs` surfaced three blocking facts this RFC must
account for **before** any phase-2 code:

- **P0a — there is no pointer type.** `ValueType` is
  `ScalarI32 | ScalarI64 | ScalarF32 | ScalarF64 | ScalarBool |
  Tensor | GradMap`. Adding a `Ptr` variant would ripple through the
  entire type-checker and threaten the µs frontend moat. **Resolution:
  the intrinsic address ABI is opaque `i64`** (a machine address is a
  64-bit integer). `Ptr<T>` is a *pure-MIND newtype* `struct Ptr { addr: i64 }`
  at the `std` level — zero type-system change. All five intrinsics
  below are re-signed in terms of `i64`, never a built-in `ptr`.
- **P0b — `Instr::Call` is not lowered to MLIR.** `emit_instr` in
  `src/mlir/lowering.rs` handles only `ConstI64`, `ConstTensor`,
  `BinOp`, `MatMul`, `Conv2d`, `Output`. `src/opt/ir_canonical.rs`
  documents that the lowerer "does not yet emit `Instr::Call` for all
  user-defined" calls. A generic call therefore never reaches LLVM.
  This means **Phase 0 of this RFC is generic `Instr::Call` → MLIR
  lowering** (emit `func.call` / `llvm.call`), a foundational path the
  self-hosting IR→LLVM-text backend (Phase 15) also requires. The five
  intrinsics are its first consumers; they cannot work before it.
- **P0c — no load/store-at-address path in MIND today.** Slices
  (`&[u8]`) read by index but can only come from a struct field or a
  function parameter — there is no built-in `addr: i64 -> &[u8]` and
  no scalar load/store intrinsic. Without one, `vec.push(&mut Vec<T>,
  T)` has *no way* to write a value into the `__mind_alloc`-returned
  backing store at offset `len * sizeof(T)`. The five intrinsics named
  in the original draft cover allocation and *file-bulk* I/O, but the
  middle layer — single-slot read/write at an opaque address — is
  missing. **Resolution: expand the intrinsic set from 5 to 7**, all
  still `i64`-signed (the P0a discipline is preserved):

  ```
  __mind_load_i64(addr: i64)                  -> i64   // *(i64*)addr
  __mind_store_i64(addr: i64, value: i64)     -> i64   // 0 == ok
  ```

  Sub-i64 widths (u8/u16/u32) are built on `__mind_load_i64` plus
  `& 0xff` etc. — `std.string` and `std.map` are not in any hot path
  the µs moat measures, so packing eight bytes per load is a clarity
  win, not a perf concession. (If a perf gap shows up later, narrow
  loads can be added as `_v1`-ABI compatible additions; the i64 pair
  is the minimum bootstrap.)

  Phase 1 (already shipped) declared the original 5 intrinsics in the
  type-checker. Phase 1.5 — small, scoped — adds the two load/store
  names to the same registry. Phase 2 then has a working write path.

Revised adoption order: **Phase 0 (call lowering) → Phase 1 (5 alloc/
I-O intrinsics) → Phase 1.5 (2 load/store intrinsics) → Phase 2+
(`std.vec` and up).** All three blockers are caught **before** any
pure-MIND `Vec` code is written.

### Two additional implementation blockers discovered before Phase 2

A read of `src/mlir/lowering.rs`, `src/eval/lower.rs`, and the existing
multi-fn pipeline surfaced two more facts blocking the *MLIR-text*
emission of `std/vec.mind`:

- **P0d — `Instr::FnDef` was not lowered to MLIR.** The lowerer
  handled `Instr::Call` (Phase 0) but `Instr::FnDef` fell through to
  the `UnsupportedOp` catch-all. User-defined functions never reached
  MLIR as `func.func` definitions. **Resolution: shipped** in commit
  `aacebe1` — the gated `Instr::FnDef` arm emits `func.func @name(%pN: i64,
  ...) -> i64 { ... return %ret : i64 }` as a sibling top-level symbol
  before `@main`. `Instr::Param` binds the param ValueId; `Instr::Return`
  emits `return %v : i64`. The assembler filters locally defined names
  out of the Phase-0 extern fwd-decl loop. No struct decisions in this
  step — all params + returns are `i64` (i64-ABI rule, P0a discipline).

- **P0e — struct codegen.** `Node::StructDef`, `Node::StructLit`,
  and `Node::FieldAccess` previously parsed and type-checked as
  placeholder `ScalarI32`, but lowered to `ConstI64(0)`. **Status:
  Option C (heap record) shipped as P0e Step 1 — write-path only.**
  The Step-1 lowering populates `IRModule.struct_defs[name]` from
  `StructDef` and emits `__mind_alloc(8*N)` + N×`__mind_store_i64`
  for each `StructLit`. Fields are reordered into canonical
  (declared) order before the stores so out-of-order literals still
  match the schema. The struct value is the i64 base address from
  `__mind_alloc`, so it threads through function boundaries just like
  any other i64 — `vec_new() -> Vec<T>` now lowers cleanly.

  **P0f follow-up — `FieldAccess` read path.** Still placeholder. The
  AST `Node::FieldAccess { receiver, field }` doesn't carry the
  receiver's struct name, so the lowering pass needs a `struct_env`
  binding-table that tracks which local `let` names map to which
  struct schemas. Once that's threaded, `obj.field` lowers to
  `__mind_load_i64(addr + 8*field_index)`. `vec_push(&mut Vec<T>, T)`
  ships in P0f.

#### P0e resolution options

The struct ABI is a one-way door — whatever layout lands becomes the
public contract for every pure-MIND program. Options:

- **Option A — flat SSA bundle.** Lower `struct Vec { addr, len, cap }`
  to three independent `i64` SSA values in the calling fn's scope.
  `StructLit` binds the three to one local name; `FieldAccess` resolves
  to the right child id. *Pros:* zero MLIR primitives added, zero ABI
  commitment. *Cons:* structs can't cross function boundaries — kills
  `vec_new() -> Vec<T>` and `vec_push(&mut Vec<T>, T)`. Phase 2 doesn't
  ship on this alone.

- **Option B — multi-return tuple ABI.** Lower fn returns / args of
  struct type to multiple `i64` values: `func.func @vec_new() -> (i64,
  i64, i64)`. *Pros:* matches existing i64-only ABI; no memref needed;
  Phase 2 ships on this. *Cons:* makes ABI versioning awkward (adding
  a field changes the arity); a `&mut Vec` argument becomes three
  inputs + three outputs.

- **Option C — heap-allocated record via `__mind_alloc` + load/store.**
  Each struct instance is an i64 address into a 3×i64 block; field
  access is `__mind_load_i64(addr + 8*field_index)`. *Pros:* uniform
  with how `Vec`'s own backing store works; trivially extends to any
  field count. *Cons:* every struct construct is a heap allocation
  (free is the caller's problem); not zero-cost; allocator hot path.

- **Option D — MLIR `llvm.struct` / `memref<3xi64>`.** Use MLIR's
  native aggregate types. *Pros:* canonical, future-proof, what LLVM
  ultimately wants anyway. *Cons:* widens the type-system surface
  inside MIND IR; couples to MLIR's struct dialect choices; tougher to
  port to the IR→LLVM-text backend in Phase 15.

The recommended order is **decide via the multi-LLM consensus +
autoresearch loop** before any implementation work, per the standing
"non-obvious architectural decisions need consensus" rule. Until then,
Phase 2 is *blocked on a design decision, not on code*.

## Reference-level explanation

### `std.vec` — `Vec<T>`

A `Vec<T>` is a struct `{ addr: i64, len: i64, cap: i64 }` where
`addr` is the opaque machine address of the backing store (P0a — no
built-in pointer type; an address is a 64-bit integer). Backing store
is obtained from the allocator intrinsics, all signed in `i64`:

```
__mind_alloc(bytes: i64)              -> i64   // 0 == allocation failed
__mind_realloc(addr: i64, n: i64)     -> i64
__mind_free(addr: i64)                -> i64   // 0 == ok
```

`std` wraps a raw `addr` in a pure-MIND newtype
`struct Ptr<T> { addr: i64 }` for type distinction at the library
level; the compiler only ever sees `i64`. Everything above the five
intrinsics is pure MIND.

- `vec.new() -> Vec<T>` — zero-cap, no allocation.
- `vec.push(&mut Vec<T>, T)` — amortised O(1); growth doubles cap.
- `vec.get(&Vec<T>, usize) -> Option<&T>`
- `vec.len(&Vec<T>) -> usize`
- `vec.as_slice(&Vec<T>) -> &[T]` — bridges to existing slice ops.

Growth policy is **fixed and documented** (double, min cap 4) so the
allocation trace is deterministic — load-bearing for the evidence
chain and for `stage1 == stage2` reproducibility.

### `std.string` — `String`

`String` is `Vec<u8>` with a UTF-8 well-formedness invariant enforced
at construction (`string.from_bytes` returns `Result`, `string.push`
takes a validated `char`). Byte-level operations reuse `std.vec`.
This mirrors the existing byte-level tokenizer discipline (every
UTF-8 input round-trips) so the Tier-3 multilingual floor holds for
compiler source too.

### `std.map` — `Map<K, V>`

**Insertion-ordered**, not hash-randomised. The default `Map` is a
deterministic open-addressing table whose iteration order is
insertion order (an ordering vector alongside the buckets). Rationale:
a hash-randomised map would make the symbol table's iteration order
non-deterministic, breaking `model_hash` / evidence reproducibility
and the bootstrap fixed point. Determinism is not optional here — it
is the reason the default is ordered rather than a `HashMap`-style
type.

- `map.new() -> Map<K, V>`
- `map.insert(&mut Map<K, V>, K, V) -> Option<V>`
- `map.get(&Map<K, V>, &K) -> Option<&V>`
- `map.iter(&Map<K, V>)` — yields entries in **insertion order**.

### `std.io` — deterministic file I/O

- `io.read_to_string(String) -> Result<String, IoError>`
- `io.write(String, &[u8]) -> Result<(), IoError>`

I/O is a **governed surface**: each call emits an evidence record
(path hash + byte-length + operation) into the evidence chain, exactly
like every other governance-relevant decision. The implementation
shells out through the same process seam RFC 0003 established for
`llc`/`clang` — no new FFI surface beyond the three allocator
intrinsics and a `__mind_read`/`__mind_write` syscall pair.

### New compiler intrinsics (the entire non-MIND surface)

Seven primitives, all `i64`-signed (P0a — no `ptr` type):

```
__mind_alloc(bytes: i64)                         -> i64
__mind_realloc(addr: i64, bytes: i64)            -> i64
__mind_free(addr: i64)                           -> i64
__mind_load_i64(addr: i64)                       -> i64   // *(i64*)addr  (P0c)
__mind_store_i64(addr: i64, value: i64)          -> i64   // 0 == ok      (P0c)
__mind_read(path_addr: i64, path_len: i64,
            buf_addr: i64, buf_cap: i64)         -> i64   // bytes read, <0 = errno
__mind_write(path_addr: i64, path_len: i64,
             buf_addr: i64, buf_len: i64)        -> i64   // bytes written, <0 = errno
```

Everything else in this RFC is pure MIND compiled by the existing
pipeline. Keeping the intrinsic set this small — and `i64`-only, so
no type-system change — is what keeps the bootstrap honest and the
moat untouched: stage-1 only needs the Rust compiler to provide these
seven (lowered via the Phase-0 generic-call path), then the std
surface is MIND.

## Compile-speed invariant (the moat)

`std.*` is **library code compiled per-program**, not frontend code.
It cannot regress the 1.8–15.5 µs frontend benches — those measure
parse→typecheck→IR of small kernels that import nothing. A new
sub-benchmark `std_surface` measures `Vec`/`Map` op throughput in
isolation; it is its own bench target (like `cross_module`), never in
the headline `compiler` group. The five intrinsics are lowered as
direct calls (no dispatch). `.bench-baseline` ±2% gate unchanged.

## Drawbacks

- Five new compiler intrinsics to keep stable across the bootstrap
  boundary (they must be ABI-frozen like `mind_fn_*_v1`).
- An ordered map is slightly slower than a hash-randomised one; the
  determinism requirement makes this non-negotiable, not a tuning
  choice.
- A real allocator in pure MIND is a meaningful body of code; phasing
  (below) ships the minimum first.

## Rationale and alternatives

- **Expose Rust `Vec`/`String`/`HashMap` via FFI.** Rejected: makes
  the std surface un-self-hostable — stage-1 would still need Rust.
  The whole point is a MIND-expressible surface.
- **Hash-randomised `HashMap`.** Rejected: breaks determinism /
  bootstrap fixed point. Ordered map is the default; a fast
  unordered map may come later behind an explicit non-deterministic
  opt-in for non-governed code.
- **Arena-only, no free.** Considered for the compiler (Zig-style
  arena-per-module — see Phase 15 / autoresearch finding). Kept as an
  *option* (`vec.with_arena`) but the general `Vec` supports `free`
  so `std` is usable outside the compiler too.

## Adoption plan (phased — this is the long pole)

### Landing record (updated 2026-05-18)

| Phase | What | Tag | Status |
| ----- | ---- | --- | ------ |
| 0     | Generic `Instr::Call` → MLIR lowering | mindc v0.2.11 | **shipped** |
| 1     | `__mind_alloc/free/realloc/read/write` intrinsics declared | mindc v0.2.11 | **shipped** |
| 1.5   | `__mind_load_i64/store_i64` declared (P0c) | mindc v0.2.11 | **shipped** |
| P0d   | `Instr::FnDef` → MLIR `func.func` | mindc v0.3.0 | **shipped** |
| P0e   | Struct heap-record ABI (Option C, 9-LLM consensus) | mindc v0.3.0 | **shipped** |
| P0f   | FieldAccess read path (Step 1 + Step 2 side-table) | mindc v0.3.0 | **shipped** |
| 2     | `std/{vec,string,map,io}.mind` lower end-to-end | mindc v0.4.0 | **shipped** |
| 2-resolver | `use std.foo` cross-module resolver (Phase A) | mindc v0.4.0 | **shipped** |
| B     | Per-arg signature matching on imported `pub fn`s | mindc v0.4.1 | **shipped** |
| C     | Bundle `std/*.mind` into mindc via `include_str!` | mindc v0.4.2 | **shipped** |
| D₁    | `$MIND_STDLIB_PATH` override for the bundled stdlib | mindc v0.4.2 | **shipped** |
| 6     | Self-host smoke — `mindc` lexer in MIND | TBD | open |

The original phases-1-through-6 sequence is preserved below for the
RFC's historical narrative.  Future Phase D items (cross-arg type
matching across Named structs; env-var stdlib root for the
single-file `mindc` CLI without project context; dependency-style
`Mind.toml` declaration of external modules) will extend the table.

0. **Generic call lowering (BLOCKING — P0b).** Emit MLIR for
   `Instr::Call` (`func.call` / `llvm.call`) so a non-tensor function
   call reaches LLVM at all. Feature-gated `std-surface`; default
   build byte-identical (the lowerer's `emit_instr` gains a gated
   `Instr::Call` arm only). This is also the path the Phase 15
   self-hosting IR→LLVM-text backend needs — it is foundational, not
   std-specific. Own sub-bench; headline benches untouched. **All
   subsequent phases depend on this.**
1. **Intrinsics — allocation + bulk I/O.** Declare the five `i64`-
   signed `__mind_*` intrinsics (`alloc`, `realloc`, `free`, `read`,
   `write`) so the type-checker accepts them with fixed signatures and
   the Phase-0 path lowers them to `llvm.call @__mind_*`. Feature-gated
   `std-surface`. Sub-bench added.
1.5. **Intrinsics — load/store (BLOCKING — P0c).** Declare the two
   scalar load/store intrinsics (`__mind_load_i64`, `__mind_store_i64`)
   in the same gated registry. Mirrors Phase 1 exactly. Without these
   two, `vec.push` has no path from `__mind_alloc`-returned address to
   actually writing the value — and no Vec means no String / Map / io
   either. Sub-bench extended.
2. **`std.vec`.** Pure-MIND `Vec<T>` on the seven intrinsics. Tests +
   `std_surface` bench.
3. **`std.string`.** `String` on `std.vec` + UTF-8 invariant; reuse
   the Tier-3 round-trip corpus as the conformance test.
4. **`std.map`.** Insertion-ordered map; determinism test asserts
   iteration order is stable across runs and architectures.
5. **`std.io`.** Evidence-emitting read/write; governance test
   asserts every call produces a chain record.
6. **Self-host smoke.** Re-express the `mindc` lexer in MIND on top
   of `std.*` and compile it with stage-0 — the first concrete
   Phase 15 deliverable-1 milestone.

## Open questions

1. `char` representation — `u32` scalar (Unicode scalar value) vs a
   1–4 byte struct? Lean: `u32` scalar; UTF-8 (de)coding lives in
   `std.string`.
2. `Result`/`Option` are assumed available (enum + generics, shipped
   Phase 10.5/10.7). Confirm generic enums are sufficient for
   `Result<T, IoError>` before phase 1.
3. Allocator: bump-only for the compiler vs general-purpose with
   `free`? Lean: ship general-purpose; expose `with_arena` as the
   compiler's fast path (Phase 15 self-hosting accelerant).
4. Do the five intrinsics get `_v1` ABI versioning like the C-ABI
   wrappers? Lean: yes — they cross the bootstrap boundary, so the
   same fractal-stability argument (RFC 0003) applies.
