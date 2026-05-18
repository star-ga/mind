# RFC 0005: Pure-MIND Standard Surface

- **Start Date**: 2026-05-18
- **RFC PR**: TBD
- **Status**: Draft
- **Target Release**: v0.4.0 (Phase 15 self-hosting long pole)
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

## Reference-level explanation

### `std.vec` — `Vec<T>`

A `Vec<T>` is a struct `{ ptr, len, cap }`. Backing store is a
heap allocation obtained from a single allocator primitive
`__mind_alloc(bytes) -> ptr` / `__mind_realloc` / `__mind_free`
(the only new compiler intrinsics this RFC requires; everything else
is pure MIND).

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

`__mind_alloc`, `__mind_realloc`, `__mind_free`, `__mind_read`,
`__mind_write`. Five primitives. Everything else in this RFC is pure
MIND compiled by the existing pipeline. Keeping the intrinsic set this
small is what keeps the bootstrap honest: stage-1 only needs the Rust
compiler to provide these five, then the std surface is MIND.

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

1. **Intrinsics.** Land the five `__mind_*` intrinsics in the Rust
   stage-0 compiler, feature-gated `std-surface`. Sub-bench added.
2. **`std.vec`.** Pure-MIND `Vec<T>` on the intrinsics. Tests +
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
