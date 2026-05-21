# RFC 0010: Memory Safety Model + C ABI in Pure MIND

| Field | Value |
|---|---|
| RFC | 0010 |
| Title | Memory safety model + C ABI in pure MIND |
| Status | **Phase F Scaffolded — Phase C Shipped, Phase D + Phases G–J planned** |
| Authors | STARGA Inc. |
| Created | 2026-05-21 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0007 (Mindcraft), RFC 0008 (mindc build), RFC 0009 (federation package layer), RFC 0011 (async + structured concurrency) |

---

## 1. Motivation

Two concerns have been deferred through the self-hosting ladder; this RFC
addresses both in a single specification because they are tightly coupled at
the implementation level.

### 1.1 Memory safety without a garbage collector

The existing MIND safety story is invariant-based: the Q16.16 overflow checks
and decision-point coverage discipline catch a class of correctness problems
at runtime, but neither mechanism provides a coherent answer for aliasing,
use-after-free, or dangling references on heap-allocated data. As MIND moves
into production workloads — the governance substrate, the native encoder, the
async task infrastructure (RFC 0011) — the absence of a heap safety model
becomes load-bearing.

MIND does not want a garbage collector. GC introduces non-deterministic pause
latency that is incompatible with the determinism contract the language already
establishes for formatting (RFC 0007 §8) and for replay-deterministic execution
(RFC 0011 §7). The memory safety model must be deterministic and
zero-collection-pause.

MIND also does not want pervasive lifetime annotations on every function
signature. The annotation overhead cost has been studied extensively in
language design; the conclusion for MIND is that explicit annotations belong
at allocation boundaries, not at every call site. The model in this RFC is
inference-friendly: most code never writes a lifetime annotation.

### 1.2 libMLIR / libLLVM FFI in pure MIND

Today `mindc` calls into MLIR and LLVM through the Rust `mlir-sys` and
`inkwell` crates. The RFC 0008 Phase G KEYSTONE shipped the pure-MIND build
orchestrator — `mindc build` self-hosts the mind repo — but Rust remains in the
dependency chain for the FFI into the MLIR C API. The claim "pure-MIND mindc
owns the full compile path" cannot be made while that dependency exists.

The keystone for removing it is `extern "C"` semantics in MIND. Once MIND can
call into the MLIR C API directly, the Rust FFI layer can be replaced with
hand-written MIND bindings in `std.mlir` and `std.llvm`. The Rust crate then
becomes a thin distribution shim — the Rust-hosted `mindc` binary is installed
via the standard Rust toolchain package installer —
and carries no load-bearing compilation logic.

These two concerns (heap safety, C FFI) are specified together because the
same memory model governs how unsafe FFI pointers are handled.

---

## 2. Non-goals

The following are explicitly outside the scope of RFC 0010.

**No garbage collector.** MIND will not add a GC. The three-tier model (§3)
achieves memory safety through scope-bounded lifetimes and generation checks,
not through tracing collection.

**No fully unrestricted unsafe blocks.** The `unsafe` surface (§4.2) is
deliberately narrow: it permits calling `extern "C"` functions marked `unsafe`
and performing raw pointer arithmetic. It does not grant permission to bypass
the type checker, forge generation counters, or alias region-interior pointers
across region boundaries.

**No pervasive lifetime annotations.** Lifetime annotations do not appear on
function signatures in the common case. They appear only on `extern "C"`
declarations and on `GenRef<T>` allocation sites. The model is
inference-friendly by design.

**No replacement of MLIR or LLVM as the codegen backend.** MLIR and LLVM remain
the backend throughout. RFC 0010 replaces only the Rust FFI layer that calls
into them; it does not replace the tools themselves.

**No cross-language reflection or runtime type information.** MIND does not
acquire RTTI beyond what the C ABI requires for calling convention purposes.
`extern "C"` functions see MIND types only at the ABI boundary; there is no
mechanism for a C caller to introspect a MIND value's type at runtime.

**No new `extern` block calling conventions beyond C, System V x86_64, Win64,
and AAPCS in this RFC.** Other calling conventions (e.g. platform-specific
intrinsic ABIs) are deferred to future RFCs.

---

## 3. Memory safety model — three tiers

MIND memory is partitioned into three tiers. Each tier has a different safety
mechanism. Code that stays within the tier's rules is safe by construction;
code that crosses a tier boundary requires explicit annotation.

### 3.1 Stack (Tier 1)

Stack allocations follow standard single-owner, scope-lifetime semantics. A
value allocated on the stack is freed when its enclosing scope exits. There is
no aliasing rule to state because stack values are not addressable across scope
boundaries without an explicit reference, and references to stack values have
their lifetime checked at compile time by the mindc type-checker.

This is the default tier. The vast majority of MIND programs operate entirely
in Tier 1 and encounter no new syntax or annotation requirements.

### 3.2 Region-interior heap (Tier 2)

Heap allocations within an explicit `region { ... }` block are owned by the
region. When control exits the region — normally, via early return, or via
panic — the region frees all allocations it owns. No manual `free` call is
needed or permitted for region-interior allocations.

Aliasing within a region is unrestricted: multiple bindings may reference the
same region-interior allocation simultaneously. Use-after-free is structurally
impossible for region-interior allocations: references are only ever formed
inside the region block, and the region owns the lifetime.

```mind
region {
    let v = vec_new();
    vec_push(v, 1);
    vec_push(v, 2);
    let alias = v;          // unrestricted aliasing inside the region
    vec_push(alias, 3);
    // v and alias are freed when the region exits
}
// v and alias are not in scope here; no use-after-free is possible
```

A reference to a region-interior allocation **MUST NOT** escape the region
block. The mindc type-checker enforces this: if a region-interior reference is
returned from the region block, or stored in a location that outlives the
region, a compile-time error is emitted.

### 3.3 Region-exterior heap — generation-checked references (Tier 3)

Long-lived heap allocations that must outlive the scope that created them use
`gen_alloc`, which returns a `GenRef<T>`. Each allocation carries a 64-bit
generation counter. A `GenRef<T>` is a (pointer, generation) pair. Dereferencing
a `GenRef<T>` checks that the stored generation matches the allocation's current
counter; on mismatch the runtime panics with a diagnostic naming the allocation
site. The program never observes a dangling pointer — it either gets a valid
reference or it panics with a clear attribution.

Freeing a `GenRef<T>` allocation increments the generation counter. Any
surviving `GenRef<T>` holding the previous generation will panic on the next
deref. This is the mechanism that converts use-after-free from undefined
behaviour into a deterministic, attributable runtime panic.

```mind
// Long-lived allocation — survives the allocating scope
let r: GenRef<Vec<i32>> = gen_alloc(Vec::new());

// Dereference returns Option<&T> — the caller handles the None case
match r.deref() {
    Some(v) => {
        vec_push(v, 1);
        print(vec_len(v));
    },
    None => panic("generation mismatch — allocation was freed"),
}

// Explicit free: increments generation counter
gen_free(r);

// Any subsequent r.deref() returns None — safe, not UB
```

`GenRef<T>` is the only long-lived heap allocation primitive at the language
level. There is no raw `malloc`/`free` surface exposed outside `unsafe` blocks
(§4.2). Code that does not use `unsafe` blocks cannot produce a dangling pointer.

### 3.4 Relationship to existing invariant checks

The three-tier model supplements — and does not replace — the existing
invariant-based safety layer. Q16.16 overflow checks and decision-point coverage
continue to apply on top of the memory model. A MIND program is correct when
both layers pass: the memory model guarantees no aliasing or lifetime violations,
and the invariant layer guarantees domain-specific correctness properties.

### 3.5 Tier boundary crossing

Moving a value from one tier to another is a well-defined operation, not a
special form. The rules are:

**Tier 1 → Tier 2 (stack value into a region):** Standard copy or move. The
region-interior allocation is a new allocation; the original stack value is
copied into it. This is the common case for initialising region-interior data
structures from stack-computed values.

**Tier 2 → Tier 3 (region-interior value into a long-lived allocation):**
Requires explicit `gen_alloc`. The region-interior value is copied into the
new `GenRef<T>` allocation before the region exits. Attempting to let a
`GenRef<T>` point inside a `region { }` block is a compile-time error: a
`GenRef<T>` must outlive any region, so its referent cannot be a
region-interior allocation.

**Tier 3 → Tier 1 or Tier 2 (copying out of a GenRef):** `r.deref()` returns
`Option<&T>`. The caller may copy the value behind the reference into a stack
or region-interior binding. The reference itself is consumed by the deref
expression; it does not escape the deref call site.

**Tier 3 value through `unsafe` (raw pointer access):** Inside an `unsafe`
block, a `GenRef<T>` may be dereferenced to a raw `*const T` or `*mut T` via
`r.as_ptr()` and `r.as_mut_ptr()`. The generation check is bypassed; the raw
pointer carries no generation information. This is the mechanism that FFI
bindings use to pass MIND-allocated data to C functions. The `# safety:`
requirement ensures the programmer has reasoned about the lifetime.

```mind
// Passing a GenRef-managed buffer to a C function
let buf: GenRef<[u8; 4096]> = gen_alloc([0u8; 4096]);

// # safety: buf outlives this call; the C function reads only and does not
//           store the pointer beyond this call site.
unsafe {
    c_read_into(buf.as_mut_ptr(), 4096);
}

let data = buf.deref().expect("buffer still live");
```

---

## 4. `extern "C"` and the C ABI

### 4.1 Calling convention declaration

MIND acquires first-class calling convention support. A calling convention is
declared on an `extern` block with the `callconv` annotation. The supported
conventions in this RFC are:

| Tag | Convention |
|---|---|
| `callconv(.c)` | Platform-default C ABI (alias for the platform's native C calling convention) |
| `callconv(.sysv)` | System V AMD64 ABI |
| `callconv(.win64)` | Microsoft x64 calling convention |
| `callconv(.aapcs)` | ARM Architecture Procedure Call Standard (AArch64) |

When no `callconv` annotation is present on an `extern "C"` block, the
compiler selects `callconv(.c)` — the platform default — which resolves to
`.sysv` on Linux/macOS x86_64 and `.win64` on Windows x86_64.

```mind
extern "C" {
    safe fn getpid() -> i32;
    unsafe fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

extern "C" callconv(.sysv) {
    fn printf(fmt: *const u8, ...) -> i32;
}

extern "C" callconv(.aapcs) {
    fn arm_specific_routine(x: u32) -> u32;
}
```

### 4.2 Safety attribution

Every symbol in an `extern "C"` block carries an explicit safety tag.

- **`safe fn`** — the function upholds MIND's invariants from the MIND side.
  The compiler verifies that the declared signature does not involve raw pointer
  arithmetic in the call itself. The callee is trusted to be correct; the
  declaration asserts that calling it cannot violate MIND's memory model.
- **`unsafe fn`** — the function may violate MIND's invariants (it takes raw
  pointers, has side effects on external state, etc.). Calls to `unsafe fn`
  symbols require an enclosing `unsafe { ... }` block in the caller, which in
  turn requires a `# safety: <reason>` doc comment immediately above the
  `unsafe` block. `mindc check` (RFC 0007) enforces the doc comment.

```mind
// # safety: getpid() reads the OS process ID — no pointer arithmetic,
//           no mutation of MIND state.
unsafe {
    let pid = getpid();
    print(pid);
}
```

Raw pointer types (`*const T`, `*mut T`) are only constructible inside `unsafe`
blocks or within the body of an `unsafe fn`. They cannot be stored in a `GenRef`
or a region-interior allocation without an explicit unsafe cast. This prevents
raw pointers from leaking into safe MIND code.

### 4.3 Variadic functions

The `...` syntax after the last concrete parameter declares a variadic function,
matching the C ABI directly.

```mind
extern "C" {
    unsafe fn printf(fmt: *const u8, ...) -> i32;
    unsafe fn snprintf(buf: *mut u8, n: usize, fmt: *const u8, ...) -> i32;
}
```

Variadic argument passing uses the platform calling convention rules for
variadic arguments (va_list promotion rules). The MIND compiler does not add
a higher-level variadic abstraction; callers must match the C ABI's promotion
rules manually.

### 4.4 ABI lowering (deferred to Phase B–D)

The ABI lowering details — register assignment, stack layout, struct passing,
return value conventions — are deferred to the Phase B, C, and D
implementations. This RFC commits to the declaration surface (§4.1–4.3). The
concrete lowering is a function of the target platform and will be validated
against the platform ABI specifications during Phase B–D.

---

## 5. Phasing

The implementation is staged to keep each phase independently shippable and
testable.

| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| A | Parse `extern "C"` blocks; parse `safe`/`unsafe` fn attribution; parse `callconv(.)` tags; parse `...` variadic syntax. Emit parse errors for invalid combinations. Type-check extern signatures (Copy-only rule). Lower `extern "C"` fn calls to `llvm.call`; emit `llvm.func` declarations. | mindc parses all new syntax; existing test suite unchanged; 7 Phase A tests pass. | **Shipped** (`e82b831`) |
| B | System V AMD64 calling convention lowering for `#[repr(C)]` structs (up to 4 Copy fields, ≤16B); `extern "C" fn(T) -> R` callback function pointer types (`TypeAnn::FnPtr` → `!llvm.ptr`); vararg call lowering with per-position type hints (`vararg_hints: Vec<String>` on `ExternFnDecl`); SysV struct classification (`sysv_classify_struct`). | 17 Phase B tests pass; Phase A tests unchanged; bootstrap fixed-point preserved. | **Shipped** |
| C | Win64 calling convention lowering (`win64_classify_struct`, `extern_type_to_mlir_multi_win64`, `cconv = #llvm.cconv<win64cc>` attribute on `llvm.func`/`llvm.call`); f32 vararg promotion to f64 (C11 §6.5.2.2p6, audit R-03). | 10 Phase C tests pass; `0 failed` full-suite gate. | **Shipped** |
| D | AAPCS (AArch64) calling convention lowering. | same round-trip gate on AArch64 Linux. | Planned |
| E | Hand-written MIND `std.mlir` bindings for the MLIR C API (~150 functions). Authored against the MLIR C API header set. Safety attribution per function: `safe` for pure query functions, `unsafe` for mutation and pointer-passing functions. | std.mlir compiles under mindc; a smoke test exercises round-trip MLIR construction from MIND code. | **Scaffolded** (`std/mlir.mind` — 209 fns, 673 LOC; `tests/std_mlir_bindings_smoke.rs` — 4 tests; Phase F migrates mindc internals) |
| F | Hand-written MIND `std.llvm` bindings for the LLVM C API. | std.llvm compiles; smoke test exercises IR construction from MIND code. | **Scaffolded** (`std/llvm.mind` — 221 fns, 669 LOC; `tests/std_llvm_bindings_smoke.rs` — 3 tests; Phase H migrates mindc LLVM-glue) |
| G | Migrate mindc's MLIR-glue from `mlir-sys` (Rust) to `std.mlir` (MIND). | mindc self-build smoke test passes end-to-end with the new path. | Planned |
| H | Migrate mindc's LLVM-glue from `llvm-sys` / `inkwell` (Rust) to `std.llvm` (MIND). | same self-build smoke test. | Planned |
| I (KEYSTONE) | Remove `mlir-sys` and `inkwell` from `Cargo.toml`. The Rust crate becomes a thin distribution shim. Pure-MIND mindc owns the full compile path. | The Rust dependency tree shows no mlir-sys or inkwell transitive deps; mindc produces a byte-identical result to the Phase G build. | Planned |
| J | Implement the three-tier memory model in mindc: parse `region { }` blocks, type-check region escape, lower region alloc/free, lower `GenRef<T>` with generation counter. | existing MIND programs compile unchanged; new tests exercise region and GenRef semantics. | Planned |

Phase A is the prerequisite for all subsequent phases. Phases B–D are
independent of each other and may be implemented in any order. Phases E–I are
sequentially dependent. Phase J is independent of Phases B–I and may be
implemented in parallel after Phase A.

---

## 6. Backwards compatibility

**Existing `Mind.toml` and `mindc` CLI are unchanged.** No new required fields,
no renamed subcommands.

**Existing MIND source compiles unchanged.** The `extern "C"` declaration syntax
is only triggered by new source that uses it. The three-tier memory model is
opt-in: code that does not use `region { }` blocks or `gen_alloc` is unaffected
by Phase J. The two features are additive, not breaking.

**Existing invariant checks remain.** Q16.16 overflow detection and
decision-point coverage continue to operate as before. The three-tier memory
model is a new safety layer on top of the existing invariant layer; it does not
modify or disable any existing check.

**The `unsafe` block doc-comment requirement (Phase A)** is a new `mindc check`
diagnostic, not a compile error. Projects that do not yet add `# safety:`
comments will see `warn`-severity diagnostics, not build failures. The severity
can be promoted to `error` per-project via `Mind.toml [mindcraft]` configuration
(RFC 0007 §5).

---

## 7. Open questions

### Region inference

Should the mindc compiler auto-introduce `region { }` blocks via escape
analysis, or always require explicit syntax?

**Proposed: explicit only in Phase J; inference deferred.** Explicit `region`
blocks are visible in the source — a reader can always identify lifetime
boundaries without running the compiler. Inference would make the language
easier to write but harder to audit, which conflicts with the governance-substrate
use case where code must be traceable. Inference is tracked as a separate
future RFC gated on Phase J stability.

### Generation counter wrap-around

The generation counter is 64 bits. After 2^64 allocations at the same memory
address, the counter wraps and a stale `GenRef<T>` would incorrectly pass the
generation check.

**Proposed: panic on counter wrap.** Wrap-around at a single allocation seat
requires 2^64 frees at that address — effectively unbounded for any real
workload. The panic-on-wrap policy is safe (no silent incorrect deref), and
the condition is never triggered in practice. A 32-bit counter is not proposed
because it could be exhausted in high-frequency allocation-heavy code.

### `unsafe` block scope for region checks

Do `unsafe { }` blocks bypass Tier 2 region escape checking?

**Proposed: yes.** An `unsafe` block is the explicit acknowledgement that the
programmer is operating outside the normal safety model. Escaping a
region-interior reference via an `unsafe` block is permitted, but the
`# safety:` doc comment requirement (enforced by `mindc check`) must explain
why the escape is safe. This mirrors the treatment of raw pointer construction:
unsafe operations are permitted with explicit justification, not prohibited.

---

## 8. Relation to other RFCs

**RFC 0007 (Mindcraft)** — `mindc check` acquires a new diagnostic:
`lint::missing_safety_doc` fires on any `unsafe` block that lacks a
`# safety: <reason>` doc comment immediately above it. The severity defaults
to `warn` and can be promoted to `error` in `Mind.toml [mindcraft]`.
RFC 0007 §5's per-target severity model applies: an `unsafe` block in a
`[mindcraft.gpu]`-scoped context can carry a different severity than one in
`[mindcraft.cpu]`.

**RFC 0008 (mindc build)** — Phase G landed the KEYSTONE: `mindc build`
self-hosts the mind repo. RFC 0010 Phase I is the next keystone: it removes
the last Rust FFI dependency from the pure-MIND compile path. The RFC 0008 Phase
G fixed-point oracle must remain byte-identical after RFC 0010 Phase I lands —
Phase I is a FFI replacement, not a codegen change.

**RFC 0009 (federation package layer)** — orthogonal. The package layer does
not depend on the FFI surface. A MIND package that wraps a C library uses
`extern "C"` in its source (RFC 0010); the package layer treats it identically
to any other MIND package. RFC 0010 is not a prerequisite for RFC 0009.

**RFC 0011 (async + structured concurrency)** — RFC 0011's `Future<T>` type is
a Tier 3 (region-exterior) allocation, managed via `GenRef<Future<T>>` (RFC 0011
§8). The generation check is the cross-task lifetime mechanism: a task holding
a `GenRef<Future<T>>` will panic cleanly if the owning task has been freed,
rather than observing a dangling pointer. RFC 0011 depends on RFC 0010 Phase J
being shipped before the `Future<T>` heap model is finalised.

---

## 9. Decision points

This section records the design choices made for this RFC. These decisions are
final for the specified phases; superseding a decision requires a new RFC or an
explicit backward-compatibility annotation.

### Region syntax: `region { }` vs `with` vs arena-object

Three syntactic forms were evaluated:

- `region { ... }` — explicit block syntax, lifetime visible at the opening
  brace, no additional binding required.
- `with arena { ... }` — familiar to developers who know resource-management
  patterns in other languages, but `with` carries implicit semantics in some
  contexts that conflict with MIND's explicit-first design.
- `let arena = Arena::new(); ...; arena.drop();` — maximally explicit but
  requires manual `drop` calls, which are easy to forget (and which forgetting
  is the error the model exists to prevent).

**Decision: `region { ... }`.**

The explicit block syntax makes the lifetime boundary visible to the reader
without requiring a separate binding. A `region { }` block is self-closing —
you cannot forget to call `drop`. The `with` form is not MIND idiom; `arena`
objects require manual discipline that `region` eliminates.

### Generation reference type: `GenRef<T>` vs `Weak<T>` vs `Rc<T>`

- `Weak<T>` — familiar from reference-counting languages, but implies the
  existence of a strong-count companion. MIND's three-tier model has no
  reference counting at all; the naming would be misleading.
- `Rc<T>` — reference-counted smart pointer. Incompatible with the non-GC
  commitment; reference counting is a restricted form of GC.
- `GenRef<T>` — semantically distinct. Deref returns `Option<&T>`, not `&T`.
  The caller is always required to handle the `None` case. The name accurately
  describes the mechanism (generation counter) without implying reference
  counting or a strong/weak hierarchy.

**Decision: `GenRef<T>`.**

The type name and the `Option<&T>` deref signature together communicate the
contract to the reader: this reference may have been freed; you must handle that
case. No implicit dereferencing, no hidden overhead beyond a single 64-bit
comparison per deref.

### Variadic syntax: `...` vs named variadic list type

The C ABI uses `...` after the last concrete parameter. Two options:

- `...` — matches the C ABI declaration syntax directly; no translation layer.
- A named type (e.g. `VarArgs`) — more MIND-idiomatic but requires a lowering
  step that maps to the C ABI's `va_list` convention, introducing complexity
  without benefit for the primary use case (calling existing C APIs).

**Decision: `...` after concrete parameters, matching the C ABI directly.**

MIND's `extern "C"` declarations are declarations of existing C functions, not
new MIND-native variadic designs. Matching the C declaration syntax reduces
friction and eliminates a translation layer. A MIND-native variadic design for
MIND-to-MIND calls is a separate future RFC.

---

## 10. Diagnostic surface

RFC 0010 introduces new compile-time and runtime diagnostics. This section
is the normative catalogue. Diagnostics are named with the `safety::` prefix
to distinguish them from the existing `lint::` and `fmt::` diagnostic
namespaces (RFC 0007 §6).

### Compile-time diagnostics (mindc type-checker)

| Diagnostic ID | Severity default | Condition |
|---|---|---|
| `safety::region_escape` | `error` (not configurable) | A reference to a region-interior allocation escapes the `region { }` block — returned, stored in a long-lived binding, or passed to a function that stores it beyond the call. |
| `safety::missing_safety_doc` | `warn` | An `unsafe { }` block has no `# safety: <reason>` doc comment immediately above it. Promotable to `error` via `Mind.toml [mindcraft]`. |
| `safety::raw_ptr_in_safe_context` | `error` (not configurable) | A `*const T` or `*mut T` is constructed outside an `unsafe { }` block or the body of an `unsafe fn`. |
| `safety::unsafe_call_in_safe_fn` | `error` (not configurable) | A function declared `unsafe fn` in an `extern "C"` block is called outside an `unsafe { }` block. |
| `safety::extern_missing_callconv` | `info` | An `extern "C"` block does not declare an explicit `callconv(.)` tag. Informational only; the platform default is used. |

### Runtime diagnostics (mindc runtime)

| Diagnostic | Condition | Behaviour |
|---|---|---|
| `GenRef generation mismatch` | `r.deref()` when the allocation's generation counter does not match the `GenRef`'s stored generation (because `gen_free(r)` or a later `gen_alloc` at the same address has incremented the counter). | Panic with message: `GenRef<T> generation mismatch at <alloc_site>: expected gen N, found gen M`. The allocation site is tracked in debug builds; release builds omit the source location. |
| `region allocator exhausted` | The region's internal allocator runs out of available memory for a new allocation. | Panic with message: `region allocator OOM at <region_entry_site>`. The region-entry site is always available (it is a compile-time constant). |
| `generation counter wrap` | A Tier 3 allocation seat has been freed and reallocated 2^64 times. | Panic with message: `GenRef generation counter wrap at <alloc_site>`. This is an unreachable condition in practice and is documented to make the model's completeness explicit. |

### Diagnostic configurability

`safety::region_escape`, `safety::raw_ptr_in_safe_context`, and
`safety::unsafe_call_in_safe_fn` are non-configurable hard errors. They
represent conditions that are structurally impossible to make safe; demoting
them to warnings would silently un-enforce the memory model.

`safety::missing_safety_doc` defaults to `warn` to allow existing codebases
to adopt RFC 0010 incrementally. A project that wants full enforcement adds:

```toml
[mindcraft]
"safety::missing_safety_doc" = "error"
```

`safety::extern_missing_callconv` is `info` because calling with the platform
default is a valid choice, not a mistake. It is provided for teams that want
to audit every `extern "C"` block for explicitness.

---

## 11. References

- RFC 0007 — Mindcraft (`mindc check` diagnostic surface; `safety::` namespace
  diagnostics share the reporter and severity model with `lint::` diagnostics).
- RFC 0008 — mindc build (Phase G KEYSTONE; RFC 0010 Phase I is the FFI
  keystone that follows it).
- RFC 0009 — federation package layer (orthogonal; `extern "C"` packages are
  valid MIND packages).
- RFC 0011 — async + structured concurrency (`Future<T>` as Tier 3 allocation;
  GenRef-checked across task boundaries).
