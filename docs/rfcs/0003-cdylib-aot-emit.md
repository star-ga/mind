# RFC 0003: cdylib AOT emit + symbol versioning

- **Start Date**: 2026-05-17
- **RFC PR**: TBD
- **Status**: Draft
- **Target Release**: v0.3.0
- **Phase**: 2 of 2 (Phase 1 = `pub fn` → C ABI wrappers, landed v0.2.6 via RFC 0002)
- **Normative reference**: `mind-spec` RFC-0007 (`spec/v1.0/ffi.md`) — C ABI
  contract, type mapping, header layout, and ABI stability guarantees. This RFC
  owns the compiler-side mechanics that satisfy them.

## Summary

Wire `mindc build --lib` into a first-class cdylib emit path. After this RFC:

1. `mindc build --lib` lowers through the existing
   `libmind::eval::mlir_build::build_all` pipeline and writes
   `target/{profile}/lib{crate}.so` (`.dylib` on macOS, `.dll` on Windows).
2. Each `pub fn` in `IRModule.exports` produces a versioned wrapper symbol
   `mind_fn_<name>_v1_invoke` — the `_v1` ABI-version suffix deferred from
   RFC 0002's Alternatives section.
3. A profile-locked C header is written alongside the library at
   `target/{profile}/include/{crate}.h` with one declaration per exported
   function.

`mindc build` without `--lib` is bit-identical to v0.2.11. All new code paths
are gated behind the `mlir-build` feature that already exists.

## Motivation

### Today (v0.2.11 baseline, commit c444c77)

- `mindc --emit-shared <path>` exists and produces a `.so` via `mlir-build`.
  It does not plumb `IRModule.exports` into the object: there are no
  `mind_fn_*` symbols in the output.
- `mindc build` dispatches to `build_project`, which produces a binary
  executable. There is no `--lib` mode in the `Build` subcommand.
- `BuildOptions` in `mlir_build.rs` carries `emit_shared: Option<&Path>` but
  the linker step emits an undecorated `.so` — no export wrapper pass runs.
- Host runtimes that want a pre-compiled MIND function have no stable ABI to
  `dlopen` against.

### After this RFC

- `mindc build --lib` resolves the project manifest, compiles `src/lib.mind`
  (falling back to `src/main.mind`), runs the RFC 0002 export wrapper pass,
  and writes the versioned `.so` + `.h` pair.
- The `_v1` suffix encodes the ABI generation. RFC 0002 chose `_invoke` as
  the base name; this RFC layers `_v1` between the function name and
  `_invoke`: `mind_fn_<name>_v1_invoke`. Host code that links against `_v1`
  can coexist with a hypothetical `_v2` from a future ABI revision without a
  `dlsym` collision.
- mind-nerve Phase 2 (`mind_fn_preselect_pre_tokenized_invoke`) and
  mind-runtime consumers adopt the versioned name in the same patch cycle.

## Guide-level explanation

### A worked example

```mind
// src/lib.mind
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

```toml
# Mind.toml
[package]
name = "mylib"
version = "0.1.0"

[exports]
c_abi = ["add"]
```

```bash
$ mindc build --lib --release
   Compiling mylib v0.1.0
$ ls target/release/
libmylib.so   include/mylib.h
$ nm -D target/release/libmylib.so | grep mind_fn_
T mind_fn_add_v1_invoke
```

Generated header:

```c
/* target/release/include/mylib.h — machine-generated, do not edit */
/* profile: default  crate: mylib  mind: 0.3.0  abi: v1 */
#pragma once
#include "mind_runtime.h"

/* pub fn add(a: i32, b: i32) -> i32 */
int mind_fn_add_v1_invoke(const MindIO *inputs,  size_t in_count,
                          MindIO       *outputs, size_t out_count);
```

Consuming host:

```c
#include "mylib.h"
#include <dlfcn.h>

void example(void) {
    void *lib = dlopen("libmylib.so", RTLD_LAZY);
    mind_fn_add_v1_invoke_t fn =
        (mind_fn_add_v1_invoke_t)dlsym(lib, "mind_fn_add_v1_invoke");

    MindIO inputs[2] = { MindIO_scalar_i32(3), MindIO_scalar_i32(4) };
    MindIO output[1];
    fn(inputs, 2, output, 1);  /* output[0] holds 7 */
}
```

The symbol `mind_fn_add_v1_invoke` is the only stable contract. MLIR dialect
choices, LLVM pass pipeline, and evidence-chain hooks stay private.

### Profile locking

The `--profile` flag (introduced in RFC 0002 deliverable 5) scopes the output
directory and embeds the profile tag in the header comment. A `default` build
and an `embedded` build of the same crate land in separate directories and
cannot silently overwrite each other.

```
target/
  default/
    libmylib.so
    include/mylib.h
  embedded/
    libmylib.so
    include/mylib.h
```

## Reference-level explanation

### CLI change

The `Build` subcommand in `mindc.rs` gains one flag:

```rust
Build {
    release:  bool,
    target:   Option<String>,
    verbose:  bool,
    /// Emit a cdylib instead of an executable.
    /// Output: target/{profile}/lib{name}.so + target/{profile}/include/{name}.h
    /// Requires the `mlir-build` feature.
    #[arg(long)]
    lib: bool,
}
```

`BuildOptions` in `src/project/mod.rs` gains `lib: bool`. When `lib` is true,
`build_project` routes to a new `build_lib` branch rather than the binary path.

### AOT lowering pipeline (build_lib)

```
Mind.toml + src/lib.mind
        │
        ▼
AST → IRModule (existing parse + lower_to_ir)
        │
        ▼  IRModule.exports populated (RFC 0002 merge step)
        │
        ▼
lower_to_mlir  → primal_mlir: String   (existing MlirProducts)
        │
        ▼  [feature = "ffi-c-user"]
 c_export_wrapper pass  (RFC 0002 D2)
 appends wrapper MLIR for each exports entry
        │
        ▼
mlir_build::build_all(primal_mlir + wrappers, tools, opts)
  mlir-opt → mlir-translate → clang -shared -fPIC
        │
        ▼
target/{profile}/lib{crate}.so
        │
        ▼  [feature = "mlir-build"]
header_gen::emit_header(&ir_module, profile, crate_name)
        │
        ▼
target/{profile}/include/{crate}.h
```

The `build_all` call is unchanged from v0.2.11 except that `emit_shared` is
now set to `target/{profile}/lib{crate}.so` and the MLIR input includes the
wrapper declarations emitted by the RFC 0002 `c_export_wrapper` pass.

### Symbol versioning

RFC 0002 specified `mind_fn_<name>_invoke`. This RFC inserts the ABI version
token between the function name and the verb:

```
mind_fn_<name>_v1_invoke
```

The version token is `v` followed by a decimal integer. It increments only
on ABI-breaking changes to the `MindIO` calling convention (type layout,
slot semantics, return code contract). `v1` is the initial stable ABI
matching mind-spec RFC-0007.

The `c_export_wrapper` pass in RFC 0002 emits `mind_fn_<name>_invoke` (no
version token) as its base symbol. This RFC changes that pass to emit
`mind_fn_<name>_v1_invoke`. The unversioned name is optionally emitted as a
weak alias — see Open Questions.

### Header generation (new: `src/codegen/header_gen.rs`)

```rust
#[cfg(feature = "mlir-build")]
pub fn emit_header(
    module:     &IRModule,
    profile:    &ProfileTag,
    crate_name: &str,
    mind_ver:   &str,
) -> String
```

Iterates `module.exports` in deterministic order (sorted by `FnId`), maps
each function's parameter and return types through the same type table used
by `spec/v1.0/ffi.md`, and writes one `int mind_fn_<name>_v1_invoke(...)`
declaration per export. The emitted header includes a `profile:` comment so
consumers can detect a profile mismatch at human-review time.

The function is called from `build_lib` after `build_all` succeeds. Header
write failures are fatal (the `.so` exists but is unusable without a matching
header).

### Compile-speed invariant

Frontend headline numbers (1.8–15.5 µs against
`.bench-baseline-2026-04-28-pratt.txt`) MUST NOT regress.

- `build --lib` is a new dispatch branch in `build_project`. The existing
  `build --bin` path has zero new instructions on its hot path.
- `header_gen::emit_header` runs after `build_all`, which is I/O-bound
  (subprocess: `mlir-opt | mlir-translate | clang`). No parser or IR-lowering
  bench can be affected.
- `emit_header` itself is gated behind `#[cfg(feature = "mlir-build")]`. When
  the feature is off the function does not exist; the linker removes it.
- A new sub-benchmark `bench_header_gen(1_export | 10_exports | 100_exports)`
  lives in `benches/compiler.rs` alongside the existing `bench_c_export_wrapper`
  sub-benchmark from RFC 0002. The headline benches (`small_matmul`,
  `medium_mlp`, `large_network`) are unchanged.

Acceptance gate: `cargo bench --bench compiler -- --quick` shows ≤ 2% mean
regression on the three headline benches (existing `bench-gate.yml` threshold).

## Drawbacks

1. Compile time grows linearly in number of exports — the wrapper pass and
   header emitter both iterate `IRModule.exports`. Most crates have ≤ 10
   exports; the sub-bench keeps this visible.
2. The `.so` symbol table grows by one entry per export. For a crate with a
   large public surface (100+ functions) this is measurable table bloat.
3. Header generation couples the compiler to the type layout defined in
   `spec/v1.0/ffi.md`. Any spec change that alters the `MindIO` struct forces
   a coordinated compiler + spec update.

## Rationale and alternatives

### Why `_v1` in the symbol name rather than ELF version scripts?

ELF symbol versioning (`.symver`, `--version-script`) provides the same
isolation at the linker level but requires a separate `.map` file and is not
portable to macOS (where `install_name_tool` / `@rpath` are the idiom) or
Windows (DEF files). Encoding the ABI generation in the symbol name is
portable, visible from `nm`, and does not require build-system cooperation
from the consumer.

### Why `target/{profile}/lib{crate}.so` rather than a flat `lib{crate}.so`?

Profile-scoped output directories prevent a `default`-profile build from
silently overwriting an `embedded`-profile build. This mirrors the cache
fingerprint separation introduced in RFC 0002 deliverable 5 and matches what
Cargo does with `target/debug` vs `target/release`.

### Why `include/{crate}.h` rather than `include/<crate>/`?

A single-file header is the lowest-friction consumption pattern: the consumer
adds one `-I target/release` flag. A subdirectory (`include/mylib/mylib.h`)
adds no information for a single-crate build and complicates pkg-config
generation. Multi-crate scenarios with name conflicts can be addressed by
renaming at install time — see Open Questions.

### Considered: expose unversioned `mind_fn_<name>_invoke` as the primary symbol

Rejected. The unversioned name from RFC 0002 is still in `ffi-c-user` gated
code that has not shipped in a release. Renaming to `_v1_invoke` now, before
any downstream consumer is pinned, costs nothing. Keeping the unversioned name
and adding `_v1` as an alias is the compromise captured in Open Questions.

## Adoption plan

Each deliverable ships as a separate mindc patch on the 0.3.0 line:

| # | Deliverable | Feature gate | Notes |
|---|-------------|-------------|-------|
| D1 | `mindc build --lib` mode dispatch in `build_project` | `mlir-build` | Calls existing `build_all` with `emit_shared`; no export wrappers yet. Sub-bench added. |
| D2 | AOT lowering through `mlir_build` path with per-fn export plumbing | `mlir-build` + `ffi-c-user` | Feeds RFC 0002's `c_export_wrapper` pass output into `build_all`. |
| D3 | `mind_fn_<name>_v1_invoke` wrapper symbol emission | `ffi-c-user` | Renames base symbol from RFC 0002 `_invoke` → `_v1_invoke`. RFC 0002 D2 must land first. |
| D4 | Header generation (`target/{profile}/include/{crate}.h`) | `mlir-build` | `header_gen::emit_header` + directory creation. |
| D5 | End-to-end smoke test: `pub fn add(a: i32, b: i32) -> i32` → `.so` → `dlopen` → call | — | Integration test in `tests/cdylib_smoke.rs`. Verifies symbol present, return value correct, header parseable by `cc` crate. |

mind-nerve and mind-runtime update their `dlsym` call sites from
`mind_fn_preselect_pre_tokenized_invoke` to
`mind_fn_preselect_pre_tokenized_v1_invoke` in the same patch cycle as D3.

## Open questions

1. **Unversioned alias.** Should `mind_fn_X_v1_invoke` be re-exported as the
   weak alias `mind_fn_X_invoke` for forward compatibility with consumers that
   linked against the RFC 0002 unversioned name? Lean: yes for the v0.3.0
   transition window, removed in v0.4.0 once downstreams have migrated.

2. **Header install layout.** Should the generated header ship flat as
   `include/{crate}.h` or under a subdirectory `include/{crate}/{crate}.h`?
   Flat is simpler for single-crate projects; subdirectory avoids name
   collisions when multiple MIND libraries share an install prefix. Lean: flat
   for v0.3.0, revisit if a collision is reported.

3. **Symbol table compression.** Should the `.so` request `--hash-style=gnu`
   by default (GNU hash, O(1) lookup) over the default SYSV hash? The
   `build_all` linker invocation passes through to `clang -shared`; adding
   `-Wl,--hash-style=gnu` is one flag. Lean: yes — no correctness risk, faster
   `dlsym` at runtime, smaller `.so` for large export tables.
