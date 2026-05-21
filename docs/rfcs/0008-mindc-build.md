# RFC 0008: mindc build + mindc test — retiring cargo from the build path

| Field | Value |
|---|---|
| RFC | 0008 |
| Title | mindc build + mindc test — retiring cargo from the build path |
| Status | **Phase F Shipped** |
| Authors | STARGA Inc. |
| Created | 2026-05-21 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0005 (pure-MIND std surface), RFC 0007 (Mindcraft), RFC 0002 (pub fn C exports), RFC 0003 (cdylib AOT emit) |

## 1. Motivation

The self-hosting credibility ladder has three rungs. The first two are
closed:

1. **Language self-hosts** — MIND is expressive enough to host its own compiler.
   Proof: `examples/mindc_mind/main.mind`, 1,084 LOC, compiles itself.
2. **Compiler self-hosts** — the pure-MIND compiler produces MLIR byte-identical
   to the Rust reference on the same input. Proof: bootstrap fixed-point at
   mindc v0.6.1 / v0.6.2 (10,889 bytes / 206 SSA, tag-reproducible).
3. **Toolchain self-hosts** — build orchestration, dependency resolution, and
   test running are themselves written in MIND. RFC 0007 (Mindcraft) closed the
   format/lint/check third-rung rungs. This RFC closes the remaining two:
   `mindc build` and `mindc test`.

Today the build pipeline is owned by cargo and the Rust `mind` crate. The
path from `.mind` source to a linked binary flows through:

- cargo resolves features, selects profile, drives `cc` for the C runtime
  support layer
- `src/project/mod.rs` (`build_project`) orchestrates compile → link via
  shell-outs to `mlir-opt` / `llc` / `clang`
- `test_project` discovers `tests/*.mind` and runs them via the same pipeline
- `Cargo.toml` carries the feature flags (`std-surface`, `mlir-build`, `pkg`,
  etc.) that control which capabilities are compiled in

None of that is wrong. It works. The problem is provenance: a MIND build of a
MIND program has Rust in its dependency chain. A developer auditing a MIND
binary cannot trace it back to pure-MIND source without passing through the
Rust compiler.

RFC 0008 specifies `mindc build` and `mindc test` as pure-MIND orchestrators.
After this RFC lands through Phase G, building the `mind` repo itself with
`mindc build` produces a byte-identical result to the cargo pipeline — and at
that point cargo becomes an optional transition scaffold, not a required tool.

### What cargo currently provides that we are replacing

| Capability | Where in Cargo.toml / mod.rs today | RFC 0008 replacement |
|---|---|---|
| Feature flag resolution | `[features]` table | `[features]` section in Mind.toml (§3) |
| Profile (debug/release) | `[profile.release]`, `[profile.dev]` | `[build] optimize = ...` in Mind.toml (§3) |
| Dependency graph | `[dependencies]` (currently empty for mind users) | `[dependencies]` in Mind.toml (§3) |
| Source collection + compile | `compile_sources` in `src/project/mod.rs` | `mindc build` pipeline (§4) |
| Test discovery + running | `test_project` in `src/project/mod.rs` | `mindc test` (§5) |
| Incremental rebuild | cargo fingerprinting | SHA256 object cache (§4.6) |
| Workspace members | not yet used by mind itself | `[workspace]` in Mind.toml (§3) |

LLVM and MLIR are not being replaced. They remain the codegen backend
throughout. This RFC is about the orchestration layer above the backend — the
plumbing that decides which source files compile in which order, what the dep
graph looks like, and how tests are discovered and reported.

## 2. Non-goals

The following are explicitly out of scope for this RFC and are tracked
separately:

- **Replacing LLVM/MLIR as the codegen backend.** LLVM is a build tool, not a
  language tool. The shell-out to `mlir-opt` / `llc` / `clang` stays. RFC 0010
  (libMLIR FFI in pure MIND) covers the long-term native-MLIR path.
- **Memory safety overhaul / ownership model.** RFC 0010.
- **Async / concurrency runtime.** RFC 0011.
- **A network-aware package registry with semver constraints.** Phase E of this
  RFC introduces git-pinned deps and a lockfile; full semver resolution,
  authentication, and an index server are RFC 0009.
- **Performance parity with cargo.** `mindc build` is not optimised to build
  the Rust mind crate faster than cargo does. It builds `.mind` programs. The
  goal is provenance correctness, not build throughput.
- **`mindc bench`.** The existing bench subcommand is owned by the Rust
  `bench_project` function; it will migrate after `mindc test` ships and the
  test runner proves stable. It is not in this RFC.
- **Windows MSVC support.** The C runtime-support layer already constrains this
  RFC; we inherit the same platform support matrix. Windows MinGW is not
  actively blocked, but is not a first-class target for Phase A–D.

## 3. Manifest schema (Mind.toml extensions)

The current `Mind.toml` schema (defined in `src/project/mod.rs`,
`ProjectManifest`) already carries `[package]`, `[build]`, `[features]`,
`[targets]`, `[dependencies]`, `[profile]`, `[exports]`, and `[mindcraft]`
tables. RFC 0008 extends four of those tables and adds two new ones
(`[workspace]` and `[test]`). All additions default sensibly; existing
`Mind.toml` files parse without modification.

### Full normative schema after RFC 0008

```toml
[package]
name        = "my-project"          # required, string
version     = "0.1.0"               # required, semver string
authors     = ["STARGA Inc."]       # optional, array of strings
license     = "Apache-2.0"          # optional, SPDX identifier string
description = "..."                 # optional
repository  = "https://..."         # optional
homepage    = "https://..."         # optional

[dependencies]
# Path dependency (Phase D)
some-lib = { path = "../some-lib" }
# Path dependency with feature selection (Phase D)
other-lib = { path = "../other-lib", features = ["gpu"] }
# Git dependency pinned to a revision (Phase E)
remote-lib = { git = "https://github.com/...", rev = "abc1234" }
# Local override of a git dep during development (Phase E)
remote-lib = { git = "...", rev = "...", path = "../remote-lib-dev" }

[features]
default = []                        # features enabled when none specified
gpu     = ["some-lib/gpu"]          # feature can activate dep features
simd    = []
full    = ["gpu", "simd"]

[build]
entry    = "src/main.mind"          # default: "src/main.mind"
output   = "app"                    # default: package.name
target   = "cpu"                    # default: "cpu"
emit     = "binary"                 # binary | cdylib | object; default: "binary"
optimize = "debug"                  # debug | release; default: "debug"

[test]
filter   = ""                       # substring filter applied to test names; default: ""
parallel = true                     # run tests in parallel; default: true
threads  = 0                        # 0 = use available parallelism; default: 0
timeout  = 30                       # per-test timeout in seconds; default: 30

[workspace]
members = [                         # paths to workspace member Mind.toml roots
    "crates/core",
    "crates/std",
]
exclude = [                         # paths excluded from workspace operations
    "scratch",
]

# [exports] — unchanged from RFC 0002; preserved verbatim
[exports]
c_abi = ["my_exported_fn"]

# [mindcraft] — unchanged from RFC 0007; preserved verbatim
[mindcraft]
# ...
```

### Field-level validation rules

`[package]`

- `name` — required. Must match `[a-zA-Z][a-zA-Z0-9_-]*`. Validated at
  manifest load; a malformed name is a hard error before any build step runs.
- `version` — required. Validated as a semver-compatible string (major.minor.patch).
  RFC 0009 (package manager) will enforce strict semver; this RFC validates format only.

`[dependencies]`

- A path dep (`{ path = "..." }`) is resolved relative to the manifest file.
  The path must contain a valid `Mind.toml`. Phase D.
- A git dep (`{ git = "...", rev = "..." }`) requires `rev` to be a full or
  abbreviated SHA. Semver version constraints are intentionally absent in this
  RFC (deferred to RFC 0009). Phase E.
- The `features` array activates named features in the dependency's own
  `[features]` table.
- Unknown fields in a dep inline table are a hard error at manifest load time,
  not a warning. Silent unknown-field tolerance in dependency specs causes
  misbuilds in ways that are extremely hard to diagnose.

`[build]`

- `entry` — path relative to the manifest file. Must end in `.mind`. Default
  `src/main.mind`. Error if the file does not exist at build time.
- `output` — output artifact name, without extension. Default: `package.name`.
- `target` — one of `cpu | gpu | tpu | npu | lpu | dpu | fpga | cerebras`.
  Inherits the backend-target vocabulary from the existing CLI. Default `cpu`.
- `emit` — one of `binary | cdylib | object`. Controls the linker invocation.
  Default `binary`. Selecting `cdylib` activates the C-ABI export path
  (RFC 0002 §3, RFC 0003).
- `optimize` — one of `debug | release`. `release` passes `-O3 -flto -fPIC` to
  the clang link step. Default `debug`.

`[test]`

- `filter` — a substring applied to test names at test-discovery time. Empty
  string matches all. Overridden by `--filter` on the CLI.
- `parallel` — boolean. When `true`, test cases run concurrently up to
  `threads`. Default `true`.
- `threads` — unsigned integer. `0` means "use available parallelism" (i.e.
  `std::thread::available_parallelism()` equivalent from MIND's runtime).
  Overridden by `--threads` on the CLI.
- `timeout` — per-test timeout in seconds. A test that does not complete within
  this limit is marked `TIMEOUT` and counted as a failure.

`[workspace]`

- `members` — array of paths relative to the workspace root's `Mind.toml`.
  Each path must contain a `Mind.toml` with a `[package]` section. Processed
  in declaration order for printing; topological sort is applied for builds.
- `exclude` — paths excluded from workspace glob expansions. Useful for
  example directories that have their own `Mind.toml` but should not be treated
  as workspace members.
- Workspace detection: `mindc build` walks parent directories looking for a
  `Mind.toml` with a `[workspace]` table, exactly as `find_project_root` walks
  for `Mind.toml` today. A manifest that is both a workspace root and a package
  (virtual manifests with only `[workspace]` are also valid) follows the same
  pattern as Cargo's workspace layout.

## 4. Build pipeline

The `mindc build` pipeline replaces `build_project` / `compile_sources` /
`link_binary` in `src/project/mod.rs` with a pure-MIND orchestrator. The Rust
functions remain as the fallback path until Phase G; in Phases A–D both paths
are active and CI asserts parity.

### 4.1 Read Mind.toml

Load and validate the manifest. Validation is fail-fast: a malformed `[package]`
section, an unknown `[build].emit` value, or an unresolvable path dep stops the
build before any compilation begins. This is stricter than the current
`load_manifest` which silently ignores unknown fields. The strictness pays off:
configuration errors are surfaced at the earliest possible moment, not after
five minutes of partial compilation.

### 4.2 Resolve the dependency graph

Phases A–C: no external deps; this step is a no-op. The std surface
(`std.vec`, `std.string`, `std.map`, `std.io`, `std.blas`) is always
available and does not appear in `[dependencies]`.

Phase D: walk `[dependencies]` entries that have `path = ...`. For each,
recursively load the dependency's `Mind.toml`, collect its own deps, and build
a directed acyclic graph. Cycle detection is O(V+E) depth-first search;
a detected cycle is a hard error that names the cycle path.

Phase E: for `git = ...` deps, fetch to `~/.mindenv/cache/<sha>/` on first
use. The lockfile (`Mind.lock`, §4.3) pins the resolved revision. Subsequent
builds that find the cached revision skip the network. `mindc fetch` populates
the cache without triggering a build; `mindc fetch --update` re-fetches
to pick up new commits (still requires updating `rev` or `Mind.lock`).

### 4.3 Mind.lock

The lockfile records the exact resolved revision for every dependency in the
graph. Format question addressed in §10; the normative answer is: TOML with a
`version = 1` envelope, one `[[package]]` array entry per dep. Example:

```toml
version = 1

[[package]]
name    = "remote-lib"
git     = "https://github.com/star-ga/remote-lib"
rev     = "abc1234def5678"
sha256  = "e3b0c44298fc1c149afb4c8996fb92427ae41e4649b934ca495991b7852b855"
```

`Mind.lock` is committed to version control for applications and excluded (via
`.gitignore`) for library crates, following the same convention as Cargo. The
sha256 field is a content hash of the fetched archive, not the git object hash,
so the cache entry can be verified offline.

`mindc lock` regenerates `Mind.lock` from the current manifest without
building. It exits non-zero if any git dep cannot be fetched.

### 4.4 Walk modules in topological order

Within a single package, modules are ordered by import graph. A `use` statement
at the top of a `.mind` file creates a dependency edge. The walker resolves
import paths using the same module-table logic RFC 0005 Phase C established
(`STDLIB_MIND_SOURCES` for std, then project sources). Cycles in the import
graph are a hard error.

The topo-walk feeds a pipeline per module:

```
lex → parse → typecheck → emit_ir → codegen
```

This is the existing single-file `compile_source_with_name` pipeline;
`mindc build` calls it in the correct order and accumulates object files.

### 4.5 Linking

After all objects are compiled, `mindc build` invokes the linker via the same
shell-out seam RFC 0003 established. The specific invocation depends on
`[build].emit`:

| emit value | linker invocation |
|---|---|
| `binary` | `clang <objs> -o <output>` |
| `cdylib` | `clang -shared -fPIC <objs> -o <output>.so` |
| `object` | no link step; the object files are the output |

The linker is `clang` by default. `MIND_LINKER` environment variable overrides
it. This matches the existing `link_binary` behaviour; the difference is that
the orchestration is MIND code, not Rust code.

The RFC 0002 `[exports] c_abi` contract is preserved unchanged: when `emit =
"cdylib"`, the list of exported names from `Mind.toml` is passed through the
compile pipeline exactly as `opts_with_exports.manifest_exports` does today
(see `src/project/mod.rs:460-464`). No change to the ABI.

### 4.6 Incremental cache

Phase F. Each object is keyed by:

```
SHA256(source_bytes || feature_flags || dep_hashes || optimize_flag || target)
```

where `dep_hashes` is the recursive SHA256 of all transitive dependency
objects. The cache lives at `target/incremental/<key>.o`. A build hit skips
lex through codegen and goes straight to the link step. The link step is always
re-run when any object is rebuilt.

Cache invalidation is conservative: any change to source bytes, feature flags,
the dependency graph, or the target invalidates the affected object and all
downstream objects that transitively import it. This is more aggressive than
cargo's fingerprinting but simpler to reason about and correct.

`mindc clean` removes `target/`. `mindc clean --all` removes `target/` and
the `~/.mindenv/cache/` entries for this project's deps.

### 4.7 Backend dispatch

The `--target` flag (or `[build].target` in `Mind.toml`) selects the MLIR
codegen path. The vocabulary is the same as the existing `parse_target`
function in `src/bin/mindc.rs`: `cpu | gpu | tpu | npu | lpu | dpu | fpga |
cerebras`. Each target selects the appropriate MLIR lowering pipeline and
passes the corresponding `--target-triple` to `llc`. This RFC does not change
target semantics; it changes who calls the backend dispatch — MIND code
rather than Rust code.

## 5. Test discovery and runner

### 5.1 The `#[test]` attribute

A MIND function marked `#[test]` is a test entry point. The attribute is
parsed by the lexer/parser as a standard MIND attribute (the attribute surface
is already present for `#[allow(...)]` from RFC 0007). Semantics:

- A test function takes no arguments and returns `()` or `Result<(), E>`.
- A function returning `Result` passes if it returns `Ok(())` and fails if it
  returns `Err(e)` — the error message is printed in the failure report.
- A test function that panics (via the `mind_panic` intrinsic) is counted as a
  failure; the panic message is captured.
- Attribute name: `#[test]`. Not `#[mind_test]`. Rationale: the two-character
  shorthand is already the universal convention (Rust, Python `pytest`, Go
  `t.Run`) and the MIND namespace is not threatened by a short built-in
  attribute name. See §12 for the naming decision record.

### 5.2 Test collection at link time

At link time, `mindc test` emits a `__mind_test_registry` section into the
object containing one entry per `#[test]` function: the function pointer
(i64-ABI) and the fully-qualified test name string. A thin runner entry point
iterates the registry, calls each function, captures the result, and reports
pass/fail/timeout/panic.

This is the same pattern Rust's test harness uses (the `#[test]` collector
and `rustc --test`); it is the correct approach for a compiled language.

### 5.3 `mindc test` subcommand

`mindc test` compiles the project with the test harness entry point injected,
links it, and executes the resulting binary. The test binary exits 0 if all
tests pass, 1 if any test failed.

Test isolation: **one process per test** (see §10 for the reasoning).

Each test is spawned as a child process of the runner binary. The child
inherits no mutable state from the runner. A crashed test is detected by a
non-zero exit code and reported as PANIC. The overhead per test is one
`fork`/`exec` pair; on modern Linux this is fast enough for test suites of up
to ~10,000 tests without needing in-process isolation. If the suite is large
enough that process overhead matters, `--no-isolation` opts into in-process
panic isolation (Phase B extension, not Phase A).

Timeout enforcement is implemented in the runner: the parent process delivers
`SIGKILL` to the child after `[test].timeout` seconds.

### 5.4 Output format

The default output matches `cargo test` UX intentionally — developers already
know it, and parity reduces cognitive friction during the transition period:

```
running 7 tests
test utils::parse_empty ... ok
test utils::parse_ident ... ok
test lowering::const_i64 ... ok
test lowering::neg_literal ... ok
test lowering::struct_heap_record ... ok
test lowering::matmul_contract ... ok
test lowering::q16_bit_identity ... FAILED

failures:

---- lowering::q16_bit_identity ----
thread 'lowering::q16_bit_identity' panicked at 'expected 0x0000ffff, got 0x00010000'

test result: FAILED. 6 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
```

`--reporter json` emits a stable machine-readable stream, one JSON object per
test result, schema-compatible with the Mindcraft reporter trait defined in
RFC 0007 §6:

```json
{"type":"test","name":"lowering::q16_bit_identity","result":"failed","message":"expected 0x0000ffff, got 0x00010000"}
```

`--no-capture` passes stdout/stderr from child test processes through to the
terminal, mirroring cargo's `--nocapture`.

`--list` prints all discovered test names and exits 0 without running any test.

## 6. CLI surface

```
mindc build [PATHS...] [--target=<t>] [--release] [--emit=<emit>] [--out=<path>]
mindc test  [PATHS...] [--filter=<substr>] [--no-capture] [--threads=<n>] [--list]
mindc fetch [--update]
mindc lock
mindc clean [--all]
```

### mindc build

| Flag / arg | Type | Default | Description |
|---|---|---|---|
| `PATHS...` | paths | cwd | Source files or project roots. When a directory is given, `mindc build` looks for `Mind.toml` in that directory. |
| `--target=<t>` | string | `Mind.toml [build].target` or `cpu` | Backend target. |
| `--release` | bool | false | Equivalent to `[build] optimize = "release"`. Overrides the manifest. |
| `--emit=<emit>` | string | `Mind.toml [build].emit` or `binary` | Output artifact type: `binary`, `cdylib`, `object`. |
| `--out=<path>` | path | `target/<profile>/<name>` | Output artifact path. Overrides `[build].output`. |
| `--features=<list>` | csv | none | Activate the named features from `[features]`. |
| `--no-default-features` | bool | false | Disable the `[features] default` list. |
| `--verbose` | bool | false | Print each compile + link invocation. |

### mindc test

| Flag / arg | Type | Default | Description |
|---|---|---|---|
| `PATHS...` | paths | cwd | Source files or project roots. |
| `--filter=<substr>` | string | `""` | Run only tests whose name contains this substring. |
| `--no-capture` | bool | false | Do not capture child stdout/stderr; print immediately. |
| `--threads=<n>` | uint | `[test].threads` or 0 | Max parallel test workers. 0 = available parallelism. |
| `--list` | bool | false | Print test names and exit 0 without running. |
| `--reporter=<r>` | string | `human` | `human` or `json`. |
| `--target=<t>` | string | `cpu` | Build target for the test binary. |

### mindc fetch

Resolves all `[dependencies]` entries and populates `~/.mindenv/cache/` without
building anything. `--update` re-fetches git deps even if the pinned revision
is already cached (useful when `Mind.lock` has been updated to a new `rev`).

### mindc lock

Regenerates `Mind.lock` from the current manifest's deps. Exits non-zero if any
dep cannot be resolved. Idempotent: running `mindc lock` on an already-current
lockfile is a no-op.

### mindc clean

Removes `target/`. `--all` additionally removes the `~/.mindenv/cache/` entries
pinned by this project's `Mind.lock`.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. All builds succeeded; all tests passed. |
| `1` | Build failure or test failure. At least one error was fatal. |
| `2` | Invalid usage. Unknown flag, conflicting arguments, or missing required argument. |

These three codes are the only defined exit codes. Pipelines that only check
for 0/non-zero work correctly; pipelines that distinguish build-failure from
usage-error use code 2.

## 7. Phasing

The phasing splits into two tiers: **Phases A–D** are the first cut and ship
together or in close succession. **Phases E–G** iterate after.

### Phase A — single-crate build (no deps) — **Shipped**

`mindc build` for a project with no `[dependencies]` entries. Covers the
compile-sources loop, the link step, and the `--target` / `--release` /
`--emit` flags. Uses the existing `compile_source_with_name` pipeline
internally. The Rust `build_project` function remains active; CI runs both for
byte-identical output parity.

Deliverables:
- `src/build/mod.rs` — the Rust build orchestrator implementing RFC 0008 §4
  Phase A pipeline (single-crate, no deps, full CLI surface)
- `src/project/mod.rs` — `BuildTarget`, `EmitKind`, `OptimizeLevel` enums;
  `TestConfig`, `WorkspaceConfig` structs; `[build]` section extended
- `src/bin/mindc.rs` — `mindc build [PATHS...] [--target|--emit|--release|--optimize|--out]`
  subcommand wired in with RFC 0008 §6 exit code semantics (0/1/2)
- `tests/mindc_build_phase_a.rs` — 23 integration + unit tests, all passing
- Mind.toml validation for `[build]`, `[test]`, `[workspace]` section fields
- Self-build smoke gate: `mindc build examples/mindc_mind/main.mind --emit=cdylib --out=/tmp/mindc_self_build.so` passes

### Phase B — test discovery and runner — **Shipped**

`mindc test` discovery via `[test]`, the in-process panic-isolation runner,
and the human + JSON reporters. `--filter`, `--list`, `--threads`, `--no-capture`.
The pure-MIND binary test runner path (process-per-test, `__mind_test_registry`
linker section) is future work gated on the MLIR compiled binary path.

Deliverables:
- `src/test/mod.rs` — RFC 0008 Phase B test runner (`run_tests`, `discover_tests_in_source`)
- `[test]` attribute parsing in the parser (`is_test: bool` on `Node::FnDef`)
- `mindc test [PATHS...] [--filter] [--list] [--threads] [--no-capture] [--reporter]`
  CLI subcommand in `src/bin/mindc.rs`
- Human and JSON reporters matching §5.4 format spec
- `tests/mindc_test_phase_b.rs` — 18 integration tests, all passing
- `tests/fixtures/test_phase_b_all_pass.mind` + `test_phase_b_one_fail.mind`
- Isolation model: in-process `catch_unwind` (process-per-test deferred to
  when MLIR compiled binary path is available — see §10 decision)

### Phase C — workspace support — **Shipped**

`[workspace]` table parsing, member enumeration, topological build order across
members, and workspace-level `mindc build` / `mindc test` that acts on all
members.

Deliverables:
- `src/workspace/mod.rs` — workspace resolution engine: glob expansion, exclude
  filtering, virtual manifest support, Kahn's topological sort, cycle detection
  (exit code 2), `WorkspaceOpts::filter_members` with BFS transitive-dep closure
- `src/project/mod.rs` — `DependencySpec::Path { path, features }` variant for
  cross-member path dep discovery
- `src/bin/mindc.rs` — `--package`/`-p` and `--workspace` flags on `mindc build`;
  `--package`/`-p` on `mindc test`; `detect_workspace_root`, `run_workspace_build`,
  `run_workspace_test` delegation using `set_current_dir`
- `tests/mindc_workspace_phase_c.rs` — 12 integration tests covering glob
  expansion, toposort, cycle detection, virtual manifests, exclude, canonical
  paths, external path deps, and error exit codes; all 12 passing

### Phase D — path deps — **Shipped**

External `path = ...` deps with `tree_sha256` drift detection and mandatory
`Mind.lock` enforcement (AP-2). The build fails with a clear message if
`Mind.lock` is absent or stale.

Deliverables:
- `src/deps/mod.rs` — `resolve_and_verify_deps`, `run_lock`, `run_fetch`,
  `run_clean`; `compute_tree_sha256` (self-contained FIPS 180-4 SHA-256);
  `MindLock` / `LockEntry` TOML schema; `DepError` with exit codes
- `src/project/mod.rs` — `DependencySpec::Git { git, rev, tag, branch }`
  variant added alongside the existing `Path` variant
- `tests/mindc_deps_phase_de.rs` — 12 Phase D tests covering external path
  deps, drift detection, lock regeneration, --check, --update, AP-2 enforcement

### Phase E — git deps + ~/.mindenv/cache + lockfile — **Shipped**

`git = ...` deps with rev/tag/branch resolution to full 40-char SHA at lock
time (AP-1: URL + rev + tree_sha256 triple). Content-addressed cache at
`~/.mindenv/cache/git/<hostname>/<owner>/<repo>/<sha>/`. Mandatory lockfile
enforcement on `mindc build` (AP-2). `mindc lock`, `mindc fetch`, `mindc clean`
subcommands.

Deliverables:
- `src/deps/mod.rs` — git dep resolution via `git` CLI (`shallow_clone` +
  `full_clone_and_checkout` fallback), `mindenv_cache_root()`,
  `git_cache_dir()`, `fetch_git_dep_into`, `run_fetch`, `run_lock --check`,
  `run_lock --update <pkg>`, `run_clean --cache`, `run_clean --all`
- `src/bin/mindc.rs` — `Lock`, `Fetch`, `Clean` subcommands wired in
- `tests/mindc_deps_phase_de.rs` — Phase E tests covering git rev/branch/tag
  resolution, cache population, `--check` on absent lock, `mindc fetch`
  idempotency, `mindc clean --all`, tree_sha256 stability

### Phase F — incremental cache — **Shipped**

SHA256-keyed object cache (§4.6). Rebuilds only changed modules and their
dependants. Integration test: touch one source file → only that file
recompiles; unchanged modules hit from cache.

Deliverables:
- `src/build/cache.rs` — `module_cache_key`, `probe`, `write_object`,
  `clean_all_caches`, `BuildManifest`, `ObjectMeta`; self-contained FIPS 180-4
  SHA-256; atomic write-via-rename for concurrency safety
- Cache layout: `target/<target>/<optimize>/.cache/{objects,meta}/<sha256>.{o,json}`
  + `manifest.json`; per-target + per-optimize-level isolation (cpu/cerebras/…
  and debug/release never share entries)
- `src/build/mod.rs` — Phase F cache probe integrated into `run_build`;
  `BuildOpts::no_cache`; `IncrementalStats` in `BuildOutput`; cache written
  on every successful compile (even with `--no-cache`)
- `src/bin/mindc.rs` — `mindc build --no-cache` flag; `mindc build --verbose`
  reports `[CACHE HIT] <module> (<key-prefix>)`; `mindc clean --cache` wipes
  `target/*/.cache/` directories leaving linked binaries intact
- `tests/mindc_cache_phase_f.rs` — 13 tests (10 spec + 3 unit), all passing
- Cache key format version `mindc-cache-v1\n` — bump invalidates all users
- Hard-gate results: 13/13 pass; full suite 0 failed; self-build smoke passes;
  bootstrap fixed-point unchanged; warm rebuild ≈3 ms vs cold ≈188 ms (63×)

### Phase G — keystone: bootstrap mind itself with mindc build

`mindc build` builds the `mind` repo itself. This is the keystone milestone.
It means:

- A fresh clone of `mind` can produce a working `mindc` binary using only
  `mindc` (plus the LLVM/MLIR toolchain it already requires) — no Rust
  toolchain.
- `cargo build` continues to work; both pipelines produce byte-identical
  artifacts (CI parity test is the proof).
- After community migration period, cargo is deprecated for the `mind` repo
  itself. The Rust `mind` crate remains available for `cargo install mind`
  users, but the canonical build is `mindc build`.

Phase G does not require RFC 0010 (libMLIR FFI in pure MIND) to land first.
The shell-out to `mlir-opt` / `llc` / `clang` remains; what changes is that
the Rust code calling those shell-outs is replaced by MIND code calling them.
RFC 0010 eliminates the shell-out itself; that is a subsequent step.

## 8. Migration plan

### Phases A–D: side-by-side

During Phases A–D, `cargo build` and `mindc build` coexist. Both are
supported; CI runs both. The parity assertion is:

```
sha256(cargo build artifact) == sha256(mindc build artifact)
```

for a fixed (source, flags, backend) triple. Any divergence fails CI before
merge. This is the same discipline used during the bootstrap fixed-point work
(v0.6.0–v0.6.1): the pure-MIND compiler was proven byte-identical to the Rust
reference before the Rust reference was declared redundant.

### Phase G: cargo deprecated for mind itself

Once Phase G lands:

1. The `Cargo.toml` and the Rust `src/project/mod.rs` build functions are
   marked deprecated in the repo's `CHANGELOG.md`.
2. The `mind` README switches its "build from source" instructions to
   `mindc build`.
3. The GitHub Actions CI removes the `cargo build` parity job and keeps only
   the `mindc build` job.
4. The Rust `mind` crate is not removed — it stays as the `cargo install mind`
   distribution path until RFC 0010 enables full Rust removal.

### Downstream project migration

A project with an existing `Mind.toml` requires no changes to use `mindc
build`. Every new manifest field defaults sensibly. The only migration step is
removing the `cargo build` invocation from the project's build script (if any)
and replacing it with `mindc build`.

Projects that build a `.so` via `--emit-shared` continue to work; Phase A
preserves the RFC 0002 `[exports] c_abi` contract.

## 9. Backwards compatibility

### Existing Mind.toml files

Every new table and field added by this RFC is optional and defaults to a
value that replicates current behaviour:

| New field | Default | Behaviour without it |
|---|---|---|
| `[build].emit` | `"binary"` | same as today |
| `[build].optimize` | `"debug"` | same as today |
| `[build].target` | `"cpu"` | same as today |
| `[test]` table | parallelism on, no filter, 30s timeout | same as test_project defaults |
| `[workspace]` table | absent = single-package project | same as today |
| `[dependencies]` with `path` / `git` | absent = no external deps | same as today |

The existing `[exports]`, `[mindcraft]`, `[profile]`, `[targets]`, and
`[features]` tables are unchanged and parsed by the same `ProjectManifest`
serde path as today.

The `serde(deny_unknown_fields)` attribute on `MindcraftConfig` is not applied
to the new tables; unknown fields in `[build]` and `[test]` are reported as
warnings, not errors, during the transition period (Phases A–D). Once Phase G
lands and `mindc build` is canonical, unknown fields become errors.

### Cargo coexistence

`Cargo.toml` is not modified by this RFC. `cargo build` continues to work. The
two build systems are entirely separate; they share only the `.mind` source
files and the MLIR/LLVM backend tools. There is no `Cargo.toml`-to-`Mind.toml`
migration step at the project level; downstreams choose their own timing.

### RFC 0002 `[exports] c_abi` contract

The `c_abi` list in `[exports]` is threaded to the compile pipeline exactly as
`opts_with_exports.manifest_exports` does today (see `src/project/mod.rs:457–464`).
The contract is: every name in `c_abi` that resolves to a `pub fn` in the entry
module gets a C-ABI wrapper emitted into the object. This is unchanged in all
phases.

## 10. Open questions

### Test isolation model

**Decision: process-per-test (§5.3).** Rationale: MIND does not yet have a
memory-safety model that makes in-process panic isolation reliable. An in-process
runner that catches panics from one test must guarantee that the test's heap
allocations, global state mutations, and partially-executed destructors do not
affect subsequent tests in the same address space. MIND's allocator (RFC 0005
`__mind_alloc`) has no per-test arena and no test-scoped leak detector. Until
RFC 0010 (memory safety) lands, process-per-test is the only isolation model
that is correct without qualification. The overhead (one `fork`/`exec` per test
on Linux) is acceptable for the test suites the MIND ecosystem will have during
this RFC's lifetime. An opt-in `--no-isolation` flag (Phase B) lets projects
trade correctness for speed when they know their tests are side-effect-free.

### Cache invalidation strategy

The SHA256-keyed incremental cache (Phase F) is described in §4.6. The open
question is whether to reuse the `~/.mindenv/cache/` path for both git dep
archives and incremental objects, or to keep them separate. Recommendation:
separate directories — `~/.mindenv/cache/` for fetched dep archives and
`target/incremental/` for compiled objects. The latter is project-local (gitignored),
the former is machine-global (shared across projects using the same dep revision).

### Lockfile format

The normative answer given in §4.3 is TOML with a `version = 1` envelope.
The alternative — a pure-MIND DSL — is attractive from a self-hosting standpoint
but has two practical problems: (1) the lockfile must be parseable before MIND
itself is built (it is read during dep resolution, which precedes compilation),
and (2) a pure-MIND lockfile would require the MIND parser to be available as a
build-time tool on the host, creating a bootstrapping circularity in Phase E.
TOML is the correct format for the lockfile; it matches `Mind.toml` itself and
is already parsed by the `toml` crate in the Rust transition scaffold.

### Test failure reporting

The human reporter (§5.4) and JSON reporter are specified. A third reporter,
LSP-compatible, is deferred to Phase B. It will share the same `ReporterKind`
trait that RFC 0007 Phase 6 defined for `mindc check`, so the diagnostic schema
is already established. The open question is whether test failures use the same
`Diagnostic` struct as lint/check diagnostics or a separate `TestResult` struct.
Recommendation: a separate `TestResult` struct — test failures are not source
diagnostics; they carry different metadata (duration, exit code, panic message).
The JSON output format for `TestResult` is specified in §5.4.

### Cross-target rebuild invalidation

When `--target` changes (e.g. from `cpu` to `cerebras`), the incremental cache
must not reuse objects built for the previous target. The SHA256 key (§4.6)
includes the target string, so different targets hash to different keys. Cross-
target builds produce independent object sets in `target/incremental/`. This is
correct but doubles the cache size for projects that build for multiple targets.
Acceptable: disk is cheap, correctness is mandatory.

### Dep version constraints (semver)

This RFC intentionally defers semver version constraints to RFC 0009. In Phase E,
`rev = "<sha>"` is the only allowed constraint — a full or abbreviated git SHA.
There is no `"^1.2"` or `">=0.4, <1.0"` syntax. This is a deliberate
narrowing: it makes the dep resolver trivial (no constraint-satisfaction step),
it makes builds deterministic by construction (a SHA is immutable), and it means
that the dep graph cannot silently change between builds without a lockfile edit.
The cost is verbosity: upgrading a dep requires updating both the `rev` in
`Mind.toml` and regenerating `Mind.lock`. RFC 0009 removes this friction.

## 11. Relation to other RFCs

**RFC 0002 — pub fn C exports** (Phase A must preserve verbatim): the
`[exports] c_abi = [...]` contract in `Mind.toml` is passed through the build
pipeline unchanged. See §4.5 and §9 for the explicit thread-through.
`src/project/mod.rs:457–464` is the current Rust reference; the MIND build
orchestrator replicates that logic.

**RFC 0003 — cdylib AOT emit**: the `emit = "cdylib"` build option is the
direct successor to RFC 0003's `--emit-shared` flag. Phase A must produce
a `.so` with the same ABI as the existing `emit_shared_if_requested` path.

**RFC 0005 — pure-MIND std surface**: `mindc build` depends on RFC 0005 to
compile any program that imports `std.*`. The std modules are bundled into the
`mindc` binary (Phase C of RFC 0005 landed in mindc v0.4.2) and remain
available transparently. Phase G (bootstrap mind itself) requires the entire
RFC 0005 std surface to be available to the build orchestrator's own source.

**RFC 0006 — mind-blas**: `mindc build` handles `use std.blas` like any other
stdlib import. The BLAS intrinsics (`__mind_blas_*`) are registered in
`STD_SURFACE_INTRINSICS` behind the `std-surface` feature gate; `mindc build`
activates this gate when the project's `[features]` includes `std-surface` or
a dep activates it.

**RFC 0007 — Mindcraft**: `mindc test` shares the project-walker pattern that
`mindc check` (RFC 0007 Phase 5) established in `src/check/mod.rs`. The file
enumeration, `.gitignore` integration, and reporter trait are reused. The
`#[test]` attribute is parsed by the same attribute-handling path that
`#[allow(...)]` uses. There is intentionally no shared code path between test
running and format/lint checking — they have different lifecycles — but the
infrastructure (project root discovery, file walking, reporter output) is shared.

**RFC 0009 — package manager** (planned): Phase E of this RFC is the
prerequisite for RFC 0009. RFC 0009 specifies the full network protocol,
authentication, package index, semver resolution, and publishing workflow. This
RFC provides only the client-side fetch-and-cache primitive that RFC 0009 will
build on. RFC 0009 must not land before Phase E of this RFC is stable.

**RFC 0010 — libMLIR FFI in pure MIND** (planned): Phase G of this RFC does
not require RFC 0010. The shell-out to `mlir-opt` / `llc` / `clang` persists
through Phase G. RFC 0010 eliminates the shell-out itself, completing full Rust
removal. RFC 0010 depends on Phase G having landed (the build system must be
able to compile the FFI bindings using `mindc build` before the Rust scaffold
can be removed from the build path).

**RFC 0011 — async / concurrency** (planned): orthogonal. The parallel test
runner in Phase B uses MIND's process-fork primitive, not an async runtime.
When RFC 0011 lands, the test runner can optionally migrate to async worker
tasks; that is a quality-of-life improvement, not a correctness dependency.

## 12. Decision points

This section records design decisions that were choices between named
alternatives. The decisions are final for this RFC; superseding decisions
require a new RFC or an explicit §9-style backwards-compat annotation.

### Naming: `mindc build` vs `mindc compile` vs `mindc bake`

**Decision: `mindc build`.**

`compile` is accurate for a single-file invocation but misleading for a
multi-file project with a link step — "build" is the conventional term for the
full source-to-artifact pipeline. `bake` is clever but creates an unnecessary
vocabulary mismatch for anyone familiar with any other build system. `build`
is unambiguous, consistent with the existing `mindc build` subcommand that
already exists in `src/bin/mindc.rs` (the Rust implementation this RFC
replaces), and consistent with the CLI surface developers already use today.
No renaming required; this RFC is implementing what the existing stub promises.

### Default target: `cpu`

**Decision: `cpu`.**

The existing `parse_target` function defaults to `cpu` for bare `mindc build`
invocations. A different default would break existing project builds silently.
The `[build].target` field in `Mind.toml` lets projects set their own default
without touching the CLI. `cpu` remains the global default.

### Workspace top-level command behaviour

**Decision: when invoked at a workspace root, `mindc build` builds all
workspace members in topological dep order. `mindc build --package <name>`
limits the build to one member.**

This is the same convention as `cargo build` in a workspace. The alternative
— requiring explicit `--workspace` to build all members — is more verbose for
the common case and requires every project to remember a flag. `--package` is
the explicit narrowing; the default is "build everything I know about."

### Test attribute name: `#[test]` vs `#[mind_test]`

**Decision: `#[test]`.**

`#[mind_test]` adds a namespace that does not need to exist. There is no
conflict risk: MIND attributes are parsed in a separate namespace from MIND
identifiers; a function named `test` can coexist with a `#[test]` attribute
on another function. The short form reduces annotation noise in test files and
matches every other compiled-language convention (Rust, Zig, Odin, Lobster).
The only argument for `#[mind_test]` is namespacing for potential future
attribute registries; that concern is deferred to when an attribute registry
is actually specified (not this RFC's scope).

### `mindc fetch` vs automatic fetch on build

**Decision: automatic fetch on `mindc build`, with `mindc fetch` as the
explicit-only variant.**

`mindc build` automatically fetches any git dep whose `rev` is not in
`~/.mindenv/cache/`. The fetch is logged (`Fetching remote-lib @ abc1234...`).
`mindc fetch` exists for CI pipelines and air-gapped environments that want to
pre-populate the cache before the build step, or that want to run fetch and
build as separate auditable steps. `--offline` on `mindc build` disables all
network access; a missing cache entry in offline mode is a hard error.
