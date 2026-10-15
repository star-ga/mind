# Captured compiler source bundles

Author: STARGA Inc.

`_selfhost_sources.py` provides internal bootstrap orchestration for composing
independently maintained MIND compiler components. It does not implement language
semantics, replace the project resolver, enable native features or qualify a
compiler pin. Component implementations remain in their original MIND files.

## Capture and views

`capture_bundle(repo, components=(), driver=...)` reads each declared file once.
The checkout must remain immutable during capture; this is not an atomic
filesystem snapshot or an untrusted-directory sandbox. Components are an explicit
ordered tuple of normalized, compiler-local `.mind` paths. Missing files,
non-UTF8 source, duplicate paths/inodes, symlinks and unsupported paths or manifest
source-selection options are refused. No compiler-component directory crawl,
topological reordering or import inference occurs.

The immutable `SourceBundle` exposes:

| Member | Contract |
|---|---|
| `library_bytes` | Original `main.mind`, then each component, separated by exactly one newline |
| `native_bytes` | Historical ordered std prefix, library view, one newline, native driver |
| `user_lo` | Exact std-prefix byte length |
| `stdin_image` | Signed little-endian64-bit user_lo and source length, followed by native bytes |
| `spans(native=...)` | Source intervals excluding added separators; each slices the original captured bytes |
| `identity` | SHA256 of a versioned, length-delimited ordered source closure, including manifest, all captured std files and driver |
| `materialize_cpu(directory)` | Captured manifest/std and library entry in a new empty owned directory |

The native driver defines `main` and is deliberately excluded from the CPU
library. An empty component list preserves the original library and native seed
bytes exactly. `self_host_loop_smoke.build_seed()` retains its three-value API
and accepts an explicitly supplied bundle; the default captures the current
checkout with no extra components.

Supplying a component bundle does not automatically rebind the LOOP gate's
global oracle resolver. Its caller must pair that bundle with
`build_captured_so` for the corresponding oracle. Existing `resolve_so()` callers
keep their historical policy; this change does not claim that every old build
path now consumes an immutable bundle.

Materialization uses captured bytes only. Original files can change or disappear
after capture without changing either view. UTF8, CRLF and trailing-newline bytes
are retained exactly. `SourceBundle.identity` describes the ordered source inputs;
raw hashes of the library and native views are separate. None is a mathematical
proof or signer authentication. The compiler's existing source-note computation
is unchanged and receives the actual composed native bytes.

This internal flat compiler-library view does not alter ordinary project
semantics. General projects still use `ProjectScope`, captured source ownership,
existing export rules and native entry-first/canonical module order. Duplicate
names, missing calls, ABI mismatches and unsupported syntax still require their
existing compiler checks; a source-bundle hash cannot authorize them.

## Captured CPU oracle

`_selfhost_so.build_captured_so(bundle, compiler, compiler_sha256, cache_dir, ...)`
is a strict opt-in path beside the unchanged `resolve_so()` policy. It checks the
compiler hash before execution, builds from the materialized library view with
the captured std directory, retains the command/streams/exit, and checks that the
owned input copies and executable did not change during the build. It does not
substitute `MINDC_SO` or use an mtime-only freshness fallback.

Fresh compilation is the default. Optional `reuse=True` requires an explicit
`dependency_files` tuple: the caller must supply the complete pinned CPU toolchain
file closure and keep its configuration fixed. Actual dependency bytes, compiler
bytes, ordered source identity and fixed build command enter the cache key. A
cache hit additionally verifies the artifact hash. This is not automatic
toolchain dependency discovery; omit reuse when that obligation is unresolved.
The owner-controlled cache is not an authentication boundary.

The CPU shell embeds its C reference shim. Updating that source requires a fresh
shell build; this helper does not retrofit old executables. Native and CPU builds
share the captured main/component bytes while retaining their documented runtime
differences. The reference shim does not become native compiler authority.

## Reproduction and limits

Run the filesystem/view/cache controls alone explicitly:

```sh
python3 examples/mindc_mind/selfhost_source_bundle_check.py --unit-only
```

The full gate requires exact compiler identities and uses an unchanged MIND
layout-planner component. Its MIND driver exercises the same four arithmetic
boundary results checked through the captured CPU library. Unit cache tests use
a deliberately synthetic compiler/artifact solely to test cache mechanics; the
full gate separately requires real native and CPU builds and execution.
Three actual helper-source mutations remove components, reread original files,
or put the driver into the library. Each must cause its designated assertion to
fail normally; syntax errors, unrelated errors, signals and timeouts do not
qualify as mutation detection.

```sh
python3 examples/mindc_mind/selfhost_source_bundle_check.py \
  --evidence /absolute/owned/evidence-directory \
  --seed /absolute/reviewed/native-compiler.elf --seed-sha256 ACTUAL_SHA256 \
  --mindc /absolute/rebuilt/mindc --mindc-sha256 ACTUAL_SHA256
```

Use the supported Linux CPU configuration, one CPU and a 2GiB/no-swap aggregate
memory cap. The full gate needs no network, credentials or consumer repositories.
Compiler paths must match the supplied hashes; a build failure, crash or timeout
cannot count as successful component execution. The gate does not reseed, change
the frozen compiler manifest or resolve a previously known seed mismatch.

Production integration still requires reviewed symbol/ABI bindings, whole-body
closure, all unchanged compiler regression gates and the consumer qualification
contract. Composing source files alone satisfies none of those obligations.
