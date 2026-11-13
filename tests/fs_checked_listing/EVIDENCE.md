# Checked directory-listing qualification

Author: STARGA Inc.

The additive module and its test contract are candidates. The MLIR CPU route
passed the checks below on 2026-10-22. Direct-native qualification is blocked.
This evidence does not replace a compiler release gate or qualify parser
compaction consumers.

## Exact sources and compiler

- Module SHA-256:
  `338467cc428802e0d5a3147e90c9c98f876fcf9645761da1904e751bce7cc7c5`.
  The file first shipped as that source after one `mindc fmt` pass (SHA-256
  `ee25ea43259b3b54fdb54ecbcfc5d24cf7495d0b8f28f204818a97356ec91d45`):
  statement-final `;` at block ends dropped and wrapped signatures and
  expressions joined, nothing else. The runs below were made on the
  pre-format bytes. The shipped file has changed since then (see the last
  section).
- Compiler source base:
  `93238e8ef2ae014a79a51b9f23ef0d90e063b2a9`, plus this isolated module and
  bundled-source/substrate-link registrations. This is an uncommitted
  development candidate, not an accepted compiler pin.
- Compiler executable SHA-256:
  `4530087ae328f3196e473b48cd5232a9ff198cb839265fe15c65b28d5a0fc12c`.
- Build: `cargo build --locked --offline --features 'mlir-build cross-module-imports' --bin mindc`.
- Rust 1.99.1; LLVM 20.1.2; Linux x86-64 CPU configuration. The retained lock
  file SHA-256 is
  `59dde284eed3b564f0aadce0cbea57c44d107b53ff0926f7ab8aeee0eb3b747f`;
  that lock was supplied by the frozen build packet, not claimed as a tracked
  file in the source base.

The original module, 36 unit cases, 36 fault cases, seven source mutations and
their expected outcomes were not changed to obtain passing results. A real
`use std.fs_checked` fixture was added. It first failed to link because the
module was missing from the shared substrate membership list. Adding that
membership made the unchanged import fixture pass.

## Observed execution

Each project was built with:

```text
mindc build --backend mlir --target cpu --emit binary --optimize debug --no-cache --out program
```

All 18 programs built and produced their predeclared process outcomes:

| Program group | Programs | Observed result |
|---|---:|---|
| Original unit assertions | 1 | Exit 0, stdout `0` plus LF, empty stderr |
| Public embedded import | 1 | Exit 0, stdout `0` plus LF, empty stderr |
| Real filesystem and 10,000 listing/free cycles | 1 | Exit 0, stdout `0` plus LF, empty stderr |
| Synthetic raw-error/allocation/dirent controls | 1 | Exit 0, stdout `0` plus LF, empty stderr |
| Mutation positive controls | 7 | Exit 0 and exact expected streams |
| Actual source mutations | 7 | Exit 1 and exact predeclared numeric diagnostics |

Every case ran under a 2 GiB address-space and cgroup limit, no swap, one CPU,
NOFILE 1024, and bounded build/run times. The maximum observed cgroup memory
peak was 53,555,200 bytes; all recorded memory-event counters were zero.
No crash or build failure counted as a detected mutation.

The retained result archive SHA-256 is
`1ab55bdb8eda2737fe6ee500fc60e73c59a5e2e5d14b58a44a6814a81566483f`.
It records exact source assemblies, commands, compiler/runtime dependencies,
program hashes, captured IR, streams, exit codes and resource receipts. The
archive also holds private build metadata; review it before distributing it.

## Direct-native limitation

The same frozen tests were attempted with `--backend native --native-profile
bounded`, using the compiler's existing embedded seed without an override.
The first unit build exited 3 with `lane.source.print` at `fct_expect`; no
artifact was written. The remaining 17 native cases were not run. The native
profile explicitly refuses printing because its runtime-call support is not
qualified. Removing that refusal is not a valid fix by itself.

The retained native refusal archive SHA-256 is
`6efa61e10cf2e55adf19affd4994ee4b0cbe3af6441b86b9f4ac9b5e6f3426e1`.
Both-backend acceptance, consumer failure propagation, final repository gates
and compiler-pin promotion remain open.

## Scoped repository checks

The following checks passed on the same implementation and registrations:

- `cargo fmt --check`.
- `cargo test --locked --offline --no-default-features --features cross-module-imports --lib project::stdlib::tests -- --test-threads=1`: six passed.
- `cargo clippy --locked --offline --no-default-features -- -D warnings`.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --no-default-features --document-private-items`.
- `python3.12 -B examples/mindc_mind/stdlib_manifest_lint.py` (repository metadata check only).

The first test-binary build, with the repository's one code-generation unit,
failed with a Rust/LLVM allocation error under the 2 GiB address-space cap.
Its receipt is retained, with archive SHA-256
`7f031eff00932dff91e937865e8cda953d048215569ab1d77a29f8857cda9af3`.
The successful test build used `CARGO_PROFILE_TEST_CODEGEN_UNITS=16`; the cap
remained 2 GiB. Its peak was 1,479,876,608 bytes, with zero recorded OOM events.
This changes host test compilation, not MIND product semantics or the pinned
compiler. The successful scoped-check archive SHA-256 is
`7d95635d6743a6fde5196295733e87ad961a533c3095fdaa8c88926ff4d19ede`.

These are scoped checks, not the full release, bootstrap, cross-substrate or
performance gates. The native refusal and consumer obligations remain open.

## Continuous integration (0.10.3)

The `std_mind_programs` job in `.github/workflows/ci.yml` now builds
and runs all 18 programs with the MLIR CPU route and requires 18 exact
outcomes. Before this, no workflow, script or Rust test ran this directory.

The module later moved its three raw syscalls into `std.posix_sys` wrappers
(commit `4687e3a4`). Three INSTRUMENTATION.json needles still named the raw
calls, so they matched zero times and the fault programs could not be made.
These needles now name `posix_sys_openat`, `posix_sys_getdents64` and
`posix_sys_close`, which are the same three call sites. The replacement text,
the counts and every expected outcome are unchanged.

INSTRUMENTATION.json and MUTATIONS.json no longer carry SHA-256 fields. They
recorded the 2026-10-22 source, instrumented source, mutant and per-case main
bytes. The module has changed since then, and the per-case mains were never in
the repository, so no check could compare them. A recorded hash that nothing
checks is not evidence. The CI step instead requires each needle and each
mutation to match its declared number of times in the current module, and it
requires the declared outcome of every program.
