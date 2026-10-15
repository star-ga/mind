# Source-lowered stream I/O candidate

Author: STARGA Inc.

This change connects actual MIND `__mind_read` and `__mind_write` calls to the
checked libc I/O bundle through `nsi_compile`. It does not enable the production
native CLI, qualify a compiler pin, or close the downstream consumer gates.
The source semantics and lifetime boundary are documented in
[native_source_image.md](native_source_image.md).

## Reproduction

Use independently identified native and official CPU/MLIR compiler instruments.
The commands below require their SHA-256 values; the available development
instruments are not interchangeable with a released or qualified pin. Run the
build and tests in an owned one-CPU service with aggregate memory 2 GiB, swap
zero and process address space 2 GiB. The scripts additionally bound child CPU
and wall time. Set LLVM tool paths and `MIND_STDLIB_PATH` explicitly to the
reviewed build configuration.

```sh
python3 examples/mindc_mind/self_host_native_emission_context_smoke.py \
  --enumerate --evidence "$BUILD_EVIDENCE"
python3 examples/mindc_mind/self_host_native_emission_context_smoke.py \
  --build-only --evidence "$BUILD_EVIDENCE" \
  --seed "$NATIVE_COMPILER" --seed-sha256 "$NATIVE_COMPILER_SHA256" \
  --mindc "$CPU_COMPILER" --mindc-sha256 "$CPU_COMPILER_SHA256" \
  --main-sha256 "$MAIN_SHA256" --context-sha256 "$CONTEXT_SHA256" \
  --source-image-sha256 "$SOURCE_IMAGE_SHA256"
python3 examples/mindc_mind/native_stream_io_check.py \
  --native "$BUILD_EVIDENCE/context.native" --native-sha256 "$DRIVER_NATIVE_SHA256" \
  --cpu "$BUILD_EVIDENCE/mlir/context.mlir" --cpu-sha256 "$DRIVER_CPU_SHA256" \
  --prefix "$BUILD_EVIDENCE/prefix.mind" --prefix-sha256 "$PREFIX_SHA256" \
  --cases examples/mindc_mind/native_stream_io_cases_v3.json \
  --cases-sha256 ebe6c9e6fdc0310ec98f7886e62ef8802d37b8e981451d91eec29e939e9a8ca5 \
  --reference "$PREINTEGRATION_DRIVER" --reference-sha256 "$PREINTEGRATION_SHA256" \
  --evidence "$RUN_EVIDENCE"
```

The optional historical reference compares the **same complete packet**, not
outputs of differently named or differently hashed sources. Omitting it does
not establish historical byte preservation. Both diagnostic builds use the
same ordered captured standard-library prefix. The current development prefix
has SHA-256 `12aabf233e961756c0e9f3d4ae7fa40e0427c5103cfee65b92db9be2a530d8a9`;
it differs from the retained historical seed manifest header. This observation
does not authorize a seed update or validate the current source using that old
header. The complete source, tool, runtime and executable closures belong in
the build evidence.

`--build-only` executes no catalogue. To run the original context/component
contract, enumerate a fresh evidence directory and omit that option. Its
original refusal expectations remain unchanged, including old source-image
stream refusals now superseded by the explicitly versioned stream admission.
Schema-1 component stream refusals still apply. Retained failures and separate
new acceptance tests must be reported independently.

## Retained test corrections

The original 37-source catalogue remains byte-for-byte intact. Its first run
recorded 352/448 expected observations; all 148 native/CPU output pairs matched.
The version-2 catalogue recorded 524/532 expected observations and all 176 output
pairs matched. Neither is an all-green contract verdict.
The separately corrected version-3 catalogue passed all 532 observations across
44 sources and 176 packet profiles, including every native/CPU output pair and
four same-packet historical no-I/O comparisons. Its 572 process invocations
completed under the 2 GiB limits with peak aggregate memory 166,891,520 bytes
and zero OOM events. This does not rewrite either earlier contract result.

The correction ledger includes exact before/after sources and hashes:

- The initial integer test buffer encoded `hI\n` instead of the intended `hi\n`.
  Version 2 corrects that source constant without changing expected output.
- Bare bool literals and integer constants follow existing frontend folding.
  Eight exact original sources were separately compiled and executed through
  the official MLIR CPU reference; each zero-count call returned zero with empty
  streams. This does not authorize typed bool or unsigned variables.
- The helper name `send` collides with a standard-library libc declaration.
  Version 3 changes only its two identifier occurrences to `sio_i32_send`.
  The original normal refusal and an inverse-rename byte check are preserved.
  Independent runtime checks established the renamed helper's expected exit
  and byte stream without changing compiler code.

## Actual source mutations

`native_stream_io_mutations.json` records the six edits frozen before their
execution. Each replacement must match exactly once in the captured diagnostic
MIND source; build the changed source in a new isolated CPU/MLIR project with
the same captured standard library and compiler identity. Replay the named
version-2 source through that newly compiled checker and the unchanged checker
under both geometries and routes. Never modify the baseline source or artifacts.
The executable fixture logic is MIND; process transport is diagnostic only.

`native_stream_io_mutation_observations.json` records source/executable hashes
and the original targeted verdicts. All six changed programs compiled normally.
Four edits produced the expected normal false acceptance or wrong runtime
outcome. Removing the float guard remained undetected because the independent
source-domain guard still rejected that input. The import-index modulo edit
was also undetected by its predeclared runtime-specific detector: a separate
layout guard normally refused it with diagnostic `[6,19,0,0]` before emitting
an image. This fail-closed observation is reported separately; it does not
rewrite the original targeted verdict. No crash or failed build was counted
as fault detection.

## Independent scope

Separate prelisted execution observed 120 paired packet profiles, 176 actual
MIND image runs, 12 same-packet pre-integration comparisons and 14 repeated
emissions with exact results. A separate 14-source parameter/namespace suite
observed 56 paired profiles, 48 runtimes, 64 source refusals and eight repeats.
These cover file and pipe behavior, positioned versus sequential offsets,
short operations, descriptor narrowing, refusal boundaries and exact output
bytes. The implementation and independent executions run on one host; they
do not establish cross-hardware determinism or a Criterion performance result.

## Whole-compiler and retained-context results

Three fresh bounded native stages are byte-identical, with SHA-256
`03e956bdfe049d43d128a387264118c2623916be487868268ec4892af57c093a`.
All 32 unchanged static byte/exit comparisons match, including two retained
baseline refusals. A fresh CPU oracle passes the original whole-source MIC
comparison (850,451 bytes), eight exact record regressions, 28 scope checks
(25 executed live) and all 46 source-note controls. The source-note gate uses
the already documented 300-second initial-build allowance; fixture limits and
expected bytes are unchanged.

Native and fresh CPU self-emission also produce identical default-layout ELF
bytes, SHA-256
`f52025bfefa387616af268952794321c5dd73ab5b3b7143032426c70ca869b9c`.
That default-layout artifact was compared only; the executed fixed point uses
the bounded layout. The CPU self-emission process exited normally with empty
stderr under the recorded allocator checks.

A final fresh diagnostic rebuild reproduces both build2 executables exactly.
Replaying the unchanged context catalogues records 662/674 component outcomes,
216/240 original runtime outcomes and 388/400 pointer outcomes. Its overall
verdict remains **FAIL**, with all 40 prior failure identities retained and
eight additional original stream-refusal expectations superseded by this
explicitly versioned feature. No failed expectation was removed or relabeled.
The previous namespace/policy corrections and unsupported pointer-positive
cases remain separately documented. Peak aggregate memory for this replay was
912,809,984 bytes, with zero OOM events.

Source-level TLS errno remains unobserved because the admitted source API has
no errno accessor. Component-level NIO errno evidence is separate. Historical
seed-parity failure, independent open review obligations, production
source-driver/seed/profile integration and downstream qualification remain
required before a qualified compiler delivery. Criterion measurements are a
separate gate; no performance result or threshold adjustment is claimed here.
