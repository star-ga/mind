# Runtime file-open errors

Author: STARGA Inc.

`__mind_open(path)` returns a nonnegative process-local file descriptor on
success and a negative value on error. Callers must test the sign. The hosted
runtime uses libc's `-1`; a direct Linux syscall can return `-errno`. Neither
exact error integers nor descriptor numbers are portable evidence values.

The runtime at base `92b3dddac9f9f12c1df096657a5ff32b20e1c170`
contained a temporary mutation returning `999` for a failed open. This repair
returns the signed platform result and preserves the existing null guard,
read-only operation and successful descriptor. A later failing read can mask
the defect, so the regression checks the open result directly.

The MIND test checks null, a missing path and a non-directory intermediate
component, then reads a real one-byte file and its EOF. It needs an owned
fixture directory. It does not modify caller files or test descriptor cleanup;
the missing general close intrinsic remains a separate obligation.

## Reproduce

Build this compiler source with `mlir-build cross-module-imports` enabled and
configure the supported CPU/MLIR tools. In an admitted, isolated environment
with one CPU, a 2 GiB memory/address-space cap and no swap, run:

```sh
MINDC=/absolute/path/to/mindc \
  prlimit --as=2147483648 --core=0 -- \
  bash scripts/open_error_regression.sh /absolute/new/evidence-directory
```

The script only prepares fixtures, invokes builds and compares frozen process
results. All executable assertions are in `tests/open_error_kat.mind`.
Both optimization profiles must produce exit `0`, stdout `0\n` and empty stderr.
A timeout, crash, missing tool or build failure is a failed regression.

## Observed narrow verification

On Linux x86-64 with LLVM/MLIR 20, the unchanged MIND test returned ordinary
exit `1` with stdout `2\n` and empty stderr against the faulty baseline. The
isolated repaired runtime produced exit `0`, stdout `0\n` and empty stderr
for both `debug` and `release` builds. Their executable SHA-256 was
`f80473b64a6b1b7935cfe82fa9d596a6bbaf55eec58bb4568cfc86b890eb90fc`.

The repaired test compiler SHA-256 was
`8983d5d1a3a2cd9727d021eb6aced1abdf15398fb392aba07d95215e886d5ad7`;
the source had the functional repair before this explanatory comment update.
This identifies the observed candidate, not a consumer-approved compiler pin.
The combined imported-library/runtime regression batch recorded all 12 expected
outcomes, 47,632,384 bytes peak cgroup memory, no OOM events and no swap use.
The baseline and repaired compiler differ in other unrelated experimental
stdlib work, so this comparison is not a full compiler equivalence claim.

A separate source review found no blocking defect in the signed result repair
or test, and requested the explicit cross-backend error-code qualification above.
Repository format, no-feature build/test/lint/documentation, claims, attribution,
JSON-evidence hygiene, gate-wiring mutation tests and the pinned cargo-deny
0.20.2 all-feature policy check passed. This gate batch peaked at
2,067,869,696 bytes under the 2 GiB cap, with zero memory events and no swap;
it does not demonstrate generous build-memory headroom.

Non-Linux execution, direct-native qualification, consumer pin qualification and
descriptor-leak repair are outside this narrow result.
