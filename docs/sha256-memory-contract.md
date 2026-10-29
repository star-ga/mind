# SHA-256 memory and failure contract

Author: STARGA Inc.

The previous `std.sha256` implementation retained a fresh 512-byte schedule
for each message block, a padded message copy and hash state. Repeated hashing
could exhaust memory, and allocation failures were not checked. The raw entry
now uses one 704-byte temporary allocation, reused for all blocks and freed
before successful return. Complete input blocks are consumed directly; only
the final one or two padded blocks are copied.

## APIs and ownership

| Entry | Success | Failure |
| --- | --- | --- |
| `sha256_checked(input, length, output)` | Returns zero after writing exactly 32 bytes. | Returns `-1` for invalid signed length/address arithmetic or `-2` for temporary allocation failure. Output remains unchanged. |
| `sha256(input, length, output)` | Existing zero-returning success ABI. | Invokes the existing nonreturning runtime trap. Ignoring a return value cannot admit an uninitialized digest. |
| `hash(bytes)` | Returns an owned 32-byte allocation; caller frees it. | Returns zero after freeing any successfully allocated temporary/output storage. |

Raw input may overlap output: output writes occur only after every input byte
has been consumed. Zero-length input may have address zero. Nonempty input and
the output must be live caller-owned buffers with the declared capacities.
Integer bounds are checked; these checks do not establish pointer ownership.
The ergonomic `bytes` wrapper checks its stride-eight record arithmetic and
copies the input into contiguous temporary storage. Its total temporary memory
is therefore linear in input length; the constant bound applies to the raw
hash operation, not to that conversion.

The legacy entry uses `__mind_assert_fail` directly because callers such as
`std.io_canon` historically ignore the hash return value. A configurable test
assertion is insufficient. The existing one-argument runtime intrinsic is
registered in the std surface, preserving its prior determinism classification.
Backends that cannot execute it must refuse the program. This change does not
establish direct-native, interpreter or cross-platform execution support.

## Reproduction

Build the compiler from the tested source with `mlir-build cross-module-imports`
enabled. Configure its supported MLIR CPU runtime and tools. In an isolated
Linux cgroup with one CPU, exactly 2 GiB memory, no swap and a matching 2 GiB
address-space limit, run:

```sh
MINDC=/absolute/path/to/mindc \
  prlimit --as=2147483648 --core=0 -- \
  bash scripts/sha256_memory_regression.sh /absolute/new/evidence-directory
```

The script checks the resource contract, freezes sources and expected outputs
before execution, and records compiler/executable hashes, build results,
stdout, stderr, exits and cgroup events. Executable assertions, allocation-fault
observers and the legacy trap parent observer are MIND programs. Host tooling
only assembles variants, invokes the compiler and compares frozen byte results.

The suite assembles the exact checked-out std source for mutation and fault
injection. It does not accidentally import a different bundled copy. Changes
to a checked-out std file alone do not update an installed compiler's embedded
stdlib; importing `std.sha256` requires rebuilding that compiler.

| Cases | Required observation |
| --- | --- |
| KAT | 17 fixed digests through 1 MiB, padding boundaries, bounds, output guards, overlapping input/output, wrapper and legacy success. |
| RPT | 100,000 one-byte hashes under an independently checked 32 MiB address-space cap. |
| F01–F04 | Forced failure at raw scratch, wrapper input, wrapper output and wrapper inner scratch sites; ordinary checked failure, unchanged guarded output where applicable. |
| M01–M02 | Removing scratch cleanup or corrupting initial state produces the designated ordinary assertion failure. |
| L01–L02 | Legacy raw/hash-anchor positive controls return normally. |
| L03–L04 | Faulted legacy child reaches SIGABRT; its independent parent returns the expected ordinary test verdict. |
| L05–L06 | Removing the trap produces a false success caught by that parent as the designated ordinary failure. |

Unrelated crashes, build failures and timeouts never count as mutation detection.
The legacy fault controls specifically observe the required child signal;
they do not turn arbitrary abnormal termination into a passing test.

## Observed evidence and limits

The final formatted source passed all 14 declared cases on Linux x86-64,
LLVM/MLIR 20, using compiler SHA-256
`5e1ccd1b6015a18ef39066d6776ee1ae2ef81700d56d1a9d1ebe615e83350a65`.
Peak cgroup memory was 33,562,624 bytes, with no OOM events or swap use.
An additional host independently exercised the compiler's embedded std imports:
known-answer checks and legacy/raw hash-anchor positives in both debug and
release modes, a specifically observed invalid-input legacy trap in both modes,
and 100,000 release-mode hashes under 32 MiB. All nine imported-library calls
matched their frozen expected results. Debug/release executable pairs matched
within this profile; this is not a universal cross-backend determinism claim.

The observed compiler was built from base
`92b3dddac9f9f12c1df096657a5ff32b20e1c170` plus the std and intrinsic-registration
changes before the final formatter pass. Standalone tests bind the final source;
import tests bind that compiler's earlier source-equivalent embedded copy.
Formatting/documentation differences and build paths are not claimed to reproduce
the same compiler binary. The candidate identity is not an accepted consumer pin.

The original accepted compiler refused the direct trap with E2024 followed by
E5005. That refusal is retained as a failed build; it is not included in the
14-case success count. The correction is in the compiler's intrinsic registry,
not a precision downgrade or an alternate product runtime.

Repository format, no-feature build/test/lint/documentation, claims, attribution,
JSON-evidence hygiene, gate-wiring mutation tests and the pinned cargo-deny
0.20.2 all-feature policy check passed. The first gate run caught an oversized
intrinsic source module and the newly added regression runner exceeding the
script-count ratchet; the final source stays within the original 800-line
module limit and explicitly accounts for its one added runner.

An additional 15-test std-surface/MLIR intrinsic suite exhausted the 2 GiB
allocation limit while compiling with the repository default test profile
(codegen-units=1). No kernel OOM kill occurred. A separately recorded run with
`CARGO_PROFILE_TEST_CODEGEN_UNITS=16` passed all 15 tests under the same limits,
peaking at 1,037,365,248 bytes with zero memory events. This is a test-only
compilation accommodation matching the repository dev profile, not a change to
the production compiler pin or a pass for the failed default-profile run.

These finite tests and a separate source review support the stated memory and
failure behavior. Consumer pin qualification and broader cryptographic
validation remain separate requirements. This work makes no
mathematical proof, research-quality or whole-application release claim.
