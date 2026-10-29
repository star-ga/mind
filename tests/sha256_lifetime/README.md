# SHA-256 lifetime and refusal controls

Author: STARGA Inc.

> **Superseded contract.** These controls were written for the reviewed source
> `a132ee35:std/sha256.mind` (sha256 `95f170a1…`): three scratch allocations per
> call and raw refusals that return `1` (invalid range) or `2` (allocation
> refusal). #289 replaced that contract in `std/sha256.mind`: one reused 704-byte
> scratch, `sha256_checked` returning `-1`/`-2`, and a legacy `sha256` entry that
> traps instead of returning. The current contract and its suite are
> `docs/sha256-memory-contract.md`, `scripts/sha256_memory_regression.sh` and
> `tests/sha256_{memory,allocation,legacy,repeated}_kat.mind`.
> `assemble.sh` refuses (exit 65) any subject whose sha256 is not a pinned
> `subject_sha256` in `cases.json`, so the current `std/sha256.mind` and patches
> applied to it are rejected; the suite still reproduces the a132ee35 evidence
> from `git show a132ee35:std/sha256.mind`. Porting it to the #289 contract would
> duplicate that suite's coverage and is not planned.

The executable assertions are MIND. `assemble.sh` only composes sources;
`cases.json` records the prelisted controls and actual source mutations.
Compilation errors, crashes and timeouts never count as mutation detection.

| Profile | Scope |
| --- | --- |
| `real` | 32 raw/wrapper digest, range, canary and repeat controls with the real allocator |
| `observed` | All 44 controls, adding exact allocation/free accounting and each allocation-refusal path |
| `raw` | The 23 raw-function controls; wrapper functions are outside the native emission roots |
| `raw-observed` | 29 raw-function controls with allocation observation and refusal injection |

The raw profiles are additive diagnostics. They do not replace the full contract
or demonstrate native support for the growable-bytes wrapper. Overlapping
profiles must not be added together as independent case coverage.

Controls cover empty input, `abc`, the standard 56-byte vector, padding boundaries,
all 256 byte values, million-a, 4 MiB, repeated calls, untouched inputs/output
canaries, signed range checks, every raw/wrapper allocation refusal and digest
ownership transfer. Fixed vector digests were cross-checked using two external
SHA-256 implementations during preparation; runtime comparisons occur in MIND.

The observer substitutes only the subject's allocation/free call sites and
threads an explicit state argument through its raw/wrapper functions. It poisons
scratch, checks allocation guards, tracks live ownership and injects one selected
null allocation. Successful allocations still delegate to the real allocator.
This checks source ownership and null-return handling; it is not a proof of
allocator reclamation, real OOM behavior or cryptographic security. Caller
pointer liveness and object extents remain API preconditions.

Assembly gives every subject function a reversible `st_subject_` prefix to avoid
accidentally testing a bundled std.sha256. The real-allocator profile makes no
other change to function bodies. The observer profile additionally makes the
documented call substitutions. `assemble.sh` reproduces all four executed
baseline source files byte for byte.

## Reproduction

Use the reviewed subject source (`a132ee35:std/sha256.mind`, not the current
file) and an explicitly selected compiler. For example:

```sh
git show a132ee35:std/sha256.mind > /tmp/sha-subject.mind
bash tests/sha256_lifetime/assemble.sh /tmp/sha-subject.mind observed /tmp/sha-observed
cd /tmp/sha-observed
"$MINDC_BIN" build main.mind --backend mlir --target cpu --emit binary \
  --optimize debug --no-cache --out program
./program
```

A passing baseline exits 0 with empty stdout/stderr. Each mutant exits its
specific failing case number from `cases.json`, also with empty streams. Copy
the reviewed subject to a fresh temporary directory, apply one `mutants/Mxx.patch`
there, and assemble that copy with `observed`; never patch the working product
in place. Patches are tied to the reviewed source, so a failed patch application
requires review rather than a fuzzy rewrite.

Run builds and programs serially under a 2 GiB address-space **and aggregate**
memory limit, no swap and one allocated CPU. Record compiler/source/executable
hashes, commands, outputs and exit codes. Do not infer acceptance from an absent
compiler or skipped execution.

For the diagnostic native compiler, use its documented MINDLAY v1 transport:
six little-endian i64 words `[magic, arena, stack, guard, user_lo, source_len]`
followed by source bytes. The tested sizes were 805306368/402653184/1048576.
Source is the exact captured std prefix plus `prelude.mind` plus `entry.mind`.
For the raw profiles, `user_lo` is the length of std prefix plus test prelude;
only the raw entry is an emission root. Preserve the exact prefix and packet,
repeat emission and compare complete ELF bytes. A diagnostic candidate is not
a qualified consumer pin.

## Observed scope

The pinned MLIR/CPU compiler (`a1444e44…`) passed the full 32/44 profiles and all
eight source mutations. The diagnostic native candidate (`03e956bd…`) refused
the full drivers with an unsupported-construct error. The additive 23/29 raw
profiles passed on both backends; six raw-function mutations produced the
expected normal failures on both. Repeated raw native emissions were identical.
Full native wrapper validation remains open. These results do not qualify a
compiler release, establish theorem authority or replace broader project gates.
