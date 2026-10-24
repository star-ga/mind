# Dynamic writer MIC compatibility

Author: STARGA Inc.

Independent review found that native and MLIR execution success did not imply
MIC emission compatibility. The original base writer emitted 17 of its 24
functions through the current compiler's MIC path; seven failed because of
early returns, `break`, or conditional-expression locals inside loops. This
correction preserves the original public APIs, input bounds, rejection codes,
ordering, ELF layout and emitted bytes. It adds no compiler admission.

The original source SHA-256 was
`468dc87cdb41da33939d23696eda9ebd980ff75bae91cd82cf5c2e68367081aa`.
The corrected source SHA-256 is
`4c8db4d85ca457809e2d19c792924894279d227cd0198d85f8ea0b381efbf6ef`.
The original 24 functions remain; 14 small helpers make loop transitions
explicit. All 38 functions pass the per-function MIC emission check.

## Preserved semantics

- Text validation folds a validity flag. Once a byte fails, later steps retain
  failure without reading more bytes. Initial pointer, length and first-byte
  checks retain their original order and bounds.
- Lexicographic comparison preserves the first unequal byte or exhausted-right
  result. Later steps cannot overwrite it; equal-prefix length ordering remains
  unchanged, including unsigned byte values above 127.
- Import and patch validation carries the first rejection code. A step with a
  prior error returns before reading a later row. Each validation phase stops
  before starting the next. Duplicate names, overlapping patches and unused
  imports retain their original precedence. The four placeholder-byte checks
  are unrolled in the same order, after the original range checks.
- Insertion sorting returns cursor zero to represent its original `break`.
  Only strictly greater keys swap, preserving stable ordering and the first
  representative of a shared version. Version/rank lookup preserves the first
  match through an absorbing result.
- Metadata construction moves conditional next-link calculations into ordinary
  helpers. Invalid version IDs remain a refusal; the retained validity flag is
  checked before an artifact is returned. Relocation steps cannot write another
  displacement after a failed link check. Existing allocation behavior and raw
  buffer preconditions remain unchanged.

No length limit, validation rule, zeroing requirement, symbol-version constraint
or displacement bound was weakened. Inputs remain valid, live, immutable
allocations; this is not a general pointer-safety or allocator-failure proof.

## Regression evidence

The original `native_dynamic_writer_check.py` and its MIND fixture are unchanged.
The image extension's original 82-case packet and its three source mutations
are also retained. `native_dynamic_mic_cases.tsv` adds 15 separately frozen
cases for first-error precedence, text grammar, byte comparison, stable symbol
ordering and deduplicated version representatives. They were executed against
the original writer on both native and MLIR before changing its implementation.
The supplemental fixture initially used positioned I/O on its stdin pipe and
returned exit90; correcting the offset to -1 established the intended stream
transport before any base-writer refactor. No expected result was changed.

`native_dynamic_legacy_outputs.tsv` records 158 distinct pre-refactor exit and
stdout/stderr hash contracts: 61 original-writer cases, 82 image-extension cases
and 15 supplementary cases. The baseline source above and the fixture/toolchain
hashes are retained in execution evidence. The runner requires every baseline
entry to be exercised and match, including both backends where applicable.
Changing both writer and inspector together cannot silently replace these
hash-locked artifact expectations.

The separate `native_dynamic_mic_mutations.tsv` prelists three actual compiled
source mutations: overwrite an earlier import error, overwrite an earlier
lexicographic difference, and reverse the sort stopping condition. All produce
normal exit0 with wrong output detected by the supplementary expectations.
Build failures, crashes and timeouts do not count as detection.

Final component execution on 2026-10-15 passed 38 base and 14 extension MIC
emissions, the original writer gate, all 82 image cases, all 15 supplementary
cases on both backends, all 158 legacy output contracts and all six compiled
mutations (three original extension plus three new base controls). Peak
aggregate memory was 680,783,872 bytes under 2 GiB with zero swap and OOM events.
The same explicit instruments are documented in the image-extension guide;
raw exit/stdout/stderr, source/tool hashes, artifacts and failed earlier runs
are retained in the private handoff.

Run `native_dynamic_image_check.py` with its documented exact native compiler,
MLIR compiler and source-bound MIC oracle arguments. Per-function emission and
finite tests do not establish whole-compiler fixed-point compatibility, source
admission, production C ABI binding or ten-program consumer qualification.
