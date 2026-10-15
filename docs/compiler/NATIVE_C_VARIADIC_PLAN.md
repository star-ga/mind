# Scalar SysV variadic argument plan, version 2

Copyright 2026 STARGA Inc. Author: STARGA Inc.

Bounded component for Linux x86-64 LP64 SysV, eight-bit bytes,
32-bit C int and binary32/binary64. Implementation is MIND only. This slice
plans types and locations; it performs no value conversion, call emission,
binding admission, dispatch integration or foreign call. Execution status: the frozen diagnostic campaign and artifact repeat passed
on both explicit instruments below. Independent implementation review passed
with the scoped notes and transport correction recorded below.
No compiler-pin, performance or call-admission result is claimed.

## API and trusted input

`ncv_validate(decl:i64,actual_types:i64,total:i64)->i64` returns zero for a
valid request or the error below, without allocation.
`ncv_plan(decl:i64,actual_types:i64,total:i64)->i64` validates first and returns
one owned allocation, or zero on validation refusal or zero allocation result.
Validation distinguishes validation refusal from allocation refusal for the
same immutable inputs. There is no partially published plan.

`decl` denotes an existing readable 64-byte internal MCABI row. Its i64 fields
are source spans at 0/8 (unused), convention at 16, safety at 24, varargs at 32,
fixed count at 40, fixed-type pointer table at 48, result-type pointer at 56.
Convention and safety each accept exactly 0 or 1; varargs must be exactly 1.
Require `1 <= fixed <= total <= 256`. This also rejects negative and excessive
counts before argument indexing or allocation, so size arithmetic cannot
overflow. The fixed table contains at least fixed readable i64 pointers;
`actual_types` contains at least total readable i64 pointers, each to the
actual source type supplied by the trusted typed frontend.

Each type record is 48 readable bytes: an i64 length in 1..33 followed by that
many type bytes. Scalars are length one: 0 void, 1 i8, 2 u8, 3 i16, 4 u16,
5 i32, 6 u32, 7 i64, 8 u64, 9 isize, 10 usize, 11 f32, 12 f64, 13 bool.
Pointers have 1..32 prefix bytes, each 32 (*const) or 33 (*mut), ending in a
scalar 1..13. Pointer-to-void, unknown tags, bad lengths and malformed chains
refuse. Pointer classification is internal tag 14, never a valid encoded
scalar byte. Results accept scalar 0..10, 12, 13 or a validated pointer;
fixed parameters accept scalar 1..10, 12, 13 or a validated pointer.
Tail sources additionally accept f32. All argument void types refuse.

All rows, tables and records must be live, readable and immutable throughout
validation and construction, including across allocation. They must have
valid nonwrapping address arithmetic and suitable internal i64 access.
Zero row, table or record pointers refuse; numeric checks cannot establish
provenance, extent, readability or lifetime. Nonzero invalid addresses are
outside this API's preconditions, not safely probeable inputs. The frontend
must supply true expression types; the planner cannot inspect expressions or
values to establish that agreement. Inputs must not overlap the fresh output.

## Errors and order

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | OK | Valid plan request |
| 1 | CONVENTION | Convention is neither 0 nor 1 |
| 2 | SAFETY | Safety is neither 0 nor 1 |
| 3 | VARARGS_REQUIRED | Variadic field is not 1 |
| 4 | COUNT_RANGE | Required count inequality fails |
| 5 | INVALID_DECLARATION_TYPE | Malformed result or malformed/void fixed declaration type |
| 6 | UNSUPPORTED_FIXED_TYPE | Fixed declaration f32 |
| 7 | UNSUPPORTED_RESULT_TYPE | Result f32 |
| 8 | NULL_INPUT | Zero declaration or required table |
| 9 | INVALID_ARGUMENT_TYPE | Malformed/void actual source, including null source record |
| 10 | SOURCE_TYPE_MISMATCH | Fixed actual differs in length or encoded bytes |

Null result and null fixed type records are type errors (5), as determined
by the type validator; code 8 is only for the declaration and required tables.

Order is null declaration, convention, safety, varargs, count, result,
fixed table, actual table, then arguments in source order. For each fixed
argument validate declared type, refuse declared f32, validate actual type,
then compare exact lengths and bytes. Tail types are validated directly.
The first error is retained. Padding never participates in type equality.
Distinct records with identical encodings agree; pointer qualifiers, depth
and pointee type all participate. Fixed f32 actual versus declared i32 is
a mismatch, while a declared fixed f32 refuses with code 6.

## Promotion and placement

Fixed arguments have identity conversion; the frontend must already have
performed any language-authorized fixed conversion. Tail promotion is:

| Source | ABI | Conversion |
| --- | --- | --- |
| i8 | i32 | 1 signed8to32 |
| u8 | i32 | 2 unsigned8to32 |
| i16 | i32 | 3 signed16to32 |
| u16 | i32 | 4 unsigned16to32 |
| bool | i32 | 5 boolto32 |
| f32 | f64 | 6 f32tof64 |
| Other supported nonvoid source | Exact source encoding | 0 identity |

All u8/u16 values fit signed C int. u32 remains u32. Conversion metadata
requests value work; it does not perform that work. In particular source f32
width is explicitly 32, and its ABI width is 64. No fixed f32 emitter support
is implied. Bool conversion must canonicalize to 0/1; integer conversions
must consume meaningful low source bits before extending. ABI normalization
does not replace conversion.

Process fixed then tail arguments with shared independent GP6 and SSE8
counters. ABI f64 gets XMM0..7; other supported arguments get RDI, RSI, RDX,
RCX, R8, R9. Exhaustion of one class does not exhaust the other. Spills use
eight-byte slots in source order, offsets 0,8,...; reservation rounds upward
to 16, with padding above the argument area. AL_count equals allocated SSE
register count, including fixed arguments, and is at most 8. Spilled f64 does
not increase AL_count.

## Exact owned bytes

All fields are native little-endian i64 words unless described as type bytes.
The allocation is exactly `96 + total*160` bytes, from 256 through 41056.
Every byte is initialized to zero before construction; only live encoding
bytes are copied, leaving deterministic zero record padding and reserved words.

| Header offset | Field |
| --- | --- |
| 0 | version = 2 |
| 8 | total |
| 16 | fixed |
| 24 | gp_used |
| 32 | sse_used |
| 40 | stack_bytes |
| 48 | aligned_reservation |
| 56 | AL_count |
| 64 | resulttag (pointer = 14) |
| 72 | resultclass (0 void, 1 INTEGER, 2 SSE) |
| 80 | resultwidth (void 0) |
| 88 | resultnormalize |

Row i starts at `96 + i*160`:

| Row offset | Field |
| --- | --- |
| 0..47 | Owned source type record |
| 48..95 | Owned ABI type record |
| 96 | role (0 fixed, 1 tail) |
| 104 | conversion (0..6 above) |
| 112 | sourcewidth in bits |
| 120 | abiwidth in bits |
| 128 | abi_normalize |
| 136 | locationclass (1 GP, 2 SSE, 3 stack) |
| 144 | location (register index or stack byte offset) |
| 152 | reserved = 0 |

Widths follow ncp_width except source f32 = 32. Normalization follows
ncp_normalize: 0 full bits, 1 signed extension, 2 zero extension, 3 Boolean.
Full exact live encodings, including all pointer bytes, are copied into both
records for identity conversion. Promoted ABI records contain length 1 and
the promoted scalar byte. No output field borrows an input record pointer.
Result metadata intentionally stores classification facts, not a result type
record pointer. Caller releases the single allocation using the established
allocator counterpart. Inputs may be released after construction completes.
Construction has no fallible step after its one allocation, under the live
immutable input and valid allocator assumptions.

Dependencies are the supplied ncp_type, ncp_width, ncp_normalize,
ncp_result_class, ncp_location_class, ncp_location and ncp_delta, plus existing
MIND load/store/allocation intrinsics. No ncp_* helper is redefined. The
ncv_type wrapper additionally rejects a negative terminal byte if i8 loads
sign-extend. Helpers are internal implementation steps despite pub visibility.

## Allocation and failure boundary

`ncv_allocate` is the single allocator adapter. A zero result yields no plan.
The zero-return test substitutes this adapter in a separately compiled fixture;
there is no runtime fault flag. An allocator which aborts on exhausted memory
may terminate instead of returning zero. This test does not establish recovery
from arbitrary host or native-runtime OOM. Valid-input construction has bounded
size and no additional allocations.

## Frozen test contract

The catalogue in `examples/mindc_mind/testdata/native_c_variadic_plan/`
contains 65 prelisted controls. C01–C16 retain the original scalar, promotion,
pointer, register-exhaustion and count-boundary proposal cases. C17–C64 add
input refusals, precise error priority, full pointer encodings, owned output,
nonzero input padding, both count-256 limits, result classes and fixed/tail
register interaction. C65 is the deterministic allocator-zero build.

The MIND driver compares every plan byte with a frozen expected plan, including
padding and reserved words. It checks that input records were not changed,
poisons them after construction to check ownership, then restores them and
changes padding before constructing an independent equal plan. Every successful
case therefore checks repeat construction and borrowed-input independence.
Each refusal calls both validation and construction. Source and actual records
have different addresses and padding. These checks cover type metadata, not
integer or float value conversion, pointer dereferencing, or foreign execution.

Six separately compiled product mutations remove fixed-type equality, break
u8 and f32 promotions, couple independent register pools, overcount AL, and
corrupt a reserved zero word. A mutation counts only when its prelisted case
returns the MIND assertion's normal exit 1, stdout `0\n`, and empty stderr on
both backends. Crashes, transport errors, build failures and timeouts fail the
campaign. The unmutated cases require exit 0, `1\n`, empty stderr and exact
fd3 bytes; both backends must match those bytes. There are 71 observations per
backend: 64 baseline, one allocator fault, six mutants.

The fixture packet is little-endian: twelve i64 header fields (version1,
convention, safety, varargs, fixed count, total count, stored fixed-record count,
stored actual-record count, null flags, expected validation code, expected
plan byte length, allocator-zero flag); one 48-byte result record; stored fixed
then actual records; expected plan. Null bits are 1 declaration, 2 fixed table,
4 actual table, 8 result record, 16 first fixed record, 32 first actual record.
Counts used for transport are separately bounded to 0..256. Only the frozen
fixtures are driver inputs; this driver is not an arbitrary untrusted-wire
parser. Expected plans are checked-in data, authored before execution. The
MIND verdict is the executable authority for these diagnostic assertions.

Fd3 receives the observed i64 error and plan length, followed by owned plan
bytes. This fixed binary receipt and stdout are deterministic. Compiler paths,
OS timing, cgroup counters and executable backend bytes are a separate
observation envelope; native and MLIR executable hashes need not agree.

## Reproduction

Run from the repository root, with a new evidence directory and explicit
hash-checked native and CPU compiler instruments. Enumeration only captures
files and applies already-authored patches; it does not build or execute MIND.
The shared `_stdlib_manifest.py` supplies the unchanged ordered standard-library
prefix. Both builds use the same captured fixed planner, variadic planner and
MIND driver bytes. The CPU build uses the captured standard library directory.

```sh
python3 examples/mindc_mind/native_c_variadic_plan_check.py \
  --enumerate --evidence "$EVIDENCE"
# Review and lock $EVIDENCE/PREEXECUTION.sha256 before execution.
# Set REVIEWED_CAPTURE_SHA256 to that independently recorded digest.
export MLIR_OPT=/path/to/mlir-opt MLIR_TRANSLATE=/path/to/mlir-translate
export CLANG=/path/to/clang
python3 examples/mindc_mind/native_c_variadic_plan_check.py \
  --evidence "$EVIDENCE" --seal-sha256 "$REVIEWED_CAPTURE_SHA256" \
  --seed "$NATIVE_COMPILER" --seed-sha256 "$NATIVE_SHA256" \
  --mindc "$CPU_COMPILER" --mindc-sha256 "$CPU_SHA256"
```

The execution must additionally run in an owned systemd unit or equivalent
with aggregate MemoryMax=2G, MemorySwapMax=0, CPUQuota=100%, CPU affinity 11 and
nice 19. The transport enforces per-process AS=2GiB and core=0; builds have
180 CPU seconds/240 wall seconds, each driver run 10 CPU/12 wall seconds.
One build runs at a time. Record exact compiler/LLVM/libc/loader hashes,
commands, stdout/stderr/exits and cgroup memory.events/peak/swap observations.
The entry point is manual in SMOKE_WIRING.tsv because CI must supply the exact
compiler prerequisites. Neither instrument becomes a qualified consumer pin.
No original MIND-Lab evaluator acceptance class is closed by this component.

## Integration boundary

This module is additive. Existing `ncp_*`, `nce_*`, source admission, libc
binding and compiler dispatch are unchanged. A future emitter must implement
all conversion operations, consume the exact plan schema and supply the
SysV AL value. A trusted typed frontend must bind each source expression to
its true type and the correct foreign declaration before dispatch. Supporting
this metadata format alone does not permit source-level variadic calls.

## Observed campaign, 2026-10-15

| Observation | Result |
| --- | --- |
| Baseline catalogue | 64/64 on native and fresh MLIR |
| Deterministic allocator-zero control | 1/1 on both |
| Actual product mutants | 6/6 normal MIND assertion failures on both |
| Diagnostic receipt parity | All 71 pairs byte-identical |
| Initial fresh builds | 16/16 normal success |
| Repeat builds | 2/2 match their original same-backend ELF bytes |
| Repeated C01/C10/C53/C64 | 8/8 exact MIND verdicts and receipts |
| Initial aggregate peak | 78,204,928 bytes; all memory.events zero, swap zero |
| Repeat aggregate peak | 72,351,744 bytes; all memory.events zero, swap zero |

The repeat is separate reproducibility evidence, not additional logical case
coverage. The native and MLIR artifacts differ from each other, as expected.
The same captured source rebuilt through each backend produced an identical
artifact for that backend in a fresh output directory. This is an observed
same-machine repeat, not a cross-machine reproducibility claim.

Exact SHA-256 identities:

| Input/artifact | SHA-256 |
| --- | --- |
| Variadic planner | `5faf7a585dd74809c192be5c0bd445628776a54d5645182ccd015c8346154ba6` |
| MIND assertion driver | `a951de0fdc7b05fe0582e7fd4759c6817d9c6fd696a9ec366cebcac880636888` |
| Unchanged fixed planner dependency | `304f66fb4b40f885983a48eae19d41a693dbcb399474c66882d50d92a082427b` |
| Native diagnostic compiler | `03e956bdfe049d43d128a387264118c2623916be487868268ec4892af57c093a` |
| CPU compiler | `62bc0f97c4cd9c7f8230bd3af765487626e06bc3918e2c67e35dc058390bc1fd` |
| mlir-opt, LLVM 20 | `c9e187870c70663e6a15129e68c064d382be6d991de91f9670daecf3c8016d4c` |
| mlir-translate, LLVM 20 | `f03e82b5b3c9fffbf1eb00f8a9e461878dbecf52d0061d41184865ab4ceb58c3` |
| clang, LLVM 20 | `7c19dc665abb898a09fe5976c44f7691d7c137534dd5fcf6a76dfaab334c90c8` |
| libc.so.6 | `85e64f97e348786a8fb4d9f3d52fec289e2fb86bba20f0731dfe61990525e0f7` |
| ld-linux-x86-64.so.2 | `bda779ea9e9ad234f60315477d7cfd0bdde861cd8ddd9df8cdaf62f18c4cec14` |
| Baseline native ELF | `e1022f43b349c4a2c3387ce40268893c7e9336cdd1be4d152bf4d491986190e1` |
| Baseline MLIR ELF | `5934e4649b90ca48161f95af833e46c66db6f9796b5c30315b6c35be7c02fdca` |

`OBSERVED.tsv` records each original-case/mutation/backend exit and stream hash.
`BUILD_HASHES.tsv` identifies all 16 original build artifacts. `RESULTS.tar.gz`
contains unmasked MIND stdout/stderr/exits/fd3 receipts, build exits/stderr,
resource observations and a relative-path `EVIDENCE.sha256`. It excludes compiled
programs and path-bearing compiler stdout; executable hashes remain recorded.
The complete original raw build evidence is retained separately. No secrets,
private keys or operational credentials are part of this fixture package.

To check the portable receipt package in a fresh directory:

```sh
tar -xzf /path/to/RESULTS.tar.gz
sha256sum --check EVIDENCE.sha256
```

Architecture scans before and after were recorded with arch-mind. Its current
summary deliberately excludes `examples.*`, including this compiler directory,
so unchanged repository scores do not validate this new module. Direct source
review checks its additive dependency on the unchanged fixed planner helpers.
No architectural session record was invented or an existing session modified.
The diagnostic performs no performance measurement; compiler Criterion gates
remain a separate integration requirement.

Portable receipt archive SHA-256: `61f2db725cf9716f2630622a2edc0af99d68a97eb12febfff00125ce544f872f`.

## Independent review and transport correction

A separate source review and 18 additional prelisted MIND outcomes found no
planner defect within the documented internal-buffer contract. Its six fresh
builds exercised all 256 terminal/prefix byte values, aliased input records,
256-argument mixed allocation, deepest pointer encoding, allocator canaries,
exact allocation extent, refusal-before-allocation and zero-return allocation.
Peak 70,520,832 bytes, all memory.events zero and no swap. This is model-assisted
engineering evidence; it is not a formal proof or independent human acceptance.
The original review report SHA-256 is
`aad1787fb90929020b8a9cf151b2f110965f4b2ffef153d55f48d598fd2b13d1`.

That review found a transport custody defect: the launcher invoked the live
repository run.sh after enumerating a captured copy. The corrected trusted
launcher requires `--seal-sha256`, verifies the owner-reviewed manifest and
all captured file hashes, checks required runner/wrapper identity, and invokes
only the captured `fixtures/run.sh`. Enumeration-only repository imports no
longer run on the execution path. Use an owner-controlled capture directory
with no concurrent writers. A malicious replacement launcher or concurrent
filesystem attacker is outside this cooperative transport boundary.

Five separately preregistered transport controls reproduced the old live-runner
substitution (normal exit 42), proved the corrected launcher ignores a modified
live runner, and proved captured-runner, manifest and reviewed-digest tampering
all refuse before any script/build starts. They use harmless task-owned marker
scripts and execute no compiler; they are not additional planner mutations.
The original 65+6 cases, product, MIND driver and all their observations remain
unchanged. The corrected launcher SHA-256 is
`0cb49b811d61f3a737bbc990d796a02970e42d7f79b6b8ae3a8d4363af4372bd`.

The captured development standard-library prefix used in both independent
campaigns is `12aabf233e961756c0e9f3d4ae7fa40e0427c5103cfee65b92db9be2a530d8a9`.
The historical manifest header remains unchanged; this development prefix is
not represented as a newly qualified historical seed or a consumer pin.

The harness ceiling increases from 305 to 307 for exactly two additions:
`native_c_variadic_plan_check.py` and its captured `run.sh`. Executable plan
assertions remain in MIND. This allocation carries no spare harness allowance.

The transport correction received a separate source/receipt review, closing the
finding under the stated boundary. Its report SHA-256 is
`ea4b444b3e7cdb13dba31f69b8e840d5e0c51267f9518badc16d522c20f42f7e`.
`CUSTODY_OBSERVED.tsv` records the five observed exits and marker/stream hashes.
These are transport controls, not MIND semantic tests.

Four patch fixtures were subsequently converted to zero-context unified diffs
to remove whitespace-only context lines rejected by the staged whitespace gate.
Every final patch reconstructs exactly its originally executed mutant source;
no mutation, case, expected outcome, product or MIND driver changed. Original
patch bytes and their replacement hash mapping remain in the raw audit record.

## Cross-machine native reproduction

The unchanged frozen baseline packet and the same `03e956bd…` diagnostic compiler
were reproduced on a second x86-64 Linux machine. The resulting executable was
byte-identical to the original native ELF (`e1022f43…`). All 64 original baseline
cases matched their exact exit status, stdout, stderr and fd3 receipts. This
reproduction does not cover the allocator-fault case, mutants or the MLIR backend.

Peak memory was 73,994,240 bytes under the 2 GiB aggregate/address-space limits,
with zero memory events and no swap. Compiler and source identities were checked
before and after execution. This is reproduction using the same compiler binary;
it does not qualify a consumer pin or establish cross-architecture equivalence.
The reproduction report SHA-256 is
`eb7434f90961e3f9557c0a5b9839949d9ce738c55bb1898d8dddd7ae78d2e9b5`;
the raw 283-file evidence seal is
`c9fff0e133d310dfa4603cd0180574b69f488cb2da8cc21dab0e06f05bdcf656`.
