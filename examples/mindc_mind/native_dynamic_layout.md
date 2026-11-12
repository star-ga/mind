# Checked dynamic MIND image layout, version 1

Author: STARGA Inc.

This standalone MIND planner describes one Linux x86-64 little-endian LP64
ET_EXEC image. It does not emit code, enable C calls, authenticate provenance,
initialize libc or qualify a compiler. Historical static layout and the existing
dynamic writer remain unchanged. Page size is4096. The planned memory budget is
2147483648 bytes; actual runtime use must still pass the2GiB execution gate.

## Interface and ownership

`ndl_plan(request: i64, output: i64) -> i64` reads ten signed64-bit words from
request and writes32 signed64-bit words to output. Both addresses must designate
valid caller-owned suitably aligned memory of at least80/256 bytes. Arbitrary
pointer validity is not checked. All ten inputs are captured before any output
write, so identical or partially overlapping buffers are supported. The caller
must not concurrently change either buffer. No allocation occurs; output has no
borrowed pointers. On every diagnosed refusal all256 output bytes are zero.

Request words in order:

| Word | Meaning |
|---|---|
| 0 | Policy version, exactly1 |
| 1 | Complete code bytes, including future startup/runtime/main adapter |
| 2 | Complete read-only dynamic lookup metadata bytes |
| 3 | Complete writable relocation/dynamic table bytes, sealed by RELRO |
| 4 | Literal bytes; zero means no literal PT_LOAD |
| 5 | Primary arena bytes, page-aligned, at least8192 (data/header page + state page) |
| 6 | Maximum future overflow bytes, zero or a multiple of1048576 |
| 7 | Dedicated MIND stack bytes, page-aligned, at least4096 |
| 8 | Minimum unmapped guard bytes, page-aligned, at least4096 |
| 9 | External runtime reserve bytes, nonnegative and page-aligned |

All byte counts must be nonnegative and no greater than the policy budget.
Code/metadata/relocation sizes must be positive. Reserve accounts explicitly for
loader/libc, original process stack and other mappings; zero is representable
for planning and proves no runtime headroom. No default reserve or claim that a
particular reserve suffices is inferred. Consumer qualification must bind and
measure the actual runtime environment.

## Layout and arithmetic

Headers occupy one read-only page at0x400000. Code starts at file offset4096,
VA0x401000 and occupies a separately rounded RX range. Read-only metadata and
RELRO follow on page boundaries; a separate zero-filled RW context page follows
RELRO in virtual memory and is never sealed as GOT data. This complete low range
must end at or below0x800000, regardless of whether literal bytes are present.
Literals have VA0x800000 and a page-aligned file offset immediately after RELRO;
their rounded range must end at or below the primary arena at0x1000000.
Context is BSS and consumes no file bytes. Literal absence sets its file offset
and map length to zero while retaining its canonical VA.

The arena's final4096-byte page is allocator state. Let Aend be arena end:
overflow_base = align_up(Aend + guard,1048576),
overflow_end = overflow_base + overflow_bytes,
stack_base = overflow_end + guard, stack_top = stack_base + stack_bytes.
The first actual guard includes any extra alignment gap. Overflow is initially
unmapped; both overflow endpoints are mandatory for the future allocator.
The stack is the highest executable-owned PT_LOAD, so the Linux ELF brk floor
is at or above stack_top rather than inside the planned overflow window.
This does not prevent other mmap users from occupying an unmapped gap. The
future allocator must retain MAP_FIXED_NOREPLACE and refuse collisions/exhaustion.
The current allocator has no overflow_end input and is incompatible with this
layout until that separate bounded runtime integration is implemented.

Initial mapped bytes count rounded header/code/metadata/RELRO/context/literals,
the primary arena and stack. Potential mapped bytes add the complete overflow
capacity. Reserved total additionally counts both actual guard gaps and the
external reserve. It must not exceed2147483648. Counting guard gaps is a
conservative planning policy, not a claim they consume Linux RLIMIT_AS mappings.
Every addition and rounding is validated before use; no wrapped sum is trusted.

Program headers:13 without literals,14 with literals. They describe PT_PHDR,
PT_INTERP, four original LOADs (headers/code/metadata/RELRO), PT_DYNAMIC,
PT_GNU_STACK, PT_GNU_RELRO, plus PT_NOTE and context/arena/stack LOADs, plus the
optional literal LOAD. The interpreter starts immediately after the ELF64
header and all56-byte program headers; reserve256 bytes including its NUL
(matching the existing writer's255-byte path bound). The note starts thereupon
at4-byte alignment and occupies exactly52 bytes: namesz5, descsz32, type1 or2,
`MIND\0` padded to8 bytes, and a32-byte digest. It lies wholly in the header
page and cannot overlap code or literals. A future writer must validate the
note bytes and retain the actual digest kind (1=IR,2=source); the planner assigns
placement only and cannot bless a digest or convert one kind into another.

## Output schema

| Word | Meaning |
|---|---|
| 0 | Version1, written only on successful complete planning |
| 1 | Header VA0x400000 |
| 2 | Code file offset4096 |
| 3 | Code VA0x401000 |
| 4 | Rounded code map bytes |
| 5 | Metadata file offset |
| 6 | Metadata VA |
| 7 | Rounded metadata map bytes |
| 8 | RELRO file offset |
| 9 | RELRO VA |
| 10 | Rounded RELRO map bytes |
| 11 | Writable context VA (one page, BSS) |
| 12 | Literal file offset, or0 when absent |
| 13 | Literal VA0x800000 |
| 14 | Rounded literal map bytes |
| 15 | Arena VA0x1000000 |
| 16 | Arena bytes |
| 17 | State page VA |
| 18 | Overflow base |
| 19 | Overflow end (exclusive) |
| 20 | Stack base |
| 21 | Stack top (exclusive) |
| 22 | First actual guard bytes |
| 23 | Second guard bytes |
| 24 | Initially mapped bytes |
| 25 | Maximum potential mapped bytes |
| 26 | Reserved total including guards and external reserve |
| 27 | File byte length (rounded literal padding retained) |
| 28 | Program-header count |
| 29 | External runtime reserve bytes |
| 30 | Interpreter file offset (256-byte reserved slot) |
| 31 |52-byte MIND note file offset |

Original byte lengths remain the captured request's responsibility when emitting
filesz fields; mapped lengths do not authorize reading padded bytes from input
buffers. The planner returns scalar geometry, not an authenticated capability.
Future consumers must preserve the source/request binding and must not accept
an attacker-supplied output record without recomputing/validating it.

Return codes, checked in this order:1 unsupported policy;2 byte-count domain
or missing required nonzero size;3 alignment or minimum arena/stack/guard;
4 low header/code/metadata/RELRO/context collision;5 literal/arena collision;
6 arithmetic/header placement refusal;7 memory budget exceeded. Success=0.
All output words are zero on refusal. A build error, crash or OOM is never a
valid refusal test result.
Code6 is a defensive arithmetic/header invariant refusal; the currently bounded
public request domain does not require a reachable code6 case. The test report
must not claim coverage of an unreachable error path.

## Frozen verification contract

`native_dynamic_layout_cases.tsv` is frozen before implementation/builds. Each
row specifies the exact ten request words, expected exit and selected exact
output words; all successful rows additionally check every output field against
an independently expressed segment model, disjoint rounded mappings, page
congruence, note placement and budget accounting. The first two positive rows
carry complete independently computed32-word golden records. Refusals require
exact zero output, not merely a nonzero process status. The harness records the
case-file hash before compilation and refuses drift before execution.

Native and fresh MLIR-compiled MIND artifacts must match expected exit/stdout/
stderr for every frozen case. Exercise identical and partially overlapping
request/output buffers, deterministic repeated calls, and success-then-refusal
output clearing. Actual source mutations remove low-range/literal/budget/alignment
requirements or break alias capture; each must yield a normal wrong-result
detection, never a crash/build refusal counted as detection. Existing dynamic
writer byte/protection fixtures remain unchanged and run separately. This is a
component gate, not final compiler or consumer qualification.

## Reproduction

Use a verified current bounded native compiler executable and the official
MLIR-capable MIND compiler. Supply their exact SHA-256 identities explicitly;
the harness never changes a repository pin. Place the supported LLVM/MLIR
tools on PATH and configure CC/CLANG/MLIR_OPT/MLIR_TRANSLATE as required by that
compiler build. Run under an aggregate2GiB memory limit, zero swap and one CPU:

```sh
python3 examples/mindc_mind/native_dynamic_layout_check.py \
  --seed /path/to/bounded-native-compiler --seed-sha256 VERIFIED_SHA256 \
  --mindc /path/to/mindc --mindc-sha256 VERIFIED_SHA256 \
  --evidence /fresh/absolute/evidence-directory
```

The harness applies a2GiB process address-space limit, serializes builds, uses a
fresh MLIR project with cache bypass, records exact source/toolchain identities
and each process's streams/status, and runs the original dynamic-writer gate.
An outer cgroup supplies aggregate memory/swap/CPU enforcement. Native diagnostic
executables use an explicit256MiB arena/128MiB stack build packet; these are test
runner resources, not default sizes of a planned dynamic image.

Observed component validation on2026-10-15:52 exact fixture cases and8
state/repeat controls passed on both freshly built native and MLIR artifacts;
six compiled source mutations returned normal wrong results detected by their
predeclared cases. The unchanged standalone dynamic-writer regression also
passed. The manual harness is classified in SMOKE_WIRING; no hosted CI run,
general C-call support, production integration or new compiler pin is claimed.
Independent integration review remains a separate requirement.
