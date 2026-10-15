# Dynamic image writer with checked runtime mappings

Author: STARGA Inc.

This standalone MIND component extends `native_dynamic.mind` with the geometry
from `native_dynamic_layout.mind`. It does not enable compiler admission,
implement typed C calls, initialize an allocator or provide a production libc
main adapter. The original `nd_write_elf` API and implementation remain intact.

## Inputs and ownership

`ndx_plan(code, entry, imports, patches, interpreter, soname, literals, policy,
output) -> i64` produces the existing 32-word layout record. The first six
arguments have exactly the original writer's meaning. `literals` is a borrowed
String of zero through 8,388,608 bytes. `policy` points to six signed 64-bit words:
version (exactly 1), primary arena bytes, maximum overflow bytes, dedicated stack
bytes, guard bytes and external runtime reserve bytes. The layout planner's
alignment, minimum-size and 2 GiB accounting rules remain unchanged.

`output` designates a valid writable 256-byte buffer. `policy` designates a valid
readable 48-byte buffer. They may overlap: policy is captured before layout
output is written. Output must be disjoint from every other borrowed allocation
(code, literal bytes, descriptor arrays/rows and their text). On refusal all
32 output words are zero. There is no generic memory-pointer validity test;
the caller must supply valid live buffers and prevent concurrent mutation.

`ndx_write_elf(code, entry, imports, patches, interpreter, soname, literals,
policy, note, expected_layout) -> String` returns an owned ELF String, or an
empty String on refusal. It does not mutate borrowed inputs. `expected_layout`
is a valid readable 256-byte buffer; all 32 words are compared against a fresh
plan derived from this call's actual code/metadata/RELRO/literal lengths and
policy. A previous successful plan is scalar data, not an authenticated token.
The writer never accepts supplied layout geometry merely because version=1.

`note` must be a valid String containing exactly 52 bytes: little-endian namesz
5, descsz 32, type 1 (IR) or 2 (source), the bytes `MIND\0` padded with three more
zero bytes, then a 32-byte digest. The writer checks structure and retains every
digest byte and the stated kind. It does not compute, authenticate or certify
the digest's meaning. Digest authentication remains a separate obligation.

Plan status 0 means success. Statuses 11..19 carry original writer input codes
1..9; 20 means invalid literal descriptor, 21 means an unexpected original
writer constructor invariant; 31..37 carry planner codes 1..7. These status
categories do not imply every defensive path is reachable through valid inputs.
The writer's empty return is its explicit refusal contract. The diagnostic
driver converts it to exit 23 and exactly `ndx image refused\n` on stderr; a
crash, timeout or allocator failure never counts as a malformed-input refusal.

## Placement and linkage

The original writer first validates inputs and constructs its linked image.
Its code, metadata and page-rounded RELRO geometry already agrees with the
checked planner. The extension validates that constructor's program headers,
exact file length and dynamic-table containment before using its sizes. It
replaces only the first header page and retains every byte from file offset
4096 through the end of the original RELRO area. Consequently symbol ordering,
version requirements, hash tables, GOT relocations and existing FF15 call
displacements are produced by the unchanged writer. No instruction scanning,
E8 expansion, implicit rebase or new symbol-name ABI inference occurs.

The new ELF contains 13 program headers, or 14 when literal bytes are present:
PT_PHDR, PT_INTERP; ascending LOADs for R headers, RX code, R metadata, RW
relocation tables, RW process context, optional R literals, RW primary arena
and RW dedicated stack; PT_DYNAMIC, GNU_STACK, GNU_RELRO and one PT_NOTE.

Headers stay at VA 0x400000 and code at 0x401000. The complete PH table,
256-byte interpreter slot and 52-byte note fit in the first page. Literals use
VA 0x800000 with their own page-aligned file offset; final padding is zero.
The context is a distinct zero-filled BSS page after RELRO. The arena is BSS at
0x1000000; its final page is the allocator state page. The bounded overflow
window and guards remain unmapped. The dedicated stack is the highest LOAD,
keeping the ELF brk floor above that window. BSS requires no file payload.

RELRO seals only the relocation/dynamic-table range, not the context page.
No LOAD is writable and executable. Both GNU_STACK and the dedicated stack
LOAD lack execute permission. Final validation checks header count/order,
exact output length, file-backed bounds, page congruence, nonoverlap of rounded
LOAD ranges, highest LOAD end and the unique complete note before returning.

The layout budget is conservative planning only. Passing it does not prove
that the actual loader, libc, original process stack and unrelated mappings
fit under 2 GiB. Runtime measurements remain mandatory for qualification.
No allocator constants, initialization marker or argc/argv cells are written by
this component; those belong to separately reviewed runtime/startup adapters.

## Temporary allocation lifetime

Only the newly returned old-writer artifact, private request/plan buffers,
new header buffer and new final output belong to the extension. Borrowed code,
notes, literal data, import descriptors and policy/layout buffers are never
freed. `ndx_plan_base` frees its 80-byte request after `ndl_plan` returns.
`ndx_plan` releases the temporary old artifact on success and refusal.
`ndx_write_elf` releases its temporary old artifact, private 256-byte plan and
header buffer on each applicable exit. The header is copied to a distinct
output allocation before appending; it never aliases a String that can be
reallocated during append. Final validation failure additionally frees that
new output. A successful caller owns the returned String buffer.

The final segment traversal owns a separate 32-byte state buffer and releases
it before every post-traversal exit. A failed segment makes the validity flag
absorbing; later iterations do not read another segment or overwrite the state.
The bounded loop always advances, avoiding early-return loop shapes that the
current MIC emitter cannot represent. Plan equality similarly folds all 32
checked word comparisons without returning from inside the loop.

The unchanged original writer's existing internal temporary-allocation behavior
is inherited, not silently redesigned by this wrapper. Arbitrary allocation
failure handling across that existing code is outside this component's claim.

## Verification contract

`native_dynamic_image_cases.tsv` was frozen before implementation. It contains
82 cases with independent expected categories, exits and marker bytes. The
transport harness builds the exact MIND source through a source-bound native
instrument and a fresh MLIR CPU artifact, requiring identical emitted ELF
bytes for each accepted image and exact normal refusal for malformed inputs.
It independently parses all segments, dynamic pointers, version tables,
relocations and note/literal bytes; it also invokes the real loader.

Positive execution checks zero-filled context/arena/state/stack words, writable
ownership, exact literal contents, startup argc and a real libc getpid call.
Separate protection probes print marker byte `A` after a validated read/write,
then deliberately attempt a forbidden write, instruction fetch or unmapped read.
Their expected SIGSEGV is a protection observation, never a malformed-input
rejection or a successful ordinary program. Missing symbol/version cases must
fail in the loader with the specific diagnostic, without reaching main. The
original frozen packet incorrectly expected exit 127 for both. Observed GNU
loader behavior is exit 1 for a missing version and 127 for a missing symbol;
`native_dynamic_image_expectation_corrections.tsv` records that single explicit
correction while preserving every original row and all other expectations.
The loader and libc identities are captured with each successful run. Large
synthetic import/version fixtures are structural cases; they are
not claimed as executable libraries or completed ABI support.

All 32 expected-layout words are independently corrupted in separate cases.
Adjacent valid note kinds and literal/page bounds remain in the packet. The
original dynamic-writer gate runs unchanged, and the baseline old/new linked
payload after offset4096 must be exactly equal.

The separately prelisted `native_dynamic_image_mutations.tsv` preserves that
82-case contract. Three actual source mutations remove layout binding, remove
note validation or mark the context read-only. Each altered MIND program must
compile and exit normally while emitting the wrong artifact; the refusal or
structural checks must detect it. Compilation failures, timeouts and crashes
do not qualify as mutation detection. Mutant artifacts are never promoted.

Run with exact instrument hashes and supported LLVM/MLIR tools on PATH:

```sh
python3 examples/mindc_mind/native_dynamic_image_check.py \
  --seed /path/to/source-bound-native --seed-sha256 VERIFIED_SHA256 \
  --mindc /path/to/official-mindc --mindc-sha256 VERIFIED_SHA256 \
  --mic-oracle /path/to/source-bound-oracle.so --mic-oracle-sha256 VERIFIED_SHA256 \
  --evidence /fresh/absolute/evidence-directory
```

The outer runner must enforce aggregate 2 GiB memory, zero swap, one CPU and
bounded scheduling. Each child also has a 2 GiB address-space cap. The harness
retains source/toolchain/dependency hashes, exact native input, commands and raw
stdout/stderr/exit receipts; it removes LD_* and GLIBC_TUNABLES overrides and
requires no system preloads. It makes no compiler-pin, CI or cross-host claim.
The explicitly hashed oracle also checks nonempty MIC emission for every
extension function. This is separate from native/MLIR runtime parity and does
not by itself establish MIC closure for dependencies or the whole compiler.

### Observed component run, 2026-10-15

The 82 cases passed with the explicitly recorded missing-version exit correction:
26 accepted images had identical native/MLIR bytes, 56 malformed cases returned
the exact normal refusal, and 44 loader/runtime executions matched their expected
outcomes. All three compiled source mutants produced normal wrong results that
the checks detected. The unchanged original writer gate passed; the baseline
linked payload after offset 4096 remained byte-identical. Peak aggregate memory
was 631,611,392 bytes under a 2 GiB limit, with zero swap and no OOM events.

The native diagnostic compiler SHA-256 was
`7f223c1ac82401fac84b490797d1770b394b05b6e525e92586c989362c139811`;
the official compiler used for a fresh MLIR CPU build was
`ba27a0ff9f0a960886246031bcb90829d0e2d0a0195df9d9e9eb9132233524cb`.
Both compiled the same standalone source, including the exact vec/string
dependency prefix. This excludes unused argv helpers unsupported by that MLIR
instrument; it does not modify the compiler's production standard-library seed.

Earlier diagnostic attempts are retained in the private handoff. They exposed
unsupported escaped literals/unused argv helpers in the chosen test composition,
an older diagnostic compiler's already-fixed source-note corruption of trailing
literals, and the single incorrect loader-exit expectation. Those failed runs
are not counted as passing evidence. No frozen compiler seed or consumer pin
was changed. Production libc startup, allocator initialization, typed C calls,
full compiler integration and ten-program qualification remain separate work.

Independent review subsequently identified two extension functions that passed
runtime checks but could not emit MIC because of early loop returns. The
refactor above passes emission for all 14 extension functions using oracle
SHA-256 `bab5c02ec407ec7c1b429f5befb39ec49d997d75dd7f1392f83ab364cb4a48ae`.
The same 82 cases and three mutations pass again, with all 164 emitted stdout/
stderr pairs unchanged from the prior runtime run. Seven functions in the
unchanged underlying writer still lack MIC compatibility; that pre-existing
dependency obligation is not hidden by this extension-level result.
