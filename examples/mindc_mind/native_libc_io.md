# Fixed libc intrinsic I/O component

Author: STARGA Inc.

`native_libc_io.mind` constructs and checks a deterministic owned recipe for the
four-argument `__mind_read` and `__mind_write` operations on Linux x86-64 LP64
SysV. It is a standalone compiler component. Production callsite admission,
compiler integration, seed qualification and the ten-program consumer gate are
separate obligations.

## Semantics

For a null buffer or a signed count less than or equal to zero, return zero
without calling libc or changing errno. Otherwise any negative offset, including
INT64_MIN, selects `read`/`write`; nonnegative offsets select `pread`/`pwrite`.
The file descriptor is converted to the low 32 bits of C `int`; positive counts
retain their bits as `size_t`. Pointer bits and nonnegative `off_t` are preserved.
The actual libc function supplies the signed `ssize_t` result and thread-local
errno. There is no synthetic errno, raw-syscall fallback, retry, or cached errno
pointer. A successful libc operation does not imply errno is zero.

The code uses ordinary four-GP-register SysV calls, aligns RSP before each libc
call and restores it on return. It does not modify callee-saved registers. Both
thunks are 50 bytes. Their four indirect-call operands are zero until the dynamic
writer applies its checked GOT fixups. Each import requires `GLIBC_2.2.5` from
`libc.so.6`; profile 1 uses `/lib64/ld-linux-x86-64.so.2`.

`__mind_open` remains outside this fixed-call component: libc `open` is actually
variadic and needs a separately reviewed generated-call recipe. Raw
`__mind_sys_*` and legacy three-argument I/O have separate contracts and are not
modified. Callback, multithreading and cancellation behavior is not claimed
beyond taking the real libc call path.

## API, ownership and trust boundary

- `nio_build(profile, diagnostic32)` returns an owned flat bundle, or zero.
- `nio_check(bundle, readable_length, profile, diagnostic32)` returns an error
  code after rebuilding and comparing the entire canonical bundle.
- `nio_free(owned_bundle)` releases that allocation.

The two checked entrypoints initialize all four diagnostic words. Error codes
are 1 invalid pointer/length, 2 unsupported profile, 3 generated declaration or
registry mismatch, 4 changed bundle, and 5 allocation/code construction failure.
Diagnostics contain `[error, source_offset, underlying_registry_error, 0]`.

Pointers must identify live allocations of the stated size. Diagnostics must be
writable and disjoint from inputs. `nio_free` requires a still-live owned
allocation. Internal helpers are exported for compiler composition but require
their documented checked inputs; they are not additional hostile-input APIs.
The component checks its length arithmetic and returned allocations. It does
not establish general parser/String safety under every allocation failure.
The existing registry builder also retains lexer/AST scratch not released by
`nb_cabi_free`; the thunk releases its own buffers and descriptor, but repeated
in-process revalidation does not yet have a proven bounded scratch lifetime.
A production collector must address that lifetime without weakening source
binding. The current isolated-process tests do not establish such a bound.

The component creates its declaration source internally, reparses it through
`nb_cabi_build`, checks all four typed rows, and obtains the complete MCABI001
encoding through `nb_cabi_encode`. Source, descriptor, code, import rows, fixups
and version strings are copied into one initialized allocation before temporary
buffers are released. No caller-supplied source, symbol name, origin flag,
registry, hash, or PASS flag establishes a generated import. The source uses
`i32` file descriptors, `u64` counts, `i64` offsets/results and explicit const/mut
byte pointers. The registry remains the source-derived declaration authority.

Every bundle has `callsite_admission=0`. Future production lowering must bind
this recipe to an actual resolved intrinsic ID and callsite in the same captured
source closure. Calling this component, or knowing its helper names, cannot
open compiler admission. The generic test fixture composes already-reviewed
image/startup/allocator components; it is not production integration.

## Flat bundle revision 1

The 24-word little-endian i64 header is:

| Word | Value |
| --- | --- |
| 0–3 | magic `NIOBND01`, total length, profile 1, recipe 1 |
| 4–9 | source offset/length, MCABI offset/length, code offset/length |
| 10–11 | read entry 0, write entry 50 |
| 12–15 | import offset/count, fixup offset/count |
| 16–21 | version offset/length, SONAME offset/length, interpreter offset/length |
| 22–23 | callsite admission 0, reserved 0 |

Each 40-byte import row holds `[owned_source_name_offset, name_length,
registry_row_index, convention_0, flags_0]`. Source order is read, pread, write,
pwrite. Each 24-byte fixup holds `[code_displacement_offset,
original_import_index, operation_id]`. The four rows are `(23,1,0)`, `(38,0,0)`,
`(73,3,1)`, `(88,2,1)`. Writer sorting must remap these original indexes normally.
All payload bytes, header words and rows are included in revalidation; no raw
pointer or uninitialized padding is serialized.

## Reproduction and evidence

`native_libc_io_check.py` is a development-only host orchestrator; the product,
registry, recipe checker, emitter and diagnostic transport are MIND. The host
assembles an independent reference, freezes the catalogue before execution,
compares native and MLIR bundles/ELFs, and executes each against the existing
MLIR runtime reference. Its runtime assertions include exit status, stdout,
stderr, errno, buffer bytes, file contents, file position and pipe contents.
Tests require Linux x86-64, glibc with the declared versions, binutils and the
supported MLIR CPU toolchain. They do not call a live service.

Run from the repository root with explicitly verified diagnostic compiler paths:

```sh
python3 examples/mindc_mind/native_libc_io_check.py --enumerate --evidence "$EVIDENCE"
python3 examples/mindc_mind/native_libc_io_check.py \
  --evidence "$EVIDENCE" --seed "$NATIVE_SEED" --seed-sha256 "$NATIVE_SHA256" \
  --mindc "$CPU_COMPILER" --mindc-sha256 "$CPU_SHA256" --mutations
```

Use one CPU with an aggregate 2 GiB cgroup memory limit, zero swap, a 2 GiB
per-child address-space cap and time limits. The harness adds child AS/CPU
limits; the caller supplies the aggregate cgroup. It captures the exact full std
prefix, compiler/component/fixture sources, reference assembly, catalogues,
compiler/tool/libc/loader hashes, source packets, output binaries and command
receipts, forces the same std path, and detects before/after input drift. Reuse
requires exact build packets, all dependencies and output identities; only a
recorded host-runner revision may differ. Optional `--reuse-mutants` accepts
trusted local development receipts only: it verifies the exact original and
mutated source packets, pinned compiler identity, build stdin/stdout hashes and
executable bytes, then executes the mutation again. A changed mutation is built
afresh. These local receipts are provenance records, not cryptographic
authentication of an arbitrary third-party executable.

The original frozen contract contains 54 metadata cases, 60 runtime cases and
six actual MIND source mutations. A mutation only counts when its compiled
program produces a normal, incorrect result; crashes, malformed ELF and build
failure do not count. MIC checks cover every product function separately.
The C-errno observation in the independent runtime reference explicitly
sign-extends the 32-bit value; `__mind_load_i32` itself intentionally zero-extends.
Negative preset errno is also covered by independent supplemental tests.

The final run produced identical native/MLIR bundles and dynamic ELFs; all
54 metadata pairs, 60 runtime triples and six compiled source mutations passed.
All 24 product functions emitted in the per-function MIC check. Separate review
found no product defect and passed 1,826 byte-corruption outcomes plus 36
supplemental runtime outcomes. It caught the signed-errno reference-fixture
error, which was corrected without changing expected behavior. The review
retains the parser scratch and production-integration obligations above.
This component evidence does not qualify a compiler pin.
