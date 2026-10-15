# Source-lowered Linux libc image candidate

Author: STARGA Inc.

`native_source_image.mind` combines the source-lowering path described in
[`native_emission_context.md`](native_emission_context.md) with the checked
dynamic layout, allocator, libc startup, stream I/O and ELF writer. It is a development
component, not a production CLI mode or a qualified compiler pin. Pointer
provenance review and whole-compiler regression checks remain open.

## Interface and lifetime

```mind
pub fn nsi_compile(source: i64, length: i64, hash: i64,
    user_lo: i64, route: i64, policy: i64, diagnostic: i64) -> String
```

The caller supplies live source bytes, a readable 32-byte compiler-note input,
a readable six-word policy and a disjoint writable 32-byte diagnostic buffer.
All inputs must remain immutable during capture. A nonzero `route` is borrowed
compiler-owned routing metadata and must remain live and immutable throughout
the entire call; zero selects normal routing. Integer range checks do not
validate arbitrary addresses or authenticate their contents. The diagnostic
buffer is cleared before other validation, so its validity is an unconditional
caller obligation.

The policy contains version, arena size, overflow allowance, stack size,
guard size and reserve size, in that order. Its interpretation and permitted
bounds come from [`native_dynamic_layout.md`](native_dynamic_layout.md).
The source length is positive and at most 64 MiB. Emitted startup, allocator
and source-body code, including any generated I/O thunks, must fit the writer's 1 MiB code limit.
Other parser, registry, literal and image limits still apply.

Success returns an independently owned ELF String. The caller releases its
allocation exactly once. Failure returns an empty String and a diagnostic
category/detail pair. The component releases its captured parts and workspace
before returning; the returned image contains no borrowed workspace addresses.
Compiler AST/parser scratch retains its existing arena/checkpoint lifetime.
This API does not establish bounded total memory for an unlimited sequence
of in-process compilations.

| Diagnostic category | Meaning |
|---|---|
| 0 | No orchestration error recorded |
| 1 | Invalid argument range |
| 2 | Workspace allocation failure |
| 3 | Invalid initial layout policy |
| 4 | Source lowering, local relocation or main metadata refused |
| 5 | Runtime or startup construction refused |
| 6 | Final image planning refused |
| 7 | Final layout changed a source-emission address |
| 8 | Final ELF writer refused |
| 9 | Stream-use, context or checked I/O recipe binding refused |

The remaining diagnostic words are reserved zero. The caller must inspect the
returned image as well as the diagnostic: a supplied or printed success flag
is never the input to code generation or proof verification.

## Source and image binding

1. Copy the policy and note input into an initialized private workspace.
2. Validate a provisional layout and capture the source before parsing.
3. Lower actual reachable source functions through the shared compiler path.
   Validate the main signature and all source-derived local relocations.
4. Patch local relative calls in the unprefixed body. Adding the same prefix
   to every local function preserves those displacements.
   If reachable source uses stream I/O, construct and recheck the fixed NIO
   recipe, then derive its imports and fixups from that owned checked bundle.
5. Plan the image using the actual code, literal, note and import sizes.
   Recompute the final context and compare every address used by source
   emission. Refuse a geometry mismatch instead of rewriting arbitrary bytes.
6. Generate final startup and allocator bytes, then invoke the checked ELF
   writer with the source-derived body and checked metadata.

The caller cannot supply replacement main bytes, import tables, call offsets
or a precomputed valid-layout flag to this interface. Internal helpers still
require trusted, live records produced by the checked entry path; they are
not a serialized-certificate verification API. The note input is existing
compiler metadata, not an authenticated source digest or theorem claim.

## Host and supported boundary

This candidate targets Linux x86-64 LP64 with the interpreter
`/lib64/ld-linux-x86-64.so.2`, dependency `libc.so.6` and
`__libc_start_main@GLIBC_2.34`. These dependencies are explicit and must be
recorded when reproducing an execution. The returning-main adapter uses the
source main's checked argument count and normal libc return semantics.

Without reachable stream calls, the only generated foreign import is
`__libc_start_main`. With stream calls, the image also imports `read`, `pread`,
`write` and `pwrite`, each at `GLIBC_2.2.5`. Both operations share the complete
checked 100-byte NIO bundle; the import set is deterministic, not a link-time
guess based on arbitrary external names. Dormant checked extern declarations
do not create imports. Reachable user foreign calls, raw process calls and
unsupported intrinsics still refuse. The fixed-scalar C-call component remains
a separate integration obligation.

## Source stream-I/O contract

The source operations are `__mind_read(fd, buffer, count, offset)` and
`__mind_write(fd, buffer, count, offset)`. A negative offset selects sequential
I/O; a nonnegative offset selects positioned I/O. The descriptor uses its low
32 bits. A null buffer or nonpositive count returns zero before calling libc.
Otherwise libc returns the actual byte count or `-1`; it owns TLS errno. These
semantics are separate from the raw `std.posix` negative-errno syscall contract.

The compiler checks four operands at the actual call site, integer register
class, no retained raw-pointer origin and no aggregate descriptor. This initial
source-domain check accepts signed i32/i64 declarations and explicit signed
casts, with the existing frontend's folded integer constants and bare literals.
Typed bool, unsigned/narrow variables and floating values require the supported
explicit conversion; unknown inferred binding types are conservatively refused.
These restrictions do not establish a complete source type system. Buffer
addresses remain explicit integer bit views with the normal live-memory
obligation; a zero or invalid process address is not made safe by compilation.

Reserved intrinsic spellings must be unambiguous bare calls. Local functions,
bindings, imports or member references using those spellings cause source
refusal when the reachable image uses stream I/O. Runtime generation uses
source-derived usage, checked local-body ownership and the NIO recipe; no user
`PASS` flag or purported generated-origin marker selects it.

The schema-2 context binds the thunk base to code base plus the actual startup
and allocator sizes. Source calls use fixed SysV argument registers and a
checked thunk entry; final import relocations are rebased through the writer's
canonical symbol ordering. Final geometry is recomputed before emission. A
no-I/O source omits the bundle entirely and retains its prior image bytes.

Direct source-image tests observe return codes, byte streams, file contents,
file cursors and pipe state. They cannot yet observe TLS errno through the
admitted source API. The separate NIO component's errno tests are evidence for
that runtime component, not newly measured source-call errno results.

## Reproduction and acceptance

`self_host_native_emission_context_smoke.py` enumerates its case list before
building. It records exact source/tool/library hashes, native and explicit
MLIR/CPU builds, raw streams, exit statuses and emitted image bytes. Executable
fixtures and compiler logic are MIND; the script transports and compares
diagnostic artifacts. Run it in a bounded owned service with the documented
2 GiB address-space and aggregate memory limits, zero swap and one CPU.

Keep original observations, explicitly versioned expectation corrections and
independent supplements separate. In particular, the original `Box` array
fixture collides with a different standard-library `Box`; its original refusal
is retained. Renaming only the fixture's three type identifiers tests the
intended array-field operation without that collision. This does not modify
the frozen downstream qualification corpus.

`native_stream_io_check.py` consumes the exact two built diagnostic drivers and
captured source prefix. Its original `native_stream_io_cases.json` remains
unchanged. `native_stream_io_cases_v2.json` and
`native_stream_io_case_corrections.json` separately record a mistaken test byte
constant and reference-checked literal folding. Version 3 additionally renames
the fixture helper `send` to `sio_i32_send`: the former collides with an existing
standard-library extern declaration. Only its two identifier occurrences
change; the expected exit code and bytes do not. The original collision is
separately tested as a required refusal. Selecting a newer catalogue
does not turn the retained original failed observations into passes. Every run
records the selected catalogue and packet hashes before executing either side.

Successful component images, matching backend bytes and mutation detections
do not qualify the compiler. Required next gates include corrected pointer
tracking, independent source review, whole-compiler static compatibility and
reproduction, production runtime wiring, and the unchanged downstream native
versus MLIR qualification. Consumer pins and seed status remain unchanged.
