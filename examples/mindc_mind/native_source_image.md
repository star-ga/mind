# Source-lowered Linux libc image candidate

Author: STARGA Inc.

`native_source_image.mind` combines the source-lowering path described in
[`native_emission_context.md`](native_emission_context.md) with the checked
dynamic layout, allocator, libc startup and ELF writer. It is a development
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
and source-body code together must fit the writer's 1 MiB code limit.
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

The only generated foreign import is `__libc_start_main`. Dormant checked
extern declarations do not create imports. Reachable foreign calls, stream
I/O, raw process calls and unsupported intrinsics still refuse. The separate
libc I/O and fixed-scalar C-call components are not admitted by this interface
merely because their isolated tests pass.

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

Successful component images, matching backend bytes and mutation detections
do not qualify the compiler. Required next gates include corrected pointer
tracking, independent source review, whole-compiler static compatibility and
reproduction, production runtime wiring, and the unchanged downstream native
versus MLIR qualification. Consumer pins and seed status remain unchanged.
