# Bounded dynamic allocator component

Author: STARGA Inc.

This MIND component emits a returning allocator runtime for the checked Linux
x86-64 dynamic layout. It is independent of the existing static runtime.
It does not enable a compiler profile, emit startup code, initialize libc,
switch stacks, qualify a consumer or update a compiler pin.

## Emitter interface

`nda_emit(request: i64, metadata: i64) -> String` consumes the ten-word request
documented in [native_dynamic_layout.md](native_dynamic_layout.md) and recomputes
`ndl_plan`. Request80 bytes and metadata64 bytes must be valid caller-owned,
nonoverlapping storage; the request remains immutable during the call. The
emitter neither trusts a supplied layout success flag nor inspects its own
process to infer addresses. It requires the complete planned code size to be
at least `nda_code_size()`. The owner must later bind the complete image's final
code/metadata sizes to the same request and revalidate the complete plan.

Refusal returns an empty String with all64 metadata bytes zero. Success returns
owned code bytes and eight scalar words:
`[1, init_offset, alloc_offset, free_offset, realloc_offset, code_bytes, 0, 0]`.
Offsets are relative to this runtime blob; internal calls and entry pointers
use RIP-relative addressing, so the whole blob can move without rewriting
arbitrary code bytes. Arena/state/overflow immediates are emitted explicitly
from the recomputed plan. The generated entry metadata is not an authenticated
capability: consumers must retain its exact source/request/code binding.

The product is `native_dynamic_allocator.mind`. The checked-in assembly and
generator are development references for auditing its byte encoding. Compiled
MIND emits the runtime without invoking Python, an assembler or C. No product
dependency on the diagnostic harness is introduced.

## Initialization and ownership

Initialization is an ordinary callable adapter, distinct from OS entry and
libc main. It returns0 on first initialization,1 on a valid repeated call and2
on dirty or inconsistent initialization metadata. It preserves RSP with
balanced pushes/pops and preserves GP registers except RAX; flags are clobbered
and the SysV clear-direction-flag precondition applies. It never switches
stacks, reads OS-stack arguments, writes argc/argv, or touches the separate
process-context page. It contains no SIMD instructions.

The caller must provide fresh zero-filled primary/state mappings from the exact
plan. Initialization verifies the entire state page and cursor are fresh before
writing. It reserves the first64 arena bytes: cursor56 is relative to A+8, so
the first block header is A+64 and the first payload is A+72. Reserved cells
including A+8/A+16/A+24 remain untouched for later adapter use. The state marker
is at offset528; allocator entry slots remain536/544/552. A repeated call checks
the marker, all three entry pointers and mapped-end range/alignment before
returning1; it never resets a live heap. This is not a full integrity validation
of every free-list link or caller-written heap byte.

Entry convention is internal MIND runtime ABI: allocation takes size in RSI,
free takes pointer in RDI, realloc takes pointer in RDI and size in RSI. Each
returns RAX and preserves the other GP registers. These are not libc malloc
function signatures. The initial component is single-threaded and non-reentrant.

Owned buffers retain free-list reuse, splitting/coalescing, zeroed returned
storage and realloc prefix preservation. `alloc(n<=0)` returns0.
`realloc(NULL,0)` returns a minimum block; negative sizes return0 with an existing
buffer left live; zero size frees an existing buffer. A pointer outside this
allocator is ignored by free and rejected by realloc with0, without reading
foreign `p-8`. The historical static foreign-copy behavior is unchanged.
Libc-owned pointers require their corresponding libc APIs. Forged headers,
interior pointers, use-after-free and concurrent mutation remain caller errors;
the raw API does not prove arbitrary pointer validity.

## Immutable capacity and mapping

The arena, state page, overflow base and **exclusive overflow end** come from
the checked plan. Runtime state only records the mapped end; it cannot enlarge
capacity. Every public operation rechecks its range/alignment and initialization
marker. Bump growth and realloc top growth validate addition before committing;
the common mapper checks requested and one-MiB-rounded endpoints against the
immutable exclusive end before the syscall. Exact-end fits are permitted;
zero-capacity overflow cannot map.

Mappings retain `MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE` and require
the exact requested address. Collision, capacity exhaustion or refused mapping
prints exactly `error: native arena exhausted\n` to stderr and exits70. The
component does not use destructive MAP_FIXED or silently expand the policy.
Fresh mappings/zero initialization and valid owned metadata are preconditions;
these checks are not a universal OOM or memory-safety guarantee.

## Reproduction and observed scope

The manual diagnostic gate first verifies the exact compiler hashes and the
frozen39-case TSV. It builds the MIND emitter natively and with fresh MLIR, then
requires identical emitted bytes and compares them to the assembly reference.
An independently compiled C instrument calls those actual bytes to check the
predeclared results, allocation contents, adjacent-buffer/reserved-cell canaries,
repeat initialization, register preservation and normal exhaustion behavior.
The C instrument is an external test; it is not a MIND runtime replacement or
an implementation of production dynamic libc startup.

Five actual mutated MIND sources are compiled and executed against separately
predeclared controls: widened capacity, ignored repeated-init entry corruption,
foreign-buffer adoption, missing reuse zeroing and destructive mapping. Only a
normal wrong result counts as detection; compilation failures, signals or OOM
kills do not. Five additional invalid requests require normal refusal and exact
zero metadata. The historical static runtime bytes are independently re-emitted
and checked against their unchanged assembly source.

```sh
python3 examples/mindc_mind/native_dynamic_allocator_check.py \
  --seed /path/to/current-bounded-compiler --seed-sha256 VERIFIED_SHA256 \
  --mindc /path/to/official-mindc --mindc-sha256 VERIFIED_SHA256 \
  --evidence /fresh/absolute/evidence-directory
```

Use the supported LLVM/MLIR PATH and an outer2GiB zero-swap cgroup with one CPU.
The harness adds2GiB per-process address-space and bounded CPU/time limits,
serializes builds, records exact source/tool identities and all process streams
and statuses. It requires diagnostic gcc, GNU as/objcopy/nm and Python; emitted
product code requires none of them. CI wiring remains manual until its exact
source-bound compiler artifact is a runner dependency.

Production image writing/startup, libc interaction, source call lowering,
concurrency, independent review and unchanged consumer qualification remain
separate requirements. Passing this standalone component does not complete
native C FFI or qualify a new compiler release.

Observed component run on2026-10-15: all39 runtime cases, five invalid-request
refusals and five source mutations passed the stated contracts. Each mutation
produced a normal wrong-result exit3, never a crash counted as detection.
Native/MLIR emitted bytes matched exactly; the original static runtime byte
control passed. The final run peaked at76,587,008 cgroup bytes, zero OOM events,
under the2GiB/zero-swap/one-CPU constraints. This is owner-run component evidence;
independent implementation review remains a separate integration requirement.

Frozen case SHA-256:
`ecfa89f8568d718ef14e532d639cdd94af15a86f5829b54f51759f56306e9d00`.
Frozen mutation SHA-256:
`ec18aae3979d086770873aec970d61c8f5b6805b34a8dd837e87eef8be7bfc5f`.
