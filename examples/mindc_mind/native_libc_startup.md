# Standalone libc startup adapter

Author: STARGA Inc.

`native_libc_startup.mind` emits a Linux x86-64 LP64 SysV startup stub and a
returning adapter for an ordinary MIND `main`. This is a separately testable
component. The production compiler does not select it. It changes no static
boot code, native admission profile, compiler seed, or qualified toolchain pin.

The implementation is MIND. Python, GNU binutils, and the small C preload
fixture are development/test instruments. They are not emitted-program runtime
dependencies. The resulting dynamic image intentionally depends on the named
system loader and libc; it is not a libc-free executable.

## Checked interface

```mind
pub fn nsa_emit(request: i64, symbols: i64, metadata: i64) -> String
```

`request` is the existing 80-byte `ndl_plan` request: ten i64 words containing
version, complete code length, metadata length, RELRO length, literal length,
arena bytes, overflow bytes, stack bytes, guard bytes, and external reserve.
The adapter recomputes the checked layout; it does not trust a supplied planner
success flag. See [the layout contract](native_dynamic_layout.md).

`symbols` is 80 readable bytes, in this order:

| Word | Meaning |
| --- | --- |
| 0 | Schema version 1 |
| 1 | Startup blob offset within complete code |
| 2 | Allocator blob offset within complete code |
| 3 | Allocator blob size |
| 4 | Callable allocator-init offset relative to allocator blob |
| 5 | Ordinary returning source-main offset within complete code |
| 6 | Source-main blob size |
| 7 | Main parameter count: 0, 1, or 2 integer/pointer GP arguments |
| 8 | Return tag 7, the existing ABI metadata tag for i64 |
| 9 | `__libc_start_main` import index, 0 through 255 |

All offsets are bytes, not virtual addresses. Each complete nonempty range
must fit the checked code length; the three ranges must be pairwise disjoint.
The init entry must lie inside its allocator blob. The checked low code window
makes relative calls fit signed rel32 without overflow. Unsupported signatures,
negative values, overflow-sized ranges and bad layouts refuse before returning
code. No varargs, callbacks, arbitrary aggregate main, Win64, or AArch64 ABI is
admitted by this interface.

`metadata` is 64 writable bytes. Its eight i64 words on success are:
`[1, entry_offset, adapter_offset, blob_size, libc_call_displacement_offset,
import_index, context_schema_version, 0]`. Offsets are relative to the returned
blob. At this revision they are `[1, 0, 135, 365, 115, index, 1, 0]`.
All eight words are cleared before validation. Refusal returns an empty owned
String and zero metadata. Allocation failure for the layout scratch also
refuses. General allocation failure safety of every String helper is not claimed.

Inputs must be live and immutable for the call; the output buffer must not
overlap either input. These raw-address preconditions are caller obligations,
not validated memory capabilities. The caller must bind the offsets, init ABI,
main signature, and import name/version to the actual selected code. Well-formed
integers alone do not establish that binding. The fixture makes this binding
explicit using allocator metadata, checked image geometry and named import rows.

The caller owns the returned String and passes its FF15 displacement record to
the checked dynamic writer. Allocator/init offsets come from allocator metadata,
not a copied historical runtime offset. Moving the three blobs is supported and
tested. The startup blob's own symbol offsets are tied to the zero-relocation
assembly reference by the diagnostic, including its complete byte sequence.

## Startup and main semantics

The initial entry saves the unmodified kernel RSP, argc, argv, original envp and
the loader-provided RDX finalizer. It validates an initially zero context marker
and phase, bounds argc by `INT_MAX`, aligns the original stack, and calls
`__libc_start_main`. The call supplies the adapter, argc/argv, zero init/fini,
original `rtld_fini`, and stack-end argument. A return from this non-returning
libc routine traps. Invalid entry state exits 70 before libc initialization.

The main adapter preserves the C callee-saved registers and balanced C stack.
It checks the context and libc argc/argv against their kernel originals, saves
libc's current envp separately, and calls the allocator's returning initializer.
Only a fresh-init result of zero is accepted. A repeated/corrupt initialization
returns 70 through libc instead of resetting a live heap.

After init, the adapter writes argc, argv and current envp at arena offsets
8, 16 and 24. These are reserved cells: the allocator cursor occupies offset 0,
and its first allocation header/payload begin at 64/72. It then saves C RSP,
switches to the checked dedicated MIND stack, and calls ordinary returning main.
The source function may clobber internal GP registers: the adapter rematerializes
the context address, restores the saved C stack and registers, and returns the
low 32 result bits as C `int`. It does not execute a raw exit syscall on this path.
Libc therefore flushes stdio and invokes registered finalizers normally.

The adapter passes argc and argv GP registers for all three accepted arities;
functions declaring fewer parameters ignore the unused registers. It does not
convert an existing exit-only compiler entrypoint into a returning function.
Production integration must select/emit the correct ordinary function body.

Context schema 1 is separate from RELRO and allocator state:

| Byte offset | Value |
| --- | --- |
| 0 | Marker `0x3154524154534d` |
| 8 | Phase: 1 entered, 2 source active, 3 returned, 4 adapter refused |
| 16 | Original kernel RSP |
| 24 | Loader finalizer |
| 32 / 40 / 48 | Kernel argc / argv / original envp |
| 56 | Saved C RSP |
| 64 | Current libc envp |
| 72 | Original i64 source result |

A constructor can replace libc's environment vector before main; the original
kernel vector and current libc vector remain distinct. This is a single-thread,
one-entry adapter. It is not a thread-local context implementation or a security
boundary against arbitrary source writes. Reentry is refused. Guard pages bound
the initial stack mapping, not the program's total stack use. The lower-guard
fixture intentionally observes a fault; such a fault never counts as mutation
or ordinary refusal success.

The ABI comparison uses the upstream [x86-64 startup contract](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/x86_64/start.S)
and [`exit(main(..., __environ))` path](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/generic/libc_start_call_main.h).
The assembly reference is an independently written STARGA test instrument;
no glibc implementation source is incorporated.

## Reproduction

The checked gate requires explicitly selected, hash-verified instruments. Do not
substitute a historical frozen seed that cannot compile the current diagnostic.
The current component's dependency commits may be sibling branches until the
integration merge; pass their actual committed paths rather than silently using
older files from the checkout.

```sh
export PATH=/usr/lib/llvm-20/bin:$PATH
export MLIR_OPT=/usr/lib/llvm-20/bin/mlir-opt
export MLIR_TRANSLATE=/usr/lib/llvm-20/bin/mlir-translate
export CLANG=/usr/lib/llvm-20/bin/clang
export MIND_STDLIB_PATH="$PWD/std"
python3 examples/mindc_mind/native_libc_startup_check.py \
  --seed "$CURRENT_NATIVE_COMPILER" --seed-sha256 "$CURRENT_NATIVE_SHA256" \
  --mindc "$OFFICIAL_MLIR_COMPILER" --mindc-sha256 "$OFFICIAL_MLIR_SHA256" \
  --allocator "$ALLOCATOR_MODULE" --allocator-sha256 "$ALLOCATOR_SHA256" \
  --image-writer "$IMAGE_MODULE" --image-writer-sha256 "$IMAGE_SHA256" \
  --base-writer "$BASE_WRITER_MODULE" --base-writer-sha256 "$BASE_WRITER_SHA256" \
  --evidence "$NEW_EVIDENCE_DIRECTORY"
```

Run inside a 2 GiB aggregate cgroup with swap disabled and a one-CPU quota and
explicit affinity. Each child also has a 2 GiB address-space cap, CPU/time caps,
and core dumps disabled. The gate uses a fresh output directory and records
commands, statuses, stdout/stderr, source and artifact hashes. The caller records
cgroup peak, limits and OOM events separately. The ELF runtime gets a minimal
environment; inherited LD settings are removed. The diagnostic preload is used
only for the four named constructor/finalizer cases and corresponding mutants.

Frozen contract: 45 rows, SHA-256
`44ab600c4edd01a31871e5a05cf55330a6bc47588511d4cef4d857d8d506c8fb`.
The original pre-enumeration used terminal empty TSV fields (hash
`4a172a7da528f033c6cd3ae79db8db640904785562c2f05b0d3ec96693fb8945`). For whitespace-clean Git packaging those fields
now use `-`, decoded as empty bytes. All 45 decoded rows were compared exactly
and are unchanged; the original file and comparison receipt are retained.

Five source mutations: SHA-256
`e08d8c3d44aa8da54197f8afeb65fcd4efc62d8a1d06f31d2863f3c700debbd9`.
The first run reached a normal unsupported-construct build refusal because the
harness incorrectly included the stdlib prefix in user roots. It was corrected
to the established prefix-length boundary without changing any fixture. A later
independent expected-byte check incorrectly treated `__libc_start_main` as the
first sorted import, overlooking `__errno_location`; the expected GOT slot was
corrected. Both failed diagnostic receipts are retained separately. Neither was
an observed product runtime defect or a relaxed expected runtime outcome.

Observed on the scoped component: all 45 native/MLIR pairs matched; 19 include
real dynamic-image executions, and 26 require normal emitter refusal with zero
metadata. Native and MLIR emitted byte-identical images. The reference checks
complete startup bytes, local calls and the libc GOT target, ELF permissions,
context/arena/stack geometry and guards. All five actual MIND source mutations
built and ran, producing the predeclared normal wrong results: missing stdio
flush, missing loader finalizer, ignored init refusal, wrong stack, and overwritten
original envp. Build failures and signals are rejected as mutation detections.

This evidence does not qualify a compiler pin, prove arbitrary C ABI integration,
establish production TLS transport, or validate every libc/loader version.
Independent component review and production integration remain separate gates.
