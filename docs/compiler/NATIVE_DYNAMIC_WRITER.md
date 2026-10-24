# Native dynamic ELF writer candidate

Author: STARGA Inc.

`examples/mindc_mind/native_dynamic.mind` is an isolated MIND library that
writes Linux x86-64 LP64 dynamic executables. It is not yet wired into the
compiler, an admitted native profile, or a qualified consumer pin. Static
compiler output is unchanged. This milestone is distinct from position
independent ET_DYN generation and the C8-DYN roadmap.

The product writer is MIND. The Python diagnostic harness composes test input,
checks bytes independently, and runs child processes; it is not part of the
generated executable. No host C compiler, assembler, or linker generates the
test executable.

## Interface and preconditions

```mind
nd_import(name: String, version: String) -> i64
nd_patch(displacement: i64, import_index: i64) -> i64
nd_validate(code: String, entry_off: i64, imports: Vec, patches: Vec,
            interpreter: String, soname: String) -> i64
nd_write_elf(code: String, entry_off: i64, imports: Vec, patches: Vec,
             interpreter: String, soname: String) -> String
nd_code_addr() -> i64
```

An import row is four i64 words: borrowed name address, name length, borrowed
version address, version length. A patch row is two words: displacement offset
within code and original import index. Vectors contain row addresses. All
descriptors and borrowed buffers must be real, live, immutable allocations for
the entire call. They must not be concurrently changed or freed. Numeric range
checks do not establish allocation ownership; forged arbitrary pointers are
outside this raw-memory API's contract. Allocation failure follows the native
runtime's checked failure path.

The caller owns raw instruction bytes and fixup provenance. A fixup must mark
the displacement of an actual `FF 15 disp32` instruction, with an initially
zero displacement. Recognizing these bytes cannot prove instruction boundaries:
the same bytes can occur inside an immediate. Likewise, an in-range entry offset
does not prove it starts a valid instruction. A reviewed typed emitter must
establish both obligations before this writer becomes a compiler backend.

The current scope is one explicitly supplied SONAME, versioned function imports,
and one interpreter path. It supports 1–256 unique imports, 1–4,096 patches and
1–1,048,576 code bytes. Every import needs at least one patch; unreachable calls
still count. Names and versions are bounded ASCII. There are no unversioned,
weak, data, callback, TLS-object, or multi-library imports in this version.
The library records a requested version; it does not invent an ABI or choose
a libc alias for a source name.

## Layout and determinism

The writer emits ET_EXEC at `0x400000`, code at `0x401000`, a bounded entry,
PT_INTERP, PT_DYNAMIC, DT_NEEDED, dynstr/dynsym, SysV hash, RELA, VERSYM and
VERNEED. Imports are sorted by symbol name, versions by version name. Original
patch indices are remapped without modifying caller inputs. Each import has
one zero-initialized GOT slot and an `R_X86_64_GLOB_DAT` relocation. There are no
lazy PLT resolvers: DT_FLAGS/BIND_NOW and DT_FLAGS_1/NOW request eager binding.

Separate page-aligned loads contain read-only headers, read/execute code,
read-only lookup metadata, and read/write GOT/dynamic data. The entire last
load is PT_GNU_RELRO. PT_GNU_STACK has no execute permission. No load is both
writable and executable. Layout arithmetic is bounded after input validation;
patch displacement and final size are checked before emitting a result.

Identical inputs produce identical bytes. Permuting import rows and adjusting
their indices produces the same output. Runtime virtual addresses obtained
from the loader may vary with ASLR; deterministic output does not promise
identical runtime addresses or portability across arbitrary libc versions.

## Refusal contract

`nd_validate` returns zero for structurally admitted inputs and these nonzero
reasons. `nd_write_elf` returns an empty String on refusal, before mutating inputs.

| Code | Reason |
|---|---|
| 1 | Code length/address/capacity or entry range |
| 2 | Import/patch counts or vector descriptor shape |
| 3 | Interpreter or SONAME text |
| 4 | Import descriptor or symbol/version text |
| 5 | Duplicate symbol name |
| 6 | Patch offset or import index |
| 7 | Wrong call bytes or nonzero displacement placeholder |
| 8 | Overlapping patch displacement bytes |
| 9 | Import without any patch |

A structurally valid artifact can still fail in the system loader because a
symbol or version is unavailable. That is a different outcome from writer
refusal. An expected protection fault in a dedicated probe is also not writer
refusal, and an arbitrary crash never counts as successful validation.

## Diagnostic reproduction

The explicit test composition uses the frozen stdlib manifest and copies five
small byte encoders verbatim from `main.mind`: `nb_push`, `nb_le`, `nb_u16`,
`nb_u32`, and `nb_i64le`. It appends the library and its generic test fixture.
This is a diagnostic input to the existing native v2 protocol, not a new
production source-composition pipeline. The future module integration must use
the compiler's approved source/dependency mechanism.

Supply an independently identified native compiler and its exact SHA-256:

```sh
export MIND_DYNAMIC_SEED=/absolute/path/to/native-compiler.elf
export MIND_DYNAMIC_SEED_SHA256=<64-lowercase-hex-digits>
export MIND_DYNAMIC_EVIDENCE=/absolute/path/to/new-evidence-directory
python3 examples/mindc_mind/native_dynamic_writer_check.py
```

Run in an outer resource group with 2 GiB memory, zero swap and one CPU. Child
processes additionally receive a 2 GiB address-space cap and disabled core
dumps. The harness refuses an incorrect seed hash and system loader preloads;
it removes all `LD_*` variables and `GLIBC_TUNABLES`. It records the interpreter,
resolved libc and loader-cache hashes, libc version output, exact composed
source/input, compiler and product hashes, child stdout/stderr/exit results,
and an execution ledger. These host dependencies are explicit, not bundled
or silently replaced. The synthetic fixture needs `GLIBC_2.34` for
`__libc_start_main` and `GLIBC_2.2.5` for `getpid`.

The suite checks source-byte fidelity before calling the writer, deterministic
emission and input non-mutation, independent ELF layout/relocations/version
tables, real libc startup with argc and getpid, 25 malformed inputs with exact
normal refusal reasons, missing symbols and versions, an unreachable missing
import that must fail eagerly before main, maximum import/patch/code limits,
and runtime RELRO and NX. Both protection tests must successfully output a
memory probe before the expected SIGSEGV; a crash before that probe fails.
An independently corrupted segment layout must also be rejected by the
Python structural checker without being executed.

The first diagnostic expected exit 127 for both missing symbols and missing
versions. Observed glibc behavior was 127 for symbols and 1 for versions; that
test-contract correction is explicit. Subsequent NX diagnostics exposed a
compiler provenance-note bug: `hm_mark_source_note` used `artifact_length-44`
after read-only literal data had been appended, changing a literal byte. The
failed evidence is retained. Golden raw-byte checking and strict fixture hex
decoding guard this boundary; padding literals to hide the defect is not a fix.

### Observed isolated validation, 2026-10-15

The complete suite passed using a corrected, **unqualified diagnostic compiler**
with SHA-256
`b976cb9ad77a6fcd2b749645ba90bb037cd9e69327588c2e11a29b8448df5dd0`.
The MIND writer source hash was
`468dc87cdb41da33939d23696eda9ebd980ff75bae91cd82cf5c2e68367081aa`.
The emitted getpid executable was byte-identical to earlier runs:
`abcde9e33d6062c32acb67cb20019f17e6afaa57f3316b7264de26e4f68adfaa`.

All 25 malformed-input controls, four loader negatives, two executed size
boundaries and both observed memory-protection faults passed their respective
contracts. Real libc startup/getpid returned the expected argc for both argument
sets. The independent structural validator rejected an overlapping-load
mutation. Raw source bytes and repeated/permuted artifact bytes matched.
The execution ledger has 76 subprocess observations, including generation,
diagnostics and repeated calls; this is not 76 distinct behavioral tests.

The test host used glibc 2.43. Peak cgroup memory was 92,405,760 bytes under the
2 GiB/one-CPU/zero-swap limits, with zero memory-limit or OOM events. A separate
run of the preserved minimal compiler-bug packet returned the exact original
62 literal bytes and verified type 2 at the actual PT_NOTE location. Historical
failed runs remain evidence of the defect and the corrected test expectation.
These observations validate this isolated milestone on that host; they do not
qualify the compiler, establish cross-host portability, or complete typed
extern admission and consumer parity.

## Remaining integration obligations

1. Retain exact source-level C types, calling convention, varargs and pointer
   shape through validation and call lowering; this writer supplies none of
   those semantics.
2. Supply reviewed libc startup preserving argc/argv and the loader finalizer,
   stack alignment, ABI register preservation and the return/exit contract.
   The test startup and main are independently written instruction bytes only.
3. Relocate real compiler call sites for the new `0x401000` code base. Existing
   `nb_hdrs` and absolute native runtime addresses cannot be reused unchanged.
4. Add MIND arena/state and literal data with separate permissions, allocation
   ownership and lifetime contracts. The current writer has no such segments.
5. Resolve libc TLS/errno, native stream I/O, raw syscall and allocation
   ownership interactions. Raw `-errno` std.posix behavior must remain distinct
   from libc `-1` plus thread-local errno.
6. Exercise typed ABI microcases and unchanged static regressions before any
   profile admission or native/MLIR consumer qualification. Shared objects,
   callbacks and concurrent threads need their own capability contracts.

ABI references: [glibc x86-64 startup](https://github.com/bminor/glibc/blob/glibc-2.39/sysdeps/x86_64/start.S),
[libc startup](https://github.com/bminor/glibc/blob/glibc-2.39/csu/libc-start.c),
and the [x86-64 System V ABI](https://gitlab.com/x86-psABIs/x86-64-ABI/-/raw/master/x86-64-ABI/low-level-sys-info.tex).
This implementation is original STARGA code. ABI documentation supplies the
contract, not copied implementation code or theorem authority.
