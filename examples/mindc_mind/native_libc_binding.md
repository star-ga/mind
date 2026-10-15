# Source-bound libc declaration metadata

Author: STARGA Inc.

This standalone MIND module matches original source extern declarations to a
versioned Linux x86-64 LP64 SysV libc contract. It does not emit a foreign call,
select a native profile, authorize a compiler pin, or resolve call arguments.
No production compiler source is modified by this component.

The contract recognizes 31 fixed-signature candidates, four genuinely variadic
library functions (`open`, `openat`, `fcntl`, `prctl`), and one callback-shaped
function (`signal`). The source declarations of those four variadic functions
have fixed supplied parameter lists. Both facts are retained. Every result has
`callready=0`; merely matching a declaration never authorizes calling it.

## Source authority and ownership

```mind
ncb_build(source, length, row_index, target_profile, diagnostic32) -> owned_plan
ncb_check(plan, readable_length, source, length, row_index, target_profile,
          diagnostic32) -> error_code
ncb_free(owned_plan) -> 0
```

`ncb_build` first validates profile and bounds, copies the immutable source bytes
into owned storage with eight initialized padding bytes, and reparses them with
the existing `nb_cabi_build`. It selects the actual original-source row, checks
its complete source signature and library contract, then encodes the registry
using `nb_cabi_encode`. The returned plan owns copies of both the source and the
complete canonical registry bytes. Registry/snapshot/temporary buffers are
released; the returned allocation contains no borrowed pointers.

`ncb_check` rebuilds the complete binding from the current source and compares
all bytes and the total length. It never trusts a caller's digest, registry,
contract ID, origin, readiness flag, library version, type table or success flag.
An exact plan from different source bytes, even a changed comment, refuses.
A modified source row or profile also refuses. Its checked readable length is
bounded before comparison; it does not parse untrusted pointer fields because
none exist in the flat plan. A generated-origin API is deliberately absent.

Raw input pointers must still reference readable live allocations; diagnostic
buffers must be writable and nonoverlapping with inputs. Source storage must
remain immutable during capture. Neither function can prove arbitrary pointer
validity or race freedom. The existing compiler parser's allocation lifetime is
unchanged; releasing a binding is not a claim that every parser scratch
allocation is individually reclaimed. Native OOM policy also remains unchanged.

Source length is at most 4 MiB, canonical registry at most 8 MiB, complete binding
at most 16 MiB. Bounds use subtraction before additions and allocation. Every
encoded byte is initialized, with no copied allocator padding. Copying and
reparsing a whole source per declaration is deliberate in this component.
A future collector may own and share an immutable source capture; it must
preserve the same binding checks instead of replacing them with a hash flag.

## Contract and exact encoding

Target1/revision1 fixes ELF64 little-endian Linux x86-64 LP64 SysV,
`libc.so.6`, and `/lib64/ld-linux-x86-64.so.2`. Symbol versions are explicit:
`close_range`2.34; `fstat`/`fstatat`2.33; `memfd_create`2.27;
`clock_gettime`2.17; `pipe2`2.9; `openat`/`renameat`/`unlinkat`/`fdopendir`2.4;
all other recognized user names2.2.5 (each prefixed `GLIBC_`). This does not
claim an oldest-supported-host run or trust the current host's default version.
Generated runtime imports such as `__libc_start_main`, read/write thunks and
startup addresses require a separate internal recipe and binding path.

The human-readable contract table is
[native_libc_binding_contracts.tsv](native_libc_binding_contracts.tsv).
It contains standard library contracts and generic source spellings, without
consumer-specific paths or private application data. Safety and explicit C
versus SysV convention are preserved as source facts. `safe` is not a proof that
a libc call is safe. Source aliases, conflicting names and unresolved syntax
remain governed by the source registry and its full-resolver integration
obligations.

The flat format is diagnostic/internal metadata revision1, not a general public
wire codec. Header:24 little-endian i64 words,192bytes:

```text
MCBND001, total_size, profile=1, contract_revision=1, declaration_index,
origin=0(USER), callready=0, class(1fixed/2variadic/3callback),
actual_library_variadic, library_fixed_prefix, nonreturning,
source_convention, source_safety, source_varargs, source_parameter_count,
source_byte_length, canonical_registry_length, symbol_length, version_length,
SONAME_length, interpreter_length, slot_count, contract_id, reserved=0
```

Slots follow in result-first, then source-parameter order. Each slot has five
i64 words: source prefix type, library prefix type, compatibility action,
semantic role, unresolved obligation mask. A packed prefix is
`(length << 32) | little_endian_prefix_bytes`, length1–3 for this contract.
Source codes are the existing registry prefix codes. The library metadata adds
opaque leaves64=DIR,65=dirent,66=stat,67=timespec,68=rlimit,69=siginfo,
70=unresolved function pointer,71=unresolved variadic argument. These are
metadata types, not new language types or inputs accepted by `ncp_plan`.

Compatibility actions:0 identity;1 source i32 to library u32 low32 view;
2 library u32 result to source i32 low32/sign-extended view;
3 character-pointer qualifier/pointee view;4 unresolved callback view;
5 unresolved variadic argument. Actual vararg tails retain their source types
while leaving the library argument representation unresolved. No false fixed
row is constructed to bypass the existing fixed-only C-call classifier.

Roles:0 scalar,1 fd,2 NUL path,3 stat buffer,4/5 input/output timespec,
6/7 input/output rlimit,8 owned DIR result,9 borrowed DIR input,
10 borrowed dirent result,11 argv,12 envp,13 borrowed thread-local errno pointer,
14 two-int pipe buffer,15 wait status,16 siginfo buffer,17 signal number,
18 callback/sentinel,19 mode bits,20 uid bits,21 flags,22 selector,23 pid,
24 offset,25 NUL name. These roles retain obligations; they do not validate a
buffer's size, alignment, ownership, lifetime, contents or nullability.

Obligation bits:1 checked call arguments(required for every slot),2 varargs,
4 callback,8 pointer role,16 explicit bit-view conversion,32 nonreturning effect.
No declaration match resolves these bits. `_exit` remains explicitly nonreturning.
A future call checker must validate source-bound arguments and resolve applicable
obligations before creating any executable-ready plan.

After the slots: exact source bytes, exact full MCABI001 descriptor, symbol,
version, SONAME, interpreter. Lengths are in the header; no terminal NUL or
uninitialized padding is included. Unknown or changed fields fail full-plan
reconstruction comparison. `ncb_free` frees the one owned allocation.

The checked entrypoints `ncb_build` and `ncb_check` initialize Diagnostic32, which contains
`[error, source_byte_offset, underlying_registry_error, 0]`.
Errors:1 invalid input/length,2 target profile,3 source registry refusal,
4 row index,5 unknown symbol,6 signature mismatch,7 allocation refusal,
8 encoded-size refusal,9 stale or forged plan. Success is0. Diagnostic transport
errors are separate; a crash, build failure or timeout is never a valid refusal.
Other `pub` helpers are component-internal linking symbols with checked-input
preconditions; they are not hostile-input APIs. In particular, `ncb_error` stores
a diagnostic into an already initialized buffer, and `ncb_free` requires a live
owned plan returned by `ncb_build`.

## Low32 views and remaining scope

`getuid`, `umask`, `memfd_create` flags and `waitid`'s id_t explicitly differ in
source/library signedness. GNU-target resource-limit selector enums and waitid's
idtype_t are also unsigned32; a local C compatibility probe confirmed their
4-byte unsigned-int representation. Their source i32 views remain explicit.
`ncb_low32_view` tests the two defined scalar conversion actions independently;
it does not establish argument validity or authorize a foreign call. The
independent C reference uses uint32_t conversion and a byte-preserving int32_t
view, including high-bit boundaries.

No arbitrary callback registration, variadic promotions, AL count, fcntl command
selection, prctl tuple, signal sentinel, foreign-function-pointer reuse, library
loading, import collection, deduplication or call emission is implemented here.
The registry's exact source declarations and complete dependency capture remain
authoritative. Current host measurements do not prove cross-host compatibility.

## Composition and testing

Direct compiler dependencies are `nb_cabi_build`, `nb_cabi_encode`, and
`nb_cabi_free`; String address/length and memory primitives supply their storage
operations. Product composition must use the existing captured-source/module
closure machinery. The diagnostic's scratch concatenation of the standard
manifest prefix, immutable compiler source, this module and its generic wrapper
is not a production integration mechanism.

The gate first enumerates its cases into a read-only hash-locked catalogue,
then builds source-bound native and MLIR diagnostics. It records the complete
standard-library MIND tree, seed manifest and reader, product and fixture source,
compiler and selected build-tool hashes. Both routes use this checkout's `std`
directory; inherited std paths and compiler override variables cannot substitute
other inputs. Input hashes and std membership are checked before and after the
builds and after execution. Built executable hashes and exact source packets are
recorded separately. This detects ordinary source drift; it does not prove
absence of adversarial concurrent file replacement or a compromised host. All successes and diagnosed
refusals must equal explicit independent expected byte sequences; matching wrong
backend results cannot pass. The host harness and C reference are test
instruments, not hidden MIND runtime dependencies or theorem authorities.
Independent source/implementation review, final component execution evidence,
and production compiler qualification are separate gates.

The 356-case catalogue includes exact expected output bytes before execution;
its final SHA-256 is
`ef7c7a868e1d494481257cf4e8afb64f72e44a853dbc8ae6b720436a80838151`.
It covers all recognized declarations under C, safe and explicit SysV spelling;
incorrect counts/results/parameter types and pointer qualifiers; unknown symbols;
source/row/profile changes; every header word, selected slot fields and payload
bytes; attempted variadic readiness bypass; plan/source lengths; released source
ownership; and low32 boundary conversions. Actual call admission is not tested
or implied by these declaration matches.

```sh
export PATH=/usr/lib/llvm-20/bin:$PATH
export MLIR_OPT=/usr/lib/llvm-20/bin/mlir-opt
export MLIR_TRANSLATE=/usr/lib/llvm-20/bin/mlir-translate
export CLANG=/usr/lib/llvm-20/bin/clang
export MIND_STDLIB_PATH="$PWD/std"
python3 examples/mindc_mind/native_libc_binding_check.py \
  --enumerate --evidence "$NEW_CASE_DIRECTORY"
python3 examples/mindc_mind/native_libc_binding_check.py \
  --evidence "$NEW_CASE_DIRECTORY" \
  --seed "$CURRENT_SOURCE_BOUND_NATIVE_COMPILER" --seed-sha256 "$NATIVE_SHA256" \
  --mindc "$OFFICIAL_COMPILER" --mindc-sha256 "$OFFICIAL_SHA256"
```

For actual source mutations, enumerate a second evidence directory and invoke
the same gate with the already built, hash-bound driver artifacts plus the exact
original source-build packet and the original full build-evidence directory.
Reuse requires exact current product/compiler/std/tool input identities, source
packet, MLIR source and catalogue, not only executable hashes. A runner-only
revision may differ; producer and runner hashes are recorded explicitly, and the
original runner snapshot must match its recorded hash. No other missing or
renamed dependency is ignored. This mode re-executes all 356 pairs and 14 independent
C view comparisons; it does not claim to rebuild the unchanged original driver.
It separately rebuilds all five altered MIND emitters and requires normal
incorrect acceptance/output. Crashes, compilation failure and nonzero process
status never count as successful mutation detection.

```sh
python3 examples/mindc_mind/native_libc_binding_check.py \
  --enumerate --evidence "$NEW_MUTATION_DIRECTORY"
python3 examples/mindc_mind/native_libc_binding_check.py \
  --evidence "$NEW_MUTATION_DIRECTORY" \
  --source-evidence "$ORIGINAL_BUILD_EVIDENCE_DIRECTORY" \
  --mindc "$OFFICIAL_COMPILER" --mindc-sha256 "$OFFICIAL_SHA256" \
  --native "$BUILT_NATIVE_DRIVER" --native-sha256 "$DRIVER_NATIVE_SHA256" \
  --oracle "$BUILT_MLIR_DRIVER" --oracle-sha256 "$DRIVER_MLIR_SHA256" \
  --seed "$CURRENT_SOURCE_BOUND_NATIVE_COMPILER" --seed-sha256 "$NATIVE_SHA256" \
  --mutation-source "$ORIGINAL_BUILD_PACKET" \
  --mutation-source-sha256 "$BUILD_PACKET_SHA256"
```

Use an aggregate 2 GiB cgroup, swap disabled, one-CPU quota and explicit affinity.
Children also enforce 2 GiB address-space, CPU/time and disabled-core limits.
The gate logs exact commands, source/build packet, stdout, stderr and statuses.
The caller records actual cgroup limits, peak and OOM events separately. Its
new C file is an independent diagnostic tool, not a product runtime dependency.

The first candidate run retained a test-category mismatch: primitive type
shadowing was correctly refused by both backends as registry error3 (unsupported
type), while the test expected error9 (namespace collision). Source registry
logic and a separate paired reproduction confirmed error3 at the same source
location. The negative remains mandatory with the corrected diagnostic category.
A second metadata refinement explicitly marks real variadic tails unresolved
rather than assigning them source-shaped library scalar types. No case was
removed, no wrong declaration was accepted, and no resource cap was raised.

At implementation handoff, the 356 declaration cases, 14 independent C scalar
views, five actual MIND source mutations, and 71 per-function MIC emission checks
have passed. The mutations remove parameter checking, complete-plan comparison,
library-varargs metadata, the disabled call-ready flag, and owned source copying.
Every mutated checker builds and exits normally with incorrect acceptance or
metadata, which the unchanged expected outputs detect. No crash counts as a pass.

Independent review added 17 multi-declaration, changed-other-row, stale-row and
nested-pointer cases, all with exact native/MLIR outcomes. It also checked all
36 signatures against local libc headers and requested symbol versions against
the local library. Those measurements establish only the local target contract.
The reviewer identified missing harness provenance coverage; the final harness
adds complete std/input/output records and drift checks, followed by a fresh
build and unchanged catalogue run. Earlier evidence and the provenance correction
remain recorded separately; neither component testing nor review qualifies the
full production compiler or resolves variadic/callback/call-argument obligations.
