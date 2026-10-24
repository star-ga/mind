# Explicit target context and source-lowered module parts

Author: STARGA Inc.

This development interface connects MIND source lowering to the reviewed Linux
x86-64 dynamic-image components. It is not selected by the production CLI and
does not qualify a compiler pin. Tests and independent implementation review are
in progress. Historical static entry points remain compatibility wrappers with
context zero; their existing gates and seed-parity obligation are unchanged.

The target context describes the emitted program. It never controls the running
compiler's arena, stack, checkpoint or low-memory guards. A CPU-backed compiler
can therefore emit the same target addresses as a native compiler without
reading or rewriting a process-global native layout cell.

## Owned context

`native_emission_context.mind` uses the existing checked `ndl_plan` geometry:

```mind
pub fn nec_new(request: i64) -> i64
pub fn nec_check(ctx: i64) -> i64
pub fn nec_free(ctx: i64) -> i64
```

The request is the documented ten-word dynamic-layout request. `nec_new` copies
it and recomputes its plan; invalid requests return zero. The returned 416-byte
block owns its request and all derived values. `nec_check` builds a fresh expected
record from that owned request and compares all 52 words. A magic number or a
stored success flag alone is insufficient. A coherent different valid request
remains a valid context; this record is not authorization or source provenance.

| Offset | Field |
|---|---|
| 0, 8 | Schema 1 or 2 and dynamic-target mode 1 |
| 16 | Allocator state address |
| 24, 32 | Reserved argc and argv cell addresses |
| 40, 48 | Literal base and available 8 MiB window |
| 56, 64 | Arena base and target stack top |
| 72 | Zero for schema 1; generated stream-thunk base for schema 2 |
| 80 | Owned ten-word request |
| 160 | Recomputed 32-word layout |

These are live raw-address compiler APIs. The caller must provide valid readable
allocations, immutable inputs during each call, and exactly-once ownership.
They do not validate arbitrary pointers or parse untrusted serialized records.
`nb_ctx_shape_ok` only checks necessary emitter-field invariants. Public component
orchestration must call `nec_check`; the internal shape check is not a substitute
for complete geometry recomputation.

`nec_new_profile(request, schema, stream_base)` and
`nec_lower_parts_profile(..., schema, stream_base)` provide the internal schema-2
variant used by source-image orchestration. The original entry points select
schema 1 with a zero stream base. Schema 2 restricts the base to the code window;
the image orchestrator must additionally bind it to the actual generated runtime
placement and final image plan. A caller-selected schema or address does not
authorize a foreign call.

## Lowering and entry semantics

The context is explicit at module and function boundaries, then carried at
`lcell+120`. All nine normal, count and diagnostic cells are 128 bytes with that
slot initialized. Both HOME and SSA routes use the same context. Allocation,
reallocation, free, argument loads, literal handles and implicit aggregate
allocation use its target addresses. Runtime calls retain the reviewed slots
536, 544 and 552. Context-zero targets retain their historical address and
entry policy; byte compatibility is subject to the separate static regression
gate, not inferred from this interface design.

Entry kinds are separate from target mode: zero is an ordinary function, one is
a historical kernel entry, and two is a libc-called source main. Kind two keeps
main-signature checks, returns normally and does not capture the kernel stack or
execute an exit syscall. Invalid context/entry combinations refuse. The first
source slice requires exactly one main, an explicit i64 result and zero, one or
two integer/pointer arguments. Aliased/aggregate/unit/float results, callbacks,
variadics and unsupported signature syntax remain refused.

The schema-1 dynamic slice permits arithmetic, memory, argument and supported
float-bit intrinsics. Schema 2 also connects the four-operand stream intrinsics
through the checked libc I/O recipe described in
[`native_source_image.md`](native_source_image.md). Open and raw process/system-call
intrinsics remain refused. Dormant top-level
extern declarations are checked block by block with the existing declaration
grammar. Every block must match its AST span; local-body/name collisions and
primitive-type shadows refuse. A full token census rejects extern declarations
hidden in an unsupported wrapper. No declaration creates an executable import:
every reachable nonintrinsic call must still resolve to a local body.

The flattened standard-library prefix contains repeated declarations across
bricks, including different opaque-pointer views. Independent checking of these
unused blocks does not select an ABI, reconcile conflicting declarations or
authorize a foreign call. Connecting real extern calls requires source-bound
typed declaration resolution. `user_lo` is never a bypass for declaration checks.
The main signature separately receives the whole-source primitive-shadow check;
a source alias named `i64` cannot pass as the primitive merely by its spelling.

## Source-owned parts

```mind
pub fn nec_lower_parts(source: i64, length: i64, hash: i64,
    user_lo: i64, route: i64, request: i64) -> i64
pub fn nb_parts_local_calls_ok(parts: i64) -> i64
pub fn nb_parts_patch_local(parts: i64) -> i64
pub fn nb_parts_free(parts: i64) -> i64
```

`nec_lower_parts` captures the source before lexing and enters one common parser,
reachability, alias, route, poison and call-resolution pipeline. It does not
compile an ELF and scrape it back into pieces. The existing static wrapper uses
that same lowering pipeline and its original runtime prefix, call patching and
ELF writer. Unsupported functions poison the entire result.

The existing 32-byte note input is supplied by the source-note driver. It is not
an authenticated digest or a proof claim. `user_lo` and routing retain their
existing compiler-owned semantics. The input length is bounded to 64 MiB in the
new dynamic entry. No filename, supplied success flag or theorem identifier can
select a less restrictive lowering path.

The returned 192-byte record owns these values:

| Offset | Field |
|---|---|
| 0 | Schema 1 |
| 8, 32, 56 | Three-word code, literal and note Strings |
| 80, 88 | Function-start and source-name-span tables |
| 96, 104 | Local E8 call records and owner-index table |
| 112, 120 | Function and local-call counts |
| 128, 136 | Original main index and checked main parameter count |
| 144, 152, 160 | Captured source address, length and ownership flag |
| 168 | Local relocation applied flag |
| 176 | Owned emission context |
| 184 | Source-derived stream-use bit, zero or one |

The stream-use collector traverses the same reachable source functions that
are lowered. With no collected use, body lowering receives schema 1, so a missed
unsupported AST shape cannot silently emit a stream-thunk reference without its
imports. Schema-1 and context-zero paths retain zero in this additional word.
The record is an internal live allocation, not a public serialized wire format.

Name spans resolve against the retained source copy, so freeing the original
input cannot invalidate them. Internal AST/count scratch is still subject to the
existing compiler allocation/checkpoint lifetime; this interface does not claim
complete reclamation for an unlimited sequence of in-process compilations.

Local relocation validates all function intervals, source spans, owner/callee
indices, E8 displacement bounds and signed rel32 reach before writing any code.
It operates on the unprefixed body: uniform placement of the body leaves local
relative offsets unchanged. Typed FF15 import relocations remain a separate
writer responsibility. An opcode-byte check does not prove arbitrary x86
instruction boundaries; records must originate from the trusted emitter.

`nb_parts_free` destroys an unconsumed record exactly once. A consumer taking a
String for destructive concatenation must first clear or update its stored
three-word descriptor. Borrowed table/byte views must not outlive their parts.
The scalar main index/arity and retained byte/span tables contain no dangling
AST pointers.

## Validation status

The first helper-shape diagnostic refused four new functions normally. Those
receipts are retained. Splitting them into checked scalar/row helpers preserved
the validation rules; the next isolated-file diagnostic accepted 39 of 40
new/refactored helper bodies. The declaration-scoped revision accepted 41 of
42. `nb_parts_string` returned no emission in both isolated diagnostics, although
the earlier full-source native and CPU driver builds succeeded. An independent
probe of that exact helper with the complete captured standard-library prefix
emitted successfully: the raw main-file probe omitted the `String` definition.
The isolated refusal remains recorded separately. An earlier 40/40 summary was
incorrect; the structured per-function results take precedence. None of these
helper diagnostics establishes runtime or semantic qualification.

The first full context run recorded 670 paired outcomes: 586 passed and 84
source-positive cases refused identically on both backends because a blanket
extern prohibition also rejected dormant standard-library declarations. Those
receipts remain intact. The declaration-scoped correction described above and
six primitive-shadow cases are being checked. The original unused-user-extern
refusal expectation is preserved separately from the explicitly documented
new dormant-declaration admission; it must not be silently rewritten.

Independent static baseline capture found that both prior layouts refused the
fixed-array field fixture. Further inspection identified a conflicting standard
library `Box` declaration. Renaming only the fixture's three type identifiers to
`ArrayFixtureBox` produced the expected exit 5 on both drivers, target layouts
and routes. Original refusals remain recorded; the corrected positive control
is a separate versioned fixture, not a changed downstream acceptance case.

Source-lowered image execution is described in
[`native_source_image.md`](native_source_image.md). Independent adversarial
review subsequently exposed a function-registry capacity bypass, malformed
parameter admission and pointer aliases bypassing cast restrictions. The
capacity correction passed the original 80 independent observations. Parameter
grammar and pointer-origin corrections have received scoped independent review.
Candidate 2 rejects the observed pointer-through-dereference laundering cases
and accepts the corrected scalar-shadow control. The original conservative
shadow expectations and unsupported positive pointer-alias/HOME-reference
cases remain separately visible: the frozen independent contract records
652/672 expected outcomes, including the superseded conservative-shadow policy.
This is not an all-green contract verdict.
That candidate's fresh whole-source CPU build and scope gate passed, but its
whole-module MIC parity gate returned no self-hosted IR. Those failed receipts
are retained. The same candidate reproduced identical native compiler bytes across three
stages and matched all 32 frozen static comparisons, including two retained
baseline refusals. The original source-note diagnostic build exceeded its
120-second CPU budget. An explicitly separate build-only 300-second allowance
completed that exact input in 126.90 seconds; all 43 structured note fixtures
and three literal regressions then passed, 46 checks total, with their original per-case
limits. The original timed failure remains recorded and no downstream campaign
budget was changed. Reproduction can explicitly select
`self_host_source_note_smoke.py --build-cpu-seconds 300`; the default remains
120 seconds. The option changes only the two initial compiler/diagnostic-driver
builds. All fixture builds and executions retain CPU120/wall150 limits.
CI explicitly selects the 300-second build allowance for this gate. This is a
documented build-budget change; the original default-budget failure remains
part of the validation record.

The subsequent record-normalization correction permits canonical MIC emission
for an unknown simple record schema even when unrelated record declarations
exist. This matches the reference's source-order opaque-record layout. Known
layouts that the word-per-field normalization does not model, including narrow
scalar fields and fixed-array fields, cannot use that unknown-schema fallback.
Native/pruned routing retains its existing unknown-schema admission policy.

That change exposed a separate value-reference defect: a descriptor-slot
environment had been passed to a record flattener expecting resolved SSA IDs.
Local call results could therefore serialize the wrong reference despite equal
artifact lengths. Separate descriptor-environment record helpers now resolve
field expressions through the existing descriptor cache; resolved-ID callers
retain their own path. The ordinary expression and array-element paths share
the corrected behavior. Eight fixed regression sources preserve the two former
wrong-output cases and six controls without changing the original whole-module
comparison.

For source SHA-256
`f0ba4ea0f61012fee90cda4e254ad975b523074d946c89285e31da26a8d9a83d`,
a fresh CPU oracle produced exactly the same 840,515-byte whole-module MIC as
the reference compiler. All eight record regressions, 28 scope cases with
25 live comparisons, and 46 source-note checks passed. Three native self-build
stages reproduced the same compiler bytes, and all 32 static comparisons matched,
including the two retained baseline refusals. These results establish the listed
regressions only. Independent validation of the latest record-boundary changes
is incomplete; constructor-free narrow-field reads remain an explicit review
obligation. Unsupported pointer-alias/HOME-reference positive cases and the
original conservative-shadow expectations remain separately reported.

The final source-context component run retains its original expectations:

| Contract | Expected outcomes observed |
| --- | --- |
| Original component checks | 662 / 674 |
| Original runtime checks | 224 / 240 |
| Separate pointer-origin supplement | 388 / 400 |

All 28 original failure identities match the prior run: the conflicting `Box`
fixture and changed dormant-extern policy remain visible. The 12 supplemental
failures are unsupported positive alias-parameter and HOME dereference-store
cases. All 264 negative pointer observations passed. Native and CPU drivers
produced identical bytes throughout this run, which recorded 252 actual image
executions and zero OOM events under the 2 GiB cap. Byte agreement does not
turn the 40 unmet expectations into passes. The gate remains explicitly manual
in `SMOKE_WIRING.tsv`; no hosted CI result is claimed.

The original failed observations are retained. Component execution, mutation
checks, whole-compiler regressions and independent review must all complete
before production adoption or any compiler qualification claim. Candidate
branch checkpoints may compose these interfaces while preserving every open
obligation; they do not enable production admission or change a consumer pin.
