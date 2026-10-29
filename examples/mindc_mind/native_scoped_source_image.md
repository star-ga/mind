# Scoped fixed C calls: NSF/1 candidate

Author: STARGA Inc.

This is an implemented, **not yet executed** source-to-image candidate. It
connects original MIND declarations to the existing checked libc binding and
fixed SysV emitter. It does not enable the official native frontend, qualify
a compiler pin, replace a seed or complete a downstream consumer gate.

## Source ownership

`native_fixed_scope_manifest.mind` is compiled toolchain data: the exact21
canonical standard-library files captured at `17ea2dff`, in the existing
`_selfhost_sources.NATIVE_STD_MODULES` order, each followed by one LF. The
prefix is395479 bytes with SHA-256
`199f304b1209959946d67bde175407719f9641bb423aa6c95fa8245d259cb202`.
Each region's offset, length and SHA-256 are checked against the actual owned
source copy. The historical seed manifest is unchanged and does not validate
this newer std capture.

The suffix is exactly one translation unit. The adapter accepts no module
labels, region list, library list, caller digest or ownership flag. Byte input
cannot prove whether somebody previously concatenated several files; this
interface gives the suffix one-unit semantics. Future official integration
must independently require `SingleFile` or an actually resolved one-source
project before flattening. General project source sets remain unsupported.

Each of the22 original source units gets its own complete `nb_cabi_build`
registry. A repeated name inside one unit refuses. Declarations in another
unit cannot authorize a call. The original source, pointer qualifiers,
parameter types and complete registry remain available through finalization.
No one-row declaration is synthesized and no declaration is selected by
first/last occurrence across units.

Every ordinary emitted local call is also checked. A same-unit body takes
priority; otherwise the target must be a unique public body from an explicit
`import std.NAME;` or `use std.NAME;`. The actual retained AST must be that
target. Private helpers, absent imports, aliases, qualified import forms and
unresolved project imports refuse. The legacy global body/return metadata
cannot safely represent duplicate body names, so this profile refuses them
when called; an extern call also refuses a same-named body elsewhere. A
primitive type-name redefinition refuses rather than contaminating global
type metadata. These are deliberate bounds of this profile, not a claim of
general module resolution. All preexisting source guards remain active.

## Admitted call tranche

The first tranche accepts zero or one exact `i32` argument, and an `i32`
result or the binding table's exact borrowed `*mut i32` TLS errno result.
Arguments must come from an explicit signed32 cast, a live explicitly typed
`i32` binding/parameter, or a checked source-resolved `i32` result. Arguments
are evaluated once by the existing lowering path. Bare untyped literals,
unsigned/bool/float values, ambiguous inferred bindings and pointer arguments
are not contextually reinterpreted as `i32`.

`ncb_build` and `ncb_check` reparse the whole original unit. Only fixed,
returning libc contracts with zero compatibility conversions are admitted.
The existing binding table selects the actual symbol, version, soname and
interpreter. A valid pointer result retains pointer origin until an explicit
address cast. This does not prove user memory accesses safe; live-pointer and
allocation obligations of the existing MIND intrinsics still apply. Invalid
file descriptors are legal inputs to libc `close`, whose normal error result
and TLS errno must be observed, not predicted by the compiler.

The scope permits at most256 emitted foreign callsites and16 distinct
unit/declaration bindings. Import and fixup storage respects the dynamic
writer's existing256-import bound. Repeated source bindings may share a
symbol/version loader import only after their separate checks have passed.
The complete source remains bounded by4MiB.

## Frame and image construction

Schema3 is internal432-byte compiler context storage. Its first416 bytes
retain the checked target geometry; the extension owns the source-unit and
call-state data. Public raw schema1/schema2 constructors and the v1 stdin
adapter are unchanged. The live-buffer API is not a serialized proof format.

Lowering reserves64 bytes of `INT3` for each foreign call. It records the
original unit/row, argument/result frame displacements and actual final
function frame. SSA-to-HOME fallback discards that function's speculative
records. HOME records its frame only after statement lowering; SSA records
its computed real frame. DTK register caching is excluded for this path.
Rebasing preserves each function's ordinary branch and local-call offsets.

Finalization checks sorted, disjoint, intact64-byte reservations, source-copy
equality and the original declaration. It invokes unchanged `nce_emit` with
the actual frame and source row. Only a valid bounded result with one real
`FF 15` fixup can replace a reservation; the remaining bytes become NOPs.
Local E8 records remain separate. Every reservation must be finalized exactly
once, with exact matching counts, before either ELF image pass can publish.
Unknown bindings, missing/stale records, invalid frames and overlong emitter
results refuse. No unresolved trap/NOP-only placeholder is publishable.

The old startup and optional NIO imports remain first in the construction
vectors; the existing writer performs canonical sorting/remapping. New
binding-plan storage remains live through both planning and final emission.
No-call output parity is an explicit test requirement, not an assumed result.

## Process interface

`native_scoped_source_driver.mind` uses the existing bounded stdin framing
with distinct eight-byte magic `MINDNSI2`. Header size is32, flags0, payload
is `user_lo:i64, source_length:i64, source_bytes`, followed by EOF. User offset
must match the committed prefix exactly. All other bounds, fixed policy,
route0, computed source-hash note, short-I/O loops, SIGPIPE handling and exits
64/65/70/74 follow [the v1 adapter contract](native_source_driver.md).
Source ownership/binding refusals use exit65, empty stdout and exactly
`source/image refused\n` on usable stderr. A separate frozen test driver may
select HOME; the product frame does not expose a route control.

The lower `nsf_compile` live-buffer API recomputes the supplied full-source
digest and constructs scope itself. Its additional diagnostic families are
10 for source scope/capture and11 for foreign binding/finalization. These
diagnostics are not caller authorizations or accepted proof flags.

## Evidence contract

The original30 independent source cases and five prefix/ownership
perturbations are retained in
[the frozen catalogue](testdata/native_scoped_fixed/README.md). Additional
cases cover bounded staging, actual frame/rollback helpers and geometry
temporary ownership. The geometry observer uses the exact extracted
production function with explicit allocation/ownership observers; it is not
a whole-allocator or sanitizer proof. An actual wrong-ownership mutant must
fail a normal assertion, never merely crash.

Both compiler paths must rebuild from one captured closure and agree on
complete output bytes for identical source packets. Successful generated
images must run with exact expected exit/stdout/stderr and owned descriptor
effects. No-C cases must match their old v1 source images. Repeated requests,
source-level errno, real mutations and normal refusal checks are separate
obligations. The CPU instrument/native seed identities are diagnostic tools,
not downstream compiler pins. All results remain pending until their actual
execution receipts exist.
