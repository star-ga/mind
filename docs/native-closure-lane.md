# Candidate native closure and namespace admission

Author: STARGA Inc.

The historical native profile still rejects unresolved runtime calls and every
collision with its standard-library prefix, except a user function that is an
exact private copy of a private seed helper (see
[exact private copies](#exact-private-copies-historical-profile)). A separate candidate lane profile
can distinguish supported intrinsic calls from missing user definitions and
handle private helper collisions without rewriting user code. These changes do
not qualify a compiler release or establish consumer backend parity.

The exact admitted types, intrinsic semantics and remaining restrictions are
defined in [the candidate lane profile](compiler/NATIVE_LANE_PROFILE.md).

## Selection and trust

The bridge selects candidate `native-lane-v1` only when the chosen native
compiler contains the distinct byte marker `48 b8 6e 62 6c 61 6e 65 30 31`
(`movabs rax` with immediate bytes `nblane01`). The existing `nbscope1` marker
and a bounded memory layout do not select this profile. Unmarked compilers
retain the historical frozen source/IR predicates and collision rules.

This is a version compatibility marker. It is not a signature, compiler hash,
independent validation or qualified release pin. A user-supplied compiler
executable is already a trusted build input. Qualification must still record
its exact source, executable and toolchain hashes and exercise the required
native/MLIR corpus under the prescribed resource limits.

## Mandatory checks

The candidate path validates the source before lowering, then validates the
source and its single IR lowering together under the lane's typed predicate.
Only exact intrinsic names and argument counts from that same profile become
terminal call bindings. A call to an unknown name is still an unresolved edge;
no placeholder function or runtime fallback is introduced. Source functions
and extern declarations cannot override the compiler's intrinsic namespace.

The closure check visits every retained user function, including uncalled
bodies, and all top-level calls. Entry and explicit exports remain mandatory
roots. Duplicate definitions and missing roots refuse. Closure validation
establishes binding, not arithmetic correctness or feature support; it cannot
replace the source/IR profile checks.

## Private standard-library helpers

When a private user helper collides with a positively parsed private seed
helper, the bridge may omit the complete unused standard-library prefix.
Public or explicitly exported collisions remain refusals. A seed that cannot
be parsed does not gain an exception from a textual guess about visibility.

Omission requires a primitive, explicit-call user image. Every source call
must bind to a retained user function or checked compiler intrinsic; the IR
closure is checked as well. Implicit library objects, methods and named types
are excluded. The source check matters because lowering could otherwise
replace library syntax with intrinsic IR while the native parser still sees
the original library-dependent syntax.

Under these conditions no retained call needs an omitted seed body, and no
seed body remains to resolve a private name differently. The complete user
region stays byte-for-byte unchanged, in its captured resolver order, with
every user function still a native root. No name is rewritten or user body
pruned. Noncolliding programs retain the exact historical prefix and image.

## Exact private copies (historical profile)

The frozen compiler keeps the last definition of each bare name, so a user
function named like a seed helper replaces that helper for every caller,
standard-library callers included. That is an ambiguity only when the two
definitions differ. On the historical path a colliding name is admitted when
the user image defines it once, as a private, unexported, top-level function
without attributes or type parameters; the complete seed parses and defines it
once in the same way, with no seed module making it public or exporting it;
and the two definitions are equal line for line after each line loses its `//`
comment and trailing spaces. A definition containing a string or character
literal or a block-comment marker is never compared, so comment removal cannot
misread a literal.

Every other collision keeps the `closure.std_symbol_collision` refusal and its
message. The image is not rewritten or pruned: both identical copies are sent
to the frozen compiler, and noncolliding images never parse the seed. Calls out
of an admitted copy must still resolve inside the user image; an open copy is
an unresolved edge, never a binding into a seed body. The candidate lane keeps
its own omission rule above. `tests/native_seed_copy_admission_controls.rs`
builds and executes an exact copy and refuses divergent, public and open copies.

## Regression evidence and limits

`tests/native_closure_admission.rs` retains the old profile controls and adds
candidate intrinsic, wrong-arity, forged-name, override and dead-body cases.
Unit tests in `src/build/native_bridge/namespace.rs` check public/exported
collisions, actual seed visibility, unresolved source calls and hidden library
dependencies. `tests/native_lane_bridge_controls.rs` captures the actual wire
image, executes simple scalar/memory programs, verifies empty output and exact
exit status, and compares unchanged noncolliding image/artifact bytes.

The CLI controls use a recording shim carrying a synthetic marker to select
the candidate contract while delegating to the historical seed for features
that seed already implements. They test selection, binding and image
composition, not the complete new feature implementation. Release validation
must run the real merged compiler with its marker emitted from committed
source. In particular, native raw syscalls are not evidence of libc-compatible
extern semantics; unsupported externs and unsupported typed constructs remain
blocked by the candidate profile.

Run the controls without a live API or consumer source dependency:

```sh
cargo test --offline --release --features mlir-build,std-surface,cross-module-imports \
  --test native_closure_admission --test native_lane_bridge_controls \
  --test native_module_bridge_controls -- --test-threads=1
cargo test --offline --release --features mlir-build,std-surface,cross-module-imports \
  --lib build::native_bridge::namespace::tests -- --test-threads=1
```
