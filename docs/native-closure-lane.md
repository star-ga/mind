# Candidate native closure and namespace admission

Author: STARGA Inc.

The historical native profile rejects unresolved runtime calls and every
collision with its standard-library prefix, except a user function that is an
exact private copy of a private seed helper (see
[exact private copies](#exact-private-copies-historical-profile)). `mindc build`
does not apply that profile at this head: it refuses a compiler without the
`nblane02` mark before any profile (see Selection and trust). A separate candidate lane profile
can distinguish supported intrinsic calls from missing user definitions and
handle private helper collisions without rewriting user code. These changes do
not qualify a compiler release or establish consumer backend parity.

The exact admitted types, intrinsic semantics and remaining restrictions are
defined in [the candidate lane profile](compiler/NATIVE_LANE_PROFILE.md).

## Selection and trust

The bridge runs a native compiler only when that compiler contains the distinct
byte marker `48 b8 6e 62 6c 61 6e 65 30 32` (`movabs rax` with immediate bytes
`nblane02`). The existing `nbscope1` marker, the earlier `nblane01` marker of
the 0.10.3 seeds and a bounded memory layout do not carry this mark. A compiler
with another mark, or with none, is refused with one message that names
`nblane02`. The host does not apply the historical frozen source/IR predicates
to that compiler, and it does not build the program.

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

## Calls into the standard-library seed

A call resolves through one chain: user function, admitted `extern "C"`
declaration (typed by its ABI classes under a compiler carrying `nbextc01`; the
compiler decides which symbols exist), std seed function, compiler intrinsic. A name
that more than one tier defines refuses `lane.call.ambiguous`; the order never
picks a winner, because the native compiler binds a bare name to the last
definition in the image, so a seed body's call would silently bind to a user
function of the same name.

A user call binds to the seed tier only when it names a public (or exported)
function of a seed module the program imports (`import std.<m>;`), the same
modules the MLIR build links. Every seed function that body calls is reached
too, whatever its visibility. Each reached function is typed and walked under
the rules for user functions, source first and then its own lowering, and its
arity becomes a terminal closure binding. Unreached seed bodies are neither
admitted nor walked, and a program that calls no seed function never parses
the seed. A reached body outside the lane refuses at its own site:
`(lane.type.named) in std/cli.mind fn `args_get` at line 63, column 5 of
std/cli.mind`. A seed body sees its parameters only: a seed module constant is
not bound (`lane.source.unbound`). A private seed helper is not callable from
user code natively (`lane.call.unknown`), although MLIR links it.

Every `import std.<m>` naming a module outside the manifest's `seed` rows
refuses `lane.import.not_seed` at its line, whether or not anything calls it:
the native image carries no such module. Register the module as a `seed` row
(a reseed event) or build with `--backend mlir`.

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

Omission is consulted first. An image that also carries an identical private
copy of a seed helper (a vendored helper) next to a divergent private
collision is admitted when it is closed: the seed is omitted and the vendored
copies are the only definitions. The identical-copy guard refuses only an
image that cannot omit (a seed call, a string literal, a struct, a method, a
non-primitive type, or a public or exported collision), with the unchanged
`closure.std_symbol_collision` message. `naes_shape_closed_vendored` and
`naes_shape_open_divergent` in `tests/native_lane_std_seed_calls.rs` pin both
sides.

In a closed image the visibility of a colliding seed symbol does not matter:
omission drops the whole seed, so no seed body, public or private, is left to
bind the name. A private user function named like a public standard-library
function (`decode`, defined public in `std/json.mind`) therefore omits the
seed and builds, as a private collision does. An open image keeps the seed and
refuses. User names that are public or exported still refuse in every image:
an exported name is an ABI, not a private helper. Each refusal names the
colliding symbol and the first seed module that defines it, for example
``"`decode` collides with public std symbol `decode` defined in std/json.mind
(closure.std_symbol_collision)"``. A name the seed reserves only textually,
with no parsed definition, still refuses. `naes_shape_public_name_closed` and
`naes_shape_public_name_open` pin both sides.

## Allocation in loops

### Allocator classes of the source tree

The pure-MIND compiler of this source tree gives each native image one of
three allocators. The host check of `mindc build` no longer has
`lane.alloc.free_in_loop`. A compiler minted from the tree, carrying `nblane02`,
builds a free reached from a loop and the image reclaims: three `__mind_free`
calls in a `while` exit 7, and the corpus program `r3_free_loop` exits 33. The
committed seeds carry `nblane02`, so a build that uses them is admitted. The
host refuses a compiler that lacks `nblane02` before any profile
(`native.compiler_mark`) and writes no image. The published 0.10.3 compiler does
not have this allocator.

- BUMP: the inline bump core. `__mind_free` releases nothing. A full arena kills
  the process by signal. An allocation that does not fit runs a trap instruction
  (SIGILL). An exact fill of the 1 GiB arena followed by one more allocation dies
  on the guard gap (SIGSEGV). Every default-profile image of a program that has
  no free at a loop position is a BUMP image, byte for byte the same as before.
- RECLAIM: the free-list runtime that explicit-layout images carry (a shared
  blob at the entry of the image), on the default 1 GiB arena. Alloc, free and
  realloc call into it; a freed block is reused and neighbours merge. The arena
  is the only region: a full arena ends the program with
  `error: native arena exhausted` on stderr and exit status 70, a stop that
  depends only on the program. A dynamic image (a program with `extern "C"`
  calls) takes the same blob at its entry.
- Bounded and emission-context images keep their own allocators, unchanged. An
  explicit layout always uses the free-list runtime; the choice below does not
  apply.

`nb_alloc_select.mind` chooses the class of a default-layout image from the
composed source alone (no environment variable, no mode word). RECLAIM is
chosen when a `__mind_free` call stands at a loop position: the condition or the
body of a `while`, or the body of a `for`. The call may be direct or reached
through calls, over every function of the image (the user functions and the
standard-library functions they reach). A `for` bound that is evaluated before
the loop is not a loop position. Recursion without a loop is not counted.

A line of the leading `//` block of the user region (its first lines that begin
with `//`) that reads exactly `// mind: native_alloc = bump` keeps BUMP. The
compiler's own source carries it, so the compiler stays on the bump allocator.
`// mind: native_opt = full` stays a directive of line 1; both lines can be
present (opt-level on line 1, allocator on line 2).

Contract of a RECLAIM image: pass only 0 or a live block to `__mind_free` and
`__mind_realloc`; never use a block after it is freed. A double free is
undefined. Blocks are
8-byte aligned and read zero. `__mind_alloc(n)` with `n <= 0` returns 0.

### Allocation through a compiler minted from this tree

`mindc build` with a compiler minted from this tree admits a free reached from
a loop and the image reclaims. One shape that allocates its scratch once,
outside the loop:

```mind
fn clock_ns(ts: i64) -> i64 {
    let rc: i64 = __mind_sys_clock_gettime(1, ts);
    if rc < 0 {
        return 0;
    }
    __mind_load_i64(ts) * 1000000000 + __mind_load_i64(ts + 8)
}
```

with `let ts: i64 = __mind_alloc(16);` in `main` before the loop.
`std.sha256`'s `sha256` frees its scratch in `sha256_checked`. A call reached
from a loop builds and runs (`std_sha256_in_loop`: exit 169, and the 32 bytes
are the SHA-256 digest of the empty message). The same program through
`--backend mlir` exits 169 and writes the same bytes.

## Exact private copies (historical profile)

Status at this head: `mindc build --backend native` with a compiler that lacks
`nblane02` (the committed seed before the advance) refuses a program that holds
an exact copy of `sha_mask32` with `native.compiler_mark`, exit 3, before the
rule below runs. The candidate minted from this tree builds the same program
(exit 0). The rule below describes the historical profile only.

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
source. `nb_cap_native_lane_v1` in `examples/mindc_mind/main.mind` emits it, so
a compiler built from that source carries the marker. The embedded default
seed was advanced from such a source by the two-hop pure-MIND advance and the
bounded seed re-blessed from it, so both carry the marker and a plain
`mindc build --backend native` selects the lane; `seed.rs`'s
`the_embedded_seeds_carry_the_native_lane_v1_and_scoped_binding_marks` and
`tests/native_embedded_seed.rs` (`*_selects_native_lane_v1`) pin that. In particular, native raw syscalls are not evidence of libc-compatible
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
