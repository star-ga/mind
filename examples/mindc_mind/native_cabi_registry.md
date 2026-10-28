# Typed native C ABI registry (candidate metadata slice)

Author: STARGA Inc.

This MIND implementation preserves original source declarations for a future
native C ABI boundary. It does **not** enable external calls, dynamic linking,
libc startup, or a new native profile. Existing extern placeholders, historical
MIC bytes and native rejection behavior are unchanged.

`nb_cabi_build(source,length)` reparses original top-level extern AST spans.
A token census refuses hidden/nested extern blocks. No type is reconstructed
from the historical MIC integer categories. Callers retain source storage
unchanged until releasing the registry with `nb_cabi_free`. As with the existing
compiler lexer API, this low-level entry requires a readable source buffer with
at least 7 padding bytes beyond `length`; both diagnostic callers provide 8 zero
bytes. Buffer validity and output-record allocation remain caller obligations.
`nb_cabi_encode` returns the deterministic descriptor; on any error it returns
no bytes. `selftest_nb_cabi(source,length,out)` exposes this to diagnostic gates.
Its four output words are descriptor address, descriptor length, error code,
and source byte offset. Descriptor bytes belong to the caller. This diagnostic
entry is not a verifier of arbitrary source program semantics.

## Descriptor bytes

All counts and lengths are unsigned 32-bit little-endian values. The format is:

```text
"MCABI001" (8 ASCII bytes)
declaration_count: u32
repeat declaration_count times, in source order:
    name_length: u32
    name: ASCII bytes (no NUL)
    callconv: u8 (0=C, 1=SysV, 2=Win64, 3=Aapcs)
    unsafe: u8 (0=safe, 1=unsafe/default)
    varargs: u8 (0 or 1)
    fixed_parameter_count: u32
    return_type: prefix type
    parameter_types: fixed_parameter_count prefix types, in source order
```

Prefix type tags: 0=Void, 1=i8, 2=u8, 3=i16, 4=u16, 5=i32, 6=u32,
7=i64, 8=u64, 9=isize, 10=usize, 11=f32, 12=f64, 13=bool.
Tag 32 is `*const` and tag 33 is `*mut`; either is followed by its pointee type.
Void represents only an absent return annotation. Explicit `void`, pointer to
void, aliases, aggregates, references and function-pointer types are rejected.
C and explicit SysV remain distinct metadata. Recording Win64/Aapcs/varargs is
not a claim that the Linux native emitter implements them. A future callable
contract must validate target support independently.

Bounds are explicit metadata admission bounds: 4096 declarations, 256 fixed
parameters per declaration, 32 pointer layers and 255 ASCII foreign-symbol bytes. Parameter identifier lengths follow the source
parser; they are not ABI symbol names. Duplicate parameter names are rejected.
Exceeding a bound is a diagnosed refusal; there is no partial successful table.
The diagnostic stdin wrapper separately accepts at most 4 MiB of source.
Identical and conflicting duplicate external names both refuse. The reserved
`__mind_` prefix and collision with a locally defined function also refuse.
Local const/type names and syntactic import aliases/path tails are checked for
collisions and primitive-type shadowing. Complete resolved-import namespace
validation needs the separately reviewed
source-AST contract; this stage never widens production admission.

The internal owned type record is 48 bytes (length word plus at most 33 byte codes).
Rows own parameter/type storage and borrow source name spans. Each parameter
allocation is 6144 bytes: 256 type pointers then 256 pairs of original name
spans, used to reject duplicate parameter names. Encoding copies
name/type bytes into a separate String. Freeing the registry does not free the
source or the returned descriptor. Existing parser/token allocations use the
compiler's established lifetime; no universal malloc-exhaustion claim is made.

Error codes: 1 invalid input; 2 malformed declaration; 3 unsupported type;
4 unknown ABI/calling convention; 5 duplicate external/parameter name; 6 reserved name;
7 unresolved/nested extern scope; 8 allocation failure; 9 namespace collision;
10 bounded-resource refusal. Offset is the first detected source byte location.

## Tests and status

`native_cabi_registry_smoke.py --enumerate --evidence DIR` writes the complete
case packet before execution. `--native DRIVER --oracle SO --evidence DIR`
checks expected descriptor bytes and refusal categories, and requires native
versus fresh malloc-backed MIND oracle exit/stdout/stderr equality. Every oracle
request runs in a fresh subprocess. The runner is diagnostic host orchestration;
the parser/registry/encoder and executable input wrapper are MIND.

The normal native compiler's output, historical extern MIC regressions, bounded
compiler fixed point, and whole-source `mic3_flip_smoke.py` are separate required
checks. Passing this registry gate does not qualify any of
the ten downstream-consumer applications or establish a compiler pin. Actual import
emission, C startup, width conversions and libc-compatible intrinsic errno
behavior remain separate work.

Independent source review found two additional candidate defects after the first
passing 64-case run: duplicate parameter names were accepted and parameter names
were unnecessarily limited to 255 bytes. Both were corrected; the final case
packet retains all previous cases and adds these controls. The first block-
comment positive expectation was corrected only after the official parser
rejected that identical source with E1001; the original input remains a required
negative, and a separate line-comment positive is retained. No block-comment
language support is claimed by this metadata slice.

The full-source MIC3 gate caught a further integration regression in the first
registry implementation: nine helper functions used early returns or nested
side effects inside loops that the native HOME emitter handled but the MIC3
emitter refused. The registry now uses bounded iterative loops and checked
per-step helpers supported by both paths. No recursion proportional to the
number of declarations was introduced, and the descriptor format, error codes,
first-error offsets, bounds and original case packet remain unchanged. The
original failed gate and pre-refactor execution evidence are retained privately.

Observed candidate validation on 2026-10-15: 69 paired registry cases, the
4096/4097-declaration boundary, and all ten frozen source metadata inventories
passed. Five old MIC fixtures and three static executables retained exact bytes.
The bounded compiler reached an exact fixed point; whole-source MIC3 output
matched the official compiler at 815746 bytes. Peak aggregate memory was
785833984 bytes under a 2 GiB limit, with no OOM events or swap. Separate
implementation review of the registry and subsequent helper refactor returned
PASS_WITH_NOTES; complete resolver checking and runtime ABI remain obligations.
