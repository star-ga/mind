# Source-derived native C ABI declaration contract

Author: STARGA Inc.

`src/ir/native_c_abi.rs` records and checks C function declaration metadata from
the original parser AST. It does **not** enable native C calls, change the native
feature profile, emit a call, resolve a library, qualify a compiler or establish
pointer safety. C extern declarations remain refused by native lane admission.

## Source authority and scope

Call `NativeCAbi::from_ast` on the same captured AST that will be lowered and
compiled. Supply the complete imported and reserved symbol sets for that source
capture. Imports without resolver information refuse. Imported aliases and
resolved bindings cannot collide with a foreign declaration. The `__mind_`
namespace is always reserved; additional compiler/std names come from the
selected capability's resolver. The completeness and source binding of those
caller-supplied sets remain an explicit integration obligation.

Legacy `Instr::ExternFnDecl` is insufficient as the source of truth: for example,
the existing SysV parameter lowering can map i32, u32, i64 and u64 to the same
`i64` tag, and pointer pointees/mutability are erased into an opaque pointer.
This module retains those distinctions before lowering. Tests derive both forms
from the same real parser AST and exhibit that loss.

Descriptors preserve i8/u8/i16/u16/i32/u32/i64/u64/isize/usize/f32/f64/bool,
recursive raw-pointer pointees and const/mut qualifiers, the declared calling
convention, fixed parameters, variadic flag and the informational safe/unsafe
annotation. Absent return means Void; explicit named `void` and pointer-to-void
are outside this initial contract. Target-sized integers and bool are distinct
metadata, not guesses about platform representation.

Duplicates, local/import/intrinsic conflicts, repeated parameter names and
non-ASCII C symbol spellings refuse. Aliases, aggregates, references and callback
function pointers refuse, including behind raw pointers. Nested foreign blocks
and aliases shadowing primitive spellings refuse regardless of declaration order;
resolved imported scalar-name conflicts are conservatively ambiguous. Foreign
data declarations refuse. No thread-local storage, callback lifetime,
re-entry or thread-safety contract exists here. A function name or an integer
handle cannot certify those properties; library-specific behavior still needs
an explicit interface contract. A `safe fn` annotation remains informational.

The limits are 4,096 declarations, 256 fixed parameters per declaration,
32 nested pointer levels and 255 bytes per symbol. Exceeding a bound returns a
normal rejection without a partial accepted inventory. These are metadata
resource bounds, not claims that the backend can call a function with 256
arguments or support 4,096 imports. Parser resource management precedes this
AST-level component.

## MCABI001 binary identity

This is a versioned declaration descriptor, separate from historical MIC extern
encoding. It is not a new execution channel and does not alter MIC or MAP.
There is no JSON runtime channel or JSON hash preimage.

The canonical bytes are:

1. Eight ASCII bytes `MCABI001` and a u32 little-endian declaration count.
2. Each declaration in captured source order: u32 little-endian name length,
   ASCII symbol bytes, one calling-convention byte, one unsafe byte, one varargs
   byte, u32 little-endian fixed-parameter count, result type, then parameter
   types in order.
3. Calling conventions: 0=C, 1=SysV, 2=Win64, 3=Aapcs. Boolean flags are 0/1.
   The omitted safety keyword follows the parser's unsafe default.
4. Type tags: 0=Void, 1=i8, 2=u8, 3=i16, 4=u16, 5=i32, 6=u32, 7=i64, 8=u64,
   9=isize, 10=usize, 11=f32, 12=f64, 13=bool. Pointer tags 32=const and 33=mut
   are immediately followed by the complete pointee descriptor.

Only absent returns may encode Void. Fields with lengths are bounded before
encoding, so u32 conversion cannot truncate a validated inventory. Parameter
names and source spans do not affect ABI identity; symbol names, declaration
order and every signature/safety field do. The descriptor hash is SHA-256 through
the existing compiler hash implementation. It is **not a source hash**: source
bytes, compiler, runtime, target and dependency identities must be bound
separately by the eventual integration.

`check_native_descriptor` compares independently generated native bytes with
the exact AST-derived canonical bytes. Unknown versions/tags, reordered or
missing declarations, altered fields, truncation, trailing bytes and a supplied
`PASS` flag cannot substitute for equality. This establishes metadata agreement
only. It does not prove that the native emitter used the descriptor correctly.

The golden test for `safe fn probe(a:u32,p:*const u8)->i32` under explicit SysV
uses hand-specified bytes from this format. Other controls mutate every byte,
declaration order, lengths and signature fields.

## Target and calling obligations

`require_fixed_sysv_amd64` is a prerequisite filter for a future fixed-signature
Linux/x86-64 SysV slice: C and explicit SysV are retained as distinct metadata;
Win64, Aapcs and variadic declarations refuse this filter. Passing it does not
classify arguments or implement calls. Cross-target convention handling must
never silently reinterpret Win64/Aapcs as the default.

Before enabling native imports, implement and verify scalar extension and
truncation, pointer representation, integer/SSE register classification, stack
arguments and alignment, return classification, target-sized integers and bool,
plus ABI-compatible process startup and runtime interactions. Calls must be
type-checked against the original signatures, not only counted or matched by
symbol name. Pointer provenance, mutability, lifetime, alignment and access
bounds need their own checks and contracts.

Variadic support additionally requires source-bound call-site argument types,
C default promotions and platform rules such as SysV AMD64's vector-register
count in AL. This descriptor deliberately supplies no variadic argument hints
or default i64 fallback. A fixed declaration of a symbol whose actual library
interface is variadic does not prove that the supplied declaration is correct;
the explicit library ABI contract must cover that discrepancy. No libc behavior
is inferred from names such as `open` or `fcntl`.

Dynamic symbol/version/SONAME/interpreter selection is a separate linker policy.
ELF metadata or a successfully loaded function does not establish a compatible
signature. Imported functions, exported shared-library invokers and raw Linux
syscalls remain separate capabilities. In particular raw negative errno is not
libc thread-local errno.

## Checks

```sh
cargo test --offline --lib --no-default-features \
  --features std-surface,cross-module-imports ir::native_c_abi -- --test-threads=1
```

Tests parse real synthetic source before checking descriptor refusals. They
cover source width loss, pointer structure, calling conventions/varargs,
namespace conflicts, unsupported aggregates/callbacks/data imports, stable
identity, exact native-byte comparison and adjacent resource bounds. A regression
asserts that descriptor acceptance still leaves native C admission refused.
These tests validate contract infrastructure, not a native C runtime.
