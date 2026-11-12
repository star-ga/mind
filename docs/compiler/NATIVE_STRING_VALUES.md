# Native string value decoding

Author: STARGA Inc.

The native compiler previously refused every escaped string value even though
its lexer already understood the escape syntax. Literal emission now uses that
same seven-entry table for both decoded length and byte copying: `\n`, `\t`,
`\r`, `\0`, `\\`, `\"` and `\'`. Other UTF-8 bytes are copied unchanged.
Unknown escapes, unsupported hexadecimal/Unicode escapes and unterminated
strings remain refused. Embedded NUL is a byte, not an end marker.

The emitter validates the complete span before copying. It checks the existing
8 MiB literal window with subtraction (`length <= capacity - offset`) after
rejecting negative values and oversized offsets. Both passes use bounded
iteration with constant call depth. The decoder's readable, immutable source
span is supplied by the parser; these helpers are not arbitrary-pointer parsers.

An uppercase constant declared as `const NAME: String = "literal"` can use the
same literal path, with or without a trailing semicolon. Duplicate declarations,
composite initializers and other annotation shapes stay unsupported. Existing
local/parameter precedence remains conservative: a matching binding anywhere
in the source prevents global substitution. This includes ordinary/mutable
lets, tuple patterns, range-for variables and the accepted enum payload match
pattern. It applies to integer constants as well as String constants. A global
use outside a same-named local's scope may therefore refuse; it must never be
silently replaced with the wrong value. Lowercase lookup stays unsupported.

Independent review found six silent wrong results in the earlier candidate:
integer and String globals overrode for, tuple and match bindings. Those exact
inputs are retained in the gate, alongside no-global controls. Diagnostic scope
absence cannot replace this conservative guard: a newline-leading binary
operator intentionally makes that diagnostic withhold its binding conclusion.
Both newline and same-line forms are regression controls.

The match parser now checks the actual `=>` tokens for every admitted pattern
kind. Previously it skipped two tokens, accepting malformed arrows such as
`+ +`. The binder guard shares the checked-arrow predicate, so ordinary enum
constructor expressions do not spuriously suppress global lookup. Any future
extension to admitted binding patterns must extend the substitution guard.

This change preserves the existing native borrowed String representation
`{addr, len, cap = 0}` and read-only literal storage. It does **not** establish
parity for arbitrary String mutation, capacity inspection or ownership operations
against the MLIR backend's owned representation. Those are separate integration
obligations. The separately reviewed standard-library append fix now copies a
borrowed prefix into a fresh owned buffer before appending; its ownership,
capacity and allocation preconditions are documented in
[Borrowed String append checks](../../examples/mindc_mind/string_borrowed_check.md).
String decoding alone does not establish those ownership guarantees. Native source provenance
remains source provenance; accepting these
values does not claim that the current MIC trace path models every String value.
No frozen compiler seed, downstream pin or production qualification is changed.

## Reproduction

Use explicitly identified source-bound native and official CPU/MLIR compiler
executables. Set `MLIR_OPT`, `MLIR_TRANSLATE`, `CLANG`, and `MIND_STDLIB_PATH` for
the supported toolchain. From the repository root:

```sh
python3 examples/mindc_mind/self_host_native_string_escape_smoke.py \
  --enumerate --evidence /tmp/mind-string-values-run
python3 examples/mindc_mind/self_host_native_string_escape_smoke.py \
  --evidence /tmp/mind-string-values-run \
  --native "$NATIVE_COMPILER" --native-sha256 "$NATIVE_SHA256" \
  --mlir "$MINDC" --mlir-sha256 "$MINDC_SHA256"
```

Each evidence directory permits one execution after enumeration. The gate
hash-checks both instruments and records source inputs, exact commands, status,
stdout and stderr. Every child has a 2 GiB address-space cap and a CPU/wall timeout;
run the whole gate in a 2 GiB cgroup with swap disabled to cap aggregate memory.
The build-tool timeout is explicitly 300 seconds: the initial 64 KiB MLIR literal
diagnostic exceeded the compiler shell's separate default 60-second clang timer.
Increasing that time budget does not change the memory or correctness contract.

The fixed catalogue contains 55 source cases and 20 decoder/bounds cases. It
includes all escapes, UTF-8/NUL, constants, local shadowing, multiple literals,
long scanning boundaries, binding precedence, malformed arrows, exact decoded
bytes, malformed syntax, the exact 8 MiB
copy boundary, arithmetic overflow bounds and destination canaries. Unsupported
constant forms are native refusal controls, not claims that MLIR rejects them.
Crashes and failed builds never satisfy a positive or malformed-input control.

The original diagnostic's `escaped-final-quote` row mistakenly expected a
refusal for an escaped quote followed by a valid closing quote. Its input is
unchanged; the corrected expectation is normal exit 4 with empty streams. The
original catalogue hash was
`d321550cbbbc91ab3cfc9ee31093931679e7a8febdd35a2c1a4738ca9861fd6a`;
the correction record hash is
`81fd8c2ed2a4b74ae7ae4c2c99c8aa15fdf964615e5143fa77e5c224a86ec1e3`.
This is a documented test correction, not a change to string syntax.

Fresh whole-source MIC parity, bounded native self-reproduction, native/MLIR
self-output equality, source-note regression and independent source review are
required before integrating this compiler change. Runtime-only test results
cannot substitute for those gates.

## Observed validation (2026-10-15)

The final source and dependency-locked run accepted all 146 expected outcomes
from the frozen 55-source/20-helper catalogue. Whole-source MIC output matched
byte for byte (820,639 bytes); bounded native stages two and three matched;
native and fresh CPU/MLIR compiler self-output matched. All 28 scope cases and
46 source-note controls passed. The largest recorded cgroup peak was
1,693,077,504 bytes, with no OOM events or swap use under the 2 GiB limit.

| Artifact | SHA-256 |
| --- | --- |
| `main.mind` | `1f376b84a92d0fa524a869d381216915fc82afac1f3142216a7323fd22dcdcbe` |
| Frozen case catalogue | `1161ed302c525fc8a2a086a0e18b7e44dae1766642c5ce22902cc50067eb1354` |
| Bounded native diagnostic compiler | `b755ba17579dd67884180e51573e5b88b491b08fc3405d26c1d131949e6c832b` |
| Native/MLIR compiler self-output | `48762d25aaa18f8a7faeb077bb42e7f7d4a86b9a5e1f854604d3dc10757111b8` |

Separate source review and 25 additional intended controls found no remaining
defect within this contract. One reviewer fixture initially used unsupported
block-comment syntax and was refused by both compilers; the retained failed run
is distinct from its separately recorded line-comment correction. The reviewer
rebuilt the test programs using the supplied compiler artifact, rather than
independently rebuilding that compiler.

These results qualify this component for integration. They do not qualify a new
consumer compiler pin: merged-source gates, bootstrap-transition checks and
consumer runtime parity still apply.
