# Scalar variadic call emission

Author: STARGA Inc.

Status: COMPONENT_TESTED, INDEPENDENT_REVIEW_PASS_WITH_NOTES. This additive component is not connected
to source admission, import binding or compiler dispatch. No consumer compiler
pin or downstream-consumer acceptance class is qualified by its presence.

`native_c_variadic_emit.mind` implements Linux x86-64 LP64 SysV calls for the
bounded scalar/pointer types accepted by the [variadic argument planner](NATIVE_C_VARIADIC_PLAN.md).
The existing fixed planner, fixed emitter, variadic planner and compiler entry
point remain unchanged.

## API and preconditions

`ncve_validate(decl, actual_types, total, slots, frame, dst, import_id)` first
returns the unchanged planner's ordered validation error. It then validates a
16-byte-aligned frame in [16, 2147483632], import identifier in [0,255], a nonnull
slot table and every signed 8-byte-aligned displacement in [-frame,-8]. A nonvoid
result requires a valid destination displacement; a void result ignores it.
Later errors are 9 for frame, 10 for import, 11 for input slots and 12 for result
slot. These overlap earlier planner errors; precedence is part of the contract.

`ncve_emit` has the same arguments followed by five writable metadata words.
Inputs are live, immutable, correctly aligned internal buffers with the planner's
specified extent. Metadata is writable and disjoint from every borrowed input.
These APIs do not validate arbitrary machine pointers or concurrent mutation.

The emitter clears metadata, revalidates original typed inputs and constructs a
fresh owned plan. A refusal or planner allocation returning zero yields an empty
string and five zero words. The latter is not a promise to survive a runtime
allocator that aborts on actual memory exhaustion. A successful result contains
unbound machine-code bytes and metadata `[2, fixup, import_id, reserve, AL]`.
`fixup` indexes the first byte of the unresolved four-byte displacement in
`FF 15 disp32`. The caller must bind it through its own admitted GOT policy.
The temporary plan is freed. The returned string follows MIND string ownership.

At the call site RBP describes the declared frame and RSP is its 16-byte-aligned
bottom. Each input slot has eight readable bytes with the meaningful value in
its low declared width. The result slot may alias an input after all arguments
have been loaded. The callee follows SysV, preserves its callee-saved registers
and the caller's FP control state, and may destroy all caller-clobbered registers.
Signal delivery, stack availability, actual pointer pointee validity and the
external function's semantics remain caller/runtime obligations.

## Emission semantics

Fixed parameters retain their declared loads. Tail i8/u8/i16/u16 extend to i32;
tail Boolean reads one byte and canonicalizes it to zero or one. Tail f32 uses
`CVTSS2SD`, including when spilled; XMM15 is the spill scratch register so that
live XMM0 through XMM7 remain intact. Pointer-to-f32 is an integer-class pointer,
not a floating value. Remaining scalar loads and result normalization reuse the
unchanged fixed emitter helpers.

GP and SSE pools are independent. Ordered eight-byte spills use the planner's
16-byte-aligned reservation. `MOV EAX, AL_count` is emitted after argument
preparation and immediately before the call, so RAX scratch work cannot corrupt
the count. Stack restoration precedes declared-width result storage. The emitter
adds no library call that could modify `errno`.

`CVTSS2SD` observes MXCSR. This component does not change global FP state.
Precise subnormal tests below require DAZ/FTZ disabled and masked exceptions,
with MXCSR control value `0x1f80`. A production strict-FP integration must establish
its own environment contract. No result here claims determinism for arbitrary
FP environments, signaling NaNs, or arbitrary NaN payloads.
See Intel's [instruction reference](https://cdrdv2-public.intel.com/868137/325462-089-sdm-vol-1-2abcd-3abcd-4.pdf)
and [FTZ/DAZ documentation](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/set-the-ftz-and-daz-flags.html).

## Frozen executable contract

The `testdata/native_c_variadic_emit/` catalogue contains 55 controls: 38 real
calls and 17 normal refusal/allocation checks. Eight actual source changes test
signed extension, Boolean normalization, f32 conversion, live-XMM preservation,
AL, spill offsets, stack restoration and narrow result normalization.
`CASES.json`, `CASES.tsv`, `MUTATIONS.json` and the byte fixtures define the exact
expectations before execution. A crash, build failure, timeout or setup failure
never counts as mutation detection.

The MIND constructor checks validation/error priority, zero-refusal metadata,
planned metadata and the unresolved call location. It emits a dedicated wrapper
with a 4096-byte canary frame, poisoned input high bits and preinitialized stack
padding. The wrapper records the actual post-call RSP before `LEAVE`; a faulty
restore therefore produces a normal observable mismatch. Result stores are
checked in a complete frame snapshot, excluding only its dynamic context pointer.

The explicit C/assembly fixture is an independent observation instrument. It
captures GP/SSE arguments, AL, entry alignment and stack slots before clobbering
arguments, restores them and tail-jumps into a C function with the exact fixed
signature and ellipsis. The C callee records actual `va_arg` values without any
expected-value comparison. Five assembly result controls deliberately poison
excess return bits and caller-clobbered registers. Only the designated fixture
GOT slot is bound. Code transitions from RW to RX; no RWX mapping is used.

During each call the fixture blocks catchable asynchronous signals, saves MXCSR,
installs `0x1f80`, records before/after values, and restores the prior state and
signal mask. The checker requires unchanged control bits; status flags are
observations and are masked from the deterministic receipt. It does not claim
canary preservation in the callee's legal red zone or stack scratch area.

The separate MIND verifier links no planner or emitter. It compares the raw
capture to frozen expected words and masks, then emits the masked canonical
receipt. Absolute addresses and unused registers are excluded; used argument
values, stack slots/padding, frame canaries, result, AL, relative RSP, alignment,
`errno` and FP controls are checked. Both native and MLIR verifiers check each
builder's calls. Unbound code/metadata and canonical receipts must agree across
builders. Runtime addresses and raw OS observations need not be identical.

## Diagnostic wire layouts

All integer fields use little-endian 64-bit two's-complement words. Constructor
input has 24 words: version, case number, calling convention, safety, variadic
flag, declared fixed count, declared total, fixed-record count, actual-record
count, frame, destination, import, null-slots flag, expected error, expected
reserve, expected AL, allocator-fault expectation, reserved, fixture mode and
five reserved words. It is followed by the result's 48-byte type record, fixed
records, actual records, actual-count slot displacements and source-value words.
Record counts are independently bounded to 256 before reading their bytes.
Before either product call, the valid-count indexing path additionally requires
declared fixed count <= captured fixed count and declared total <= captured
actual count. An incomplete extent returns setup status 72. Deliberately invalid
declared counts still exercise the unchanged product count diagnostics.

Constructor fd3 output has 12 words: version 2, case number, error, wrapper size,
call start, absolute fixup, GOT offset, frame, import, reserve, AL and fixture
mode; followed by the unbound wrapper. Refusals carry only the header, with the
first three words retained and all remaining words zero.

The capture has 1046 words: version, case number, AL, entry RSP modulo 16,
post-call RSP minus RBP, errno, MXCSR before, MXCSR after, six GP registers,
eight low XMM words, 256 stack words, 256 typed-value words and 512 frame words
in ascending address order. The verifier reads 1046 expected words, 1046 bitmask
words, then this capture, and requires end-of-input. A normal pass is exit 0,
stdout `1\n` and empty stderr; a normal assertion failure is exit 1 and `0\n`.
Setup and transport failures return other statuses and do not establish fault
detection. These are diagnostic fixtures, not a general untrusted-code sandbox.

## Reproduction and custody

Capture once before execution, record/review the printed manifest digest, then
execute with explicit instrument identities:

```sh
python3 examples/mindc_mind/native_c_variadic_emit_check.py --enumerate --evidence /owned/capture
sha256sum /owned/capture/PREEXECUTION.sha256
python3 examples/mindc_mind/native_c_variadic_emit_check.py --evidence /owned/capture \
  --seal-sha256 REVIEWED_DIGEST --seed /tool/native-compiler --seed-sha256 NATIVE_DIGEST \
  --mindc /tool/mindc --mindc-sha256 CPU_DIGEST
```

Set `MLIR_OPT`, `MLIR_TRANSLATE` and `CLANG` to the reviewed LLVM tools. Run the
transport under a dedicated owned cgroup with aggregate memory 2 GiB, swap zero,
one CPU and nice 19; it also imposes a 2 GiB address-space limit. Builds receive
180 CPU seconds / 240 wall seconds; diagnostic children receive 10 CPU seconds /
12 wall seconds. Preserve the entire captured closure and raw failed results.

The trusted wrapper verifies the owner-reviewed seal and every captured file,
then invokes the captured runner. Both the wrapper and the capture directory
must be owner-controlled with no concurrent mutation. Enumeration copies the
unchanged fixed-call diagnostic sources to preserve their original 58-case
contract; this new suite does not replace or waive those original outcomes.
Existing fixed-call execution remains a separate integration gate.

Execution and independent review results will be appended only after observation.


## Observed execution and retained correction

The final guarded constructor passed all 55 original controls and all eight
actual source mutations on both builder backends. This used 22 fresh MIND builds,
126 constructor executions, 92 real C/assembly calls and 184 MIND verifier
executions. Every unbound wrapper/metadata artifact and checked canonical receipt
matched between builder backends. All eight mutants produced normal verifier
exit 1, stdout `0\n` and empty stderr. Their capture processes exited normally;
the XMM clobber and wrong-stack-restoration cases were not counted from traps.
Peak memory was 91,508,736 bytes, all memory events were zero, and swap was zero.

A separate same-machine repeat rebuilt both constructors and both verifiers in
fresh directories. Each executable matched its own original backend bytes. The
six unchanged selected controls produced 12 repeated calls and 24 exact verifier
outcomes. Peak memory was 71,102,464 bytes with zero memory events and no swap.
These are repeat observations, not new cases or cross-machine evidence.

Independent source review found a diagnostic transport defect in the original
constructor: declared type counts could exceed captured record extents. Two
prelisted malformed packets produced normal assertion failure at the wrong
stage. The corrected constructor checks extents before either product call.
The complete original suite was rerun with unchanged cases, expectations,
mutations, verifier, runner and emitter product. Original source and raw failure
receipts remain preserved; the two malformed packets were independently replayed unchanged on this
corrected version and produced four normal setup refusals (both backends). This finding does not change the product's
explicit live-buffer precondition or imply malformed packets were originally safe.

| Source or artifact | SHA-256 |
| --- | --- |
| Emitter product | `1cb4617a17fb1b3cf6717cdb01c58dfa4f93fb7336157000fe8a49aca5088069` |
| Final MIND constructor | `94c1e97fab6cc2a72ab1a743b05e0aaf90ebdd44eda5f9f90e6c26022f424e62` |
| Independent MIND verifier | `0caf3c4745e7d9313e9cd074e38c0bd6d41cd354f3d939171fd69b864f5bd72f` |
| Native diagnostic compiler | `03e956bdfe049d43d128a387264118c2623916be487868268ec4892af57c093a` |
| CPU compiler | `62bc0f97c4cd9c7f8230bd3af765487626e06bc3918e2c67e35dc058390bc1fd` |
| Captured development standard library | `12aabf233e961756c0e9f3d4ae7fa40e0427c5103cfee65b92db9be2a530d8a9` |
| Final native constructor ELF | `1667405d3616f8f01afd48732206dfa503278692eac993b28296e8e56d0b88e5` |
| Final MLIR constructor ELF | `8d9acffd46a9fba82a7390d651cb0f4869735e873b53c68d4458c195a6301417` |
| Portable observation archive | `e1422a603d06931f4fe69b8239a220d7e940c90f646e096accafb2d7013a3467` |

`BUILD_HASHES.tsv` identifies all 22 original final-source build artifacts.
`OBSERVED.tsv` records the 310 constructor/verifier outcomes.
`RESULTS.tar.gz` contains unmodified MIND protocol streams, unbound code,
canonical receipts, build statuses and resource observations, with an inner
SHA-256 manifest. It excludes compiled ELFs, path-bearing compiler stdout and raw
capture frames containing process-specific addresses; exact raw-capture hashes
are recorded and complete raw runs are retained separately. No raw frame is
relabeled as a canonical receipt.

The tested compiler binaries are explicit diagnostic prerequisites, not a new
release or consumer pin provided by this fixture. Missing or mismatched identities
block reproduction. The development standard-library closure is not represented
as a refreshed historical seed. Architecture-summary tooling currently excludes
`examples.*`; an unchanged repository score cannot validate this compiler module.
Source review checks the additive dependency on unchanged ncp/nce/ncv helpers.
Compiler Criterion/performance and the original fixed58 gate remain integration
obligations, not inferred successes from these component tests.

The standalone harness inventory rises from 307 to exactly 309 for
`native_c_variadic_emit_check.py` and its captured `run.sh`. The wrapper owns
capture/identity and the shell owns bounded build/run transport; executable
construction and semantic assertions remain MIND. No spare allowance is added.


## Independent review outcome

The independent component review returned `PASS_WITH_NOTES` on the exact product,
corrected constructor and verifier hashes above. Its 16 separately prelisted
logical controls produced 74 processes: 14 real C captures, 36 MIND verifier
outcomes, six normal product-bound refusals and four corrected setup refusals on
the original malformed packets. The remaining processes construct the call
artifacts. Controls include additional signed minima, poisoned widths, maximum
finite/subnormal f32 values, mixed register exhaustion, 256 arguments and shared
slots, plus forged AL, MXCSR control and result receipts. Altering an allowed
MXCSR status bit correctly preserves the canonical receipt. Peak memory was
67,092,480 bytes, with zero memory events and no swap.

This was a separate model-assisted source review and independent execution using
copied, hash-checked author-built artifacts. It was not an independent fresh
compiler rebuild, independent human acceptance or a formal proof. No core emitter
defect was confirmed within the stated live-buffer and FP-environment contract.
The diagnostic extent finding was corrected and closed; its original failures
remain in the record. The original 55/8 author denominator is unchanged.

The full independent report SHA-256 is
`2273e834feb105c1998844115b2b8b94e6e77904d9589543c404c1e5bca522ea`;
its 375-file evidence seal is
`4ee4c9520661bed33fefd8a002746bca9080331182b76d8953056b06e4d3e7cd`.
`INDEPENDENT_RESULTS.tar.gz` publishes a provenance-labeled subset: fixed input
and expectation bytes, original extent-defect source/streams, unchanged MIND
receipts and a path-free projection of the 74 process records. It explicitly
omits compiled instruments, private tool paths and raw address-bearing capture
frames; complete original evidence is retained separately. Its inner manifest
can be checked with `sha256sum --check EVIDENCE.sha256` after extraction.
