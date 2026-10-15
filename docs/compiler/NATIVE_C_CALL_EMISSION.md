# Fixed scalar C call emission

Author: STARGA Inc.

`native_c_call_emit.mind` emits one Linux x86-64 LP64 SysV call from the
source-typed declaration row used by the [call planner](NATIVE_C_CALL_PLAN.md).
It recomputes the plan from the row; it does not trust a supplied successful
plan. This standalone component does not enable extern declarations in the
production compiler or establish that a library symbol has the declared ABI.

## Inputs and ownership

```text
nce_validate(row, slots, frame_bytes, result_disp, import_index) -> status
nce_emit(row, slots, frame_bytes, result_disp, import_index, metadata) -> String
```

The row and its type records have the planner's immutable, readable storage
preconditions. `slots` contains one signed i64 RBP-relative displacement per
argument. `metadata` is a separate writable four-word buffer; it must not alias
any input. The row and slots are borrowed, never changed or retained.

The caller has an allocated RBP frame with a positive, 16-byte-aligned size of
at most 2,147,483,632 bytes. Each slot must be eight-byte aligned and between
`-frame_bytes` and `-8`. A nonvoid result requires a destination in that range;
it may reuse an argument slot after the call. Void does not use the destination.
This validation checks displacement bounds, not whether a caller actually
allocated the promised frame. The call starts with RSP at the frame bottom,
aligned to 16 bytes, and no live values held only in caller-clobbered registers.

The import index is in 0–255, matching the current writer's bounded table.
It is a table index, not permission to call an arbitrary symbol. The caller
must bind the original declaration, source argument types, library ABI and
symbol/version before using the emitted call.

Success returns an owned String and:

```text
[version=1, displacement_offset, import_index, aligned_stack_reservation]
```

The offset identifies the four zero bytes in `FF 15 disp32`, relative to the
returned code. The dynamic writer must resolve this explicit GOT relocation
before execution. Refusal returns an empty String and all-zero metadata.
The internal plan allocation is released once before success returns. Callers
own and release the returned String's backing allocation.

## Instruction semantics

The emitter reserves the planner's aligned stack space. It loads signed narrow
arguments with sign extension and unsigned narrow arguments with zero extension;
bool uses its valid low-byte representation. Full-width integers and pointers
retain all 64 bits. GP arguments occupy RDI, RSI, RDX, RCX, R8 and R9; f64
arguments occupy XMM0–7. The two pools are independent. Ordered spills occupy
eight-byte slots at the planner's exact RSP offsets, with padding above them.
RAX is a scratch register outside the argument-register pools. f64 spills copy
their raw 64-bit payload. The implementation does not rely on the red zone.

After the call, the emitter restores RSP and stores the result in its frame
slot. Narrow INTEGER returns are normalized from their meaningful low bits;
unspecified high register bits cannot affect the value. f64 comes from XMM0.
Void stores nothing. Valid C bool values, SysV callee-saved registers and foreign
memory safety remain the callee's obligations. There is no vararg AL count,
promotion, callback, aggregate, f32 or alternate-convention support here.

Validation retains planner statuses 1–8 and adds 9 for invalid frame size,
10 for an invalid import index, 11 for an invalid source slot and 12 for an
invalid result slot. No code or relocation is returned on these refusals.

## Reproduction

Use a current source-bound native compiler and the supported official CPU/MLIR
configuration. Supply exact hashes; the harness checks them before building.
Configure `MLIR_OPT`, `MLIR_TRANSLATE`, `CLANG`, `CC` and `MIND_STDLIB_PATH` as
for the existing CPU build. Diagnostic Python and Clang compile and run the
independent C/assembly ABI fixture; neither is an emitted-product dependency.

```sh
python3 examples/mindc_mind/native_c_call_emit_check.py \
  --enumerate --evidence /fresh/absolute/call-emission-evidence
python3 examples/mindc_mind/native_c_call_emit_check.py \
  --mutations --evidence /fresh/absolute/call-emission-evidence \
  --seed /path/to/current-native-compiler --seed-sha256 "$NATIVE_SHA256" \
  --mindc /path/to/mindc --mindc-sha256 "$MINDC_SHA256" --clang /path/to/clang
```

The harness freezes 58 cases before builds, requires native/MLIR emitted bytes
to match and checks each expected refusal. Valid cases execute the emitted code
against independently compiled ABI fixtures, including mixed spilled arguments,
all supported scalar widths, a maximum-size signature, excess return-bit
canaries and exact post-call stack restoration. The test host maps executable
memory for this diagnostic; that is not the production image's W^X policy.
Crashes fail a positive case. Three actual changed MIND sources exercise the
import bound, zero-on-refusal metadata and signed return normalization; only
normal wrong outputs or a normal result-comparison failure count as detection.

Observed author runs passed 116 driver outcomes, including 72 actual calls,
and detected all three source mutations without counting signals or build
failures as detection. All ten emitter functions passed a separate current
MIC emission probe. Source/compiler/artifact hashes, case packets, observed
streams, status codes and diagnostic fixture sources are retained by the
harness. The case catalogue SHA-256 is
`e16d94d16157fd9d53391e026b1133bae5c7092879df32370c0065e026555e8e`.

Independent implementation review is pending. This component is not production
C FFI qualification. Dynamic startup, complete source/library binding, runtime
ownership and errno semantics, relocations in the complete executable, and the
unchanged downstream qualification remain separate obligations. Existing local
call emission and frozen compiler seeds are unchanged.
