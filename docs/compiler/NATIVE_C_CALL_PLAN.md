# Fixed scalar C call plans

Author: STARGA Inc.

`examples/mindc_mind/native_c_call_plan.mind` classifies a bounded, fixed
signature for Linux x86-64 LP64 SysV. It produces compiler metadata; it does
not emit a call or enable foreign functions. Source argument checking, symbol
and library ABI binding, call relocation, stack setup, runtime initialization
and production qualification remain separate obligations.

The input is the live immutable internal row produced by the
[source-typed registry](../../examples/mindc_mind/native_cabi_registry.md),
not a serialized untrusted pointer table. All row, parameter and type buffers
must remain readable and unchanged for the operation. The row is64 bytes;
parameter slots are pointers to48-byte type records. Source name spans are
irrelevant to register classification and remain owned by the registry.
Pointer validity is a caller precondition, not something this classifier proves.

`ncp_validate(row)` returns zero or a diagnosed category. `ncp_plan(row)`
validates again and returns a fresh allocation, or zero on refusal/allocation
failure. The caller releases a successful allocation with `__mind_free`.
It never modifies its input or returns a partial plan. Native allocator
exhaustion may terminate according to its documented resource policy; this
component does not claim universal recoverable allocation failure.

Supported declarations have convention C or explicit SysV, valid safety
metadata, no varargs, and at most256 parameters. Integer widths/signedness,
LP64 size types, bool, f64 and validated raw pointers are preserved. Void is
allowed only as the result. Scalar f32, unsupported conventions, varargs and
malformed type records refuse. A pointer to f32 is a pointer and is supported;
it is never mistaken for an f32 value. Pointer mutability/depth/pointee must
still be checked by the source type and library contract; classification does
not discard the original registry's authority over those distinctions.

## Plan format

The owned plan is80 +40×parameter_count bytes. All fields are signed64-bit
compiler words. Its ten-word header is:

```text
version=1, parameter_count, GP_registers_used, SSE_registers_used,
stack_argument_bytes, aligned_stack_reservation,
result_type_code, result_class, result_width_bits, result_normalization
```

Each parameter, in source order, has five words:

```text
type_code, location_class, location, width_bits, normalization
```

Type codes retain scalar registry codes;14 denotes a validated pointer. This
pointer marker is a machine-placement class, not a substitute source type.
Location1 uses GP registers0–5 in RDI, RSI, RDX, RCX, R8, R9 order. Location2
uses XMM0–7. Location3 is an eight-byte stack slot at the given byte offset
from RSP immediately before CALL. GP and SSE pools are independent; exhaustion
of one does not consume the other. Spilled scalar arguments retain source
order. Any alignment padding follows the highest-address argument. The
reservation rounds up to16 bytes and assumes the caller's RSP was already
16-byte aligned before subtracting it; it cannot repair an unknown alignment.
The callee sees these stack slots starting at RSP+8 after CALL's return address.

Result class0 means no result;1 means RAX;2 means XMM0. Width is the meaningful
value width, not the full register width. Normalization0 preserves all64 bits;
1 sign-extends the low width;2 zero-extends it;3 handles a valid low-byte C bool
value0/1. Do not test unspecified high bits when reading a narrow foreign
result. For outgoing arguments these actions specify canonical compiler slot
representations; the ABI does not require the same canonical excess bits.
Bool normalization is not permission to treat an invalid foreign bool as valid.
No vararg AL count, default argument promotions, aggregate classification,
callbacks, hidden result pointers or calling-convention adaptation is provided.

These rules follow the upstream
[x86-64 psABI parameter/return specification](https://gitlab.com/x86-psABIs/x86-64-ABI/-/blob/master/x86-64-ABI/low-level-sys-info.tex).
The inspected2026-10-15 source SHA-256 is
`2d42f2ab76b99a3b8456e6bc07e18314e85991119f72db90ca1175328e27b705`.
No upstream implementation code is copied. Tests validate the implemented
metadata subset; actual external-call execution requires a separate ABI gate.

## Refusals and tests

Validation checks, in order:1 unsupported convention;2 invalid safety bit;
3 varargs;4 parameter count;5 malformed or unsupported type structure;
6 scalar f32. Null row or a missing nonempty parameter table returns8.
The test driver reports allocation refusal as7. Within a parameter list the
first detected type refusal is preserved.

The MIND diagnostic driver takes four i64 words(convention,safety,varargs,count),
one48-byte return descriptor and count48-byte parameter descriptors. An invalid
count has no parameter payload. It rejects truncated/trailing transport data
with a separate nonzero process status; those failures never count as a valid
classifier refusal. It returns error,size followed by exactly size plan bytes.

Use a compiler built from the current source, not the historical bounded seed
directly. The frozen `e9de4480...` seed refuses this diagnostic with exit 3;
that refusal is retained as a bootstrap prerequisite result. For example,
`self_host_source_note_smoke.py --evidence-dir /tmp/source-note-build` rebuilds
the current compiler and checks its source-note regression. Its
`current-compiler.elf` can then be supplied below, with its recorded SHA-256.
This is a diagnostic compiler build, not a qualified compiler-pin update.

Freeze the 60-case catalogue before either executable runs:

```sh
python3 examples/mindc_mind/native_c_call_plan_check.py \
  --enumerate --evidence /tmp/call-plan-cases
python3 examples/mindc_mind/native_c_call_plan_check.py \
  --evidence /tmp/call-plan-cases \
  --seed /path/to/bounded-native-compiler --seed-sha256 "$NATIVE_SHA256" \
  --mindc /path/to/mindc --mindc-sha256 "$MINDC_SHA256"
```

The second command builds both drivers from the exact library and diagnostic
driver, using the explicit hash-checked compilers. Configure MLIR_OPT,
MLIR_TRANSLATE, CLANG/CC and MIND_STDLIB_PATH for the supported CPU toolchain.
The harness records source/compiler/executable hashes and build streams in
the evidence directory. Alternatively provide independently built artifacts
with `--native` and `--oracle`; that mode never claims to have rebuilt them. The
harness requires all120 expected outcomes, exact stderr/status and the frozen
case hash; two matching wrong results cannot pass. It uses no live service.
Original failed setup attempts, when present, must remain distinguishable from
compiler defects and successful execution evidence.

The standalone gate is classified as manual in `SMOKE_WIRING.tsv`. CI execution
is deferred until its current-compiler prerequisite is wired into the runner.

## Observed component verification

The final library SHA-256 is
`304f66fb4b40f885983a48eae19d41a693dbcb399474c66882d50d92a082427b`.
The unchanged case-catalogue SHA-256 is
`8c407f164e35f05d27cff5262c9ce29bcd3cc2892dd9b3857749511a716f1e2d`.

- All 120 native/MLIR expected outcomes passed, including a run using the
  documented build wrapper. Both builds used hash-checked compiler artifacts.
- A separate implementation review reproduced 74 outcomes from 37 additional
  prewritten cases. Six additional ownership/null controls passed through a
  fresh reviewer-built MLIR library; native execution of those six is not claimed.
- The review found an MIC-emission incompatibility in the original placement
  routine. The corrected implementation preserves the test contract and all
  12 of its functions emit successfully through the current MIC path.
- The independent verdict is `PASS_WITH_NOTES` for this metadata component.
  Whole-source integration, real foreign-call execution, runtime semantics and
  the downstream ten-program qualification remain separate gates.

This component does not enable foreign calls, qualify a release, or change a pin.
