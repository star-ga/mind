# Typed scalar delivery profile

Author: STARGA Inc.

`src/ir/native_lane_profile/mod.rs` provides a conservative, source-aware native
capability predicate. It does not change the historical frozen profile, select
an executable, qualify a compiler pin, or establish consumer parity. Selection
must separately bind a compiler with the corresponding verified capabilities.

The caller checks the source before lowering, lowers that same source once,
then supplies the source and resulting IR to the combined predicate. The
predicate is not an equivalence checker for independently supplied source/IR.
Closure, namespace composition and runtime resource bounds remain separate
mandatory checks. All source functions, including unreachable functions, are
checked. Unknown names and intrinsic overrides refuse.

The initial surface comprises explicitly typed i64/bool/f64 parameters and
results, scalar literals and constants, typed direct calls, scalar arithmetic,
bit operations on i64, comparisons, structured if/while/range-for, return and
scoped loop exits. Conditions use the existing i64 truth-value representation.
Unary and binary operations check operand types. Only f64 floating-point
operations are admitted. Runtime unary f64 negation currently refuses because
its legacy lowering emits integer zero minus the floating operand; explicit
`0.0 - x` and negated float literals have a uniformly typed representation.
Source annotations are essential because legacy IR
can erase signedness and widths.

Full-width `i64 as f64` and `f64 as i64` conversions are admitted. Both
backends convert i64 to f64 with IEEE round-to-nearest-even and f64 to i64 by
saturating truncation: NaN gives 0, values at or beyond the i64 range give
`i64::MIN` or `i64::MAX`. In IR these are `__mind_conv_f64` and
`__mind_conv_i64`, typed by their source value; an explicit source call to a
conversion intrinsic still refuses. A plain top-level `import module;` is a
declaration the bridge's resolver already linked into the image; an `as`
rename refuses, and calls still bind only to uniquely named functions of the
one composed image.

Fixed i64 array reads are admitted in one shape: `let a = [1, 2, 3];` or
`let a: [i64; 3] = [1, 2, 3];` (each element an integer constant expression,
such as `0 - 1`, folded once by the spec 3.2 rule; the annotated length equal
to the literal's) and `a[i]` with an i64 index, typed i64. The compiler
emits the read with a runtime bounds trap (exit 77, the same code as MLIR) and
refuses a constant out-of-bounds index. Any other array use refuses by name: the
array as a value, operand, argument, return value or assignment target
(`lane.source.array_value`), an element store (`lane.source.index_store`), an
index on anything but an array binding (`lane.source.index`), a non-literal
element that does not fold, including a division by zero or an overflow
(`lane.source.array_element`) and an array-typed parameter or
mismatched annotation (`lane.type.array`). In IR only an `ArrayLoad` may read a
`ConstArray` result; the array value never enters the scalar type env.

Unsigned integers, narrow integer expressions and casts, f32 and
float-to-float casts, pointers, other aggregates, tensors, methods,
implicit standard-library objects, closures and unrecognized syntax refuse.
A typable `extern "C"` block is admitted when the selected compiler carries
the lane mark, and the compiler's rows name a refusal. A `print` of an i64
builds on a compiler minted from this tree (it writes the integer and exits
0). The committed seeds carry `nblane02`, so the host admits that build; a compiler that carries `nblane01` is refused before this predicate runs (`native.compiler_mark`).
Each refusal
names its construct (`lane.type.tensor`, `lane.type.int_width`,
`lane.source.struct`, `lane.source.trait`, `lane.ir.tensor`, ...); the generic
`*.unsupported` kinds remain only for what has no name of its own. A raw intrinsic accepting an i64 address does not
prove memory safety of that address. The predicate is a compiler feature fence,
not a memory-safety or theorem verifier.

The type named exactly `String`, a string literal and a field read are typed
as the i64 word, and this predicate decides nothing more about them: a field
read walks its receiver first, a field write still refuses
(`lane.source.struct`), and every other type name keeps its refusal
(`lane.type.named`, `lane.type.int_width`). The rules for these three belong to
the compiler that carries the `nblane02` mark. A compiler that does not carry
`nblane02` is refused by the host before this predicate runs. The host does
not apply the historical frozen profile to that compiler.

Canonical intrinsic contracts supply the existing exact i64 ABI for allocation,
loads, stores, read and write. The canonical contract marks store results
DiscardOnly because the historical frozen seed leaves their result slot
unwritten. This profile applies only to a compiler carrying the lane marker,
whose value-position store lowering writes the C runtime's defined 0, the value
the MLIR lowering materialises. A store result is therefore a statement, or the
i64 value 0 where a value is read (the `w_set` helper shape returns it). It is
never an f64, and a constant initializer still refuses it. Additional
explicit contracts cover load/store i32, free/realloc, signed word conversion,
f64 bit conversion, and the implemented raw Linux syscall family. The raw
syscall surface takes word arguments and returns raw results, including negative
errno. It does not implement libc external declarations or thread-local errno.
The name/arity inventory alone never authorizes a call: both source and IR types
must also pass.

Native-only rows are registry rows whose profile is FrozenNative alone: the AVX2 ymm rows `__mind_vpaddd_i32x8`, `__mind_vpsubd_i32x8`,
`__mind_vpaddb_i8x32`, `__mind_vpshufb_i8x32`, `__mind_vmovdqu_i32x8`,
`__mind_vpaddq_i64x4`, `__mind_vpsubq_i64x4` (seed ids 60-66, operands are vector
addresses) and `__mind_cpu_features_v1` (67, a World row: a program reaching it is
attested nondeterministic). A seed id is the self-host intrinsic kind. The type
checker arity-checks them in any profile, so `--emit-mic3` and `--emit-evidence`
carry the call by name. E2024 fires only for a `__mind_*` name registered in no
profile (an unregistered or misspelled name, which then fails type checking); a
registered native-only row carries no E2024. An MLIR build refuses a
native-only row by name with E2041 before it lowers or links. argc/argv (seed
ids 10, 11) are in both profiles: the MLIR path calls the runtime-support pair.

The AVX2 group family is three FrozenNative value rows (each returns 0 and
writes memory): `__mind_group_bank_i64(ptrs, bank, plane, pixels, group)` (87),
`__mind_group_acc32_i64(out, bank, off, pos, neg, init)` (88) and
`__mind_group_accrows_i64(desc)` (89, one call walks many accumulator rows).
Kinds 87 and 88 probe the CPU and take a scalar path without AVX2; kind 89 has one
AVX2 path behind the program's ymm entry guard. Probe-skip contract: bit 4 of
`group` (kind 87, the group used is `group & 15`) and an `init` in [2^62, 2^63)
(kind 88, bit 62 is cleared and "keep" is decided on the cleared value) skip the
probe and force the AVX2 path. A program that sets a probe-skip bit must also hold
a guarded row (any ymm row or kind 89); without one, a CPU with no AVX2 stops at
the first VEX op (SIGILL) and prints no `isa.avx2_missing` line.

| Kind | Bits | Mode |
|---|---|---|
| 87 | `group` bit 5 | store the low 32 bits of each code (4-byte rows) |
| 87 | `group` bits 5 and 6 | u32 channels, 8 pixels per vector step |
| 87 | `group` bit 6 alone, bits 7..63 | trap (ud2) before any store |
| 88 | `init` above 3 after the probe-skip arm | trap (ud2) before any store |
| 89 | `desc[7]` bit 0 | keep or add to `out` |
| 89 | `desc[7]` bit 2 | i32 walk over a 4-byte bank, 64 lanes; bits 63..32 = segment bound, 2^31 or more takes the exact i64 fallback |
| 89 | `desc[7]` bits 3, 4 | term loop: two terms per step (3), u16 lane offsets (4) |
| 89 | `desc[7]` bit 5 | epilogue tail over the stored rows (`desc[8]` records, `desc[9]` trip flag, `desc[10]` limit) |
| 89 | `desc[7]` bit 7 | fused row finish in registers (`desc` is i64[15]; `desc[11..14]` src, src step, dst, dst step) |
| 89 | `desc[7]` bit 8 | u32 src and dst for bit 5 or bit 7; `desc[10]` must fit in 32 bits |
| 89 | bit 1 with bit 2, 3 or 4; bit 6; bits 9..31; bits 63..32 without bit 2; bit 8 without bit 5 or bit 7 | trap (ud2) before any store |
A `desc[7]` value of 2 (bit 1 set, no other mode bit) is valid and starts the accumulator row from zero.
This profile types the rows; the compiler that lowers them carries a capability
mark in its own bytes (`nbymm001`, `nbcpuf01`). A compiler without the row
refuses the call at its own closure check (exit 3, no artifact).

An allocation size must be proven positive (`lane.alloc.size_positive`), because
the marked compiler bumps its arena by any size while the C runtime and the
evaluator return 0 for a size of 0 or less. The proof (`alloc_size.rs`) accepts a
positive integer literal, or an i64 name the function never assigns after a
statement `if n <= 0 { ... return; }` with no `else` (`n < 1`, `0 >= n`, `1 > n`
and `break`/`continue` exits too); it proves no arithmetic. The native bridge
respells every other `__mind_alloc(e)` in a user function body as a call to an
appended guard, `__lane_alloc`, which returns 0 for `n <= 0` and calls
`__mind_alloc(n)` otherwise (`src/build/native_bridge/alloc_guard.rs`). It
respells every other `__mind_realloc(p, e)` as `__lane_realloc`, which calls
`__mind_realloc` for a positive size and returns 0 for a non-positive size,
except `realloc(NULL, 0)`, which returns a 24-byte block (the C runtime's
`malloc(0)`, and the same block the evaluator allocates). Each guard name has
the callee's length, so no offset moves, and an image with no dynamic size is
unchanged. The guards give the C-runtime result on native and MLIR; they are
a defined result, not a trap. An image that already names a guard is not
respelled for that callee, and the unproven call keeps `lane.alloc.size_positive`.

The IR walker exhaustively handles every instruction variant; unsupported
variants refuse. It checks SSA, function binding, typed operands and results,
parameter indices, branch-local scopes, loop-carried values, break/continue
snapshots and the i64 typing of store and conversion results. Optional scalar
metadata is checked,
never used as an authorization shortcut. New IR variants require an explicit
decision because the instruction match has no wildcard admission.

Run the source-independent positive and adversarial tests with:

```sh
cargo test --lib --no-default-features --features std-surface,cross-module-imports ir::native_lane_profile -- --test-threads=1
cargo test --lib --no-default-features --features std-surface,cross-module-imports ir::frozen_profile -- --test-threads=1
```

The tests include genuine source lowering and malformed IR: erased non-f64 or
unsigned/narrow types, mixed float/integer operations, unknown/dead calls,
wrong arity, intrinsic overrides, wrong return type, altered parameter index,
branch scope escape, malformed loop tuples, a store result forged into a float
operation, a float forged into the int-to-float conversion, refused casts and
renamed imports. A refusal carries the function name and, for source checks,
the byte offset of the refused node; the bridge reports it as a line and column
of the composed user image. Passing
these profile tests does not substitute for execution parity on a reseeded
compiler or for the final consumer acceptance contract.
