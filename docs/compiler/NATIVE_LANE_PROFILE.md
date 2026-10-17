# Typed scalar delivery profile

Author: STARGA Inc.

`src/ir/native_lane_profile.rs` provides a conservative, source-aware native
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

Unsigned integers, narrow integer expressions and casts, f32 and
float-to-float casts, pointers, C externs, aggregates, tensors, methods,
implicit standard-library objects, closures and unrecognized syntax refuse. A raw intrinsic accepting an i64 address does not
prove memory safety of that address. The predicate is a compiler feature fence,
not a memory-safety or theorem verifier.

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
