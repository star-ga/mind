# f64 call-ABI negative control (paired with the #298 self-host f64 MLIR surface)

`scale_neg.mind` passes a typed **i64 variable** to an `f64` parameter — an invalid
f64 call ABI. It MUST be rejected at the MLIR-lowering ABI boundary with:

    error: use of value '%N' expects different type than prior uses: 'f64' vs 'i64'

`scale_pos.mind` is byte-identical except the argument variable is `f64` — it MUST
compile. The gate `tests/f64_abi_negative_control.rs` asserts both machine-checkably
(negative fails for that exact reason — not parse, not link; positive compiles).
