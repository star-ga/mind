// Regression test for #4: lowering a `for` loop to IR (the path `mindc --emit-ir`
// uses) must not PANIC. Previously the mic@1 value-position lowering (lower_expr)
// had no `For` arm and hit the fail-closed `panic!`. The fix desugars For→While
// and reuses the existing While lowering. (The MLIR build path lowers For
// separately and is unaffected.)

use libmind::eval::lower;
use libmind::parser;

// For→While desugar reuses the `While` lowering arm, which is gated behind
// `std-surface`; without that feature there is no For/While lowering and
// `lower_to_ir` fail-closes (panics) by design. Gate the test to the config
// where the feature under test is actually compiled in.
#[cfg(feature = "std-surface")]
#[test]
fn for_loop_lowers_to_ir_without_panic() {
    let src = "fn main() -> i64 { let mut s: i64 = 0; for i in 0..10 { s = s + i; } return s; }";
    let module = parser::parse(src).expect("parse failed");
    // Must not panic on the For (was: `no IR lowering for For in value position`).
    let ir = lower::lower_to_ir(&module);
    // Sanity: lowering produced a non-empty module.
    assert!(
        !ir.instrs.is_empty(),
        "lowering a for-loop program must yield IR instructions"
    );
}
