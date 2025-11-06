//! MLIR lowering stub (Phase 1).
//!
//! Feature `mlir` enables this pure-Rust placeholder with no external deps.
//! For the real integration, use feature `mlir_backend` (melior) in a future PR.

#[cfg(feature = "mlir")]
pub fn lower_placeholder(source: &str) -> String {
    let one_line = source.lines().collect::<Vec<_>>().join(" ");
    format!("mlir.module {{ // lowered from: {} }}", one_line)
}
