//! MIND core library (Phase 1 scaffold)
pub mod ast;
pub mod parser;
pub mod eval;
pub mod diagnostics;
pub mod opt;
pub mod stdlib;
pub mod types;
pub mod type_checker;
pub mod lexer;

#[cfg(feature = "mlir")]
pub mod ir;

#[cfg(feature = "autodiff")]
pub mod autodiff;
