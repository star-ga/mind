//! MIND core library (Phase 1 scaffold)
pub mod ast;
#[cfg(feature = "autodiff")]
pub mod autodiff;
pub mod eval;
pub mod lexer;
pub mod parser;
pub mod stdlib;
pub mod type_checker;
pub mod types;

#[cfg(feature = "mlir")]
pub mod ir;
