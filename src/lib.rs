//! MIND core library (Phase 1 scaffold)
pub mod ast;
pub mod lexer;
pub mod parser;
pub mod types;
pub mod type_checker;
#[cfg(feature = "autodiff")]
pub mod autodiff;

#[cfg(feature = "mlir")]
pub mod ir;
