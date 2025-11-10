//! MIND core library (Phase 1 scaffold)
pub mod ast;
pub mod diagnostics;
pub mod eval;
pub mod exec;
pub mod lexer;
pub(crate) mod linalg;
pub mod opt;
pub mod parser;
pub mod stdlib;
pub mod type_checker;
pub mod types;

pub mod ir;

#[cfg(feature = "autodiff")]
pub mod autodiff;
