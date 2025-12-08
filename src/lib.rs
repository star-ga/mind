// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! MIND core library (Phase 1 scaffold)
pub mod ast;
pub mod diagnostics;
pub mod eval;
pub mod exec;
pub mod ir;
pub mod lexer;
pub(crate) mod linalg;
#[cfg(feature = "mlir-lowering")]
pub mod mlir;
pub mod opt;
pub mod parser;
pub mod pipeline;
pub mod runtime_interface;
pub mod stdlib;
pub mod type_checker;
pub mod types;

#[cfg(feature = "autodiff")]
pub mod autodiff;
#[cfg(feature = "autodiff")]
pub use autodiff::{differentiate_function, AutodiffError, GradientResult};
pub use pipeline::{compile_source, CompileError, CompileOptions, CompileProducts};
#[cfg(feature = "mlir-lowering")]
pub use pipeline::{lower_to_mlir, MlirProducts};

#[cfg(feature = "mlir-lowering")]
pub use mlir::{compile_ir_to_mlir_text, MlirLowerError};

#[cfg(feature = "ffi-c")]
pub mod ffi;

#[cfg(feature = "pkg")]
pub mod package;
