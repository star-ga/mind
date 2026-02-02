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

//! MIND core library for the Phase-2 Core pipeline defined in mind-spec Core v1.
//!
//! The pipeline progresses through the following stable stages:
//!
//! * **Surface front-end** → parses MIND source into the typed AST.
//! * **Public IR** → a stable, verified representation of ops, shapes, and
//!   broadcasting/reduction semantics.
//! * **Autodiff** → static gradient generation built on the public IR API.
//! * **Canonicalization** → deterministic, semantic-preserving rewrites prior to
//!   lowering.
//! * **MLIR lowering** → text MLIR emission for backends (feature gated).
//! * **Runtime** → execution of canonical IR or lowered artifacts.
//!
//! # Stability & versioning
//!
//! MIND Core follows the mind-spec Core v1 stability contract:
//!
//! * **Stable**: public IR structure/semantics, autodiff API, canonicalization
//!   guarantees, CLI base flags, and textual IR form.
//! * **Conditionally stable**: MLIR lowering is stable within a given minor
//!   release when the `mlir-lowering` feature is enabled.
//! * **Experimental**: new ops, experimental flags, and future non-CPU backends.
//!
//! Diagnostics follow the Core v1 error model with structured spans and
//! machine-readable JSON output (`mindc --diagnostic-format json`).
//!
//! See `docs/versioning.md` for the full policy and surface definitions.
pub mod ast;
pub mod conformance;
pub mod diagnostics;
pub mod eval;
pub mod exec;
pub mod ir;
pub mod lexer;
pub(crate) mod linalg;
#[cfg(feature = "mlir-lowering")]
pub mod mlir;
pub mod ops;
pub mod opt;
pub mod parser;
pub mod pipeline;
pub mod runtime;
pub mod runtime_interface;
pub mod shapes;
pub mod stdlib;
pub mod type_checker;
pub mod types;

#[cfg(feature = "autodiff")]
pub mod autodiff;
#[cfg(feature = "autodiff")]
pub use autodiff::{
    differentiate_function, differentiate_with_options, AutodiffError, GradientOptions,
    GradientResult,
};
pub use conformance::{
    run_conformance, ConformanceFailure, ConformanceOptions, ConformanceProfile,
};
pub use pipeline::{
    compile_source, compile_source_with_name, CompileError, CompileOptions, CompileProducts,
};
#[cfg(feature = "mlir-lowering")]
pub use pipeline::{lower_to_mlir, MlirProducts};
pub use runtime::types::{BackendTarget, DeviceKind};

#[cfg(feature = "mlir-lowering")]
pub use mlir::{compile_ir_to_mlir_text, MlirLowerError};

#[cfg(feature = "ffi-c")]
pub mod ffi;

#[cfg(feature = "pkg")]
pub mod package;

pub mod project;

#[cfg(feature = "python-bindings")]
pub mod python;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn mind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)
}
