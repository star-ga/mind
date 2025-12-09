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

//! High-level compiler pipeline utilities for the public MIND front-end.
//!
//! The helpers in this module parse, type-check, and lower MIND source code
//! into the public IR. They optionally run the static autodiff pipeline and
//! can lower the resulting IR into MLIR text when the `mlir-lowering` feature
//! is enabled.

use std::collections::HashMap;

use crate::diagnostics;
use crate::eval;
use crate::ir;
use crate::opt;
use crate::parser;
use crate::runtime::types::BackendTarget;
use crate::type_checker;

#[cfg(feature = "autodiff")]
use crate::autodiff;
#[cfg(feature = "mlir-lowering")]
use crate::mlir;

/// Options controlling the compiler pipeline.
#[derive(Debug, Default, Clone)]
pub struct CompileOptions {
    /// Optional function name to focus on (used by autodiff and MLIR).
    pub func: Option<String>,
    /// Whether to run autodiff for the selected function.
    pub enable_autodiff: bool,
    /// Requested execution backend (CPU is the only supported target).
    pub target: BackendTarget,
}

/// Artifacts produced by [`compile_source`].
#[derive(Debug, Clone)]
pub struct CompileProducts {
    /// Verified and canonicalized IR for the input source.
    pub ir: ir::IRModule,
    /// Gradient IR when autodiff is enabled.
    #[cfg(feature = "autodiff")]
    pub grad: Option<autodiff::GradientResult>,
}

/// Errors surfaced by the high-level compilation pipeline.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// Parsing failed with one or more diagnostics.
    #[error("parse error")]
    ParseError(Vec<diagnostics::Diagnostic>),
    /// Type checking failed with one or more diagnostics.
    #[error("type error")]
    TypeError(Vec<diagnostics::Diagnostic>),
    /// The IR module did not pass verification.
    #[error("IR verification failed: {0}")]
    IrVerify(#[from] ir::IrVerifyError),
    /// Autodiff requested without specifying a function.
    #[error("autodiff requested but no function was provided")]
    MissingFunctionName,
    /// Requested backend target is not available in this build.
    #[error("backend unavailable: {target}")]
    BackendUnavailable { target: BackendTarget },
    /// Autodiff failed with a structured error.
    #[cfg(feature = "autodiff")]
    #[error("autodiff failed: {0}")]
    Autodiff(#[from] autodiff::AutodiffError),
    /// Autodiff was requested but the feature is not enabled.
    #[cfg(not(feature = "autodiff"))]
    #[error("autodiff requested but the 'autodiff' feature is not enabled")]
    AutodiffDisabled,
}

/// High-level compiler pipeline entry point: parse, type-check, lower to IR,
/// verify, and canonicalize. Optionally run autodiff for the requested
/// function.
pub fn compile_source(
    source: &str,
    opts: &CompileOptions,
) -> Result<CompileProducts, CompileError> {
    if matches!(opts.target, BackendTarget::Gpu) {
        return Err(CompileError::BackendUnavailable {
            target: opts.target,
        });
    }

    let module = parser::parse_with_diagnostics(source).map_err(CompileError::ParseError)?;

    let type_diags = type_checker::check_module_types(&module, source, &HashMap::new());
    if !type_diags.is_empty() {
        return Err(CompileError::TypeError(type_diags));
    }

    let mut ir = eval::lower_to_ir(&module);
    ir::verify_module(&ir)?;
    opt::ir_canonical::canonicalize_module(&mut ir);
    ir::verify_module(&ir)?;

    #[cfg(feature = "autodiff")]
    let grad = if opts.enable_autodiff {
        let func = opts
            .func
            .as_deref()
            .ok_or(CompileError::MissingFunctionName)?;
        let grad = autodiff::differentiate_function(&ir, func)?;
        Some(grad)
    } else {
        None
    };

    #[cfg(not(feature = "autodiff"))]
    if opts.enable_autodiff {
        return Err(CompileError::AutodiffDisabled);
    }

    Ok(CompileProducts {
        ir,
        #[cfg(feature = "autodiff")]
        grad,
    })
}

/// MLIR lowering artifacts for the canonical IR (and optional gradient IR).
#[cfg(feature = "mlir-lowering")]
#[derive(Debug, Clone)]
pub struct MlirProducts {
    /// Textual MLIR for the canonical IR module.
    pub primal_mlir: String,
    /// Textual MLIR for the gradient IR module when autodiff is enabled.
    pub grad_mlir: Option<String>,
}

/// Lower canonical IR (and optional gradient IR) into MLIR text.
#[cfg(feature = "mlir-lowering")]
pub fn lower_to_mlir(
    ir: &ir::IRModule,
    grad: Option<&autodiff::GradientResult>,
) -> Result<MlirProducts, mlir::MlirLowerError> {
    let mut canonical_ir = ir.clone();
    ir::verify_module(&canonical_ir)?;
    opt::ir_canonical::canonicalize_module(&mut canonical_ir);
    ir::verify_module(&canonical_ir)?;

    let primal_mlir = mlir::lower_ir_to_mlir(&canonical_ir)?.text;

    let grad_mlir = if let Some(grad) = grad {
        let mut grad_ir = grad.gradient_module.clone();
        ir::verify_module(&grad_ir)?;
        opt::ir_canonical::canonicalize_module(&mut grad_ir);
        ir::verify_module(&grad_ir)?;
        Some(mlir::lower_ir_to_mlir(&grad_ir)?.text)
    } else {
        None
    };

    Ok(MlirProducts {
        primal_mlir,
        grad_mlir,
    })
}
