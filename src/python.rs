// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Python bindings for MIND compiler benchmarks

use crate::pipeline::{compile_source, CompileOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compile a MIND program and return compilation output.
///
/// # Arguments
///
/// * `source` - A string containing the MIND program source code to compile
///
/// # Returns
///
/// Returns the compiled IR (Intermediate Representation) as a formatted string.
///
/// # Errors
///
/// Returns a `PyValueError` if the compilation fails due to syntax errors,
/// type errors, or other compilation issues.
///
/// # Example
///
/// ```python
/// import mind
/// ir = mind.compile("let x: Tensor[f32,(10)] = 0; x + 1")
/// ```
#[pyfunction]
fn compile(source: &str) -> PyResult<String> {
    let options = CompileOptions {
        func: None,
        enable_autodiff: false,
        target: crate::runtime::types::BackendTarget::Cpu,
    };

    match compile_source(source, &options) {
        Ok(products) => Ok(format!("{:#?}", products.ir)),
        Err(e) => Err(PyValueError::new_err(format!(
            "Compilation failed: {:?}",
            e
        ))),
    }
}

/// Compile a MIND program with autodiff enabled to generate gradient computation.
///
/// # Arguments
///
/// * `source` - A string containing the MIND program source code with function definitions
/// * `func` - Optional name of the function to differentiate (defaults to "main")
///
/// # Returns
///
/// Returns both the forward IR and the gradient module IR as formatted strings.
/// When the autodiff feature is enabled, this includes the generated gradient computation.
/// Without the autodiff feature, only the forward IR is returned.
///
/// # Errors
///
/// Returns a `PyValueError` if:
/// - The compilation fails due to syntax or type errors
/// - The specified function is not found
/// - Autodiff generation fails
///
/// # Example
///
/// ```python
/// import mind
/// program = '''fn main(x: Tensor<F32, [1000]>) -> Tensor<F32, []> {
///     let x_squared = mul(x, x);
///     tensor.sum(x_squared)
/// }'''
/// result = mind.compile_with_autodiff(program)
/// # Or specify a different function:
/// result = mind.compile_with_autodiff(program, func="forward")
/// ```
#[pyfunction(signature = (source, func = None))]
fn compile_with_autodiff(source: &str, func: Option<String>) -> PyResult<String> {
    // Determine which function to differentiate; default to "main" for backwards compatibility
    let func_name = func.unwrap_or_else(|| "main".to_string());

    let options = CompileOptions {
        func: Some(func_name),
        enable_autodiff: true,
        target: crate::runtime::types::BackendTarget::Cpu,
    };

    let products = compile_source(source, &options)
        .map_err(|e| PyValueError::new_err(format!("Compilation failed: {:?}", e)))?;

    // Return the gradient result if autodiff was enabled
    #[cfg(feature = "autodiff")]
    {
        if let Some(grad) = products.grad {
            Ok(format!(
                "Forward IR:\n{:#?}\n\nGradient Module:\n{:#?}",
                products.ir, grad.gradient_module
            ))
        } else {
            Ok(format!("IR:\n{:#?}", products.ir))
        }
    }

    #[cfg(not(feature = "autodiff"))]
    {
        Ok(format!("IR:\n{:#?}", products.ir))
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(compile_with_autodiff, m)?)?;
    Ok(())
}
