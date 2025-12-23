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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::pipeline::{compile_source, CompileOptions};
use std::collections::HashMap;

/// Compile a MIND program and return compilation output
#[pyfunction]
fn compile(source: &str) -> PyResult<String> {
    let options = CompileOptions {
        func: None,
        enable_autodiff: false,
        target: crate::runtime::types::BackendTarget::Cpu,
    };

    match compile_source(source, &options) {
        Ok(products) => {
            Ok(format!("{:#?}", products.ir))
        }
        Err(e) => {
            Err(PyValueError::new_err(format!("Compilation failed: {:?}", e)))
        }
    }
}

/// Compile a MIND program with autodiff enabled
#[pyfunction]
fn compile_with_autodiff(source: &str) -> PyResult<String> {
    // Compile with autodiff enabled
    let options = CompileOptions {
        func: Some("main".to_string()),
        enable_autodiff: true,
        target: crate::runtime::types::BackendTarget::Cpu,
    };

    let products = compile_source(source, &options)
        .map_err(|e| PyValueError::new_err(format!("Compilation failed: {:?}", e)))?;

    // Return the gradient result if autodiff was enabled
    #[cfg(feature = "autodiff")]
    {
        if let Some(grad) = products.grad {
            Ok(format!("Forward IR:\n{:#?}\n\nGradient Module:\n{:#?}",
                      products.ir, grad.gradient_module))
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
