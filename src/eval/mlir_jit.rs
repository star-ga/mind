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

use std::path::PathBuf;

use libloading::Library;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JitError {
    #[error("MLIR libraries not found")]
    NotFound,
    #[error("MLIR invocation failed: {0}")]
    Invoke(String),
    #[error("Unsupported shape/dtype for JIT")]
    Unsupported,
}

pub struct MlirJit {
    #[allow(dead_code)]
    lib: Library,
}

impl MlirJit {
    pub fn new() -> Result<Self, JitError> {
        let path = find_any_mlir_c_api_path().ok_or(JitError::NotFound)?;
        match unsafe { Library::new(&path) } {
            Ok(lib) => Ok(Self { lib }),
            Err(_) => Err(JitError::NotFound),
        }
    }

    pub fn run_mlir_text(
        &self,
        mlir: &str,
        entry: &str,
        args: &[MemRefArg],
    ) -> Result<(), JitError> {
        let _ = mlir;
        let _ = entry;
        let _ = args;
        Err(JitError::Unsupported)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemRefArg {
    pub data: *mut std::ffi::c_void,
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
    pub dtype: String,
}

fn find_any_mlir_c_api_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("MLIR_C_API_LIB") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    None
}
