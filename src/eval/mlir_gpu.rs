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

use crate::eval::EvalError;

use super::GpuBackend;

#[derive(Clone, Copy, Debug)]
pub struct GpuLaunchCfg {
    pub blocks: (u32, u32, u32),
    pub threads: (u32, u32, u32),
}

pub fn run_mlir_gpu_text(
    _mlir: &str,
    backend: GpuBackend,
    _cfg: GpuLaunchCfg,
) -> Result<(), EvalError> {
    let backend_name = match backend {
        GpuBackend::Cuda => "CUDA",
        GpuBackend::Rocm => "ROCm",
    };
    Err(EvalError::UnsupportedMsg(format!(
        "mlir-gpu runtime for {backend_name} not available; falling back"
    )))
}
