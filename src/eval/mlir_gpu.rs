// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
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
