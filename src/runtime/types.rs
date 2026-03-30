//! Shared runtime surface types for execution backends.
//!
//! GPU-related variants are **experimental**. CPU remains the only stable and
//! implemented backend in this crate.

use std::fmt;

/// Logical device on which computations are executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Tpu,
    Npu,
    Lpu,
    Dpu,
    Fpga,
}

/// High-level backend target requested by the user or tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendTarget {
    /// Stable, default execution path.
    #[default]
    Cpu,
    /// GPU target — CUDA, ROCm, Metal, WebGPU backends (CUDA is production-ready).
    Gpu,
    /// ASIC target for XRM-SSD hardware (direct SSA IR execution).
    Asic,
    /// TPU target — Google Tensor Processing Units (systolic array, XLA HLO lowering).
    /// Planned: Q3–Q4 2026.
    Tpu,
    /// NPU target — on-device Neural Processing Units (Apple ANE, Qualcomm Hexagon, Intel NPU).
    /// Planned: Q3–Q4 2026.
    Npu,
    /// LPU target — SRAM-resident Language Processing Units (Groq-style deterministic execution).
    /// Planned: Q3–Q4 2026.
    Lpu,
    /// DPU target — SmartNIC Data Processing Units (NVIDIA BlueField, AMD Pensando, Intel IPU).
    /// Planned: Q3–Q4 2026.
    Dpu,
    /// FPGA target — Field-Programmable Gate Arrays (HLS4ML-style IR → RTL synthesis).
    /// Planned: Q3–Q4 2026.
    Fpga,
}

impl fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendTarget::Cpu => write!(f, "cpu"),
            BackendTarget::Gpu => write!(f, "gpu"),
            BackendTarget::Asic => write!(f, "asic"),
            BackendTarget::Tpu => write!(f, "tpu"),
            BackendTarget::Npu => write!(f, "npu"),
            BackendTarget::Lpu => write!(f, "lpu"),
            BackendTarget::Dpu => write!(f, "dpu"),
            BackendTarget::Fpga => write!(f, "fpga"),
        }
    }
}

/// Opaque handle to runtime-managed tensor storage.
pub type TensorHandle = usize;

/// Structured runtime error for backend implementations.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Backend functionality requested but not available.
    #[error("backend unavailable: {target}")]
    BackendUnavailable { target: BackendTarget },
    /// Generic backend failure message.
    #[error("backend error: {message}")]
    Message { message: String },
}
