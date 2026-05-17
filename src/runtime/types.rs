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
    /// Wafer-scale accelerator (Cerebras CS-2/CS-3, Tesla Dojo D1/D2). Distinct
    /// from GPU because the fabric is 2-D mesh of cores with weight residency,
    /// not a discrete-die SIMT model — the compiler reasons about region
    /// placement rather than streaming-multiprocessor occupancy.
    Wafer,
}

/// High-level backend target requested by the user or tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendTarget {
    /// Stable, default execution path.
    #[default]
    Cpu,
    /// GPU target — CUDA, ROCm, Metal, WebGPU backends (CUDA is production-ready).
    Gpu,
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
    /// Cerebras Wafer-Scale Engine (CS-2 with WSE-2 / CS-3 with WSE-3).
    ///
    /// Distinct from `Gpu` because the runtime backend (in
    /// `mind-runtime/src/backend/cerebras/`) emits Cerebras Software
    /// Language (CSL) and reasons about a 2-D fabric mesh of 850k–900k
    /// cores with fabric-resident weights, rather than CUDA-style
    /// streaming-multiprocessor blocks. The wafer generation is selected
    /// at runtime through the `mind-runtime` capability descriptor, not
    /// at the source-level target.
    ///
    /// MIND's three structural fits for the WSE:
    ///
    /// 1. **Local-stencil hot path.** Programs whose dominant kernel is
    ///    a 5-point Laplacian (or any short-range 2-D stencil) map onto
    ///    the fabric without the O(N²) fragmentation cost that pure
    ///    attention models pay when sharded across cores.
    /// 2. **Q16.16 fixed-point end-to-end.** mindc emits canonical IR
    ///    in deterministic Q16.16; Cerebras Weight Streaming is
    ///    precision-agnostic, so the same Q16.16 weights run on WSE-2
    ///    and WSE-3 without per-format kernel rewrites.
    /// 3. **Cross-substrate hash identity.** Q16.16 weights re-serialized
    ///    through the canonical wire format produce a byte-identical
    ///    32-byte hash across x86, CUDA, and the wafer — a compliance
    ///    primitive for regulated-AI deployments where "same model on
    ///    different substrate" must be cryptographically verifiable.
    ///
    /// Selecting this target tells mindc to lower tensor ops into
    /// `mind.cerebras.*` MLIR ops that downstream tools can specialize
    /// for the wafer fabric.
    Cerebras,
}

impl fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendTarget::Cpu => write!(f, "cpu"),
            BackendTarget::Gpu => write!(f, "gpu"),
            BackendTarget::Tpu => write!(f, "tpu"),
            BackendTarget::Npu => write!(f, "npu"),
            BackendTarget::Lpu => write!(f, "lpu"),
            BackendTarget::Dpu => write!(f, "dpu"),
            BackendTarget::Fpga => write!(f, "fpga"),
            BackendTarget::Cerebras => write!(f, "cerebras"),
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
