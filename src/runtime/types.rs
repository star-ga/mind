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
}

/// High-level backend target requested by the user or tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendTarget {
    /// Stable, default execution path.
    #[default]
    Cpu,
    /// Experimental GPU target (interface only in this crate).
    Gpu,
}

impl fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendTarget::Cpu => write!(f, "cpu"),
            BackendTarget::Gpu => write!(f, "gpu"),
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
