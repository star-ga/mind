//! Runtime abstractions for execution backends.
//!
//! This module hosts shared runtime surface types and experimental backend
//! contracts. CPU remains the only implemented backend in this crate; GPU
//! interfaces are provided solely for forward compatibility.

pub mod gpu;
pub mod types;

pub use types::{BackendTarget, DeviceKind, RuntimeError, TensorHandle};
