//! Experimental GPU backend contract for MIND.
//!
//! This module defines the abstract interface a GPU backend must implement.
//! No concrete GPU implementation is provided in this crate. The GPU surface is
//! **experimental** and not covered by the Core v1 stability guarantees.

use crate::ir::Instr;
use crate::runtime::types::{RuntimeError, TensorHandle};
use crate::runtime_interface::TensorDesc;

/// Abstract contract for GPU execution backends.
///
/// A concrete implementation is responsible for device memory management and
/// implementing the semantics of the IR operations it chooses to support.
pub trait GPUBackend {
    /// Allocates a device tensor with the given description.
    fn allocate(&self, desc: &TensorDesc) -> Result<TensorHandle, RuntimeError>;

    /// Executes a single IR instruction on device-resident tensor handles.
    fn run_op(
        &self,
        op: &Instr,
        inputs: &[TensorHandle],
        outputs: &[TensorHandle],
    ) -> Result<(), RuntimeError>;

    /// Synchronises device execution, ensuring all prior operations are visible.
    fn synchronize(&self) -> Result<(), RuntimeError>;
}
