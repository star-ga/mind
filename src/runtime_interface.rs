use crate::types::{DType, ShapeDim};

/// Logical device on which computations are executed.
///
/// This is intentionally minimal and may be extended in the future
/// with concrete device identifiers or backend-specific metadata.
#[derive(Clone, Copy, Debug)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Other,
}

/// Describes a tensor visible to the runtime.
///
/// Shape is represented as a sequence of `ShapeDim` values
/// (static / symbolic dimensions).
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub shape: Vec<ShapeDim>,
    pub dtype: DType,
    /// Optional execution device for this tensor.
    pub device: DeviceKind,
}

/// Core interface for the MIND runtime.
///
/// Implementations bridge the compiler IR and concrete backends
/// (CPU, GPU, accelerators). The open-core repo only defines this
/// interface; production backends live in the private `mind-runtime`
/// repository.
///
/// Error handling: for now, implementations may panic or log on
/// irrecoverable failures. The interface may be extended to return
/// `Result` in a future revision once error taxonomy is stabilized.
pub trait MindRuntime {
    /// Allocate storage for a tensor with the given descriptor.
    ///
    /// Returns an opaque handle that can later be passed to `run_op`.
    fn allocate(&self, desc: &TensorDesc) -> usize;

    /// Execute an operation identified by `op` over the input and output handles.
    fn run_op(&self, op: &str, inputs: &[usize], outputs: &[usize]);

    /// Ensure that all enqueued operations are visible to the host.
    fn synchronize(&self) {}
}

/// Default no-op runtime used in the open-core build.
///
/// This implementation does not perform any real computation and is
/// only intended for compilation testing and non-numerical flows.
pub struct NoOpRuntime;

impl MindRuntime for NoOpRuntime {
    fn allocate(&self, _desc: &TensorDesc) -> usize {
        0
    }

    fn run_op(&self, _op: &str, _inputs: &[usize], _outputs: &[usize]) {
        // no-op
    }
}
