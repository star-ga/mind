use crate::runtime::types::TensorHandle;
pub use crate::runtime::types::DeviceKind;
use crate::types::{DType, ShapeDim};

/// Describes a tensor visible to the runtime.
///
/// Shape is represented as a sequence of `ShapeDim` values
/// (static / symbolic dimensions).
#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub shape: Vec<ShapeDim>,
    pub dtype: DType,
    /// Optional execution device for this tensor.
    pub device: Option<DeviceKind>,
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
    fn allocate(&self, desc: &TensorDesc) -> TensorHandle;

    /// Execute an operation identified by `op` over the input and output handles.
    fn run_op(&self, op: &str, inputs: &[TensorHandle], outputs: &[TensorHandle]);

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
