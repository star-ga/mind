use crate::types::{DType, Shape};

#[derive(Clone, Copy, Debug)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Other,
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: DType,
}

pub trait MindRuntime {
    fn allocate(&self, desc: &TensorDesc) -> usize;
    fn run_op(&self, op: &str, inputs: &[usize], outputs: &[usize]);
    fn synchronize(&self) {}
}

pub struct NoOpRuntime;

impl MindRuntime for NoOpRuntime {
    fn allocate(&self, _desc: &TensorDesc) -> usize {
        0
    }

    fn run_op(&self, _op: &str, _inputs: &[usize], _outputs: &[usize]) {
        // no-op
    }
}
