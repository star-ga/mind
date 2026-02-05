# FFI & Runtime Integration

The open-core repository exposes the surface needed to embed compiled MIND modules from a host application. The actual execution engines live in proprietary backends (for example, the private `mind-runtime` repository), while this crate focuses on stable interfaces and data types.

## Runtime Interface

The `runtime_interface` module defines the traits used by evaluators and FFI shims:

- `MindRuntime` – a backend-agnostic trait for allocating tensors, launching operations, and synchronizing devices.
- `TensorDesc` – a simple descriptor that pairs shape dimensions (`Vec<ShapeDim>`) with a `DType`.
- `DeviceKind` – identifies a broad execution target (CPU, GPU, or other accelerators).

Backends implement `MindRuntime` to provide real allocators and kernel dispatch. The default `NoOpRuntime` included here is a stub suitable for compiler smoke tests.

## Embedding Workflow

1. Build a module with the desired features (e.g., IR lowering or MLIR emission).
2. Link against a `MindRuntime` implementation supplied by your runtime backend.
3. Use the runtime to allocate inputs, run operations, and collect outputs through the backend-specific API.

```rust
use libmind::runtime_interface::{DeviceKind, MindRuntime, NoOpRuntime, TensorDesc};
use libmind::types::ShapeDim;

fn run_demo(runtime: &dyn MindRuntime) {
    let buffer = runtime.allocate(&TensorDesc {
        shape: vec![ShapeDim::Known(2), ShapeDim::Known(3)],
        dtype: libmind::types::DType::F32,
        device: Some(DeviceKind::Cpu),
    });
    runtime.run_op("demo_op", &[buffer], &[]);
    runtime.synchronize();
}
```

## FFI Surface

When the `ffi-c` feature is enabled, the crate exports C-compatible entry points that mirror the open runtime interface. These bindings are intentionally thin: they surface opaque handles and descriptor structs without exposing backend-specific data layouts or device management details.

Backends are responsible for providing the actual implementations and for documenting any additional knobs (such as device selection, stream control, or custom allocators).
