# FFI & Runtime Integration

The MIND runtime exposes safe and ergonomic hooks for embedding compiled programs into host applications.

## Runtime Layers

1. **Tensor Buffers** – `mind-runtime` allocates and manages device/host tensors with reference counting and view semantics.
2. **Executors** – CPU, interpreter, and MLIR-backed JITs implement a shared `Runtime` trait for launching compiled modules.
3. **ABI Bindings** – The `ffi-c` feature exports C-compatible functions, structs, and error codes.

## Embedding Workflow

1. Compile a module with the desired features (`mind build --features cpu-exec`).
2. Load the emitted artifact via the runtime API (`mind_runtime::Module::load`).
3. Prepare inputs using `TensorHandle` builders.
4. Invoke entry points; results materialize as borrowed tensor views or owned buffers.

```rust
let module = mind_runtime::Module::load("model.mindpkg")?;
let mut session = module.session()?;
let output = session.call("inference", &[input_tensor])?;
```

## Memory Management

- Tensor handles use interior mutability and RAII to manage device buffers.
- Zero-copy views allow slicing without extra allocations.
- Custom allocators can be registered for embedded targets.

## Error Handling

FFI calls return `mind_status` codes paired with diagnostic strings. Rust bindings surface these as `Result<T, MindError>` with context captured from the compiler.

## Extending the Runtime

- **New Devices** – Implement `Device` and register it with the dispatcher.
- **Custom Ops** – Define host-callable kernels and expose them through the op registry.
- **Telemetry** – Feature-gated tracing integrates with `tracing` subscribers for observability.

For benchmark integration and performance tracking see [`benchmarks.md`](benchmarks.md).
