# MIND Architecture (high level)

- `src/parser`: lexer+parser → `ast`
- `src/types`: dtype/shape/value typing
- `src/eval`: front-end evaluation (preview), IR, MLIR export, optional exec backends
- `src/exec`: optional CPU kernels (feature-gated)
- `src/ffi`: optional C FFI (feature-gated)
- `src/package`: optional packaging (feature-gated)
- `src/main.rs`: CLI (gates subcommands by features)

**Design principles**
- `--no-default-features` always builds & tests
- Feature gates: `cpu-exec`, `cpu-conv`, `mlir-*`, `pkg`, `ffi-c`
- Tests prefer integration tests in `tests/`
