# MIND Examples

Comprehensive examples demonstrating MIND's capabilities for tensor computation, automatic differentiation, GPU parallelism, scientific computing, and ML workflows.

## Quick Start

| Example | Description | Difficulty |
|---------|-------------|------------|
| `hello_tensor.mind` | Basic tensor operations and device placement | Beginner |
| `autodiff_demo.mind` | Comprehensive autodiff demonstrations | Intermediate |
| `cnn_classifier.mind` | CNN training for image classification | Advanced |
| `tiny_edge_model.mind` | Edge-optimized model (<200 KB) | Advanced |
| `mlir_pipeline_demo.sh` | Full MLIR/LLVM compilation pipeline | Advanced |
| `remizov_solver.mind` | Universal ODE solver (Remizov 2025) with Richardson extrapolation | Advanced |
| `remizov_gpu.mind` | GPU-parallel ODE solver and batch lambda sweep | Advanced |
| `remizov_inverse.mind` | Inverse ODE solver — coefficient recovery via autodiff | Advanced |
| `remizov_verify.mind` | ODE solver verification suite (5 tests) | Intermediate |
| `remizov_feynman.mind` | Monte Carlo Feynman path integral ODE solver | Advanced |
| `remizov_benchmark.mind` | 6-benchmark ODE solver performance suite | Advanced |
| `policy.mind` | Execution boundary kernel — fail-closed access control | Intermediate |

## Example Descriptions

### 1. hello_tensor.mind
**Topics**: Basic tensor ops, device placement
**Lines**: ~20

Minimal introduction to MIND syntax.

---

### 2. autodiff_demo.mind
**Topics**: Reverse-mode autodiff, gradients, composition
**Lines**: ~180

Comprehensive tour of automatic differentiation covering scalar gradients, tensor gradients, broadcasting, reductions, nonlinear activations, and composite layers.

```bash
mind run examples/autodiff_demo.mind
```

---

### 3. cnn_classifier.mind
**Topics**: CNNs, training, SGD, cross-entropy
**Lines**: ~120

Complete CNN training pipeline for MNIST-style classification with Conv2d layers, pooling, and cross-entropy loss.

---

### 4. tiny_edge_model.mind
**Topics**: Edge deployment, quantization, memory optimization
**Lines**: ~150

Ultra-compact binary classifier optimized for embedded devices with <200 KB total footprint including runtime.

---

### 5. mlir_pipeline_demo.sh
**Topics**: MLIR lowering, LLVM codegen, optimization passes
**Lines**: ~180

Shell script demonstrating the complete MIND → MLIR → LLVM → Binary compilation pipeline.

```bash
chmod +x examples/mlir_pipeline_demo.sh
./examples/mlir_pipeline_demo.sh
```

---

## Remizov Universal ODE Solver Suite

The following examples implement the Remizov (2025) universal formula for solving second-order
linear ODEs with variable coefficients — the first constructive solution formula for this
problem class. See the [std::ode specification](https://github.com/star-ga/mind-spec/blob/main/std/ode.md)
and [documentation](https://mindlang.dev/docs/remizov-ode-solver) for full details.

### 6. remizov_solver.mind
**Topics**: ODE solving, Chernoff approximation, Gauss-Laguerre quadrature, Richardson extrapolation
**Lines**: ~330

Core Theorem 6 (translation-based) solver. Implements the shift operator, Gauss-Laguerre quadrature
for the Laplace integral, and Richardson extrapolation that boosts convergence from O(1/n) to O(1/n²).
Includes three worked examples: constant-coefficient, variable-coefficient (Airy-type), and convergence
rate verification.

```bash
mindc examples/remizov_solver.mind --verify-only
```

---

### 7. remizov_gpu.mind
**Topics**: GPU parallelism, batch solving, spectral parameter sweeps
**Lines**: ~280

GPU-parallel solver using `on(gpu0) { parallel for }`. Each grid point is independent
(embarrassingly parallel), yielding up to 25x speedup on large grids. Also includes batch
lambda sweep for spectral parameter studies.

---

### 8. remizov_inverse.mind
**Topics**: Inverse problems, autodiff, Scientific ML, gradient descent
**Lines**: ~280

Differentiable inverse solver: given observed solution f(x) and known source g(x), recovers
the ODE coefficients a(x), b(x), c(x) by differentiating through the entire Remizov solver
via `backward()`. Uses exp(log_a) parameterization to enforce positivity. Demonstrates MIND's
`diff tensor` type for Scientific Machine Learning without neural networks.

---

### 9. remizov_verify.mind
**Topics**: Verification, error analysis, convergence testing
**Lines**: ~300

Five-test verification suite:
1. Constant-coefficient test (vs analytical Green's function)
2. Variable drift self-consistency and symmetry
3. Reaction term boundedness and positivity
4. O(1/n) convergence rate validation
5. Lambda sensitivity analysis

---

### 10. remizov_feynman.mind
**Topics**: Monte Carlo, Feynman path integrals, stochastic methods
**Lines**: ~340

Theorem 5 (Feynman formula) solver using Monte Carlo sampling of Gaussian random paths.
Implements xoshiro256** PRNG with Box-Muller normal sampling. Convergence is O(1/√N_samples),
independent of path length. Includes both CPU and GPU-parallel versions.

---

### 11. remizov_benchmark.mind
**Topics**: Benchmarking, performance comparison, finite differences
**Lines**: ~430

Comprehensive 6-benchmark performance suite:
1. **Richardson extrapolation speedup** — plain Chernoff vs Richardson at various n_iter
2. **Accuracy vs compute budget** — total operations vs L∞ error
3. **Grid scaling** — CPU vs GPU wall-clock at increasing n_grid
4. **Quadrature node efficiency** — accuracy vs number of Gauss-Laguerre nodes
5. **Remizov vs finite differences** — comparison with Thomas algorithm baseline
6. **Lambda sensitivity** — solution accuracy across spectral parameter range

Includes a complete finite-difference tridiagonal solver as a comparison baseline.

---

### 12. policy.mind
**Topics**: Enums, structs, byte slicing, deterministic access control, fail-closed design
**Lines**: ~200
**Status**: Targets upcoming systems programming features (`enum`, `struct`, `if/else`, `while`, `const`, `&[u8]`). Not compilable with mindc 0.2.x which currently supports tensor operations only. Included as a design reference for the planned systems programming extension.

Execution boundary kernel for AI agent governance. Demonstrates MIND's systems programming capabilities beyond tensor computation: enum-based action/resource/environment typing, byte-level string matching (case-insensitive without allocations), packed confirmation codes with bit shifting, and exhaustive match-style control flow. The kernel enforces fail-closed access control with prompt injection detection, sensitive path blocking, human confirmation requirements for high-risk actions, and default-deny semantics. Three gate entry points (`evaluate_fleet`, `evaluate_memory`, `evaluate_git`) route through a single `evaluate` function.

**Language features used** (pending compiler support):
- `enum` with explicit discriminants — ADTs for action, resource, environment, effect types
- `struct` with typed fields — zero-copy request/effect structs
- `if/else` branching — exhaustive match-style control flow
- `while` loops — byte-level iteration without allocation
- `const` — compile-time constants
- `&[u8]` byte slices — raw byte access for path and justification validation
- Bitwise operations — packed confirmation codes with shift encoding

---

## Building and Running

```bash
# Interpret mode
mind run examples/hello_tensor.mind

# Compile to IR
mind compile --emit-ir examples/autodiff_demo.mind

# Verify ODE solver examples
mindc examples/remizov_solver.mind --verify-only

# Compile to binary (requires MLIR/LLVM)
mind build --features=mlir-lowering examples/cnn_classifier.mind -o cnn_binary
```

## Additional Resources

- [MIND Language Specification](https://github.com/star-ga/mind-spec)
- [Autodiff Specification](../docs/autodiff.md)
- [MLIR Lowering Guide](../docs/mlir-lowering.md)
- [ODE Solver Specification (std::ode)](https://github.com/star-ga/mind-spec/blob/main/std/ode.md)
- [ODE Solver Documentation](https://mindlang.dev/docs/remizov-ode-solver)

---

**Last Updated**: 2026-02-22
**Total Examples**: 12
**Coverage**: Basics, Autodiff, CNNs, Edge deployment, MLIR pipeline, ODE solving, GPU parallelism, Scientific ML, Monte Carlo methods, Benchmarking, Systems/Policy
