# MIND Examples

Comprehensive examples demonstrating MIND's capabilities for tensor computation, automatic differentiation, and ML workflows.

## Quick Start

| Example | Description | Difficulty |
|---------|-------------|------------|
| `hello_tensor.mind` | Basic tensor operations and device placement | Beginner |
| `autodiff_demo.mind` | Comprehensive autodiff demonstrations | Intermediate |
| `cnn_classifier.mind` | CNN training for image classification | Advanced |
| `tiny_edge_model.mind` | Edge-optimized model (<200 KB) | Advanced |
| `mlir_pipeline_demo.sh` | Full MLIR/LLVM compilation pipeline | Advanced |

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

## Building and Running

```bash
# Interpret mode
mind run examples/hello_tensor.mind

# Compile to IR
mind compile --emit-ir examples/autodiff_demo.mind

# Compile to binary (requires MLIR/LLVM)
mind build --features=mlir-lowering examples/cnn_classifier.mind -o cnn_binary
```

## Additional Resources

- [MIND Language Specification](https://github.com/cputer/mind-spec)
- [Autodiff Specification](../docs/autodiff.md)
- [MLIR Lowering Guide](../docs/mlir-lowering.md)

---

**Last Updated**: 2025-12-18
**Total Examples**: 5
**Coverage**: Basics, Autodiff, CNNs, Edge deployment, MLIR pipeline
