# MIND Model Zoo

Five certified model examples demonstrating MIND's type-safe tensor programming, reverse-mode automatic differentiation, and end-to-end training workflows.

## Models

| Model | File | Architecture | Parameters | Difficulty |
|-------|------|-------------|------------|------------|
| Linear Regression | `linear_regression.mind` | Linear (features -> 1) | features + 1 | Beginner |
| Logistic Classifier | `logistic_classifier.mind` | Linear + Sigmoid (features -> 1) | features + 1 | Beginner |
| MLP MNIST | `mlp_mnist.mind` | 784 -> 256 -> 128 -> 10 | 235,146 | Intermediate |
| CNN Classifier | `conv_classifier.mind` | Conv-Pool-Conv-Pool-FC | 12,810 | Advanced |
| Transformer Block | `transformer_block.mind` | Self-Attention + FFN | 45,632 (d=64) | Advanced |

## What Each Model Demonstrates

### 1. linear_regression.mind

Simple linear regression with MSE loss. Demonstrates the core MIND workflow: differentiable tensors (`diff tensor<f32[...]>`), forward pass, scalar loss computation, `backward()` for gradient extraction, and SGD parameter updates. Includes an R-squared evaluation metric.

### 2. logistic_classifier.mind

Binary classification with sigmoid activation and binary cross-entropy loss. Shows numerically stable log-probability computation with epsilon clamping, boolean thresholding for class prediction, and accuracy computation.

### 3. mlp_mnist.mind

3-layer MLP (784 -> 256 -> 128 -> 10) for MNIST digit classification. Demonstrates deep network composition with ReLU activations, cross-entropy loss via `log_softmax` + `gather`, and He weight initialization. Each layer is a separate function for modularity.

### 4. conv_classifier.mind

CNN with two convolutional blocks (Conv2d -> ReLU -> MaxPool) followed by a fully connected classification head. Demonstrates 4D tensor operations (`[batch, channels, height, width]`), spatial dimension tracking through convolutions and pooling, and the flatten-to-FC transition.

### 5. transformer_block.mind

Single-head self-attention with feed-forward network and layer normalization. Demonstrates scaled dot-product attention (`Q @ K^T / sqrt(d_k)`), pre-norm residual connections, batched 3D tensor operations, and sequence classification via the [CLS] token pattern.

## Common Patterns

All models follow the same structure:

1. **Forward pass** -- Type-annotated function composing tensor operations
2. **Loss function** -- Scalar loss with full shape annotations
3. **`train_step`** -- Forward -> `backward()` -> SGD update, returns new parameters
4. **`predict`** -- Inference without gradient tracking, returns class labels or values
5. **`main`** -- Example usage with initialization, training loop, and evaluation

## Building and Running

```bash
# Interpret a model
mind run examples/zoo/linear_regression.mind

# Compile to IR (inspect generated code)
mind compile --emit-ir examples/zoo/mlp_mnist.mind

# Compile to binary
mindc examples/zoo/conv_classifier.mind -o conv_classifier

# Compile with deterministic mode for reproducibility verification
mindc --deterministic examples/zoo/transformer_block.mind -o transformer

# Verify SHA-256 hash of compiled output
mindc --emit-hash examples/zoo/mlp_mnist.mind
```

## Deterministic Verification

All models support SHA-256 deterministic verification when compiled with the `--deterministic` flag. This ensures bit-identical outputs for identical inputs across runs, enabling reproducible training and auditable model artifacts.

```bash
# Generate hash for compiled model
mindc --emit-hash --deterministic examples/zoo/mlp_mnist.mind
# Output: sha256:a1b2c3d4...

# Verify against known hash
mindc --verify-hash sha256:a1b2c3d4... examples/zoo/mlp_mnist.mind
```

## MIND Language Features Used

| Feature | Syntax | Used In |
|---------|--------|---------|
| Differentiable tensors | `diff tensor<f32[batch, features]>` | All models |
| Reverse-mode autodiff | `backward(loss, param)` | All models |
| Dynamic batch dimension | `batch` in tensor shapes | All models |
| ReLU activation | `relu(x)` | MLP, CNN, Transformer |
| Sigmoid activation | `sigmoid(x)` | Logistic Classifier |
| Softmax / log_softmax | `softmax(x, axis=1)` | MLP, CNN, Transformer |
| 2D convolution | `conv2d(x, w, stride, padding)` | CNN |
| Max pooling | `maxpool2d(x, kernel, stride)` | CNN |
| Matrix multiply | `matmul(a, b)` | All models |
| Reshape | `reshape(x, shape)` | MLP, CNN |
| Gather | `gather(x, indices, axis)` | MLP, CNN, Transformer |
| Argmax | `argmax(x, axis)` | MLP, CNN, Transformer |
| Transpose | `transpose(x, perm)` | Transformer |
| Reduction ops | `sum(x)`, `mean(x)` | All models |
| Broadcasting | `tensor + bias` | All models |

---

**Models**: 5 certified
**Last Updated**: 2026-03-06
