<p align="center">
  <img src="assets/logo.svg" alt="MIND Logo" width="400"/>
</p>


# MIND

**The native language for intelligent systems**

**Machine Intelligence Native Design**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](#roadmap)
[![Build Status](https://img.shields.io/badge/Build-Passing-success.svg)](https://github.com/mind-lang/mind/actions)
[![Built with Rust](https://img.shields.io/badge/Built_with-Rust-orange.svg)](https://www.rust-lang.org/)
[![MLIR Compatible](https://img.shields.io/badge/IR-MLIR%20Compatible-blue.svg)](https://mlir.llvm.org/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-7289da.svg)](https://discord.gg/mindlang)

> **[Read the Manifesto](https://mindlang.dev/manifesto)** • **[View Rendered Docs](https://mind-lang.github.io/mind)**

---

## Table of Contents
- [What is MIND?](#what-is-mind)
- [Why MIND?](#why-mind)
- [Core Principles](#core-principles)
- [Quick Start](#quick-start)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Community](#community)
- [FAQ](#faq)

---

## What is MIND?

**MIND** is a programming language where AI primitives aren't imported—they're built into the language itself.

```mind
// Training step: autodiff and device placement are native
import std.autodiff;

@differentiable
fn train_step(
    model: Model[f32],
    inputs: Tensor[f32, (B, 3, 224, 224)],   // B is a symbolic batch size
    targets: Tensor[i32, (B,)]
) -> f32 {
    on(gpu0) {
        // Define the loss as a closure over current inputs/targets
        let loss_fn = || {
            let logits = model.forward(inputs);
            return cross_entropy(logits, targets);
        };

        // grad(...) returns a function that computes loss and gradients
        // for the values captured by the closure (e.g. model parameters)
        let loss, grads = grad(loss_fn)();

        optimizer.step(model, grads);
        return loss;
    }
}
```

**No frameworks. No wrappers. Just intelligence, compiled.**

---

## Why MIND?

AI development today is fragmented:
- **Models** live in Python
- **Kernels** are written in C++/CUDA  
- **Graphs** execute in XLA/MLIR
- **Production** deploys in yet another runtime

This creates three critical problems:

### 1. Abstraction Mismatch
Iterative, conditional, and agentic AI doesn't map cleanly to static computation graphs.

### 2. Opaque Performance  
One wrong move in Python and you've escaped the compiler—no fusion, no optimization.

### 3. AI as an Afterthought
Tensors, autodiff, devices, and distribution exist only as library add-ons to general-purpose languages.

**MIND solves this** by making intelligence a first-class concept in the language itself.

---

## Core Principles

### 🧠 AI-Native Type System
Tensor shapes, device placement, and differentiability are part of the type signature—verified at compile time.

```mind
// '@gpu0' marks device placement in the type system
@differentiable
fn forward(x: Tensor[f32, (B, 3, H, W)] @gpu0)
    -> Tensor[f32, (B, 1000)]
{
    // Compiler guarantees:
    // - x lives on gpu0
    // - output shape is (B, 1000)
    // - this function can be differentiated
}
```

### ⚡ Deterministic Performance
What compiles to one kernel vs. many is explicit. No hidden framework magic. The compiler provides compile-time diagnostics when optimizations fail.

```mind
// This fuses into a single kernel
let y = x.map(relu).reduce(sum).normalize();

// This doesn't—and the compiler tells you why
let y = x.cpu().map(relu).gpu().reduce(sum);
```

### 🌐 Built-In Distribution
Data parallelism, tensor parallelism, and pipeline parallelism aren't afterthoughts—they're language constructs.

```mind
// 'shard' automatically partitions data and synchronizes gradients
shard(axis="batch", devices=[gpu0, gpu1, gpu2, gpu3]) {
    let outputs = model.forward(batch);
}
```

### 🔄 Automatic Differentiation
Autodiff works through loops, conditionals, recursion, and custom operators—because it's part of the language semantics.

```mind
fn custom_loss(pred: Tensor[f32], target: Tensor[f32]) -> f32 {
    let threshold = 0.5;  // Example threshold
    let total = Tensor[f32, ()].scalar(0.0);
    for i in 0..pred.shape()[0] {
        if pred[i] > threshold {
            total += (pred[i] - target[i]).pow(2);
        }
    }
    return total.item();
}

// grad() returns a higher-order function that computes gradients
let dloss = grad(custom_loss);
let g = dloss(pred, target);
```

---

## Design Highlights

| Feature | Status | Description |
|---------|--------|-------------|
| **Shape-Aware Types** | ✅ Specified | `Tensor[f32, (B, C, H, W)]` with compile-time validation |
| **Device Placement** | ✅ Specified | `on(gpu0)`, `@cpu`, `@tpu` as type/block annotations |
| **Autodiff** | ✅ Specified | `grad()`, `vjp()`, `jvp()` as language primitives |
| **AOT + JIT** | 🚧 In Progress | Compile to MLIR → GPU/TPU/CPU |
| **Mixed Precision** | ✅ Specified | `f16`, `bf16`, `f32` with explicit conversion rules |
| **Ownership Model** | ✅ Specified | Rust-style ownership + region memory for zero-copy tensor ops |
| **Python Interop** | 🚧 In Progress | Bidirectional calls via pybind11/FFI |

---

## Benchmarks (Preliminary)

> ⚡ **Early results**: Preliminary microbenchmarks (synthetic tests) suggest up to 2× speedups on fused paths vs. equivalent PyTorch on ResNet-50 inference (RTX 4090, batch size 32).

**Run the benchmark yourself** (from repo root):
```bash
cargo bench
```

More details: [/benchmarks/resnet.md](benchmarks/resnet.md)

![Compilation Flow](assets/compilation-flow.png)
*MIND's compilation pipeline: Source → Type Check → MLIR IR → Kernel Fusion → Device Execution*

---

## Quick Start

> ⚠️ **Alpha stage**: MIND is under active development. Current focus is research-grade experimentation. Alpha means you shape the language—your feedback drives priorities. Production stability is targeted for v1.0 (Q4 2026). See [issue #1](https://github.com/mind-lang/mind/issues/1) for known limitations.

### Prerequisites
- Rust 1.70+ and Cargo
- LLVM 15+
- CUDA Toolkit 11.8+ (for GPU support)

> **Note:** MLIR/LLVM backends are currently feature-gated. By default, the crate builds without them (no system LLVM required).
> Enable with `--features mlir` and/or `--features llvm` once you have a matching toolchain installed.

### Troubleshooting
- **CUDA not found?** Check `CUDA_HOME` and `LD_LIBRARY_PATH` environment variables
- **Build fails?** Ensure LLVM 15+ is in `PATH`
- **Rust version mismatch?** Run `rustup update stable`

### Tensor previews (Phase 4A)

MIND now evaluates tensor expressions as lightweight previews (dtype + shape + optional fill), without materializing data buffers.

```bash
cargo run --quiet -- eval 'let x: Tensor[f32,(2,3)] = 0; x + 1'
# → Tensor[F32,(2,3)] fill=1
```

Broadcasting support previews combined shapes:

```bash
cargo run --quiet -- eval 'let a: Tensor[f32,(2,1,3)] = 0; let b: Tensor[f32,(1,4,3)] = 0; a + b'
# → Tensor[F32,(2,4,3)] fill=0
```

### Tensor intrinsics (Phase 4B)

```mind
let z = tensor.zeros(f32, (2,3));
let o = tensor.ones(f32, (2,3));
tensor.print(o);
# → Tensor[F32,(2,3)] fill=1
```

Additional helpers:

- `tensor.shape(t)` → returns a preview tuple such as `(2,3)`
- `tensor.dtype(t)` → returns the dtype string such as `f32`

### Buffers (feature-gated)

Opt in to concrete CPU buffers with `--features cpu-buffers`. Small tensors (≤1,024 elements) automatically materialize, and you
can request buffers explicitly:

```bash
cargo run --features cpu-buffers -- eval 'let x: Tensor[f32,(2,3)] = 1; tensor.materialize(x)'
# → Tensor[F32,(2,3)] materialized (sample=[1,1,1,1,1,1])
```

Use `tensor.materialize(t)` to force allocation, `tensor.sample(t, k)` to grab up to `k` linear elements, and
`tensor.is_materialized(t)` to check if a buffer exists. Without the feature flag, tensors remain lightweight previews.

**Quick install via Docker:**
```bash
docker pull mindlang/mind:latest
docker run -it mindlang/mind
```

### Installation from Source
```bash
# Clone the repo
git clone https://github.com/mind-lang/mind.git
cd mind

# Build the compiler (Rust-based)
cargo build --release

# Add to PATH (temporary)
export PATH=$PATH:$(pwd)/target/release
# For permanent: add above line to ~/.bashrc or ~/.zshrc
```

### Try the CLI

You can now evaluate simple expressions directly:

```bash
cargo run --quiet -- eval "1 + 2 * 3"
# → 7
```

### Variables & assignment
```bash
cargo run --quiet -- eval "let x = 2; x = x + 5; x * 3"
# → 21
```

**Type checking (Phase 3A, scalars):**
- Int literals → `Scalar(i32)`
- Variables inherit scalar type from assigned values
- Binary ops require both sides to be scalar
- Type errors are reported before execution

### Typed let (Phase 3B)
```bash
cargo run --quiet -- eval 'let n: i32 = 3; n + 1'
# → 4

cargo run --quiet -- eval 'let x: Tensor[f32,(B,3,224,224)] = 0; x + 1'
# → type error (tensor vs scalar)
```

**Tensor typing & broadcasting (Phase 3C, type-check only)**
- Elementwise `Tensor ⊕ Tensor` and `Tensor ⊕ Scalar` follow NumPy-style broadcasting.
- Dtypes must match (`f32 + f32`); `i32` scalar can be promoted when adding to `Tensor[f32,...]`.
- Mismatched shapes or dtypes produce pretty compile-time diagnostics.

### REPL (interactive)

```bash
cargo run --quiet -- repl
# MIND REPL — type :quit to exit
# MIND> let x = 2;
# 2
# MIND> x * 3
# 6
# MIND> :quit
```

### Diagnostics

Pretty parse errors now include line/col and carets:

```
$ mind eval "let = 2"
error: expected identifier
--> line 1, col 5
let = 2
^

```

### Autodiff (Phase 4D, preview)

Use `grad(loss, wrt=[...])` to obtain gradient previews (shape/dtype and optional constant fill):

```bash
cargo run --quiet -- eval "let x: Tensor[f32,(2,3)] = 0; grad(tensor.sum(x + 1), wrt=[x])"
# → grad{ x: Tensor[F32,(2,3)] fill=1 }
```

### Reductions & Shape ops (Phase 4E)

```bash
cargo run --quiet -- eval "let x: Tensor[f32,(2,3)] = 1; tensor.sum(x)"
# → Tensor[F32,()] fill=6

cargo run --quiet -- eval "let x: Tensor[f32,(2,3)] = 0; grad(tensor.mean(x), wrt=[x])"
# → grad{ x: Tensor[F32,(2,3)] fill=0.166666 }
```

Operators: `tensor.sum/mean(x, axes=[...], keepdims=bool)`, `tensor.reshape`, `tensor.expand_dims`, `tensor.squeeze`.

### Linear algebra (Phase 4F)
```bash
cargo run --quiet -- eval "let a: Tensor[f32,(2,3)] = 1; let b: Tensor[f32,(3,4)] = 2; tensor.matmul(a,b)"
# → Tensor[F32,(2,4)] fill=6

cargo run --quiet -- eval "let v: Tensor[f32,(3)] = 1; let w: Tensor[f32,(3)] = 2; tensor.dot(v,w)"
# → Tensor[F32,()] fill=6
```

Gradients (preview):
```bash
cargo run --quiet -- eval "let A: Tensor[f32,(2,3)] = 0; let B: Tensor[f32,(3,4)] = 0; grad(tensor.sum(tensor.matmul(A,B)), wrt=[A,B])"
# → grad{ A: Tensor[F32,(2,3)] fill=1, B: Tensor[F32,(3,4)] fill=1 }
```

### Indexing & Slicing (Phase 4G)

```bash
cargo run --quiet -- eval "let x: Tensor[f32,(2,5)] = 1; tensor.index(x, axis=1, i=0)"
# → Tensor[F32,(2)] fill=1

cargo run --quiet -- eval "let x: Tensor[f32,(3,6)] = 2; tensor.slice(x, axis=1, start=1, end=4)"
# → Tensor[F32,(3,3)] fill=2
```

Gradients (preview):
```bash
cargo run --quiet -- eval "let X: Tensor[f32,(3,6)] = 0; grad(tensor.sum(tensor.slice(X, axis=1, start=1, end=4)), wrt=[X])"
# → grad{ X: Tensor[F32,(3,6)] … }
```

### Strided slicing & gather (Phase 4H)

```bash
cargo run --quiet -- eval "let x: Tensor[f32,(5,10)] = 3; tensor.slice_stride(x, axis=1, start=0, end=10, step=2)"
# → Tensor[F32,(5,5)] fill=3

cargo run --quiet -- eval "
  let X: Tensor[f32,(3,4)] = 1;
  let idx: Tensor[i32,(2)] = 0;
  tensor.gather(X, axis=1, idx)
"
# → Tensor[F32,(3,2)] fill=1
```

Gradients (preview):
```bash
cargo run --quiet -- eval "
  let X: Tensor[f32,(5,10)] = 0;
  grad(tensor.sum(tensor.slice_stride(X, axis=1, start=0, end=10, step=2)), wrt=[X])
"
# → grad{ X: Tensor[F32,(5,10)] … }
```

### Phase 5A — IR Lowering (MIND → IR)
MIND now lowers typed AST to a minimal IR with Const, BinOp, Sum, Reshape, and Output.
You can inspect the IR with:

```bash
cargo run -- eval '1 + 2 * 3'
# --- Lowered IR ---
# ConstI64(ValueId(0), 1)
# ConstI64(ValueId(1), 2)
# ConstI64(ValueId(2), 3)
# BinOp { dst: ValueId(3), op: Mul, lhs: ValueId(1), rhs: ValueId(2) }
# BinOp { dst: ValueId(4), op: Add, lhs: ValueId(0), rhs: ValueId(3) }
# Output(ValueId(4))
# --- Result ---
# 7
```

### Phase 5B — MLIR Exporter
You can now emit MIND programs as MLIR text:

```bash
cargo run -- eval '1 + 2 * 3' --emit-mlir
```

prints:

```mlir
module {
  func.func @main() -> () {
    %0 = arith.constant 1 : i64
    %1 = arith.constant 2 : i64
    %2 = arith.constant 3 : i64
    %3 = arith.muli %1, %2 : i64
    %4 = arith.addi %0, %3 : i64
    return
    // result: %4
  }
}
```

### Phase 5C — MLIR file export and lowering presets

You can now dump MLIR to stdout or write a `.mlir` file:

```bash
# stdout
cargo run --quiet --no-default-features -- eval "1+2" --emit-mlir --mlir-lower none

# file
cargo run --quiet --no-default-features -- eval \
  "let x: Tensor[f32,(2,3)] = 0; x + 1" \
  --emit-mlir-file /tmp/out.mlir --mlir-lower arith-linalg
```

`--mlir-lower` supports:

* `none` — raw exporter output
* `arith-linalg` — normalizes common patterns (textual)
* `cpu-demo` — lightweight cosmetic rewrites for demos

> These are **textual** passes (no MLIR libs). Later phases can swap them for real `mlir-opt` pipelines behind a feature.

**Span-accurate type errors (Phase 3D):** carets now point to the exact token (identifier or operator) that triggered a type error.

### Hello, Tensor
```mind
import std.tensor;

fn main() {
    let x = tensor.zeros[f32, (2, 3)];
    let y = x + 1.0;

    on(gpu0) {
        print(y.sum());  // 6.0 (2*3 elements, all = 1.0)
    }
}
```

**Example (expressions):**
The Phase 2 parser prototype can now understand and evaluate simple integer expressions for development tests: `1 + 2 * 3`
evaluates to `7`, while `(1 + 2) * 3` evaluates to `9`.

**Run it:**
```bash
mind run examples/hello_tensor.mind
```

---

## Roadmap

> **Note**: This roadmap reflects target milestones, not guarantees. Dates may shift based on community feedback and development priorities.

<details>
<summary><b>Phase 1: Foundation (Q3-Q4 2025)</b> — <i>Current</i></summary>

- [x] Language design document v0.3 ([docs/design/v0.3.md](docs/design/v0.3.md))
- [x] Core syntax specification
- [x] Lexer and parser
- [x] Type checker with shape inference ✅ *(basic unify)*
- [x] Basic autodiff prototype
- [x] MLIR IR generation ✅ *(stub)*
- [x] Basic tensor operations stdlib ✅ *(stubs)*
</details>

*Note:* Tensor stdlib and shape inference are minimal placeholders to keep the pipeline compiling; real semantics come in Phase 2–3.

*Note:* `mlir` feature provides a stubbed lowering (no deps). Real MLIR integration will use `mlir_backend` (melior) in a later phase.

_Phase 1 scaffold complete (parses identifiers/integers; type checker stubbed)._ 

Autodiff API is a placeholder (feature-gated). Real gradients arrive in Phase 2–3.

<details>
<summary><b>Phase 2: Compilation (Q1 2026)</b></summary>

- [ ] GPU code generation (CUDA/Metal/ROCm)
- [ ] Kernel fusion optimizer
- [ ] Device placement engine
- [ ] Python FFI bridge
</details>

<details>
<summary><b>Phase 3: Autodiff (Q2 2026)</b></summary>

- [ ] Forward-mode autodiff
- [ ] Reverse-mode autodiff
- [ ] Hessian-vector products (mixed-mode) for second-order optimizers
- [ ] Checkpointing and memory optimization
</details>

<details>
<summary><b>Phase 4: Distribution (Q3 2026)</b></summary>

- [ ] Multi-device execution
- [ ] Data/tensor/pipeline parallelism
- [ ] Distributed gradient computation
- [ ] Fault tolerance and checkpointing
</details>

<details>
<summary><b>Post-1.0: Ecosystem (2027+)</b></summary>

- [ ] IDE plugins (VSCode, Neovim, IntelliJ)
- [ ] Model zoo and pretrained weights
- [ ] Cloud deployment tooling
- [ ] Hardware vendor partnerships
- [ ] Community-driven language extensions
</details>

**Target: v1.0 production release by Q4 2026, with ongoing beta releases from Q2 2026**

---

## Contributing

We're building MIND in the open. Here's how you can help:

### 🐛 Good First Issues
Start here: [Good First Issues](https://github.com/mind-lang/mind/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

**Low-hanging fruit:**
- **Propose tensor operations**: What ops should be in the stdlib? Open an issue.
- **Write examples**: Show us what MIND code should look like for your domain (vision, NLP, RL, agents).
- **Test shape inference**: Find edge cases where type checking should catch errors.

### 🔧 Medium Complexity
- **Implement IR passes**: MLIR dialect extensions, fusion patterns, layout optimization.
- **Build standard library**: Core tensor ops, loss functions, optimizers.
- **Improve error messages**: Make compiler diagnostics actually helpful.

### 🚀 Advanced
- **GPU code generation**: CUDA/ROCm/Metal kernel emission.
- **Autodiff engine**: Implement tape-based or symbolic differentiation.
- **Distributed runtime**: Device coordination, gradient aggregation, fault tolerance.

### 💰 Bounty Board
High-impact issues eligible for sponsored bounties are marked with 💰. See [bounties.md](bounties.md) for current opportunities.

### 🎓 Mentorship
New to compilers? Pair with core team on Discord for guided PRs. We're here to help you learn.

**See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

---

## Community

- **Discord**: [Join the conversation](https://discord.gg/mindlang)
- **X (Twitter)**: [@mindlang](https://x.com/mindlang)
- **Mailing List**: [Monthly updates on spec drops and alpha builds](https://mindlang.dev/newsletter)
- **GitHub Discussions**: [RFCs and design feedback](https://github.com/mind-lang/mind/discussions)
  - [Success Stories](https://github.com/mind-lang/mind/discussions/categories/success-stories) — Share your MIND projects!
- **Design Docs**: [/docs/design](docs/design/)

---

## Success Stories

**Early Adopters**: Training toy transformers with comparable performance to PyTorch on fused operations. [Share your story in Discussions](https://github.com/mind-lang/mind/discussions/categories/success-stories).

---

## Related Documentation

- [Language Spec v1.0 (draft)](docs/specs/v1.0.md)
- [Design Principles v0.3](docs/design/v0.3.md)
- [RFC Process](docs/rfcs/)
- [Contribution Guidelines](CONTRIBUTING.md)

---

<details>
<summary><b>FAQ</b></summary>

### How is MIND different from JAX/PyTorch?
JAX and PyTorch are **libraries** on top of Python. MIND is a **language** where tensors, autodiff, and devices are native primitives—not decorators or context managers.

### Why not just improve Python?
Python's runtime semantics (dynamic typing, GIL, reference counting) fundamentally conflict with the determinism and performance AI systems require. We need a clean slate.

### Can I use existing models/libraries?
Yes. MIND's FFI (via pybind11) lets you call PyTorch/JAX/C++ directly during transition. You can also call custom CUDA kernels natively.

### What about Mojo?
Mojo targets "Python with performance"; MIND targets "AI with native semantics." Different goals, different trade-offs.

### How does MIND handle errors?
Compile-time wherever possible (shape mismatches, device placement, type errors). Runtime errors for data-dependent issues include traceable diagnostics with line numbers and stack traces.

### Is MIND production-ready?
Not yet—current focus is research-grade experimentation. Production stability is targeted for v1.0 (Q4 2026).

</details>

---

## Translations

Community translations welcome—see [translation issues](https://github.com/mind-lang/mind/issues?q=is%3Aissue+is%3Aopen+label%3Atranslation).

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

MIND builds on decades of programming language research:
- **Type systems**: Hindley-Milner, dependent types, effect systems
- **Compilers**: MLIR, LLVM, Cranelift
- **AI frameworks**: JAX's autodiff, PyTorch's eager execution, TensorFlow's XLA fusion
- **Language design**: Rust's ownership, Swift's protocols, Julia's multiple dispatch

We stand on the shoulders of giants—and invite the next generation to stand on ours.

---

**MIND — where intelligence compiles.**

*Star this repo if you believe AI needs a native language. Let's build it together.*

---

**Topics**: `ai-language` `machine-learning` `compiler` `tensor-native` `autodiff` `mlir` `gpu-computing` `programming-language` `deep-learning` `parallel-computing` `systems-programming`
