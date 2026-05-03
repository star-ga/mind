# Roadmap

This roadmap outlines upcoming milestones for the MIND language, runtime, and tooling.

## Completed

- ✅ **v1.0 Stabilization** – Core syntax and IR semantics locked, conformance tests passing.
- ✅ **GPU Backends** – CUDA (16K LOC), Metal (2.9K), ROCm (3.8K), WebGPU (5.1K) all complete.
- ✅ **Autodiff Engine** – Reverse-mode AD for all Core v1 ops.
- ✅ **FFI Stabilization** – ABI frozen, C header generation.
- ✅ **Distributed Runtime** – NCCL/Gloo collectives, RingAllReduce, pipeline parallelism (3.3K LOC).
- ✅ **Deployment & Serving** – HTTP/gRPC inference, dynamic batching, Prometheus metrics (4.7K LOC).
- ✅ **Package Manager** – PubGrub resolver, SLSA provenance, SBOM, sparse registry (9.5K LOC).
- ✅ **Docs Migration** – Documentation published to `mindlang.dev/docs/*`.
- ✅ **Link-Check Automation** – CI enforces documentation link health.

## In Progress

- 🚧 **Phase 10.5: Governance Logic** – enum, struct, if/else, while, const, bitwise ops for policy kernels.
- 🚧 **Improved Diagnostics** – Structured error messages with fix-it hints.

## Planned

- **Formal Verification** – Proof-carrying IR passes for safety-critical deployments.
- **Ecosystem Integrations** – Official bindings for Python, Swift, and WebAssembly targets.
- **Package Registry** – Public `mindpkg` registry with curated models.

## Contributing

Roadmap items are tracked as GitHub issues with the `roadmap` label. Proposals should follow the RFC template in [`docs/rfcs`](rfcs/).

---

# Extended 2025–2026 Roadmap

The following phases extend the existing roadmap and define the next development milestones for MIND as it moves toward full v1.0 readiness, broader ecosystem adoption, and enterprise deployment.

---

## Phase 10 — Developer SDK & Examples

### Goals
Establish a developer-friendly ecosystem that showcases real-world usage of MIND across training, inference, MLIR workflows, and edge targets.

### Deliverables
- Add `examples/` directory with:
  - CNN classifier (training + inference)
  - Tiny edge model (optimized for <200 KB footprint)
  - Autodiff demonstration (reverse-mode)
  - MLIR/LLVM pipeline example (`mind → mlir → opt → ll → bin`)
- Improve CLI ergonomics and error messaging
- Provide initial Python bindings (`mindlang`)
- Add minimal model zoo with reproducible builds
- Publish example-focused documentation

---

## Phase 10.5 — Systems Programming Foundation

### Goals
Extend the MIND compiler to support governance logic alongside tensor computation. These features enable execution boundary kernels, access control policies, and deterministic decision logic — all compiled with the same verification guarantees as tensor programs.

### Motivation
The `policy.mind` execution boundary kernel (see `examples/policy.mind`) demonstrates the use case: fail-closed access control with enum-based typing, struct-based requests, and bitwise-packed confirmation codes. MIND becomes a *verifiable behavior language* — expressing certified decisions alongside tensor computation.

### Design Decision
**Tiers 1–2 ship. Tier 3 deferred to Rust FFI.** MIND owns governance logic (typed decisions, policy rules). Rust owns byte-level implementation (string matching, memory) via FFI. This preserves MIND's identity as a certified tensor + governance language without scope-creeping into general systems programming. (5-model LLM consensus: unanimous Option A.)

### Deliverables

**Tier 1 — Control Flow & Constants (2–3 days):**
- `if` / `else` expressions (already in EBNF grammar, not yet in parser)
- `while` loops (already in EBNF grammar, not yet in parser)
- `const` declarations (compile-time constants)
- Comparison operators in non-tensor context (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- Bitwise operators (`|`, `&`, `<<`, `>>`, `^`)
- Boolean operators (`||`, `&&`, `!`)

**Tier 2 — Algebraic Data Types (3–5 days):**
- `struct` with typed fields (already in EBNF grammar, not yet in parser)
- `enum` with unit variants and explicit discriminants
- Struct literal construction (`Effect { tag: ..., code: ... }`)
- Field access (`req.target.path`)
- Enum-to-integer casting (`code as u32`)
- `u32` integer type (for packed codes and bitfields)

**Deferred — Byte-Level Operations (use Rust FFI):**
- `&[u8]` byte slices, `b"..."` literals, `u8` type, `mut` bindings
- Byte-level string matching (`starts_with`, `contains_ascii_ci`) implemented as Rust FFI helpers
- Rationale: low-level memory types are redundant with Rust and would blur MIND's certified-AI niche

### Benchmark Impact
All new features are gated behind new keyword/token checks that fail immediately for existing tensor programs. Estimated regression on existing Criterion benchmarks: **<1%**. New parser arms compile to jump tables (O(1) dispatch).

### Spec Status
- `if`, `while`, `struct`, `match`, `for` are already defined in the EBNF grammar (`grammar-syntax.ebnf`)
- `enum`, `const` require grammar additions (see `mind-spec` future-extensions)

---

## Phase 11 — Benchmarks & Cloud Compiler Prototype

### Goals
Demonstrate performance, determinism, and flexibility. Introduce cloud-assisted builds.

### Deliverables
- Official benchmark suite:
  - matmul (CPU)
  - conv2d
  - small CNN
  - compile-time comparison (MIND vs PyTorch 2.0 AOT)
  - memory footprint analysis
- Publish results under `docs/benchmarks/*`
- Add deterministic build mode documentation
- Prototype cloud compiler endpoint:
  - `mind build --remote`
  - REST API for compilation requests
- Regression suite to ensure deterministic builds and MLIR→LLVM stability
- TurboQuant benchmark module (`bench/turboquant.mind`):
  - Validate that PolarQuant rotation + Lloyd-Max codebook + QJL residual correction
    can be expressed and compiled as pure MIND (requires Phase 10.5 struct/enum)
  - Benchmark: quantize/dequantize 1M KV vectors at 3-bit, measure throughput vs Python
  - This is the first real-world mind-inference kernel that exercises struct, enum,
    const, bitwise ops, and tensor ops together — ideal Phase 10.5 integration test
  - Ref: arXiv:2504.19874 (TurboQuant, ICLR 2026)

---

## Phase 12 — Enterprise Runtime & Edge Deployments

### Goals  
Enable enterprise and safety-critical adoption through stable runtime features, reproducibility, and mature edge support.

### Deliverables
- Finalize enterprise runtime:
  - feature-gated modules
  - deterministic execution
  - stable configuration API
- Edge deployment pipeline:
  - ARM Cortex-M support
  - RISC-V minimal backend
  - QEMU demo environment
- Deployment guidelines for:
  - cloud + on-prem
  - hybrid architectures
  - reproducible inference workloads
- Runtime safety documentation (Rust-based guarantees)

---

## Phase 13 — Neuroscience & Brain-Computer Interfaces

### Goals
Establish MIND as the premier language for real-time neural signal processing and brain-computer interface (BCI) applications. Leverage MIND's native tensor operations, static shape inference, and high-performance compilation for ultra-low-latency neural decoding in next-generation BCI systems (Neuralink-style implants, research BMIs, clinical neurotechnology).

### Deliverables

**Near-term (Phase 13.1):**
- Add `mind::neuro` standard library module:
  - Specialized tensor types for neural time-series (Channel × Time × Batch)
  - Common signal processing filters: bandpass, notch, ICA decomposition
  - Feature extraction primitives: CSP (Common Spatial Patterns), wavelet transforms, spectral power
- IO support for neuroscience data formats:
  - EDF/EDF+ (European Data Format) for clinical EEG
  - GDF (General Data Format) for BCI research
  - Optional integration with existing Rust neuro crates
- Example implementations:
  - Spike detection algorithms for invasive recordings
  - Motor intent decoding from ECoG/multi-unit activity
  - Basic P300 and SSVEP classifiers for non-invasive BCI

**Medium-term (Phase 13.2):**
- Optimized neural decoders:
  - RNN/LSTM templates for sequence-to-intent decoding
  - Transformer-based models for multi-modal neural data
  - Integration with autodiff for on-device decoder adaptation
- Tutorials and examples:
  - BCI Competition datasets (motor imagery, P300)
  - OpenNeuro integration for reproducible research
  - Real-time decoding pipeline examples
- Performance benchmarks:
  - Sub-millisecond inference for invasive BCI
  - Comparison with existing frameworks (MNE-Python, FieldTrip)

**Long-term (Phase 13.3):**
- Closed-loop BCI support:
  - Real-time feedback mechanisms
  - Online learning and decoder adaptation
  - Safety-critical execution guarantees for medical devices
- Research collaborations:
  - Partner with neuroscience labs for validation
  - Contribute to open BCI standards (IEEE, INCF)
  - Support for neuromorphic hardware targets

### Integration with Existing Features

- **Tensors**: Multi-channel neural data naturally maps to MIND's tensor model (channels, time, trials, batches)
- **Autodiff**: Enable gradient-based decoder optimization directly on implanted devices
- **MLIR/LLVM**: Generate highly optimized code for real-time constraints (<1ms latency)
- **Edge Runtime**: Deploy to resource-constrained BCI hardware (ARM Cortex-M, RISC-V)
- **Determinism**: Critical for reproducible neuroscience and FDA-regulated medical devices

---

## Phase 13.5 — Observer-Dependent Cognition (ODC) Language Primitives

### Goals
Introduce language-level support for observation-dependent computation. MIND already has `axis` for tensor operations — extend this to cognitive/governance contexts where the result of a computation depends on which observation basis is chosen.

**Spec:** `specs/observer-dependent-cognition.md`

### Deliverables
- **`@axis` annotation** — declare which observation basis a function/block operates in
  ```mind
  @axis("semantic")
  fn recall(query: Tensor[1, 768]) -> Tensor[K, 768] { ... }
  
  @axis("governance", invariants=[1,2,5,7])
  fn verify(txn: Transaction) -> Evidence { ... }
  ```
- **Axis contracts** — compile-time verification that composed computations use compatible observation bases
- **Determinism certificates** — compiler emits proof that a code path is:
  - `axis_independent` — fully deterministic, same result regardless of observation basis
  - `axis_dependent` — result varies with declared axis (must be explicitly declared)
- **`observe` keyword** — explicit measurement operator for superposition states
  ```mind
  let candidates = speculate(draft_model, context)  // superposition
  let result = observe(candidates, axis="verifier_a")  // collapse
  ```
- **Multi-axis composition** — compose results from different observation bases with explicit fusion strategy (RRF, voting, weighted)

### Integration
- MIND compiler validates `@axis` annotations at type-check time
- MLIR lowering preserves axis metadata for runtime evidence chain
- 512-mind governance modules use `@axis` to declare invariant coverage

---

## Phase 13.6 — Deterministic Distributed Primitives

### Goals

Today the runtime ships TCP transport, NCCL/Gloo backends, RingAllReduce,
pipeline parallelism, and fault tolerance — production-grade but
**non-deterministic by reduction order**. Different runs of the same
sharded model produce slightly different logits because IEEE-754
floating-point reductions are not associative under parallel scheduling.

For audit, regulatory, and replay-required workloads this is a blocker.
Phase 13.6 closes the gap with **language-level primitives** that emit a
deterministic TP+PP plan from a single `.mind` source — sharding stays,
the reduction race goes away.

The driving use case is the BitNet b1.58 ternary-weight LLM described in
[`bitnet-mind-governance/docs/parallel_pipeline.md`](https://github.com/star-ga/bitnet-mind-governance/blob/main/docs/parallel_pipeline.md):
3-way TP+PP across consumer GPUs with bit-identical replay across runs
*and* across hardware vendors (verified by per-stage merkle roots).

### Surface (proposed in `.mind`)

```mind
use mind.distributed.shard
use mind.distributed.allreduce
use mind.distributed.allgather
use mind.distributed.pipeline
```

### New invariants enforced by `mindc`

| Invariant | Effect |
|-----------|--------|
| `deterministic_all_reduce` | refuses to emit if any cross-shard reduction is not lexicographic-order |
| `reduction_order_lexicographic` | extends the existing local-matmul reduction-order rule to cross-device gathers |
| `gather_order_lexicographic` | shard contributions concatenate in fixed shard-id order, never timestamp-arrival order |
| `evidence_chain_continuous` | every pipeline stage must verify the previous stage's evidence emit before processing — bypassing the chain refuses to compile |

### Compiler-side IR (✅ scaffolded in `mindc` — Apr 2026)

The [`mind/src/distributed/`](../src/distributed/) module ships the
IR-layer primitives now. 31 unit tests pass on `cargo test --lib distributed::`.
Exposed types:

* `ShardSpec` / `ShardLayout` — tensor sharding spec with replicated /
  split / split-2D variants and an even-divisibility check at typecheck
  time.
* `AllReduceOp` / `ReductionOrder::Lexicographic` — lexicographic
  shard-ID schedule baked into the op at compile time.
* `AllGatherOp` / `GatherOrder` — same discipline applied to all-gather.
* `PipelineGraph` / `PipelineStage` / `StageBoundary` — `[pipeline_stage(N)]`
  attribute lowering, with the evidence-chain-continuous invariant
  attached to every transition.
* `DistributedInvariant` / `InvariantViolation` — typed enforcement of
  the four invariants above.

### Engineering scope to reach proof-of-concept

Six to eight weeks of focused work, scoped for paper-validation:

1. **`mind.distributed.shard` + `allreduce` + `allgather`** (~3 weeks)
   - Direct CUDA P2P for proof-of-concept (no NCCL needed initially)
   - Or shared-memory IPC for single-machine multi-process validation
   - Lexicographic-order all-reduce + all-gather primitives
2. **`mind.distributed.pipeline` with `[pipeline_stage(N)]`** (~2 weeks)
   - Send/recv between stages; evidence-chain continuity verification
3. **Port `bitnet.mind` to use the new primitives** (~1 week)
   - Re-shard along attention heads (TP), split layer groups (PP),
     combine for 3-way memory split
4. **Cross-shard determinism harness** (~1 week)
   - Capture per-stage merkle roots, compare to single-device run
   - Validate across (x86 + 3× consumer GPU) and (ARM + 3× different
     GPU) for cross-hardware bit-identicality

### Hypothesis the work proves

> **H4**: Cross-shard determinism is achievable on TP+PP at the cost of
> < 5% throughput overhead vs. non-deterministic FP16 baseline.

Two measurable claims (governance overhead < 1% on local BitNet from
H1, and cross-shard determinism overhead < 5% from H4) — together this
is the shape of an MLSys / OSDI methods paper.

### Speed-preservation discipline

The 1.8–15.5 µs frontend latency is the IP moat. Distributed primitives
must **never** widen it. Same rules as the language-profiles plan:

| Risk | What it looks like | Forbidden in mindc |
|------|--------------------|--------------------|
| Tax non-distributed code | Single-device compiles pay an analysis pass for collectives that aren't there | Distributed analysis only runs if at least one `mind.distributed.*` symbol is imported. Zero-import cost stays at 0 ns. |
| Linear shard-count blowup | Adding shards reruns shape inference per shard | `ShardSpec` is one struct attached to an SSA value; per-shard shape is computed once via `local_shape()`. World size doesn't enter typecheck cost. |
| Runtime collective dispatch in the binary | Binary contains a switch over collective backend / order | Reduction order is **compile-time-fixed** to lexicographic. The dispatch table never exists. |
| Per-statement invariant evaluation | Every op gets re-checked for `deterministic_all_reduce` | Invariant is a single flag on the op; verifier reads it in O(1). |

**Expected per-profile latency under Phase 13.6:**

| Profile + scenario | Expected frontend latency | vs. baseline |
|--------------------|---------------------------|--------------|
| `default`, no `mind.distributed.*` import | 1.8 – 15.5 µs | identical |
| `default`, with TP+PP collectives | ≤ 1.10 × baseline | bounded by O(num_collectives) shard-spec checks |
| `systems`, with TP collectives | ~0.9 – 3.3 µs | still faster than `default` baseline |

**Hard refuses (these stay out permanently):**

- Statement-level `#[cfg(distributed)]`. Module-level imports only.
- Runtime collective-order detection. Order is compile-time constant.
- Cross-profile distributed linking. A `default` shard cannot be linked
  into a `systems`-built peer (their stdlibs differ). Linker enforces
  via the MIC binary header.

The cache key in [`mind/src/cache/`](../src/cache/) already incorporates
`ProfileTag`; Phase 13.6 extends it with `WorldSize` so cross-shard
re-compilation never hits a stale entry.

### Dependencies

- mindc 0.3 primitives (`Vec`, struct fields, recursive functions)
  needed for the example modules to compile. Same blocker as the
  Language Profiles roadmap entry.

---

## Phase 14 — Full-Stack AI Framework

### Vision

MIND is evolving from a single-stack language focused on AI computation into a comprehensive full-stack AI framework that supports end-to-end machine learning. This stack will cover everything from building and training models to deploying them at scale in cloud and edge environments, providing a unified and cohesive experience for machine learning engineers, data scientists, and developers.

### Goals

- **End-to-End AI Development**: From research and prototyping to large-scale deployment
- **Seamless Integration**: Work naturally with popular data engineering and AI tools
- **Optimized Performance**: Efficient execution across CPU, GPU, and cloud environments
- **Ease of Use**: Well-integrated tools, APIs, and deployment frameworks

### Key Areas of Full-Stack Development

#### 1. Frontend Layer

**Language Enhancements:**
- Extend the language to improve developer experience through enhanced IDE support
- Advanced syntax highlighting and semantic code analysis
- Improved debugging tools with step-through tensor inspection

**Interactive Notebooks:**
- Browser-based interactive notebooks for writing, testing, and visualizing MIND code
- Jupyter-style cells with native MIND kernel support
- Real-time tensor visualization and model architecture diagrams

**Visualization Tools:**
- Integrated tools for visualizing tensor data and intermediate representations
- Model architecture visualization with automatic graph generation
- Training progress dashboards with loss curves, metrics, and resource utilization

#### 2. Backend Layer

**Compiler Optimization:**
- Extend the MIND Compiler to support advanced optimizations:
  - Automatic model pruning and quantization
  - Tensor fusion and operator scheduling
  - Enhanced support for TPUs, GPUs, and specialized AI accelerators

**Distributed Execution:**
- Distributed execution framework for training on clusters and multi-GPU setups
- Data parallelism and model parallelism strategies
- Gradient synchronization with NCCL-style collectives

#### 3. Middleware Layer

**API Layer:**
- REST and gRPC APIs for MIND-based model inference and serving
- OpenAPI/Swagger documentation generation
- Authentication and rate limiting for production deployments

**Data Integration:**
- Data engineering integrations for preprocessing and managing data pipelines:
  - Apache Kafka for streaming data
  - Apache Spark for batch processing
  - Native connectors for cloud storage (S3, GCS, Azure Blob)

**Model Serving:**
- Scalable, real-time model serving APIs for production deployment
- Batching and request queuing for throughput optimization
- A/B testing and canary deployment support

#### 4. Deployment Layer

**Containerization:**
- Pre-configured Docker images for training and serving MIND models
- Multi-stage builds optimized for minimal production footprint
- NVIDIA CUDA and ROCm base images for GPU workloads

**CI/CD Pipelines:**
- Continuous integration and deployment pipelines for automatic testing, training, and deployment
- GitHub Actions and GitLab CI templates
- Model versioning and artifact management

**Kubernetes Integration:**
- Helm charts for deploying MIND-based applications
- Horizontal pod autoscaling based on inference load
- Integration with Kubernetes operators for ML workflows (Kubeflow, MLflow)

#### 5. Ecosystem Support

**Pre-trained Models:**
- Library of pre-trained models for standard use cases:
  - Image classification (ResNet, EfficientNet variants)
  - Natural language processing (transformers, embeddings)
  - Time-series forecasting and anomaly detection

**Framework Interoperability:**
- Import/export compatibility with other frameworks:
  - ONNX model import/export
  - TensorFlow SavedModel conversion
  - PyTorch model migration tools

**Data Pipeline Connectors:**
- Seamless integration with data lakes and cloud platforms
- Connectors for popular databases (PostgreSQL, MongoDB, ClickHouse)
- Integration with feature stores (Feast, Tecton)

### Phased Rollout

**Phase 14.1 — Frontend Language and Developer Tools:**
- Enhanced language features for easier model development
- Full IDE support with syntax highlighting and auto-completion
- Integration of interactive notebooks for AI experimentation

**Phase 14.2 — Compiler Optimizations and Backend Support:**
- Extension of MIND's compiler to support advanced tensor optimizations
- Development of runtime support for distributed training
- Full support for cloud-based, multi-GPU, and distributed computing environments

**Phase 14.3 — Middleware Development:**
- Creation of MIND's API layer for inference and model serving
- Integration with data engineering frameworks (Kafka, Spark, etc.)
- Development of real-time serving APIs for production-ready AI models

**Phase 14.4 — Full-Stack Deployment and Ecosystem Integration:**
- Docker images and Kubernetes deployment support for MIND models
- Pre-trained models and ecosystem tools for end-users
- Full integration with CI/CD pipelines and cloud-native deployments

---

## Summary

These extended phases (10–14) guide MIND into the next major development era:
- richer SDK and examples,
- systems programming foundation (enum, struct, control flow, byte slices),
- formal benchmarks,
- early cloud compiler,
- enterprise-grade runtime,
- robust edge interfaces,
- and comprehensive full-stack AI framework capabilities.

Together they represent the roadmap toward **MIND v1.0** and production-ready adoption across multiple industries, enabling end-to-end machine learning from research to deployment.

---

## TRIZ-Driven Direction

This section captures the *criterion* against which roadmap items are evaluated, separate from the items themselves. It is direction-setting, not feature-generating. Every entry above is auditable against the Ideal Final Result below; every future entry should be too.

### Ideal Final Result (IFR)

> A developer writes governance and tensor logic at its natural level of abstraction. The compiler produces code that is provably safe under declared invariants, runs at the speed of unverified code, and requires no runtime checks the developer did not declare.

The IFR is *unreachable*. That is the point. It defines the gradient along which every roadmap decision should move, and it makes regression visible: any change that pushes the system away from this state is structurally suspect, regardless of how attractive it looks in isolation.

### Laws of System Development Applied to MIND

Five trajectories that govern how engineered systems evolve. Each suggests where to invest and where *not* to.

| Law | What it means for MIND | Status |
|---|---|---|
| **Uneven subsystem development** | One subsystem lags and gates the others | Frontend is fast (≤15.5 µs frontend stages). Tooling, IDE integration, diagnostics lag. Investing further compiler optimization yields less than investing tooling. |
| **Mono → bi → poly system** | Single → multi → many, with shared semantics | MIND is bi-system today (multiple targets via `mindc`). Poly-system goal: unified invariant semantics across MIND-language consumers, with cross-project contract verification. |
| **Increasing controllability** | Rigid → parameterized → automatic | `[invariant]` annotations are parameterization. Next stage: invariant *inference* under bounded conditions, not only assertion. |
| **Transition to micro-level** | Coarse mechanisms → fine-grained mechanisms | **Anti-pattern for MIND.** Statement-level configuration or runtime dispatch would break the compile-speed property. The micro-level transition is forbidden at the language frontend; if forced, it migrates to the runtime layer instead. |
| **Rhythm coordination** | Synchronization across subsystems | Q16.16 fixed-point already coordinates numerical rhythm across backends. Next stage: coordinating compile-time invariants with runtime evidence chains so consumers receive matched compile-time and runtime guarantees. |

### Separation Principles for Conflict Resolution

When two desirable properties conflict (e.g. *more invariants* vs *no compile-speed regression*), the design discipline is to separate the conflict rather than compromise:

- **Separation in time** — the property holds at one phase of the pipeline, not another (e.g. invariant verified at compile time, not re-checked at runtime).
- **Separation in space** — the property holds in one module or scope, not another (e.g. governance invariants on `governance/` paths, not on tensor kernels).
- **Separation in condition** — the property holds under one configuration, not another (e.g. debug-mode trace, release-mode silent).
- **Separation in scale** — the property holds at one granularity, not another (e.g. module-level feature gates, never statement-level).

This is the same separation discipline used by `[invariant]` conflict resolution in MIND-language consumers (see consumer-side governance roadmaps). It is *not* a 40-principles invocation. It is the small subset of TRIZ that has direct mechanical use here; the rest is left as catalog reference, not method.

### Anti-Patterns Made Explicit

- **Statement-level `cfg` or runtime dispatch in the frontend.** Violates compile-speed preservation. Any feature requiring it is rejected at design time, not after benchmarking.
- **Adding invariants without a separation strategy.** New invariants must declare *which separation principle* makes them coexist with existing ones. If none applies, the conflict is real and a redesign is required.
- **Driving design from the analogy.** TRIZ is a checklist for direction, not a generator. Features come from product needs and consumer pull; TRIZ filters them, it does not invent them.

---

## AGI Integration

This repo is part of the STARGA AGI stack. See `naestro-bot/specs/AGI-ROADMAP.md` for:
- Gap 3: Rule induction → .mind programs (compile + verify)
- Gap 8: Reasoning primitives (hypothesis, evidence, inference_step)
- Phase 10.5 is prerequisite for all AGI .mind modules
