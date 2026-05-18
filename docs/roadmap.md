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
- ✅ **Phase 10.5: Governance Logic** – enum, struct, if/else, while,
  const, bitwise/boolean ops, struct literals, enum-variant `::`
  access shipped in v0.2.5–v0.2.10.

## In Progress

- 🚧 **Phase 10.6: Surface Syntax & Library Output** – tuple types,
  references, generics, slices, fixed-size arrays, struct literals,
  indexed/field assignment, multi-line arithmetic, RFC 0002 C-ABI export
  (D1/D3/D5 shipped in 0.2.6–0.2.8; D2 codegen pending).
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

## Phase 10.6 — Library Output & C ABI (mindc 0.2.6 / 0.3.0)

### Goals
Turn `mindc` from an executable-only compiler into a library-emitting one,
so MIND code can be consumed by C, Python, Rust, and other runtimes via a
stable C ABI. This is the foundation Phase-2 native inference paths
(mind-nerve, MindLLM, rfn-mind) compile against.

### Motivation
Today: `Mind.toml`'s `[exports]` section is parsed but silently dropped.
`Node::Export` is parsed and discarded. The .so produced has no AOT
symbols for user `pub fn` — runtimes embed source and re-parse on dlopen.
The mind-runtime embedded parser has drifted from mindc's surface parser;
shipping `.mic` IR instead of source is the canonical path forward.

### Compile-speed invariant (non-negotiable)
The 1.8–15.5 µs frontend latency is the IP moat. Every Phase 10.6 feature
is gated **module-level only** — no statement-level cfg, no runtime
dispatch. Each item below ships a dedicated sub-benchmark that does not
move the headline numbers. CI gate: ±2% per size / ±1% mean.

### Deliverables — mindc 0.2.6 (library-emit foundation)

1. **`Node::Export` → IR `FnExport` variant.** Lower the parsed
   `export { foo, bar }` into `IRModule.exports: HashSet<FnId>` and verify
   the named functions exist. Gate: `feature = "ffi-c-user"`, parse-time
   early-out when no `export` keyword present.
2. **C wrapper codegen for `pub fn`.** For each `IRModule.exports` entry,
   emit `mind_fn_{name}_invoke(inputs, in_count, outputs, out_count) -> int`
   using the existing `MindIO` calling convention. Static FFI stubs stay
   for backward compat.
3. **`Mind.toml [exports]` honored end-to-end.** `ProjectManifest.exports`
   already parses; wire the consumer in the build pipeline to prepend
   manifest-declared symbols into the IR export list before codegen.
4. **Document `mic@1` IR consumption pattern.** Canonical parse happens at
   compile time; runtimes consume `.mic` via `ir::load()`. Embedded
   runtime parser is deprecated. mind-nerve, MindLLM, rfn-mind switch from
   shipping source to shipping `.mic` artifacts.
5. **`--profile=<default|systems|embedded>` CLI flag.** Wire the
   ProfileTag from `src/cache/fingerprint.rs` through the build pipeline
   so the same `Mind.toml` produces three distinct artifacts.

### Deliverables — mindc 0.3.0 (cdylib era + protection actions)

6. **`mind build --lib` cdylib emit.** Output `.so`/`.dylib`/`.dll`
   directly with symbol visibility limited to exported user wrappers +
   standard FFI stubs. Entry point not invoked; library is dlopen'd.
7. **AOT native codegen with symbol versioning.** LLVM lowering via
   `inkwell`; emit object files with `mind_fn_{name}_v1` symbols.
   Runtimes call `dlsym()` instead of embedded parser. Gates: existing
   `mlir-build` feature + new `aot-exports` feature.
8. **`Mind.toml [protection]` action transforms.** Three tiers documented
   in a new RFC (`docs/rfcs/0002-protection.md`): `obfuscate_strings`
   (compiler-side string table), `anti_debug` (runtime canary hook),
   `anti_tamper` (binary hash check). Replaces the hand-rolled
   `build.sh`-driven FORTRESS post-processing in mind-mem-protected /
   mind-nerve-protected.
9. **Cross-directory imports (`use crate::foo::bar`).** Module-qualified
   resolver populates a module table during project load; type checker
   resolves identifiers across module boundaries. Gate:
   `feature = "cross-module-imports"`.

### Deprecated in this phase

- `Mind.toml [targets.cpu] sources / c_sources / headers` — decorative
  fields with no consumer. MIND's compilation target is IR → MLIR →
  LLVM IR, not mixed C+MIND. C interop comes via `[package.ffi]` in a
  separate RFC if needed.

### Sequencing

Items 1–5 ship in mindc 0.2.6 (~2 weeks). Items 6–9 ship in mindc 0.3.0
(~6 weeks). Item 8's `[protection]` action transforms unlock per-customer
FORTRESS builds without external post-processing scripts.

### Anti-regression discipline

- Every PR touching parser / type_checker / lowering / codegen runs
  `cargo bench --bench compiler` and reports baseline-vs-current.
- New features ship as separate sub-benchmarks (e.g.,
  `export_list_1item`), never as deltas to the headline numbers.
- Gated features verify their cfg cost is <0.5% on headline benchmarks.
- `CacheKey` extends for every feature that changes compilation behavior.

### Acceptance gates

- mind-nerve compiles with `pub fn preselect_pre_tokenized` exported and
  callable as `mind_fn_preselect_pre_tokenized_invoke()`.
- mind-nerve's `Mind.toml [exports] c_abi = [...]` works end-to-end.
- mind-nerve build flow documents `mind compile --emit-ir` and the
  runtime loads `.mic` instead of `.mind`.
- MindLLM, rfn-mind, mind-mem repeat the same pattern.
- Cross-arch bit-identity gate (mind-nerve task #57) unblocked by 0.3.0.
- p95 ≤ 30 ms on 4-core CPU — already met on the Phase 1 PyTorch path via
  the `mind-nerve-routed` UNIX-socket daemon (23 ms p95 after warmup,
  closed in 0.1.0-alpha.7). The 0.3.0 AOT path tightens this further by
  replacing the embedded-parser dlopen path.

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

### Runtime Layer (Mind-Runtime)

The frontend IFR above governs the compiler. The runtime layer has a *different* IFR with different optima — including one law that is explicitly inverted from the frontend rule. Both layers live in this repository and must stay coherent under both criteria.

**Runtime IFR.** Compiled MIND programs execute with bit-exact determinism across backends, record verifiable evidence of every governance-relevant decision, and enforce declared invariants at zero overhead on the legitimate execution path. Failures degrade into sealed evidence rather than silent corruption.

**Laws applied at the runtime layer.**

| Law | Runtime application | Status |
|---|---|---|
| Uneven subsystem development | Runtime backends evolve unevenly (CPU, GPU, accelerator). Slowest backend gates portability claims. | Invest in the slowest backend, not the fastest. |
| Mono → bi → poly | Runtime is poly today (multiple backends). | Next stage: unified evidence chain across backends so a program's evidence trail is backend-independent. |
| Increasing controllability | Runtime policy is largely compile-time-fixed. | Next stage: runtime-tunable severity tiers within a compile-time-bounded envelope (the envelope is immutable; the position inside it is configurable). |
| Transition to micro-level | **Inverted from frontend.** At runtime, instruction-level scheduling, branch hints, vectorization are appropriate and often required. The forbidden boundary is *crossing* a compile-time invariant, not *implementing* it efficiently. | Pursue micro-level wherever the legitimate path demands it; never let it weaken an invariant. |
| Rhythm coordination | Runtime emits evidence; compile-time consumers verify invariants against it. | The cycle closes only when both timescales agree on the schema and meaning of each evidence record. |

**Separation principles at runtime.**

- **Time** — compile-time verification vs runtime check vs runtime evidence record. Three phases, three different cost/value trade-offs. A property proven at compile time is *not* re-verified at runtime; that path has been replaced with evidence emission, not redundancy.
- **Space** — hot path vs cold path vs governance audit path. Hot path admits no overhead beyond the IFR; cold path admits diagnostic instrumentation; audit path admits full evidence-chain materialisation.
- **Condition** — release vs debug vs forensic mode. The same program produces different evidence detail under different modes; the compile-time invariants are unchanged.
- **Scale** — per-instruction, per-basic-block, per-module, per-program. Each scale has its own legitimate optimisation surface; mixing scales is a code smell.

**Runtime anti-patterns.**

- **Re-verifying at runtime what was proven at compile time.** Wasted overhead, no added safety, violates the IFR.
- **Silent failure paths.** Any failure that does not produce a sealed evidence record breaks the chain and the contract with governance consumers.
- **Backend-specific semantics leaking into program behaviour.** Determinism is a per-program property; backend differences must be invisible above the FFI boundary.
- **Performance regressions justified by future features.** The IFR demands zero overhead on the legitimate path *now*; deferred-cost arguments are rejected at design time.

---

## Phase 15 — Self-Hosting & AI-Era Language Efficiency

The frontend moat and the invariant/evidence/determinism core only
compound if the language can host itself and is optimised for the
metric that actually matters now: an AI writes the code, a compiler
must prove it, and the result must be reproducible without re-running
it. Phase 15 makes both explicit and tracked.

### Guiding principle (holds for every Phase 15+ feature)

**AI-era efficiency = minimise tokens-to-a-trusted-reproducible
artifact, never at the cost of the µs frontend.** Concretely the
product to minimise is:

```
(tokens to express correct intent)
  × P(the generator got it wrong)
  × (cost to detect it is wrong)
  × (cost to reproduce / audit the result)
```

C optimises only the first factor's runtime; dynamic languages
optimise only token count. MIND optimises the whole product. Any
feature that improves one factor by regressing the frontend envelope
(`.bench-baseline` ±2% gate on the three headline benches) is
rejected at design time — the same anti-regression discipline already
in force.

### Deliverables — self-hosting bootstrap

1. **Stage-0 → Stage-1 → Stage-2 bootstrap.** Rust `mindc` (stage-0)
   compiles a MIND-sourced `mindc.mind` (stage-1); stage-1 compiles
   itself (stage-2). `stage1 == stage2` byte-identical is the
   self-hosting acceptance gate. Rust leaves the build path on pass.
2. **`IRModule → LLVM IR text` backend.** The self-hosted compiler
   emits textual LLVM IR (`.ll`) and shells out to `llc`/`clang` —
   the C-ABI emitted by RFC 0002/0003 is the shell-out seam. MLIR
   stays a stage-0 / multi-backend concern; the bootstrap path skips
   it to minimise dependencies. `IRModule` is the fixed point shared
   by both stages — it is not redesigned for self-hosting.
3. **Pure-MIND std surface.** Growable `Vec`, `String` operations,
   hash map, deterministic file I/O. This is the long pole: a
   lexer/parser/symbol-table cannot be written without it.
4. **Cross-module imports** (`use crate::x::y`) — already roadmap
   item 9 of Phase 10.6; restated here as a hard self-hosting
   prerequisite (a compiler is many files).

### Deliverables — AI-era efficiency surface

5. **Token-efficiency as a first-class design constraint.** Every
   surface-syntax decision is evaluated by "how many tokens does a
   generator spend to express this *correctly*?" Terse-by-default;
   ceremony is a regression. Tracked as a review checklist item, not
   aesthetics.
6. **AI-era efficiency benchmark.** A published harness measuring the
   four-factor product above on a fixed task suite, reported per
   release alongside the frontend µs benches. The yardstick is
   defined by this metric, not SPECint.
7. **Deterministic concurrency model (RFC pending).** Structured,
   replayable, evidence-carrying parallelism. MIND today is
   deterministic-sequential; AI systems are inherently parallel, so
   this is the largest single gap between "efficient language" and
   "efficient language for AI systems." Prerequisite for any
   concurrent MIND program. Highest-priority new RFC.
8. **Index-refinement types (RFC pending).** Presburger-decidable
   refinements of the form `{x: T | lo <= x < hi}` over integer
   indices — SMT-free, so frontend-safe. Encodes tensor-shape bounds
   and governance ranges as types. (Flux-style; full SMT refinement
   is explicitly rejected as moat-incompatible.)
9. **Tensor layout + uniqueness annotations (RFC pending).**
   Type-level memory-layout (row/col-major, alignment) and
   Futhark-style uniqueness for in-place mutation proofs — the
   missing primitives for provably-aliasing-free GPU kernels. O(1)
   parse cost, single-bit IR flag.

### Sequencing

Items 3 + 4 are the long pole and gate everything else. 1 + 2 are
mechanical once 3 + 4 land (the backend mirrors the existing
`src/mlir/lowering.rs` structural-fold pattern). 5 is continuous
discipline from now. 6–9 are independent RFCs; 7 (deterministic
concurrency) is the highest priority because it gates both concurrent
MIND programs and the AI-era efficiency claim.

### Acceptance gates

- `stage1 == stage2` byte-identical; Rust removed from the documented
  build path.
- A non-trivial MIND program (target: the lexer/parser of `mindc`
  itself) compiles and runs through the self-hosted backend with
  cross-arch bit-identical output.
- The AI-era efficiency benchmark is published and tracked per
  release; no frontend headline-bench regression beyond the ±2% gate
  across all Phase 15 work.

---

## AGI Integration

This repo is part of the STARGA AGI stack. See `naestro-bot/specs/AGI-ROADMAP.md` for:
- Gap 3: Rule induction → .mind programs (compile + verify)
- Gap 8: Reasoning primitives (hypothesis, evidence, inference_step)
- Phase 10.5 is prerequisite for all AGI .mind modules
