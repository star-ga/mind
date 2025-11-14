# Roadmap

This roadmap outlines upcoming milestones for the MIND language, runtime, and tooling.

## Near Term (0–3 months)

- **v1.0 Stabilization** – Lock core syntax and IR semantics, add conformance tests.
- **Package Registry Preview** – Publish early `mindpkg` registry with curated models.
- **Improved Diagnostics** – Structured error messages with fix-it hints in the CLI.
- **Link-Check Automation** – Enforce documentation link health in CI (see `.github/workflows/link-check.yml`).

## Mid Term (3–6 months)

- **GPU Backend GA** – Graduate MLIR CUDA backend with memory residency analysis.
- **Autodiff Optimizations** – Integrate checkpointing heuristics and sparse gradients.
- **FFI Stabilization** – Freeze ABI for embedding and release C++ helpers.
- **Docs Migration** – Publish documentation to `mindlang.dev/docs/*` via Eleventy.

## Long Term (6–12 months)

- **Distributed Runtime** – Add multi-node execution with NCCL-style collectives.
- **Formal Verification** – Explore proof-carrying IR passes for safety-critical deployments.
- **Ecosystem Integrations** – Official bindings for Python, Swift, and WebAssembly targets.

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

## Summary

These extended phases (10–12) guide MIND into the next major development era:
- richer SDK and examples,
- formal benchmarks,
- early cloud compiler,
- enterprise-grade runtime,
- and robust edge interfaces.

Together they represent the roadmap toward **MIND v1.0** and production-ready adoption across multiple industries.
