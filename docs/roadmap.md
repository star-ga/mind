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
