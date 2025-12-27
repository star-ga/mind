# GitHub Setup (Quick)
Enable Issues, Discussions, Projects. Add topics. See CI in `.github/workflows/ci.yml`.

Tarpaulin coverage in CI runs without optional LLVM/MLIR features by default.

### Coverage with optional features (opt-in)

By default, CI coverage runs with `--no-default-features` to avoid pulling optional dependencies (e.g., LLVM/MLIR).

To enable an opt-in coverage matrix for optional features, set a repository variable:

- **Settings → Secrets and variables → Actions → Variables**
- Add: `ENABLE_OPTIONAL_FEATURE_COVERAGE = true`

This will run an additional Tarpaulin job over a small feature matrix (e.g., `mlir`, `llvm`, `autodiff`). Ensure any required toolchains (such as LLVM) are available on runners before enabling.

### Quality & Security guardrails
- CI enforces rustfmt, clippy (core build), and supply-chain checks (cargo-deny & cargo-audit).
- Release notes are auto-drafted by Release Drafter on pushes/PRs to `main`.
