# GitHub Setup (Quick)
Enable Issues, Discussions, Projects. Add topics. See CI in `.github/workflows/ci.yml`.

Tarpaulin coverage in CI runs without optional LLVM/MLIR features by default. To collect feature-specific coverage, define the repository variable `ENABLE_OPTIONAL_FEATURE_COVERAGE=true` and ensure the runners provide the necessary LLVM/MLIR toolchains before enabling the optional coverage job matrix.
