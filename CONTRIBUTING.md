# Contributing to MIND

Thank you for your interest in contributing to MIND! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- **Rust**: stable toolchain (1.82+), nightly for some optional features
- **Tools**: `cargo`, `rustfmt`, `clippy`
- **Optional**: LLVM/MLIR for backend features

### System Requirements

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64) | Fully supported | Primary development platform |
| macOS (x86_64/ARM64) | Fully supported | CI tested |
| Windows (x86_64) | Fully supported | CI tested |

### Setup

```bash
git clone https://github.com/star-ga/mind.git
cd mind
cargo build --no-default-features
cargo test --no-default-features
```

## Development Workflow

### Toolchain Requirements

- No default features must compile: `cargo check --no-default-features`
- Format with rustfmt: `cargo fmt --all`
- Lint with clippy: `cargo clippy --no-default-features -- -D warnings`

### CI Gates

All PRs must pass these checks:

| Check | Command |
|-------|---------|
| Format | `cargo fmt --all -- --check` |
| Build | `cargo build --no-default-features` |
| Test | `cargo test --no-default-features` |
| Clippy | `cargo clippy --no-default-features -- -D warnings` |
| License | `cargo deny check` |
| Docs | `cargo doc --no-default-features --document-private-items` |

### Running Tests

```bash
# All tests (no features)
cargo test --no-default-features

# With specific features
cargo test --features autodiff
cargo test --features cpu-exec

# Single test
cargo test test_name
```

## Making Contributions

### Issue Triage

Before starting work:

1. **Check existing issues**: Search for related issues/PRs
2. **Open a discussion**: For large changes, discuss first
3. **Claim the issue**: Comment to indicate you're working on it

Issue labels:

- `good first issue`: Suitable for newcomers
- `help wanted`: Open for contributions
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements

### Pull Request Process

1. **Fork and branch**: Create a feature branch from `main`
2. **Make changes**: Keep commits small and focused
3. **Test locally**: Ensure all CI checks pass
4. **Update docs**: Document user-visible changes
5. **Submit PR**: Use the PR template

### PR Guidelines

- **Small, focused commits**: One logical change per commit
- **Tests required**: Add tests for new behavior (prefer `tests/` integration tests)
- **Documentation**: Update README/docs for user-visible changes
- **No breaking changes**: Without prior discussion

### Commit Style

We use conventional commit format:

```
type: short description

Longer explanation if needed.

Fixes #123
```

Types:

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |

Examples:

```
feat: add symbolic shape unification
fix: handle edge case in broadcast inference
docs: update autodiff examples
refactor: simplify IR canonicalization pass
```

## Developer Certificate of Origin (DCO)

By contributing, you certify that:

1. The contribution was created by you, or
2. You have the right to submit it under the project license, or
3. It was provided to you by someone with rights to submit

You can sign off commits with:

```bash
git commit -s -m "feat: your feature"
```

This adds a `Signed-off-by` line to your commit message.

## Code Review

### What We Look For

- Correctness and test coverage
- Code clarity and maintainability
- Documentation quality
- Performance considerations
- Security implications

### Review Timeline

- Initial review: within 3 business days
- Follow-up reviews: within 2 business days
- Merge after approval from at least one maintainer

### Responding to Feedback

- Address all comments before requesting re-review
- Use "Resolve conversation" when addressed
- Ask for clarification if feedback is unclear

## Architecture Guidelines

### Project Structure

```
src/
├── ast/          # Abstract syntax tree
├── lexer/        # Tokenization
├── parser/       # Parsing
├── types/        # Type system
├── shapes/       # Shape inference
├── ir/           # Intermediate representation
├── opt/          # Optimizations
├── autodiff/     # Automatic differentiation
├── mlir/         # MLIR lowering
├── eval/         # Evaluation/runtime
├── exec/         # Execution backends
└── ffi/          # Foreign function interface
```

### Code Style

- Follow existing patterns in the codebase
- Use descriptive names for functions and variables
- Add doc comments for public APIs
- Keep functions focused and small
- Avoid unnecessary complexity

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: File an issue with reproduction steps
- **Security**: See [SECURITY.md](SECURITY.md)
- **Chat**: Join discussions in issues/PRs

## Recognition

Contributors are recognized in:

- Release notes
- GitHub contributor graphs

Thank you for contributing to MIND!
