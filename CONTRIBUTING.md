# Contributing to MIND

Thank you for your interest in contributing to MIND!

## Code of Conduct

Be respectful. No harassment or discrimination. Assume good intent. Help newcomers.

## Getting Started

### Prereqs

```bash
# Rust (1.70+)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# LLVM 15+
# Ubuntu:
sudo apt-get install -y llvm-15 llvm-15-dev
# macOS:
brew install llvm@15
```

### Build & Test

```bash
git clone https://github.com/cputer/mind.git
cd mind
cargo build
cargo test
```

### Linting

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
```

## Pull Requests

1. Fork & branch (`feat/*` or `fix/*`)
2. Add tests & docs
3. Make sure CI passes
4. Open PR with clear description and link to issues

## RFCs

For major changes, copy `docs/rfcs/0000-template.md` and open a PR to discuss.

## Thanks ❤️
Your contributions make MIND better.
