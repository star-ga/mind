# Contributing to MIND

## Toolchain
- Rust stable (pinned by CI), `cargo`, `rustfmt`, `clippy`
- No default features must compile: `cargo check --no-default-features`

## CI gates
- fmt: `cargo fmt --all -- --check`
- build/tests: `cargo test --no-default-features`
- clippy: `cargo clippy --no-default-features -- -D warnings`
- license/advisories: `cargo deny check`

## PR rules
- Small, focused commits
- Tests for new behavior (`tests/` integration preferred)
- Update README/docs when user-visible

## Commit style
- Conventional-ish: `feat:`, `fix:`, `docs:`, `refactor:`, `ci:`, `chore:`
