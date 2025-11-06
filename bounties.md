# MIND Bounty Board

This page tracks high-impact issues eligible for bounties. Bounties are awarded for merged PRs that fully address the listed requirements.

## How Bounties Work

1. **Find an issue** marked with 💰 on this page
2. **Comment on the issue** to claim it (first come, first served)
3. **Submit a PR** that meets the acceptance criteria
4. **Get reviewed** by core team
5. **Receive bounty** after PR is merged

## Current Bounties

### 💰 $500 - Implement MLIR Tensor Dialect Integration
**Issue**: [#TBD]  
**Difficulty**: Advanced  
**Skills**: Rust, MLIR, compilers

**Description**: Build the bridge between MIND's type system and MLIR's tensor dialect.

**Requirements**:
- Map MIND tensor types to MLIR tensor types
- Implement shape inference in MLIR
- Handle device placement annotations
- Add comprehensive tests
- Document the implementation

**Timeline**: 4-6 weeks

---

### 💰 $300 - Build Shape Checker with Error Messages
**Issue**: [#TBD]  
**Difficulty**: Medium  
**Skills**: Rust, type systems, compiler diagnostics

**Description**: Implement compile-time shape checking with helpful error messages.

**Requirements**:
- Implement shape unification algorithm
- Add symbolic shape support (e.g., batch size B)
- Generate clear error messages with suggestions
- Add 50+ test cases covering edge cases
- Write developer documentation

**Timeline**: 3-4 weeks

---

### 💰 $200 - Create VSCode Extension
**Issue**: [#TBD]  
**Difficulty**: Medium  
**Skills**: TypeScript, VSCode API, language servers

**Description**: Build syntax highlighting and basic LSP support for MIND in VSCode.

**Requirements**:
- Syntax highlighting for all MIND constructs
- Basic LSP server with:
  - Go to definition
  - Hover information
  - Error diagnostics
- Code snippets for common patterns
- Published to VSCode marketplace

**Timeline**: 2-3 weeks

---

### 💰 $150 - Implement Autodiff for Loops
**Issue**: [#TBD]  
**Difficulty**: Advanced  
**Skills**: Rust, automatic differentiation, compilers

**Description**: Extend autodiff to work through for and while loops.

**Requirements**:
- Implement loop unrolling for static bounds
- Add tape-based recording for dynamic bounds
- Handle break/continue statements
- Add benchmarks comparing with PyTorch
- Write technical documentation

**Timeline**: 3-4 weeks

---

### 💰 $100 - Write Comprehensive Benchmarks Suite
**Issue**: [#TBD]  
**Difficulty**: Low-Medium  
**Skills**: Rust, benchmarking, MIND

**Description**: Create a comprehensive benchmark suite comparing MIND with PyTorch/JAX.

**Requirements**:
- Implement 10+ common ML operations
- Equivalent implementations in PyTorch and JAX
- Run on CPU and GPU
- Generate comparison graphs
- Document methodology

**Timeline**: 2 weeks

---

## Bounty Rules

### Eligibility
- Open to everyone
- Can claim only one bounty at a time
- Must not be a core team member (for paid bounties)

### Process
1. Bounties are paid via GitHub Sponsors, Open Collective, or crypto
2. Payment within 2 weeks of PR merge
3. Disputes resolved by core team vote

### Quality Standards
- Code must pass CI (tests, clippy, formatting)
- Must include tests and documentation
- Must follow contribution guidelines
- Breaking changes require RFC approval

## Proposing New Bounties

Open a discussion with problem, acceptance criteria, difficulty, timeline, and suggested amount.

## Sponsoring Bounties

- Email: bounties@mindlang.dev
- Discord: #bounties channel

**Last Updated**: 2025-11-06  
**Total Active Bounties**: $1,250
