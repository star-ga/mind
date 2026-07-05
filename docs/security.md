# Security Guide

> **Audience:** MIND users deploying in regulated environments

> **Status note (2026-07):** This guide describes the *intended* production
> security surface. Flags and annotations shown here that are not in
> [`docs/cli.md`](cli.md) (`--release`, `--fast-math`, `--sandbox`,
> `@validate`) are planned, not yet shipped. The currently-verified
> determinism scope is defined in [`docs/determinism.md`](determinism.md):
> bit-identical integer/Q16.16 execution gated across x86 == ARM, with scalar
> IEEE-754 f64/f32 on the strict path (run-to-run bit-identical, and verified
> byte-identical across x86_64 and ARM64 on real hardware, 2026-07-05).

## Overview

MIND is designed for certified AI systems in safety-critical and regulated industries. This guide covers security best practices for production deployments.

## Deterministic Execution

MIND targets bit-exact reproducibility by default (see the status note above
for the currently-verified scope). This is critical for:
- FDA-regulated medical devices
- Auditable AI systems
- Reproducible research

```bash
# Verify determinism
mindc --release model.mind -o model
./model input.tensor > output1.txt
./model input.tensor > output2.txt
diff output1.txt output2.txt  # Should be empty
```

### Enabling Fast Math (Disables Determinism)

```bash
# Only for non-certified workloads where performance is critical
mindc --release --fast-math model.mind
```

**Warning:** Fast math mode sacrifices bit-exact reproducibility for performance. Never use in regulated deployments.

## Input Validation

Always validate untrusted inputs at system boundaries:

```mind
@validate(range=[-1.0, 1.0], no_nan=true)
fn process(input: Tensor<f32, [batch, 224, 224, 3]>) -> Tensor<f32, _> {
    // Input guaranteed valid by static annotation
    model(input)
}
```

## Supply Chain Security

MIND uses `cargo-deny` for dependency auditing:

```bash
# Check for known vulnerabilities
tools/cargo-deny-sanitize.sh check

# Audit all dependencies
cargo audit
```

> **Note:** Until `cargo-deny` ships CVSS v4 support, run it through
> `tools/cargo-deny-sanitize.sh` so the advisory database is sanitized (the
> script removes CVSS v4 lines from affected advisories after `cargo deny
> fetch`). This keeps the check working without mutating the ignore list.

The `deny.toml` configuration enforces:
- License compliance
- No unmaintained dependencies
- CVE blocking

## Sandboxed Execution

For untrusted model evaluation:

```bash
# Run with resource limits
mindc --sandbox=strict \
      --max-memory=4G \
      --max-time=60s \
      untrusted_model.mind
```

## MAP Protocol Resource Budgets

The Mind AI Protocol (MAP) server (`mind-ai`) is a long-lived, stateful,
line-oriented protocol that accepts untrusted input from AI agents
(`load.mic`, `patch.*`, `dump`, `check`, `query.*`). Because a single session
persists across many requests and holds a resident module in memory, the
server enforces explicit, named resource budgets so a hostile or buggy peer
cannot exhaust memory or CPU. Each budget is enforced with a **clear,
structured error** — the server never panics, aborts, or silently truncates.

| Budget | Constant | Default | Guards against | Error code |
|--------|----------|---------|----------------|------------|
| Single line size | `MAX_LINE_BYTES` | 1 MiB | One enormous request or heredoc-body line forcing an unbounded allocation | `E101` |
| Heredoc body size | `MAX_SESSION_BYTES` | 16 MiB | An unterminated or hostile heredoc accumulating without bound before `EOF` | `E102` |
| Module node count | `MAX_MODULE_NODES` | 1,000,000 | A module whose node count makes per-node validation/patch scans or resident memory unbounded (checked on `load.mic` and after any `patch.*`) | `E103` |
| Patch operation rate | `MAX_PATCH_OPS` | 100,000 / session | An unbounded stream of mutating `patch.*` operations triggering repeated whole-module rewrites | `E104` |

### Enforcement behaviour

- **Bounded parsing.** Oversized physical lines are rejected before any command
  is dispatched. The body of an over-budget heredoc is *drained* (discarded)
  until its closing `EOF` so it is never reinterpreted as protocol commands.
- **Pre-commit node check.** `load.mic`, `patch.insert`, and `patch.replace`
  compute the candidate module's node count and reject it *before* it becomes
  the resident module, so a rejected operation leaves session state unchanged.
- **Per-session patch budget.** The patch-operation counter is charged on every
  mutating `patch.*` call and reset to zero on `bye` (session teardown).
- **Structured errors.** Every rejection is returned as
  `=<seq> err code=E1xx msg="..."`, so a client can match on the stable code
  while still receiving a human-readable reason.

These budgets are compile-time constants in `src/bin/mind-ai.rs`; adjust them
there if a deployment legitimately needs larger modules or higher patch rates.

## Secure Defaults

MIND's defaults prioritize safety:

| Feature | Default | Security Rationale |
|---------|---------|-------------------|
| Determinism | Enabled | Reproducible audits |
| Bounds checks | Enabled | Memory safety |
| Integer overflow | Checked | Defined behavior |
| Fast math | Disabled | Bit-exact results |

## Rust Safety Guarantees

The MIND compiler is written in Rust, providing:
- Memory safety without garbage collection
- Thread safety guarantees
- No undefined behavior in safe code

## Reporting Vulnerabilities

Report security issues privately to the repo owner (@star-ga). See [SECURITY.md](../SECURITY.md) for the full policy.

## Compliance Considerations

For deployments requiring certification:

1. **Reproducible Builds**: Use `--release` (not `--fast-math`)
2. **Audit Trail**: Enable logging for inference operations
3. **Version Pinning**: Lock compiler and runtime versions
4. **SBOM Generation**: Document all dependencies

See [docs/versioning.md](versioning.md) for stability guarantees.
