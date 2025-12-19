# Security Guide

> **Audience:** MIND users deploying in regulated environments

## Overview

MIND is designed for certified AI systems in safety-critical and regulated industries. This guide covers security best practices for production deployments.

## Deterministic Execution

MIND guarantees bit-exact reproducibility by default. This is critical for:
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
cargo deny check

# Audit all dependencies
cargo audit
```

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

Report security issues privately to the repo owner (@cputer). See [SECURITY.md](../SECURITY.md) for the full policy.

## Compliance Considerations

For deployments requiring certification:

1. **Reproducible Builds**: Use `--release` (not `--fast-math`)
2. **Audit Trail**: Enable logging for inference operations
3. **Version Pinning**: Lock compiler and runtime versions
4. **SBOM Generation**: Document all dependencies

See [docs/versioning.md](versioning.md) for stability guarantees.
