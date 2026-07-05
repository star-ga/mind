# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

We provide security updates for the latest release and the `main` branch. Older versions are not supported.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT file a public GitHub issue** for security vulnerabilities
2. **Email**: Send details to `security@star.ga`
3. **Alternative**: Contact the repo owner (@star-ga) directly via GitHub's private vulnerability reporting

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial acknowledgment | Within 48 hours |
| Preliminary assessment | Within 7 days |
| Fix development | Within 30 days (severity dependent) |
| Public disclosure | After fix is released |

### Disclosure Policy

- We follow coordinated disclosure practices
- Credit will be given to reporters (unless anonymity is requested)
- We aim to release fixes before public disclosure
- Critical vulnerabilities may receive expedited handling

## Security Best Practices

When using MIND in production (as a dependency in your own applications):

1. **Pin dependencies**: Commit your application's `Cargo.lock` to ensure reproducible builds
2. **Verify checksums**: Validate release artifacts before deployment
3. **Sandbox execution**: Run compiled MIND programs in isolated environments
4. **Audit inputs**: Validate all external data before processing
5. **Update regularly**: Keep MIND and dependencies up to date

## Security Audits

- `cargo deny check` runs on all PRs for license and advisory checks
- Clippy with `-D warnings` enforces safe Rust patterns
- No `unsafe` code in core compiler paths

## Scope

This security policy covers:

- The MIND compiler (`mindc`)
- The MIND interpreter (`mind`)
- Core library (`src/`)
- Build tooling and CI

Out of scope:

- Third-party dependencies (report to upstream)
- The proprietary `mind-runtime` (separate policy)
- Example code and documentation

## Threat Model

`mindc` runs on a single operator's machine. The primary **untrusted inputs** are
third-party MIND source and compiled artifacts (`mic@3` bundles + their embedded
`evidence_chain` MAP). Those are the surfaces we harden and the ones we most want
vulnerability reports about:

- **Untrusted bundle / artifact parsing.** The `mic@3` reader and evidence-chain
  verifier are bounds-checked and fail **closed** on malformed input; a parser DoS,
  out-of-bounds read, or unbounded allocation on a crafted artifact is in scope.
- **Evidence-chain integrity, not authenticity (yet).** The embedded evidence chain
  (RFC 0016) is **tamper-evident**: `trace_hash` is a SHA-256 over the canonical
  `mic@3` bytes, and `mindc verify <artifact>` recomputes and checks it. It is
  **not** cryptographically signed yet — Ed25519 / ML-DSA signing is a pending
  milestone (RFC 0016 Phase C). Do **not** rely on the chain as a signature or as
  proof of *origin*; today it proves an artifact was not altered after emission
  relative to its own recorded hash, nothing about *who* produced it.
- **Not a sandbox.** Compiling or running third-party MIND does not sandbox it; run
  untrusted programs in your own isolation (see Security Best Practices above).

## Contact

- Security issues: `security@star.ga`
- General inquiries: `info@star.ga`
- Repository owner: [@star-ga](https://github.com/star-ga)
