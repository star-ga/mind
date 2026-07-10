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
- **Evidence-chain integrity + opt-in authenticity.** The embedded evidence chain
  (RFC 0016) is **tamper-evident** by default: `trace_hash` is a SHA-256 over the
  canonical `mic@3` bytes, and `mindc verify <artifact>` recomputes and checks it.
  `trace_hash` authenticates those **canonical (type-erased) IR bytes**: two source
  programs that differ *only* in a scalar type they never observe — e.g. a
  pass-through `fn f(x: i64) -> i64 { x }` vs `fn f(x: f64) -> f64 { x }`, whose
  bodies carry no type-dependent instruction — encode to identical `mic@3` and so
  share a `trace_hash`. This is a **canonicalization-completeness boundary, not a
  tamper-detection failure**: a *given* artifact's bytes still cannot be altered
  without changing its hash, and any program that actually uses a scalar (arithmetic
  or width-sensitive ops) emits type-distinct instructions and does not collide.
  Authenticity is **opt-in** (RFC 0016 Phase C): an artifact may additionally carry
  a `signature.*` block — Ed25519 (RFC 8032), ML-DSA-65 (FIPS-204 PQC), or the
  hybrid — over the canonical provenance preimage. Signing is **never enabled by
  default**, and the `signature.*` keys sort *after* the hashed body so an unsigned
  artifact stays byte-identical. An **unsigned** artifact is tamper-evident but says
  nothing about *who* produced it — do not treat the bare chain as proof of origin.
  When a signature is present, verify it against a **trusted-key allowlist**
  (`mindc verify --signer-pubkey <hex>` / `MIND_EVIDENCE_VERIFY_PUBKEYS`); without
  an allowlist `mindc verify` reports the signature's presence and structural
  validity but **refuses to report it as trusted** — a valid signature by an
  unknown key is not authenticity. A crafted-signature forgery, PQC scheme-downgrade,
  or trust-anchor bypass on a signed artifact is in scope.
- **Not a sandbox.** Compiling or running third-party MIND does not sandbox it; run
  untrusted programs in your own isolation (see Security Best Practices above).

## Contact

- Security issues: `security@star.ga`
- General inquiries: `info@star.ga`
- Repository owner: [@star-ga](https://github.com/star-ga)
