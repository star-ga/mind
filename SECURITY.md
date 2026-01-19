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

## Contact

- Security issues: `security@star.ga`
- General inquiries: `info@star.ga`
- Repository owner: [@star-ga](https://github.com/star-ga)
