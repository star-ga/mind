# RFC 0020: mind-bench Public Reproducibility Harness

| Field | Value |
|---|---|
| RFC | 0020 |
| Title | mind-bench Public Reproducibility Harness |
| Status | **Partial** — internal gate (§10) shipped; compiler evidence signing is available **opt-in** with the PQC hybrid (§5.3). Public `mind-bench` CLI, signed reference/receipt integration, and operational release signing remain pending. |
| Authors | STARGA Inc. |
| Created | 2026-05-25 |
| Related | RFC 0006 (mind-blas Q16.16 cross-arch baseline), RFC 0013 §8 (FFI discipline + refuse-list), RFC 0014 (per-substrate lowering tiers), RFC 0015 (cross-substrate bit-identity proof obligation), RFC 0016 (#288 evidence-chain MAP key — wedge-score embedding site), RFC 0019 (#294 agent-state-replay reference workload — future) |

---

> **Status correction (2026-09-13).** The internal harness exists:
> `tests/cross_substrate_identity.rs` has 26 tests over an inventory of 25
> workload manifests, with asserting x86/ARM CI and separate coverage receipts
> (RFC 0015 §5A). The public CLI below is still a specification. The compiler's
> ML-DSA-87 + SLH-DSA-SHAKE-256s signing capability is implemented behind
> `evidence-mldsa` + `evidence-slhdsa`, both off by default; this does **not**
> mean public benchmark receipts or STARGA releases are signed. Offline key
> custody, publication of a trust anchor and the signed-reproduction release
> gate remain open in `docs/roadmap.md`, Phase 19.1.

## 1. Motivation

The property this harness exposes is **bit-identical computational output**
for a defined workload and inputs across supported substrates and machines,
with attributable evidence when signing is enabled. RFC 0015 §5A records
the current Q16.16, exact-integer and scoped strict f32/f64 coverage across
AVX2 and NEON. Target-specific MLIR and native instruction bytes may differ;
they are not the cross-ISA comparison subject. Additional substrates and
workloads need their own execution evidence before joining the claim.

The property has an implemented in-repository gate, but there is no public,
standalone one-command verification path. RFC 0015 defines the proof
obligation and RFC 0014 the tiers. Today verification requires the repository,
a Rust toolchain and the MLIR execution tools. This RFC packages that existing
proof surface for evaluators who should not need to build the compiler.

Backed by two rounds of independent cross-review:
- **First round, evidence-chain convergence** — without a publicly
  reproducible artifact, the bit-identity claim sits in the same epistemic
  bucket as unreproducible benchmark numbers; it cannot enter a downstream
  auditor's evidence chain.
- **Second round, convergence on the missing deliverable** — a public
  wedge-proving harness running identical kernels under mindc and Rust -O2/-O3
  with a byte-diff report; a one-command `mind-verify ./blas.matmul`, else the
  public verification remains inconvenient; `mind-bench` is proposed to run identical workloads on
  PyTorch/JAX/MIND, publishes bitwise diffs, and a "wedge score" 0–100.

`mind-bench` converts the internal property into a public one.

**Missing deliverable:** independent evaluators must currently reconstruct the
repository test environment. A standalone verifier should make the workload,
inputs, output encoding, reference provenance and tested coverage inspectable
without a source migration. It does not make a foreign repository's arbitrary
computations deterministic.

## 2. Non-goals

- **Not a replacement for criterion micro-benches.** The one-sided `+10%` frontend µs
  regression gate (RFC 0013 §5, Phase 15) stays in the existing criterion harness.
  mind-bench is reproducibility, not regression.
- **Not a throughput claim.** mind-bench does NOT claim MIND is faster than
  PyTorch/JAX/Rust. Timing in receipts is informational, marked as such.
- **Not NeurIPS-paper statistical rigor.** Fixed inputs/seeds, exact-byte-match.
  The bar is "every byte matches, every run, every substrate" — binary, not
  statistical.
- **Not a GPU vendor benchmark.** Per RFC 0013 §8 it does NOT compare against
  cuBLAS default mode, cuDNN, oneDNN default, OpenBLAS default — comparing
  byte-equality against a nondeterministic baseline is a category error.
- **Not a substitute for `tests/cross_substrate_identity/` (RFC 0015 §5).** That
  matrix gates merge; mind-bench consumes the same workload manifest after a
  release ships. Internal gate stays internal; public proof goes public.

## 3. Surface — the `mind-bench` binary CLI

**Planned surface, not an installed command.** Single binary at `bin/mind-bench`, built via `mindc build --release`. No Python,
no shell wrappers, no Docker-only path.

| Subcommand | Purpose | Exit |
|---|---|---|
| `mind-bench list` | enumerate workloads (`name substrate-coverage hash-prefix`) | 0 |
| `mind-bench run <w>` | execute one; print computed hash + timing | 0 ok / nonzero on workload error |
| `mind-bench verify <w>` | execute + compare against published per-substrate reference hash | 0 on byte-equality; nonzero + diff on mismatch; nonzero on missing reference |
| `mind-bench verify --all` | verify every workload the host substrate supports | 0 iff all pass |
| `mind-bench compare <w> --against <pytorch\|jax\|rust-O2\|rust-O3>` | byte-diff report vs baseline | 0 always (informational) |
| `mind-bench wedge-score` | emit 0–100 metric (§9) for host substrate | 0 always |
| `mind-bench wedge-score --signed` | same, opt-in PQC-hybrid-signed receipt (§5.3) | 0 on successful signing; nonzero if signing is unavailable or fails |
| `mind-bench --version` | mind-bench + linked mindc + reference-manifest versions | 0 |

Global flags: `--substrate <id>` (override detection, testing only), `--out <path>`,
`--format text|json`, `--no-network` (default already no-network), `--mindc-version <ver>`
(verify against a historical release's reference hashes).

Deliberately absent: `install` (binary ships standalone), `upload` (no telemetry,
local-only), `tune` (no autotuner — tuning fractures byte-identity).

## 4. Workload suite

Each workload is `(input_bytes, expected_output_bytes, reference_hash)` in a fixed
deterministic format.

### 4.1 Phase-1 workloads

This is the proposed public suite, not the inventory of already gated
fixtures. RFC 0015 §5A and `tests/cross_substrate_identity/*/manifest.toml`
enumerate the existing evidence; adding a public workload requires a real
kernel, inputs, output encoding and reference on each claimed substrate.

| ID | Description | Substrates | Weight |
|---|---|---|---|
| `matmul-q16-32` | 32×32 Q16.16 gemm, seed 0xDEADBEEF | AVX2, NEON | 1.0 |
| `matmul-q16-256` | 256×256 Q16.16 gemm | AVX2, NEON | 2.0 |
| `matmul-q16-1024` | 1024×1024 Q16.16 gemm | AVX2, NEON | 4.0 |
| `dot-l2-q16` | L2 dot reduction, len 65536 | AVX2, NEON | 1.0 |
| `dot-l1-q16` | L1 dot reduction | AVX2, NEON | 1.0 |
| `dot-linf-q16` | L∞ (max-abs) reduction | AVX2, NEON | 1.0 |
| `encoder-forward-pass` | mind-nerve reference encoder, Q16.16 | AVX2, NEON | 3.0 |
| `bitnet-ternary-matmul-256` | RFC 0001 ternary primitive | AVX2, NEON | 2.0 |

### 4.2 Future workloads (named for sequencing)
`agent-state-replay-{small,medium,large}` (gated on RFC 0019), `training-step-rfn-mind`
(rfn-mind R3.x), `conv-q16-2d-3x3` (RFC 0015 §3.2 Tier 3), `reduce-q16-max` (`.max` work).

### 4.3 Workload file format
`workloads/<id>/`: `manifest.toml` (name/version/weight/coverage/desc), `input.bin`,
`expected_output.bin` (cross-check, not load-bearing), `reference_hashes.toml`
(per-substrate sha256, signed once §5 is implemented), `README.md`. Public
packaging MUST consume RFC 0015's in-tree workload definitions as its source
of truth. This directory layout and signature transport are planned; the
current internal references are unsigned committed TOML files.

## 5. Reference-hash storage

The output reference hash is the load-bearing comparison value. Signing
authenticates its declared provenance; it does not replace execution of the
workload or prove the signer's claim correct.

### 5.1 Current storage and planned distribution

The shipped internal references are
`tests/cross_substrate_identity/<id>/reference_hashes.toml`. They are committed
hashes with recorded substrate provenance, **not signed release manifests**.
The proposed public distribution under
`mind-spec/wedge-reference-hashes/<mindc-version>` must bundle the references,
workload/input/encoding identities and verification material for offline use.
The final public file format and receipt integration remain implementation
work; the former Ed25519-per-line format is retired from this specification.

### 5.2 Content and version binding

The signed payload MUST bind the workload and input identities, canonical
output encoding, substrate coverage, compiler/reference-manifest versions
and expected output hash. Per RFC 0015 §3.1, covered `avx2` and `neon`
computations MUST share the same output hash, while their substrate provenance
may differ. A signature over an unbound hash string is insufficient.

Published versioned manifests and release tags are immutable. A correction
requires a new version; old manifests and trust anchors remain available for
verification of historical artifacts. References must never be silently
regenerated or re-signed to make an unexpected output mismatch pass (§13).

### 5.3 Signing profile: PQC hybrid, opt-in

The required profile is **ML-DSA-87 + SLH-DSA-SHAKE-256s**, using the existing
scheme identifier `pqc-hybrid-ml-dsa-87-slh-dsa-256s`. Its combiner is **AND**:
both signatures over the same canonical payload must verify. The planned
release harness must pin the authorized two-key tuple from its published
trust anchor. Missing, malformed, invalid or unsupported
legs fail closed. The harness MUST NOT fall back to a single leg, standalone
ML-DSA-65, Ed25519, the retired Ed25519/ML-DSA-65 hybrid, or unsigned output
when this signed profile is required.

The compiler already implements the opt-in evidence signing/verification
capability in `src/ir/compact/v3/evidence.rs`, `mldsa.rs`, `slhdsa.rs`, and
`src/bin/mindc.rs`. Both Cargo features `evidence-mldsa` and `evidence-slhdsa`
are required and are off by default. The release build recipe enables both;
that establishes capability in that build profile, not a signed release.
`tests/pqc_hybrid_cli_signing.rs` covers CLI signing, both-key pinning,
tampering, incomplete configuration and retired schemes. The library control
`pqc_hybrid_non_degradable` strips each actual signature leg and requires
rejection. Existing `mindc verify --require-signed` and repeated
`--signer-pubkey` arguments support requiring a signature and checking both
hybrid keys against an operator-supplied allowlist. That allowlist is not a
published STARGA trust anchor, and the current wire format does not establish
a key-pair identity. The public harness must additionally enforce the exact
profile and authorized tuple described above.

Reuse this implementation and its canonical provenance binding; do not
introduce a second cryptographic implementation or assume a mind-mem key can
sign compiler releases. Existing MIC evidence signing is not yet an API for
signing arbitrary benchmark JSON or reference files. Binding those payloads
to the evidence format remains part of the public harness integration.

### 5.4 Operational release signing remains pending

Follow `docs/roadmap.md`, Phase 19.1: offline key generation and custody,
publication of the trust anchor, then a signed-reproduction release gate.
CI prepares and independently reproduces artifacts; the operator signs
offline; finalization verifies the signatures against the published anchor
and checks the reproduced bytes before publication. Release signing keys
must not be placed in the benchmark binary, public image or ordinary CI job.

Until that process and the first release verification are complete, describe
the capability as **"signed (opt-in)"**. Do not claim STARGA releases, public
benchmark receipts, or reference bundles are already signed.

## 6. Docker image — `starga/mind-bench`

The planned one-command reviewer path: `docker run starga/mind-bench verify matmul-q16-1024`
→ output-identity verification against authenticated references. The execution
time target is an acceptance goal, not a measured result; generating a signed
local receipt is a separate opt-in operation requiring the caller's signing
configuration. No signing seeds ship in the image. Contents: static `mind-bench` binary, full
`wedge-reference-hashes/` tree + public key, full `workloads/` tree, and for `compare`
the pinned PyTorch/JAX/Rust toolchains as layered tags
(`:pytorch-2.4`, `:jax-0.4.30`, `:rust-1.81`). Dockerfile committed at
`docker/mind-bench/Dockerfile`; published image SHA recorded per release; the
mind-bench binary inside is itself byte-identical across rebuilds. Not the only
channel — also `cargo install`, pre-built static binaries per GitHub Release
(Linux musl / macOS universal / Windows MSVC), and `mindc build --release` post
self-host distribution.

## 7. Receipt format

Planned text (default) and JSON (`--format json`) receipt. This unsigned example
is illustrative, not a shipped or finalized wire schema:
```json
{ "schema_version": 1, "workload": "matmul-q16-1024",
  "substrate": {"id":"avx2","detected_cpu":"...","override_used":false},
  "mindc_version":"0.7.0","mind_bench_version":"0.7.0",
  "hash_computed":"...","hash_expected":"...","result":"match",
  "byte_diff_offset": null, "duration_ms": 47,
  "duration_note":"informational, not load-bearing",
  "signature": null }
```
On mismatch, `result:"mismatch"`, `byte_diff_offset` = first divergence offset +
a 16-byte hex window; full diff to `--out` if larger than a fixed cap. Per RFC 0016
the receipt is intended for the `evidence.wedge_score` MAP key. Its signed
form must use §5.3's exact scheme, carry both signatures and identify both
trusted public keys, with a defined canonical payload binding all claimed
results. That serialization and MAP integration remain pending.

Computational-output identity is distinct from complete receipt-byte identity.
Substrate metadata, measured duration, versions and signer identity may differ.
Deterministic signing reproduces signature bytes for the same complete payload
and keys; it cannot make different receipt payloads identical. Any canonical
score comparison must define its stable fields and keep observational metadata
explicitly scoped. Neither signature validity nor an output match is a
throughput or SOTA score.

## 8. Competitor-comparison harness

`compare` runs identical workloads on PyTorch/JAX/Rust/MIND and emits bytewise diffs.
The point is not "is MIND faster" but "is MIND byte-identical where the others diverge".

**Hard constraint (RFC 0013 §8):** mind-bench does NOT FFI to the competitors. It is a
deterministic MIND binary; admitting Python C-API / ATen / cuBLAS would violate its own
determinism boundary. **Resolution:** comparisons run as **separate subprocesses** via
`std.process` (libc — allowed). The baseline subprocess writes output bytes to a file;
mind-bench reads, hashes externally, compares. The baseline's nondeterminism is expected
and is the point. Subprocesses are hermetically pinned via layered Docker tags; on a
non-Docker host, `compare` refuses with a pointer to the Docker path. One PyTorch version
per mind release, pinned in `mind-spec/comparison-baselines.toml`, no auto-bumping.

Diff schema includes `byte_diff_count`, `first_diff_offset`, `max_magnitude_diff`,
`across_run_baseline_drift_count/max` (baseline run twice → how much it diverges from
*itself*), `mind_across_run_drift_count/max` (always 0). `compare` exits 0 always — the
baselines are not on trial.

## 9. The wedge-score metric

For host substrate H, workload set W with weights w:
```
wedge_score(H) = 100 * Σ_i [ w(W_i) * I(hash_i(H) == reference_hash_i(H)) ] / Σ_i w(W_i)
```
0 = no workload matches; 100 = every workload matches on every shipped substrate the
host can run. Per-substrate sub-scores reported; the top-level score is the **minimum**
of sub-scores (the wedge holds only if it holds everywhere it claims). A substrate the
host cannot run is `null` (deferred), never 0 — zeros are reserved for tested-and-mismatched.
Weights live in `manifest.toml`, normative, change only via RFC amendment.
An empty executed workload set must not yield 100 or a successful verification.
This score describes the declared identity checks only. The deterministic
score payload must be distinguished from duration and host metadata (§7);
the complete signed receipt is byte-identical only when its full payload and
signing keys are identical. `evidence.wedge_score` integration remains planned.

## 10. CI integration

**Shipped internal gate:** `.github/workflows/ci.yml` runs
`tests/cross_substrate_identity.rs` on `ubuntu-24.04` (`avx2`) and
`ubuntu-24.04-arm` (`neon`), with `MIND_BENCH_REQUIRE=1`. The
`cross_substrate_receipts` target verifies the manifest-to-execution receipt
inventory. See RFC 0015 §5A for the covered workloads and deferred cases.
Missing toolchains fail; bless-mode collection is separate from asserting
verification and is not proof of a pass.

**Planned public integration:** `cargo test --workspace --no-fail-fast` (the standing full-workspace-no-features gate)
does NOT run `mind-bench verify` — the bench is external by design. The internal gate
remains `tests/cross_substrate_identity/` (RFC 0015 §5.1), which produces the **same**
hashes the reference manifest publishes at release. A separate workflow
`.github/workflows/mind-bench.yml` builds mind-bench, runs `verify --all` across the
runner matrix (Linux x86_64 AVX2 / Linux aarch64 NEON / macOS arm64 NEON / Windows MSVC
AVX2). Publication of signed references and receipts must follow §5.4's
offline-signing and verification process; a tag-triggered CI job does not hold
the signing seeds or silently regenerate reference hashes. A
`verify --all` failure post-tag is a release-blocker. Per RFC 0015 §5.3, unavailable
substrate = deferred (`null`), never = pass.

## 11. Timeline (6 weeks, 3 phases)

- **W1-2 Surface + format:** CLI skeleton; workload-dir + reference-hash file formats;
  PQC hybrid integration via §5.3's existing evidence implementation;
  `matmul-q16-1024` end-to-end; receipt
  text + JSON.
- **W3-4 Suite + verify:** remaining 7 Phase-1 workloads; `verify --all` on AVX2 host;
  Docker image; first public docker-pull-and-verify green; `mind-bench.yml` CI.
- **W5-6 Comparison + wedge-score + release:** PyTorch/JAX/Rust subprocess comparison;
  `compare` diff schema; `wedge-score` 0–100 + per-substrate; first public mindc release
  ships a signed versioned reference bundle + Docker tag + wedge-score receipt;
  publication depends on §5.4, not merely elapsed time. mindlang.dev cites the
  one-command verify path.

## 12. Acceptance

1. `mind-bench verify matmul-q16-1024` < 1s on the reference Linux x86_64 AVX2 host,
   exit 0 on byte-equality against authenticated references. This is a pending
   performance target; opt-in receipt signing is measured separately.
2. Same on macOS arm64 (NEON) + Windows MSVC (AVX2), identical content hashes (RFC 0015).
3. `docker run starga/mind-bench verify matmul-q16-1024` works on a clean host.
4. `wedge-score --signed` → 100/100 on reference host for the Phase-1 suite.
5. `compare matmul-q16-1024 --against pytorch` reports measured byte differences
   and across-run drift, including zero when the baseline matches. A passing
   comparison must not depend on forcing another implementation to diverge.
6. The versioned public reference bundle exists and verifies under §5.3's
   exact PQC hybrid against both published trusted keys. Missing or stripped
   legs, wrong keys, tampering and downgrade attempts fail closed. This
   acceptance item remains open until §5.4 is operational.
7. CI gates the release tag on `verify --all`.
8. mindlang.dev cites the one-command verify path on the front page.

## 13. Open questions

- **Wedge-score deterministic across runs of mind-bench itself?** The planned
  canonical score payload must be deterministic for identical declared inputs.
  Complete receipts include observational metadata (§7); a dedicated public
  harness gate must be implemented before claiming its own byte-identity.
- **Workload diverges on Windows MSVC but matches Linux/macOS?** Release-blocker, not a
  bench failure — a bug in mindc lowering / std-surface / libc-syscall / input-generator.
  During dev the receipt (substrate, version, byte-diff offset, OS) drives iteration. A
  `--investigate` mode (intermediate-tensor + per-MLIR-stage hashes) is Phase 4, post-release.
- **PyTorch comparison pin?** Latest stable at branch time, pinned for the lifetime of the
  mindc release, no auto-bumping. Same for JAX. Rust pinned by `rustc` version; `-O2`/`-O3`
  is a flag.
- **Reference hash changes across mindc releases?** Expected only on an intentional
  lowering-pipeline change; RFC-amendment-worthy, documented in release notes with the
  specific transition. Unexpected changes are caught at the internal gate pre-release.
- **Does mind-bench gate the +10% frontend µs regression?** No — different property, different
  surface; the criterion gate stays independent.

## 14. Risks

- **Wedge-score gaming** (trivially-matching workload inflates score) — mitigated:
  workload-add is RFC-amendment, weights normative, manifest publicly auditable.
- **Docker dependency for the lowest-friction path** — pre-built static binary path
  (§6) is first-class, both exercised in CI per release.
- **Comparison-baseline rot** — §13 lifetime-of-release pin; refresh is mindc-release work.
- **Signing-key compromise** — follow the release custody, rotation, revocation
  and archival process in `docs/roadmap.md`, Phase 19.1. Shared implementation
  does not imply shared private keys across products.
- **CI runner heterogeneity** — runner image SHA pinned per release (RFC 0015 §5.2);
  bumps are RFC-amendment-worthy.
- **Workload-format breaking change** — `version="1"` explicit; v2 coexists, not replaces.

## 15. References

RFC 0006 §5.2 (mind-blas Q16.16 cross-arch, source of `matmul-q16-1024`); RFC 0013 §8
(FFI discipline — cuBLAS/NumPy/Python-C-API forbidden, libc + `std.process` subprocess
allowed); RFC 0014 (per-substrate lowering tier IDs); RFC 0015 (cross-substrate bit-identity
proof obligation — same manifest, two consumers); RFC 0016 (#288 `evidence.wedge_score`
MAP key); RFC 0019 (#294 agent-state-replay future workload); `tests/blas_vec_q16_smoke.rs`
(in-tree precedent); `tests/cross_substrate_identity/` (internal gate producing the
reference hashes); `src/ir/compact/v3/evidence.rs`, `mldsa.rs`, `slhdsa.rs` and
`tests/pqc_hybrid_cli_signing.rs` (opt-in PQC capability); `docs/roadmap.md`,
Phase 19.1 (pending operational signing); internal cross-review round 1
(evidence-chain) + round 2 (missing-deliverable), both 2026-05-24.
