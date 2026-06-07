# RFC 0020: mind-bench Public Reproducibility Harness

| Field | Value |
|---|---|
| RFC | 0020 |
| Title | mind-bench Public Reproducibility Harness |
| Status | **Partial** — internal gate (§10) first slice shipped; public CLI (§3) + signing (§5.3) pending std-crypto |
| Authors | STARGA Inc. |
| Created | 2026-05-25 |
| Related | RFC 0006 (mind-blas Q16.16 cross-arch baseline), RFC 0013 §8 (FFI discipline + refuse-list), RFC 0014 (per-substrate lowering tiers), RFC 0015 (cross-substrate bit-identity proof obligation), RFC 0016 (#288 evidence-chain MAP key — wedge-score embedding site), RFC 0019 (#294 agent-state-replay reference workload — future) |

---

## 1. Motivation

The wedge claim — Q16.16 cross-substrate bit-identity, deterministic agent
traces, byte-equal MLIR outputs across AVX2 / NEON / future deterministic
tensor cores — is MIND's strongest single differentiator. No competing
project (IREE, OpenXLA, Triton, Mojo, PyTorch, JAX) makes a cross-substrate
bit-identity claim at the dialect level; JAX/XLA explicitly disclaims it.

Today the claim is **whitepaper-only**. There is no public, one-command
verification path. RFC 0015 defines the proof obligation, RFC 0014 the tiers,
`tests/blas_vec_q16_smoke.rs` closes the Linux↔Windows CPU half — but all of it
lives **inside the mind repo test surface**, behind clone + Rust toolchain +
`cargo test`. That is the proof surface for STARGA engineers, not for the public.

Backed by two rounds of independent cross-review:
- **First round, evidence-chain convergence** — without a publicly
  reproducible artifact, the bit-identity claim sits in the same epistemic
  bucket as unreproducible benchmark numbers; it cannot enter a downstream
  auditor's evidence chain.
- **Second round, convergence on the missing deliverable** — a public
  wedge-proving harness running identical kernels under mindc and Rust -O2/-O3
  with a byte-diff report; a one-command `mind-verify ./blas.matmul`, else the
  claim stays a whitepaper assertion; `mind-bench` runs identical workloads on
  PyTorch/JAX/MIND, publishes bitwise diffs, and a "wedge score" 0–100.

`mind-bench` converts the internal property into a public one.

**Failure mode if shipped without this:** the wedge claim stays whitepaper-only;
adoption stalls at "interesting language, unverified property"; regulated-industry
buyers cannot cite the property in compliance docs; RFC 0013 §8 + RFC 0015 look
like internal-test hygiene rather than the load-bearing property they are.

## 2. Non-goals

- **Not a replacement for criterion micro-benches.** The `<±2%` frontend µs drift
  gate (RFC 0013 §5, Phase 15) stays in the existing criterion harness.
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

Single binary at `bin/mind-bench`, built via `mindc build --release`. No Python,
no shell wrappers, no Docker-only path.

| Subcommand | Purpose | Exit |
|---|---|---|
| `mind-bench list` | enumerate workloads (`name substrate-coverage hash-prefix`) | 0 |
| `mind-bench run <w>` | execute one; print computed hash + timing | 0 ok / nonzero on workload error |
| `mind-bench verify <w>` | execute + compare against published per-substrate reference hash | 0 on byte-equality; nonzero + diff on mismatch; nonzero on missing reference |
| `mind-bench verify --all` | verify every workload the host substrate supports | 0 iff all pass |
| `mind-bench compare <w> --against <pytorch\|jax\|rust-O2\|rust-O3>` | byte-diff report vs baseline | 0 always (informational) |
| `mind-bench wedge-score` | emit 0–100 metric (§9) for host substrate | 0 always |
| `mind-bench wedge-score --signed` | same, Ed25519-signed receipt | 0 always |
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
(per-substrate sha256, signed), `README.md`. This is **the same manifest** consumed
by RFC 0015's `tests/cross_substrate_identity/` — single source of truth, two consumers.

## 5. Reference-hash storage

Per-substrate reference hash is the load-bearing artifact: tamper-evident,
version-anchored, retrievable without network at verify time.

- **Location:** `mind-spec/wedge-reference-hashes/<mindc-version>.txt`, committed (not
  release-artifact-only) so a shallow clone at a tag carries the proof bundle;
  duplicated into the Docker image for the network-free path.
- **Format:** `<workload-id> <substrate-id> <mindc-version> <sha256> <ed25519-sig>` per
  line. Per RFC 0015 §3.1, `avx2` and `neon` MUST share the same content hash for
  Q16.16 workloads (the same hash on both lines, substrate-specific signatures) —
  cross-substrate bit-identity made inspectable in the manifest.
- **Signing:** each line signed with the STARGA mindc-release Ed25519 key (the same
  key chain as mind-mem `model_signing` — single anti-fragmentation key chain). Public
  half at `mind-spec/signing-keys/mindc-release-<keyid>.pub`, fingerprint pinned in
  `docs/security/signing-keys.md`. RFC 0015's canonical CI runner produces + signs at
  release-tag time; no other path admitted.
- **Versioning:** a new `<version>.txt` ships every mind release tag; older files
  remain forever (regression archaeology); never rewritten in place. A workload added
  after a release first appears in the next release's manifest (monotone-add).

## 6. Docker image — `starga/mind-bench`

The one-command reviewer path: `docker run starga/mind-bench verify matmul-q16-1024`
→ signed pass/fail in under one second. Contents: static `mind-bench` binary, full
`wedge-reference-hashes/` tree + public key, full `workloads/` tree, and for `compare`
the pinned PyTorch/JAX/Rust toolchains as layered tags
(`:pytorch-2.4`, `:jax-0.4.30`, `:rust-1.81`). Dockerfile committed at
`docker/mind-bench/Dockerfile`; published image SHA recorded per release; the
mind-bench binary inside is itself byte-identical across rebuilds. Not the only
channel — also `cargo install`, pre-built static binaries per GitHub Release
(Linux musl / macOS universal / Windows MSVC), and `mindc build --release` post
self-host distribution.

## 7. Receipt format

Text (default) and JSON (`--format json`). JSON schema:
```json
{ "schema_version": 1, "workload": "matmul-q16-1024",
  "substrate": {"id":"avx2","detected_cpu":"...","override_used":false},
  "mindc_version":"0.7.0","mind_bench_version":"0.7.0",
  "hash_computed":"...","hash_expected":"...","result":"match",
  "byte_diff_offset": null, "duration_ms": 47,
  "duration_note":"informational, not load-bearing",
  "signature":"...","signing_key_id":"...","signing_key_fingerprint":"..." }
```
On mismatch, `result:"mismatch"`, `byte_diff_offset` = first divergence offset +
a 16-byte hex window; full diff to `--out` if larger than a fixed cap. Per RFC 0016
the signed JSON receipt is the value of the `evidence.wedge_score` MAP key — a
first-class evidence-chain artifact for downstream agent surfaces.

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
Weights live in `manifest.toml`, normative, change only via RFC amendment. The
`wedge-score --signed` receipt is itself byte-identical across runs (the tool that
measures the wedge is part of the wedge); it populates RFC 0016's `evidence.wedge_score`.

## 10. CI integration

`cargo test --workspace --no-fail-fast` (the standing full-workspace-no-features gate)
does NOT run `mind-bench verify` — the bench is external by design. The internal gate
remains `tests/cross_substrate_identity/` (RFC 0015 §5.1), which produces the **same**
hashes the reference manifest publishes at release. A separate workflow
`.github/workflows/mind-bench.yml` builds mind-bench, runs `verify --all` across the
runner matrix (Linux x86_64 AVX2 / Linux aarch64 NEON / macOS arm64 NEON / Windows MSVC
AVX2), and on tag regenerates + signs `wedge-reference-hashes/<version>.txt`, publishes
the Docker image, and attaches the signed wedge-score to the GitHub Release. A
`verify --all` failure post-tag is a release-blocker. Per RFC 0015 §5.3, unavailable
substrate = deferred (`null`), never = pass.

## 11. Timeline (6 weeks, 3 phases)

- **W1-2 Surface + format:** CLI skeleton; workload-dir + reference-hash file formats;
  Ed25519 signing wired (reusing model_signing); `matmul-q16-1024` end-to-end; receipt
  text + JSON.
- **W3-4 Suite + verify:** remaining 7 Phase-1 workloads; `verify --all` on AVX2 host;
  Docker image; first public docker-pull-and-verify green; `mind-bench.yml` CI.
- **W5-6 Comparison + wedge-score + release:** PyTorch/JAX/Rust subprocess comparison;
  `compare` diff schema; `wedge-score` 0–100 + per-substrate; first public mindc release
  ships signed `<version>.txt` + Docker tag + wedge-score receipt; mindlang.dev cites the
  one-command verify path.

## 12. Acceptance

1. `mind-bench verify matmul-q16-1024` < 1s on the reference Linux x86_64 AVX2 host,
   signed receipt, exit 0 on byte-equality.
2. Same on macOS arm64 (NEON) + Windows MSVC (AVX2), identical content hashes (RFC 0015).
3. `docker run starga/mind-bench verify matmul-q16-1024` works on a clean host.
4. `wedge-score --signed` → 100/100 on reference host for the Phase-1 suite.
5. `compare matmul-q16-1024 --against pytorch` shows non-zero `byte_diff_count` AND
   non-zero `across_run_baseline_drift_count` (wedge visible: MIND deterministic, PyTorch not).
6. `mind-spec/wedge-reference-hashes/0.7.0.txt` exists, every line Ed25519-signed, public
   key in `mind-spec/signing-keys/`.
7. CI gates the release tag on `verify --all`.
8. mindlang.dev cites the one-command verify path on the front page.

## 13. Open questions

- **Wedge-score deterministic across runs of mind-bench itself?** YES — §9.4 makes it
  normative; the bench tool's own output byte-equality is a `tests/cross_substrate_identity/`
  case.
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
- **Does mind-bench gate the ±2% frontend µs drift?** No — different property, different
  surface; the criterion gate stays independent.

## 14. Risks

- **Wedge-score gaming** (trivially-matching workload inflates score) — mitigated:
  workload-add is RFC-amendment, weights normative, manifest publicly auditable.
- **Docker dependency for the lowest-friction path** — pre-built static binary path
  (§6) is first-class, both exercised in CI per release.
- **Comparison-baseline rot** — §13 lifetime-of-release pin; refresh is mindc-release work.
- **Signing-key compromise** — same lifecycle as mind-mem model_signing key (rotation /
  revocation / sunset); one key chain.
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
reference hashes); `mind-mem model_signing` (Ed25519 reused for §5.3); internal cross-review round 1
(evidence-chain) + round 2 (missing-deliverable), both 2026-05-24.
