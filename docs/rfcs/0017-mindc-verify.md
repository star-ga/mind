# RFC 0017: `mindc verify` — Artifact Verification Surface

| Field | Value |
|---|---|
| RFC | 0017 |
| Title | `mindc verify` — Artifact Verification Surface |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-29 |
| Task | #290 |
| Related | RFC 0016 (evidence-chain emission — the carrier being verified), RFC 0021 (mic@3 binary `IRModule` + MAP epilogue — the container format), RFC 0015 (cross-substrate bit-identity — the property `verify` discharges), RFC 0020 (mind-bench — `wedge-score` receipt is a `verify.*` sibling block) |

---

## 1. Summary

`mindc verify` is a first-party subcommand that validates a compiled MIND artifact's
embedded evidence chain. It recomputes the `trace_hash` over the artifact's IR body,
compares it to the value embedded in the MAP epilogue, and — on signed release
artifacts — validates the Ed25519 signature. The subcommand is the human-facing and
CI-facing discharge surface for the RFC 0016 evidence-chain claim.

## 2. Motivation

RFC 0016 defines what a compile-time evidence chain *asserts* and RFC 0021 ships the
mic@3 container that carries it. Without a first-party verifier, the chain has no
observable discharge surface: an auditor holding a `.mic` artifact cannot confirm that
the embedded `trace_hash` is correct, that the substrate matches, or that the
Ed25519 signature is valid. The claim stays internal.

`mindc verify` closes that gap. It turns an embedded evidence chain into a
verifiable, human- and CI-readable report — the same artifact that a downstream agent
or compliance auditor can run without cloning the MIND source tree.

This subcommand is **not** a new primitive; it is the CLI wrapper for the verifier
library core already shipped as `mic3_evidence_report` in
`src/ir/compact/v3/evidence.rs` (RFC 0021 step 2, mind@c64bd0b) and
`verify_evidence_chain` in `src/ir/compact/v2/evidence.rs` (RFC 0016 Phase B,
mind@cadca87). This RFC scopes the surface, the two verification modes (hash-equality
vs SMT-proven), and the acceptance criteria.

## 3. Guide-level explanation

After compiling with evidence emission:

```
mindc src/matmul.mind --emit-evidence out/matmul.mic3
mindc verify out/matmul.mic3
```

Expected output on a valid artifact:

```
artifact:   out/matmul.mic3
format:     mic@3 + evidence MAP
trace_hash: <64 hex chars>  [OK — matches recomputed]
substrate:  x86_avx2
toolchain:  0.7.1
determinism: deterministic
parent:     <64 hex chars>  (IR compilation step)
signature:  unsigned (local build)
result:     PASS
```

On a tampered artifact the `trace_hash` line reports `[MISMATCH]` and the exit
code is nonzero.

On a signed release artifact:

```
mindc verify --check-sig out/matmul.mic3
```

The output additionally includes:

```
signature:  Ed25519 [OK] — key-id: <fingerprint>
```

### Verification flags

| Flag | Meaning |
|---|---|
| `--check-sig` | Require and validate the Ed25519 signature (fails on unsigned) |
| `--pubkey <path>` | Provide the verification key explicitly (default: bundled STARGA release key) |
| `--format text\|json` | Output format; `json` emits a machine-readable receipt |
| `--out <path>` | Write the JSON receipt to a file |
| `--chain` | Follow `parent` pointers and verify the full DAG (requires sibling artifacts in the same directory) |

## 4. Reference-level explanation

### 4.1 Hash-equality verification (primary gate)

The verifier executes the RFC 0016 §3.2 self-reference rule:

1. Parse the mic@3 artifact; peel the MAP epilogue.
2. Extract `evidence_chain.trace_hash` from the MAP.
3. Recompute `ir_trace_hash` over the canonical mic@1 IR bytes embedded in the
   mic@3 body (`SHA-256` per FIPS 180-4, the same seam as `deps::mini_sha256`).
4. Compare the recomputed hash to the stored value byte-for-byte.
5. Optionally: validate the Ed25519 signature over MAP-minus-`signature.*` (mic@2.1
   §6 discipline, reused for mic@3).
6. Report `substrate`, `determinism`, `toolchain`, `parent` from the MAP.
7. Exit 0 on full pass; nonzero (with a diff line indicating first divergence) on
   any mismatch or signature failure.

This is the **default** and **always-available** mode. It requires only the artifact
— no reference database, no network, no second substrate.

### 4.2 Cross-substrate equality check

```
mindc verify --cross-substrate out/matmul-avx2.mic3 out/matmul-neon.mic3
```

For a Q16.16 graph, both artifacts must have the same `trace_hash` value
(RFC 0015 §3.1 inspectable identity). The verifier checks:
- Both artifacts share the same `parent` hash (same IR source step).
- The `trace_hash` values are byte-equal.
- `substrate` fields differ (expected: `x86_avx2` vs `arm_neon`).
- Both signatures valid (if `--check-sig` supplied).

Exit 0 iff all pass. This is the in-band form of the RFC 0015 cross-substrate
proof obligation, expressed over two local artifacts without running a second
toolchain.

### 4.3 SMT-proven identity (scoped future extension — §7)

The hash-equality gate is **sufficient** for the RFC 0015 claim: two artifacts
produced by `mindc` from the same source for different substrates whose `trace_hash`
values are equal have byte-identical IR, by construction of the hash. However,
hash-equality is a *sampling check* on the specific compiled output — it does not
prove that *every* input to the lowering pipeline produces the same result.

SMT-proven cross-substrate identity would demonstrate, via a symbolic model of the
lowering pipeline, that the Q16.16 lowering function is substrate-independent for
any IR input in the supported fragment. This is the strongest possible discharge of
the RFC 0015 claim. It is out of scope for the initial `mindc verify` subcommand
(§8) but is the logical extension of the `--cross-substrate` gate.

The decision point for RFC 0017 is therefore: **ship hash-equality as the primary
gate; scope SMT-proven identity as a future `--symbolic` mode**.

Rationale for this decision:

- Hash-equality is already implemented (`mic3_evidence_report`); it is available
  for CI and auditors today.
- The SMT approach requires a formal model of the MLIR lowering pipeline that does
  not yet exist; committing to it in this RFC would block the `mindc verify`
  subcommand on multi-year foundational work.
- The RFC 0020 mind-bench `wedge-score` provides an empirical, public complement to
  the SMT ideal: it runs the actual lowered binary on real hardware and compares
  output hashes. Empirical + formal is stronger than either alone.
- When the SMT path ships (a separate RFC), it can carry the same `verify.*` MAP
  block structure and be invoked via `mindc verify --symbolic`, composing cleanly.

### 4.4 The `verify.*` MAP block

A signed verification receipt can be embedded as a sibling `verify.*` MAP block
in a mic@3 artifact. This block is written by the RFC 0015 canonical CI runner at
release-tag time and is also the format used by RFC 0020's `wedge-score --signed`
receipt.

| Key | Type | Meaning |
|---|---|---|
| `verify.schema` | `uint` | Version of this block (`1`). |
| `verify.verifier_version` | `string` | `mindc` version that produced this receipt. |
| `verify.mode` | `string` | `"hash_equality"` or (future) `"cross_substrate"` / `"symbolic"`. |
| `verify.result` | `string` | `"pass"` or `"fail"`. |
| `verify.timestamp` | `uint` | Unix seconds of the verification run (informational). |
| `verify.substrate_a` / `verify.substrate_b` | `string` | Substrates compared (cross-substrate mode). |
| `verify.signature` | `bytes` | Ed25519 signature over MAP-minus-`signature.*` (mic@2.1 §6). |

The `verify.*` block is optional in the carrier; its presence signals that the
artifact has been independently re-verified by a second tool invocation (or CI pass)
and the result is embedded for downstream auditors.

### 4.5 Exit codes

| Exit | Meaning |
|---|---|
| `0` | All checks passed. |
| `1` | `trace_hash` mismatch — artifact body and embedded hash disagree. |
| `2` | Signature invalid or required but absent. |
| `3` | Cross-substrate hashes differ (unexpected for a Q16.16 graph). |
| `4` | Artifact parse error (not a mic@3 file, or truncated). |
| `5` | Parent chain walk failed (missing sibling artifact in `--chain` mode). |

### 4.6 JSON receipt schema

```json
{
  "schema_version": 1,
  "artifact": "<path>",
  "format": "mic3",
  "trace_hash_stored": "<hex>",
  "trace_hash_recomputed": "<hex>",
  "trace_hash_match": true,
  "substrate": "x86_avx2",
  "toolchain": "0.7.1",
  "determinism": "deterministic",
  "parent": "<hex or null>",
  "signature_checked": false,
  "signature_valid": null,
  "signing_key_id": null,
  "result": "pass",
  "verifier_version": "0.7.1"
}
```

On cross-substrate mode, `substrate_a`/`substrate_b` replace `substrate` and
`trace_hash_equal` is added.

## 5. Drawbacks

- Adding a CLI subcommand that wraps an existing library is low-risk but adds CLI
  surface area to maintain.
- The `--chain` parent-walk mode requires sibling artifacts to be present locally;
  a split-artifact deployment where substrates are on different machines needs a
  protocol for fetching siblings (not scoped here).
- Hash-equality does not prove general cross-substrate correctness for all inputs
  (§4.3); the SMT gap must be clearly communicated to auditors.

## 6. Rationale and alternatives

**Alternative: detached receipt file.** The receipt could be a separate `.receipt`
sidecar rather than embedded in the artifact. Rejected: RFC 0021 §2 and the
unanimous 5/5 cross-model finding (2026-05-27) establish that evidence must be
inseparable from the artifact bytes. A detached file introduces desync risk.

**Alternative: external verification tool.** Shipping `mind-bench verify` (RFC 0020)
as the only public verification path. Rejected: `mind-bench` verifies *output hashes*
against a reference manifest; `mindc verify` verifies the *artifact's own embedded
evidence* without a reference database. Both are necessary; they are complementary,
not substitutes.

**Alternative: verify as part of `mindc build` output.** Automatically verify every
emitted artifact. Rejected: verification is a read-only check on the artifact, not
a build step; conflating them obscures the audit trail. The correct model is:
`mindc build --emit-evidence` emits, `mindc verify` independently re-checks.

## 7. Prior art

- SLSA / in-toto / Sigstore: detached attestations keyed by content hash. MIND's
  embedded MAP approach (RFC 0021 §2) is superior for inseparability; the `verify.*`
  block discipline mirrors in-toto's link-metadata concept.
- WASM custom-section tooling (`wasm-opt --print-custom-sections`, `wasm-tools
  parse`): per-section inspection of embedded metadata. `mindc verify` is the MIND
  equivalent.
- LLVM `opt --verify-each`: mid-pass IR integrity checks. `mindc verify` is the
  post-compilation external equivalent.
- `cosign verify` (Sigstore): CLI verifier for OCI image signatures. Same
  separation-of-concerns pattern: a dedicated verification subcommand, not a
  build-time side-effect.

## 8. Unresolved questions

1. **SMT-proven mode design.** Which SMT solver (Z3, CVC5), what fragment of the
   MLIR lowering pipeline to model, and what the resource budget is. Scoped to a
   follow-on RFC; not a gate for this RFC.
2. **Parent-walk across repos.** If two sibling artifacts live in different repos or
   registries, `--chain` needs a fetch protocol. Deferred; local artifact set is
   sufficient for CI use.
3. **Key rotation / revocation UX.** `--pubkey` accepts a PEM file; how a verifier
   discovers the correct key for an artifact produced by an older `mindc` release is
   covered by RFC 0020 §5 (per-version key archive) but not fully specified here.

## 9. Future possibilities

- **`mindc verify --symbolic`**: SMT-proven cross-substrate identity gate once a
  formal lowering model exists.
- **`mindc verify --chain` with remote fetch**: follow parent pointers across a
  content-addressed artifact registry.
- **`verify.*` block as a CI badge input**: the JSON receipt populates a
  `shields.io`-style badge on mindlang.dev (linked to RFC 0020 §12.8).
- **Governance integration**: 512-mind DIFC `proof_chain` records the
  `trace_hash` from a `mindc verify` pass as a proof input (RFC 0016 §7 / RFC 0021
  §3.4 convergence deliverable).

## 10. Acceptance

1. `mindc verify <artifact>` exits 0 on a valid mic@3 artifact and nonzero on a
   tampered one.
2. A single flipped byte anywhere in the artifact body causes `trace_hash`
   mismatch and exit 1.
3. `mindc verify --check-sig <artifact>` exits 2 on an unsigned artifact.
4. `mindc verify --cross-substrate <avx2> <neon>` exits 0 for two Q16.16
   artifacts produced from the same source, exits 3 if their `trace_hash` values
   differ.
5. `--format json` emits a receipt conforming to §4.6.
6. The subcommand reuses `mic3_evidence_report` and `verify_evidence_chain` from
   the shipped library core; no new hash or signing primitive is introduced.
7. Phase B of RFC 0016 (`mindc verify --evidence`) is satisfied by this subcommand
   landing; the two specs are consistent.

## 11. References

RFC 0015 §3.1 (cross-substrate bit-identity — the property this verifier discharges);
RFC 0016 §3.2 (self-reference rule — hash computation the verifier replicates),
§6 (Ed25519 reused), Phase B (verifier core now in `src/ir/compact/v2/evidence.rs`);
RFC 0020 §5 (per-version reference hash archive), §7 (receipt JSON schema, basis for
`verify.*` block);
RFC 0021 §4.4 (`mindc verify --evidence` CLI step, mind@7fc10d2 precedent),
`src/ir/compact/v3/evidence.rs::mic3_evidence_report` (shipped library core),
`src/ir/compact/v2/evidence.rs::verify_evidence_chain` (shipped verifier);
mic@2.1 §6 (Ed25519 signing discipline reused for mic@3).
