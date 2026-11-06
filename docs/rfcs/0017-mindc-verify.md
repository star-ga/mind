# RFC 0017: `mindc verify` — Artifact Verification Surface

| Field | Value |
|---|---|
| RFC | 0017 |
| Title | `mindc verify` — Artifact Verification Surface |
| Status | **Partially shipped** — `trace_hash` verification + SSA well-formedness SHIPPED; full SMT-proven verification is future work |
| Authors | STARGA Inc. |
| Created | 2026-05-29 |
| Task | #290 |
| Related | RFC 0016 (evidence-chain emission — the carrier being verified), RFC 0021 (mic@3 binary `IRModule` + MAP epilogue — the container format), RFC 0015 (cross-substrate bit-identity — the property `verify` discharges), RFC 0020 (mind-bench — `wedge-score` receipt is a `verify.*` sibling block) |

---

## 1. Summary

`mindc verify` is a first-party subcommand that validates a compiled MIND artifact's
embedded evidence chain. It recomputes the `trace_hash` over the artifact's IR body,
compares it to the value embedded in the MAP epilogue, and — on a signed artifact
(signing is opt-in) — validates its `pqc-hybrid-ml-dsa-87-slh-dsa-256s` signature, whose
ML-DSA-87 and SLH-DSA-SHAKE-256s legs must both verify. The subcommand is the human-facing and
CI-facing discharge surface for the RFC 0016 evidence-chain claim.

## 2. Motivation

RFC 0016 defines what a compile-time evidence chain *asserts* and RFC 0021 ships the
mic@3 container that carries it. Without a first-party verifier, the chain has no
observable discharge surface: an auditor holding a `.mic` artifact cannot confirm that
the embedded `trace_hash` is correct, that the substrate matches, or that a
signature the artifact carries is valid. The claim stays internal.

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

Expected report on stdout for a valid, unsigned artifact (verdict lines go to stderr):

```
artifact:         out/matmul.mic3
substrate:        <substrate>
determinism:      deterministic
toolchain:        <mindc version>
parent:           (root)
trace_hash:       <64 hex chars>
trace_hash_kind:  mic3-bytes
trace_hash_valid: yes
fp_mode:          strict
ssa_valid:        yes
signature:        absent
```

On a tampered artifact `trace_hash_valid` reads `NO`, stderr carries
`error[verify]: trace_hash MISMATCH — artifact has been tampered with`, and the exit
code is 1.

On an artifact signed with the PQC hybrid scheme — signed (opt-in): the emitter signs
only when the key seeds are supplied through `MIND_EVIDENCE_MLDSA87_KEY` and
`MIND_EVIDENCE_SLHDSA_KEY`, and unsigned is the default — pin both signer keys:

```
mindc verify --signer-pubkey <ml-dsa-87 pubkey hex> --signer-pubkey <slh-dsa pubkey hex> out/matmul.mic3
```

The report then names the scheme and both keys:

```
signature:        pqc-hybrid-ml-dsa-87-slh-dsa-256s
signature_mldsa87_pubkey: <hex>
signature_slhdsa_pubkey:  <hex>
```

Both legs must verify and both keys must be in the pinned set; a stripped leg, a
rewritten scheme tag or an unpinned key fails closed (exit 1).

### Verification flags

**Shipped** (`mindc verify --help` is authoritative):

| Flag | Meaning |
|---|---|
| `--json` | Emit the report as a JSON object (incl. `trace_hash_valid`, `ssa_valid`, `signature`, `provenance_authenticated`) instead of human-readable text |
| `--require-strict-fp` | Fail (exit 1) unless the FP-contract mode is `strict` (no FMA-contraction / f32-reassociation). Fail-closed on `relaxed` / `unknown` / unattested |
| `--require-deterministic` | Fail (exit 1) unless the artifact is `deterministic`. The mode is RE-DERIVED from the hashed body (not the forgeable MAP field), so this is fail-closed against a tampered `determinism` label |
| `--signer-pubkey <hex>` | Trust anchor: pin the expected signer key(s) (repeatable; also `MIND_EVIDENCE_VERIFY_PUBKEYS`). Pinning makes a valid signature REQUIRED — an unsigned / signature-stripped artifact fails closed. A trusted signature authenticates the provenance preimage (`substrate` / `toolchain` / `parent`) |
| `--require-signed` | Fail (exit 1) unless the artifact carries a VALID signature (any signer). Weaker than `--signer-pubkey`, which also requires the signer be pinned/trusted; use this for an "every artifact must be signed" policy without a specific key allowlist |

**Not shipped** (documented in earlier drafts; superseded or deferred — see *Shipping status*):

| Flag | Status |
|---|---|
| `--check-sig`, `--pubkey <path>` | Superseded by `--signer-pubkey <hex>` (pinning a key makes a valid signature required). |
| `--format text\|json`, `--out <path>` | Superseded by `--json` (stdout only; no file-receipt flag). |
| `--cross-substrate` | Not a verify subflag: cross-substrate byte-identity is enforced by the `tests/cross_substrate_identity.rs` CI gate against pinned reference hashes. |
| `--chain` | Deferred (parent-DAG walk); `parent` is emitted and verified per-artifact, but the multi-artifact walk is future work. |

## 4. Reference-level explanation

### 4.1 Hash-equality verification (primary gate)

The verifier executes the RFC 0016 §3.2 self-reference rule:

1. Parse the mic@3 artifact; peel the MAP epilogue.
2. Extract `evidence_chain.trace_hash` from the MAP.
3. Recompute `ir_trace_hash` over the canonical mic@3 bytes embedded in the
   mic@3 body — `trace_hash = SHA-256(canonical mic@3 bytes)` per FIPS 180-4, the
   same seam as `deps::mini_sha256` (re-anchored from mic@1 text on 2026-05-31
   after a collision audit — mic@1 text can drop function-body semantics; mic@3
   binary commits the full `IRModule`; supersedes the original RFC 0016 GAP-1
   mic@1-text rule).
4. Compare the recomputed hash to the stored value byte-for-byte.
5. When the artifact is signed (opt-in): validate the
   `pqc-hybrid-ml-dsa-87-slh-dsa-256s` signature; both the ML-DSA-87 and the
   SLH-DSA-SHAKE-256s legs must verify.
6. Report `substrate`, `determinism`, `toolchain`, `parent` from the MAP.
7. Exit 0 on full pass; nonzero (with a diff line indicating first divergence) on
   any mismatch or signature failure.

This is the **default** and **always-available** mode. It requires only the artifact
— no reference database, no network, no second substrate.

### 4.2 Cross-substrate equality check

Cross-substrate equality is not a `mindc verify` flag. For a Q16.16 graph, artifacts
built for different substrates must carry the same `trace_hash` value (RFC 0015 §3.1
inspectable identity); the `tests/cross_substrate_identity.rs` CI gate enforces this
against pinned reference hashes. Two local artifacts can be compared by running
`mindc verify --json` on each and comparing their `trace_hash` and `parent` fields.

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
(§8) but is the logical extension of the cross-substrate gate (§4.2).

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
| `verify.signature` | `bytes` | Signature under the evidence signing scheme (`pqc-hybrid-ml-dsa-87-slh-dsa-256s`, opt-in). |

The `verify.*` block is optional in the carrier; its presence signals that the
artifact has been independently re-verified by a second tool invocation (or CI pass)
and the result is embedded for downstream auditors.

### 4.5 Exit codes

| Exit | Meaning |
|---|---|
**Shipped** (`mindc verify --help` is authoritative):

| `0` | All checks passed — attested → `trace_hash` valid; or unattested-but-SSA-valid with `attested: false` (attestation is opt-in). |
| `1` | Verification failed: `trace_hash` mismatch (tampered body), malformed artifact, SSA fault, or a failed `--require-strict-fp` / `--require-deterministic` / pinned-signer gate. |
| `2` | I/O or CLI error (artifact not found, bad flag). |

> Earlier drafts assigned distinct codes (`2` signature, `3` cross-substrate, `4` parse, `5` `--chain` walk). The shipped CLI collapses every verification failure into `1` and reserves `2` for I/O / CLI errors; the distinct codes are deferred together with their (not-yet-shipped) flags.

### 4.6 JSON receipt schema

**Shipped** (`mindc verify --json`, one object on stdout):

- An attested artifact prints 17 keys: `artifact`, `substrate`, `determinism`,
  `toolchain` (strings); `parent` (hex string or `null`); `trace_hash` (hex string);
  `trace_hash_kind` (`"mic3-bytes"` or `"mic1-text"`); `trace_hash_valid` (bool);
  `fp_mode` (`"strict"`, `"relaxed"` or `"unknown"`); `ssa_valid` (bool); `ssa_reason`
  (string or `null`); `signature` (the scheme name, or `"absent"`, `"invalid"`,
  `"malformed"`, `"unsupported"`, `"retired (<scheme>)"`); `signature_ed25519_pubkey`,
  `signature_mldsa_pubkey`, `signature_mldsa87_pubkey`, `signature_slhdsa_pubkey` (hex
  string or `null`); `provenance_authenticated` (bool). It is printed before the
  signature, strict-FP and determinism gates run, so a later gate can still exit 1.
- An artifact with no evidence chain, or one whose IR body fails the SSA check, prints
  4 keys: `artifact` (string), `ssa_valid` (bool), `ssa_reason` (string or `null`),
  `attested` (`false`).
- A non-canonical artifact and a malformed evidence chain print nothing on stdout; the
  diagnostic goes to stderr and the exit code is 1.

**Planned, not shipped** (the original draft; kept as the target schema):

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
sidecar rather than embedded in the artifact. Rejected: RFC 0021 §2 and an
independent cross-review finding (2026-05-27) establish that evidence must be
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
3. **Key rotation / revocation UX.** `--signer-pubkey` takes hex keys; how a verifier
   discovers the correct key for an artifact produced by an older `mindc` release is
   covered by RFC 0020 §5 (per-version key archive) but not fully specified here.

## 9. Future possibilities

- **`mindc verify --symbolic`**: SMT-proven cross-substrate identity gate once a
  formal lowering model exists.
- **`mindc verify --chain` with remote fetch**: follow parent pointers across a
  content-addressed artifact registry.
- **`verify.*` block as a CI badge input**: the JSON receipt populates a
  `shields.io`-style badge on mindlang.dev (linked to RFC 0020 §12.8).
- **Governance integration**: MIND-Law DIFC `proof_chain` records the
  `trace_hash` from a `mindc verify` pass as a proof input (RFC 0016 §7 / RFC 0021
  §3.4 convergence deliverable).

## 10. Acceptance

1. `mindc verify <artifact>` exits 0 on a valid mic@3 artifact and nonzero on a
   tampered one.
2. A single flipped byte anywhere in the artifact body causes `trace_hash`
   mismatch and exit 1.
3. Signature trust — SHIPPED as `--signer-pubkey <hex>` (not `--check-sig`):
   pinning a signer key makes a valid signature REQUIRED, so an unsigned or
   signature-stripped artifact fails closed (exit 1). Without a pin, an
   internally-consistent signature passes but authenticity is not claimed.
4. Cross-substrate byte-identity — NOT a verify subflag: enforced by the
   `tests/cross_substrate_identity.rs` CI gate against pinned reference hashes,
   not `mindc verify --cross-substrate`.
5. `--json` emits a machine-readable receipt (fields per §4.6 and
   `mindc verify --help`, including `provenance_authenticated`).
6. The subcommand reuses `mic3_evidence_report` from the mic@3 evidence module.
   NOT MET as stated: `verify_evidence_chain` is the mic@2 Graph verifier and has no
   caller on the CLI path — `mindc verify` reads mic@3 and pairs `mic3_evidence_report`
   with `mic3_canonical_check`. This criterion described an intended reuse that the
   implementation never took; recorded rather than silently treated as satisfied.
   the shipped library core; no new hash or signing primitive is introduced.
7. Phase B of RFC 0016 (`mindc verify --evidence`) is satisfied by this subcommand
   landing; the two specs are consistent.
8. `mindc verify` reports `ssa_valid` (single-assignment + define-before-use over
   the mic@3 IR body) **independent of attestation**: an unattested-but-SSA-valid
   artifact reports `ssa_valid` with `attested: false` and exits 0; an SSA-faulted
   artifact exits 1 naming the violated rule and value id; an attested artifact
   reports BOTH `ssa_valid` and `trace_hash_valid`.

### Shipping status (2026-06-06)

`trace_hash` verification (criteria 1, 2, 5, 6, 7) and the **SSA well-formedness
slice** (criterion 8) are SHIPPED — see `check_ssa_well_formed` in `src/ir/verify.rs`
and `run_verify` in `src/bin/mindc.rs`, exercised by `tests/verify_ssa.rs`. Full
SMT-proven verification (`--symbolic`, §4.3 / §9) remains future work. Criteria 3 and 4
shipped in the forms they now state: `--signer-pubkey` in place of `--check-sig`, and the
cross-substrate CI gate in place of a `--cross-substrate` flag.

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
