# RFC 0019: Deterministic Agent Substrate

| Field | Value |
|---|---|
| RFC | 0019 |
| Title | Deterministic Agent Substrate — `agent.*` trace fields and evidence-chain linkage |
| Status | **Draft** |
| Authors | STARGA Inc. |
| Created | 2026-05-29 |
| Task | #294 |
| Related | RFC 0016 §3.3 (the `trace_hash` fold — runtime evidence meets compile-time evidence here), RFC 0016 §7 (ecosystem fit — `agent.*` is the runtime layer of the one-chain design), RFC 0011 (async + structured concurrency — `ReplayScheduler` provides the runtime trace-hash that agent steps fold in), RFC 0020 (mind-bench — `agent-state-replay-*` workloads defined here are future bench workloads), RFC 0021 (mic@3 + evidence MAP — the `agent.*` block is a sibling MAP namespace in the same carrier) |

---

## 1. Summary

This RFC defines the `agent.*` namespace in the mic@3 MAP epilogue (RFC 0021): a set
of trace fields that link an agent step's runtime execution to the compile-time
evidence chain of the IR that ran. It specifies how an agent step's `trace_hash`
is computed, how it folds into the evidence chain (RFC 0016 §3.3), and how the
resulting chain carries the "deterministic cognition" property — the claim that
an agent step is as reproducible and auditable as the artifact it executed.

## 2. Motivation

RFC 0016 §7 establishes a three-layer evidence spine:

> **Compile → Runtime/agent → Governance**

Phase A/B of RFC 0016 shipped the compile layer. The governance layer (512-mind
`proof_chain`) is a tracked deliverable. The runtime/agent layer — `agent.*` — is
the missing link that connects compile-time evidence to runtime reproducibility.

Without it:
- An agent step that executes a MIND artifact carries no in-band record of *which
  artifact it ran*, *on which substrate*, or *whether the execution was
  deterministic*.
- The RFC 0011 `ReplayScheduler` produces a `trace_hash` for each execution, but
  that hash is a local scheduler artifact with no connection to the artifact's
  compile-time evidence chain.
- The "deterministic cognition" claim — that agent behaviour is as reproducible as
  the compiled artifact — has no formal definition and no inspectable representation.

RFC 0016 §3.3 defines the fold that connects the two hashes:

```
trace_hash = H(mic1_ir_hash || replay_trace_hash)
```

This RFC specifies the `agent.*` MAP fields that carry the inputs to that fold,
the fold itself, and how a verifier reconstructs the link.

## 3. Guide-level explanation

### 3.1 What an agent step produces

When a MIND agent step executes under the RFC 0011 `ReplayScheduler`, it produces
an `agent.*` block embedded in the execution record:

```
agent.artifact_hash    <64 hex chars>   // trace_hash of the compiled artifact that ran
agent.replay_trace     <64 hex chars>   // ReplayScheduler trace_hash for this step
agent.step_hash        <64 hex chars>   // H(artifact_hash || replay_trace) — the fold
agent.substrate        x86_avx2         // substrate the step ran on
agent.step_id          <uint>           // monotone step counter within the agent session
agent.session_id       <hex>            // identifies the agent session (for DAG correlation)
agent.determinism      deterministic    // "deterministic" or "nondeterministic"
agent.toolchain        0.7.1            // mindc version of the artifact that ran
```

### 3.2 How the chain reads

A verifier reconstructing the agent's evidence chain:

1. Loads the compiled artifact (`out/inference.mic3`) and extracts
   `evidence_chain.trace_hash` — the compile-time hash of the IR.
2. Runs `mindc verify out/inference.mic3` to confirm the artifact is valid.
3. Reads the agent step record and finds `agent.artifact_hash`.
4. Confirms `agent.artifact_hash == evidence_chain.trace_hash`.
5. Computes `H(agent.artifact_hash || agent.replay_trace)` and confirms it equals
   `agent.step_hash`.
6. Chains step records: `step[n].agent.parent` == `step[n-1].agent.step_hash`.

Step 4 is the *link*: it binds the runtime execution to the compiled artifact, so
the compile-time evidence (substrate, toolchain, determinism, parent DAG) flows
into the agent step record without duplication.

### 3.3 Determinism claim

An agent step is `deterministic` in the `agent.determinism` field if and only if:
- The artifact it ran has `evidence_chain.determinism = "deterministic"`.
- The `ReplayScheduler` is active (RFC 0011 Phase A — scheduler injection; a step
  run without the scheduler cannot produce a deterministic `replay_trace`).
- The substrate matches the artifact's declared `evidence_chain.substrate`.

A step that fails any of these conditions emits `agent.determinism =
"nondeterministic"` — it declines to attest reproducibility rather than silently
omitting the field.

## 4. Reference-level explanation

### 4.1 The `agent.*` MAP namespace

The `agent.*` block is a sibling namespace in the mic@3 MAP epilogue (RFC 0021 §3.2).
It is written by the agent runtime (the RFC 0011 `ReplayScheduler` integration layer),
not by `mindc` itself — `mindc` emits `evidence_chain.*`; the scheduler emits
`agent.*` for the step record. A step record is a separate mic@3 artifact (or a MAP
entry in a session log), not a mutation of the compiled artifact.

| Key | Type | Meaning |
|---|---|---|
| `agent.schema` | `uint` | Block schema version (`1`). |
| `agent.artifact_hash` | `bytes` (32) | `evidence_chain.trace_hash` of the compiled artifact that executed in this step. The link to compile-time evidence. |
| `agent.replay_trace` | `bytes` (32) | The RFC 0011 `ReplayScheduler` `trace_hash` for this step (the `std.async.trace_hash` value). |
| `agent.step_hash` | `bytes` (32) | `H(agent.artifact_hash \|\| agent.replay_trace)` — the fold (RFC 0016 §3.3). The identity of this step in the chain. |
| `agent.parent` | `bytes` (32) | `agent.step_hash` of the preceding step (absent for the first step of a session). |
| `agent.substrate` | `string` | RFC 0014 canonical substrate ID where the step executed. Must match `evidence_chain.substrate` of the artifact. |
| `agent.step_id` | `uint` | Monotone step counter within the session (starts at 1). |
| `agent.session_id` | `bytes` (16) | Identifies the agent session (128-bit random, assigned at session start). Used to group step records into a DAG. |
| `agent.determinism` | `string` | `"deterministic"` or `"nondeterministic"` (§3.3 rule). |
| `agent.toolchain` | `string` | `mindc` version of the artifact that executed. |
| `agent.signature` | `bytes` | Ed25519 signature over the step record's MAP-minus-`signature.*` (optional; required for governance-layer consumption). |

### 4.2 The fold function (normative)

```
agent.step_hash = SHA-256(agent.artifact_hash || agent.replay_trace)
```

- **Concatenation order is normative**: artifact hash first, replay trace second.
- **Both inputs are 32-byte values** (SHA-256 outputs).
- **The hash function is FIPS 180-4 SHA-256**, the same seam as `evidence_chain.trace_hash`
  and `deps::mini_sha256`. No BLAKE3, no SHA-3. Rationale: the Rust bootstrap and
  the pure-MIND `std.sha256` endpoint must produce bit-identical fold outputs.
- **The fold is associative at the session level**: `session_hash = SHA-256` of all
  `step_hash` values in `step_id` order is a session-level summary hash (informational;
  not normative in v1).

### 4.3 Verification

`mindc verify --agent-step <step-record.mic3>` (a future extension of RFC 0017):

1. Extract `agent.artifact_hash`, `agent.replay_trace`, `agent.step_hash`.
2. Recompute `H(artifact_hash || replay_trace)`.
3. Compare to `agent.step_hash`.
4. If `--artifact <artifact.mic3>` is supplied, confirm
   `artifact.evidence_chain.trace_hash == agent.artifact_hash`.
5. Optionally validate `agent.signature`.

Exit 0 on pass; nonzero with a specific exit code per §4.4.

### 4.4 Exit codes for step verification

| Exit | Meaning |
|---|---|
| `0` | All checks passed. |
| `10` | `agent.step_hash` does not match recomputed fold. |
| `11` | `agent.artifact_hash` does not match supplied artifact's `trace_hash`. |
| `12` | `agent.substrate` does not match artifact's `evidence_chain.substrate`. |
| `13` | `agent.signature` invalid or required but absent. |
| `14` | Step record parse error. |

Exit codes 10–14 are in the `agent.*` namespace; exit codes 1–5 (RFC 0017) cover
artifact verification. Ranges are non-overlapping.

### 4.5 Relationship to governance (512-mind `proof_chain`)

RFC 0016 §7 and RFC 0021 §3.4 state that 512-mind's DIFC `proof_chain` should
record `evidence_chain.trace_hash` as a proof input. The `agent.*` layer extends
this: a governed decision that references an agent step records
`agent.step_hash` as the proof input, so the governance chain is anchored on the
*step identity* (which includes the runtime trace), not just the compiled artifact.

This is a **tracked deliverable**, not a shipped fact. The present state (RFC 0021
§3.4): two chains, convergence in progress. This RFC is the specification of the
`agent.*` block that makes the convergence possible; wiring it into 512-mind's
`EvidenceChain` consumer is a separate cross-repo task.

### 4.6 Session DAG

Step records form a directed acyclic graph via `agent.parent`:

```
[step 1, parent=∅]
     ↓ agent.step_hash
[step 2, parent=step1.step_hash]
     ↓
[step 3, parent=step2.step_hash]
```

The DAG is analogous to the compile-time DAG in RFC 0016 §4 but operates at
*session* granularity rather than *transformation* granularity. A branching
agent (parallel steps) produces a fork:

```
[step 2a, parent=step1]      [step 2b, parent=step1]
```

Both children share the same `parent`; their `step_hash` values differ. The
session-level summary hash covers all terminal nodes.

### 4.7 RFC 0020 integration

RFC 0020 §4.2 reserves `agent-state-replay-{small,medium,large}` as future
mind-bench workloads, gated on RFC 0019. This RFC defines the trace fields; the
mind-bench workload format will include a reference `agent.step_hash` column
alongside the artifact hash columns.

## 5. Drawbacks

- The `agent.*` block is only meaningful when the RFC 0011 `ReplayScheduler` is
  active. Without the scheduler, `agent.replay_trace` is absent and
  `agent.determinism` must be `"nondeterministic"` — the agent step record is
  a partial record, not a full one.
- The fold ties the runtime layer to the compile-time layer via
  `agent.artifact_hash`. If the same artifact runs on multiple substrates in
  different steps, each step records the same `artifact_hash` but different
  `substrate` — a correct and expected state, but reviewers must understand the
  separation.
- The session-level DAG is reconstructed by collecting step records; there is no
  single session-manifest artifact in v1.

## 6. Rationale and alternatives

**Alternative: agent step hash is the replay trace only.** Simpler. Rejected: this
severs the link to compile-time evidence. An auditor cannot confirm which compiled
artifact ran, so the "deterministic cognition" claim has no inspectable foundation.

**Alternative: inline the compile-time evidence in each step record.** Copy the
full `evidence_chain.*` block into every step record. Rejected: this duplicates
data proportional to session length and introduces desync risk if the step record
is produced independently of the artifact.

**Alternative: use a detached receipt per step.** A separate sidecar file for each
step record. Rejected: the same reasoning as RFC 0021's rejection of detached
provenance — desync risk and no artifact inseparability.

## 7. Prior art

- **W3C PROV-O**: a provenance ontology for agents and activities. The `agent.*`
  DAG mirrors PROV-O's `Activity → wasGeneratedBy → Entity → wasDerivedFrom`
  structure. The MIND form is more constrained (hash-linked, cryptographic,
  compiler-native) and makes no claim of OWL compatibility.
- **OpenTelemetry trace IDs + span IDs**: a precedent for step counters and
  parent-child trace linking in distributed systems. The `agent.step_id` /
  `agent.session_id` / `agent.parent` tuple mirrors span+trace+parent-span IDs.
  The MIND form adds cryptographic binding (hashes, not opaque IDs).
- **in-toto link metadata**: per-step signed records in a supply-chain attestation.
  The `agent.*` block is the MIND analog for inference-time steps.

## 8. Unresolved questions

1. **Scheduler availability check.** How does the agent runtime know whether the
   `ReplayScheduler` is active before committing `agent.determinism =
   "deterministic"`? The RFC 0011 Phase A scheduler injection must provide a
   query API.
2. **Session-log format.** Where do step records live? A mic@3 session log (a MAP
   with multiple `agent.*` blocks keyed by step_id), a flat binary log, or
   individual per-step mic@3 artifacts? Needs a format RFC or §4.6 extension.
3. **Fold ordering for branching DAGs.** The session-level summary hash over
   parallel branches needs a canonical ordering (by `step_id`? by `step_hash`
   lexicographically?). Normative pin deferred to implementation.
4. **Replay of an agent step.** To recompute `agent.replay_trace`, the verifier
   must replay the step under the same `ReplayScheduler`. The replay protocol
   (which inputs, which seed, which state) is RFC 0011 Phase A's concern; this
   RFC depends on it but does not define it.

## 9. Future possibilities

- **`mindc verify --agent-session`**: verify an entire session DAG, not just a
  single step.
- **mind-bench `agent-state-replay` workloads**: RFC 0020 §4.2 workloads that
  use `agent.*` records as the verification unit instead of raw artifact hashes.
- **Governance wiring**: 512-mind `proof_chain` records `agent.step_hash` as a
  proof input (RFC 0021 §3.4 convergence deliverable).
- **Cross-agent session correlation**: multiple agent instances sharing a
  `session_id` produce step records whose `parent` pointers form a multi-agent
  DAG — the compile-time Merkle-DAG extended to runtime collaboration.

## 10. Acceptance

1. The `agent.*` MAP block is specified with enough precision that a second
   independent implementation can produce bit-identical `agent.step_hash` values
   given the same `artifact_hash` and `replay_trace` inputs.
2. The fold (`H(artifact_hash || replay_trace)`) is pinned by a frozen test vector
   (analogous to the RFC 0016 FIPS conformance vector) before any implementation
   lands.
3. `mindc verify --agent-step` (RFC 0017 extension) recomputes the fold and
   confirms the link to the compiled artifact.
4. An agent step record with `agent.determinism = "deterministic"` can only exist
   if the artifact has `evidence_chain.determinism = "deterministic"` — a
   constraint enforced by the scheduler integration, not just convention.
5. The `agent.*` block introduces no new hash primitive — SHA-256 (FIPS 180-4)
   throughout.

## 11. References

RFC 0011 (async + structured concurrency — `ReplayScheduler` produces
`std.async.trace_hash`, the input to the fold);
RFC 0016 §3.3 (the fold definition — `H(mic1_ir_hash || replay_trace_hash)`),
§7 (agent layer in the one-chain ecosystem design);
RFC 0020 §4.2 (`agent-state-replay-*` future workloads);
RFC 0021 §3.2 (mic@3 MAP epilogue — the container `agent.*` is a namespace in),
§3.4 (512-mind governance convergence — `agent.step_hash` as proof input);
RFC 0017 (the verifier surface that will grow to cover `--agent-step`);
`src/ir/compact/v3/evidence.rs` (the `mic3_evidence_report` library the verifier
builds on); W3C PROV-O (prior art); OpenTelemetry span model (prior art for
session/step/parent IDs); in-toto link metadata (supply-chain step records).
