# RFC 0025: MIND Intent — Intent Contracts (goal + constraints → verifiable Contract IR)

| Field | Value |
|---|---|
| RFC | 0025 |
| Title | MIND Intent — Intent Contracts (goal + constraints → verifiable Contract IR) |
| Status | **Draft — In Review** (I1). Normative surface below; no implementation has shipped. `mind-intent` is at I0 (skeleton). |
| Authors | STARGA Inc. |
| Created | 2026-08-05 |
| Repo | `mind-intent` (private); CLI surface `mindc intent <file>.intent.mind` |
| Depends | RFC 0015 (cross-substrate bit-identity — the property a determinism envelope may claim), RFC 0016 (evidence-chain emission — carrier for the verification report), RFC 0017 (`mindc verify` — re-derives obligations), RFC 0021 (canonical mic@3 IR — the bytes a `contract_id` hashes) |

> **Numbering note.** This RFC is **0025**, not 0023. `0023` looks free in `docs/rfcs/`
> (the directory runs 0013…0022, then 0024) but is **reserved-and-cancelled**: it was held
> for a "nice sets" language feature whose split verdict on 2026-07-14 was *do not write it*
> (it requires dependent types). Reusing 0023 would collide with that recorded decision.
> 0024 is Loop Collapse. This RFC takes 0025.

---

## 1. Summary

**MIND Intent** compiles a human-stated **goal plus constraints** into a typed,
machine-checkable **Contract**, and **refuses to produce a plan it cannot satisfy.** It sits
in front of MIND Flow: it consumes an `IntentSpec`, lowers it to a trusted **Contract IR**,
and either emits a MIND Flow execution graph whose obligations are discharged, or emits
`E5104 CONTRACT_UNSATISFIABLE` and does not produce a plan. It does not replace MIND Flow,
does not duplicate it, and is not a runtime.

The load-bearing idea is a **trust boundary**. Natural language and LLM output are
*proposals*. `IntentSpec` is a *reviewable artifact*. `Contract IR` is the *first trusted
layer*. Nothing executes until contract checks and policy gates pass. An LLM may draft an
`IntentSpec`; an LLM may never authorize execution.

## 2. Motivation

Agent systems today translate a goal into an action plan through opaque, unreviewable steps,
and "determinism" is asserted as a bare boolean that quietly overpromises. MIND already has
the missing primitive — cross-substrate bit-identity and a hash-anchored evidence chain
(RFC 0015/0016/0017/0021). MIND Intent applies it one layer up: it makes the *plan itself* a
verifiable object with named guarantees and an explicit failure mode, so that an impossible
requirement is a **compile-time rejection with an actionable explanation**, not a runtime
surprise. The rejection path is the headline capability, not error handling.

## 3. Guide-level explanation

An `IntentSpec` states a goal and typed constraints:

```
intent DeployClassifier {
  goal accuracy >= 0.92 on eval_set("cifar10-v1")
  require memory.peak <= 4 GiB
  require effects.network == deny
  require determinism >= numerical(tolerance = 1e-6)
  prefer  latency.p95 <= 20 ms weight 10
  prefer  build_time <= 60 s weight 2
  assume  target == cuda(sm = 90)
  assume  tensorrt.version in "10.x"
  approval required when effects.external_write
}
```

`mindc intent DeployClassifier.intent.mind` parses this to a `Contract`, hashes it to a
`contract_id`, runs the analyzers, and prints a **verification report** (§4.6) — or rejects.

The rejection is the demo. Given an intent whose required peak memory (5.18 GiB) exceeds its
declared cap (4 GiB) with a fixed batch and recompute disallowed:

```
E5104 CONTRACT_UNSATISFIABLE
Required peak GPU memory: 5.18 GiB
Declared maximum:         4.00 GiB

Unsatisfiable core:
  require memory.peak <= 4 GiB
  require batch_size == 64
  require activation_recompute == false

Candidate repairs (not applied automatically):
  1. batch_size <= 32
  2. allow activation_recompute
  3. raise memory.peak to >= 5.18 GiB
```

Candidate repairs are **never auto-applied**. The compiler reports the smallest unsatisfiable
core it can isolate, plus relaxations the human may choose.

## 4. Reference-level explanation

### 4.1 Namespacing (normative, mandatory)

The computational contract surface is **`intent.compute.*`**. It is distinct from the
existing economic-commitment layer **`intent.commit.*`**
(`mind-agents/src/intent_commit.mind`: agent, action, target, risk tier, stake, governance
checks, deadline, lifecycle). **This RFC does not modify `intent_commit.mind` semantics.**
The economic layer *consumes* a `contract_id` hash from the computational side; it is never
repurposed as the computational intent language. Any change that would require editing
`intent_commit.mind` is out of scope for this RFC and is an operator decision.

### 4.2 Five constraint classes

| Class | Semantics | Failure behavior |
|---|---|---|
| `require` | Must be proven statically, attested by target, or verified at runtime before a protected action | Compilation/deployment **rejects** |
| `prefer` | Weighted optimization objective; may be unsatisfied | Report the trade-off; **never** silently promote to a guarantee |
| `assume` | Condition outside the plan that a guarantee depends on | Embed in the envelope; invalidate the guarantee when false |
| `observe` | Metric/evidence that does NOT control trusted computation | Sidecar data; excluded from proof hashes |
| `authorize` | Capability granted by user/policy/delegator | **Deny by default** when missing |

### 4.3 Contract IR

```
Contract {
  contract_id:      Hash              // mic@3 canonical bytes of the normalized Contract (RFC 0021)
  inputs:           [TypedPort]
  outputs:          [TypedPort]
  hard_obligations: [Predicate]       // from require
  soft_objectives:  [WeightedObjective]  // from prefer
  assumptions:      [Assumption]      // from assume
  effects:          [EffectCapability]   // from authorize / effects
  randomness:       [RandomSource]    // stochastic islands (§4.5)
  guarantee:        DeterminismEnvelope   // §4.4
  approval_policy:  ApprovalPolicy
  evidence_policy:  EvidencePolicy
  source_map:       IntentSourceMap   // spans back to IntentSpec for diagnostics
}
```

`contract_id = mini_sha256(emit_mic3(normalize(Contract)))` — anchored on the canonical
mic@3 bytes (RFC 0021), consistent with how MIND already anchors `trace_hash`. Two
contracts with identical semantics and identical envelopes hash equal; `observe` data and
stochastic annotations are excluded from the hashed preimage.

### 4.4 Determinism taxonomy — never a bare boolean

`deterministic = true` alone is **rejected** in strict profiles. Every claim names a level
**and** an envelope (compiler, target, backend, precision, seeds, reduction order).

| Level | Promise |
|---|---|
| D0 | none — unpinned external LLM/tool call |
| D1 | trace-replay — recorded outputs replayable without re-invoking the stochastic source |
| D2 | order-deterministic — same inputs → same command/edge ordering |
| D3 | numerical(tol) — outputs within a declared tolerance under the envelope |
| D4 | bitwise-same-target — bit-identical on the same target/environment |
| D5 | cross-substrate — bit-identical across a proven target set (MIND int/Q16.16 CPU paths) |
| D6 | build-reproducible — source + environment recreate the same artifact bytes |

Envelope shape:

```
determinism numerical {
  tolerance absolute = 1e-6
  target          = cuda.sm90
  compiler        = mindc@0.10.1
  runtime         = mind-runtime@<artifact-hash>
  backend         = tensorrt@10.x
  tactic_cache    = <hash>
  random_sources  = [key("augmentation", 982145)]
  reduction_order = fixed
}
```

**Honesty constraint (normative).** A node may claim only the level the underlying toolchain
has proven for its envelope. D4/D5/D6 are demonstrated today for MIND's integer / Q16.16 CPU
paths (RFC 0015); floating-point and broader cross-substrate coverage are an **open track in
the `mind` compiler**. No partial determinism claim may be upgraded into a universal one.
MIND Intent describes what is *declared and checked* — it does not assert a substrate-wide
guarantee the compiler has not proven.

### 4.5 Stochastic islands

The deterministic spine orchestrates stochastic islands; it does not pretend they are
deterministic. Every stochastic node declares *why* it is stochastic (model sampling,
network state, wall clock, sensor, human decision). Under strict replay the runtime reads the
**recorded output object** — it does not re-invoke the provider. Stochastic annotations
(e.g. `confidence`) are excluded from proof hashes and must never contaminate a byte-identity
claim for a deterministic node.

### 4.6 Verification report

Every field renders a real value; a report with placeholders is not a working report.

```
MIND CONTRACT VERIFICATION REPORT
Contract ID: 8d7c...91e2
Plan hash:   17ab...44f0
Artifact:    pending target build
Hard obligations: 12 satisfied / 2 target-dependent / 0 violated
Soft objectives:  latency target met; build-time preference exceeded by 8 s
Assumptions:      6 declared / 5 attested / 1 pending deployment check
Effects:          network denied; filesystem read-only; external write approval required
Determinism envelope: D3 numerical, tolerance 1e-6, CUDA SM90, pinned tactic cache
Stochastic islands:   1 LLM node, strict replay by captured output hash
Evidence root:        pending execution
Result: PLAN VALID; DEPLOYMENT REQUIRES TARGET ATTESTATION AND HUMAN APPROVAL
```

### 4.7 The `E5104` rejection (normative)

`E5104 CONTRACT_UNSATISFIABLE` reports the smallest unsatisfiable core the analyzer can
isolate over the `require` set, plus candidate relaxations. Relaxations are **never
auto-applied**; they are suggestions for a human to choose. A protected action whose
`authorize` capability is absent is denied by default and surfaces as a distinct
`E51xx` authority error, not a silent skip.

### 4.8 Hierarchical contracts

```
OperatorContract → ComponentContract → NodeContract → FlowContract → DeploymentContract
composition: local guarantee + dependency guarantees + environment assumptions
             ⇒ parent guarantee, OR a structured unresolved obligation
```

The parent guarantee is **computed** from children, adapters, and explicit assumptions —
never asserted in prose.

### 4.9 Versioning

`IntentSpec` and `Contract IR` carry a schema version. The `contract_id` preimage includes
the schema version, so a schema change is a hash change (no silent reinterpretation).
Backward-compatible additive fields bump a minor version; any change to the hashed preimage
layout bumps a major version, mirroring the mic@3 wire-versioning discipline (RFC 0021).

## 5. Drawbacks

- A new surface to maintain and keep honest; the status ledger must not go stale (a sibling
  repo's ledger drifting is precisely the failure this RFC's honesty rails guard against).
- Constraint solving for `E5104` cores is undecidable in general; the analyzer must be
  explicit about what it proves versus defers to target attestation or runtime checks.
- Naming proximity to a third-party commercial "Mindflow" and to the internal
  `intent.commit` layer — mitigated by namespacing (§4.1); a trademark screen for the
  product name remains an operator/legal decision, out of scope here.

## 6. Rationale and alternatives

- **Why build now, not defer.** An evaluation recommended deferring a separate repo until
  ≥3 independent consumers. The operator overrode that and decided to build now. This RFC
  records the decision; the two flagged risks (naming/namespace and honesty) are carried as
  binding constraints (§4.1, honesty rails), not dropped.
- **Why a separate trusted IR rather than annotations on Flow graphs.** The trust boundary
  needs a first-class, hashable artifact that exists *before* a plan is generated, so that
  rejection happens without producing an execution graph. Annotations on an already-built
  graph cannot reject before construction.
- **Why determinism levels, not a boolean.** MIND's wedge is *partial-but-proven*
  determinism. A boolean invites upgrading a CPU-path guarantee into a substrate-wide one;
  the D0–D6 + envelope form makes that category error unrepresentable.

## 7. Prior art

Contract- and intent-oriented compilation, refinement types, capability-based effect
systems, and reproducible-build/attestation work all inform this design; recent research on
structure-first and intent-driven program construction motivates the goal→contract framing.
The specific external ideation source that prompted this evaluation is treated as **external
and unvetted**: its architecture (trust boundary, constraint classes, determinism taxonomy,
fail-loud rejection) was adopted; its framing and unsubstantiated central claim were
rejected and are **not** cited in this or any public artifact. Provenance is retained
privately.

## 8. Honesty rails (normative, binding)

1. No pseudo-scientific or speculative-physics framing, and no invented numbered "laws", in
   the repo, code, comments, docs, commits, or public copy.
2. No "own native backend, no MLIR/LLVM dependency" claim. `mindc` 0.10.1 lowers through
   MLIR/LLVM; full Rust-independence is a separate open compiler track. Describe the native
   backend as **in progress**, or omit it.
3. `mind-intent/docs/status.md` is a three-column ledger (operational / experimental /
   roadmap); every operational row carries a real command + real output.
4. Never claim a gate passed without the verbatim command and its output.
5. No partial determinism claim may be upgraded into a universal one (§4.4).

## 9. Unresolved questions

- The exact `IntentSpec` grammar (units, comparison operators, `on eval_set(...)` binding)
  is settled by this RFC's review before I2 implements the parser.
- The minimal-unsatisfiable-core algorithm and what it proves vs. defers to attestation.
- The precise `EvidencePolicy` binding into the RFC 0016 evidence chain.

## 10. Future possibilities

- I5 lowering into MIND Flow. **Gated**: mind-flow's native build is currently broken
  (tracked outside this repo), so an end-to-end native demo is blocked on a fix that is not
  in this project's scope. I0–I4 are unaffected.
- I6: binding the `contract_id` hash into the economic ICL (`intent.commit.*`) without
  changing its semantics.
- Contract composition across agents (hierarchical contracts, §4.8) as a governance surface
  shared with 512-mind.
