# RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue

| Field | Value |
|---|---|
| RFC | 0021 |
| Title | Canonical IR Unification (mic@1e epilogue; model-exchange demotion) |
| Status | **Draft** — design locked, implementation pending (bootstrap-parser blast radius) |
| Authors | STARGA Inc. |
| Created | 2026-05-26 |
| Supersedes-in-part | RFC 0016 §3.2/§5.4 (evidence anchor + carrier), RFC 0015 (enforcement seam) |
| Depends | RFC 0001 (mic@1 determinism), RFC 0014 (per-substrate lowering), RFC 0016, RFC 0020 |

## 1. Problem — two IRs, and the governance layer is bolted to the wrong one

A 2026-05-26 architecture audit (mind-architect + mind-auditor) found MIND carries
**two disjoint IRs**:

- **`IRModule`** (`src/ir/mod.rs`) — the rich, canonical compiler IR: control flow,
  functions, SIMD, 39+ instructions. Serialized as **mic@1** text (`ir::save`),
  RFC-0001 deterministic (`save → load → save` is a cross-platform byte-identical
  fixed point). This is what the real pipeline produces
  (`source → AST → IRModule → MLIR → native`, `src/pipeline.rs:267`) and what the
  **self-host compiler** (`libmindc_mind.so`, emits MLIR-text) tracks.
  `docs/ir-stability.md` names mic@1/IRModule the canonical contract.
- **v2 `Graph`** (`src/ir/compact/v2/`) — a separate, weaker IR: ~19–22 pure-dataflow
  opcodes, **no control flow, no functions, no SIMD** (`types.rs:146`). Carries the
  mic@2 / mic@2.1 / MIC-B binary lineage, the MAP epilogue, **all** of the RFC 0016
  evidence-chain machinery, and Ed25519 signing.

The governance/provenance layer (RFC 0016 evidence chains, mic@2.1 MAP, signing) was
built entirely on the v2 `Graph`. But the v2 `Graph` **cannot represent a real
program** (no control flow/functions), there is **no `IRModule → Graph` bridge**, the
only non-test `Graph` is the hardcoded fixture `Graph::residual_block()`
(`types.rs:542`), `mindc` emits **no** mic@2.1 artifact, and the loader **rejects**
mic@2/MIC-B (`ir/mod.rs:80-81`). Concretely:

- `attach_evidence_chain` has **zero non-test call sites**.
- The mic@1 evidence anchor `ir::ir_trace_hash` (`src/ir/evidence.rs`, landed
  2026-05-26 as the first GAP-1 fix) is **wired to nothing** — referenced only by its
  own unit tests.
- `tests/cross_substrate_identity/` is **data files only** (`manifest.toml` +
  `reference_hashes.toml`) — there is **no asserting harness** comparing
  `H_substrate_A == H_substrate_B`.

**RFC 0016 is currently a spec with a test fixture, not a feature, and the
cross-substrate wedge (RFC 0015) is a reference-hash table, not a gate.** For a
language whose entire defensible wedge is *cross-substrate bit-identity +
compile-time evidence chains + determinism-by-default*, the IR foundation must be
coherent: provenance must attest **the artifact the compiler actually ships**, on
**the IR the self-host converges toward**.

## 2. Prior art — the unanimous pattern

Every production system that attaches provenance to compiled output keeps **one
canonical IR + a separable, versioned metadata section** — none maintains a second
IR to hold metadata:

| System | Canonical IR/artifact | Provenance attachment |
|---|---|---|
| **MLIR** | bytecode | properties / attributes / discardable attrs |
| **LLVM** | bitcode | `!metadata`, module flags, embedded-bitcode section |
| **WASM** | module | **custom sections** (ignored by engines, read by tools) |
| **SLSA / in-toto / Sigstore** | the build artifact itself | a *detached* signed attestation keyed by the artifact's content hash |

The lesson is decisive: **attach, do not fork.** MIND's second IR is the anti-pattern.

## 3. Decision

### 3.1 One canonical compiled-artifact IR: mic@1 / `IRModule`
`IRModule`/mic@1 is the sole IR for compiled artifacts and the sole evidence anchor.
`trace_hash = SHA-256(canonical mic@1 text)` (already `src/ir/evidence.rs::ir_trace_hash`,
via the `deps::mini_sha256` FIPS-180-4 seam — bit-identical to a future pure-MIND
`std.sha256` over the same bytes, preserving the RFC 0016 §5.4 duality). This survives
the self-host boundary because the self-host endpoint converges on `IRModule → MLIR`.

### 3.2 Provenance as a versioned epilogue: **mic@1e**
mic@1 today **fails closed on unknown lines** (`src/ir/compact/parse.rs:142-143`), so
it has no metadata slot. Add a back-compatible **fence**: the mic@1 IR body, then a
sentinel line (`---MAP---`), then the canonical MAP key/values. Rules:

- **Pre-fence bytes are byte-identical to current mic@1.** An artifact with no
  evidence emits exactly today's bytes (no fence) — so the RFC-0001 `save→load→save`
  fixed point, the self-host bootstrap byte-identity (`phase_g_keystone_bootstrap`),
  and the RFC 0020 bench baselines are **untouched**.
- `parse_mic` treats the fence as a clean EOF for the IR body and hands the remainder
  to the **existing v2 `Map` parser** — the MAP / canonical-sort / Ed25519 machinery
  in `v2/evidence.rs` is **reused verbatim**, fed mic@1 bytes instead of `emit_micb`
  bytes. No rewrite of the signing surface.
- **Do not change mic@1's IR grammar.** Version the epilogue independently:
  `evidence_chain.schema = 1`. mic@1's stability contract is preserved.

### 3.3 Demote v2 `Graph` to a model-exchange format: **`mind-model@2`**
The v2 lineage is honest as a *tensor model-exchange* format for the graphs that **are**
expressible in its dataflow opcodes — rfn-mind / MindLLM tensor graphs, and 512-mind's
MAP `dump` (its only documented v2 use). Rename the lineage in docs to break the
"compiler IR" implication; keep `emit_micb`/`parse_micb` for that role. It is **not**
the compiled-artifact IR and **not** the general evidence anchor.

### 3.4 Honesty about the governance chains (supersedes RFC 0016 §7's aspiration)
512-mind ships its **own** DIFC evidence chain (`observer.mind::EvidenceChain`,
sha3_512 / Hash512) — today it is a **genuinely separate** chain, not a consumer of
mic@1e `trace_hash`. RFC 0016 §7's "consumes, does not parallel" is **aspirational
until wired**. Action: 512-mind records the mic@1e `trace_hash` as a proof input, or
the RFC stops claiming unification. State the present truth plainly: **two chains
today; convergence is a tracked deliverable**, not a shipped fact.

### 3.5 Enforce cross-substrate bit-identity, don't assert it
- **IR-level link (now, cheap, by construction):** one `IRModule` fans out to N
  per-substrate binaries; the IR is substrate-invariant, so mic@1 `trace_hash` is
  trivially identical across targets. Add the missing `oracle.rs` asserting it — and,
  crucially, run the existing `tests/cross_substrate_identity` **output-hash** gate in
  a toolchain-equipped CI job that **hard-fails (not self-skips)** on release tags.
- **Output-hash matrix (deferred):** Q16.16 is an i32 *encoding convention*, not a
  mic@1 `DType` — so mic@1 identity is **necessary but not sufficient**. True
  bit-identity is an output-hash property of the *lowered* binary and needs the
  per-substrate runtime matrix (aarch64 runner / QEMU). Until a real second-substrate
  run blesses it, stop committing `neon` reference hashes as hand-copied duplicates of
  `avx2` (mark them `deferred`, never green-by-copy).

## 4. Migration path (must not regress the self-host bootstrap)

1. **mic@1e fence** in `emit_mic`/`parse_mic` + a fixed-point test proving pre-fence
   bytes are byte-identical to current mic@1 (protects bootstrap + bench baseline).
2. **Re-point** `v2/evidence.rs::compute_trace_hash` to accept externally-supplied
   canonical bytes (mic@1), so the MAP/Ed25519 container is reused unchanged.
3. **Wire** `ir_trace_hash` into a real `mindc build --emit=evidence` path that writes
   the mic@1e epilogue (unsigned local; Ed25519 at release-tag time). Turn the orphan
   into the actual artifact anchor.
4. **`mindc verify --evidence`** reads a mic@1e artifact, recomputes `ir_trace_hash`
   over its IR body, compares to the stored `trace_hash`, validates the signature.
   (Shares the RFC 0017 verifier surface, #290.)
5. **Demote** v2 → `mind-model@2` in `docs/ir-stability.md` + `versioning.md`; document
   the mic@1e schema; resolve the `MICB_VERSION`/`Mind.toml` drift (#308).
6. **`oracle.rs`** + CI wiring for the bit-identity gate (#307).

Each step is independently testable and additive; none regresses mic@1 byte-output for
evidence-free artifacts.

## 5. Acceptance

1. An evidence-free `mindc build` emits byte-identical mic@1 to pre-RFC output;
   `phase_g_keystone_bootstrap` 7/7 unchanged.
2. `mindc build --emit=evidence` produces a mic@1e artifact whose `trace_hash` equals
   `ir_trace_hash` of its own IR body; `mindc verify --evidence` round-trips it.
3. The cross-substrate gate runs in CI and hard-fails on a deliberately corrupted
   reference hash (no self-skip on release tags).
4. `docs/ir-stability.md` describes exactly one compiled-artifact IR (mic@1/mic@1e) and
   names `mind-model@2` as the model-exchange format.

## 6. First steps (highest leverage, in order)

1. mic@1e fence + fixed-point byte-identity test (§4.1) — the keystone-protecting step.
2. Re-point the evidence container at mic@1 bytes + wire `mindc build --emit=evidence` (§4.2–4.3).
3. `oracle.rs` + CI gate for bit-identity (§3.5).

> Implementation note: §4.1 touches the frozen mic@1 parser that the self-host
> bootstrap consumes — it must be done with a keystone re-verification in a focused
> session, not bundled into unrelated work.
