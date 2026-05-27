# RFC 0021: Canonical IR Unification — one IR, provenance as a versioned epilogue

| Field | Value |
|---|---|
| RFC | 0021 |
| Title | Canonical IR Unification (mic@3 binary `IRModule` + embedded MAP; model-exchange demotion) |
| Status | **Partial** — design locked + 5/5 cross-model validated (§3.2). **Steps 1–2 SHIPPED** (mic@3 codec mind@5c29f0d + evidence MAP mind@c64bd0b); step 3 (CLI emit) in progress; steps 4–6 pending (§4). "Demote v2" re-scoped as a byte-preserving cross-repo migration (§4.5). |
| Authors | STARGA Inc. |
| Created | 2026-05-26 (rev. 2026-05-27) |
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

> **Design refinement (2026-05-27).** A first draft proposed a text epilogue on
> mic@1 (`mic@1e`). That conflated two independent axes and was **rejected** — see
> §3.0. The decision below separates the IR **data model** (which must be the full
> `IRModule`) from the **container/encoding** (which should be the superior binary
> MAP-bearing format, not text).

### 3.0 The real axis: data model vs container
Two independent properties, and neither existing format has both:

| | data model | container |
|---|---|---|
| **mic@1** (text) | ✅ full `IRModule` (control flow, functions, SIMD) | ❌ text; **no metadata slot** (parser fails closed, `parse.rs:142-143`) |
| **mic@2/2.1/MIC-B** (binary) | ❌ v2 `Graph` (≤22 dataflow opcodes; **cannot** encode `if`/`while`/`call`) | ✅ binary, compact, MAP epilogue, Ed25519 — already built |

The v2 `Graph`'s **data model**, not its encoding, is what disqualified it — and is the
root cause of the audit (evidence built on the good container, but it could only hold
toy graphs). A text epilogue on mic@1 (`mic@1e`) would retreat to the worse *container*
just to reuse mic@1's right *data model*. The correct move is the converse: give the
**superior binary+MAP container the real `IRModule` data model.**

### 3.1 Canonical data model: the full `IRModule`
`IRModule` is the sole compiled-artifact IR (the v2 `Graph` data model cannot represent
real programs and is demoted, §3.3). `trace_hash` is `SHA-256` of the **canonical IR
bytes** via the `deps::mini_sha256` FIPS-180-4 seam (bit-identical to a future pure-MIND
`std.sha256`, preserving RFC 0016 §5.4). Today's anchor `src/ir/evidence.rs::ir_trace_hash`
hashes the canonical **mic@1 text** — the deterministic interim encoding (RFC-0001 fixed
point, already what the pipeline + self-host produce); it remains valid as the authoritative
bytes until §3.2's binary form ships, after which the hash is taken over the canonical
binary IR bytes.

### 3.2 Canonical container: **mic@3** — binary `IRModule` + embedded MAP
Extend the binary mic@2/MIC-B encoding to carry the **full `IRModule`** instruction set
(control flow, functions, SIMD) **plus** the mic@2.1 MAP epilogue (`evidence_chain.*` +
Ed25519). This is `mic@3` — strictly better than both existing formats: compact binary,
**embedded** signed provenance (the mic@2.1 spec's stated stance — "embedded so verifiers
need no separate sidecar"), *and* able to represent real programs.

- **Reuse, don't rebuild, the container.** The MAP / canonical-sort / `emit_micb` /
  Ed25519 machinery in `v2/` is kept verbatim; only the **value/instruction section**
  grows from the 22 dataflow opcodes to the full `IRModule` instruction set. The MAP is
  versioned independently (`evidence_chain.schema = 1`); the binary body is `mic@3`.
- **mic@1 text is retained** as the human-readable/auditable view (a *feature* for an
  auditable language) and the debug/round-trip format — not deprecated, just no longer
  the shipped binary artifact.
- **Determinism is the gate — and it lives in a PUBLISHED canonical-encoding spec,
  not in the format choice.** `mic@3` emit must be byte-canonical with an explicit,
  documented ruleset: **sorted/lexicographic map keys, fixed-width length prefixes, no
  padding, deterministic float encoding (`to_bits`), and a fixed depth-first/pre-order
  traversal of nested control-flow regions.** The acceptance gate is **differential
  fuzzing**: a single reference encoder/decoder, fuzzed against the mic@1 text IR, until
  `SHA-256(canonical mic@3) == embedded trace_hash` holds for all semantically-equivalent
  modules. RFC-0001 `save→load→save` fixed-point is necessary but **not sufficient** —
  the published canonical ruleset is what makes the hash reproducible *across independent
  encoder implementations* (the self-host endpoint being a second one), which is the
  cross-substrate bit-identity wedge restated.

> **Cross-model validation (2026-05-27, 5/5 unanimous: deepseek-v4-pro / mistral-large /
> grok-4.3 / glm-5 / kimi).** mic@3 is the right call; both alternatives are rejected on
> auditability grounds — text-only fails bit-identity (formatting is non-canonical) and a
> detached SLSA/in-toto/Sigstore sidecar introduces a separable trust root with
> **desync risk** (signature survives while its binding to the artifact drifts). The
> unanimous principle: *evidence must be inseparable from the exact bytes the compiler
> emits.* The unanimous caveat is the canonical-spec + differential-fuzz gate above. One
> dissent worth weighing (glm-5): prefer a **schema-driven serializer over hand-rolled
> bit-packing** to guarantee cross-implementation reproducibility — though off-the-shelf
> protobuf is not canonical by default, so the same canonicalization discipline still
> applies. Q3 was also unanimous: **converge to two formats** (text mic@1 + binary mic@3);
> the weak dataflow binary is demoted, never a first-class third format (§3.3).

This is more work than a text epilogue (a full binary `IRModule` encoding incl.
control-flow regions), but it is the correct endpoint and avoids a throwaway mechanism.
Rejected alternative `mic@1e` (text fence on mic@1) is recorded in §3.0.

### 3.3 Demote v2 `Graph` to a model-exchange format: **`mind-model@2`**
The v2 lineage is honest as a *tensor model-exchange* format for the graphs that **are**
expressible in its dataflow opcodes — rfn-mind / MindLLM tensor graphs, and 512-mind's
MAP `dump` (its only documented v2 use). Rename the lineage in docs to break the
"compiler IR" implication; keep `emit_micb`/`parse_micb` for that role. It is **not**
the compiled-artifact IR and **not** the general evidence anchor.

### 3.4 Honesty about the governance chains (supersedes RFC 0016 §7's aspiration)
512-mind ships its **own** DIFC evidence chain (`observer.mind::EvidenceChain`,
sha3_512 / Hash512) — today it is a **genuinely separate** chain, not a consumer of
the mic@3 `trace_hash`. RFC 0016 §7's "consumes, does not parallel" is **aspirational
until wired**. Action: 512-mind records the mic@3 `trace_hash` as a proof input, or
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

0. **(SHIPPED, db5cb76)** `ir::ir_trace_hash` anchors `trace_hash` on canonical mic@1
   IR bytes — immediate correctness for "evidence attests the real IR," independent of
   the container work below.
1. **(SHIPPED, 5c29f0d)** `mic@3` binary encoding of the full `IRModule`
   (`src/ir/compact/v3/`: `emit_mic3`/`parse_mic3`/`Mic3Error`) — all 37 `Instr`
   variants incl. nested control-flow regions + the std-surface registries. Canonical +
   RFC-0001-deterministic (string table fixed traversal order, exports sorted, BTreeMap
   registries, f64 via `to_bits`); tests cover per-variant round-trip, save→load→save
   fixed point, byte-determinism, and a mic@1-decode cross-check. Additive `MIC3` magic +
   `MicFormat::Mic3`/`LoadError::Mic3`; mic@1, mic@2/MIC-B, std, and the self-host
   bootstrap are byte-untouched (verified: builds 0 warnings, full workspace 0 failures,
   bootstrap byte-identity 7/7).
2. **(SHIPPED, c64bd0b)** Evidence MAP epilogue on mic@3 — `src/ir/compact/v3/evidence.rs`:
   `emit_mic3_with_evidence` appends a self-contained, lexicographically-sorted MAP
   (`evidence_chain.{determinism,schema=1,substrate,toolchain,trace_hash[,parent]}`) after
   the canonical mic@3 body; `mic3_evidence_report` peels the body, recomputes
   `ir_trace_hash` (canonical mic@1), and verifies. Reuses v2 `Determinism`/`EvidenceReport`/
   `EvidenceError` (one vocabulary). 12 differential-fuzz/canonical-gate tests
   (byte-determinism, round-trip, tamper-detection, sorted-keys); no-evidence bytes
   byte-identical; verified v3 33/45, full workspace green, bootstrap 7/7. `trace_hash` is
   over the canonical mic@1 IR (the §3.1 anchor), MAP excluded per §3.2.
3. **(SHIPPED, 7fc10d2)** `mindc <file> --emit-mic3 <path>` / `--emit-evidence <path>` emit
   a mic@3 artifact (the latter with the embedded `evidence_chain` MAP: substrate =
   `--target` id, toolchain = mindc version, determinism = Deterministic w/ TODO(#289),
   trace_hash = `ir_trace_hash`). 6 CLI round-trip tests (emit → `mic3_evidence_report`
   verifies). Additive — default build + `--emit-mic`/`--emit-obj`/`--emit-shared`
   byte-unchanged; bootstrap 7/7. (Ed25519 signing at release-tag time still pending §3.4.)
4. **`mindc verify --evidence`** reads a mic@3 artifact, recomputes `trace_hash` over its
   IR body, compares, validates the signature (shares the RFC 0017 surface, #290).
5. **Demote** v2 `Graph` → `mind-model@2`. ⚠️ **This is a byte-preserving cross-repo
   migration, NOT a doc rename** (arch review 2026-05-27). The v2 `Graph`/MIC-B is a
   published `mind-spec` wire contract with live external consumers: **512-mind** hashes
   over the MIC-B byte layout (`canonical_ir.mind:142` `SPEC_HASH`), **mind-mem** ships a
   second implementation (`src/mind_mem/mic_map.py`, `MICB_VERSION=0x02`), **mind-nerve**
   uses mic@2-text IPC. The rename must **not** change `MICB_VERSION` or byte layout, and
   must land as a coordinated `mind-spec` version bump with 512-mind's `SPEC_HASH` +
   mind-mem's pin updated in lockstep — tracked as cross-repo deliverables. Resolve the
   `MICB_VERSION`/`Mind.toml` drift first (#308).
6. **`oracle.rs`** + CI wiring for the bit-identity gate (#307).

Each step is independently testable and additive; mic@1 text byte-output and the
keystone bootstrap are never regressed (mic@3 is added alongside, not in place of, mic@1).

## 5. Acceptance

1. An evidence-free `mindc build` emits byte-identical mic@1 to pre-RFC output;
   `phase_g_keystone_bootstrap` 7/7 unchanged.
2. `mindc build --emit=evidence` produces a **mic@3** artifact whose `trace_hash` equals
   the canonical-IR-bytes hash of its own IR body; `mindc verify --evidence` round-trips it.
3. The cross-substrate gate runs in CI and hard-fails on a deliberately corrupted
   reference hash (no self-skip on release tags).
4. `docs/ir-stability.md` describes exactly one compiled-artifact IR (mic@1 text /
   mic@3 binary, same `IRModule` data model) and names `mind-model@2` as the
   model-exchange format.

## 6. First steps (highest leverage, in order)

1. Spec + implement the **mic@3** binary `IRModule` encoding (§4.1) with the
   `save→load→save` fixed-point test — the foundational, keystone-safe addition
   (a new emit target; mic@1 and the bootstrap are untouched).
2. Attach the MAP/Ed25519 container to mic@3 + wire `mindc build --emit=evidence` (§4.2–4.3).
3. `oracle.rs` + CI gate for bit-identity (§3.5 / #307).

> Note: mic@3 is **added alongside** mic@1 — it does not modify the frozen mic@1 parser,
> so it does not risk the self-host keystone (a key advantage over the rejected mic@1e
> fence, which would have). The §306 std heap-OOB remains the separate keystone-touching
> item.
