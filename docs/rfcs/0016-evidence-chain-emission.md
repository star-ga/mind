# RFC 0016: Compile-Time Evidence-Chain Emission

| Field | Value |
|---|---|
| RFC | 0016 |
| Title | Compile-Time Evidence-Chain Emission |
| Status | **Partial** — Phase A shipped (Rust-bootstrap emit) + Phase B verifier core (lib) + **Phase B-CLI shipped** (`mindc <file> --emit-evidence <path>` emits an embedded, **unsigned but well-formed** `evidence_chain` MAP on a mic@3 artifact, with a built-in verifier-core self-check; `mindc verify <artifact>` recomputes + reports); **Phase C SHIPPED** (crypto-agile Ed25519 / ML-DSA-65 / hybrid signing over the canonical provenance preimage — additive, optional, **opt-in via a key-seed env var, never signed-by-default**; unsigned artifacts stay byte-identical, `schema` stays Int 1, no `mic@N` bump); Phases A.5 (pure-MIND emit) / D (#289 determinism-default) / E (agent + governance links) pending. The chain is **no longer inert** — it is reachable from the CLI, round-trip-verifiable, and optionally signed; only wiring signing into the release CI runner is deferred (RFC 0020 §5.3 std-crypto). |
| Authors | STARGA Inc. |
| Created | 2026-05-26 |
| Task | #288 |
| Carrier | mic@2.1 MAP `evidence_chain.*` namespace ([mic2.1-spec.md](../../../mind-spec/spec/mic/mic2.1-spec.md) §4, §6) |
| Related | RFC 0011 (ReplayScheduler trace-hash — the runtime trace source), RFC 0014 (per-substrate lowering — the `substrate` field), RFC 0015 (cross-substrate bit-identity — the attested property), RFC 0017 (`mindc verify` — the cross-target certificate the chain can carry), RFC 0019 / #294 (deterministic agent substrate — `agent.*` links INTO the chain), RFC 0020 (#303 mind-bench — the public wedge-score is a chain link), 512-mind DIFC `proof_chain` (governance consumer, NOT a parallel chain) |

---

## 1. Motivation

The wedge is cross-substrate bit-identity (RFC 0015): the same MIND source
produces a byte-identical artifact and a byte-identical execution trace on AVX2,
NEON, and every claimed substrate. Today that property is **asserted** — provable
by a STARGA engineer running `tests/cross_substrate_identity/`, or a reviewer
running `mind-bench verify` (RFC 0020), but **not carried by the artifact itself**.
A downstream auditor holding a compiled `.mic` binary has no in-band, tamper-evident
record of *how it was produced, on what substrate, from what parent, and whether
that production was deterministic*.

Backed by the strongest signal from independent cross-review: **evidence-chain
emission as a language-native output** is the single most-agreed-upon move. The
finding: a determinism claim that lives only in a test harness sits in the same
epistemic bucket as an unreproducible benchmark number; it cannot enter a
*downstream* agent's or auditor's evidence chain. Emission turns the internal
property into a portable, composable, signed artifact-level fact.

**This RFC defines the semantics of `evidence_chain.*`** — what it attests, how
`mindc` emits it at compile time, the parent-pointer DAG that links a compilation
through its transformation steps, and how it unifies the compile / runtime /
governance layers through a single carrier (mic@2.1 MAP) and a single signer
(mind-mem Ed25519) — explicitly **without** inventing a fourth hash chain.

**Failure mode if not shipped:** the bit-identity property stays whitepaper +
test-harness only; "deterministic compute" never becomes "deterministic, *auditable*
compute"; RFC 0015 and RFC 0020 look like internal hygiene rather than the
load-bearing, externally-citable property they are; and the AGI/ASI-substrate
thesis ("agents emit evidence, not just output") has no compile-layer foundation
for RFC 0019 to build on.

## 2. Non-goals

- **Not a new wire format.** The carrier is mic@2.1 MAP (already specified). This
  RFC defines key *semantics*, not encoding.
- **Not a new signer.** Signing is mic@2.1 §6 (Ed25519 via `mind_mem.model_signing`).
  One primitive across mind-mem / mic@2.1 / evidence-chain (anti-fragmentation; §7).
- **Not a new hash chain.** `evidence_chain` reuses the IR canonical hash (mic@2.1
  binary) and the RFC 0011 trace-hash. 512-mind's `proof_chain` and mind-mem's
  governed-write **consume** evidence-chain hashes; they do not parallel them (§7).
- **Not a network/consensus protocol.** No distributed ledger, no global ordering.
  An evidence chain is a local, per-artifact Merkle-DAG; trust is rooted in the
  signer's pubkey, resolved out-of-band (mic@2.1 §6).
- **Not a throughput claim.** Evidence carries determinism + provenance, never
  "faster than X". Timing, if present, is informational (RFC 0020 §2 discipline).

## 3. The evidence-chain object

An **evidence link** is the `evidence_chain.*` MAP block of a single mic@2.1
artifact. It records, for the graph it is attached to:

| Key | Type | Meaning |
|---|---|---|
| `evidence_chain.trace_hash` | `bytes` (32) | The deterministic production hash of THIS artifact: **SHA-256** (FIPS 180-4) of the canonical mic@2.1 **binary** of the graph (MAP minus `signature.*` and minus `evidence_chain.trace_hash` itself — §3.2). SHA-256 is fixed (not BLAKE3) so the bootstrap's Rust hash and pure-MIND `std.sha256` are bit-identical (§10 Q1). For an *executed* artifact it additionally folds the RFC 0011 `ReplayScheduler` trace-hash (§3.3). |
| `evidence_chain.substrate` | `string` | The RFC 0014 canonical lowering-tier id this artifact was produced for/on (`x86_avx2`, `arm_neon`, `nvptx_sm80`, `portable_scalar`, …). The bit-identity claim is *per-substrate-set*; this names the member. |
| `evidence_chain.parent` | `bytes` (32) | The `trace_hash` of the predecessor artifact in the transformation chain (source-parse → typecheck → IR → lowered → binary), or absent for a root. Forms the DAG (§4). |
| `evidence_chain.determinism` | `string` | `"deterministic"` (default) or `"nondeterministic"`. Set by the §5 emission rule from the graph's determinism attribute (RFC 0012 Phase C / #289). A `nondeterministic` link is a *refusal-to-attest-identity* marker, not a silent omission. |
| `evidence_chain.toolchain` | `string` | `mindc` version that produced the link (e.g. `0.7.0`), so a verifier can re-derive with the same lowering pipeline (RFC 0015 reproducibility anchor). |
| `evidence_chain.identity_set` | `string` (optional) | Comma-sorted list of substrates this artifact claims byte-identity *across* (`avx2,neon`), when the producer asserts a cross-substrate set rather than a single member. Verified by RFC 0017 / RFC 0020. |

### 3.1 What a link attests (and what it does not)

A signed evidence link asserts: **"`mindc <toolchain>` produced an artifact whose
canonical hash is `trace_hash`, for substrate `substrate`, from parent `parent`,
and the production was `determinism`."** That is a statement about *provenance +
determinism of one production step*.

It does **not** by itself prove cross-substrate identity — that is RFC 0015's
proof obligation, surfaced two ways: (a) `identity_set` + the RFC 0020 mind-bench
signed wedge-score receipt embedded as a sibling `verify.*` block (RFC 0017),
which a third party re-runs; or (b) two sibling artifacts with the same logical
`parent` but different `substrate` whose `trace_hash` values are equal for a
Q16.16 graph (the inspectable cross-substrate equality, RFC 0015 §3.1). The link
*carries* the claim; the verifier *discharges* it. This separation is deliberate:
the artifact never self-certifies a property it cannot locally prove.

### 3.2 The self-reference rule

`trace_hash` cannot hash itself. The bytes-to-hash are the canonical mic@2.1
binary with **both** `signature.*` **and** `evidence_chain.trace_hash` removed
(every other `evidence_chain.*` key IS hashed — substrate/parent/determinism/
toolchain are part of what's attested). The verifier recomputes over
MAP-minus-{`signature.*`, `evidence_chain.trace_hash`} and checks equality, then
checks the §6 signature over MAP-minus-`signature.*` (which *includes* the
now-populated `trace_hash`). Two nested exclusions, two checks — mirrors mic@2.1
§6.1's signature-omits-itself discipline.

### 3.3 What `trace_hash` covers — the canonical mic@3 IR (GAP-1, re-anchored)

> **Correction (architecture audit 2026-05-26).** Phase A/B computed `trace_hash`
> over the v2 compact [`Graph`](../../src/ir/compact/v2) (mic@2.1). But the
> compile pipeline never produces that graph for a real program — it has 22
> pure-dataflow opcodes (no control flow / functions / SIMD) and the only
> non-test `Graph` is a hardcoded fixture, so that hash attested a toy, not the
> shipped artifact. The 2026-05-26 fix re-anchored the attested object onto the
> canonical `IRModule` (`ir::save(&IRModule)`), computed by
> [`ir::ir_trace_hash`](../../src/ir/evidence.rs).
>
> **Re-anchor (collision audit 2026-05-31).** The 2026-05-26 fix anchored on
> mic@1 IR *text*, but mic@1 text is **lossy for function bodies**: `ir::print`
> emits an `Instr::FnDef` as a bare `// fn <name>` comment and drops its
> `params`/`ret_id`/`body`, so two exported functions that differ only in body
> (e.g. `f(x) { x + 2 }` vs `f(x) { x * 999 }`) serialize to byte-identical
> mic@1 text and collide to the same SHA-256 — attesting the signature surface,
> not the computation. **The attested object is therefore the canonical mic@3
> bytes** (`emit_mic3(&IRModule)`), the full-fidelity binary IR that commits
> complete function bodies. mic@3 is round-trip equivalent to mic@1 (the
> documented stable contract, `docs/ir-stability.md`), RFC-0001 deterministic (a
> byte-identical fixed point across runs and platforms), and is itself the object
> the cross-substrate byte-identity gate compares — the substrate-independent
> **IR link** of the §4 DAG whose cross-substrate equality RFC 0015 asserts. This
> supersedes the original GAP-1 mic@1-text rule. The mic@2.1 `Graph` path is the
> **model-exchange** carrier (rfn-mind / MindLLM tensor graphs that *are*
> expressible in the v2 opcodes) and the evidence **container** (the MAP holds
> `evidence_chain.*` + the §6 signature); it is not the general compiled-artifact
> anchor.

For a *compiled* artifact the production is pure (parse/lower are deterministic
functions of source + toolchain), so `trace_hash = SHA-256(canonical mic@3 bytes)`
via the FIPS-180-4 seam (§5.4). For an artifact that has been *executed* under
the RFC 0011 `ReplayScheduler` (e.g. a mind-bench workload, an agent step), the
executed trace-hash (`std.async.trace_hash`) is folded in:
`trace_hash = H(mic1_ir_hash || replay_trace_hash)`. This is the seam where
compile-time evidence meets runtime evidence — and where RFC 0019's `agent.*`
links attach (§7).

## 4. The chain — a per-artifact Merkle DAG

`parent` pointers link transformation steps into a directed acyclic graph rooted
at the source. A canonical compile chain:

```
source.mind ──parse──▶ [IR link, parent=H(source)]
                          │ lower(avx2)            │ lower(neon)
                          ▼                        ▼
              [avx2 binary link,        [neon binary link,
               parent=H(IR)]             parent=H(IR)]
                          └────────── RFC 0015: trace_hash(avx2) == trace_hash(neon)
                                       for a Q16.16 graph ◀── inspectable identity
```

- **Linearity is not required.** A lowering fan-out (one IR → N substrate binaries)
  is N links sharing one `parent` — exactly the structure RFC 0015's
  "same-hash-across-substrates" gate inspects.
- **The DAG is local and bounded.** No global chain, no cross-artifact ordering
  beyond `parent`. Cross-artifact references (e.g. RFC 0017 referencing another
  target's certificate) ride `verify.*`/`bytes()`, not `evidence_chain.parent`.
- **Root links** (no `parent`) are source artifacts or externally-supplied inputs;
  their trust is the signer's, not a predecessor's.
- **Verification is a fold:** walk child→parent, at each step recompute the child's
  `trace_hash`, confirm it commits to `parent`, confirm the signature. A broken or
  forged intermediate link fails its own hash recomputation — tamper-evident.

## 5. Emission model — compile-time, determinism-gated

Evidence is emitted by `mindc build`/`mindc emit`, not a separate tool — it is a
language-native output (the 4/4 finding). Rules:

1. **On by default for deterministic graphs.** Any graph whose effective
   determinism is `deterministic` (the default once #289 lands; today, any graph
   not marked `#[nondeterministic]`) emits a full `evidence_chain` block. No flag
   enables it; emission is part of producing the artifact. (A `--no-evidence`
   *diagnostic* flag MAY exist for unsigned scratch builds, but the signed release
   path always emits — never a determinism escape hatch, per RFC 0013 §8.)
2. **Nondeterministic graphs emit a refusal marker.** A `#[nondeterministic]`
   graph emits `evidence_chain.determinism = "nondeterministic"` and **omits**
   `identity_set` — it explicitly declines to attest cross-substrate identity
   rather than silently shipping a bare artifact. This makes the absence of an
   identity claim itself an inspectable, signed fact (you cannot launder
   nondeterminism by simply not emitting evidence).
3. **`parent` is threaded by the compile driver.** Each pass that produces a new
   canonical artifact (`mindc build`'s IR-emit, then per-substrate lower) sets
   `parent` to the prior step's `trace_hash`. The driver owns the threading; the
   passes stay pure.
4. **Signing is a release-pipeline step**, not every local build. **The Rust
   bootstrap `mindc`** emits the unsigned `evidence_chain` (the self-host MIND
   compiler emits MLIR-text today and has no mic backend; it gains mic-emit +
   evidence parity at Phase A.5 — §8). The RFC 0015 canonical CI runner (the same
   one that produces RFC 0020's `wedge-reference-hashes`) signs at release-tag time
   via mic@2.1 §6. Unsigned local artifacts carry evidence; signed release artifacts
   carry evidence + `signature.*`.
   - **Codec-agnostic trace_hash (load-bearing — keeps the pure-MIND endpoint open).**
     `trace_hash` is computed over the **canonical mic@2.1 binary serialization
     bytes** (`emit_micb` of the graph with its MAP stripped of `signature.*` and
     `evidence_chain.trace_hash` per §3.2), hashed as a flat byte string with the
     FIPS-180-4 primitive. It is NEVER a walk of an in-memory codec struct. Because
     the input is canonical bytes and the hash is FIPS-180-4, the value is
     **bit-identical** whether computed by the bootstrap's Rust SHA-256
     (`src/deps/mod.rs`) or by pure-MIND `std.sha256` at the self-host endpoint —
     the same duality already shipped for the lexer/parser (Rust mindc + MIND
     self-host, both legitimate). The hash call site is a thin one-function seam so
     swapping Rust→`std.sha256` is a one-line change. A frozen FIPS conformance
     vector pins the byte definition for the future MIND emitter.
5. **Determinism of emission itself.** The `evidence_chain` block is canonical
   (mic@2.1 §5 sorts keys); emitting it must not perturb byte-identity of the graph
   proper (MAP is a trailing epilogue, mic@2.1 §3.1). Re-emitting an artifact
   reproduces identical evidence bytes — the emitter is part of the wedge it measures.

## 6. Signing

Reuses mic@2.1 §6 verbatim: Ed25519 over the canonical binary with `signature.*`
removed, primitive = `mind_mem.model_signing` (RFC 8032). The evidence chain adds
no new signing surface — `evidence_chain.*` keys are ordinary MAP keys covered by
the §6 signature. A signed artifact therefore commits, in one signature, to its
graph **and** its provenance/determinism evidence. Key resolution
(`signature.pubkey` inline or out-of-band registry) is mic@2.1 §6's concern.

## 7. Ecosystem fit — one chain, consumed at three layers

The architect's load-bearing catch (banked 2026-05-24): #288 must **unify**, never
fork, the ecosystem's hashing/signing. Three governance-ish chains exist; this RFC
makes them one chain with three consumers:

| Layer | Surface | Relationship to `evidence_chain` |
|---|---|---|
| **Compile** | RFC 0016 `evidence_chain.*` (this RFC) | The chain itself. |
| **Runtime / agent** | RFC 0019 `agent.*` (#294) | An agent step's `agent.trace` links to the `evidence_chain.trace_hash` of the IR that ran (the §3.3 fold). Agent cognition inherits compile-time evidence. |
| **Governance** | 512-mind DIFC `proof_chain` (Hash256/Hash512) | **Consumes** evidence-chain hashes as proof inputs; does NOT re-derive a parallel chain. A governed write records the `evidence_chain.trace_hash` it authorized. |

Signer unification: **one** Ed25519 chain (mind-mem `model_signing`) signs mic@2.1
artifacts, mind-mem manifests, RFC 0020 wedge-score receipts, and these evidence
chains. 512-mind's Hash256/512 are *content* hashes feeding its DIFC proofs, not a
competing signature authority — they reference evidence_chain outputs. **No fourth
chain is introduced; #288 closes the fragmentation risk rather than opening it.**

This is the "deterministic compute → deterministic cognition" spine: evidence at
compile, `agent.*` at runtime, `proof_chain` at governance, all carried by mic@2.1
MAP and rooted in one signer.

## 8. Phased plan

- **Phase A — schema + emit (inert, unsigned), Rust bootstrap. ✅ SHIPPED**
  (`src/ir/compact/v2/evidence.rs`: `attach_evidence_chain` /
  `compute_trace_hash` / `remove_evidence_chain` / `Determinism`; `trace_hash`
  hashes `emit_micb` bytes via the `crate::deps::mini_sha256` seam; 15 tests
  incl. the frozen FIPS 180-4 conformance vector + a pinned residual-block
  `trace_hash`; bootstrap byte-identity 7/7 preserved.) Define the
  `evidence_chain.*` keys (§3) in `mind/src/ir/compact/v2/` MAP support; the **Rust
  bootstrap `mindc`** emits `trace_hash`/`substrate`/`parent`/`determinism`/`toolchain`
  for deterministic graphs. `trace_hash` = SHA-256 of `emit_micb(MAP-stripped per
  §3.2)` **bytes** (NOT a struct walk — Risk 1), via a thin one-function seam over
  the existing Rust `src/deps/mod.rs` SHA-256. No signing yet. Gates:
  `emit(parse(emit(G))) == emit(G)` with evidence present (mic@2.1 §5); the cdylib
  bootstrap stays byte-identical (evidence is a mic-epilogue, the `.so` carries no
  MAP — §9.3); **and a frozen FIPS conformance vector** pins the canonical-byte
  definition so the future MIND emitter has an exact target.
- **Phase A.5 — self-host mic-emit + evidence parity (deferred endpoint).** The
  self-host MIND compiler emits MLIR-text today and has no mic backend. When it
  gains one, it emits the identical `evidence_chain` using pure-MIND `std.sha256`
  over the same canonical bytes — bit-identical to Phase A by FIPS spec. This is
  the pure-MIND endpoint; it is NOT a precondition for Phase A (same Rust-bootstrap /
  MIND-self-host duality as the lexer/parser). Tracked, not omitted.
- **Phase B — verify.** `mindc verify --evidence <artifact>` recomputes `trace_hash`
  (§3.2 self-reference rule), folds the chain (§4), reports root + determinism +
  substrate. Shares the RFC 0017 verifier surface. **Library core SHIPPED**
  (`verify_evidence_chain` → `EvidenceReport`/`EvidenceError` in
  `src/ir/compact/v2/evidence.rs`): recomputes §3.2, flips `trace_hash_valid`
  on any post-seal mutation, decodes substrate/determinism/toolchain/parent and
  reports `is_root()`; survives a MIC-B round-trip; 5 tests incl. tamper
  detection. **CLI wrapper SHIPPED (Phase B-CLI, #309, RFC 0021 step 3+):**
  `mindc <file> --emit-evidence <path>` writes a mic@3 artifact with the
  embedded `evidence_chain.*` MAP (substrate = `--target` id, toolchain = mindc
  version, determinism = `Deterministic` w/ `TODO(#289)`, `trace_hash` =
  canonical mic@3 anchor), and runs a built-in verifier-core **self-check**
  (peel → recompute `trace_hash` → confirm `trace_hash_valid`) before the
  artifact is written — an emitter that produces an unverifiable artifact fails
  fast rather than shipping it. `mindc verify <artifact>` recomputes §3.2, folds
  the chain, and reports root + determinism + substrate + `trace_hash_valid`
  (human + `--json`); a single flipped byte fails verification. Emission is
  **unsigned but well-formed** — the `signature.*` block is the only deferred
  field (Phase C, below). Still pending on top: the cross-artifact chain *walk*
  (following `parent` across sibling artifacts) and the §6 signature check, which
  land with the RFC 0017 `mindc verify` certificate surface (#290).
- **Phase C — sign (release path). SHIPPED (additive, optional, opt-in).**
  `signature.*` keys are now emitted and verified as an **additive, optional**
  layer, selected per-artifact for crypto-agility: Ed25519 (RFC 8032), ML-DSA-65
  (FIPS-204, behind the optional `evidence-mldsa` feature, fail-closed when off),
  or the hybrid of both (both halves must verify). The signature covers the
  **canonical provenance preimage** — the 32-byte mic@3 `trace_hash` plus a
  canonical, lexicographically-ordered serialization of every other
  `evidence_chain.*` key (substrate, toolchain, determinism, parent, schema,
  trace_hash_kind), with `signature.*` and `trace_hash` themselves excluded per
  §3.2 / §6.1's sign-omits-itself discipline — so editing any provenance field
  (including the `parent` chain link) fails `mindc verify` closed. **Opt-in
  only:** signing runs solely when an operator supplies a key seed out-of-band
  (`MIND_EVIDENCE_ED25519_KEY` / `MIND_EVIDENCE_MLDSA_KEY`); there is **no
  "signed by default".** The wire is **additive** — `signature.*` keys sort after
  every `evidence_chain.*` key and are absent when no key is supplied, so an
  unsigned artifact is **byte-identical to the pre-signing encoder**, the evidence
  `schema` stays **Int 1**, and there is **no `mic@N` bump**. The signature never
  feeds back into `trace_hash`, so the determinism/keystone gate is untouched.
  Still deferred on top: wiring signing into the RFC 0015 canonical CI runner for
  release artifacts, and the RFC 0020 wedge-score receipt as a sibling `verify.*`
  block.
- **Phase D — determinism-default integration (#289).** Once `Tensor<f32>`
  requires explicit `#[nondeterministic]`, the §5.2 refusal-marker becomes the
  default-deny surface: an unmarked nondeterministic graph fails to emit a
  `deterministic` link and is caught at the gate.
- **Phase E — agent + governance links.** RFC 0019 `agent.*` and 512-mind
  `proof_chain` consume evidence hashes (§7). Long-horizon.

## 9. Acceptance

1. `mindc build` on a deterministic Q16.16 graph emits a canonical
   `evidence_chain` block; re-emit is byte-identical (mic@2.1 §5).
2. The same source lowered for `avx2` and `neon` yields two links with equal
   `parent` and — for a Q16.16 graph — equal `trace_hash` (RFC 0015 §3.1 made
   inspectable in-band).
3. Bootstrap byte-identity (Phase G keystone) is preserved with evidence emission
   on. **Two distinct byte-identity domains — neither is threatened:**
   (a) the Phase G keystone compares the compiled **cdylib `.so`** (MLIR→LLVM
   output), which carries no mic MAP, so evidence emission into the mic@2/MIC-B
   serialization cannot perturb it; (b) the RFC 0015 cross-substrate property is
   measured on the **graph / `trace_hash`** (MAP-stripped, mic@2.1 §3.1's
   separable epilogue). The `evidence_chain.substrate` field intentionally
   *differs* between an avx2 and a neon artifact — that is the recorded fact, not
   a violation; the graph and (for a Q16.16 graph) the `trace_hash` are identical
   across them. "On by default" (§5.1) governs mic-artifact emission, a different
   output than the cdylib the bootstrap pins.
4. `mindc verify --evidence` recomputes `trace_hash` and validates the §6 signature
   on a release artifact; a single flipped byte anywhere fails verification.
5. A `#[nondeterministic]` graph emits `determinism = "nondeterministic"` and omits
   `identity_set` — never a bare artifact.
6. One Ed25519 pubkey verifies a mind-mem manifest, a mic@2.1 artifact, and an
   evidence chain (signer unification, §7).
7. No new hash-chain type is added to the codebase (anti-fragmentation grep gate).

## 10. Open questions

1. **`trace_hash` function — RESOLVED: SHA-256, not BLAKE3.** mind-mem
   `model_signing` and 512-mind use SHA-256/Hash256, the Rust bootstrap has a
   FIPS-180-4 SHA-256 (`src/deps/mod.rs`), and `std.sha256` shipped pure-MIND
   FIPS-180-4. SHA-256 is now **load-bearing for the Rust↔MIND duality**: the
   bootstrap and the self-host endpoint must compute bit-identical `trace_hash`
   over the same canonical bytes, which only holds if both run the same FIPS
   algorithm. The cross-implementation-identity requirement outranks any BLAKE3
   throughput argument; content hashing is not the critical path. BLAKE3 dropped.
2. **Chain depth ceiling.** mic@2.1 §3.5 caps MAP size; a deep transformation DAG
   could exceed it via accumulated `parent` history. Resolution: a link carries
   only its *immediate* `parent` (one hash), not the transitive history — the DAG
   is reconstructed by collecting sibling artifacts, not by inlining ancestry.
   Confirm this keeps every link within mic@2.1 §3.5.
3. **Cross-toolchain-version verification.** A verifier on `mindc 0.8` re-deriving
   a `0.7`-produced `trace_hash` needs the 0.7 lowering pipeline. Resolution:
   `toolchain` field + RFC 0015's "reference hash per mindc version" archive
   (RFC 0020 §5) — verify against the recorded version, not the local one.
4. **Executed-artifact trace folding (§3.3) ordering.** Pin the fold as
   `H(canonical_binary_hash || replay_trace_hash)` (concatenation order normative)
   so it is itself reproducible; finalize when RFC 0019's agent-step profile lands.

## 11. References

mic@2.1 spec §4 (reserved `evidence_chain.*`), §5 (canonicalisation byte-identity),
§6 (Ed25519 signing — reused); RFC 0011 (ReplayScheduler `trace_hash`); RFC 0014
(substrate tier ids); RFC 0015 (cross-substrate bit-identity — the attested
property + the same-hash-across-substrates gate); RFC 0017 (`mindc verify` +
`verify.*` certificates); RFC 0019 / #294 (`agent.*` links); RFC 0020 / #303
(mind-bench wedge-score as a chain link); `mind/src/ir/compact/v2/` (MAP reference
impl); `mind-mem/src/mind_mem/model_signing.py` (the one signer); 512-mind DIFC
`proof_chain` (governance consumer); internal cross-review (evidence-chain
convergence).
