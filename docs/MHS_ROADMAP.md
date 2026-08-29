# MHS / Governed Device I/O Roadmap — SUPERSEDED

> ## SUPERSEDED — do not execute this roadmap
>
> **Superseded by [`docs/rfcs/0027-governed-physical-device-plane.md`](rfcs/0027-governed-physical-device-plane.md)**
> (2026-08-28), which is the authoritative architecture decision, taken against the live codebase
> rather than against a proposal document. This roadmap sequences
> [`docs/rfcs/DRAFT-governed-device-io-mhs.md`](rfcs/DRAFT-governed-device-io-mhs.md) — a draft
> whose central proposal was **rejected on review** — so every milestone below is superseded with
> it. It is kept rather than deleted so the public record shows the change instead of hiding it;
> the body is unedited except for the correction callouts marked `> **SUPERSEDED ...**`. No
> milestone below is a plan of record, and nothing here has shipped.
>
> **Which claims here are wrong, specifically:**
>
> 1. **`M0 — Contract freeze` and `M1 — Canonical Device IR` are rejected in full.** RFC 0027
>    §0.2 forbids a **"Canonical Device IR"**, a `device.*` opcode, and any `mic@N` bump.
>    Actuation produces a **record, not a computation**, and records already have a carrier — the
>    MAP epilogue. The decided design spends exactly **one reserved MAP key**,
>    `evidence_chain.device_receipts`, reusing the RFC 0024 collapse-receipt pattern
>    byte-for-byte: a TLV blob, **omitted entirely when empty** (so an artifact that actuates
>    nothing stays byte-identical and no `mic@N` bump is required), **outside** the `trace_hash`
>    preimage, and **inside** the signature preimage. There is nothing to "freeze": the semantic
>    objects M0 would freeze — manifest, capability, state snapshot, action request/result — and
>    the four effect classes are not the schema. The schema is the single `DeviceActionReceipt`
>    of RFC 0027 §4.1.
>
> 2. **There is no `device.*` MAP namespace in this codebase, and a bare one would be a forgery
>    surface.** The reserved prefixes are exactly `["evidence_chain.", "signature."]`; any other
>    dotted key is an **application** key — unreserved, and writable by any program. An actuation
>    record under an application key could be forged by the very program it is supposed to
>    constrain (RFC 0027 §0.2).
>
> 3. **The external hardware standard is a boundary codec, never an internal representation.**
>    M0's "explicitly separate MIND Device IR from the external MHS wire/driver specification"
>    presumes two representations; RFC 0027 §0.3 has one codec spoken by **exactly one
>    component** — the gateway edge codec in the private `mind-runtime` — never an internal
>    representation, and never a type that leaks inward.
>
> 4. **`M4 — MHS reconciliation gate` may not be read as a promise of future conformance.**
>    Conformance to an unpublished external specification is not claimed, and the correct public
>    statement is that we speak *a* device codec at the edge — not that we conform to any named
>    standard (RFC 0027 §0.3). RFC 0027 accordingly names no external vendor and no standard.
>
> 5. **A device is a stochastic island.** The *decision* is deterministic and bit-identical
>    across substrates; the *physical outcome* is not — a motor stalls, a sensor drifts, a
>    gripper slips. The phrase "deterministic device control" is a doc-honesty violation, and
>    `OutcomeUnknown` is a **first-class outcome, not an error case** (RFC 0027 §0.7). No
>    milestone below provides for it.
>
> 6. **Ownership is three repos, not a public device stack.** `mind` **defines** (schema +
>    verifier — a verifier that needs a private repo to know what it is verifying is not a
>    verifier), `512-mind` **governs** (admissibility), `mind-runtime` **performs** (actuation +
>    edge codec). Every other repo is a consumer (RFC 0027 §0.1, §0.6). Actuation must not be
>    coupled to the intent, nerve, flow, or agent components: coupling a physical actuator to a
>    skeleton component puts a motor behind a build failure (§0.4).

---

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

This is the public MIND-language side of the ecosystem MHS initiative. It owns types, canonical
bytes, effect semantics, compiler/verifier behavior, a mock backend, and conformance vectors. It does
**not** own production device drivers, enterprise policy, fleet management, or user-facing device
orchestration.

> **SUPERSEDED (RFC 0027 §0.1, §0.2, §0.3) —** The ownership split above is wrong in three
> places. `mind` does not own "types" or "effect semantics" for devices — it owns the receipt
> **schema** and the **verifier**, and nothing else (§0.1, §0.2). It owns no conformance vectors
> for an external standard, because conformance to an unpublished external specification is not
> claimed (§0.3). What is correct is the negative half: production drivers, enterprise policy,
> fleet management and device orchestration are not this repo's — actuation and the edge codec
> live in the private `mind-runtime`, and admissibility in `512-mind` (§0.6).

## M0 — Contract freeze

- Adopt `docs/rfcs/DRAFT-governed-device-io-mhs.md`.
- Freeze semantic objects: manifest, capability, state snapshot, physical action request/result.
- Freeze effect classes: Observe / Configure / Actuate / Procedure.
- Define canonical value encoding with no raw floats in signed cross-language payloads.
- Explicitly separate MIND Device IR from the external MHS wire/driver specification.

## M1 — Canonical Device IR

- Add canonical serialization and hash vectors.
- Add typed mock device descriptors and state snapshots.
- Bind `manifest_hash` and `pre_state_hash` into action identity.
- Add idempotency semantics.

> **SUPERSEDED (RFC 0027 §0.2, §0.3) —** M0 and M1 are rejected in full: no Canonical Device IR,
> no frozen device object set, no `manifest_hash`/`pre_state_hash` action identity, no
> separation of a "MIND Device IR" from an external wire specification — because there is no
> Device IR to separate. The one thing that lands is a single reserved MAP key,
> `evidence_chain.device_receipts`, carrying a fixed-width TLV `DeviceActionReceipt` (RFC 0027
> §4.1) that binds `(ingress_hash, decision_hash, command_bytes_hash, outcome)` plus
> `artifact_trace_hash` and `prev_receipt_hash`. `command_bytes_hash` is taken *after* codec
> encoding, so the record covers what the hardware actually received rather than what we
> intended to send.

## M2 — Compiler physical-effect awareness

- Preserve physical-effect classification through lowering.
- Add fail-closed diagnostics for undeclared/unknown effects.
- Ensure a generic FFI/network/process/MCP call cannot silently claim `Observe`.
- Add replay mode that consumes recorded results and cannot actuate.

> **SUPERSEDED (RFC 0027 §0.2, §0.5, §4.2) —** There is no physical-effect classification to
> preserve through lowering and no new IR, so M2's first three items are rejected. `mindc verify`
> gains exactly **one** check: if `evidence_chain.device_receipts` is present, every receipt must
> be well-formed, `artifact_trace_hash` must equal the artifact's own `trace_hash`, and
> `prev_receipt_hash` must chain; absent the key, behaviour is unchanged. The replay item is
> narrowed to two modes, not a mode flag: **replay** re-derives and checks an evidence chain and
> never actuates, while **re-execution** is a new authority requiring a fresh admissibility
> proof.

## M3 — Public host ABI + mock backend

- Stable DeviceGateway/DeviceBackend host ABI.
- In-tree deterministic mock backend.
- Golden tests on x86_64 + ARM64.
- Reference examples: temperature sensor, bounded actuator, deterministic multi-step procedure.

> **SUPERSEDED (RFC 0027 §0.5, §0.6, §7) —** The gateway lives in the private `mind-runtime`,
> not behind a public host ABI, and it accepts a command from **any** producer carrying a valid
> admissibility proof (§0.4). Receipts are **not** part of the cross-substrate identity gate —
> the artifact's bytes are unchanged by actuation, and two substrates may legitimately produce
> different receipts (§5) — so the x86_64/ARM64 golden test here is not the gate. The decided
> gate is: **replace the device adapter with a panicking stub and run `mindc verify` — if verify
> needs the device, the boundary is fake** — asserting a **positive count** of receipts verified,
> because a run that verifies zero receipts and exits 0 proves nothing. Second gate: a replay run
> must actuate **zero** times against that stub.

## M4 — MHS reconciliation gate

Triggered only when Anthropic publishes a normative open-source MHS specification:

- pin a supported MHS version;
- publish a field mapping from MHS to Device IR;
- add compatibility and negative vectors;
- document unsupported semantics explicitly;
- only then advertise MHS compatibility/conformance at the supported profile.

> **SUPERSEDED (RFC 0027 §0.3) —** Conformance to an unpublished external specification is not
> claimed, and this milestone is not a commitment to claim it later. The standard is a boundary
> **codec** spoken by one component in the private runtime; the correct public statement is that
> we speak *a* device codec at the edge, not that we conform to any named standard. RFC 0027
> deliberately names no external vendor and no standard; the naming in this document is retained
> as historical record, not as a claim.

## M5 — Spec promotion

After the public RFC and implementation stabilize, promote the interoperable semantics to
`star-ga/mind-spec` without changing Core v1 retroactively.

> **SUPERSEDED (RFC 0027 §0.1, §0.6) —** The schema does not get promoted out of `mind`. It lives
> where the verifier lives, because `mindc verify` ships in the **public** compiler and a
> verifier that needs a private repo to know what it is verifying is not a verifier — the
> definition site is RFC 0027 §4.1, extending `src/ir/compact/v3/evidence.rs`. A private
> specification repo is the coordination home for threat model, deployment tiers, hardware notes
> and rollout; where it restates a schema detail it is a **mirror** and must be marked as one.
