# Draft RFC: Governed Device I/O and MHS Compatibility — SUPERSEDED

> ## SUPERSEDED — do not implement this document
>
> **Superseded by [`docs/rfcs/0027-governed-physical-device-plane.md`](0027-governed-physical-device-plane.md)**
> (2026-08-28), which is the authoritative architecture decision, taken against the live
> codebase rather than against a proposal document. This draft was written *before* that
> decision and its central proposal was **rejected on review**. It is kept rather than deleted
> so the public record shows the change instead of hiding it; the body below is unedited except
> for the correction callouts marked `> **SUPERSEDED ...**`. Nothing below is a plan of record,
> and no implementation of either document has shipped.
>
> **Which claims here are wrong, specifically:**
>
> 1. **[`## Canonical Device IR`](#canonical-device-ir) is rejected in full.** RFC 0027 §0.2
>    forbids a "Canonical Device IR", a `device.*` opcode, and any `mic@N` bump. Actuation
>    produces a **record, not a computation**, and records already have a carrier — the MAP
>    epilogue. The decided design spends exactly **one reserved MAP key**,
>    `evidence_chain.device_receipts`, reusing the RFC 0024 collapse-receipt pattern
>    byte-for-byte: a TLV blob, **omitted entirely when empty** (so an artifact that actuates
>    nothing stays byte-identical to one compiled before the RFC existed, and no `mic@N` bump is
>    required), **outside** the `trace_hash` preimage (a device outcome is not a property of the
>    program), and **inside** the signature preimage (a record editable under a valid signature
>    is worthless). Every object named in that section — `DeviceManifest`,
>    `PhysicalActionRequest`, `PhysicalActionResult`, `DeviceStateSnapshot` and the rest — is
>    **not** the schema. The schema is the single `DeviceActionReceipt` of RFC 0027 §4.1.
>
> 2. **There is no `device.*` MAP namespace in this codebase, and a bare one would be a forgery
>    surface.** The reserved prefixes are exactly `["evidence_chain.", "signature."]`. Any other
>    dotted key is an **application** key: unreserved, and writable by any program. An actuation
>    record under an application key could be forged by the very program it is supposed to
>    constrain, so the receipt belongs under the reserved `evidence_chain.` prefix (RFC 0027
>    §0.2).
>
> 3. **The external hardware standard is a boundary codec, never an internal representation.**
>    The Summary below calls it "an external compatibility surface, while Device IR is the
>    canonical internal representation". RFC 0027 §0.3 inverts that: the standard is a wire
>    encoding spoken by **exactly one component** — the gateway edge codec in the private
>    `mind-runtime` — and it never leaks inward as a type. Conformance to an unpublished
>    external specification is **not claimed**; the correct public statement is that we speak
>    *a* device codec at the edge, not that we conform to any named standard. RFC 0027
>    accordingly names no external vendor and no standard.
>
> 4. **A device is a stochastic island — "deterministic device control" is a doc-honesty
>    violation.** The *decision* is deterministic and bit-identical across substrates; the
>    *outcome* is not, because a motor stalls, a sensor drifts, a gripper slips (RFC 0027 §0.7).
>    `OutcomeUnknown` is a **first-class outcome, not an error case**: a command that was issued
>    but whose result was never observed is the *normal* failure mode of physical hardware, and
>    a schema that cannot represent it will be lied to. The result vocabulary below has no such
>    outcome. Verification asserts a record is internally consistent and unforged; it does not
>    and cannot assert that the device physically moved.
>
> 5. **Two replay modes, not three.** *Replay* re-derives and checks an evidence chain and
>    **never actuates**. *Re-execution* is a new authority requiring a fresh admissibility
>    proof — not "replay with a flag". The three-mode split below is superseded by RFC 0027
>    §0.5, and the distinction is enforced by a gate, not by documentation.
>
> 6. **Three repos, not a public device stack.** `mind` **defines** (schema + verifier — a
>    verifier that needs a private repo to know what it is verifying is not a verifier),
>    `512-mind` **governs** (admissibility), `mind-runtime` **performs** (actuation + edge
>    codec). Every other repo is a consumer (RFC 0027 §0.1, §0.6). Actuation must not be coupled
>    to the intent, nerve, flow, or agent components: coupling a physical actuator to a skeleton
>    component puts a motor behind a build failure (§0.4).

---

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

## Summary

MIND needs a deterministic boundary between compiled/agentic intent and the physical world.
This RFC defines a vendor-neutral **Device IR** and explicit **physical-effect semantics** in the
public language/runtime contract. An MHS adapter may translate between this stable Device IR and
Anthropic's Model Hardware Standard. The MIND contract is deliberately not a competing wire
standard: MHS remains an external compatibility surface, while Device IR is the canonical internal
representation that protects the language from changes in a research-preview protocol.

> **SUPERSEDED (RFC 0027 §0.2, §0.3) —** There is no Device IR. Actuation produces a *record*,
> not a computation, and the record rides one reserved MAP key,
> `evidence_chain.device_receipts`. The external hardware standard is a boundary **codec**
> spoken by one component in the private runtime — never "the canonical internal
> representation", and never a type that leaks inward.

The core rule is:

```text
reason/propose -> compile/normalize -> authorize -> physical-effect barrier -> device -> observe -> receipt
```

A program may compute arbitrary proposals. A physical side effect is a distinct effect class and
MUST cross one declared gateway. A normal function call, FFI call, network write, process spawn, or
MCP invocation MUST NOT be able to masquerade as an unclassified physical effect.

## Goals

1. Give MIND programs one typed representation for discoverable devices, capabilities, state,
   physical actions, and action results.
2. Preserve deterministic canonicalization and evidence linkage across x86/ARM and future
   substrates.
3. Make physical side effects visible to the compiler and verifier.
4. Keep authorization policy outside the public compiler while providing an explicit hook for a
   governance provider.
5. Support MHS as a first-class adapter without coupling Core v1 to an unpublished external schema.
6. Support direct devices, remote devices, lab/manufacturing hardware, robotics, sensors, embedded
   targets, and future device classes without a separate language feature per transport.

> **SUPERSEDED (RFC 0027 §0.2, §0.4, §0.6) —** Goals 1 and 3 are rejected: there is no "one
> typed representation" for devices, and no physical-effect class is carried through the
> compiler IR. Goal 4's governance hook survives in shape only — admissibility is issued by
> `512-mind`, and the gateway accepts a command from **any** producer that carries a valid
> admissibility proof, so the actuation path must not depend on which planner produced the
> command. Goal 5's "first-class adapter" is narrowed to a single edge codec in the private
> runtime.

## Non-goals

- MIND does not define USB, serial, CAN, ROS, vendor SDK, or instrument protocols.
- The public compiler does not own user/operator policy or enterprise fleet management.
- A language-level effect annotation does not itself make an unsafe device safe.
- This RFC does not claim that Anthropic MHS is stable or that MIND is conformant to it today.
- Device drivers are not allowed to become a second policy engine.

## Relationship to existing MIND RFCs

This draft composes with, rather than replaces:

- RFC 0018 Bare-Metal Substrate: low-level execution reaches physical/embedded targets.
- RFC 0019 Deterministic Agent Substrate: runtime step identity and evidence linkage.
- RFC 0022 Deterministic I/O: external observations are explicit inputs to replayable execution.
- RFC 0025 Intent Contracts: user/model intent is normalized before execution authority exists.
- RFC 0016/0021 evidence + canonical mic@3: compile-time artifact identity remains the root anchor.

## Canonical Device IR

> **REJECTED IN FULL (RFC 0027 §0.2) —** This section is the rejected part of this document.
> RFC 0027 §0.2, verbatim: *"Forbidden: a 'Canonical Device IR', a `device.*` opcode, or any
> `mic@N` bump."* Every object named below is superseded by the single `DeviceActionReceipt`
> of RFC 0027 §4.1 — a fixed-width TLV record (no floats, no clocks, no locale) carried under
> the reserved key `evidence_chain.device_receipts`, omitted when empty, outside the
> `trace_hash` preimage and inside the signature preimage. One actuation binds four hashes into
> a single preimage — `(ingress_hash, decision_hash, command_bytes_hash, outcome)` — where
> `command_bytes_hash` is taken *after* codec encoding, so the record covers what the hardware
> actually received rather than what we intended to send. A receipt whose fields do not bind is
> not a weaker receipt; it is not a receipt. Everything from here to *Single physical-effect
> barrier* is retained as historical record only.

The names below are semantic requirements, not frozen surface syntax.

```text
DeviceId
DeviceManifest
DeviceCapability
DeviceStateSnapshot
PhysicalActionRequest
PhysicalActionResult
PhysicalActionReceiptRef
DeviceError
```

### DeviceManifest

A manifest identifies what a device is allowed to expose to a MIND host. It MUST be canonicalizable
and hashable independent of transport. Minimum semantic fields:

```text
schema_version
stable_device_id
manufacturer/model identifiers where available
firmware/software identity where available
capabilities[]
state_fields[]
operation/procedure descriptors[]
declared bounds/units
adapter identity/version
```

Human-readable descriptions may accompany the manifest but MUST NOT be the sole machine-enforced
source for numerical limits or effect classification.

`manifest_hash = SHA-256(canonical_manifest_bytes)` is the identity bound into action requests and
receipts.

### Capability classes

Every exposed operation MUST carry one of four effect classes:

```text
Observe          read-only observation; no intended physical state change
Configure        changes device configuration or a persistent control value
Actuate          causes direct physical motion/energy/material/state change
Procedure        executes a bounded multi-step device program
```

If an adapter cannot prove an operation is `Observe`, it MUST classify it as effectful. Unknown
operations fail closed.

### Values

Canonical signed/evidence payloads MUST NOT rely on host-language JSON floats. Device values use a
tagged representation such as:

```text
bool
signed/unsigned integer
Q16.16 or another explicitly named fixed-point representation
canonical decimal string
f32_bits / f64_bits when exact hardware IEEE bit preservation is required
bytes
UTF-8 string
unit + value wrapper
```

The representation is part of the hash preimage. Unit conversion occurs before authorization and is
recorded, not inferred after execution.

### DeviceStateSnapshot

A state snapshot is an explicit observation input:

```text
manifest_hash
state_fields
snapshot_sequence
source_adapter
state_hash
```

Wall-clock time may be recorded as metadata but is not required for deterministic identity. When a
policy needs time, time is passed as an explicit input just like any other nondeterministic
observation.

### PhysicalActionRequest

Minimum semantic fields:

```text
schema_version
request_id                  # canonical hash-derived identity, not a random hidden value
artifact_or_agent_step_ref
intent_contract_ref         # optional at public layer, opaque to compiler
device_id
manifest_hash
capability_id
effect_class
canonical_parameters
pre_state_hash              # binds the decision to the state it inspected
preconditions
idempotency_key
requested_bounds
```

The request contains no raw driver handle and no credential.

### Authorization seam

Before an effectful request crosses the physical-effect barrier the runtime calls a governance
provider:

```text
authorize(request, current_state) -> grant | deny | require_approval
```

The public ABI treats the returned grant as opaque authenticated data. Policy semantics belong to the
governance provider, not the driver.

`Observe` MAY use a lighter policy profile but still crosses the same device gateway so state
provenance remains uniform.

### PhysicalActionResult

```text
request_id
device_id
manifest_hash
capability_id
result_code
device_reported_result
post_state_hash             # when readback exists
observation_hash            # optional separate measurement artifact
adapter_result_hash
```

A successful transport write is not sufficient to claim a successful physical action. When the
operation declares a readback/postcondition, success requires the declared postcondition to be
observed or the result is partial/failed.

> **SUPERSEDED (RFC 0027 §0.7, §4.1, §4.2) —** The decided outcome vocabulary is exactly
> `Completed` | `Refused` | `OutcomeUnknown`, and `OutcomeUnknown` is first-class rather than an
> error case — a command that was issued but whose result was never observed is the normal
> failure mode of physical hardware. "A successful transport write is not sufficient to claim a
> successful physical action" survives as intent, but the conclusion is stronger than
> `partial/failed`: verification asserts the record is internally consistent and unforged, and
> it does not and cannot assert that the device physically moved.

## Single physical-effect barrier

The public contract defines one semantic gateway. Implementations may have many transports behind
it, but MIND code above it MUST NOT directly invoke device SDKs for effectful operations.

Forbidden architectural shape:

```text
agent/flow -> vendor SDK -> device
agent/flow -> raw serial/network write -> device
agent/flow -> arbitrary MCP hardware tool -> device
```

Required shape:

```text
agent/flow -> Device IR request -> governance hook -> device gateway -> adapter -> device
```

This is an architectural invariant, not a style preference.

> **SUPERSEDED IN PART (RFC 0027 §0.4) —** The single gateway survives; the required shape
> above does not. There is no "Device IR request", and the gateway accepts a command from
> **any** producer carrying a valid admissibility proof — it does not privilege one planner, and
> planning is orthogonal to the actuation path.

## MHS compatibility seam

> **SUPERSEDED (RFC 0027 §0.3) —** The external hardware standard is a boundary **codec**, not
> a representation, and it is spoken by exactly one component: the gateway edge codec in the
> private `mind-runtime`. There is no public compatibility adapter translating it into a Device
> IR, because there is no Device IR. Conformance to an unpublished external specification is not
> claimed — the correct public statement is that we speak *a* device codec at the edge, not that
> we conform to any named standard. RFC 0027 deliberately names no external vendor and no
> standard; the naming in this section is retained as historical record, not as a claim.

The MHS research preview publicly describes discoverable device metadata, a shared state/interface,
read/write-style primitives, MCP/CLI/API access, and deterministic code files for long-running or
fast device procedures. MIND maps those concepts into Device IR at one boundary:

```text
MHS driver/reference/state/procedure
             <->
      MHS compatibility adapter
             <->
         MIND Device IR
```

When the final MHS specification is open sourced:

1. freeze the exact supported MHS version/profile;
2. publish field-by-field mapping and unsupported cases;
3. add golden conformance vectors;
4. test native-MHS and wrapped legacy devices;
5. never silently reinterpret a changed MHS field;
6. do not claim generic MHS conformance from compatibility with the research preview.

## Determinism and replay

Physical reality is not required to be deterministic. The **decision and evidence boundary is**.
Given identical canonical request bytes, explicit observed state, manifest, governance input, and
adapter response bytes, MIND must derive identical request/result/receipt hashes across supported
deterministic substrates.

Replay therefore distinguishes:

```text
Decision replay     deterministic; re-evaluates the same recorded inputs
Transport replay    uses recorded adapter responses; never re-actuates hardware
Live replay         a new physical execution and therefore a new action request
```

A replay command MUST NOT accidentally re-run an actuator.

> **SUPERSEDED (RFC 0027 §0.5, §0.7, §5) —** Two replay modes, not three. **Replay** re-derives
> and checks an evidence chain and **never actuates**; **re-execution** is a new authority
> requiring a fresh admissibility proof, not "replay with a flag" — and the distinction is
> enforced by a gate, not by documentation. On determinism, the honest statement is narrower
> than the one above: the *decision* is deterministic and bit-identical across substrates, the
> *physical outcome* is not. A device is a stochastic island. Because receipts sit outside the
> `trace_hash` preimage, two runs on different substrates produce identical artifacts and may
> legitimately produce different receipts — which is exactly why receipts are **not** part of
> the byte-identity gate.

## Security invariants

- No device credential in model, skill, flow, or intent payloads.
- No bypass around the gateway for effectful operations.
- Manifest hash and device identity are bound to authorization.
- State used for authorization is bound by `pre_state_hash`.
- Adapter re-reads critical state immediately before actuation when the device supports it, closing
  the obvious TOCTOU window.
- Duplicate `idempotency_key` under the same device/capability scope does not produce a second side
  effect unless the operation explicitly declares repeatable semantics.
- Unknown effect class, unit, bound, or manifest version fails closed.
- Long-running `Procedure` execution has a bounded envelope; code files are not an escape hatch from
  per-device safety limits.
- A driver may enforce stricter physical limits than the grant, never looser ones.

## Compiler/verifier work

Planned public surfaces:

1. Device IR data types and canonical encoding library.
2. Effect classification in the compiler IR or an equivalent preserved annotation.
3. `mindc check` diagnostics for undeclared effectful device operations.
4. `mindc verify` support for request/result linkage and no-actuation replay mode.
5. Mock device backend for CI.
6. Golden vectors: manifest -> state -> request -> result hashes.
7. Conformance hooks used by private/commercial runtimes without putting vendor SDKs in the public
   compiler.

> **SUPERSEDED (RFC 0027 §0.2, §4.2) —** Items 1, 2, 3, 6 and 7 are rejected: no Device IR
> encoding library, no effect classification preserved through lowering, no `mindc check`
> device diagnostics, no manifest -> state -> request -> result hash vectors. `mindc verify`
> gains exactly **one** check (item 4, narrowed): if `evidence_chain.device_receipts` is
> present, every receipt must be well-formed, `artifact_trace_hash` must equal the artifact's
> own `trace_hash`, and `prev_receipt_hash` must chain. Absent the key, behaviour is unchanged —
> which is why the key is omitted-when-empty.

## Acceptance gates

- Same mock manifest/action/state produces byte-identical canonical artifacts on x86_64 and ARM64.
- Unknown operation/effect fails closed.
- A direct bypass fixture is rejected by the architectural/conformance gate.
- Replaying an evidence log cannot actuate a mock device.
- Duplicate idempotency action cannot produce a second mock side effect.
- Changing one manifest bound or one pre-state value changes the authorization/request identity.
- MHS interoperability claims remain disabled until a public normative MHS version is pinned.

> **SUPERSEDED (RFC 0027 §5, §7) —** Receipts are not part of the cross-substrate identity
> gate; the artifact's bytes are unchanged by actuation, and differing receipts across
> substrates are legitimate (§5). The decided gate is: **replace the device adapter with a
> panicking stub and run `mindc verify` — if verify needs the device, the boundary is fake.**
> It must assert a **positive count** of receipts verified, because a run that verifies zero
> receipts and exits 0 proves nothing. Second gate: a replay run must actuate **zero** times
> against that stub — if replay can reach the device, replay is re-execution and the mode
> distinction is fictional.

## Open questions

1. Whether effect class lives as a dedicated mic@3 opcode/flag or as a typed standard-library ABI
   that lowering cannot erase.
2. Whether `Procedure` is compiled MIND bytecode, an external MHS code file, or both behind distinct
   manifests.
3. Canonical unit vocabulary and dimensional analysis scope.
4. Minimum public signature verification surface for an opaque authorization grant.
5. How much of device discovery belongs in `std.device` versus the host runtime.

> **PARTLY ANSWERED / SUPERSEDED (RFC 0027 §0.2, §8) —** Question 1 is answered and closed:
> effect class is neither a `mic@3` opcode nor a flag, because there is no new IR at all.
> Questions 2, 3 and 5 lapse with the Device IR they belong to. The open questions of record are
> now RFC 0027 §8: receipt batching for high-rate actuators without weakening per-command
> binding, and whether `512-mind` can issue admissibility proofs at actuation latency — it
> currently compiles 0 of 144 modules, so this is unmeasured and no schedule should assume it.
