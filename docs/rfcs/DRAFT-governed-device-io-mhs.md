# Draft RFC: Governed Device I/O and MHS Compatibility

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

## Summary

MIND needs a deterministic boundary between compiled/agentic intent and the physical world.
This RFC defines a vendor-neutral **Device IR** and explicit **physical-effect semantics** in the
public language/runtime contract. An MHS adapter may translate between this stable Device IR and
Anthropic's Model Hardware Standard. The MIND contract is deliberately not a competing wire
standard: MHS remains an external compatibility surface, while Device IR is the canonical internal
representation that protects the language from changes in a research-preview protocol.

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

## MHS compatibility seam

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

## Acceptance gates

- Same mock manifest/action/state produces byte-identical canonical artifacts on x86_64 and ARM64.
- Unknown operation/effect fails closed.
- A direct bypass fixture is rejected by the architectural/conformance gate.
- Replaying an evidence log cannot actuate a mock device.
- Duplicate idempotency action cannot produce a second mock side effect.
- Changing one manifest bound or one pre-state value changes the authorization/request identity.
- MHS interoperability claims remain disabled until a public normative MHS version is pinned.

## Open questions

1. Whether effect class lives as a dedicated mic@3 opcode/flag or as a typed standard-library ABI
   that lowering cannot erase.
2. Whether `Procedure` is compiled MIND bytecode, an external MHS code file, or both behind distinct
   manifests.
3. Canonical unit vocabulary and dimensional analysis scope.
4. Minimum public signature verification surface for an opaque authorization grant.
5. How much of device discovery belongs in `std.device` versus the host runtime.
