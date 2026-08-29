# MHS / Governed Device I/O Roadmap

> Status: roadmap/specification work. Anthropic announced the Model Hardware Standard (MHS) research preview on 2026-08-27. The final open-source MHS specification is not public yet. This document therefore defines MIND's stable internal boundaries and an MHS compatibility seam without claiming conformance to an unpublished specification. When the normative MHS specification is released, the adapter and conformance layer MUST be reconciled before any interoperability claim is made.

This is the public MIND-language side of the ecosystem MHS initiative. It owns types, canonical
bytes, effect semantics, compiler/verifier behavior, a mock backend, and conformance vectors. It does
**not** own production device drivers, enterprise policy, fleet management, or user-facing device
orchestration.

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

## M2 — Compiler physical-effect awareness

- Preserve physical-effect classification through lowering.
- Add fail-closed diagnostics for undeclared/unknown effects.
- Ensure a generic FFI/network/process/MCP call cannot silently claim `Observe`.
- Add replay mode that consumes recorded results and cannot actuate.

## M3 — Public host ABI + mock backend

- Stable DeviceGateway/DeviceBackend host ABI.
- In-tree deterministic mock backend.
- Golden tests on x86_64 + ARM64.
- Reference examples: temperature sensor, bounded actuator, deterministic multi-step procedure.

## M4 — MHS reconciliation gate

Triggered only when Anthropic publishes a normative open-source MHS specification:

- pin a supported MHS version;
- publish a field mapping from MHS to Device IR;
- add compatibility and negative vectors;
- document unsupported semantics explicitly;
- only then advertise MHS compatibility/conformance at the supported profile.

## M5 — Spec promotion

After the public RFC and implementation stabilize, promote the interoperable semantics to
`star-ga/mind-spec` without changing Core v1 retroactively.
