# RFC 0027: Governed Physical-Device Plane — evidence-bound actuation (`device_receipts`)

| Field | Value |
|---|---|
| RFC | 0027 |
| Title | Governed Physical-Device Plane — evidence-bound actuation (`device_receipts`) |
| Status | **Draft — Architecture Decided** (D0). The load-bearing architecture (§0) was decided against the live codebase by the architecture decision authority; §4–§6 record those decisions. **No implementation has shipped.** |
| Authors | STARGA Inc. |
| Created | 2026-08-28 |
| Repo | `mind` (public) defines the schema and the verifier; `512-mind` governs admissibility; `mind-runtime` (private) performs actuation and speaks the external codec |
| Depends | RFC 0015 (cross-substrate bit-identity), RFC 0016 (evidence-chain emission), RFC 0017 (`mindc verify`), RFC 0021 (canonical mic@3 IR + MAP epilogue), RFC 0024 (the collapse-receipt carrier pattern this reuses verbatim) |

> **Supersedes** `docs/rfcs/DRAFT-governed-device-io-mhs.md` and `docs/MHS_ROADMAP.md`. Both were
> written before the architecture was decided and both assert a **"Canonical Device IR"**, which
> §0.2 rejects. They are superseded rather than deleted so the public record shows the change.

---

## 0. Architecture decisions

These were decided against the live codebase, not against a proposal document. Each is stated as a
rule plus the concrete thing it forbids.

### 0.1 The schema lives where the verifier lives — `mind`, and nowhere else

`mindc verify` is the thing that decides whether an actuation record is well-formed. It ships in
the **public** compiler. Therefore the schema it validates must also be public: a verifier that
needs a private repo to know what it is verifying is not a verifier.

The evidence chain already lives at `src/ir/compact/v3/evidence.rs`. This RFC extends *that* file.
**Forbidden:** defining the receipt schema in a private specification repo and mirroring it here.

### 0.2 No new IR — the receipt is a MAP entry, not an instruction

**Forbidden:** a "Canonical Device IR", a `device.*` opcode, or any `mic@N` bump.

Actuation produces a *record*, not a computation. Records already have a carrier: the MAP epilogue.
This RFC adds exactly one key, `evidence_chain.device_receipts`, following the RFC 0024
collapse-receipt pattern byte-for-byte:

- a TLV blob, **omitted entirely when empty** — so an artifact that actuates nothing is byte-identical
  to one compiled before this RFC existed, and no `mic@N` bump is required;
- **outside** the `trace_hash` preimage, because `trace_hash` anchors the *program*, and a device
  outcome is not a property of the program;
- **inside** the signature preimage, because an actuation record that can be edited under a valid
  signature is worthless. `evidence.rs` already asserts exactly this for collapse receipts:
  *"editing the receipt blob under a valid signature must break the signature."*

**Grounding correction.** An earlier narrowing described `device.*` as a MAP namespace "sibling to
`agent.*`". There is no `agent.*` namespace in the codebase. The real namespaces are the two
reserved prefixes — `RESERVED_KEY_PREFIXES = ["evidence_chain.", "signature."]` — plus application
keys, which must contain a `.` and must not collide with a reserved prefix. A bare `device.*` key
would therefore be an **application** key: unreserved, and writable by any program. For an actuation
record that is a forgery surface. The receipt belongs under the reserved `evidence_chain.` prefix.

### 0.3 MHS is a codec, not a representation

The external hardware standard is spoken by **exactly one component**: the gateway edge codec in
`mind-runtime`. It is a wire encoding at the boundary, never an internal representation, and never
a type that leaks inward.

**Forbidden:** claiming conformance to an unpublished external specification. Until that spec is
published and a conformance suite exists, the correct public statement is that we speak *a* device
codec at the edge — not that we conform to any named standard.

### 0.4 Planning is orthogonal to the actuation path

The gateway accepts a command from **any** producer that carries a valid admissibility proof. It
does not care which planner produced it.

**Forbidden:** making actuation depend on the intent, nerve, flow, or agent components. Coupling a
physical actuator to a skeleton component — one of which does not currently build natively — puts a
motor behind a build failure.

### 0.5 Two replay modes, not three

- **Replay** re-derives and checks an evidence chain. It **never actuates.**
- **Re-execution** is a new authority requiring a fresh admissibility proof. It is not "replay with
  a flag".

The distinction is enforced by §7's gate, not by documentation.

### 0.6 Three repos, not seventeen

`mind` **defines** (schema + verifier). `512-mind` **governs** (admissibility). `mind-runtime`
**performs** (actuation + edge codec). Every other repo is a consumer.

### 0.7 A device is a stochastic island — use the determinism vocabulary we already have

The **decision** is deterministic and bit-identical across substrates. The **outcome** is not: a
motor stalls, a sensor drifts, a gripper slips.

**Forbidden:** the phrase "deterministic device control". It is a doc-honesty violation of the kind
RFC 0017 exists to prevent. `OutcomeUnknown` is a first-class outcome, not an error case — a command
that was issued but whose result was never observed is the *normal* failure mode of physical
hardware, and a schema that cannot represent it will be lied to.

---

## 1. Summary

Bind every physical actuation into the artifact's own evidence chain, so that "this machine moved
because this governed decision authorized it" is checkable after the fact by the public verifier,
without trusting the device, the gateway, or the operator.

One actuation binds four hashes into a single preimage:

```
(ingress_hash, decision_hash, command_bytes_hash, outcome)
```

- `ingress_hash` — the sensor input the decision was made from.
- `decision_hash` — the governed decision, anchored to the `trace_hash` of the artifact that made it.
- `command_bytes_hash` — the exact bytes put on the wire, hashed *after* codec encoding, so the
  record covers what the hardware actually received rather than what we intended to send.
- `outcome` — `Completed` | `Refused` | `OutcomeUnknown`.

A receipt whose four fields do not bind is not a weaker receipt; it is not a receipt.

## 2. Motivation

Physical action is the one domain where an unverifiable claim has a body count. The existing wedge —
bit-identical output plus a hash-anchored tamper-evident chain — is worth more here than anywhere
else, because the question "what authorized this movement?" has a correct answer that survives an
adversarial operator.

What makes it tractable is that we are not inventing the mechanism. The carrier (MAP epilogue), the
anchor (`trace_hash`), the tamper-evidence (signature preimage), and the receipt pattern (RFC 0024)
all already ship. This RFC spends one reserved key and reuses the rest.

## 3. Guide-level explanation

A governed device action has four parties and one rule.

1. A **producer** proposes a command. Any producer; the gateway does not privilege one (§0.4).
2. **512-mind** issues an *admissibility proof* — this command, from this state, is permitted.
3. The **gateway** in `mind-runtime` validates the proof, encodes the command through the edge codec
   (§0.3), actuates, and observes.
4. The gateway emits a **`DeviceActionReceipt`** into `evidence_chain.device_receipts`.

The rule: **the gateway will not actuate without a valid admissibility proof, and cannot emit a
receipt that does not bind the four hashes.** Everything else is detail.

`mindc verify` can then re-derive the chain and answer "was this movement authorized?" — using only
the public compiler and the artifact.

## 4. Reference-level explanation

### 4.1 `DeviceActionReceipt`

Defined in `mind` (§0.1), encoded TLV exactly as `collapse_receipt::encode_collapse_receipts` does:

| Field | Type | Notes |
|---|---|---|
| `schema` | u16 | receipt schema version; additive-only |
| `ingress_hash` | 32 B | SHA-256 of the canonical sensor ingress record |
| `decision_hash` | 32 B | SHA-256 of the governed decision |
| `artifact_trace_hash` | 32 B | the deciding artifact's `trace_hash` — this is what makes it a *chain* |
| `command_bytes_hash` | 32 B | SHA-256 of post-codec wire bytes (§1) |
| `outcome` | u8 | 0 `Completed`, 1 `Refused`, 2 `OutcomeUnknown` |
| `prev_receipt_hash` | 32 B | zero for the first receipt on a device chain |

Fixed-width, no floats, no clocks, no locale — the same discipline as every other preimage in the
chain. A duration is carried as an integer plus a unit and a scale, never as a float.

### 4.2 Verification

`mindc verify` gains one check: if `evidence_chain.device_receipts` is present, every receipt must
be well-formed, `artifact_trace_hash` must equal the artifact's own `trace_hash`, and
`prev_receipt_hash` must chain. Absent the key, behaviour is unchanged — this is why the key is
omitted-when-empty (§0.2).

Verification asserts the record is **internally consistent and unforged**. It does not and cannot
assert the device physically moved; that is what `OutcomeUnknown` is for (§0.7).

## 5. Determinism preservation

The receipt is outside `trace_hash` (§0.2), so an artifact's bytes are unchanged by actuation and
cross-substrate byte-identity is untouched. Two runs on different substrates produce identical
artifacts and may legitimately produce different receipts — that is the stochastic island (§0.7),
and it is why receipts are not part of the identity gate.

## 6. Relationship to the private specification repo

`mind-physical` (private) is the coordination and specification home for the program: threat model,
deployment tiers, hardware notes, rollout. It **does not** define the schema — this RFC does (§0.1).
Where a `mind-physical` spec file restates a schema detail, it is a mirror and must be marked as one;
the definition site is this document.

## 7. CI gate

**Replace the device adapter with a panicking stub and run `mindc verify`. If verify needs the
device, the boundary is fake.**

This is the same shape as the byte-identity gates: prove a property by removing the thing it must not
depend on, and require a positive assertion rather than a clean exit. The gate must assert a positive
count of receipts verified — a run that verifies zero receipts and exits 0 proves nothing, which is
the failure mode catalogued across this repo's other gates.

Second gate, for §0.5: a replay run must actuate **zero** times against the panicking stub. If replay
can reach the device, replay is re-execution and the mode distinction is fictional.

## 8. Unresolved questions

- Receipt batching for high-rate actuators: one MAP entry per command will not hold at kHz rates.
  Batching must not weaken per-command binding.
- Whether `512-mind` can issue admissibility proofs at actuation latency. It currently compiles
  0 of 144 modules, so this is unmeasured, and no schedule should assume it.
