# RFC 0028: Deterministic Field Calculus

| Field | Value |
|---|---|
| RFC | 0028 |
| Title | Deterministic field calculus — Q16.16 fields, differential operators, and prove-or-refuse conservation obligations |
| Status | **Draft — Not Scheduled.** No implementation has shipped. Explicitly gated behind the Rust-independence (RI) track and behind RFC 0012 Phase B.2 (§9.1). |
| Authors | STARGA Inc. |
| Created | 2026-08-31 |
| Depends | RFC 0012 (tensor-native surface — `field` extends `Tensor`, not a parallel system), RFC 0024 (loop collapse — the prove-or-refuse + receipt pattern this RFC copies structurally), RFC 0015 (cross-substrate bit-identity — the property a field program may claim), RFC 0016 (evidence-chain emission — carrier for the conservation receipt), RFC 0017 (`mindc verify` — re-derives the receipt) |
| Related | RFC 0006 (mind-blas numeric tiers), RFC 0021 (canonical mic@3 IR), RFC 0027 (governed physical device plane — the *actuation* lane this RFC deliberately does not touch) |

> **Numbering note.** This RFC takes **0028**. `0026` is an unexplained gap in
> `docs/rfcs/` — no reservation record exists anywhere in the tree (unlike `0023`,
> which RFC 0025 documents as reserved-and-cancelled). It is left alone rather than
> reused, because an undocumented gap is more likely a lost reservation than a free
> slot, and the cost of skipping a number is zero.

---

## 1. Summary

This RFC adds **spatial semantics** to MIND's existing shape-typed tensors, so that
the compiler knows a value is a discretized quantity over a domain rather than an
opaque block of numbers — and can therefore *prove* things about it.

Four concepts, no more:

1. `domain` — compile-time spatial metadata (rank, extents, spacing, boundary rule).
2. `field` — `Tensor<T, Shape>` **plus** a domain. Not a parallel type system.
3. Compiler-known differential operators — `grad`, `div`, `laplacian`. Three, not twelve.
4. **Conservation obligations** — `#[conserves(...)]`, which the compiler either proves
   for the given discretization or **refuses to compile**, emitting an `E23xx` diagnostic
   and never silently degrading to a runtime check.

The first normative numeric profile is **Q16.16 only**. This is the load-bearing
restriction and §3 explains why an `f32` field cannot honour MIND's wedge.

The reference workload is not hypothetical: `rfn-mind`'s `laplacian.mind` and
`field_step.mind` already express a Q16.16 5-point stencil with three boundary
conditions in MIND today, by hand. This RFC's success criterion is that those files
become expressible in the field calculus **and lower to byte-identical output**.

### 1.1 What this RFC is not

It is not "Physical AI syntax." An earlier framing of this work proposed roughly
forty keywords spanning physics, meshes, autodiff surface, distribution, device
semantics, precision declarations, optimization and evidence. That framing is
rejected here. An audit of the live tree found that of twelve proposed areas, four
had already shipped, four were owned by other RFCs, and only four were genuinely
absent. This RFC takes only the four absent ones, and takes them at the smallest
size that admits a proof.

---

## 2. Motivation

### 2.1 The compiler currently cannot see space

`rfn-mind/src/laplacian.mind` computes a 5-point stencil. To the compiler that file
is index arithmetic over a flat buffer: a `%` here, a bounds compare there, some
adds. The compiler cannot tell that:

- the loop is a **stencil** (so it has a known halo, and a known parallel structure);
- the `idx` function implements a **boundary condition** (so `Periodic` vs `Reflective`
  is a semantic choice, not an arithmetic accident);
- the result is a **discrete Laplacian** (so `sum(laplacian(F)) == 0` holds exactly
  under periodic BC — a checkable conservation law).

Every one of those facts exists in the programmer's head and in a comment. None is
available to the type-checker, the optimizer, or the evidence chain.

### 2.2 Why a comment is not good enough

The third fact above is the interesting one. Under a periodic boundary condition, a
discrete Laplacian is a difference of a value and its neighbours in which **every
term appears exactly once with coefficient `+1` and once with coefficient `-1`**
across the whole grid. In exact integer arithmetic the global sum is therefore
identically zero — not approximately, exactly. In Q16.16 it is *still* exactly zero,
because Q16.16 addition is integer addition and integer addition is associative.

That is a theorem the compiler could check and the evidence chain could record. Today
it is a sentence in a design doc that nobody re-derives, and a discretization bug that
breaks it produces a plausible-looking simulation that is quietly wrong.

This is precisely the class of failure MIND exists to eliminate.

---

## 3. The Q16.16 restriction (normative, load-bearing)

**The v1 field calculus admits `q16` fields only.** `f32` and `f64` fields are
rejected at the type level with `E2310`.

This is not conservatism about effort. It follows from what the compiler already
knows, and getting it wrong would puncture the wedge:

1. **RFC 0012 already draws this line.** A `#[deterministic]` function operating on
   `f32` without a fixed-point target is specified to emit
   `determinism::f32_non_bit_identical`, signalling that the guarantee is
   within-substrate reproducible but *not* cross-substrate byte-identical
   (RFC 0012 §Open Questions, `docs/rfcs/0012-tensor-native-syntax.md:874-881`).
   Precision matters here: that is an **info** diagnostic and a *proposed resolution*,
   promotable to `warn`/`error` by Mindcraft config — not a hard refusal, and not in
   RFC 0012's normative body. So it is not a precedent for rejecting `f32`; it is
   evidence that the distinction is already recognized. This RFC takes the stronger
   position (`E2310`, a hard error) for the specific reason in point 2: a field
   carrying a *conservation obligation* is making a claim an `f32` field cannot
   discharge, which is a narrower and firmer case than `#[deterministic]` in general.

2. **A conservation proof over floats is not a proof.** `sum(laplacian(F)) == 0` is a
   theorem in `Z/2^64` and in Q16.16 (both are rings under integer addition — see
   §5.4 for why MIND's overflow rule is what makes this true rather than aspirational). Over
   IEEE-754 it is false in general — summation is not associative, so the result
   depends on reduction order, which is exactly the choice MIND pins and other
   compilers leave unspecified. A `#[conserves]` obligation over `f32` could only ever
   be discharged approximately, and an approximate proof recorded in a tamper-evident
   evidence chain is worse than no proof, because it *looks* like one.

3. **The tolerance comparison is itself non-deterministic.** This is the subtle case
   and it is why this RFC has no `tolerance` construct. A contract of the form
   `ensures abs(div(v)) < 1e-5` requires evaluating a float comparison. If the
   residual lands within one ULP of the threshold, x86 and ARM can legitimately
   disagree on PASS/FAIL — and the evidence chain then records a *different verdict
   per machine*. The verification layer would become the one non-reproducible part of
   a system whose entire premise is reproducibility. A float tolerance check does not
   merely fail to help; it actively destroys the property being verified.

Consequently: **v1 obligations are exact integer identities, not tolerances.** There
is no `tolerance` keyword in this RFC. If a future RFC wants approximate conservation,
it must first specify how the comparison itself is made byte-identical (for example
by expressing the bound as an exact Q16.16 quantity and comparing in the ring), and
that is a separate piece of design work with its own proof burden.

An `f32` field lane may be added later **as a non-wedge tier** — clearly marked as
providing shape and operator semantics but *not* cross-substrate identity, in the same
way RFC 0006 has both deterministic and relaxed numeric tiers. It is out of scope here.

---

## 4. Surface syntax

Syntax is presented last-in-importance deliberately: §6's semantics are the
normative content, and punctuation may change before implementation without
re-opening the semantic contract.

### 4.1 `domain`

```mind
domain Grid2D {
    rank: 2,
    extents: [H, W],          // symbolic dims, unified per RFC 0012 Phase A
    spacing: [Q16_ONE, Q16_ONE],
    boundary: Periodic,       // Periodic | Reflective | Zero
}
```

A `domain` is **compile-time metadata**. It emits no code and allocates nothing.
It lowers into existing mic@3 structures as attributes on the field's type; it does
**not** introduce a new IR node. The three boundary rules are exactly the three
`rfn-mind` already implements — not an open-ended set.

### 4.2 `field`

```mind
field F: q16 over Grid2D;            // shorthand
// equivalent to: Tensor<q16, [H, W]> carrying domain Grid2D
```

A `field` **is** a `Tensor` with a domain attached. Every RFC 0012 operator (`@`,
`.+`, `.*`, `.T`, `.sum`, `.mean`) applies unchanged. `field` adds spatial meaning;
it does not add a value representation, a layout, or an allocation strategy.

Field-to-field assignment between different domains is a compile error (`E2311`) —
the same class of protection RFC 0012 gives shape mismatch.

### 4.3 Differential operators

```mind
let L: field q16 over Grid2D = laplacian(F);
let g: field Vec2<q16> over Grid2D = grad(P);
let d: field q16 over Grid2D = div(V);
```

Exactly three operators in v1. `curl`, `dt`, spectral derivatives, arbitrary
coordinate systems, meshes and FEM are **out of scope** — each would need its own
conservation theory, and a field calculus that ships three provable operators is
worth more than one that ships twelve unprovable ones.

Each operator is a **semantic compiler operation**, not a library call. That is the
whole point: because the compiler owns `laplacian`, it may lower the same source to a
CPU stencil, a fused CUDA stencil, or a distributed halo exchange without the program
changing — and it may *prove* things about it, which §5 uses.

---

## 5. Conservation obligations — the prove-or-refuse core

This section is the reason the RFC exists. It copies RFC 0024 `#[collapse]`
**structurally**, because that is the shipped precedent for "the compiler proves it or
refuses to compile," and it is what makes this MIND-native rather than generic
design-by-contract.

### 5.1 The four-part pattern

| Part | `#[collapse]` (shipped, RFC 0024) | `#[conserves]` (this RFC) |
|---|---|---|
| Opt-in attribute | `#[collapse]` on a loop | `#[conserves(...)]` on a field function |
| Prover | `src/opt/collapse.rs` + `scev.rs` | `src/field/conserve.rs` (**does not exist — proposed**) |
| Fail-closed diagnostic | `E22xx`, never silently keeps the loop | `E23xx`, never silently degrades to a runtime check |
| Re-derivable receipt | `evidence_chain.collapse.*`, re-derived by `mindc verify` | `evidence_chain.field.*`, re-derived by `mindc verify` |

The critical inherited property is **fail-closed**. An unprovable `#[collapse]` loop
does not compile. An unprovable `#[conserves]` function must likewise not compile. It
must never fall back to inserting a runtime assertion, because a runtime check is a
different and weaker claim wearing the same syntax.

### 5.2 The v1 obligation set

Exactly two obligations, both exact integer identities in the Q16.16 ring:

**`#[conserves(sum)]`** — the global sum of the output field equals the global sum of
the input field, exactly.

```mind
#[conserves(sum)]
fn diffuse(F: field q16 over Grid2D) -> field q16 over Grid2D {
    return F .+ laplacian(F);        // unscaled, ring-`q16` ops only: provable
}
```

Provable because: under `Periodic` boundary, each grid cell contributes `+1` and `-1`
of each neighbour term to the global sum, so `sum(laplacian(F)) == 0` identically in
any ring; therefore `sum(F .+ laplacian(F)) == sum(F)`. The prover checks the boundary
rule is `Periodic`, that the expression is a sum of the input and terms whose global
sum is provably zero, and that every operation between the field and the sum stays in
one ring (§5.4).

Read that condition strictly. `.+` here denotes the **compiler-known ring-`q16`
addition** this RFC introduces — native `i32` addition on the Q16.16 representation,
wrapping per `docs/determinism.md` §1. It is **not** a call to a saturating helper that
happens to add Q16.16 values. The distinction is not pedantic: §5.4.1 shows the
reference implementation's own `laplacian` fails this condition, on a constant field.

Note what is **absent** from that example. The natural diffusion form carries a
coefficient — `F .+ (A .* laplacian(F))` — and that form is **not** provable in v1:
a Q16.16 multiply lowers to a multiply plus a `>>16` rescale, the shift is not
additive, and the obligation is refused with `E2305`. §5.4 works this through in full.
The v1 obligation deliberately covers the unscaled case only, rather than stretching
the certificate to cover an expression whose identity has not been proven.

Under `Reflective` or `Zero` boundary the identity does **not** hold — flux crosses
the boundary — and the prover emits `E2302` naming the boundary rule as the reason.
This is a feature: the most common discretization bug in this domain is assuming a
conservation law that the chosen boundary condition does not actually provide.

**`#[conserves(zero_sum)]`** — the output field sums to exactly zero. This is the
direct property of `laplacian` under periodic BC and is the base case the first
obligation builds on.

### 5.3 The `E23xx` diagnostic family

`E22xx` is taken by loop collapse, so field calculus claims `E23xx`:

| Code | Meaning |
|---|---|
| `E2301` | `#[conserves]` expression is not a recognized conservation shape. |
| `E2302` | The obligation does not hold under the domain's boundary rule (names the rule). |
| `E2303` | Mixed-ring arithmetic in a conserved expression — the identity is not ring-closed. |
| `E2304` | Conserved function has a side effect, escaping store, or I/O. |
| `E2305` | **Proof domain escaped.** Expected closed ring arithmetic (`ring_q16`); encountered a non-ring operation between the field and the conserved sum — saturating/clamping call (incl. `q16_add`/`q16_sub`/`q16_mul`), rounded fixed-point multiply, `>>` rescale, division, narrowing cast, or normalization. The domain is inferred from the lowered ops, never declared. See §5.4 / §5.4.1. |
| `E2310` | Field dtype is not `q16` (see §3 — `f32`/`f64` fields are not admitted in v1). |
| `E2311` | Field domain mismatch across an assignment or operator. |

Following RFC 0024 §4.3, `E2305` is a **permanent** rejection, not a deferral: an
expression that leaves the ring has no ring identity to prove, and silently widening or
re-associating it to manufacture one would be exactly the kind of invisible semantic
change the wedge forbids.

### 5.4 Why wraparound does not break the proof (and what does)

An obvious objection: algebraic cancellation over the mathematical integers is not
automatically cancellation under machine arithmetic. If an intermediate saturates or
clamps, `+x` and `-x` no longer annihilate, and the conservation certificate would be
issued for an implementation that does not actually conserve.

MIND's normative integer rule resolves this, and resolves it in the *stronger*
direction than a saturating language could:

> Integer overflow **wraps two's-complement** (defined; identical on x86 and ARM).
> — `docs/determinism.md` §1

Wraparound is exactly the statement that MIND's `i32`/`i64` arithmetic **is** the ring
`Z/2^32` / `Z/2^64`. Ring identities hold unconditionally in a ring, overflow included:
`(a + b) - b == a` is true for every pair of machine integers under wraparound. So
`sum(laplacian(F)) == 0` needs no range analysis, no no-overflow precondition, and no
bounds on the field's magnitude. Each neighbour value is added once and subtracted
once, and in `Z/2^64` those cancel exactly whether or not any intermediate wrapped.

Crucially, the same doc records that the rule holds at **every** layer — interpreter,
MLIR/native artifact (plain `arith.addi`, no `nsw`/`nuw`), and constant folding, where
folding is exact-or-skip rather than emitting a wrapped or saturated constant. A proof
discharged at compile time therefore describes the same algebra the artifact executes.
Had MIND chosen saturating arithmetic instead, this RFC would need a no-overflow
precondition and `E2305` would be a range-analysis diagnostic. It does not, and it is
not.

**What actually breaks the proof is leaving the ring**, and that is what `E2305`
detects. Non-ring operations that may not appear between the field and its conserved
sum:

| Operation | Why it breaks cancellation |
|---|---|
| `>>` (the Q16.16 rescale after a product) | Arithmetic shift is not additive: `(a>>16) + (b>>16) != (a+b)>>16` in general. Each dropped low bit is an unrecoverable rounding. |
| Integer division | Same reason; also MIND defines `x/0` as a trap (exit status 79), which is not a ring operation. |
| Saturating / clamping calls (e.g. `kernel.clamp`) | Deliberately non-wrapping — that is their purpose — so `+x`/`-x` no longer annihilate. |
| Narrowing casts (`i64 -> i32`) | Changes modulus mid-expression; terms cancel in different rings. |

This has a concrete consequence for the reference workload, and it is the sharpest
thing in this RFC. `rfn-mind`'s `field_step.mind:73` applies a Q16.16 multiply —
`fixed_point.q16_mul(cfg.diffusion, buffers.lap[i])` — and `q16_mul`
(`fixed_point.mind:120`) is *not* a ring operation. It widens to `i64`, then calls
`q32_to_q16_sat` (`fixed_point.mind:83`), which does three non-ring things in
sequence:

```mind
let shifted: i64 = (wide + 32768) >> 16   // 1. round-half-up  2. arithmetic shift
if shifted >  2147483647 { return Q16_MAX }   // 3. saturating clamp
if shifted < -2147483648 { return Q16_MIN }
```

Rounding discards information, the shift is not additive, and the saturation is
*deliberately* non-wrapping — the exact operation that breaks `+x`/`-x` cancellation.
`cfg.norm_mode` then applies either `group_norm` or `clamp_all`
(`groupnorm.mind:101`), adding a fourth. So the honest statement of what v1 can prove
about that file is:

- The **compiler-known** ring-`q16` `laplacian(F)`, under `Periodic` boundary, carries
  `#[conserves(zero_sum)]` — **provable**.
- `rfn-mind`'s **existing hand-written** `laplacian` (`laplacian.mind:79-84`) —
  **`E2305`**. It calls the saturating `q16_add`/`q16_sub`, not ring addition. See
  §5.4.1; this is a correction to an earlier draft of this RFC, which wrongly listed it
  as provable.
- `F .+ (A .* laplacian(F))` with a Q16.16 scalar multiply — **`E2305`**, because
  `q16_mul`'s rounding, `>>16` rescale and saturating clamp all sit between the field
  and the sum.
- The full normalized `field_step` — **`E2305`**, for the same reason plus
  `group_norm`/`clamp_all`.

### 5.4.1 The reference `laplacian` is not ring-closed (verified counterexample)

The saturation table above lists saturating calls as non-ring, but an earlier draft
applied that rule only to `q16_mul` and let the reference `laplacian` through. It does
not pass. `laplacian.mind:79-84` reads:

```mind
let sum_neighbors: fixed_point.Q16_16 = fixed_point.q16_add(
    fixed_point.q16_add(up, dn),
    fixed_point.q16_add(lf, rt)
)
let four_center: fixed_point.Q16_16 = center << 2
let lap: fixed_point.Q16_16 = fixed_point.q16_sub(sum_neighbors, four_center)
```

`q16_add`/`q16_sub` (`fixed_point.mind:101-109`) widen to `i64` and return
`sat_i64_to_q16` (`:113`), which clamps at `±2^31`. So a **single cell** of the
reference Laplacian mixes two algebras in one expression: three **saturating** adds,
one **wrapping** `<< 2`, and one **saturating** subtract.

The failure is not hypothetical or edge-case-only. Take the constant field
`F[c,y,x] = Q16_MAX` under `Periodic` boundary, where the Laplacian is mathematically
zero in every cell. Two independent breakages compound:

| Quantity | True value | Reference implementation |
|---|---|---|
| `sum_neighbors` = `4 * Q16_MAX` | `8589934588` | `2147483647` (saturated) |
| `four_center` = `Q16_MAX << 2` | `8589934588` | `-4` (wrapped) |
| `lap` | `0` | **`2147483647`** |

Every cell returns `Q16_MAX` where the theorem demands `0`. A control value that
saturates nothing (`v = 1000`) returns `0` as expected — which is exactly why a
sampled runtime assertion would pass and a compile-time proof must not.

Two consequences, both narrowing:

1. **The v1 theorem is about ring-`q16` arithmetic, not about "Q16.16".** `Q16.16` is a
   *representation*; it does not by itself name an algebra. The proof domain is
   `ring_q16` — native wrapping `i32` ops on that representation — and it must be
   **inferred from the lowered operations**, never declared by the author. This mirrors
   `fp_mode`, which derives strict/relaxed from the ops rather than trusting a
   declaration: inferred-and-attested beats programmer-asserted.
2. **`rfn-mind` is a reference specimen, not an all-input equivalence oracle.** Gate 1
   (§7) compares lowering on accepted reference vectors within the non-saturating
   subset; it cannot claim all-input result equality, because the two implementations
   genuinely disagree on saturating inputs — and where they disagree, the ring form is
   the one with a theorem.

Whether `rfn-mind` should itself migrate to ring-`q16` for the Laplacian is an RFN
semantic question (does that path *want* saturation as clipping?), owned by that repo
and explicitly out of scope here.

The unscaled **ring-`q16`** case (`A == 1`, no rescale, compiler-known addition) is
provable and is the v1 acceptance target.
This is a narrower claim than "RFC 0028 proves conservation for RFN's field step," and
it is stated here deliberately, because the failure mode this RFC exists to prevent is
a certificate that is broader than its proof. A future obligation covering scaled
updates must first specify what is conserved under rounding — which is a genuinely
different theorem (an inequality or an error bound, not an identity), with its own
proof burden, and is out of scope here.

---

### 5.5 The receipt

On success the compiler records under `evidence_chain.field.*`:

| Key | Meaning |
|---|---|
| `theorem_id` + `theorem_version` | Which named theorem was discharged. Versioned so a later change to the theorem cannot silently re-interpret an old receipt. |
| `stencil_id` | Which enumerated stencil (`laplacian_5pt_v1`). A closed set, not free-form. |
| `boundary` | The domain's boundary rule (`Periodic`) — a premise of the theorem. |
| `ring` | `z_mod_2_64` / `q16_16`. The algebra the identity holds in (§5.4). |
| `subject_hash` | Hash of the canonical mic@3 bytes of the proven function body — binds the receipt to the exact computation, not to the source text. |

`mindc verify` **re-derives** this rather than trusting it: it confirms `subject_hash`
identifies the expected lowered computation, then re-runs the named theorem from the
recorded premises. Same discipline as RFC 0024's collapse receipt. A receipt that
cannot be re-derived is a verification failure, not a warning.

These are additive MAP keys carried by RFC 0016's existing envelope — no new evidence
primitive, no new IR node. Whether that is *sufficient* for independent re-derivation
is open question 4 (§11), and is a prerequisite for Phase D rather than a detail to be
settled during it.

---

## 6. What this RFC explicitly does NOT own

This section is normative and exists to prevent the scope creep that the earlier
framing of this work exhibited. Each line names the actual owner.

| Not owned here | Owner |
|---|---|
| New IR nodes or a new IR level | RFC 0021 — field metadata rides existing mic@3 structures |
| New evidence primitive or envelope | RFC 0016; `Evidence[K]` is RFC 0004 |
| `precision { ... }` declarations | RFC 0012 `#[target(...)]` + `src/ir/fp_mode.rs`, which *infers* strict/relaxed from the ops rather than trusting an author's assertion — strictly stronger than a declaration |
| Generic `requires`/`ensures` framework | Not proposed. §5 is two named obligations, not a contract language |
| Intent contracts | RFC 0025 |
| Device, actuation, MHS semantics | RFC 0027 — and MHS is an edge codec, never internal representation |
| `jacobian` / `vjp` / `jvp`, `#[autodiff]` surface | The autodiff engine ships (`src/autodiff/`, `diff` type at mic@3 tag `0x08`); its surface annotation is named future work in RFC 0012 §2 and deserves its own small RFC |
| Distributed transport, NCCL, collectives surface | `mind-runtime` (see §9.2 — there is a determinism blocker there) |
| CUDA / kernel surface syntax | `mind-runtime`; lowering is RFC 0014 |
| `optimize` / `solve` / constraint framework | Not proposed by anyone; would need its own RFC |
| Neural-operator architectures | `rfn-mind` incubates; not a language concept |
| Any `f32` cross-substrate identity claim | Nobody — it is not true (§3) |

Note also that `operator` is **not** introduced as new syntax. A field-to-field `fn`
carries the semantics adequately. If evidence later shows a distinct construct buys
something, it can be added; introducing it speculatively would create a second
function-like declaration form with no demonstrated benefit.

---

## 7. Reference workload and success criterion

The gate for this RFC is not a synthetic test. It is `rfn-mind`'s `laplacian.mind`
and `field_step.mind` — but the two halves of the gate cover **different scopes**, and
conflating them would over-claim:

**Gate 1 — lowering parity (covers both files).**

> `laplacian.mind` and `field_step.mind`, re-expressed in the field calculus, must
> lower to output **byte-identical** to the current hand-written Q16.16
> index-arithmetic implementation **on the accepted reference vectors**, which are
> drawn from the non-saturating subset of the input space.

This mirrors RFC 0012 §7.2's IR-text byte-identity gate (`A @ B` ≡
`tensor.matmul(A, B)`) and RFC 0024's requirement that a collapsed loop equal its
un-collapsed form. A field calculus that changes results is not an abstraction, it is
a rewrite. Structurally the gate applies to the *whole* reference workload — scaling
and normalization included — because it is a statement about lowering, not about
proofs.

**Why the subset qualifier is not a loophole.** Per §5.4.1 the reference `laplacian`
calls saturating helpers, so on saturating inputs it *provably* disagrees with the
ring-`q16` form — on `F = Q16_MAX` it returns `Q16_MAX` where the ring form returns
`0`. An unqualified all-input byte-identity claim would therefore be unsatisfiable, and
the honest response is to narrow the gate rather than weaken the theorem. Two
requirements keep this from becoming a hiding place:

- The reference vector set is **committed to the repo and enumerated**, not selected at
  gate time; adding a vector is a reviewable diff.
- The gate additionally asserts the **known divergence** — a saturating vector (`F =
  Q16_MAX`, `Periodic`) must produce the §5.4.1 table's two different answers. A
  parity gate that cannot demonstrate where parity ends is measuring its own test set.

**Gate 2 — prove-or-refuse (covers only the ring-closed subset).**

> `laplacian(F)` under `Periodic` boundary must compile with a `#[conserves(zero_sum)]`
> receipt that `mindc verify` independently re-derives; and the scaled/normalized
> `field_step` forms must be **refused** with `E2305`.

Per §5.4, the Q16.16 rescale (`>>16`) and the normalization step leave the ring, so no
v1 obligation covers them. Both halves of Gate 2 are load-bearing: the refusal is
tested as rigorously as the acceptance, because a prove-or-refuse feature that never
refuses has not been shown to be sound — it has only been shown to be permissive.

**Additive-receipt caveat.** Gate 1 compares the *computational lowering* — the mic@3
bytes of the function body and its execution result. It is not a comparison of whole
artifacts: emitting an `evidence_chain.field.*` receipt necessarily changes the
artifact's MAP epilogue, so a whole-artifact byte comparison would be self-defeating
(it would forbid the receipt this RFC exists to add). The invariant is: same
computation bytes, same result, plus an additive receipt that `mindc verify`
re-derives.

**Ordering caveat (honest):** `rfn-mind` does not currently compile end-to-end under
public `mindc` (`forensic_audit.md:219`). That blocker (its "E0") must close before
this gate can run. This is a dependency, not an excuse — it is recorded here so the
RFC cannot be declared implementable while its only reference workload does not build.

---

## 7.1 Determinism and performance constitution (normative)

This RFC ships only if it costs the existing compiler nothing. The ordering below is
lexicographic, not a weighting — a lower item never buys a higher one:

1. Correctness
2. Determinism / reproducibility
3. No-regression compatibility
4. Runtime performance
5. Compiler performance
6. New capability

So `+20% faster but non-deterministic` is a failure, and so is `elegant field
abstraction but 15% slower`.

**G0 — zero-regression default path (exact, absolute).** For every existing keystone
and canary program that does **not** use RFC 0028 syntax:

| Property | Requirement |
|---|---|
| Canonical mic@3 bytes | **EXACT** match, pre- vs post-RFC compiler |
| `trace_hash` | **EXACT** |
| Execution result | **EXACT** |
| `fp_mode` classification | **EXACT** |
| Native code (where emitted) | **EXACT** |

One changed byte in a program that does not use the feature fails the release. This is
strictly stronger than "tests still pass."

**G1 — structural default-off.** The prover must not merely be cheap when unused; it
must not **initialize**. No field analysis, stencil recognition, domain solving, or
proof-obligation collection may run unless the opt-in construct is present in the
parsed AST. Reviewable as a code-structure property, not only as a benchmark result.

**G2 — compiler performance (statistical).** Criterion benchmarks over parse /
typecheck / lower / canonicalize / native-emit / full-compile / warm-cache, across
tiny / medium / large / self-host workloads. Requirement: **no statistically
significant regression beyond a pre-committed noise envelope.** Baselines are frozen
and committed *before* implementation begins, with raw Criterion output retained — not
remembered numbers or screenshots.

**G3 — runtime performance (parity floor, SOTA target).** The compiler-known stencil
must be **at least as fast** as the hand-written reference on the same workload.
Falling short means the implementation stays experimental; "more elegant" is not a
defence. The target is higher than parity: once `laplacian` is compiler-known, the
compiler gains the semantic room for stencil fusion, tiling, boundary specialization
and allocation elimination that hand-written index arithmetic denies it. Abstraction
must create optimization headroom, not consume it.

**G4 — no proof machinery in the hot path.** Proving happens at compile time and
re-derivation happens in `mindc verify`. The emitted program computes; it does not
carry, check, or re-derive its own certificate at runtime. Runtime cost of a proven
kernel over an unproven one: zero.

**Tracked beyond wall-clock,** because a 5% speedup that doubles allocations or binary
size is not a win: ns/op, cycles/op, instructions, allocations, peak RSS, binary size,
IR size, compile latency, warm-cache latency, artifact size delta, `mindc verify` cost
delta.

**Two kill switches.**

- *Determinism* — if a strict-profile construct breaks cross-substrate canonical
  identity or execution identity, it does not ship as strict.
- *Performance* — if the abstraction cannot reach parity with the hand-written
  reference on its target workload, the implementation stays experimental.

**Gate matrix, not stars.** Status is reported as GREEN / OPEN / RED per gate. A star
rating hides exactly the distinction that matters — an excellent implementation with
unproven determinism and a mediocre one with proven determinism both read as four
stars.

| Gate | Meaning | v1 status |
|---|---|---|
| `D0` | Deterministic correctness (ring identity holds as proven) | OPEN |
| `D1` | Canonical byte identity, x86 ↔ ARM | OPEN |
| `G0` | Zero-regression default path | OPEN |
| `G1` | Prover structurally default-off | OPEN |
| `C0` | Compiler no-regression (Criterion) | OPEN |
| `R0` | CPU runtime parity vs hand-written reference | OPEN |
| `R1` | CPU runtime beats reference (SOTA target) | OPEN |
| `P0` | `mindc verify` independently re-derives the proof | OPEN |
| `P1` | Refusal gate — non-ring forms rejected with `E2305` | OPEN |
| `GPU0` | Q16 GPU deterministic profile | OPEN (out of scope, §6) |
| `X0` | Distributed deterministic collective | OPEN (out of scope, §6) |

`GPU0` and `X0` are listed to make explicit that this RFC does **not** advance them. A
field being a language construct does not make it deterministic on a substrate; the
*operations* determine the profile, exactly as `fp_mode` already works.

---

## 8. Why only MIND can do this

`docs/determinism.md:226` states the parallel claim for loop collapse:

> *"No float compiler can do this: it requires a determinism contract on the
> arithmetic itself."*

The same argument carries here, and it is the reason this is a wedge feature rather
than a convenience. A conservation obligation is only checkable if the arithmetic is
associative and the reduction order is pinned. Over IEEE-754 floats with unspecified
reduction order — the default in every mainstream compiler — `sum(laplacian(F))` is
not a well-defined quantity at compile time, so there is nothing to prove. MIND can
prove it precisely because Q16.16 addition is integer addition and MIND pins the
order.

A competitor could add `field` and `laplacian` as syntax tomorrow. They could not add
`#[conserves]`, because they have no ring to prove it in.

---

## 9. Dependencies and blockers

### 9.1 RFC 0012 Phase B.2 (hard prerequisite for the lowering claim)

B.2 is **partial**: `.T`, `.sum` and `.mean` have shipped; `.max` and MLIR-level
byte-identity with `matmul_rmajor_f32_v`/`dot_*_v` remain deferred, both blocked on
**shape-dim threading from the type-checker into `lower_expr`**.

That threading is the same mechanism a field lowering needs — an operator that does
not know its extents cannot emit a stencil. The RFC document does not block on B.2,
but **the field layer may not be declared complete while the compiler loses the
dimensions its own lowering requires.** A beautiful surface over a substrate that
drops shape information is precisely the failure mode MIND exists to avoid.

`.sum` shipping is directly load-bearing here: the v1 obligations in §5.2 are
statements about global sums.

**Scoped.** `docs/design/rfc0012-b2-shape-threading-scope.md` maps the work to
file:line. Two findings sharpen this dependency:

- The blocker is narrower than RFC 0012 §7.2 implies. `.max` and the norm
  shorthands are *not* gated on shape threading — only `.reshape` and the MLIR
  byte-identity target are.
- `.T`/`.sum`/`.mean` shipped by lowering to **shape-agnostic** IR nodes
  (`Instr::Transpose` carries `{dst, src, perm}` and no dims,
  `src/ir/mod.rs:382`). They did not solve shape threading; they avoided needing
  it. A stencil lowering cannot — which is exactly why this RFC's Phase B
  inherits the dependency that Phase B.2's shipped operators escaped.

### 9.2 The NCCL determinism question (blocker, not a task)

`mind-runtime` has a real TCP transport but its NCCL path is not complete — the
generic backend returns `Ok(())` with the real `ncclAllReduce` calls left in comments
(`mind-runtime` `src/distributed/backend.rs:231,252`), and the CUDA path logs
`"NCCL not initialized, simulating AllReduce"` (`mind-runtime` `src/backend/cuda/nccl.rs:359`).

**A collective that returns `Ok(())` without communicating is worse than a missing
one, because it is silently green.** That should be fixed on its own merits.

But "finish NCCL" is not merely unfinished plumbing, and must not be scheduled as
such: **NCCL ring-allreduce does not pin reduction order across topologies.** The same
eight GPUs in a different ring produce a different float sum. Completing NCCL as-is
would therefore *puncture* the wedge rather than extend it. Distributed field
execution requires either (a) integer/Q16.16 collectives, where reduction order does
not affect the result, or (b) a pinned deterministic reduction tree. That is an
architectural decision that must be made before any distributed field lowering is
designed, and it is out of scope for this RFC.

### 9.3 Priority

This RFC is **not scheduled**. Rust-independence (RI) is the #1 compiler priority and
this sits behind it. Nothing here should be read as competing for that slot.

---

## 10. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| A | `domain` + `field` types. Compile-time only: no new MLIR, no new IR node. `E2310`/`E2311` diagnostics. | Existing suite byte-identical; bootstrap oracle unchanged. Mirrors RFC 0012 Phase A exactly. |
| B | `laplacian` as a semantic operator, lowering to the stencil the hand-written form produces. | **Gate 1** lowering parity against `rfn-mind`'s `laplacian.mind` + `field_step.mind` (§7). Requires B.2 shape-dim threading (§9.1). |
| C | `grad` + `div`. | Byte-identity against hand-written equivalents. |
| D | `#[conserves(zero_sum)]` + `#[conserves(sum)]`, prover, `E23xx`, `evidence_chain.field.*` receipt. | **Gate 2** prove-or-refuse verified both ways: the unscaled/ring-closed cases compile with a re-derivable receipt, and each `E23xx` — including the scaled `A .* laplacian(F)` form — has a test that *fails to compile*. Blocked on open question 4 (§11). |

Phase D is the phase that matters. Phases A–C are the scaffolding that makes it
expressible; D is the part no other compiler can copy.

---

## 11. Open questions

1. **Should `domain` extents be required to be compile-time constant?** Symbolic dims
   (RFC 0012 Phase A) would be more expressive, but a conservation proof may need
   ground extents. Proposed resolution: symbolic dims allowed in the type, ground
   values required at the point a `#[conserves]` obligation is discharged.

2. **Is `Vec2<q16>` needed for `grad`'s result, or should `grad` return a field of
   rank+1?** The latter avoids a new type constructor. Proposed resolution: rank+1,
   pending a look at how `rfn-mind` actually consumes gradient-like quantities.

3. **Does `#[conserves]` compose across function boundaries?** If `f` and `g` each
   conserve sum, `g(f(x))` does too — but proving that requires interprocedural
   reasoning. Proposed resolution: v1 is intraprocedural only; composition is
   explicitly deferred rather than silently assumed.

4. **How much must `mindc verify` see in order to re-derive the receipt — and does
   that force `domain` to stop being purely compile-time metadata?** This is the
   load-bearing open question and it is deliberately not answered here.

   §4 states `domain` is compile-time-only semantic metadata that erases before
   lowering. §5.5 requires `mindc verify` to *independently re-derive* the
   conservation proof rather than trust the receipt. Those two statements are in
   tension: re-deriving `sum(laplacian(F)) == 0` requires knowing the boundary rule
   (`Periodic`), the stencil's coefficient structure, and that the operation is
   ring-closed. If the verifier needs that, it must be *in the artifact*.

   Two possible resolutions, with very different costs:

   - **(a) Receipt-carried.** The `evidence_chain.field.*` MAP entry names a
     `theorem_id` + `stencil_id` + boundary rule + ring, and the verifier re-derives a
     *closed-form* theorem from those tokens plus the mic@3 hash of the proven
     function body. `domain` stays erased; only the proof's premises are recorded.
     This is additive MAP data and does **not** create a new IR concept — consistent
     with §6's "no new IR."

   - **(b) IR-carried.** The verifier needs to re-walk the field's structure in the
     IR, which means domain topology must survive lowering as first-class IR data.
     This **would** contradict both §4 and §6 and would make this a much larger RFC.

   (a) is strongly preferred and is what §5.5's receipt contents are written toward.
   But (a) is only sound if the enumerated `stencil_id` set is closed and each entry's
   theorem is fixed at a known version — i.e. the verifier re-derives *a specific
   named theorem*, not an arbitrary proof. Confirming that is a prerequisite for
   Phase D, not an implementation detail to be discovered during it. If (a) turns out
   to be insufficient, this RFC's framing needs revision before Phase D starts.
