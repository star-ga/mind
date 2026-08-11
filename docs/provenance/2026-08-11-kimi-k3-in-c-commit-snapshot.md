# Forensic provenance snapshot — `FareedKhan-dev/kimi-k3-in-c` vs MIND

> **Captured:** 2026-08-11 (PDT)
> **Purpose:** preserve public commit chronology and technically relevant prior-art anchors for later provenance review.
> **Status:** evidence log only; **not an allegation of copying or theft**.
>
> This record distinguishes three questions that must not be conflated:
> 1. whether MIND work was public earlier;
> 2. whether another project uses similar engineering ideas;
> 3. whether source code or other copyrightable expression was actually copied or adapted.
>
> The evidence below establishes (1) and documents potentially relevant similarities for (2). It does **not**, by itself, establish (3).

## 1. External repository snapshot

Repository: `FareedKhan-dev/kimi-k3-in-c`

Observed default branch: `main`

Observed HEAD at capture time:

- `ff11dce858a2eb8a781224facdffd33a1fa48d25`
- commit title: `Release v1.0.0: verified end to end, and faster, output unchanged`
- author timestamp: `2026-08-07T16:38:50Z` (`2026-08-07 21:38:50 +05:00`)
- canonical commit: `https://github.com/FareedKhan-dev/kimi-k3-in-c/commit/ff11dce858a2eb8a781224facdffd33a1fa48d25`

The commit object is signed/verified by GitHub at the captured HEAD. The commit message explicitly states that the release preserves **byte-identical output** while adding fused matmul kernels, KDA head-parallel recurrence, speculative decode, and other optimizations.

A Git commit SHA identifies the committed tree/history at that point. Recording the SHA therefore gives a durable reference even if `main` later moves.

## 2. Relevant initial K3-in-C public commit sequence

The early repository history appears as a tightly clustered staged import on **2026-08-01**. The timestamps establish when these commits were publicly represented in Git history; they do **not** prove when the author first wrote the underlying material.

| SHA | Timestamp | Commit | Relevance |
|---|---|---|---|
| `1e10dfc764f5ac97a3dee3bfe44bdd76d2dc88c7` | 2026-08-01 16:04:39 +05:00 | `chore: add license, attribution and editor configuration` | Apache-2.0 project license + NOTICE/attribution baseline. |
| `5fca44a754877b86499ae87407988163eb8bb7e4` | 2026-08-01 16:04:39 +05:00 | `feat(core): add the public API and the configuration reader` | Public API/configuration invariants. |
| `f3495cae5e5c5c7af6e84c4fa8a4913ee9b840de` | 2026-08-01 16:04:40 +05:00 | `chore(third-party): vendor the JSON parser and the BPE tokenizer` | Third-party provenance recorded in NOTICE. |
| `4f6b4a48548b73fefc5f06b090946f549c99b107` | 2026-08-01 16:04:40 +05:00 | `feat(core): implement the numeric kernels` | **Primary similarity anchor:** floating-point contract, scalar/OpenMP/AVX2 bit identity, explicitly pinned summation order, `-ffp-contract=off`. |
| `4a1730ef2c99853772f6b08de7714c95aa30b19e` | 2026-08-01 16:04:40 +05:00 | `feat(io): add the safetensors reader` | K3-specific I/O work; no MIND provenance claim made here. |
| `5b0944c6d6d0a7e314e390477797eefaa56928d1` | 2026-08-01 16:04:44 +05:00 | `test: add the weightless gate suite` | Per-kernel fixtures + end-to-end oracle/gate suite. |
| `4bb5e035b728339bcbafbc588073692e0e38bf44` | 2026-08-01 16:04:44 +05:00 | `test: add fixtures generated from the PyTorch reference` | Adversarial fixture/oracle methodology. |
| `c181cf0196a36cb00c035d3716a0d18e3cf2584f` | 2026-08-01 16:04:45 +05:00 | `feat(tools): add the fixture and oracle generators` | Reproducible oracle generation. |
| `224dfda330d25e4544569e2bb26c1f2d9c172ec3` | 2026-08-01 16:04:45 +05:00 | `feat(tools): add the verification and analysis tools` | Layer-by-layer conformance and independent verification. |

### Load-bearing external commit

`4f6b4a48548b73fefc5f06b090946f549c99b107` states in its commit message:

- the **floating-point contract** is the load-bearing part;
- scalar, OpenMP and AVX2 paths must produce **bit-identical** results;
- loops are deliberately partitioned and reduced with a fixed tree so the vector path can reproduce the scalar arithmetic;
- `-ffp-contract=off` is required.

Initial `src/core/k3_ops.c` at that commit:

`https://github.com/FareedKhan-dev/kimi-k3-in-c/blob/4f6b4a48548b73fefc5f06b090946f549c99b107/src/core/k3_ops.c`

The initial matmul used four double accumulators and an explicitly fixed reduction:

```text
(a0 + a1) + (a2 + a3)
```

The accompanying comments explain that the topology is intentional because floating-point addition is non-associative and the scalar/vector implementations must agree bit-for-bit.

This is a similarity in **engineering principle and execution contract**. It is not an exact implementation match to MIND's strict-f32 8-lane schedule.

## 3. MIND public prior-art chronology

The following MIND commits predate the K3-in-C Aug-1 public history and are the strongest provenance anchors for deterministic / byte-identical numerical execution.

| SHA | Timestamp | Commit | Prior-art significance |
|---|---|---|---|
| `7293f0235cec22e6f8dd4806dbdefded0f0b580f` | 2026-05-24 18:12:07 -07:00 | `docs(rfcs): draft RFC 0014 ... + RFC 0015 (cross-substrate bit-identity)` | Defines cross-substrate bit-identity proof obligation, normative reduction order and oracle infrastructure. RFC itself records `Created: 2026-05-25`. |
| `c9c38d3fbc4216a7d534471272cb1c412623ca45` | 2026-05-29 00:00:58 -07:00 | `ci: enforce cross-substrate bit-identity gate on x86_64 + ARM` | Makes cross-ISA output identity a CI-enforced property rather than documentation only. |
| `8059d8b770e9c1e24d48ef2a404c3886af1492cd` | 2026-06-05 20:34:11 -07:00 | `docs: promote RFC 0015 to Accepted (CI-enforced)` | Pins conformance evidence and exact cross-substrate reference hashes. |
| `e3026545e49d9a523e8db63cd43f2df40682b42d` | 2026-06-28 00:13:52 -07:00 | `docs: add the Determinism Contract` | Specifies strict/fast tiers, fixed reduction order, strict math, exact behavior, and verifiable deterministic output. |
| `4454b3978b7250e5c9090207900967ec035c6388` | 2026-07-02 10:49:02 -07:00 | `feat(codegen): pin -ffp-contract=off — explicit strict-FP determinism contract` | Explicitly disables FMA contraction across codegen paths to protect cross-substrate float identity. |
| `2d9d421a0c2b6ba2be6e2dca1e36bd91205c6f7f` | 2026-07-03 00:05:00 -07:00 | `docs(determinism): f32 vector BLAS reductions are now strict` | Records unfused FMA + **pinned fixed-order fold**, zero fused FMA on x86, bit-exact vector kernels. |
| `8b3cff7895a9bedfbb49fa156d37e6d8d9db506c` | 2026-07-05 10:41:43 -07:00 | `feat(determinism): strict dot_f32/dot_l1_f32 Track-A BLAS` | Rewrites scalar/AVX2 paths to one strict **8-lane + left-to-right fold** with `-ffp-contract=off`; verifies byte identity. |
| `ea59540bfd1b45f1774571b1eca146f8af14d61d` | 2026-07-06 22:38:02 -07:00 | `test(blas): compile smoke .so with -ffp-contract=off` | Demonstrates a real cross-platform failure caused by FMA contraction and closes it in the test path. |
| `e16b98b2fa3eed406ec3505e4cb3b380945e5b49` | 2026-07-14 06:47:08 -07:00 | `test: bless strict-f32 neon canaries on real aarch64 + CI hash-harvest` | Executes strict-f32 bit-identity canaries on real ARM hardware and makes the cross-ISA result fail-closed. |

Canonical anchors:

- RFC 0015: `https://github.com/star-ga/mind/blob/main/docs/rfcs/0015-cross-substrate-bit-identity.md`
- Determinism contract: `https://github.com/star-ga/mind/blob/main/docs/determinism.md`
- Runtime strict-f32 implementation: `https://github.com/star-ga/mind/blob/main/runtime-support/mind_intrinsics.c`

## 4. Technically relevant similarity cluster

The strongest overlap found in the review is a **cluster**, not one generic technique:

1. deterministic/bit-identical numerical execution as a first-class correctness contract;
2. reference scalar arithmetic paired with optimized SIMD/parallel paths;
3. explicit fixed/pinned reduction topology to stop reassociation from changing final bits;
4. explicit FMA-contraction control using `-ffp-contract=off`;
5. correctness gates/oracles around the deterministic contract;
6. optimization treated as acceptable only if it preserves exact output.

MIND has public commits implementing/specifying this cluster before the observed K3-in-C Aug-1 history.

### Important implementation difference

MIND's strict f32 dot path uses an 8-lane schedule followed by a pinned left-to-right horizontal fold. The initial K3-in-C matmul uses four double accumulators reduced as `(a0+a1)+(a2+a3)`.

Therefore this review has **not** found a verbatim copy of the relevant numerical kernel. The similarity is presently at the design/engineering-contract level.

## 5. Negative evidence / limits of current review

As of this capture, the review has **not found** a smoking-gun MIND fingerprint in K3-in-C such as:

- STARGA/MIND copyright text;
- `mic@3` / MIND evidence-chain identifiers;
- MIND canary hashes;
- MIND function/symbol names;
- a substantial verbatim MIND function/comment block.

Targeted GitHub code search can have indexing limitations, so the absence of a hit is not a mathematical proof that no transformed or derivative material exists.

Also, the techniques involved — fixed reduction orders, disabling FMA contraction, deterministic reference kernels, fixtures/oracles — are individually known numerical-computing techniques. The forensic relevance comes from their **specific combination, chronology, and framing**, not from ownership of any one generic idea.

## 6. Licensing boundary

Both repositories use Apache-2.0 licensing. That fact alone does not resolve provenance.

- **Conceptual influence / prior art:** generally a credit/provenance question, not evidence of source-code copying by itself.
- **Actual adaptation of MIND source or copyrightable expression:** may trigger Apache-2.0 redistribution/notice obligations depending on what was reused and how.

No licensing violation is asserted by this note. Determine the reuse category first.

## 7. Recommended contact posture

If contacting the K3-in-C author, ask neutrally whether they had seen or used MIND's deterministic execution work while developing the project.

If the answer is **yes**, distinguish:

- conceptual influence only → request a prior-art acknowledgement/citation;
- implementation/comments/tests adapted → identify affected files and review attribution/NOTICE obligations;
- direct or translated source reuse → preserve the admission and conduct a file-level derivative-work review before making public claims.

Do **not** describe the repository publicly as “stolen” on the basis of this evidence log alone.

## 8. Preservation note

The corresponding machine-readable manifest is stored at:

`docs/provenance/2026-08-11-kimi-k3-in-c-commit-snapshot.json`

For stronger evidentiary preservation later, create an offline clone/bundle of the external repository at the pinned HEAD and hash the bundle independently. The Git SHAs recorded here already provide durable content-addressed commit references, but an offline copy protects against remote deletion or history disappearance.