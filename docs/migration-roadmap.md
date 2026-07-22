<!-- MIND migration roadmap — any source language -> pure, executing, byte-identical MIND.
     Companion to docs/runs-burndown-roadmap.md (the internal RUNS surface this depends on). -->

# MIND Migration Roadmap — Any Language → Pure Executing MIND

> **North star:** migrate code from any source language into pure, **executing**,
> **byte-identical** MIND. The artifact's whole value is the determinism claim, so the
> bar is not "it runs" — it is **"it runs AND reaches byte-identity (claimable)."**
> Owner agent: `mind-migrator` (already exists). Foundation: [`runs-burndown-roadmap.md`](runs-burndown-roadmap.md).

---

## 1. Ground truth & referee architecture

The **Rust `mindc` is ground truth.** During translation we do **not** use the
byte-identity keystone (7/7) as the in-loop judge — wrong latency, wrong granularity.
Instead:

- **Build a behavioral referee against the Rust `mindc`.** Compile the ported MIND with
  the shipped `mindc`, run it, and diff its **output** against the source program's
  reference output. That is the in-loop correctness judge — the original codebase is the
  ground truth either way, so a referee can always be built even when none is inherited.
- **Keep byte-identity as the FINAL gate, not the loop gate.** Claimability is proven
  once, batched, at the end of a pass — never per-unit.

### The latency split forces a two-tier checker (not a choice)

| Tier | Check | Cost | Where |
|------|-------|------|-------|
| **In-loop, per-unit** | `mindc check` (fmt+lint+type-check) + behavioral referee (run + output-diff vs ground truth) | seconds | inside the port loop, every unit |
| **Batched, per-pass** | keystone 7/7 byte-identity + cross-substrate (avx2==neon) | ~25 min | outside the loop, once per pass — the **claimability** gate |

A port that passes the in-loop tier but fails the batched tier **runs but cannot be
claimed** — see §3.

---

## 2. The queue primitive is SYMBOL-level (resolved, not assumed)

A file-on-disk queue ("does the translated file exist") **does not fit MIND.** The backlog
is **539 `aot_only` symbols** keyed on their **dominant blocker construct**
(from `runs-burndown-roadmap.md`, baseline 878/1417 RUNS = 62.0%):

| Blocker construct | Symbols | Share of deficit |
|-------------------|--------:|-----------------:|
| `non_i64_param`   | 186 | 34.5% |
| `struct`          | 170 | 31.5% |
| `strings`         | 94  | 17.4% |
| `non_i64_let`     | 31  | 5.8% |
| `Vec`/`Map`       | 29  | 5.4% |
| `non_i64_return`  | 21  | 3.9% |
| `borrow`          | 7   | 1.3% |

**66% of the deficit is `non_i64_param` + `struct`** — the same root gap: the shipped
binary lowers only the i64-scalar ABI, so any aggregate / collection / non-i64 scalar in a
signature, body, or `let` forces `aot_only`. The queue primitive is therefore:

> Rebuild the **symbol** worklist from the codegraph each pass → group by blocker construct
> → fix the construct's deterministic-MLIR lowering **once** (unblocks the whole bucket) →
> `mg` verifies each emitted symbol actually **RUNS** → fan-out fixers over what still doesn't.

Fixing a construct once clears a whole bucket, so the burndown order is share-weighted:
struct-ABI + non-i64-ABI first (356 symbols), then `String`, then `Vec`/`Map`.

---

## 3. Three things that make MIND a harder case (stated before anyone starts)

1. **The gap inventory is the hard part — and it is not memory management.** MIND has **no
   host-language escape hatch**; the gap is *"what does the source language do that MIND
   cannot express or run at all."* Two tiers, both first-class artifacts that must exist
   before serious porting:
   - **Doesn't-compile / crashes** — e.g. ~30-deep nesting overflows lowering recursion →
     SIGABRT (workaround: extract a helper fn). The deep structural refusals.
   - **Compiles-but-AOT-only** — the 539-symbol RUNS surface in §2.

   Deliverable: a mechanically-regenerated **gap inventory** seeded from (a) the RUNS
   blocker histogram and (b) a crash/refuse corpus. *This artifact does not exist yet and
   is the prerequisite for everything below.*

2. **"Worst case is delete the branch" does not hold for us.** A port from a shipping
   language leaves the original shipping either way; a MIND port that **runs but cannot
   reach byte-identity ships nothing**, because the value is the determinism claim. The
   failure mode is not a wasted branch — it is a **port that runs and cannot be claimed**,
   which is a liability (it invites a determinism claim we cannot back). Migration ROI must
   be argued against *that* risk, not against branch-cost.

3. **Compiler placement is forced** by the latency split in §1 — `mindc check` (seconds)
   in-loop, keystone 7/7 (~25 min) batched-outside. Not negotiable.

---

## 4. Model/agent-tier discipline (adopt regardless of any port)

Reserve the **strongest tier for reviewers and for anything that writes rules other agents
follow** — the gap inventory, the referee spec, the per-construct lowering patterns.
Fan-out per-symbol implementation uses lighter tiers. Rule-authoring and review are where
tier actually matters; bulk implementation is not. (Default today is the opposite — fix it.)

---

## 5. Phases

- **P0 — Gap inventory + queue enumerability** *(prerequisite; enumerability RESOLVED:
  symbol-level)*. Build `docs/mind-gap-inventory.md`: the RUNS blocker histogram (have it) +
  a crash/refuse corpus (nesting-overflow SIGABRT + siblings). Mechanically regenerated.
- **P1 — Behavioral referee harness.** `port_referee`: compile ported MIND with the shipped
  `mindc`, run, diff output vs the source's reference output. In-loop, seconds. Wire
  `mind-migrator` to it as the loop judge; deterministic reference-capture for the source.
- **P2 — `aot_only` burndown (internal migration, the honest first target).** Symbol
  worklist from the codegraph, grouped by blocker; ship the construct's deterministic-MLIR
  lowering once (struct ABI → `String` → `Vec`/`Map` → f32/f64), `mg`-verify RUNS, batched
  keystone claimability gate. Target: **62% → 100% RUNS.** This ships the surface P3/P4 need.
- **P3 — Single-file cross-language port.** Port ONE small external file (Rust/Python/C)
  into pure MIND **within the shipped RUNS surface**. In-loop `mindc check` + referee;
  batched byte-identity claimability. Proves the loop end-to-end on foreign code.
- **P4 — Project-scale migration.** Scale the loop: rebuild queue from disk each pass,
  fan-out fixers, gap-inventory-driven **loud refusal** for out-of-surface constructs. Gated
  on P2 having shipped enough surface for the target's construct profile.

---

## 6. Gates (every port, every phase)

- **In-loop:** `mindc check` PASS **and** behavioral-referee output-match vs ground truth.
- **Batched (claimability):** keystone 7/7 byte-identity + cross-substrate. A port that
  runs but is not byte-identical is **not done** — it is a liability (§3.2).
- **No fake wins:** `mg` RUNS verification per symbol; no `aot_only`→RUNS demotion without a
  cited, confirmed classifier bug (the burndown audit found zero).

---

## 7. Open 15-minute looks (before committing more)

- ✅ **Queue primitive** — resolved: symbol-level (§2).
- **Crash/refuse corpus** — enumerate the "doesn't compile at all" constructs beyond the
  nesting overflow (the P0 crash tier).
- **Referee reference-capture** — how to capture a source program's reference output
  deterministically when the source language admits nondeterminism (clock/rand/hashorder).
