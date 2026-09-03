# Self-Evolution in MIND

**Status: research track. One rung (L0) is launchable today; the rest are planned.**

This document is the single place where MIND's self-evolution story lives. It exists
because the work was previously scattered across three repos under three different
names — `~/mind-lab` ("bounded RSI"), `naestro/ROADMAP.md` ("R11 Self-Evolution"), and
this repo's roadmap ("evolutionary search / AB-MCTS") — which made it unreadable as one
thing from the outside, and easy to describe wrongly.

---

## 1. The claim, stated precisely

> **Evolutionary search in which every surviving candidate is cryptographically provably
> deterministic.**

Not "an optimizer that occasionally breaks reproducibility." A search where
non-determinism is not a bug to trade off — it is instant death for the candidate.

The public phrasing is **"bounded self-improvement behind a fail-closed gate with a
mandatory human merge."** Never "a self-evolving language."

### Why this is a defensible position rather than a fashionable one

Most self-improving-code systems share one shape: an agent edits code, the tests go
green, the change merges. The weakness is always the same — **the fitness function is
negotiable.** "The benchmark got faster" can be overfitted, gamed, or measured wrong,
and the search will find those exploits because that is what search does.

MIND has a fitness signal that cannot be talked around: **cross-substrate byte-identity.**
`x86 == ARM`, hash-for-hash, recomputed from the actually-emitted bytes. It is binary,
non-negotiable, and machine-checkable. A candidate either reproduces the pinned hash or
it does not exist.

That gate is not a clever piece of code someone could copy in an afternoon; it is the
accumulated discipline of the CI matrix, the canaries, the differential fuzzer, and the
keystone bootstrap. **The moat is the examination, not the student.**

---

## 2. What is already built (and what is not)

| Piece | Where | State |
|---|---|---|
| Search engine + safety model | `~/mind-lab/autoresearch/ar_rsi.py` | Built, 36/36 tests green, **never run** |
| Fitness gate | `tests/cross_substrate_identity.rs` (26 tests) + `tests/mindfuzz_cross_substrate.rs` | Live and CI-gated |
| Pinned canaries | int8 `917d353b`, Q16 `92e2cb75`, gemv-i16 `3238e8c7` | Live |
| L0 search space | `src/mlir/gemm_tuning.rs` (~15 integer constants) | Exists, hand-tuned |
| L0 campaign harness | `~/mind-lab/autoresearch/config.mind-perf-l0.yaml` + `run_mind_gemm.sh` | Built 2026-09-03, baseline in progress |

Honest gaps, stated rather than smoothed over:

- The RSI campaign has **never been run**. "Written and tested" is not "produced a result."
- L1–L3 below are **design, not implementation**.
- `naestro`'s R11 track assumed two `.mind` modules (`hypothesis.mind`, `self_modify.mind`)
  that **do not exist anywhere** in the ecosystem (verified 2026-09-03). See that repo's
  ROADMAP R11 *Prerequisites* block.

---

## 3. The ladder

Rungs are ordered by blast radius. Each rung must produce a measured result before the
next is attempted.

| Rung | Search space | Fitness | Risk | State |
|---|---|---|---|---|
| **L0** | ~15 blocking constants in `src/mlir/gemm_tuning.rs` | canary hashes unchanged ∧ criterion GMAC/s up | ~zero — constants change loop order/blocking, never accumulation order | **launchable now** |
| **L1** | synthesized peephole / rewrite rules | keystone 7/7 ∧ differential fuzzer ∧ objdump-pure | low — rules verified offline before adoption | designed |
| **L2** | `mic@4` codec: opcode + string-table layout | emit→parse→emit fixed point ∧ `trace_hash` stability ∧ size+speed | medium — wire-format version bump | roadmap §4c |
| **L3** | the compiler itself | all of the above ∧ self-host reseed | high — only with digest veto | research |

**Why L0 is safe by construction, not by hope.** The constants change *cache blocking and
loop order*, not the accumulation order inside the microkernel. Integer accumulation is
associative, so regrouping cannot change the result. That makes byte-identity preservation
a structural property of the search space — and the gate then *proves* it every round
rather than trusting the argument.

---

## 4. The three rules that make this SOTA rather than ordinary

### 4.1 The search may never touch its own examination

If a search can weaken the test that judges it, the whole construction is theatre. This is
the single most common way self-improving systems fail.

`ar_rsi.py` already enforces this on the **harness** side with a SHA-256 digest veto
(`protected_digests.json`), which pins both whole files (`ar_chain.py`, `ar_spine/gate.py`,
`ar_spine/decision.py`) and *individual function bodies* (`autorun.py::_append_row`,
`autorun.py::cost_veto_downgrades_keep`). Any mutation to that authority surface auto-discards
the round even though the containing file is otherwise editable.

For compiler-side rungs the protected region **must** be:

- `tests/cross_substrate_identity.rs` and every `reference_hashes.toml`
- `tests/mindfuzz_cross_substrate.rs`
- the keystone bootstrap gate

Today these are protected only by *exclusion* — they are not in any campaign's `target_files`,
and `check_diff_scope` discards out-of-scope edits. That is a belt without suspenders: the
existing `protected_digests.json` covers harness authority, **not** the compiler's gate files.
Extending the digest manifest to cover them is a prerequisite for L1 and above, and is not
done yet.

### 4.2 Every surviving candidate carries its evidence chain

Artifacts already carry `trace_hash = sha256(emit_mic3(ir))`. So an evolution log is not
"we ran 10k candidates, trust us" — it is a **reproducible chain a third party can replay
step by step.**

No other self-improving system can show the provenance of its own evolution. *This* is the
distinctive claim — not "our agent edits code," which anyone can say.

### 4.3 Promotion is a human merge, always

`ar_rsi.py` has no auto-bind path from model output into the live loop, by construction.
Mutated code runs **only on an isolated clone**; the live tree is never written to.

This is a design invariant, not a phase-one precaution to be relaxed later.

---

## 5. Fail-closed, concretely

`run_mind_gemm.sh` is the L0 evaluator. It emits `gmacs: 0` — an automatic discard — on
every one of:

- `mindc` fails to build
- **any** canary hash shifts (byte-identity broken)
- the bench self-skips or its output cannot be parsed

A real number is emitted **only** when every gate passes. The hashes are recomputed from
the emitted bytes, so a proposer cannot self-report its way past the gate.

Two failure modes this caught on its first runs (2026-09-03), both worth recording because
they are the failures a self-improving loop is *most* likely to mistake for signal:

1. A non-interactive shell had no `~/.cargo/bin` on `PATH`; `cargo` was not found and the
   gate returned `gmacs: 0` rather than a false green.
2. Bare `mlir-opt` on `PATH` resolved to **LLVM 18** instead of the pinned **LLVM 20**, so
   every gate died on `failed to legalize builtin.unrealized_conversion_cast` — and the
   wrapper reported it as *"BYTE-IDENTITY GATE FAILED — a canary shifted."*

Both correctly discarded, so no false green. But (2) exposed a real defect: **an environment
fault was being reported as a determinism failure.** Left alone, the search would have spent
every round hunting a canary drift that never happened. The wrapper now separates the two —
a toolchain fault still discards (an unmeasurable candidate is worth zero either way) but is
labelled as such.

The general lesson, and the reason it is written down here: *a fail-closed gate keeps you
honest about **whether** a candidate passed; it does not automatically keep you honest about
**why** it failed.* A search optimizes against the reason it is given. Both the campaign
config and the evaluator now pin `MLIR_OPT` / `MLIR_TRANSLATE` explicitly rather than
inheriting whatever the launching shell happened to resolve.

---

## 6. Relationship to the other repos

- **`~/mind-lab/autoresearch`** — owns the search engine and the safety model. Generic, not
  MIND-specific.
- **this repo (`~/mind`)** — owns the fitness function (the gates) and the L0–L2 search
  spaces. Roadmap entries: `docs/roadmap.md:475` (AB-MCTS on the structural pack) and
  `docs/roadmap.md:1521` (`mic@4` codec search).
- **`~/naestro`** — R11 is a *different* track: it self-modifies the agent OS from research
  signal, not the compiler from a determinism gate. It shares the safety shape (isolated
  execution, fail-closed gate, human promotion) and should port `ar_rsi.py`'s pattern rather
  than invent one.

---

## 7. What would falsify the thesis

Stated up front, because a claim that cannot fail is not a claim:

- L0 runs and finds **no** improvement over the hand-tuned constants → the search space is
  already saturated and the ladder's first rung is worthless.
- A candidate passes the gate but is later found non-deterministic on a substrate not in
  the CI matrix → the fitness function is weaker than claimed.
- The digest veto is bypassable → rule 4.1 is unenforced and the construction is theatre.

Each is a concrete experiment, not a rhetorical hedge.
