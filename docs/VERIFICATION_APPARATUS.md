# Self-Host Port Verification Apparatus & SOTA Roadmap

> Verification design for the source-position type-checker rule ports of the MIND
> self-hosting compiler. Scope: the `self_host_tc_*` smoke corpus under
> `examples/mindc_mind/`, the differential-fuzz mechanism that replaces its
> hand-curated coverage, and the architectural + north-star gaps this exposes for a
> compiler meant to be #1 for the agentic era.
>
> Status legend used throughout: **[LIVE]** already in the tree · **[FIX-LANDED]**
> bug fixed reactively, mechanism to prevent recurrence not yet built ·
> **[BUILD-NOW]** designed, highest-ROI, next to land · **[PROPOSED]** designed,
> scheduled behind the build-now item · **[DEBT]** known liability, tracked.
>
> No result below is claimed as done that is not in the tree. The three corpus bugs
> were caught by humans after the fact; the apparatus exists to make that mechanical.

---

## 0. Why this document exists

MIND's wedge is a **trust** claim: a deterministic compiler whose output is
bit-identical across CPU/ARM/GPU, carrying a hash-anchored, tamper-evident evidence
chain in the artifact itself. A self-hosting compiler that makes that claim cannot
have diagnostic rules that silently diverge from its own reference implementation.
The point where the pure-MIND port of a type-checker rule disagrees with live
`mindc` is precisely the point where the trust claim fails at its root — the
compiler would be lying about its own faithfulness.

Over one hardening session, three real divergences were found in the source-position
rule ports (E2002 twice, E2003 once). All three lived on the **one axis the harness
never systematically exercised**: `token-class × position-context`. All three were
found by a human building fixtures by hand, in three separate review rounds. This
document specifies the apparatus that converts that axis from "a shape somebody
remembered to write down" into "a generated cross-product diffed against the
authoritative oracle," and sets the architectural direction that removes the axis as
a bug surface entirely.

The four target ports and their canonical exports (verified in
`examples/mindc_mind/main.mind`):

| Rule | Code | Export | main.mind |
|------|------|--------|-----------|
| `fn_value_call`    | E2012 | `pub fn selftest_tc_fn_value_call(src, src_len, pos, std_src, std_len) -> i64` | 39368 |
| `unknown_ident`    | E2002 | `pub fn selftest_tc_unknown_ident(...)`    | 40037 |
| `undeclared_assign`| E2009 | `pub fn selftest_tc_undeclared_assign(...)`| 40292 |
| `unknown_call`     | E2003 | `pub fn selftest_tc_unknown_call(...)`     | 40656 |

Shared contract: return `1` = fires, `0` = in-domain / no-fire, `-3` = fail-closed
decline. The `0`-vs-`-3` distinction is internal and invisible to a differential
diff against live `mindc` — which is exactly why a decline-at-a-firing-position
(fail-open) is caught as a divergence.

---

## 1. Bug taxonomy — what the apparatus targets

Every rule is a boolean over `(source, position)`: does code `C` fire at token `T`?
Four failure classes, three of which were observed live this session and one of which
is the harness failure that let the others hide.

### 1.1 UNDER-FIRE (false negative / suppression)
The rule declines (`0` or `-3`) at a position where live `mindc` emits the code.
**Observed — E2002 round 1:** six value positions had zero fixtures — `if` / `while`
/ `match` condition-tail, struct-literal field value, method receiver, match-guard —
and the port suppressed E2002 on all six. Net effect: undefined identifiers accepted
silently. Proximate cause: the divergent shapes were never enumerated, so they were
untested by construction.

### 1.2 OVER-FIRE (false positive)
The rule fires where live `mindc` is silent. **Observed — E2002 round 2:** boolean
literals `true`/`false` (which lex as `Literal::Int`, not `tk_ident`) and
keyword-spelled identifiers were classified as unknown idents and wrongly flagged.
Under-fire and over-fire are **duals of the same rule** yet took two rounds, because a
single reviewer chasing "where does it wrongly decline" does not simultaneously chase
"where does it wrongly fire."

### 1.3 FAIL-OPEN / FOLDED-TOKEN DECLINE
The rule fail-closes (`-3`) at a firing position, so it reads as no-fire against a
live diagnostic. **Observed — E2003:** an `import`/`use`/`pub`/`else` token in callee
position folds to a non-`tk_ident` lexeme; the port's callee arm declined (`-3`) while
live `mindc` emitted E2003. Mechanically an under-fire, but via a distinct
cause (lexer fold), and detectable only because the diff treats `-3` and `0` alike.

### 1.4 ORACLE COLLUSION / NULL-GATE (the meta-bug)
Not a rule bug — a **harness** bug that makes 1.1–1.3 invisible. Two forms:
- **Collusion.** The intended safety is 3-leg agreement `got(port) == leg2(Python
  recompute) == live(mindc)`. In E2002 round 1, leg-2 re-implemented the port's own
  token-window classifier, so `got == leg2` held trivially — a null gate that
  manufactured a false-green over a real defect. Round 2's leg-2 "shared the blind
  spot" (no independent bool-literal knowledge).
- **No live leg.** 8 of 17 tc smokes have no live-`mindc` oracle at all (§4.2). For
  those rules, if the Python leg-2 shares a port bug, there is **no third independent
  check** — the null-gate condition, still live across a third of the corpus.

This class is ranked most severe: it is the gate lying green over a real defect, not
merely staying silent.

### 1.5 Taxonomy → axis map

| Class | This-session instance | Axis coordinate |
|-------|----------------------|-----------------|
| Under-fire | E2002 r1 (6 positions) | `ident × {cond-tail, field-value, receiver, guard}` |
| Over-fire | E2002 r2 | `{bool-literal, keyword-ident} × value-use` |
| Fail-open / folded | E2003 | `{folded keyword token} × callee` |
| Null-gate | E2002 r1 collusion; 8 NOLIVE smokes | (orthogonal — a property of the *gate*, not the grid) |

The first three all live in the `token-class × position-context` product. The fourth
is why the first was not caught for a round. The apparatus is built around exactly
this: a generator for the grid, plus a meta-gate that proves the gate itself cannot
collude.

---

## 2. The layered apparatus

Four stages, fail-loud and deterministic, each gating the next. Stage 1 is the net;
Stage 4 stops the net from ever lying green.

```
 source-position rule port (.so)
        │
   ┌────▼─────────────────────────────────────────────┐
   │ STAGE 1  DIFFERENTIAL FUZZER  (tcdiff)  [BUILD-NOW]│  the net
   │  template-bank × filler-matrix × light nesting     │
   │  live `mindc check` = authoritative oracle         │
   │  catches under-fire + over-fire + fail-open in one │
   │  seeded sweep; self-tested by a planted-bug gate   │
   └────┬───────────────────────────────────────────────┘
        │ shrunk fixtures + per-cell coverage report
   ┌────▼───────────────────────────────────────────────┐
   │ STAGE 2  COVERAGE-GAP FLOOR             [BUILD-NOW]  │  proof of reach
   │  every (token-class × position-context) cell must   │
   │  have ≥1 parseable case per rule, or RED            │
   │  a template at ~100% parse-discard = a hole, not a  │
   │  pass; shared uniformly across all 4 rules          │
   └────┬───────────────────────────────────────────────┘
        │ shrunk fixtures + port-vs-live diff
   ┌────▼───────────────────────────────────────────────┐
   │ STAGE 3  MULTI-LENS ADVERSARIAL PANEL   [PROPOSED]   │  semantic reach
   │  fixed cross-family roles, in parallel:             │
   │  under-fire · over-fire · fail-open/folded ·        │
   │  position-context enumerator · oracle-independence  │
   │  runs AFTER the fuzzer is green, only on shrunk set │
   └────┬───────────────────────────────────────────────┘
        │
   ┌────▼───────────────────────────────────────────────┐
   │ STAGE 4  ORACLE-INDEPENDENCE GATE (OIA)  [PROPOSED]  │  guard the guards
   │  A static: sentinel-disjointness, no port FFI in    │
   │    leg-2 call graph, no classifier fingerprint      │
   │  B/C dynamic: mutation kill-matrix + non-redundancy │
   │  promotes the 8 NOLIVE smokes into scope            │
   └─────────────────────────────────────────────────────┘
```

### 2.1 Stage 1 — tcdiff differential fuzzer  [BUILD-NOW]

`examples/mindc_mind/self_host_tc_diff_fuzz.py` (stdlib only). Reuses ~300 lines of
verified plumbing from `self_host_tc_unknown_ident_smoke.py` verbatim rather than
re-deriving it: `build_so()` (line 1134, `mindc main.mind --emit-shared`, loaded
once), the `mind_verdict` ctypes wrapper (1187, `argtypes=[c_int64]*5`,
`restype=c_int64`), `bundled_modules()` + std concat, `bare_builtins()`,
`stmt_keywords()` + `MIND_STMT_KW_GATE`, `collect_decl_names()`, `DIAG_RE`, and the
`line/col` math (verified against live).

**Differential invariant** (uniform across all four rules). For each generated source
`src` with a marked hole at `pos`:
```
codes_all = all_live_codes(src)
if "E1001" in codes_all:  skipped_unparseable += 1; continue   # never scored
codes_at  = live_codes_at(src, pos)                            # (line,col) match
for rule in RULES:
    port_fires = (port_call(rule, src, pos) == 1)
    live_fires = (RULE_CODE[rule] in codes_at)
    DIVERGENCE iff port_fires != live_fires
```
The `0`-vs-`-3` internal distinction never enters the comparison — that is *why*
fail-open (§1.3) is caught: `-3` collapses to `port_fires = False`, live E2003 is
present, divergence.

**Generator = template-bank × filler-matrix × light nesting.** Templates span the
**position axis** (`T_LET_RHS`, `T_RETURN`, `T_STRUCT_FIELD_VAL`, `T_IF_COND`,
`T_WHILE_COND`, `T_MATCH_SCRUT`, `T_MATCH_GUARD`, `T_METHOD_RECV`, `T_BINOP`, `T_ARG`,
`T_CALLEE`, `T_ASSIGN_TGT`, `T_ANNOT`, plus `T_NESTED` at random `if`/loop/block depth
0–3 to exercise scope-frame depth). Each carries one hole marked by a unique token so
the shrinker can re-find it and recompute `pos`. Fillers span the **token-class axis**,
regenerated each run so they cannot drift: fresh undefined names, pre-declared
bound-locals (flip polarity → no-fire cases), each `stmt_keywords()` entry as ident,
folded non-idents `{import,use,pub,else,fn,let,if}`, bools `{true,false}`, ints,
every `bare_builtins()` name, a sampled std export, qualified `Foo::Bar`, and
prefixes `__mind_alloc`/`tensor.add`. A parse gate discards non-parsing combos
(counted, never scored); per-template discard-rate feeds Stage 2.

**Shrinker.** Delta-debug on any divergence: remove a top-level fn, remove a non-hole
statement, unwrap one nesting level, rename long idents to one char — accept iff the
parse gate passes AND the divergence still holds at the recomputed hole; iterate to
fixpoint; emit the minimal parseable fixture + `{rule, port_verdict, live_codes_at}`.

**Two mandatory self-tests (non-null proof).**
- **planted mode (runs first).** Copy `main.mind`, apply one deterministic patch
  reproducing a historical class (canonical: force `unknown_call`'s folded-keyword
  callee arm to return `-3` — the E2003 fail-open), build `bad.so`, rerun; assert
  ≥1 divergence AND the shrunk class == planted class. If not detected → exit FAIL.
  **A gate that cannot catch a planted bug is not a gate.**
- **ci mode.** Budget 1500, fixed seed → assert `divergences == 0`; assert two live
  position sentinels (E2002 value-use present, E2003 callee present) proving
  position-matching intact; assert the Stage 2 coverage floor.
- **nightly mode.** Random seed, large budget, optional batch (N unique-named
  single-fn modules concatenated behind one `mindc check`, amortizing subprocess
  10–50×); any divergence uploads shrunk fixtures.

**CI wiring.** Append to `fast_keystone.sh` after the tc block, beside the other tc
smokes:
```
chk "tc_diff_fuzz (E2002/3/9/12 token × position differential)" \
    python3 examples/mindc_mind/self_host_tc_diff_fuzz.py
```
`.so` built once; ~5–15 ms/case; 1500 cases ≈ 15–25 s + build. ci mode runs the
planted-bug gate then a fixed-seed bounded sweep — deterministic, safe for the gate.
**Out of scope:** the 16 selector-int rules (enum indices) need a separate
enum-enumeration harness, not this token×position grid.

### 2.2 Stage 2 — coverage-gap floor  [BUILD-NOW]

Promotes tcdiff's per-template / per-filler discard report into a **hard** gate.
Every `(token-class × position-context)` cell must have ≥1 parseable case **per rule**
or the gate is RED. A template discarding ~100% is a coverage hole, not a pass. The
matrix is **generated** — hand-listing positions in a `CASES` literal is the failure
mode, not the fix. Applied **uniformly across all four rules**, so a class fixed in
one is forced onto its siblings — this is what would have stopped the E2002 class from
recurring as E2003. This converts "we happened to fuzz it" into "the axis is provably
covered."

### 2.3 Stage 3 — multi-lens adversarial panel  [PROPOSED]

Cross-family reviewer fan-out with **fixed adversarial roles** — under-fire hunter,
over-fire hunter, fail-open/folded-token hunter, position-context enumerator,
oracle-independence auditor — run in parallel over the shrunk fixtures Stage 1
surfaces plus the port diff. This is agent orchestration, not code: it collapses the
three sequential single-reviewer rounds (which converged on one class each) into one
pass, and it runs **after** the fuzzer is green, only on the shrunk set — never as the
first-line detector. A single blind reviewer locks onto one hypothesis per round; that
is the mechanical reason E2002 needed two rounds and E2003 a third.

### 2.4 Stage 4 — oracle-independence gate (OIA)  [PROPOSED]

The meta-gate that guards the guards — it proves the two oracles cannot collude and
that no rule's only smoke lacks a live leg.

- **Layer A (static, cheapest — ships right behind tcdiff).** Add a machine-readable
  header to each 3-leg smoke:
  `# oracle-legs: port=... leg2=... live=... position_sentinel=-3`
  and statically parse leg-2's AST call graph for **(A1)** sentinel-disjointness (only
  the port may return `-3`; leg-2 may return only its declared resolution sentinels
  `{0,1,-99}`), **(A2)** no reference to the ctypes port handle, **(A3)** no classifier
  fingerprint (leg-2 must not replicate the tokeniser / position / `tc_ui_shape`
  classifier). **Layer A alone would have caught the E2002 round-1 collusion.**
- **Layers B+C (dynamic).** Mutation kill-matrix (each planted mutation must be killed
  by ≥1 leg) + a non-redundancy ledger (no two legs are the same check) + promotion of
  the 8 NOLIVE smokes into scope. Highest generality, built last.

tcdiff itself must pass its own planted-bug mutation gate (§2.1) before it counts as a
Stage-4-blessed gate.

### 2.5 Which layer catches which corpus bug

| Session bug | Class | Caught by |
|-------------|-------|-----------|
| E2002 r1 — 6 suppressed value positions | under-fire | **S1** fuzz surfaces all six in one sweep; **S2** floor forces the cells |
| E2002 r1 — colluding leg-2 | null-gate | **S4-A** sentinel/classifier-fingerprint static gate (would have caught it standalone) |
| E2002 r2 — bool/keyword over-fire | over-fire | **S1** bool/keyword fillers flip polarity → divergence; **S2** lexer-fold cell |
| E2003 — folded keyword callee | fail-open | **S1** `-3`→`port_fires=False` vs live E2003 → divergence; **S2** folded-token×callee cell; **S2 cross-rule floor** forces the E2002 class onto E2003 |
| 8 NOLIVE smokes | null-gate (latent) | **S4-B/C** live-leg promotion + mutation kill-matrix |

Every observed bug is caught by Stage 1 or Stage 4; Stage 2 turns "caught because we
fuzzed it" into "cannot regress because the axis is a gated floor"; Stage 3 reaches the
semantic classes the filler matrix does not yet parameterize.

---

## 3. Build order (value/effort weighted — not the runtime order)

1. **tcdiff (Stage 1)** — medium effort, reuses ~300 lines of verified plumbing,
   mechanically catches all three historical classes in one seeded sweep. Highest ROI.
   Build now.
2. **OIA Layer A (Stage 4 static)** — near-zero build (manifest header + call-graph
   parse of the 3 existing 3-leg smokes). Would have caught the actual collusion
   incident. Ships right behind tcdiff.
3. **Coverage-gap floor (Stage 2)** — the fuzz→coverage bridge; promotes the discard
   report to a hard, cross-rule-uniform gate.
4. **Multi-lens panel (Stage 3)** — agent orchestration; codified in the agent defs,
   used on the next port.
5. **OIA Layers B+C (Stage 4 dynamic)** — mutation kill-matrix + non-redundancy ledger
   + no-live-leg promotion of the 8 NOLIVE smokes. Highest generality, build last.

**Runtime pipeline** of any future source-position rule port:
`FUZZ (tcdiff, live mindc authoritative)` → `COVERAGE-GAP FLOOR (grid provably
covered)` → `MULTI-LENS PANEL (semantic classes)` → `ORACLE-INDEPENDENCE GATE (the
guards cannot collude)`. Stage 1 is the net; Stage 4 stops the net from lying green;
tcdiff must pass its own planted-bug gate before it counts.

---

## 4. Current-corpus liabilities (honest state)

### 4.1 Coverage is hand-enumerated — the proximate cause of all three bugs
Every tc smoke drives its oracle off a hand-authored `CASES` list or a hand-picked
combinatorial sweep. The port is only ever checked on shapes a human wrote down. The
reactive fixes are literally commented `(review 2026-07-23)` / "the 6 previously
divergent shapes" — added only after each miss. **[FIX-LANDED for the found shapes;
DEBT for the axis]** — Stages 1+2 are the durable fix.

### 4.2 A third of the corpus has no live leg — un-auditable  [DEBT, HIGH]
8 of 17 tc smokes have **no** live-`mindc` oracle — 2-leg (port vs Python recompute)
or self-golden: `class_mismatch`, `class_rules`, `narrowing`, `let_class_mismatch`,
`fixed_bytes_into_vec`, `shape_rules`, `shape_annot_compat`, `classify_error_code`.
For each of those rules (E2004/5/6/10/11/13/15/16/23/2101/2/3, `match::*`), a shared
port bug in leg-2 has no third check — the same null-gate condition that produced the
E2002 collusion, still live. Each must gain a live leg **or** a hand truth-table
adjudicated independently of the port. Blocked on OIA Layers B/C.

### 4.3 Oracle-independence is guarded in only 3 files  [DEBT]
The Rule-3a independence marker exists only in `unknown_ident`, `unknown_call`,
`undeclared_assign`. It is a **comment convention**, not a machine check. OIA Layer A
turns it into a parsed manifest gate.

### 4.4 The type-pair axis is over-tested, the token×position axis untested
`shape_rules` runs 168 combinations, `fixed_bytes` 100, several class rules a full
4×4 `{i32,i64,f64,bool}` matrix — but each exercises only 1–3 distinct front-end
**shapes**. No file enumerates `token-class × position-context`. The type-pair axis
proves one shape 168 times while the axis carrying the bugs carries them untested.
Stage 2 is the correction.

### 4.5 The emitter has a fuzz corpus; the diagnostic rules do not
`gap_corpus_smoke.py` is a fuzz-discovered regression set — but static, and scoped to
the mic@3 **emitter** byte-identity path, not the type-checker verdict path where all
three bugs live. The project already knows fuzzing finds front-end divergences; it
never pointed a generator at the diagnostic rules. tcdiff closes this gap.

---

## 5. Architectural recommendation — token-window classifier vs shared AST shape layer

**The question.** Position-context ("is this token a value-use? a callee? an
assign-target?") is currently decided at diagnosis time by `tc_ui_shape`, a
**token-window classifier**: it inspects a sliding window of tokens around a source
position and infers the role heuristically. Every source-position rule calls into it.
The alternative is a **shared AST shape layer**: the parser, which already resolves
each token into a syntactic role, tags the node with its position-context once, and
every rule reads that tag instead of re-deriving it.

**Finding.** All three corpus bugs are token-window failures, not rule-logic failures:
- Over-fire (E2002 r2): the window mis-read `true`/`false` (which lex as
  `Literal::Int`) and keyword-spelled idents — a **lexeme-fold ambiguity** the window
  cannot see through.
- Fail-open (E2003): a folded keyword token in callee position — again a fold the
  window mis-classifies.
- Under-fire (E2002 r1): six positions the window's heuristic never handled.

The token window is a heuristic re-derivation of information the parser **already
computed and threw away**. Its failure modes are structural: any token whose lexeme
folds to a different kind, and any position the heuristic did not special-case, is a
latent bug — and the space of both is open-ended.

**Recommendation: two-phase — contain now, cure structurally.**

- **Phase A (defense, now).** Ship the apparatus (§2) to **contain** the token-window
  classifier. tcdiff + the coverage floor make the entire `token-class × position`
  axis a gated grid; OIA makes the gate honest. This is deployable behind the current
  architecture with no compiler surgery and is the prerequisite for Phase B.
- **Phase B (cure, structural).** Migrate position-context to an **AST-carried shape
  layer**: the parser emits a shape tag per node, once; `tc_ui_shape` becomes a lookup,
  not a heuristic; every rule reads the same tag. This **eliminates the fold-ambiguity
  class entirely** — the parser has already resolved the token into an AST role, so a
  folded keyword can no longer be silently re-read in a downstream window — and it kills
  the `token-class × position` axis as a bug surface instead of testing it forever.

**Why the apparatus is the enabler, not a competitor, of Phase B.** A shared shape
layer is a larger change and must itself satisfy MIND's load-bearing invariants: the
shape tags must be **deterministic**, part of the versioned wire format, and their
derivation bit-identical across substrates. The way to land such a change safely is to
prove the new layer is byte-equivalent to the old classifier **across the whole
generated grid** before cutover — which is exactly what tcdiff + the coverage floor
provide. **Build the apparatus first; it is what makes the architectural cure
provably safe.** Until Phase B lands, the token-window classifier stays and the
apparatus is the containment; do not ship Phase B without a green differential over
the full grid on both x86 and ARM.

---

## 6. SOTA / north-star gap analysis

**North star.** MIND as the #1 compiler/language for the agentic AGI era — chosen by
AI agents, differentiated by innovation on **determinism + a tamper-evident evidence
chain** (bit-identical across x86/ARM/GPU, hash-anchored evidence embedded in the
artifact). Positioning discipline: say **tamper-evident**, not "signed," until opt-in
signing is on.

**Why this apparatus sits on the critical path of that thesis, not beside it:**

1. **The wedge is a trust claim; a self-hosting compiler must prove its own port
   faithful.** "Deterministic + tamper-evident" is worthless if the compiler's own
   diagnostic rules can silently diverge from its reference. The apparatus is
   **meta-determinism**: the mechanism by which the compiler demonstrates its pure-MIND
   port agrees with the authority on every generated case. This is the innovation an
   agent-era compiler needs and that mainstream toolchains do not offer.
2. **Agents consume compilers at a scale that breaks human-curated coverage.** Agents
   emit huge volumes of programs, hit rare shapes constantly, and cannot eyeball a
   diagnostic. Correctness coverage must be **generated, not enumerated**, and error
   verdicts must be **reproducible and auditable byte-for-byte** so an agent's tool
   loop over `mindc` is deterministic. The hand-curated `CASES` corpus is precisely the
   human-scale assumption that fails at agent scale — the taxonomy in §1 is that failure
   made concrete.

**Gaps to SOTA (honest, ranked):**

| # | Gap | State | Closer |
|---|-----|-------|--------|
| 1 | Correctness coverage is hand-enumerated (human-scale) | DEBT | Stages 1+2 |
| 2 | 8/17 tc rules have no independent live oracle — un-auditable | DEBT, HIGH | OIA B/C |
| 3 | Oracle independence is a comment, not a machine gate | DEBT | OIA A |
| 4 | No differential fuzzing on diagnostic rules (only the emitter) | DEBT | tcdiff |
| 5 | Position-context via token-window heuristic, not AST | ARCH DEBT | §5 Phase B |
| 6 | Verification result is CI green/red, not a tamper-evident receipt in the evidence chain | GAP | §7 move 4 |

Gap 6 is the one that is *new* here: the apparatus currently would produce a CI signal,
but a compiler whose differentiator is a tamper-evident evidence chain should make "this
port was proven faithful to its reference" a **checkable artifact**, not a screenshot.

---

## 7. Top architectural moves toward "#1 language for AI/agents by innovation"

1. **Generative-over-curated correctness.** Every source-position rule's coverage
   becomes a generated cross-product with a hard floor (Stages 1+2). Retire
   hand-listed `CASES` as a *sole* oracle. This is the reflex the whole taxonomy
   demands: a shape nobody enumerates is a shape nobody tests.
2. **Meta-determinism as a machine gate (guard the guards).** OIA — manifest header +
   sentinel-disjointness + mutation kill-matrix — so no gate can lie green. A
   trust-differentiated compiler must be able to prove its *verifiers* are independent,
   not just assert it in a comment. This is the structural form of the "default to FAIL
   unless proven" discipline.
3. **AST-carried shape layer (§5 Phase B).** Make position-context a first-class,
   versioned, deterministic AST property shared by every rule; delete the token-window
   heuristic. Removes the `token-class × position` bug axis structurally rather than
   containing it forever.
4. **Verification receipts in the evidence chain.** Emit a deterministic,
   hash-anchored **verification receipt** — which rules were fuzzed, the seed, the
   coverage-floor status, the OIA pass/fail — that rides with the artifact. This fuses
   the apparatus with the tamper-evident-evidence wedge: "this compiler was proven
   faithful to its own reference on a covered grid" becomes a tamper-evident,
   independently checkable claim. (Tamper-evident, not signed, until signing is on.)
5. **Cross-substrate parity of the verification itself.** Run tcdiff + the coverage
   floor on x86 **and** ARM (and the emitter path on GPU) so the faithfulness proof is
   itself bit-identical across substrates — the same invariant that sells the compiler,
   applied to the compiler's own self-proof. A Phase-B cutover (§5) must show a green
   grid on both ISAs before it lands.
6. **Deterministic, agent-facing diagnostics.** Diagnostic codes and positions must be
   reproducible byte-for-byte so an agent's loop over `mindc` is stable; the apparatus
   enforces this as a side effect (live `mindc` authoritative + position-matching
   sentinels). Determinism of *errors*, not just of *output*, is part of being the
   compiler agents choose.

Moves 1–3 are engineering already designed above; moves 4–6 are the north-star
extensions that turn a green CI gate into a differentiating, tamper-evident capability.

---

## 8. Codified agent-hardening invariants

These bind on any future source-position rule port (recorded here; the operative copies
live in the agent definitions):

- **DIFFERENTIAL-FUZZ-BEFORE-BLIND-REVIEW.** No source-position tc rule port
  (E2002/E2003/E2009/E2012 and any future src+pos rule) is "done" until
  `self_host_tc_diff_fuzz.py` is green against it in the port-.so-vs-live-mindc
  differential. Blind review runs **after** the fuzzer is green, only on the shrunk
  fixtures it surfaces — never as the first-line detector.
- **TOKEN × POSITION COVERAGE FLOOR.** Every new/edited source-position rule must be
  exercised across the full generated grid `{ident, stmt-keyword, bool-literal,
  int-literal, folded-non-ident, builtin-name, std-export, qualified Foo::Bar,
  operator} × {value-use, callee, assign-target, annotation-type, struct-field-value,
  if/while/match cond-tail, method-receiver, match-guard, let-name, pattern-binder}`.
  An unexercised cell (or a template at ~100% parse-discard) is RED. The matrix is
  generated; hand-listing positions is the failure mode, not the fix.
- **ORACLE-INDEPENDENCE / RULE 3a (HARD).** Leg-2 must never replicate the port's
  tokeniser/position/shape (`tc_ui_shape`) classifier and never call back into the
  ctypes port `.so` handle. Leg-2 may return only its declared resolution sentinels
  `{0,1,-99}`; only the port may return `-3`. Live `mindc check` is authoritative on
  every case. Every 3-leg smoke carries the `# oracle-legs:` manifest header and must
  pass the OIA static gate. A leg-2 that can emit `-3`, or whose call graph reaches the
  `.so`, is a null gate → reject the port.
- **NO-LIVE-LEG = UN-AUDITABLE.** A 2-leg smoke (port vs Python recompute, no live
  `mindc check`) is null-gate-prone and may not be the sole gate for a rule. The 8
  current NOLIVE smokes are flagged HIGH until each gains a live leg or a
  hand-adjudicated truth table.
- **LEXER-FOLD PROBE.** Every source-position rule must be probed with tokens that fold
  to a non-`tk_ident` lexeme in expression/callee/assign position — keywords, bools,
  int literals, operators. A dead/commented fold entry is tested-shape **debt**, not
  evidence the shape was handled; it must carry a live fixture.
- **CROSS-RULE FLOOR.** When a divergence class is found and fixed in one rule, the same
  class is added to the generated matrix for **every** sibling rule sharing the
  tokeniser / `tc_ui_shape` substrate (E2002/E2003/E2009/E2012 and the source+name
  rules). This is what stops an E2002 class from recurring as E2003.

---

*Verification is not a phase that ends; for a compiler whose product is trust, it is
the product. Build the net first, prove the net cannot lie, then remove the surface the
net was catching.*
