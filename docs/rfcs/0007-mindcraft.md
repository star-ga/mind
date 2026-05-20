# RFC 0007: Mindcraft — the pure-MIND format / lint / check toolchain

| Field | Value |
|---|---|
| RFC | 0007 |
| Title | Mindcraft — pure-MIND format / lint / check toolchain |
| Status | Phase 2A shipped (mindc v0.6.8) — sequencing gate CLEARED 2026-05-19, see §9 |
| Authors | STARGA Inc. |
| Created | 2026-05-19 |
| Supersedes | — |
| Superseded by | — |
| Related | RFC 0005 (pure-MIND std surface), RFC 0006 (mind-blas) |

## 1. Summary

**Mindcraft** is the MIND language's first-party source-quality toolchain:
deterministic formatting, lint diagnostics, and static checks for `.mind`
source files. It is exposed as three `mindc` subcommands —
`mindc fmt`, `mindc lint`, `mindc check` — not a separate installable.
The umbrella product name for documentation and the website is
**Mindcraft**.

The defining property: Mindcraft's front-end **is the MIND compiler's own
self-hosted front-end**. It does not parse `.mind` source with a
re-implemented parser in another language. The lexer, parser, AST, and
type-checker that Mindcraft analyses are the same pure-MIND components that
the compiler self-hosts (the integrated `examples/mindc_mind` pipeline,
proven byte-identical to the bootstrap reference). Mindcraft is the MIND
toolchain analysing MIND source through MIND's own eyes.

Each lint rule is a `.mind` file declaring an AST pattern plus a
diagnostic. The rule engine, the formatter, and the check pass are all
compiled by the bootstrapped compiler. When Mindcraft ships, the
credibility ladder is complete: the language self-hosts, the compiler
self-hosts, and the toolchain self-hosts.

## 2. Motivation

A language is judged on three self-hosting milestones. The first two are
done: MIND-the-language is expressive enough to host its own compiler, and
the pure-MIND compiler compiles its own source byte-identically (the
bootstrap fixed-point). The third — the toolchain maintains itself in the
language it serves — is the milestone that converts evaluators into
adopters. A formatter or linter written in another language is a standing
admission that the language was not sufficient for its own tooling.

Mindcraft exists to close that gap and to give MIND developers a single,
deterministic, zero-configuration-by-default quality surface:

- **`mindc fmt`** — canonical formatting. Same source in → byte-identical
  formatted output out, on every machine, every run, forever.
- **`mindc lint`** — diagnostic rules over the typed AST. Each rule is a
  `.mind` file; the rule set is data, not hard-coded compiler internals.
- **`mindc check`** — fast static verification (the existing `check`
  subcommand, rebuilt on the shared rule/config surface).

## 3. Design principles

1. **One front-end.** Mindcraft reuses the self-hosted pure-MIND
   lexer/parser/AST/type-checker. There is exactly one MIND front-end in
   the ecosystem and Mindcraft shares it with the compiler. A
   second front-end implemented in any other language is explicitly
   out of scope and would regress the self-hosting property (§7).
2. **Rules are data.** A lint rule is a `.mind` file declaring an AST
   pattern and a diagnostic. Adding a rule does not require modifying the
   engine. The rule set is versioned, inspectable, and itself lintable.
3. **Determinism is a contract, not a goal.** `mindc fmt` output is
   byte-identical for a given (source, Mindcraft version) pair across
   machines, operating systems, and time. `mindc lint` emits a
   byte-identical diagnostic stream for a given (source, rule set,
   config) triple. This extends the build-determinism tier already
   defined for the compiler down to the developer-tooling layer.
4. **Zero config by default, total control when needed.** A project with
   no configuration gets the canonical format and the default rule set.
   Projects that need control get a single declarative configuration
   surface (§5) — never scattered inline pragmas as the only mechanism.
5. **Toolchain, not product.** Mindcraft is `mindc` subcommands, installed
   and versioned with the compiler. There is no separate runtime, no
   separate release cadence, no second binary to keep in sync.

## 4. Command surface

```
mindc fmt   [--check] [--stdin] [PATHS...]
mindc lint  [--fix] [--reporter human|json] [PATHS...]
mindc check [--reporter human|json] [PATHS...]
```

- `mindc fmt` rewrites files to canonical form; `--check` exits non-zero
  on drift without writing (CI mode).
- `mindc lint` evaluates the active rule set; `--fix` applies the
  machine-applicable subset of diagnostics; `--reporter json` emits a
  stable machine-readable stream.
- `mindc check` runs the fast static pass and shares the configuration
  and reporter surface with `lint`.

All three share one configuration file, one severity model, and one
diagnostic schema.

## 5. Configuration surface

Configuration lives in `Mind.toml` under a `[mindcraft]` table. The design
is deliberately small and entirely MIND's own:

- **`$schema` pointer** — a published JSON schema URL so editors offer
  completion and validation on `Mind.toml` without bespoke plugins.
- **Per-rule severity** — every rule takes one of `off` / `info` /
  `warn` / `error`, not a binary enable flag, so projects can tune
  signal without forking the rule set.
- **`[[mindcraft.overrides]]` blocks** — glob-scoped overrides
  (`includes = ["tests/**"]`) layer rule and format settings per
  directory without separate config files.
- **Per-target sections** — `[mindcraft.cpu]`, `[mindcraft.gpu]`,
  `[mindcraft.cerebras]` apply backend-specific rules first-class
  (e.g. a Q16.16-overflow rule that is `error` on a fixed-point target
  and `warn` elsewhere).
- **VCS integration** — `[mindcraft.vcs] use_ignore_file = true` makes the
  default file set follow the repository's ignore rules.
- **Inline suppression** — a single canonical form, Rust-flavoured to
  match MIND's surface syntax: `#[allow(lint::q16_overflow)]` on the
  declaration above the diagnostic, with a mandatory justification
  string: `#[allow(lint::q16_overflow, reason = "...")]`. This is a
  scoped escape hatch, never the primary control mechanism.

The configuration surface is normative MIND design. It is described here
on its own terms; it is not derived from, nor compatible with, any
external tool's configuration format.

## 6. Rule model

A rule is a `.mind` file exporting a pattern function over the typed AST
and a diagnostic constructor. The engine walks the AST once, dispatches
matching nodes to active rules, and collects diagnostics. Because rules
are `.mind`:

- The default rule pack is compiled by the bootstrapped compiler like any
  other MIND program.
- A project may add local rules in its own repository without a plugin
  ABI — they are just more `.mind` files on the rule path.
- Rules are themselves subject to `mindc fmt` and `mindc lint`. The
  toolchain lints its own rule set.

The diagnostic schema (rule id, severity, span, message, optional
machine-applicable fix) is shared verbatim with `mindc check` and is the
stable contract for the `--reporter json` output.

## 7. Self-hosting constraint (load-bearing)

Mindcraft's parser/AST/type-checker **MUST** be the self-hosted pure-MIND
front-end. It **MUST NOT** be a re-implementation of a MIND front-end in
any other language, however convenient an existing analysis framework in
another language might appear.

Rationale: the self-host ladder and bootstrap fixed-point established
that MIND's front-end exists, in MIND, byte-identically. Introducing a
second MIND front-end in another language to power the toolchain would
silently un-self-host the toolchain layer — destroying the exact property
Mindcraft exists to demonstrate. A transitional bridge that drives the
compiler's *existing* front-end is acceptable as a temporary scaffold; a
new foreign-language MIND parser is not, at any phase.

## 8. Determinism contract

| Tier | Mindcraft obligation |
|---|---|
| Build (already normative for the compiler) | `mindc fmt` output is byte-identical for a fixed (source, Mindcraft version). |
| Lint reproducibility | `mindc lint` / `mindc check` emit a byte-identical diagnostic stream for a fixed (source, rule set, config). |
| Rule-set integrity | The default rule pack carries a content hash; a given hash + source ⇒ a given diagnostic set, verifiable in CI. |

`mindc fmt` is idempotent: formatting already-formatted source is a
no-op, byte-for-byte. `mindc fmt --check` in CI is therefore a total,
reproducible gate.

## 9. Sequencing

Mindcraft implementation was **gated** behind the completion of the active
mind-nerve work (native-encoder end-to-end measurement and the criterion
benchmark publication). **That gate CLEARED on 2026-05-19**: the native
pure-MIND encoder runs end-to-end, measured and numerically validated
against the reference encoder at cosine 0.999996 / top-5 route overlap
0.9975 (≥0.92 gate, n=160) with the cross-arch Q16.16 bit-identity
invariant preserved, and the criterion benchmark publication is complete.
The spec and roadmap were published ahead of the gate; the build may now
begin. (Further native-encoder latency optimization continues
independently and is not a Mindcraft prerequisite — the gate is
measurement + numerical correctness, both satisfied.)

### Phase 2A — shipped (mindc v0.6.8, 2026-05-20)

Phase 2A (`mindc fmt` scaffolding, no line-wrap) is complete. All six PR
steps landed on `main`:

| Step | Commit | Description |
|---|---|---|
| 1 | `4cfe7b9` | `feat(parser)`: trivia layer for comment-preserving CST |
| 2 | `bfeffbe` | `feat(fmt)`: scaffolding + canonical walker (Phase 2A, no wrap) |
| 3 | `434da71` | `test(fmt)`: stdlib stability + idempotence + IR-preservation |
| 4 | `696027a` | `feat(mindc)`: fmt subcommand (`--check`/`--diff`/`--stdin`) |
| 5 | `d1f10f6` | `bench(fmt)`: criterion benchmark for mindc fmt (Phase 2A Step 5) |
| 6 | `6df49e5` | `docs(mindcraft)`: Phase 2A fmt reference + RFC 0007 status update |

The normative formatter reference is at `docs/mindcraft/fmt.md`.

**Phase 2B (soft line-wrap at `max_line_length`) is deferred.** The
`max_line_length` setting is validated and stored but has no effect in Phase
2A. Phase 2B remains a separate PR gated on post-Phase 2A review.

Phased delivery going forward:

1. ~~Configuration surface in `Mind.toml`.~~ (Phase 1, shipped `6526029`)
2. ~~`mindc fmt` (canonical pretty-printer, idempotent, `--check` CI mode).~~ (Phase 2A, shipped above)
3. Phase 2B: soft line-wrap at `max_line_length`. (deferred)
4. `mindc check` rebuilt on the shared configuration/diagnostic surface. (Phase 3)
5. Rule engine + default rule pack, rules as `.mind` files, driven by the
   self-hosted front-end. (Phase 3)
6. `mindc lint` (engine + `--fix` + `--reporter json`). (Phase 4)

Bench-gate discipline applies throughout: the compiler frontend latency
floor is preserved; Mindcraft passes are additive and feature-gated, never
in the default-build hot path.

## 10. Non-goals

- Not a multi-language tool. Mindcraft formats and lints `.mind` source
  only.
- Not a separate binary or separate release train. It is `mindc`.
- Not an architecture analyser. Whole-system / cross-module structural
  analysis is a distinct concern handled elsewhere and is out of scope
  for Mindcraft, which is strictly file-level.
- Not a plugin host with a foreign ABI. Extensibility is "add more
  `.mind` rule files," nothing more.

## 11. References

- RFC 0005 — pure-MIND std surface (the front-end substrate Mindcraft
  reuses).
- RFC 0006 — mind-blas (sibling toolchain-adjacent surface; shares the
  feature-gating and determinism discipline).
- The MIND self-host ladder and bootstrap fixed-point (the property
  Mindcraft completes at the toolchain layer).
