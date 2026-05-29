# MIND Ecosystem — Version Matrix

> **Last Updated:** 2026-05-29
> **Purpose:** Version-of-record for all MIND-ecosystem repos, with drift flags
> that require attention. Canonical source: grep of each repo's manifest files
> (Cargo.toml, Mind.toml, pyproject.toml, package.json). Update this file
> whenever a repo cuts a release or changes its stated mindc target.

---

## Version table

| Repo | Path | Manifest | Version | mindc target | IR format | Notes |
|---|---|---|---|---|---|---|
| **mind** | `~/mind` | `Cargo.toml` | **0.7.0** | self (the compiler) | mic@1 (text), mic@3 (binary, post-0.7.0) | Public repo `star-ga/mind`. Cargo.toml `version = "0.7.0"`. `Mind.toml` reads `0.6.8` — see drift flag below. |
| **mind-mem** | `~/mind-mem` | `pyproject.toml` | **4.0.15** | n/a (Python package) | uses `mic_map.py` (MICB_VERSION=0x02 / mic@2) | PyPI `mind-mem 4.0.15`. `__version__ = "4.0.15"`. MICB_VERSION pin is mic@2 — see drift flag. |
| **mind-nerve** | `~/mind-nerve` | `Mind.toml` | **0.1.0-alpha.2** | mindc ≥0.5.0, <0.8.0 | mic@1 (ir-format = "mic@1") | Python package: `pyproject.toml` uses dynamic version via `mind_nerve.__version__`; `__version__ = "0.3.0b8"`. See drift flag. |
| **mind-inference** | `~/mind-inference` | `Mind.toml` | **0.2.0** | not pinned (see drift flag) | mic@1 (standard pipeline) | STARGA Commercial license. No CI verified as of last audit. |
| **512-mind** | `~/512-mind` | CHANGELOG.md | **1.10.1** | mindc 0.2.5 (CHANGELOG §v1.10.1) | MIC-B / mic@2 (`canonical_ir.serialize_mic_b`) | DIFC governance framework. 65+ .mind source modules. See drift flag on mindc target. |
| **mindlang.dev** | `~/mindlang.dev` | `package.json` | **0.1.0** | n/a (Next.js / Cloudflare Pages) | n/a | Site name in package.json is `"v2"`, version `"0.1.0"`. TypeScript/Next.js. |

---

## Drift flags

### FLAG-1: mind Cargo.toml vs Mind.toml version mismatch

| File | Value |
|---|---|
| `~/mind/Cargo.toml` | `version = "0.7.0"` |
| `~/mind/Mind.toml` | `version = "0.6.8"` |

**Assessment:** The Cargo-hosted Rust compiler and the pure-MIND `Mind.toml`
manifest track different version numbers. `Cargo.toml` is the authoritative release
version (the compiler binary, `mindc`, is built from it). `Mind.toml` appears not to
have been bumped when the Cargo version advanced from 0.6.8 to 0.7.0.

**Action:** Bump `Mind.toml` `version` to `0.7.0` to match `Cargo.toml`. Low-risk
one-line change; no functional impact. Tracked as part of #308 / RFC 0021 step 5
(cross-repo version alignment).

---

### FLAG-2: mind-nerve Mind.toml version vs Python __version__ mismatch

| File | Value |
|---|---|
| `~/mind-nerve/Mind.toml` | `version = "0.1.0-alpha.2"` |
| `~/mind-nerve/python/mind_nerve/__init__.py` | `__version__ = "0.3.0b8"` |

**Assessment:** Two independent version strings in the same repo. `Mind.toml`
tracks the MIND-language source package version; the Python `__version__` tracks
the Python MCP/CLI adapter package version. These are legitimately different
artefacts (the MIND binary vs the Python wrapper), but there is no documented
correspondence policy: a user/CI reading the repo cannot easily determine which
component they have at what version.

**Action:** Document the version-skew policy in mind-nerve's README or CONTRIBUTING:
are these expected to be independent, or should they be kept in sync? If independent,
add a note to Mind.toml clarifying the scope of each version string.

---

### FLAG-3: 512-mind mindc target is mindc 0.2.5, compiler is at 0.7.0

| File | Value |
|---|---|
| `~/512-mind/CHANGELOG.md` §v1.10.1 | `mindc 0.2.5 alignment` (last recorded bump) |
| `~/mind/Cargo.toml` | `mindc 0.7.0` |

**Assessment:** 512-mind targets mindc 0.2.5, which is 5 minor versions behind the
current compiler. The gap is large: mindc 0.3.x through 0.7.x added the full
pure-MIND stdlib (RFC 0005), mind-blas (RFC 0006), Mindcraft (RFC 0007), `mindc build`
(RFC 0008), extern "C" + SysV ABI (RFC 0010), `mic@2.1`/`mic@3` (RFC 0014/0021),
and evidence chains (RFC 0016). 512-mind's .mind source (1863 bitwise ops, 65+
modules) may compile on the current mindc but this has not been verified after each
major RFC ship.

The existing audit (2026-05-29 genesis audit) found that 65/67 modules compile; 2
fail. The mindc target version in the CHANGELOG is informational (512-mind has no
`Mind.toml` at the repo root), so there is no machine-enforced compatibility gate.

**Action (P1):** Test 512-mind against the current mindc 0.7.x to confirm/refute
the 65/67 compile claim and identify which two modules fail. Create a `Mind.toml`
at the repo root with a `mindc-min` / `mindc-max` range so future mindc upgrades
are gated mechanically.

---

### FLAG-4: 512-mind and mind-mem both consume mic@2 (MIC-B) — not mic@3

| Component | mic format consumed |
|---|---|
| `~/512-mind/src/spec_hash.mind` | `canonical_ir.serialize_mic_b` (MIC-B = mic@2) |
| `~/mind-mem/src/mind_mem/mic_map.py` | `MICB_VERSION = 0x02` |

**Assessment:** RFC 0021 §4.5 identifies this as a **byte-preserving cross-repo
migration** requirement: the mic@2 / MIC-B wire format must not change bytes during
the `mic@2` → `mind-model@2` rename, and both 512-mind and mind-mem must update their
`SPEC_HASH` / `MICB_VERSION` pins in a coordinated version bump. This flag is already
tracked in RFC 0021 step 5 as a blocker on the rename.

**Action:** Coordinate the mic@2 → `mind-model@2` migration across mind-spec, 512-mind,
and mind-mem as a single cross-repo PR set. Do not rename until all three are ready.

---

### FLAG-5: mind-inference has no mindc version pin

| File | Value |
|---|---|
| `~/mind-inference/Mind.toml` | `version = "0.2.0"` — no `mindc-min` / `mindc-max` |

**Assessment:** mind-inference is the MIND-native LLM inference pipeline. Its
`Mind.toml` declares a package version but no mindc toolchain compatibility range.
This means a toolchain upgrade cannot be caught at the project-manifest level; a
breaking change in mindc would be discovered only at build time.

**Action:** Add `mindc-min` and `mindc-max` to `mind-inference/Mind.toml`, consistent
with the range used in mind-nerve (`0.5.0` ≤ mindc < `0.8.0`). Low-cost hygiene.

---

## Quick-reference: key version anchors

| Property | Value | Source |
|---|---|---|
| Current mindc release | 0.7.0 | `~/mind/Cargo.toml` |
| mic@1 (canonical IR text) | stable since 0.2.5 | `docs/ir-stability.md` |
| mic@3 (binary codec) | post-0.7.0 (unreleased) | RFC 0021 step 1, mind@5c29f0d |
| Evidence chain (Phase A/B) | post-0.7.0 (unreleased) | RFC 0016, mind@e7c8c28/cadca87 |
| MICB_VERSION in mind-mem | 0x02 (mic@2) | `mind-mem/src/mind_mem/mic_map.py` |
| mind-mem PyPI version | 4.0.15 | `mind-mem/pyproject.toml` |
| mind-nerve Python __version__ | 0.3.0b8 | `mind-nerve/python/mind_nerve/__init__.py` |
| mind-nerve Mind.toml version | 0.1.0-alpha.2 | `mind-nerve/Mind.toml` |
| 512-mind version | 1.10.1 | `512-mind/CHANGELOG.md` §v1.10.1 |
| 512-mind last mindc bump | 0.2.5 | `512-mind/CHANGELOG.md` §v1.10.1 |
| mind-inference version | 0.2.0 | `mind-inference/Mind.toml` |
| mindlang.dev version | 0.1.0 | `mindlang.dev/package.json` |
