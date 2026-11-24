# MIND Ecosystem — Version Matrix

> **Last Updated:** 2026-11-23
> **Purpose:** Version-of-record for all MIND-ecosystem repos, with drift flags
> that require attention. Canonical source: grep of each repo's manifest files
> (Cargo.toml, Mind.toml, pyproject.toml, package.json). Update this file
> whenever a repo cuts a release or changes its stated mindc target.

---

## Version table

| Repo | Path | Manifest | Version | mindc target | IR format | Notes |
|---|---|---|---|---|---|---|
| **mind** | `~/mind` | `Cargo.toml` | **0.10.4** | self (the compiler) | mic@1 (text), mic@3 (binary); NATIVE-ELF (normative self-host, `src/native`) | Public repo `star-ga/mind`. Native-ELF self-host fixed point closed 2026-06-25. |
| **mind-mem** | `star-ga/mind-mem` | `pyproject.toml` | **5.0.3** | n/a (Python package) | uses `mic_map.py` (MICB_VERSION=0x02 / mic@2) | PyPI `mind-mem 5.0.3`. MICB_VERSION pin is mic@2 — see drift flag. |
| **mind-nerve** | `star-ga/mind-nerve` | `Mind.toml` | **0.3.1** | mindc ≥0.10.2, <0.11.0 (`[mind]` pin, now ENFORCED) | mic@2 (`ir-format = "mic@2"`) | Python package: `pyproject.toml` uses dynamic version via `mind_nerve.__version__`; `__version__ = "0.3.1"` (PyPI `mind-nerve 0.3.1`). The `ir-format` pin is REFUSED by mindc: no invocation emits mic@2 (see `docs/ir-stability.md`). |

---

## Drift flags

### FLAG-1: mind Cargo.toml vs Mind.toml version mismatch

| File | Value |
|---|---|
| `~/mind/Cargo.toml` | `version = "0.10.4"` |
| `~/mind/Mind.toml` | `version = "0.10.4"` |

**Assessment:** RESOLVED. Both manifests now read `0.10.4`.

---

### FLAG-4: MIND-Law and mind-mem both consume mic@2 (MIC-B) — not mic@3

| Component | mic format consumed |
|---|---|
| `~/mind-law/src/spec_hash.mind` | `canonical_ir.serialize_mic_b` (MIC-B = mic@2) |
| `star-ga/mind-mem/src/mind_mem/mic_map.py` | `MICB_VERSION = 0x02` |

**Assessment:** RFC 0021 §4.5 identifies this as a **byte-preserving cross-repo
migration** requirement: the mic@2 / MIC-B wire format must not change bytes during
the `mic@2` → `mind-model@2` rename, and both MIND-Law and mind-mem must update their
`SPEC_HASH` / `MICB_VERSION` pins in a coordinated version bump. This flag is already
tracked in RFC 0021 step 5 as a blocker on the rename.

**Action:** Coordinate the mic@2 → `mind-model@2` migration across mind-spec, MIND-Law,
and mind-mem as a single cross-repo PR set. Do not rename until all three are ready.

---

## Quick-reference: key version anchors

| Property | Value | Source |
|---|---|---|
| Current mindc release | 0.10.4 | `~/mind/Cargo.toml` |
| mic@1 (canonical IR text) | stable since 0.2.5 | `docs/ir-stability.md` |
| mic@3 (binary codec) | post-0.7.0 (unreleased) | RFC 0021 step 1, mind@5c29f0d |
| Evidence chain (Phase A/B) | post-0.7.0 (unreleased) | RFC 0016, mind@e7c8c28/cadca87 |
| MICB_VERSION in mind-mem | 0x02 (mic@2) | `mind-mem/src/mind_mem/mic_map.py` |
| mind-mem PyPI version | 5.0.3 | PyPI `mind-mem` |
| mind-nerve Python __version__ | 0.3.1 | `mind-nerve/python/mind_nerve/__init__.py` |
| mind-nerve Mind.toml version | 0.3.1 | `mind-nerve/Mind.toml` |
