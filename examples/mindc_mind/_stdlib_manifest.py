"""Shared reader for the committed std/*.mind manifest.

enforced-by: STDLIB-MANIFEST

Why this exists
---------------
"Which std modules does the native backend link" used to live as SEVENTEEN
hand-copied literal lists — one in ``src/bin/mindc.rs::run_native_backend_bridge``
and sixteen ``_STDLIB_MODULES`` / ``STD_MODULES`` copies across
``examples/mindc_mind/*.py`` — plus a fourth, DIFFERENT list
(``STDLIB_MIND_SOURCES``, 23 entries) in ``src/project/stdlib.rs``, with nothing
asserting any of them agreed. Two hand-copied lists that must agree, with no check,
is the most common defect class in this tree.

``testdata/stdlib_manifest.txt`` is the one committed contract. This module is the
Python reader; ``mindc.rs::frozen_std_seed_modules`` is the Rust reader, and it
parses the same file with the same rules (``include_str!`` at compile time, so the
shipping binary keeps no runtime file dependency).
``stdlib_manifest_lint.py`` gates the whole thing in both directions.

ORDER IS LOAD-BEARING: ``seed_modules()`` returns the seed rows in FILE ORDER,
which is the byte order of the std blob the frozen ``stage1.elf`` was minted from.
Reordering, adding or removing a seed row changes the compiled bytes of every
native build — a whole-corpus RESEED event, never a routine edit.
"""

import hashlib
import pathlib
import re

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]

MANIFEST_PATH = _HERE / "testdata" / "stdlib_manifest.txt"
STD_DIR = _REPO / "std"

# Kept in lockstep with mindc.rs::FROZEN_STD_SEED_COUNT. Both are deliberate
# tripwires, not redundancy: the seed set cannot change without a stage1.elf
# re-mint, so a quiet one-line manifest edit must fail loudly on both sides.
FROZEN_STD_SEED_COUNT = 21


def _rows():
    """(module, native_seed, bundled) for every manifest row, in file order.

    Parsing rules are the exact twin of ``mindc.rs::frozen_std_seed_modules``:
    drop ``#`` comment lines and blank lines, split the rest on TAB.
    """
    out = []
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{MANIFEST_PATH}:{lineno}: expected 3 tab-separated fields, got "
                f"{len(fields)}: {line!r}"
            )
        module, native_seed, bundled = fields
        if native_seed not in ("seed", "no"):
            raise ValueError(
                f"{MANIFEST_PATH}:{lineno}: native_seed must be 'seed' or 'no', "
                f"got {native_seed!r}"
            )
        if bundled not in ("bundled", "no"):
            raise ValueError(
                f"{MANIFEST_PATH}:{lineno}: bundled must be 'bundled' or 'no', "
                f"got {bundled!r}"
            )
        out.append((module, native_seed, bundled))
    if not out:
        raise ValueError(f"{MANIFEST_PATH}: no rows — refusing to return an empty std set")
    return out


def all_modules():
    """Every std module the manifest knows about, in file order."""
    return [m for m, _, _ in _rows()]


def seed_modules():
    """The std blob the native bridge feeds to the frozen stage1.elf, IN ORDER."""
    return [m for m, seed, _ in _rows() if seed == "seed"]


def bundled_modules():
    """The modules ``STDLIB_MIND_SOURCES`` resolves for ``use std.<m>``."""
    return [m for m, _, bundled in _rows() if bundled == "bundled"]


def seed_blob(std_dir=None):
    """The exact std-blob prefix of the stage1.elf stdin image.

    Byte-identical to the Rust bridge's composition: each module's bytes followed
    by a single ``\\n``.
    """
    d = pathlib.Path(std_dir) if std_dir is not None else STD_DIR
    return b"".join((d / f"{m}.mind").read_bytes() + b"\n" for m in seed_modules())


def declared_seed_sha256():
    """The ``SEED_BLOB_SHA256`` golden recorded in the manifest header, or None.

    The lint recomputes the blob and compares; the golden is a positive pin on the
    seed set AND its order, so a reseed cannot land as a silent edit.
    """
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    m = re.search(r"^#\s*SEED_BLOB_SHA256\s+([0-9a-f]{64})\s*$", text, re.M)
    return m.group(1) if m else None


def actual_seed_sha256(std_dir=None):
    """sha256 of the blob composed from the manifest + the std/ files on disk."""
    return hashlib.sha256(seed_blob(std_dir)).hexdigest()


# The manifest's own header counts: each stated once, in its canonical form, equal to the rows (lint check 7).
_HEADER_COUNT_RE = re.compile(
    r"^#\s*(SEED_COUNT|BUNDLED_COUNT|TOTAL_COUNT)\s+(0|[1-9][0-9]*)\s*$"
)
_NON_SEED_RE = re.compile(r"^#\s*(0|[1-9][0-9]*) modules are not `seed`:")
_NON_SEED = "non-seed"
# Any comment line that LOOKS like a count statement — any case, `_`/`-`/space
# separators, a misspelt name, or a leading "<n> modules" — must be one of the
# canonical forms above, so no spelling of a count escapes the check.
_COUNT_LIKE_RE = re.compile(
    r"^#\s*(?:[a-z]+(?:[_-][a-z]+)*[_-]count\b"
    r"|(?:seed|bundled|total|non[\s_-]*seed)\s+count\b"
    r"|[0-9]+\s+modules?\b)",
    re.I,
)


def count_header_errors(text, total, seed, bundled):
    """Every count stated in the manifest's comment lines, judged against the rows.

    The whole file is read, not a header window: a count line a merge left after
    the rows is still a count line. Empty result = each count appears exactly
    once, in canonical form, with the real value.
    """
    want = {
        "SEED_COUNT": seed,
        "BUNDLED_COUNT": bundled,
        "TOTAL_COUNT": total,
        _NON_SEED: total - seed,
    }
    seen = {name: [] for name in want}
    errors = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.startswith("#"):
            continue
        m = _HEADER_COUNT_RE.match(line)
        if m:
            seen[m.group(1)].append((lineno, int(m.group(2))))
            continue
        m = _NON_SEED_RE.match(line)
        if m:
            seen[_NON_SEED].append((lineno, int(m.group(1))))
            continue
        if _COUNT_LIKE_RE.match(line):
            errors.append(
                f"manifest line {lineno} states a count in a form the lint cannot "
                f"check: {line!r} (write `# SEED_COUNT <n>`, `# BUNDLED_COUNT <n>`, "
                f"`# TOTAL_COUNT <n>` or `# <n> modules are not `seed`:`)"
            )
    for name, real in want.items():
        hits = seen[name]
        if not hits:
            errors.append(f"manifest header has no {name} count line (rows give {real})")
        elif len(hits) > 1:
            lines = ", ".join(str(lineno) for lineno, _ in hits)
            errors.append(
                f"manifest states {name} {len(hits)} times (lines {lines}) — exactly "
                f"one count line per count"
            )
        for lineno, n in hits:
            if n != real:
                errors.append(
                    f"manifest line {lineno}: {name} says {n} but the rows give {real}"
                )
    return errors


def count_header_probes(text, total, seed, bundled):
    """Hostile variants of the REAL manifest text; the check must refuse each one.

    Built from the real header so the probes track its format: a probe that no
    longer applies leaves the text unchanged and is then reported as accepted.
    """
    t, s, b, ns = total, seed, bundled, total - seed
    tot, sd, bd = f"# TOTAL_COUNT {t}", f"# SEED_COUNT {s}", f"# BUNDLED_COUNT {b}"
    nsd = f"# {ns} modules are not `seed`:"
    body = text.rstrip("\n")
    return {
        "duplicated, same value": text.replace(tot, f"{tot}\n{tot}", 1),
        "duplicated, merge shape": text.replace(bd, f"{bd}\n# BUNDLED_COUNT {b - 12}", 1),
        "stale value": text.replace(sd, f"# SEED_COUNT {s + 1}", 1),
        "missing line": text.replace(tot + "\n", "", 1),
        "stale non-seed": text.replace(nsd, f"# {ns - 2} modules are not `seed`:", 1),
        "duplicated non-seed": text.replace(nsd, f"{nsd} (copy)\n{nsd}", 1),
        "missing non-seed": text.replace(nsd, "# Some modules are not `seed`:", 1),
        "lower-case spelling": text.replace(tot, f"{tot}\n# total_count {t + 1}", 1),
        "dash spelling": text.replace(tot, f"{tot}\n# TOTAL-COUNT {t + 1}", 1),
        "space spelling": text.replace(tot, f"{tot}\n# Total count {t + 1}", 1),
        "misspelt name": text.replace(tot, f"{tot}\n# BUNDELD_COUNT {b}", 1),
        "trailing junk": text.replace(sd, f"{sd}x", 1),
        "second number": text.replace(sd, f"{sd} {s + 1}", 1),
        "leading zero": text.replace(sd, f"# SEED_COUNT 0{s}", 1),
        "after the rows": f"{body}\n# BUNDLED_COUNT {b - 12}\n",
        "uncheckable module count": text.replace(tot, f"{tot}\n# {t} modules are bundled", 1),
    }
