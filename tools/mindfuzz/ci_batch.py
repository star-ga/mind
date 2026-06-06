#!/usr/bin/env python3
"""MIND-Fuzz deterministic CI batch -> cross-substrate candidate staging.

This is the reproducible batch the `cross_substrate_identity` CI job runs. It
fuzzes a FIXED set of (seed, iters) pairs with the deterministic template
mutator (--no-llm equivalent: never calls an LLM, no clock, no RNG, counter-only
instruction selection), and stages every survivor as a candidate cross-substrate
workload (the survivor program + the host's canonical output hash).

The staged batch is committed under tests/mindfuzz_cross_substrate/ and checked
by tests/mindfuzz_cross_substrate.rs, which the EXISTING two-runner
cross_substrate_identity matrix runs on BOTH avx2 (ubuntu-24.04) and neon
(ubuntu-24.04-arm). Each runner recomputes each survivor's output hash and
asserts it equals the committed reference, so the neon runner asserts
byte-identity against the avx2-blessed hash — that IS the wedge oracle, now real.

Determinism contract (every run produces byte-identical staging):
  * fixed BATCH below (seed file + iteration budget), in fixed order;
  * template mutator only (use_llm=False), counter-seeded instruction choice;
  * canonical-driver args are LCG-seeded from a fixed constant;
  * manifest.tsv lines are written in batch order; the survivor .mind files are
    the exact mutated text.

Usage:
  python3 ci_batch.py --out <staging_dir>          # regenerate the staging dir
  python3 ci_batch.py --out <dir> --check          # fail if it differs from
                                                   #   the committed staging dir
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import fuzz_loop

HERE = Path(__file__).resolve().parent

# The FIXED deterministic batch. Each entry fuzzes a seed for a fixed budget with
# the template mutator. Keep this small + scalar-entry so CI is fast and every
# survivor is canonical-drivable (exposes `f(i64)->i64`). The scalar_arith seed
# accretes redundant lets / dead conditionals / counting loops / widenings onto a
# pure-integer function — exactly the SSA-lowering surface the wedge must keep
# byte-identical across ISAs.
BATCH: list[tuple[str, int]] = [
    ("scalar_arith.mind", 6),
    ("scalar_accum.mind", 6),
]

# Default committed staging dir (relative to the repo root, two levels up).
DEFAULT_OUT = HERE.parents[1] / "tests" / "mindfuzz_cross_substrate" / "staged"


def generate(out: Path) -> int:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    total = 0
    for seed_name, iters in BATCH:
        seed = HERE / "seeds" / seed_name
        print(f"\n=== batch: seed={seed_name} iters={iters} (template mutator) ===")
        # use_llm=False -> fully deterministic; stage_dir=out -> survivors staged.
        rc = fuzz_loop.run(seed, iters, False, HERE / "mutations.txt", out)
        if rc not in (0,):
            # rc 3 = a violation was found; that is a genuine fuzz finding, not a
            # batch-staging failure. Surface it loudly.
            print(f"!! batch seed {seed_name} returned rc={rc} (violation found)")
            return rc
    manifest = out / "manifest.tsv"
    if manifest.exists():
        total = sum(1 for ln in manifest.read_text().splitlines() if ln.strip())
    print(f"\n[ci_batch] staged {total} candidate(s) into {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MIND-Fuzz deterministic CI batch")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and assert it byte-matches "
                         "the committed staging dir (CI reproducibility gate)")
    args = ap.parse_args()

    if not fuzz_loop.oracles.MINDC.exists():
        print(f"ERROR: mindc not found at {fuzz_loop.oracles.MINDC}", file=sys.stderr)
        return 2

    if not args.check:
        return generate(args.out)

    # --check: regenerate fresh and diff against the committed dir.
    import tempfile

    committed = args.out
    tmp = Path(tempfile.mkdtemp(prefix="mindfuzz_ci_batch_"))
    rc = generate(tmp)
    if rc != 0:
        return rc
    cm = (committed / "manifest.tsv").read_text() if (committed / "manifest.tsv").exists() else ""
    nm = (tmp / "manifest.tsv").read_text() if (tmp / "manifest.tsv").exists() else ""
    if cm != nm:
        print("!! ci_batch --check FAILED: staging manifest drifted.", file=sys.stderr)
        print(f"  committed:\n{cm}\n  regenerated:\n{nm}", file=sys.stderr)
        return 1
    print("[ci_batch] --check OK: regenerated staging byte-matches committed manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
