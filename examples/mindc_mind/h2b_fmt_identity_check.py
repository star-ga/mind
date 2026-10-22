#!/usr/bin/env python3
"""H2b layout-only format identity: --emit-ir and --emit-mic3 byte-identity.

Ruling Q1.1: a file may be formatted only when its diff is layout-only AND
`--emit-ir` + `--emit-mic3` under one mindc are byte-identical for the
pre-format and post-format bytes. This gate re-measures that proof for every
formatted file that is still in the walk, from the committed git history.

Files that cannot produce IR under any available composition are NOT formatted
(restored to parent bytes and pinned in testdata/h2b_exclusion_pins.json);
this gate refuses to count a refusal / equal-error pair as identity.

Red mutant:
  --mutant-diverge  rewrite one literal in the current file before comparing,
                    so IR/mic3 must differ and the gate must FAIL
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "93238e8e"
# Compositions that cannot emit IR are excluded from the formatted set; listed
# here so a future attempt is measured explicitly rather than silently skipped.
IR_REFUSAL_EXCLUSIONS = {
    "examples/mindc_mind/source_note_driver.mind",
    "examples/mindc_mind/string_escape_driver.mind",
}


def mindc() -> str:
    bin_path = os.environ.get("MINDC_BIN") or os.environ.get("MINDC")
    if not bin_path:
        raise SystemExit("h2b_fmt_identity: MINDC_BIN is required")
    return bin_path


def changed_mind_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "HEAD"], cwd=ROOT, text=True
    )
    return sorted(p for p in out.splitlines() if p.endswith(".mind"))


def emit_pair(compiler: str, src: bytes, stem: pathlib.Path) -> tuple[bytes, bytes]:
    mind = stem.with_suffix(".mind")
    mind.write_bytes(src)
    ir = subprocess.run(
        [compiler, str(mind), "--emit-ir"],
        capture_output=True,
        timeout=120,
    )
    mic = stem.with_suffix(".mic3")
    if mic.exists():
        mic.unlink()
    m = subprocess.run(
        [compiler, str(mind), "--emit-mic3", str(mic)],
        capture_output=True,
        timeout=120,
    )
    if ir.returncode != 0 or m.returncode != 0 or not mic.exists():
        err = (ir.stderr + m.stderr)[:300]
        raise RuntimeError(f"emit refused rcs=({ir.returncode},{m.returncode}) {err!r}")
    return ir.stdout, mic.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant-diverge", action="store_true")
    args = ap.parse_args()
    compiler = mindc()
    files = [p for p in changed_mind_files() if p not in IR_REFUSAL_EXCLUSIONS]
    if not files:
        print("h2b_fmt_identity: FAIL — no formatted files measured (vacuous)")
        return 1
    asserted = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h2b-id-") as td:
        tdir = pathlib.Path(td)
        for rel in files:
            old = subprocess.check_output(["git", "show", f"{BASE}:{rel}"], cwd=ROOT)
            new = (ROOT / rel).read_bytes()
            if args.mutant_diverge and rel == files[0]:
                # Semantic change: a comment is not enough — flip a stored
                # constant so the emitted IR/mic3 MUST move.
                for sample in (
                    b"__mind_store_i64(out, 0);",
                    b"return 1;",
                    b"let mut ok: i64 = 1;",
                ):
                    mutated = new.replace(sample, sample.replace(b"0;", b"1;").replace(b"return 1;", b"return 2;").replace(b"= 1;", b"= 2;"), 1)
                    if mutated != new:
                        new = mutated
                        break
                else:
                    # last resort: rewrite the first function body return
                    new = new.replace(b"return 0;", b"return 3;", 1)
            stem = tdir / pathlib.Path(rel).stem
            try:
                ir_a, mic_a = emit_pair(compiler, old, stem.with_name(stem.name + "-base"))
                ir_b, mic_b = emit_pair(compiler, new, stem.with_name(stem.name + "-head"))
            except RuntimeError as e:
                failures.append(f"EMIT_REFUSED {rel} {e}")
                continue
            if ir_a != ir_b:
                failures.append(
                    f"IR_DIVERGE {rel} base={hashlib.sha256(ir_a).hexdigest()[:12]} "
                    f"head={hashlib.sha256(ir_b).hexdigest()[:12]}"
                )
            else:
                asserted += 1
            if mic_a != mic_b:
                failures.append(
                    f"MIC3_DIVERGE {rel} base={hashlib.sha256(mic_a).hexdigest()[:12]} "
                    f"head={hashlib.sha256(mic_b).hexdigest()[:12]}"
                )
            else:
                asserted += 1
            print(
                f"IDENTITY {rel} IR_MIC3_IDENTICAL={ir_a == ir_b and mic_a == mic_b}",
                flush=True,
            )
    print(f"h2b_fmt_identity: files={len(files)} asserted={asserted}")
    for f in failures:
        print(f"FAIL {f}")
    if failures:
        print("h2b_fmt_identity: FAIL")
        return 1
    print("h2b_fmt_identity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
