#!/usr/bin/env python3
"""MIND-Fuzz loop -- LLM-mutation differential testing for the MIND compiler.

Technique: arXiv:2501.00655 ("Finding Missed Code Size Optimizations in
Compilers using LLMs"), §2. Seed program + a predetermined list of mutation
instructions -> an off-the-shelf LLM incrementally mutates the program -> a
differential test runs at EACH step -> iterate until it no longer compiles or an
oracle triggers a violation. Violations are saved and minimized.

MIND-native twist: the differential oracle is MIND's cross-substrate
byte-identity (determinism / reference / verify / mic@3), not C/Rust code size.

Determinism of the LOOP itself (autorun.py-style, replayable):
  * the mutation instruction at step i is instructions[i % N] (COUNTER, not rand)
  * no clock / random in any decision path
  * the only nondeterminism is the LLM itself; --no-llm forces the template
    fallback for a fully reproducible run.

Usage:
  python3 fuzz_loop.py --seed seeds/dot_q16.mind --iters 6
  python3 fuzz_loop.py --seed seeds/scalar_arith.mind --iters 8 --no-llm
  python3 fuzz_loop.py --inject-fault   # SANITY: prove the oracles catch a bug
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import mutate
import oracles

HERE = Path(__file__).resolve().parent
VIOL_DIR = HERE / "violations"


def apply_oracles(src: Path, work: Path) -> list[oracles.OracleVerdict]:
    """Run all LOCAL oracles that do not need a successful runnable .so first.

    Order matters: compile-success gates the rest (a program that does not
    compile yields a compile verdict only -- it is the loop's stop condition,
    not necessarily a bug, per the paper).
    """
    verdicts: list[oracles.OracleVerdict] = []

    out_so = work / "out.so"
    ok, detail = oracles.compile_so(src, out_so)
    verdicts.append(oracles.OracleVerdict("compile", ok, detail or "compiled"))
    if not ok:
        return verdicts  # stop condition; caller decides bug-vs-expected

    verdicts.append(oracles.oracle_determinism(src, work))
    verdicts.append(oracles.oracle_mic3_roundtrip(src, work))
    verdicts.append(oracles.oracle_verify(src, work))
    verdicts.append(oracles.oracle_reference(out_so))
    verdicts.append(oracles.cross_substrate_hook(src, out_so))
    return verdicts


def first_violation(
    verdicts: list[oracles.OracleVerdict], tag: str
) -> oracles.OracleVerdict | None:
    """Return the first failing oracle that is a genuine violation.

    For a [semantics] mutation the reference oracle is advisory (the computed
    value is allowed to change), so a reference failure is NOT a violation there.
    The determinism / verify / mic3 / compile oracles never depend on the
    computed value, so they are violations regardless of tag.
    """
    for v in verdicts:
        if v.ok:
            continue
        if v.name == "reference" and tag == "semantics":
            continue
        if v.name == "compile":
            # handled by the loop as a stop condition, not auto-flagged here
            continue
        return v
    return None


def minimize(src_text: str, src_path: Path, work: Path, failing: str, tag: str) -> str:
    """Simple line-deletion minimization (ddmin-lite): try removing each line; keep
    the deletion if the SAME oracle still fails. Returns the reduced program."""
    lines = src_text.splitlines(keepends=True)
    i = 0
    cur = lines[:]
    while i < len(cur):
        trial = cur[:i] + cur[i + 1 :]
        trial_text = "".join(trial)
        if not trial_text.strip():
            i += 1
            continue
        tp = work / "min_trial.mind"
        tp.write_text(trial_text)
        try:
            verdicts = apply_oracles(tp, work)
        except Exception:
            i += 1
            continue
        v = first_violation(verdicts, tag)
        # keep deletion only if the SAME failing oracle still fires
        if v is not None and v.name == failing:
            cur = trial
        else:
            i += 1
    return "".join(cur)


def save_violation(
    counter: int, seed_name: str, code: str, v: oracles.OracleVerdict, work: Path, tag: str
) -> Path:
    VIOL_DIR.mkdir(exist_ok=True)
    stem = f"{seed_name}_step{counter:03d}_{v.name}"
    prog_path = VIOL_DIR / f"{stem}.mind"
    prog_path.write_text(code)
    # minimized variant
    minimized = minimize(code, prog_path, work, v.name, tag)
    (VIOL_DIR / f"{stem}.min.mind").write_text(minimized)
    (VIOL_DIR / f"{stem}.report.txt").write_text(
        f"MIND-Fuzz violation\n"
        f"seed:        {seed_name}\n"
        f"step:        {counter}\n"
        f"mutation:    {tag}\n"
        f"oracle:      {v.name}\n"
        f"detail:      {v.detail}\n"
        f"program:     {prog_path.name}\n"
        f"minimized:   {stem}.min.mind ({len(minimized.splitlines())} lines, "
        f"from {len(code.splitlines())})\n"
    )
    return prog_path


def stage_candidate(
    stage_dir: Path, seed_name: str, counter: int, code: str, work: Path
) -> str | None:
    """Stage a survivor as a candidate cross-substrate workload.

    Writes `<seed>_step<NNN>.mind` (the survivor program) and appends a line to
    `manifest.tsv`: `<id>\t<entry>\t<arg_seed>\t<arg_count>\t<avx2_hash>`. The
    avx2_hash is the host's canonical-driver output hash — the reference the
    neon CI runner asserts equality against via tests/mindfuzz_cross_substrate.rs.
    Returns the staged id, or None if the survivor exposes no canonical entry
    (still fuzz-covered locally, just not a cross-substrate fixture).
    """
    # Compile fresh into a UNIQUE staging .so per survivor. The path must be
    # unique because ctypes.CDLL caches a library handle by path within a
    # process; reusing one path would make the canonical driver read a stale
    # survivor and stage the wrong hash.
    cand_id = f"{seed_name}_step{counter:03d}"
    out_so = work / f"stage_{cand_id}.so"
    src = work / f"stage_{cand_id}.mind"
    src.write_text(code)
    ok, detail = oracles.compile_so(src, out_so)
    if not ok:
        return None
    digest = oracles.canonical_output_hash(out_so, oracles.CANON_ENTRY)
    if digest is None:
        return None
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{cand_id}.mind").write_text(code)
    line = (
        f"{cand_id}\t{oracles.CANON_ENTRY}\t{oracles.CANON_ARG_SEED}\t"
        f"{oracles.CANON_ARG_COUNT}\t{digest}\n"
    )
    manifest = stage_dir / "manifest.tsv"
    with manifest.open("a") as fh:
        fh.write(line)
    return cand_id


def run(
    seed: Path,
    iters: int,
    use_llm: bool,
    mut_path: Path,
    stage_dir: Path | None = None,
) -> int:
    instructions = mutate.load_instructions(mut_path)
    seed_name = seed.stem
    code = seed.read_text()
    work = Path(tempfile.mkdtemp(prefix="mindfuzz_"))

    print(f"[mindfuzz] seed={seed.name} iters={iters} llm={'on' if use_llm else 'off'}")
    print(f"[mindfuzz] {len(instructions)} mutation instructions, workdir={work}")
    if stage_dir is not None:
        print(f"[mindfuzz] staging survivors -> {stage_dir}")

    # step 0: the seed must itself pass (sanity that the baseline is good).
    sp = work / "step000.mind"
    sp.write_text(code)
    verdicts = apply_oracles(sp, work)
    print(f"  step 0 (seed): " + ", ".join(f"{v.name}={'ok' if v.ok else 'FAIL'}" for v in verdicts))
    if stage_dir is not None and first_violation(verdicts, "preserve") is None:
        cid = stage_candidate(stage_dir, seed_name, 0, code, work)
        if cid:
            print(f"  step 0 staged as candidate '{cid}'")

    for counter in range(1, iters + 1):
        res = mutate.mutate(code, counter - 1, instructions, use_llm=use_llm)
        sp = work / f"step{counter:03d}.mind"
        sp.write_text(res.code)

        verdicts = apply_oracles(sp, work)
        compile_v = next(v for v in verdicts if v.name == "compile")

        summary = ", ".join(
            f"{v.name}={'ok' if v.ok else 'FAIL'}" for v in verdicts
        )
        print(
            f"  step {counter} [{res.engine}] '{res.instruction[:42]}' ({res.tag}): {summary}"
        )

        viol = first_violation(verdicts, res.tag)
        if viol is not None:
            path = save_violation(counter, seed_name, res.code, viol, work, res.tag)
            print(f"  !! VIOLATION ({viol.name}): {viol.detail}")
            print(f"  !! saved + minimized -> {path}")
            return 3  # violation found

        if not compile_v.ok:
            # paper stop condition: mutation no longer compiles. This is EXPECTED
            # (the LLM accreted something invalid), not a bug -- restart/stop.
            print(f"  -- stop: mutated program no longer compiles ({compile_v.detail[:80]})")
            return 0

        # accept the mutation and continue accreting complexity
        code = res.code

        # survivor: stage it as a cross-substrate candidate (avx2 host hash).
        if stage_dir is not None:
            cid = stage_candidate(stage_dir, seed_name, counter, code, work)
            if cid:
                print(f"  step {counter} staged as candidate '{cid}'")

    print(f"[mindfuzz] budget reached ({iters} iters), no violation. clean.")
    return 0


def inject_fault() -> int:
    """SANITY: hand a deliberately-broken program to the oracles and prove one of
    them catches it (the paper's 'does the harness actually detect bugs' check).

    We forge a non-deterministic-LOOKING corruption that the COMPILE oracle and
    VERIFY oracle catch: a program with a malformed body the compiler rejects,
    plus a separately-demonstrated determinism injection.
    """
    work = Path(tempfile.mkdtemp(prefix="mindfuzz_fault_"))
    print("[mindfuzz] SANITY fault-injection: oracles must CATCH an injected bug\n")

    caught = 0

    # Fault A: a hand-corrupted mutation (a dangling `+` -> parse error). A program
    # that SHOULD compile but is corrupted -> the compile oracle must flag it. This
    # is exactly the paper's "LLM corrupts the program" case the harness must catch.
    bad = work / "fault_compile.mind"
    bad.write_text(
        "pub fn f(a: i64) -> i64 {\n    let x: i64 = 5;\n    return a + ;\n}\n"
    )
    ok, detail = oracles.compile_so(bad, work / "fa.so")
    print(f"  Fault A (corrupted body, parse error): compile oracle ok={ok}")
    if not ok:
        last = detail.strip().splitlines()[0] if detail.strip() else "compile failed"
        print(f"    CAUGHT -> {last}")
        caught += 1
    else:
        print("    MISSED -- oracle failed to catch the fault!")

    # Fault B: corrupt an evidence artifact and prove the VERIFY oracle (trace_hash)
    # rejects it -- the consumer-side wedge check actually bites.
    good = work / "fault_verify.mind"
    good.write_text("pub fn f(a: i64) -> i64 {\n    let x: i64 = 5;\n    return a + x;\n}\n")
    ev = work / "good.ev"
    p = oracles._run([str(good), "--emit-evidence", str(ev)])
    if p.returncode == 0 and ev.exists():
        raw = bytearray(ev.read_bytes())
        # flip a byte in the body region (after the header) to break trace_hash
        if len(raw) > 40:
            raw[20] ^= 0xFF
        tampered = work / "tampered.ev"
        tampered.write_bytes(bytes(raw))
        pv = oracles._run(["verify", "--json", str(tampered)])
        rejected = pv.returncode != 0 or '"trace_hash_valid":true' not in pv.stdout.replace(" ", "")
        print(f"  Fault B (tampered evidence): verify rejected={rejected} (rc={pv.returncode})")
        if rejected:
            print("    CAUGHT -> mindc verify detected the tamper")
            caught += 1
        else:
            print("    MISSED -- verify accepted a tampered artifact!")
    else:
        print("  Fault B: could not emit evidence to tamper (skipped)")

    print(f"\n[mindfuzz] sanity: {caught}/2 injected faults caught.")
    return 0 if caught >= 1 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="MIND-Fuzz: LLM-mutation differential testing for mindc")
    ap.add_argument("--seed", type=Path, default=HERE / "seeds" / "dot_q16.mind")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--no-llm", dest="use_llm", action="store_false",
                    help="force the deterministic template mutator (fully reproducible)")
    ap.add_argument("--mutations", type=Path, default=HERE / "mutations.txt")
    ap.add_argument("--inject-fault", action="store_true",
                    help="sanity: prove the oracles catch a deliberately-broken program")
    ap.add_argument("--emit-candidates", type=Path, default=None,
                    help="stage every survivor (program + avx2 output hash) into DIR as a "
                         "candidate cross-substrate workload for the CI avx2==neon gate")
    args = ap.parse_args()

    if not oracles.MINDC.exists():
        print(f"ERROR: mindc not found at {oracles.MINDC}.\n"
              f"Build it: cargo build --release --no-default-features "
              f"--features mlir-build,std-surface,cross-module-imports --bin mindc",
              file=sys.stderr)
        return 2

    if args.inject_fault:
        return inject_fault()

    return run(args.seed, args.iters, args.use_llm, args.mutations, args.emit_candidates)


if __name__ == "__main__":
    raise SystemExit(main())
