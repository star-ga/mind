#!/usr/bin/env python3
"""Gate test: the assertion count must be EVIDENCE, never a printed integer.

Background — the defect this test exists to prevent
---------------------------------------------------
`scripts/gate_assert.py` publishes the one number `scripts/run_gate.py` trusts:
`asserted=<N>`, refused when it is 0. That number was derivable from a line the
gate simply PRINTED — any `ran=<n>` or `scored=<n>` anywhere in the output
replaced every other piece of evidence. So a gate could report a healthy count
while checking nothing:

  * a smoke printing `(ran={len(CASES)})` reports the LENGTH OF A LIST, not the
    number of cases it executed — emptying its case loop left the count at 8;
  * a gate writing `asserted=99` to stderr outranked the shim's own line,
    because the runner concatenated stdout+stderr and took the last match;
  * a helper line that merely CONTAINED `ran=0` zeroed three real assertions.

The contract enforced here:
  1. a count marker is read only in the two shapes the repo already publishes
     (`SDLC-GATE <name> ran=<n> fail=<k>`, `scored=<n> divergences=<k>`);
  2. a marker never manufactures evidence — with zero evaluated asserts and
     zero verdict lines the count is 0, whatever the marker says;
  3. a gate that writes the contract line itself fails closed;
  4. gate sources may not print those shapes by hand (the lint in
     examples/mindc_mind/smoke_wiring_lint.py).

Run:  python3 tests/gate_assert_count_contract_test.py
Exit: 0 = every case passed, 1 = a case failed (prints the offending case).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHIM = REPO / "scripts" / "gate_assert.py"
RUNNER = REPO / "scripts" / "run_gate.py"

sys.path.insert(0, str(REPO / "examples" / "mindc_mind"))
import smoke_wiring_lint  # noqa: E402

FAILURES: list[str] = []


def _run(src: str, *, args: list[str] | None = None) -> tuple[int, str, str]:
    """Run a synthetic gate through the shim.

    Returns (rc, stdout, stdout+stderr). The two streams are kept apart on
    purpose: the contract line lives on STDOUT and that is the only stream
    scripts/run_gate.py reads — folding stderr in is exactly the mistake that
    let a gate's own `asserted=99` on stderr outrank the shim's verdict.
    """
    with tempfile.TemporaryDirectory() as td:
        gate = Path(td) / "synthetic_gate.py"
        gate.write_text(src, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SHIM), str(gate), *(args or [])],
            capture_output=True, text=True, cwd=str(REPO), timeout=120,
        )
    return proc.returncode, proc.stdout, proc.stdout + proc.stderr


def _asserted(out: str) -> int | None:
    """The count the shim published on stdout (its line is the last one)."""
    n: int | None = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("asserted="):
            try:
                n = int(s.split("=", 1)[1])
            except ValueError:
                return None
    return n


def check(label: str, got: object, want: object) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(label)


# ── the shim's counting contract ───────────────────────────────────────────

def case_bare_marker_is_not_a_count() -> None:
    """A hand-printed `ran=5` with nothing behind it counts ZERO."""
    _, out, _ = _run('print("r" + "an=5")\n')
    check("bare marker, no evidence", _asserted(out), 0)


def case_bare_marker_never_overrides_evidence() -> None:
    """Two real verdict lines + a bare marker: the evidence wins, not the 5."""
    src = 'print("[PASS] leg one")\nprint("[PASS] leg two")\nprint("r" + "an=5")\n'
    _, out, _ = _run(src)
    check("bare marker beside 2 verdicts", _asserted(out), 2)


def case_helper_marker_cannot_zero_real_asserts() -> None:
    """A helper line containing `ran=0` must not discard evaluated asserts."""
    src = ('assert 1 == 1\nassert 2 == 2\nassert 3 == 3\n'
           'print("helper: r" + "an=0")\n')
    _, out, _ = _run(src)
    check("`ran=0` helper line beside 3 asserts", _asserted(out), 3)


def case_sdlc_marker_needs_evidence() -> None:
    """The sanctioned SDLC shape with no evidence behind it fails closed."""
    src = 'print("SDLC-G" + "ATE synthetic ran=7 fail=0")\n'
    _, out, allout = _run(src)
    check("SDLC marker alone", _asserted(out), 0)
    check("SDLC marker alone is diagnosed",
          "count marker without evidence" in allout, True)


def case_sdlc_marker_with_evidence_is_authoritative() -> None:
    src = ('print("[PASS] synthetic leg")\n'
           'print("SDLC-G" + "ATE synthetic ran=7 fail=0")\n')
    _, out, _ = _run(src)
    check("SDLC marker + evidence", _asserted(out), 7)


def case_tcdiff_marker_with_evidence_is_authoritative() -> None:
    src = ('print("RESULT: PASS — synthetic sweep")\n'
           'print("tcdiff synthetic sweep: sco" + "red=9 divergences=0")\n')
    _, out, _ = _run(src)
    check("tcdiff marker + evidence", _asserted(out), 9)


def case_tcdiff_zero_scored_is_zero() -> None:
    """`scored=0` is the sweep reporting it scored nothing — count 0."""
    src = ('print("RESULT: PASS — synthetic sweep")\n'
           'print("tcdiff synthetic sweep: sco" + "red=0 divergences=0")\n')
    _, out, _ = _run(src)
    check("tcdiff marker scored=0", _asserted(out), 0)


def case_forged_line_on_stderr_fails_closed() -> None:
    """A gate writing the contract line itself must not be believed."""
    src = ('import sys\nprint("[PASS] a real leg")\n'
           'print("asser" + "ted=99", file=sys.stderr)\n')
    _, out, allout = _run(src)
    check("forged count on stderr", _asserted(out), 0)
    check("forged count is diagnosed", "published its own" in allout, True)


def case_forged_line_on_stdout_fails_closed() -> None:
    src = ('print("[PASS] a real leg")\nprint("asser" + "ted=99")\n'
           'print("[PASS] another leg")\n')
    _, out, _ = _run(src)
    check("forged count on stdout", _asserted(out), 0)


def case_string_system_exit_preserves_its_diagnostic() -> None:
    """Python prints a string-valued SystemExit before exiting non-zero.

    Gate helpers use this form for fail-closed setup errors.  The shim must not
    turn the named error into a bare ``asserted=0`` failure.
    """
    rc, out, allout = _run(
        'raise SystemExit("FAIL[synthetic]: required artifact is stale")\n'
    )
    check("string SystemExit remains non-zero", rc, 1)
    check(
        "string SystemExit diagnostic is visible",
        "FAIL[synthetic]: required artifact is stale" in allout,
        True,
    )
    check("string SystemExit verdict is counted", _asserted(out), 1)


def case_runner_ignores_a_forged_count() -> None:
    """End to end: run_gate.py must reject the forged gate, not report 99."""
    with tempfile.TemporaryDirectory() as td:
        gate = Path(td) / "forged_gate.py"
        gate.write_text('import sys\nprint("asser" + "ted=99", file=sys.stderr)\n',
                        encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(RUNNER), str(gate)],
            capture_output=True, text=True, cwd=str(REPO), timeout=120,
        )
    check("run_gate rejects a forged count (rc)", proc.returncode, 1)
    check("run_gate did not read 99",
          "asserted=99 (<" not in proc.stdout + proc.stderr, True)


# ── the runner's announced-skip rule ───────────────────────────────────────
# `SKIP` was matched only as `^\s*SKIP\b`. That is NARROWER than the ci.yml
# tee-loop backstop it was meant to supersede (`^[[:space:]]*SKIP`, which does
# match `SKIPPED`), and it cannot see a leg a multi-leg gate drops INSIDE a
# wider line. Measured on this tree with the oracle `.so` hidden:
# self_host_loop_smoke.py printed `  NOTE  [ORACLE] ... — SKIPPED (...)`,
# dropped its entire Rust-oracle leg and graded `run_gate: PASS asserted=1`.
# The synthetic gate sources below spell the token by concatenation so THIS
# gate's own output never carries one.

def _runner(src: str, *args: str) -> tuple[int, str]:
    """Run a synthetic gate through run_gate.py; return (rc, output)."""
    with tempfile.TemporaryDirectory() as td:
        gate = Path(td) / "skip_gate.py"
        gate.write_text(src, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(RUNNER), *args, str(gate)],
            capture_output=True, text=True, cwd=str(REPO), timeout=120,
        )
    return proc.returncode, proc.stdout + proc.stderr


def case_runner_rejects_an_anchored_skip_line() -> None:
    """The original rule, unchanged: `SKIP  <thing> not built` is red."""
    rc, _ = _runner('print("[PASS] one leg")\nprint("SK" + "IP  mindc not built")\n')
    check("runner rejects an anchored skip line (rc)", rc, 1)


def case_runner_rejects_an_indented_skipped_line() -> None:
    """`  SKIPPED: ...` — the exact spelling `SKIP\b` could not match."""
    rc, _ = _runner('print("[PASS] one leg")\nprint("  SKIP" + "PED: leg x")\n')
    check("runner rejects an indented skipped line (rc)", rc, 1)


def case_runner_rejects_a_midline_skipped_report() -> None:
    """A leg dropped INSIDE a wider line is still a leg that did not run."""
    src = ('print("[PASS] primary leg")\n'
           'print("  NOTE  [ORACLE] drift .so not present — SKIP" + "PED (x).")\n')
    rc, out = _runner(src)
    check("runner rejects a mid-line dropped leg (rc)", rc, 1)
    check("runner names the announced skip", "unasserted invariant" in out, True)


def case_runner_keeps_a_per_case_skip_classification() -> None:
    """Deliberate limit: mindfuzz_self_host.py prints one
    `[  17] SKIP  rust rejected` line per generated CASE. That is a
    classification inside a gate that IS asserting, not a dropped gate leg —
    broadening to a mid-line `SKIP` would turn a working fuzzer red."""
    src = ('print("[PASS] real leg")\n'
           'print("  [  17] SK" + "IP  rust rejected")\n')
    rc, _ = _runner(src)
    check("runner keeps a per-case skip classification (rc)", rc, 0)


# ── the same contract for a NON-Python gate ────────────────────────────────
# A `.py` gate is executed THROUGH gate_assert.py, so its count is the shim's
# verdict and a gate that prints the contract line itself is caught as forgery.
# A `.sh` gate has no shim: the runner counts its verdict lines directly. That
# asymmetry was a hole — the runner read a self-published `asserted=<N>` line
# from ANY gate and believed it, so
#   #!/bin/bash
#   echo "asserted=42"
# graded `run_gate: PASS ... asserted=42` having checked nothing. Two `.sh`
# gates are routed today (scripts/check_no_ai_attribution.sh,
# scripts/check_json_not_evidence.sh); both report their verdict on STDOUT, so
# the count is derived from stdout verdict lines only and a printed contract
# line on EITHER stream is forgery — the same rule the shim applies, not a
# second, weaker one.

def _runner_shell(src: str, *args: str) -> tuple[int, str]:
    """Run a synthetic `.sh` gate through run_gate.py; return (rc, output)."""
    with tempfile.TemporaryDirectory() as td:
        gate = Path(td) / "synthetic_gate.sh"
        gate.write_text(src, encoding="utf-8")
        gate.chmod(0o755)
        proc = subprocess.run(
            [sys.executable, str(RUNNER), *args, str(gate)],
            capture_output=True, text=True, cwd=str(REPO), timeout=120,
        )
    return proc.returncode, proc.stdout + proc.stderr


def case_runner_rejects_a_forged_shell_count() -> None:
    """A `.sh` gate publishing the contract line must not be believed."""
    rc, out = _runner_shell('#!/bin/bash\necho "asser""ted=42"\nexit 0\n')
    check("runner rejects a forged shell count (rc)", rc, 1)
    check("runner names the shell forgery", "published its own" in out, True)
    # (the runner ECHOES gate output, so the forged line is present in `out`;
    # what must be absent is the runner BELIEVING it)
    check("runner did not grade the forged gate green",
          "run_gate: FAIL" in out, True)


def case_runner_rejects_a_forged_shell_count_on_stderr() -> None:
    """Same rule on the other stream: the shim fails closed on a forged line
    written to stderr, and the non-Python path may not be weaker."""
    src = ('#!/bin/bash\necho "[PASS] a real leg"\n'
           'echo "asser""ted=42" >&2\n')
    rc, out = _runner_shell(src)
    check("runner rejects a forged shell count on stderr (rc)", rc, 1)
    check("runner names the stderr forgery", "published its own" in out, True)


def case_runner_counts_shell_verdict_lines() -> None:
    """The honest shape: N reported verdicts on stdout count as N."""
    src = ('#!/bin/bash\necho "[PASS] leg one"\necho "[PASS] leg two"\n'
           'echo "[PASS] leg three"\n')
    rc, out = _runner_shell(src)
    check("runner accepts a shell gate with 3 verdicts (rc)", rc, 0)
    check("runner counts the shell verdicts", "asserted=3" in out, True)


def case_runner_ignores_shell_verdicts_on_stderr() -> None:
    """The count is read from the stdout stream only. Folding stderr in is what
    let a gate's own line land last in the concatenation and win."""
    rc, out = _runner_shell('#!/bin/bash\necho "[PASS] wrong stream" >&2\n')
    check("runner rejects a shell gate that reported nothing on stdout (rc)",
          rc, 1)
    check("runner names the empty count", "checked nothing" in out, True)


def case_runner_still_reads_a_skip_from_either_shell_stream() -> None:
    """Unchanged: the announced-skip rule spans stdout AND stderr."""
    src = ('#!/bin/bash\necho "[PASS] a real leg"\n'
           'echo "SK""IP  mindc not built" >&2\n')
    rc, out = _runner_shell(src)
    check("runner rejects a shell skip on stderr (rc)", rc, 1)
    check("runner names the announced skip", "unasserted invariant" in out, True)


def case_runner_enforces_the_min_asserted_floor() -> None:
    """The floor a two-leg gate is wired with must actually bite."""
    rc, out = _runner('print("[PASS] only one leg")\n', "--min-asserted", "2")
    check("runner enforces --min-asserted 2 (rc)", rc, 1)
    check("runner names the shortfall", "asserted=1 (< 2)" in out, True)


def case_loop_gate_is_wired_with_both_legs_required() -> None:
    """The self-host LOOP gate has TWO legs (PRIMARY reproduction + the Rust
    drift ORACLE) and only the pair is the gate. Every runner invocation of it
    must therefore demand both, or the wedge's loop gate grades green with the
    oracle leg silently dropped. The scope is READ OUT of the wiring files
    rather than hand-copied, so a new call site cannot quietly omit the floor."""
    checked = 0
    for rel in (".github/workflows/ci.yml", "scripts/preflight.sh"):
        text = (REPO / rel).read_text(encoding="utf-8")
        # Join shell line continuations so `ENV=... \<nl> python3 run_gate.py ...`
        # is one command, the way the shell sees it.
        joined = text.replace("\\\n", " ")
        for line in joined.splitlines():
            if "run_gate.py" not in line or "self_host_loop_smoke" not in line:
                continue
            checked += 1
            check(f"{rel}: loop gate demands both legs",
                  "--min-asserted 2" in line, True)
    check("loop-gate call sites found", checked >= 2, True)


# ── the lint that keeps the shapes out of gate sources ─────────────────────

def _lint(src: str) -> list[str]:
    return smoke_wiring_lint.marker_violations(Path("synthetic_gate.py"), src)


def case_lint_flags_a_hand_printed_ran() -> None:
    check("lint: print(f\"(ran={n})\")",
          bool(_lint('n = 5\nprint(f"passed  (ran={n})")\n')), True)


def case_lint_flags_a_forged_asserted() -> None:
    check("lint: print(\"asserted=3\")",
          bool(_lint('print("asserted=3")\n')), True)


def case_lint_allows_the_two_families() -> None:
    sdlc = 'print(f"SDLC-GATE synthetic ran={t} fail={len(p)}")\n'
    tcd = ('print(f"tcdiff sweep: generated={g} scored={s} '
           'divergences={len(d)}")\n')
    check("lint: SDLC-GATE shape allowed", _lint(sdlc), [])
    check("lint: scored/divergences shape allowed", _lint(tcd), [])


def case_lint_allows_prose_mentioning_a_marker() -> None:
    """A docstring or comment naming a marker is not a gate printing one."""
    src = ('"""ran=<n>: the repo marker, described in prose."""\n'
           '# scored=5 divergences=0 in a comment\n'
           'print("[PASS] a real leg")\n')
    check("lint: prose mention allowed", _lint(src), [])


def case_lint_scans_the_real_corpus() -> None:
    """The corpus itself must be clean — this is the wave-1 regression."""
    bad = smoke_wiring_lint.scan_marker_violations()
    check("lint: corpus clean", bad, [])


# ── the per-case verdict contract ──────────────────────────────────────────
# The count rule above made a MARKER unable to stand in for evidence. It left
# the other half open: a single UNCONDITIONAL summary line is itself a verdict,
# so a gate that printed only `ALL PASS — {n}/{n} byte-identical` reported
# `asserted=1` whether it compared eight cases or none. Measured on this tree:
# self_host_array_smoke.py with `CASES = []` printed
# `ALL PASS — 0/0 byte-identical (0 diff)` and graded `run_gate: PASS
# asserted=1`, byte-identical to the unmutated control — so the count carried no
# information about the work done, and the mandated mutation could not go red.
# The structural rule: a gate must emit evidence PER CHECKED CASE on a green
# run — a verdict print whose LITERAL carries a PASS token inside a
# For/While/AsyncFor body, an `assert` in such a loop, a `gate_assert.check()`
# in such a loop, or a `--min-asserted N >= 2` floor at its runner call sites.
# `If` was deliberately REMOVED from the accepted bodies: `if fails:
# print("FAIL")` is one line whatever the corpus held, and a failure-path print
# inside a loop emits nothing at all on a green run. Measured before that was
# tightened: self_host_tc_unknown_ident_smoke compared 419 cases, printed one
# `ALL PASS`, graded `asserted=1`, and satisfied the old rule purely through two
# `print("FAIL: ... drifted")` lines in a table-check loop a green run never
# reaches.

def _shape(src: str) -> list[str]:
    return smoke_wiring_lint.verdict_shape_violations(Path("synthetic_gate.py"), src)


def case_lint_flags_a_summary_only_gate() -> None:
    """An empty case loop plus a trailing summary must be reported."""
    src = ('CASES = []\n'
           'for c in CASES:\n'
           '    print(f"  [OK] {c}")\n'
           'print(f"ALL PASS — {len(CASES)}/{len(CASES)} byte-identical")\n')
    check("lint: summary-only gate flagged", bool(_shape(src)), True)


def case_lint_allows_a_per_case_verdict_gate() -> None:
    """The same gate reporting one verdict per case is the fixed shape."""
    src = ('CASES = []\n'
           'for c in CASES:\n'
           '    print(f"  [PASS] {c}")\n'
           'print(f"{len(CASES)}/{len(CASES)} byte-identical")\n')
    check("lint: per-case verdict gate allowed", _shape(src), [])


def case_lint_flags_an_if_only_verdict_gate() -> None:
    """An `if`-guarded recap is unconditional in effect: one line per RUN.

    This is the shape the old rule accepted. `fails` is a whole-corpus
    aggregate, so the branch fires at most once however many cases ran and the
    count cannot tell a full corpus from an empty one.
    """
    src = ('CASES = []\n'
           'fails = 0\n'
           'for c in CASES:\n'
           '    fails += c\n'
           'if fails:\n'
           '    print("FAIL: corpus diverged")\n'
           'print("ALL PASS")\n')
    check("lint: if-only verdict gate flagged", bool(_shape(src)), True)


def case_lint_flags_a_failure_only_loop_verdict() -> None:
    """A verdict printed only on the FAILURE path is not green-run evidence.

    Being inside a `for` body is not enough: this line emits nothing when the
    gate passes, so the count still comes from the blanket recap alone.
    """
    src = ('CASES = []\n'
           'for c in CASES:\n'
           '    if not c:\n'
           '        print("FAIL: case diverged")\n'
           'print("ALL PASS")\n')
    check("lint: failure-only loop verdict flagged", bool(_shape(src)), True)


def case_lint_allows_a_check_call_per_case() -> None:
    """`gate_assert.check()` composes the token, so its call site is the proof.

    It prints exactly one `[PASS]`/`[FAIL]` line per call, which the shim
    counts; an emptied loop calls it zero times and the count is zero. This is
    the sanctioned upgrade path out of VERDICT_SHAPE_RESIDUAL.txt.
    """
    src = ('from gate_assert import check\n'
           'CASES = []\n'
           'for c in CASES:\n'
           '    check(c == 1, f"case {c}")\n'
           'print("all reported cases agreed")\n')
    check("lint: per-case check() gate allowed", _shape(src), [])


def case_lint_residual_is_declared_and_shrink_only() -> None:
    """The residual is a checked artifact, not a comment.

    Every listed name must be a `gate` row that STILL violates the rule, so a
    converted gate cannot linger on the list and quietly regress. The corpus
    scan below is what enforces it; this case pins that the file exists and is
    parseable, because a missing list would silently disable the declaration.
    """
    names = smoke_wiring_lint.load_residual()
    rows, _ = smoke_wiring_lint.parse_manifest()
    check("lint: residual is non-empty and declared", bool(names), True)
    check("lint: every residual entry is a gate row",
          sorted(n for n in names if rows.get(n, (None, None))[1] != "gate"), [])


def case_lint_allows_an_assert_only_gate() -> None:
    """Deliberate limit: an `assert`-based gate prints no verdict at all, and
    an evaluated assert is already counted per iteration — flagging it would
    demand output from a gate whose evidence is not output."""
    src = 'for c in []:\n    assert c\n'
    check("lint: assert-only gate allowed", _shape(src), [])


def case_lint_scans_the_real_corpus_for_verdict_shape() -> None:
    """Every declared `gate` reports per case, not one blanket summary."""
    check("lint: corpus reports per case",
          smoke_wiring_lint.scan_verdict_shape_violations(), [])


# ── gap_corpus_smoke: an exemption is a classification, not an assertion ────
#
# The AST-kind coverage lint in examples/mindc_mind/gap_corpus_smoke.py prints
# one row per declared kind. A SOURCE_PROBES kind's row is a coverage check that
# ran against the corpus; a SYNTHETIC / NO_ORACLE_CONSTRUCT kind's row records
# that nothing was byte-compared for it. The shim counts every line carrying a
# verdict token, so those exemption rows must carry none — otherwise the gate
# reports N kinds "covered" while N-4 were checked. These cases exercise the
# pure classification function with the module's REAL tables; no oracle, no
# compiler, no `.so` is involved (the gate resolves those only inside main()).

def _gap_kind_rows(fixture_texts: list[str], **tables: dict[str, str]
                   ) -> tuple[bool, int, list[str], list[str]]:
    """Run ast_kind_coverage with optional table overrides; capture its rows."""
    import contextlib
    import io

    import gap_corpus_smoke as gap

    saved = {name: getattr(gap, name) for name in tables}
    for name, value in tables.items():
        setattr(gap, name, value)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ok, ran, failures = gap.ast_kind_coverage(fixture_texts)
    finally:
        for name, value in saved.items():
            setattr(gap, name, value)
    rows = [ln for ln in buf.getvalue().splitlines() if "AST node kind" in ln]
    return ok, ran, failures, rows


def _gap_corpus_texts() -> list[str]:
    corpus = REPO / "tests" / "selfhost_gaps"
    paths = sorted(corpus.glob("*.mind")) + sorted((corpus / "never_wrong").glob("*.mind"))
    return [p.read_text(encoding="utf-8") for p in paths]


def case_gap_corpus_exempt_kinds_are_not_counted() -> None:
    """On the real corpus: one row per kind; only probed kinds carry a token."""
    import gap_corpus_smoke as gap
    from gate_assert import VERDICT_RE, is_skip_line

    ok, ran, failures, rows = _gap_kind_rows(_gap_corpus_texts())
    exempt = set(gap.SYNTHETIC) | set(gap.NO_ORACLE_CONSTRUCT)
    exempt_rows = [r for r in rows if any(f" {k}:" in r for k in exempt)]
    counted = [r for r in rows if VERDICT_RE.search(r)]
    check("gap: real corpus classification is clean", (ok, failures), (True, []))
    check("gap: one row per declared kind", len(rows), ran)
    check("gap: one row per exempt kind", len(exempt_rows), len(exempt))
    check("gap: exempt rows carry no verdict token",
          [r for r in exempt_rows if VERDICT_RE.search(r)], [])
    check("gap: exempt rows are not skip lines",
          [r for r in exempt_rows if is_skip_line(r)], [])
    check("gap: counted rows == probed kinds", len(counted), len(gap.SOURCE_PROBES))
    check("gap: every counted row is a literal PASS",
          [r for r in counted if not r.startswith("[PASS] ")], [])


def case_gap_corpus_unexercised_probe_is_a_failed_row() -> None:
    """An empty corpus reds EVERY probed kind, one [FAIL] row each; exempt rows
    stay token-free — an empty corpus must not turn an exemption into a pass."""
    import gap_corpus_smoke as gap
    from gate_assert import VERDICT_RE

    ok, ran, failures, rows = _gap_kind_rows([])
    fails = [r for r in rows if r.startswith("[FAIL] ")]
    exempt = set(gap.SYNTHETIC) | set(gap.NO_ORACLE_CONSTRUCT)
    exempt_rows = [r for r in rows if any(f" {k}:" in r for k in exempt)]
    check("gap: empty corpus is not ok", ok, False)
    check("gap: every probed kind is a FAIL row", len(fails), len(gap.SOURCE_PROBES))
    check("gap: no PASS row survives an empty corpus",
          [r for r in rows if r.startswith("[PASS] ")], [])
    check("gap: exempt rows still uncounted on an empty corpus",
          [r for r in exempt_rows if VERDICT_RE.search(r)], [])


def case_gap_corpus_stale_classification_is_a_failed_row() -> None:
    """A kind classified in the lint but no longer declared in main.mind is a
    visible [FAIL] row, not only a summary line the per-case count cannot see."""
    import gap_corpus_smoke as gap

    stale = dict(gap.SYNTHETIC)
    stale["ast_never_declared_probe"] = "planted by the contract test"
    ok, _, failures, rows = _gap_kind_rows(_gap_corpus_texts(), SYNTHETIC=stale)
    check("gap: stale classification is not ok", ok, False)
    check("gap: stale classification names itself",
          [f for f in failures if f.startswith("ast_never_declared_probe:")] != [], True)
    check("gap: stale classification is a FAIL row",
          [r for r in rows if r.startswith("[FAIL] AST node kind ast_never_declared_probe:")] != [],
          True)


def case_gap_corpus_unclassified_kind_is_a_failed_row() -> None:
    """A declared kind with no entry in any table is a [FAIL] row: a kind
    nobody classified is a construct that can hide a miscompile."""
    import gap_corpus_smoke as gap

    probes = dict(gap.SOURCE_PROBES)
    victim = "ast_while"
    check("gap: mutation target is a probed kind", victim in probes, True)
    del probes[victim]
    ok, _, _, rows = _gap_kind_rows(_gap_corpus_texts(), SOURCE_PROBES=probes)
    check("gap: unclassified kind is not ok", ok, False)
    check("gap: unclassified kind is a FAIL row naming the gap",
          [r for r in rows if r.startswith(f"[FAIL] AST node kind {victim}:")
           and "NEW AST node kind" in r] != [], True)


def main() -> int:
    for fn in sorted(
        (v for k, v in globals().items() if k.startswith("case_")),
        key=lambda f: f.__name__,
    ):
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("\nRESULT: PASS — the assertion count is evidence, not a printed integer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
