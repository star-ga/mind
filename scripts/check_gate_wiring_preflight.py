#!/usr/bin/env python3
"""Preflight cargo-test parity checks extracted from check_gate_wiring.

The wiring lint calls `check_preflight_feature_parity`. This module holds
that check and the shell-command readers it needs. It is harness, counted
by the harness ratchet, and it is not a gate of its own.

scripts/test_exec_semantics_gate.py calls `check_red_shown` and the red-branch
readers at the end of this file.

scripts/preflight.sh printed the first 12 matching lines of a red gate output: the
"did NOT run" lists filled them and cut the FAIL[pkg] line that named the cause.
Its red_lines helper then tested for a cause with `printf | grep -q` under
`set -o pipefail`: grep -q exits at the first match, printf gets SIGPIPE on an
output of about 128 KB or more, and the helper printed "no FAIL line" and hid
every FAIL line.  These checks read preflight.sh as bash and run its own
red_lines helper under its own shell options, on a red that is over 256 KB.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

from workflow_scan import (  # noqa: E402
    cargo_test_execution_argv,
    cargo_test_matrix,
    workflow_jobs,
)

WORKFLOW_DIR = ".github/workflows"


def shell_logical_lines(text: str) -> list[str]:
    """Join backslash continuations so one shell command is one string.

    preflight writes its keystone invocation across three physical lines.  A
    lint that reads physical lines sees ``--features`` and ``--test`` on
    different lines and cannot tell which command they belong to, so it falls
    back to searching the whole file -- which comments and unrelated commands
    can satisfy.
    """
    joined: list[str] = []
    buf = ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1].rstrip() + " "
            continue
        buf += stripped
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    return joined


def preflight_keystone_commands(text: str) -> list[str]:
    """Every EXECUTABLE keystone cargo-test command in preflight.

    Comment lines are dropped: a gate named only in prose is not a gate.  This
    returns commands rather than literals so a caller compares what preflight
    actually runs, not the first ``--features`` string that appears anywhere in
    the file.
    """
    out: list[str] = []
    for line in shell_logical_lines(text):
        body = line.strip()
        if body.startswith("#"):
            continue
        if "cargo test" in body and "phase_g_keystone_bootstrap" in body:
            out.append(body)
    return out


def command_features(command: str) -> set[str]:
    """The feature set one shell command declares, empty when it declares none.

    Fails closed on a shape it does not understand: an unquoted or variable
    ``--features`` yields the empty set, which then mismatches CI and reports,
    rather than silently passing an unparsed command.
    """
    for pattern in (r'--features\s+"([^"]*)"', r"--features\s+'([^']*)'",
                    r"--features\s+([A-Za-z0-9_,-]+)"):
        found = re.search(pattern, command)
        if found:
            return set(found.group(1).replace(",", " ").split())
    return set()


def preflight_test_exclusions(text: str) -> list[str] | None:
    """The declared shrink-only exclusion set, derived and never hardcoded.

    ``None`` means preflight declares no exclusion array at all.  An empty list
    means it declares one and it is empty, which is the only direction this
    contract allows the set to move without a new decision.
    """
    found = re.search(r"PREFLIGHT_TEST_EXCLUSIONS=\(([^)]*)\)", text)
    if not found:
        return None
    return [word.strip().strip('"').strip("'")
            for word in found.group(1).split() if word.strip()]


def check_preflight_feature_parity(root: Path, failures: list[str]) -> None:
    """Require preflight's cargo-test loop to mirror CI's build_test matrix."""
    ci_path = root / WORKFLOW_DIR / "ci.yml"
    preflight_path = root / "scripts/preflight.sh"
    if not ci_path.is_file() or not preflight_path.is_file():
        failures.append("WSP-06(a): ci.yml or scripts/preflight.sh is missing")
        return
    jobs = workflow_jobs(ci_path)
    text = preflight_path.read_text(encoding="utf-8")
    rows = cargo_test_matrix(jobs.get("build_test", ""))
    if len(rows) < 10:
        failures.append(
            f"WSP-06(a): ci.yml build_test exposes only {len(rows)} cargo-test "
            "invocations; expected at least 10 (matrix derivation is vacuous)"
        )
    ci_features = {row["features"] for row in rows}
    # The keystone is a separate CI job and must have the exact feature set
    # preflight runs. Compare parsed invocations, both directions, rather than
    # searching prose (comments and unrelated test commands are not evidence).
    keystone_rows = []
    for body in jobs.values():
        keystone_rows.extend(
            row for row in cargo_test_matrix(body)
            if "phase_g_keystone_bootstrap" in row.get("selector", "")
        )
    expected_keystone = {"mlir-build", "std-surface", "cross-module-imports"}
    if len(keystone_rows) != 1:
        failures.append(
            f"WSP-06(a): expected exactly one parsed keystone invocation, found {len(keystone_rows)}"
        )
    else:
        actual_keystone = set(keystone_rows[0]["features"].replace(",", " ").split())
        if actual_keystone != expected_keystone:
            failures.append(
                "WSP-06(a): keystone CI feature set differs from required exact set: "
                f"{sorted(actual_keystone)} vs {sorted(expected_keystone)}"
            )
        # Compare the ACTUAL preflight keystone commands against CI, in both
        # directions. Scanning for the first --features literal that mentions
        # mlir-build accepts a comment, a different gate, or a stale line.
        for command in preflight_keystone_commands(text):
            preflight_keystone = command_features(command)
            if preflight_keystone != actual_keystone:
                failures.append(
                    "WSP-06(a): preflight keystone feature set does not equal CI keystone: "
                    f"{sorted(preflight_keystone)} vs {sorted(actual_keystone)}"
                )
                break
    required = {"", "std-surface", "cross-module-imports",
                "std-surface,cross-module-imports", "autodiff",
                "mlir-lowering", "cpu-buffers", "ffi-c"}
    missing_ci = sorted(required - ci_features)
    if missing_ci:
        failures.append(
            "WSP-06(a): ci.yml build_test is missing expected cargo-test "
            f"feature rows: {missing_ci}"
        )
    text = preflight_path.read_text(encoding="utf-8")
    if "scripts/workflow_scan.py run_cargo_test_matrix" not in text:
        failures.append(
            "WSP-06(a): preflight has no workflow_scan run_cargo_test_matrix call; "
            "CI feature rows have no derived preflight leg"
        )
    if "matrix_rc=$?" not in text or "workflow cargo-test matrix execution FAILED" not in text:
        failures.append(
            "WSP-06(a): preflight does not preserve the typed matrix executor's failure status"
        )
    if any(token in text for token in ("base64 -d", "mapfile -t row_fields", "eval ")):
        failures.append("WSP-06(a): preflight reparses typed matrix argv through a shell decoder")
    for row in rows:
        original = list(row.get("argv", []))
        if not original:
            failures.append("WSP-06(a): scanner accepted a cargo-test row with empty argv")
            continue
        unlocked = cargo_test_execution_argv(row, False)
        locked = cargo_test_execution_argv(row, True)
        for label, argv in (("unlocked", unlocked), ("locked", locked)):
            separator = argv.index("--") if "--" in argv else len(argv)
            cargo_args = argv[:separator]
            if "--no-fail-fast" not in cargo_args:
                failures.append(f"WSP-06(a): {label} replay lacks Cargo --no-fail-fast before --")
            if ("--locked" in cargo_args) != (label == "locked"):
                failures.append(f"WSP-06(a): {label} replay violates Cargo.lock branch parity")
        separator = original.index("--") if "--" in original else len(original)
        policy = ([] if "--no-fail-fast" in original[:separator] else ["--no-fail-fast"]) + ["--locked"]
        if locked != original[:separator] + policy + original[separator:]:
            failures.append("WSP-06(a): typed replay does not preserve the scanner's exact argv")
    if not rows or any(row.get("lock_modes") != ["locked"] for row in rows):
        failures.append("WSP-06(a): a CI build_test cargo-test row runs without --locked")
    # Assert what the keystone commands DO, not the exact byte string they were
    # once written as. The previous form required a `BUILD_INCOMPLETE=1` env
    # assignment that no longer exists; a contract test that pins a removed
    # implementation detail forces bad code back in to placate itself.
    keystone_cmds = preflight_keystone_commands(text)
    if len(keystone_cmds) != 2:
        failures.append(
            "WSP-06(a): expected exactly two executable keystone commands in preflight "
            f"(one --no-run build phase, one run phase), found {len(keystone_cmds)}"
        )
    if any("--no-default-features" not in cmd for cmd in keystone_cmds):
        failures.append(
            "WSP-06(a): keystone preflight invocation lacks --no-default-features"
        )
    scanner_text = (root / "scripts/workflow_scan.py").read_text(encoding="utf-8")
    if 'command = ["cargo", "test", *argv]' not in scanner_text or "subprocess.run(" not in scanner_text:
        failures.append(
            "WSP-06(a): typed matrix executor does not execute cargo test (fake success is not evidence)"
        )
    if "passed < 1" not in scanner_text or "failed != 0" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor accepts a cargo test row without positive result evidence"
        )
    if "cargo_test_execution_argv(row, locked)" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor bypasses the typed argv replay path"
        )
    if "timeout=timeout" not in scanner_text or "except subprocess.TimeoutExpired" not in scanner_text:
        failures.append(
            "WSP-06(a): matrix executor does not preserve per-row CI timeout protection"
        )
    # The split build phase is a STRUCTURE (one --no-run command, one run
    # command) plus a label the build-failure branch emits. It is not an
    # environment variable, and requiring the variable let a build interruption
    # be reported as a determinism regression.
    build_phase = [cmd for cmd in keystone_cmds if "--no-run" in cmd]
    run_phase = [cmd for cmd in keystone_cmds if "--no-run" not in cmd]
    if len(build_phase) != 1 or len(run_phase) != 1:
        failures.append(
            "WSP-06(a): keystone lacks the required split build phase "
            f"(--no-run commands={len(build_phase)}, run commands={len(run_phase)})"
        )
    if "BUILD_INCOMPLETE" not in text:
        failures.append(
            "WSP-06(a): keystone build failure emits no BUILD_INCOMPLETE label; an "
            "interrupted build must not be reported as a determinism result"
        )
    if "byte-identity regression" in text:
        failures.append(
            "WSP-06(a): preflight still claims a byte-identity regression on a path "
            "that may never have evaluated determinism; label the cause it proved"
        )
    if "ks_build_rc" not in text or "ks_rc" not in text:
        failures.append(
            "WSP-06(a): keystone build/run results are not independently fail-closed"
        )
    # Derive the exemption set from preflight instead of naming it here. The old
    # form hardcoded 'mindfuzz_cross_substrate' and so demanded that CI keep a
    # target the exemption existed to skip -- the lint outlived the thing it
    # guarded and then failed on its own removal.
    ci_text = ci_path.read_text(encoding="utf-8")
    declared = preflight_test_exclusions(text)
    if declared is None:
        failures.append(
            "WSP-06(a): preflight declares no PREFLIGHT_TEST_EXCLUSIONS array; the "
            "shrink-only exemption set must be explicit even when it is empty"
        )
    else:
        for excluded in declared:
            if not re.search(
                rf"cargo\s+test[^#]*--test\s+{re.escape(excluded)}(?:\s|$)", ci_text
            ):
                failures.append(
                    f"WSP-06(a): exclusion '{excluded}' has no matching executable CI "
                    "cargo-test target; delete this stale exemption"
                )
    # An exemption applied outside the declared array is an undeclared one.
    for stray in re.findall(r"for\s+excluded\s+in\s+([A-Za-z0-9_][^;\n]*)", text):
        if '"${PREFLIGHT_TEST_EXCLUSIONS[@]}"' not in stray:
            failures.append(
                "WSP-06(a): preflight filters failures through an exemption name that "
                f"is not in the declared array: {stray.strip()!r}"
            )


# --- WSP-PIPE: a pipe never feeds a reader that can end before its writer ------
# A program that writes twice into a pipe whose reader (head, grep -q, ...) ends
# after the first line dies of SIGPIPE, and `pipefail` then fails the step. The
# compiler exits 101 on a closed pipe, so `mindc --version | head -n 1` went red in
# a release step. Bash `printf` of a captured value is no exception: it makes one
# write(2) per LINE, also into a pipe. Safe forms: a here-string, `[[ ]]`, a
# parameter expansion, or a reader that reads to EOF (`sed -n 1,5p`, `grep >/dev/null`).
_SED_QUIT = re.compile(r"(?:^|[;{}\s/\d$])q\d*(?=$|[;}\s])")
_PREFIX = ("sudo", "nice", "command", "exec", "!", "env", "stdbuf", "timeout")
_OPT_ARG = ("-n", "-u", "-g", "-k", "-s", "-C", "-i", "-o", "-e")  # a prefix command option that takes a word


def _piped_commands(line: str) -> list[list[str]]:
    """Words of every command that follows a single `|`, quote-aware."""
    out: list[list[str]] = []
    words: list[str] = []
    word = quote = ""
    after, i = False, 0

    def cut(pipe: bool) -> None:
        nonlocal words, word, after
        if word:
            words.append(word)
        if after and words:
            out.append(words)
        words, word, after = [], "", pipe

    while i < len(line):
        c = line[i]
        if quote:
            word += c
            quote = "" if c == quote else quote
        elif c in "'\"":
            quote, word = c, word + c
        elif c == "\\":
            word, i = word + line[i:i + 2], i + 1
        elif c == "#" and not word:
            break
        elif c == "|":
            double, both = line[i + 1:i + 2] == "|", line[i + 1:i + 2] == "&"  # `|&` pipes stderr too
            cut(not double)
            i += double or both
        elif c in "({" and after and not word and not words:
            pass  # `| (head -n 1)` and `| { read -r l; }` stay one pipeline stage
        elif c in ";&)}(`" or line.startswith("$(", i):
            cut(False)
            i += c == "$"
        elif c.isspace():
            if word:
                words.append(word)
            word = ""
        else:
            word += c
        i += 1
    cut(False)
    return out


def _ends_early(words: list[str]) -> bool:
    """Can this command stop reading stdin before the writer has finished?"""
    while words:
        if words[0] in _PREFIX or re.match(r"\w+=", words[0]):
            timeout, words = words[0] == "timeout", words[1:]
            while words and words[0].startswith("-"):
                words = words[2 if words[0] in _OPT_ARG and len(words) > 1 else 1:]
            words = words[timeout:]  # timeout's duration
        else:
            break
    if not words:
        return False
    cmd, args = words[0].rsplit("/", 1)[-1], [a for a in words[1:]]
    opts = [a for a in args if a.startswith("-")]
    if cmd in ("grep", "egrep", "fgrep", "rg"):
        return any(re.fullmatch(r"-[A-Za-z]*[qlm][A-Za-z0-9]*", a) or a in ("--quiet", "--silent", "--files-with-matches")
                   or a.startswith("--max-count") for a in opts)
    if cmd == "sed":
        return any(_SED_QUIT.search(a.strip("'\"")) for a in args if a not in opts)
    if cmd in ("awk", "gawk", "mawk"):
        return any(re.search(r"\bexit\b", a) for a in args)
    return cmd in ("head", "read", "cmp")


def early_reader_pipes(text: str) -> list[tuple[int, str]]:
    """(line, command) of every pipe into an early-ending reader in shell text.

    A pipeline whose status is discarded (`|| true`) cannot fail its step."""
    found, buf, start = [], "", 0
    for n, raw in enumerate(text.split("\n"), 1):
        if raw.lstrip().startswith("#"):
            continue
        start = start if buf else n
        buf += " " + re.sub(r"\\$", "", raw.strip())
        if re.search(r"(\\|\||&&)\s*$", raw):
            continue
        if not re.search(r"\|\|\s*(true|:)\s*(\)|$|;)", buf) and any(map(_ends_early, _piped_commands(buf))):
            found.append((start, buf.strip()[:160]))
        buf = ""
    return found


def run_blocks(text: str) -> list[tuple[int, str, str]]:
    """(line, body, shell) of every `run:` step; shell comes from the step, else its job."""
    lines, blocks = text.split("\n"), []
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)(- )?run:\s*(.*)$", ln)
        if not m:
            continue
        kc = len(m.group(1)) + 2 * bool(m.group(2))
        val = re.sub(r"\s+#.*$", "", m.group(3)).strip()
        j, body = i + 1, val
        if val[:1] in ("|", ">"):
            while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > kc):
                j += 1
            body = "\n".join(lines[i + 1:j])
        s = i
        while s > 0 and not re.match(rf"^\s{{{kc - 2}}}- ", lines[s]):
            s -= 1
        e = j
        while e < len(lines) and not re.match(rf"^\s{{0,{kc - 2}}}(- |\S)", lines[e]):
            e += 1
        shell = ""
        for sib in lines[s:e]:
            ms = re.match(rf"^(?:\s{{{kc}}}|\s{{{kc - 2}}}-\s+)shell:\s*(.*?)\s*(#.*)?$", sib)
            shell = ms.group(1) if ms else shell
        if not shell:  # the job's defaults.run.shell
            js = i
            while js > 0 and not re.match(r"^  [A-Za-z_][\w-]*:\s*$", lines[js]):
                js -= 1
            for sib in lines[js:i]:
                if re.match(r"^\s*steps:\s*$", sib):
                    break
                ms = re.match(r"^\s+shell:\s*(.*?)\s*(#.*)?$", sib)
                shell = ms.group(1) if ms else shell
        blocks.append((i + 1, body, shell))
    return blocks


def pipefail_on(body: str, shell: str) -> bool:
    """A default step is `bash -e {0}` and ignores a writer's SIGPIPE; `shell: bash` has pipefail."""
    if "${{" in shell or (shell and "bash" not in shell and "pipefail" not in shell):
        return False  # pwsh, or a shell chosen by a matrix: not bash text
    return shell == "bash" or any(re.search(r"\bpipefail\b", ln) for ln in (shell + "\n" + body).split("\n")
                                  if not ln.lstrip().startswith("#"))


def pipe_sites(text: str) -> list[tuple[int, str]]:
    """(file line, command) of every early-reader pipe in a step that runs under pipefail."""
    return [(ln + n, cmd) for ln, body, shell in run_blocks(text) if pipefail_on(body, shell)
            for n, cmd in early_reader_pipes(body)]


def script_sites(text: str) -> list[tuple[int, str]]:
    """The same for a whole shell script: pipefail is on when the script sets it."""
    return early_reader_pipes(text) if pipefail_on(text, "") else []


def hidden_run_files(tracked: dict[str, str]) -> list[str]:
    """Tracked yml files that hold run steps but sit outside the scanned workflow and action paths."""
    return sorted(f for f, tx in tracked.items() if not _PIPE_SCOPE.fullmatch(f) and _RUN_KEY.search(tx))


def run_step_gaps(texts: dict[str, str]) -> list[str]:
    """Files where run_blocks reads another number of steps than the run keys the file holds."""
    return [f"{f}: {len(run_blocks(tx))} steps read, {len(_RUN_KEY.findall(tx))} run keys"
            for f, tx in sorted(texts.items()) if len(run_blocks(tx)) != len(_RUN_KEY.findall(tx))]


def scripts_named_by(text: str) -> list[str]:
    """Gate scripts a workflow's run steps name; a comment names nothing."""
    code = "\n".join(ln for _, body, _ in run_blocks(text) for ln in body.split("\n") if not ln.lstrip().startswith("#"))
    return sorted(set(re.findall(r"(?:scripts|examples/mindc_mind)/[\w./-]+\.sh", code)))


_PIPE_BAD = (
    '"$dir/mindc" --version | head -n 1 | tr -d "\\r"',
    "first=$(printf '%s\\n' \"$v\" | head -n 1 | tr -d '\\r')",
    "printf '%s\\n' \"$out\" | grep -q 'SHA-256 verified' || exit 1",
    "apt-cache policy \"$1\" | awk '/Candidate:/ {print $2; exit}'",
    "cmd | sed -n '/x/{p;q}'",
    "cmd | tee o | grep -m1 y",
    "cmd \\\n  | grep -Eq 'a|b'",
    "printf x | sed 1q",
    "cmd | timeout 5 head -n 1",
    "cmd | sudo env X=1 grep -q y",
    "cmd |& head -n 1",
    "cmd | nice -n 5 head -n 1",
    "cmd | stdbuf -oL head",
    "cmd | timeout -k 1 5 head -n 1",
    "cmd | { read -r first; }",
    "cmd | (head -n 1)",
)
_PIPE_GOOD = (
    'v=$("$dir/mindc" --version)',
    "first=$(head -n 1 <<<\"$v\" | tr -d '\\r')",
    "grep -q 'SHA-256 verified' <<<\"$out\" || exit 1",
    "sed -n '/^iters=\\([0-9]*\\)$/{s//\\1/p;q;}' f",
    "first=${v%%$'\\n'*}",
    "cmd | grep -E 'a|b' >/dev/null",
    "sed -n 's/^q//p' f | sort",
    "cmd | sed -n '1,5p'",
    "head -n 1 f | tr a b",
    "x=$(grep a f | head -c 3 || true)",
    "cmd | while read -r l; do echo \"$l\"; done",
    "echo 'a | head' # | head",
    "cmd |& sort",
    "cmd | nice -n 5 sort",
    "cmd | { cat; }",
    "cmd | timeout -k 1 5 sort",
)
_RUN_KEY = re.compile(r"^\s*(?:- )?run:", re.M)
_PIPE_SCOPE = re.compile(r"\.github/(?:workflows/[^/]+|actions/[^/]+/[^/]+)\.ya?ml")
_SCOPE_ROWS = (  # (tracked files under .github, files that hide run steps outside the scanned paths)
    ({".github/other/x.yml": "  - run: a\n"}, [".github/other/x.yml"]),
    ({".github/workflows/a.yml": "run: a\n", ".github/actions/b/action.yml": "- run: b\n"}, []),
    ({".github/dependabot.yml": "version: 2\n"}, []),
)
_GAP_ROW = "steps:\n  - run: echo a\n  - name: b\n    run: |\n      echo b\n  - run: >\n      echo c\n"
_STEP = "steps:\n  - name: x\n    {sh}run: |\n      {extra}clang --version | head -1\n"


def check_ci_pipe_readers(root: Path, failures: list[str]) -> None:
    """WSP-PIPE: no step that runs under pipefail pipes into a reader that can end early.

    Reads its scope out of the tree: every tracked workflow and action, and every
    gate script their run steps name. Carries its own positive controls (the
    detector fires on each bad shape, stays quiet on each safe one) so that a
    detector which stopped firing is a failure here, not a green."""
    before = len(failures)
    for files, want in _SCOPE_ROWS:
        if hidden_run_files(files) != want:
            failures.append(f"WSP-PIPE: hidden_run_files read {hidden_run_files(files)} for {sorted(files)}, want {want}")
    for s in _PIPE_BAD:
        if not early_reader_pipes(s):
            failures.append(f"WSP-PIPE: the detector no longer flags {s.splitlines()[0]!r}")
    for s in _PIPE_GOOD:
        if early_reader_pipes(s):
            failures.append(f"WSP-PIPE: the detector flags the safe form {s!r}")
    for sh, extra, want in (("", "", 0), ("shell: bash\n    ", "", 1), ("", "set -o pipefail\n      ", 1),
                            ("shell: bash -e {0}\n    ", "", 0), ("shell: pwsh\n    ", "", 0)):
        got = len(pipe_sites(_STEP.format(sh=sh, extra=extra)))
        if got != want:
            failures.append(f"WSP-PIPE: {got} sites read for shell={sh.strip()!r} extra={extra.strip()!r}, want {want}")
    for text, want in (("set -uo pipefail\nclang --version | head -1\n", 1), ("clang --version | head -1\n", 0)):
        if len(script_sites(text)) != want:
            failures.append(f"WSP-PIPE: script_sites read {len(script_sites(text))} for {text!r}, want {want}")
    if scripts_named_by(_STEP.format(sh="", extra="bash scripts/a.sh\n      # scripts/b.sh\n      ")) != ["scripts/a.sh"]:
        failures.append("WSP-PIPE: the gate scripts named by a run step are not read from code lines only")
    if gap := run_step_gaps({"x.yml": _GAP_ROW}):  # one-line, block and folded steps: the reader must see all three
        failures.append(f"WSP-PIPE: run_blocks reads a planted file wrong: {gap}")
    tracked = subprocess.run(["git", "ls-files", "--", ".github"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.split()
    listed = [f for f in tracked if _PIPE_SCOPE.fullmatch(f)]
    texts = {f: (root / f).read_text(encoding="utf-8", errors="replace") for f in tracked if f.endswith((".yml", ".yaml"))}
    for f in hidden_run_files(texts):  # the scope is read out of the tree, not copied: no run step may hide outside it
        failures.append(f"WSP-PIPE: {f} holds run steps but is outside the scanned workflow and action paths")
    for gap in run_step_gaps({f: texts[f] for f in listed}):  # the steps read must equal the run keys in the scanned files
        failures.append(f"WSP-PIPE: the pipe scan skips run steps: {gap}")
    steps = sum(len(run_blocks(texts[f])) for f in listed)
    if not steps:
        failures.append("WSP-PIPE: no workflow run step was read; the scan would assert nothing")
    scripts = sorted(s for s in {s for f in listed for s in scripts_named_by(texts[f])} if (root / s).is_file())
    sites = [f"{f}:{n}: {cmd}" for f in listed for n, cmd in pipe_sites(texts[f])]
    for f in scripts:
        text = (root / f).read_text(encoding="utf-8", errors="replace")
        sites += [f"{f}:{n}: {cmd}" for n, cmd in script_sites(text)]
    for site in sites:
        failures.append(f"WSP-PIPE: a pipe into a reader that can end before its writer, under pipefail: {site}")
    if len(failures) == before:
        print(f"[PASS] WSP-PIPE: {steps} run steps and {len(scripts)} gate scripts hold no pipe into an early-ending reader")


# --- a red section of preflight.sh shows its cause ---------------------------
PREFLIGHT = Path(__file__).resolve().parent / "preflight.sh"
_CUT_RE = re.compile(r"(^|[|;&]\s*)(head|tail)\b")
_BRANCH_END_RE = re.compile(r'\s*(fi|else|elif|done)\b|\s*(bad|step) "')
# This file's own copy of a FAIL verdict line: a test that read it from
# preflight.sh would lose a form together with the helper.
CAUSE_RE = re.compile(r"^(FAIL|\[FAIL\]|VACUOUS|run_gate: FAIL)")
BIG_RED = 256 * 1024


def _code(line: str) -> str:
    """`line` with its bash comment removed, honouring double quotes.

    A `#` only opens a comment at the start of a word and never inside `"..."`, so
    this is bash's own rule rather than `split("#", 1)`.
    """
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def red_branch_cuts(text: str) -> tuple[int, list[str]]:
    """(red branches, lines of them that cut output with head or tail outside red_lines)."""
    lines = text.splitlines()
    seen, cuts, i = 0, [], 0
    while i < len(lines):
        code = _code(lines[i])
        if 'bad "' not in code:
            i += 1
            continue
        seen += 1
        j = i
        while j < len(lines):
            code = _code(lines[j])
            if j > i and _BRANCH_END_RE.match(code):
                break
            rest = code.split('bad "', 1)[1].split('"', 1)[-1] if j == i else code
            if "red_lines" not in rest and "=$(" not in rest and _CUT_RE.search(rest.strip()):
                cuts.append(f"{j + 1}: {lines[j].strip()}")
            j += 1
        i = j
    return seen, cuts


def silent_red_outputs(text: str) -> list[str]:
    """Each captured gate output (`<name>_out=$(...)`) that no red_lines call prints."""
    names = set(re.findall(r"\b(\w+_out)=\$\(", text))
    return sorted(n for n in names if f'red_lines "${n}"' not in text)


def _red_call(text: str, var: str) -> str:
    calls = [ln.strip() for ln in text.splitlines()
             if ln.strip().startswith(f'red_lines "${var}"')]
    if len(calls) != 1:
        raise AssertionError(f"expected one red_lines call on ${var}, found {len(calls)}")
    return calls[0]


def _red_lines_shown(text: str, var: str, output: str) -> tuple[int, list[str]]:
    """Run preflight's own red_lines helper and its own call on `$var`, under its own
    `set` options, with the output on stdin (one argv string is capped at 128 KB)."""
    m = re.search(r"^RED_FAIL_RE=.*?^red_lines\(\) \{.*?^\}$", text, re.M | re.S)
    if not m:
        raise AssertionError("scripts/preflight.sh defines no red_lines helper")
    opts = re.findall(r"^set -.*$", text, re.M)
    if not opts:
        raise AssertionError("scripts/preflight.sh sets no shell options")
    script = "\n".join((*opts, m.group(0), f"{var}=$(cat)", _red_call(text, var))) + "\n"
    proc = subprocess.run(["bash", "-c", script, "red"], input=output, capture_output=True, text=True)
    return proc.returncode, proc.stdout.splitlines()


def check_red_shown(text: str, var: str, out: str, old: str) -> None:
    """`out`, a red of `$var` over BIG_RED bytes whose old print (`old`, first 12) hid a
    FAIL line: preflight prints every FAIL line, and nothing but cause lines before them."""
    cause = [ln for ln in out.splitlines() if CAUSE_RE.match(ln)]
    old_shown = set([ln for ln in out.splitlines() if re.match(old, ln)][:12])
    if len(out.encode()) <= BIG_RED or not cause or not set(cause) - old_shown:
        raise AssertionError(f"{var}: the recorded red is not over {BIG_RED} bytes with a FAIL "
                             "line hidden behind 12 list lines")
    rc, shown = _red_lines_shown(text, var, out)
    shown_set, cause_set = set(shown), set(cause)
    missing = [ln for ln in cause if ln not in shown_set]
    if missing:
        raise AssertionError(f"{var}: preflight cuts {len(missing)} of {len(cause)} cause "
                             f"line(s), first shown {shown[:1]}: {missing[:3]}")
    args = shlex.split(_red_call(text, var))
    extra = re.compile(args[2]) if len(args) > 2 and args[2] else None
    lead = shown[:max(i for i, ln in enumerate(shown) if ln in cause_set) + 1]
    early = [ln for ln in lead if not (CAUSE_RE.match(ln) or (extra and extra.search(ln)))]
    if early:
        raise AssertionError(f"{var}: preflight prints other lines before its FAIL lines: {early[:3]}")
    if rc != 0:
        raise AssertionError(f"{var}: preflight's red_lines exits {rc} under its own shell options")
