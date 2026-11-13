#!/usr/bin/env python3
"""workflow_scan.py — dependency-free structural reader for the GitHub Actions
workflow YAML in this repo.

WHY NOT PyYAML
--------------
The lints that consume this run in two places with different Pythons: the
`actions/setup-python` interpreter used by `.github/workflows/docs-claims.yml`
(a clean 3.12 with no third-party packages) and whatever `python3` a maintainer
has locally.  A lint that is the gate for a supply-chain property must not be
able to fail *open* because an import was missing, so this module parses the
subset of YAML our workflows actually use with the stdlib only:

  * block mappings with consistent 2-space indentation
  * block sequences (`- item`)
  * inline flow sequences (`[a, b]`)
  * block scalars (`|`, `>`) kept verbatim as opaque text

That is deliberately not a YAML parser.  It answers structural questions
("which jobs exist", "what does this job declare in `needs:`", "what raw text
sits inside this job") and nothing else; anything subtler belongs in a real
parser and a real dependency.
"""

from __future__ import annotations

import re
import shlex
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

_KEY_RE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_.\-]+):(?P<rest>.*)$")


def _significant(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def blocks(text: str, indent: int) -> dict[str, str]:
    """Map every `key:` at exactly `indent` spaces to the raw text beneath it.

    The value excludes the `key:` header line itself but includes any inline
    remainder on that line as the first entry, so `needs: build` and

        needs:
          - build

    both come back as text containing "build".  Order is source order.
    """
    lines = text.splitlines()
    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if not _significant(line):
            continue
        m = _KEY_RE.match(line)
        if m and len(m.group("indent")) == indent:
            starts.append((i, m.group("key"), m.group("rest").strip()))

    out: dict[str, str] = {}
    for n, (i, key, rest) in enumerate(starts):
        end = len(lines)
        for j in range(i + 1, len(lines)):
            line = lines[j]
            if not _significant(line):
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading <= indent:
                end = j
                break
        body = "\n".join(lines[i + 1 : end])
        out[key] = (rest + "\n" + body) if rest else body
    return out


def scalar_list(body: str) -> list[str]:
    """Read a `needs:`-shaped value: scalar, `[a, b]` flow list, or `- a` block."""
    body = body.strip()
    if not body:
        return []
    if body.startswith("["):
        inner = body[1 : body.index("]")] if "]" in body else body[1:]
        return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
    items = [
        ln.strip()[1:].strip().strip("'\"")
        for ln in body.splitlines()
        if ln.strip().startswith("- ")
    ]
    if items:
        return items
    first = body.splitlines()[0].strip().strip("'\"")
    return [first] if first else []


def workflow_jobs(path: Path) -> dict[str, str]:
    """job id -> raw body text, for one workflow file."""
    text = path.read_text(encoding="utf-8")
    top = blocks(text, 0)
    if "jobs" not in top:
        return {}
    return blocks(top["jobs"], 2)


def job_display_name(job_body: str) -> str | None:
    """The literal `name:` a job declares, or None when it relies on the id."""
    fields = blocks(job_body, 4)
    name = fields.get("name")
    if name is None:
        return None
    return name.splitlines()[0].strip().strip("'\"")


def name_prefix(display_name: str) -> str:
    """The stable prefix of a job's check-run name.

    GitHub renders a matrix job as `<name> (<matrix values>)`, and a `name:`
    that interpolates `${{ matrix.* }}` renders with those values substituted.
    Neither expansion is knowable from the file, so the checked contract is the
    part before the first interpolation — everything to its left is literal and
    stable across matrix edits.
    """
    cut = display_name.find("${{")
    return display_name if cut < 0 else display_name[:cut]


_RUN_BLOCK_RE = re.compile(r"^(?P<indent> +)run:\s*[|>][0-9+\-]*\s*$")
_ASSIGNED_COMMAND_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\$\((?P<body>.*)\)$")
_CARGO_TEST_RE = re.compile(
    r"^(?:timeout\s+(?P<timeout>[1-9][0-9]*)\s+)?cargo\s+test(?:\s+(?P<args>.*))?$"
)


def _shell_run_blocks(job_body: str) -> list[str]:
    """Return dedented shell bodies from YAML ``run: |``/``run: >`` fields."""
    lines = job_body.splitlines()
    scripts: list[str] = []
    i = 0
    while i < len(lines):
        match = _RUN_BLOCK_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        header_indent = len(match.group("indent"))
        start = i + 1
        end = start
        while end < len(lines):
            line = lines[end]
            if line.strip() and len(line) - len(line.lstrip(" ")) <= header_indent:
                break
            end += 1
        body = lines[start:end]
        indents = [len(line) - len(line.lstrip(" ")) for line in body if line.strip()]
        if indents:
            block_indent = min(indents)
            scripts.append("\n".join(line[block_indent:] if line.strip() else "" for line in body))
        i = end
    return scripts


def _logical_shell_lines(script: str) -> list[str]:
    """Join only shell backslash-newline continuations, preserving other spaces."""
    lines = script.splitlines()
    logical: list[str] = []
    pending = ""
    for line in lines:
        pending += line
        if pending.endswith("\\"):
            pending = pending[:-1]
            continue
        logical.append(pending)
        pending = ""
    if pending:
        raise ValueError("cargo-test command ends with an unterminated continuation")
    return logical


def _cargo_test_command(line: str) -> tuple[str, str] | None:
    """Parse one supported simple shell command, or reject ambiguous execution."""
    command = line.strip()
    if not command or command.startswith("#"):
        return None
    assigned = _ASSIGNED_COMMAND_RE.fullmatch(command)
    if assigned is not None:
        command = assigned.group("body").strip()
    match = _CARGO_TEST_RE.fullmatch(command)
    if match is not None:
        return match.group("args") or "", match.group("timeout") or ""
    if "cargo test" not in command:
        return None
    # Quoted labels and diagnostics are data, not executable commands.
    try:
        words = shlex.split(command, comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"unparseable shell line mentioning cargo test: {line!r}: {exc}") from exc
    if words and words[0] in {"echo", "printf"} and "$(" not in command and "`" not in command:
        return None
    if (len(words) == 1 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
            and "$(" not in command and "`" not in command):
        return None
    raise ValueError(f"unsupported shell wrapper around cargo test: {line!r}")


def _without_cargo_option(argv: list[str], option: str) -> list[str]:
    """Remove a Cargo option before ``--`` while preserving libtest arguments."""
    separator = argv.index("--") if "--" in argv else len(argv)
    return [word for n, word in enumerate(argv) if n >= separator or word != option]


def _features_and_selectors(argv: list[str]) -> tuple[str, list[str]]:
    """Summarise feature and selector args without changing the executable argv."""
    separator = argv.index("--") if "--" in argv else len(argv)
    cargo_argv = argv[:separator]
    features = ""
    selectors: list[str] = []
    options_with_values = {
        "--bench", "--bin", "--example", "--exclude", "--features", "--jobs",
        "--manifest-path", "--package", "--target", "--target-dir", "--test", "-j", "-p",
    }
    selector_options = {"--bench", "--bin", "--example", "--test"}
    n = 0
    while n < len(cargo_argv):
        word = cargo_argv[n]
        if word == "--lib":
            selectors.append(word)
        elif word.startswith("--features="):
            features = word.split("=", 1)[1]
        elif any(word.startswith(prefix + "=") for prefix in selector_options):
            selectors.append(word)
        elif word in options_with_values:
            if n + 1 >= len(cargo_argv):
                raise ValueError(f"cargo test option {word!r} has no value")
            value = cargo_argv[n + 1]
            if word == "--features":
                features = value
            if word in selector_options:
                selectors.extend((word, value))
            n += 1
        elif not word.startswith("-"):
            selectors.append(word)
        n += 1
    if separator < len(argv):
        selectors.extend(argv[separator:])
    return features, selectors


def cargo_test_matrix(job_body: str) -> list[dict[str, object]]:
    """Extract every cargo-test invocation from one job body.

    The result keeps the invocation text and its feature expression.  This is
    deliberately scoped to a job body: other workflow jobs may run different
    tests and must not silently expand the preflight contract.
    """
    scripts = _shell_run_blocks(job_body) or [job_body]
    rows: list[dict[str, object]] = []
    for line in (line for script in scripts for line in _logical_shell_lines(script)):
        parsed = _cargo_test_command(line)
        if parsed is None:
            continue
        args, timeout = parsed
        try:
            words = shlex.split(args, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"unparseable cargo test argv: {line!r}: {exc}") from exc
        features, selector = _features_and_selectors(words)
        locked = "--locked" in words[:words.index("--") if "--" in words else len(words)]
        executable_argv = _without_cargo_option(words, "--locked")
        rows.append({"command": line.strip(), "features": features,
                     "selector": " ".join(selector), "selector_argv": selector,
                     "argv": executable_argv, "timeout": timeout,
                     "lock_modes": ["locked" if locked else "unlocked"]})
    # Collapse only the locked/unlocked forms of the exact same Cargo argv.
    # Release mode, default-feature state, filters, selectors, and future flags
    # are all part of the key and therefore remain distinct CI invocations.
    unique: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row["argv"])
        prior = unique.get(key)
        if prior is None:
            unique[key] = row
            continue
        prior["lock_modes"] = sorted(set(prior["lock_modes"]) | set(row["lock_modes"]))
        if int(row.get("timeout") or 0) > int(prior.get("timeout") or 0):
            prior["timeout"] = row["timeout"]
            prior["command"] = row["command"]
    return list(unique.values())


def cargo_test_execution_argv(row: dict[str, object], locked: bool) -> list[str]:
    """Build the exact Cargo argv, inserting local policy before libtest's ``--``."""
    raw = row.get("argv")
    if not isinstance(raw, list) or not raw or not all(isinstance(word, str) for word in raw):
        raise ValueError("cargo test row has no valid parsed argv")
    argv = _without_cargo_option(list(raw), "--locked")
    separator = argv.index("--") if "--" in argv else len(argv)
    cargo_options = argv[:separator]
    inserted: list[str] = []
    if "--no-fail-fast" not in cargo_options:
        inserted.append("--no-fail-fast")
    if locked and "--locked" not in cargo_options:
        inserted.append("--locked")
    return argv[:separator] + inserted + argv[separator:]


_TEST_RESULT_RE = re.compile(
    r"^test result: (?:ok|FAILED)\. ([0-9]+) passed; ([0-9]+) failed;"
)


def run_cargo_test_matrix(root: Path) -> int:
    """Execute the derived matrix without a shell or an argv serialization layer."""
    jobs = workflow_jobs(root / ".github/workflows/ci.yml")
    rows = cargo_test_matrix(jobs.get("build_test", ""))
    if len(rows) < 10:
        raise ValueError(f"build_test cargo matrix has {len(rows)} semantic rows; need >=10")
    locked = (root / "Cargo.lock").is_file()
    failures = 0
    for index, row in enumerate(rows, 1):
        argv = cargo_test_execution_argv(row, locked)
        timeout_value = row.get("timeout")
        if not isinstance(timeout_value, str) or (timeout_value and not timeout_value.isdigit()):
            raise ValueError(f"cargo test row {index} has invalid timeout {timeout_value!r}")
        timeout = int(timeout_value) if timeout_value else None
        command = ["cargo", "test", *argv]
        label = row.get("features") or "<none>"
        try:
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            print(f"FAIL feature row {label!r} timed out after {timeout}s; argv={json.dumps(argv)}")
            if exc.stdout:
                print(str(exc.stdout)[-2000:])
            failures += 1
            continue
        except OSError as exc:
            print(f"FAIL feature row {label!r} could not execute cargo: {exc}")
            failures += 1
            continue
        output = result.stdout + result.stderr
        counts = [_TEST_RESULT_RE.match(line) for line in output.splitlines()]
        matched = [match for match in counts if match is not None]
        passed = sum(int(match.group(1)) for match in matched)
        failed = sum(int(match.group(2)) for match in matched)
        if result.returncode != 0 or failed != 0 or passed < 1:
            print(
                f"FAIL feature row {label!r} (passed={passed} failed={failed} "
                f"rc={result.returncode}); argv={json.dumps(argv)}"
            )
            for output_line in output.splitlines()[-15:]:
                print(output_line)
            failures += 1
        else:
            print(
                f"ok feature row {label!r} (passed={passed} failed={failed} "
                f"rc={result.returncode}); argv={json.dumps(argv)}"
            )
    print(f"cargo test matrix: rows={len(rows)} failed_rows={failures}")
    return 1 if failures else 0


def _self_test() -> int:
    """Exercise the build-test cargo matrix contract without third-party YAML."""
    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/ci.yml").read_text()
    top = blocks(text, 0)
    jobs = blocks(top["jobs"], 2)
    rows = cargo_test_matrix(jobs["build_test"])
    if len(rows) < 10:
        raise AssertionError(f"build_test cargo matrix has {len(rows)} invocations; need >=10")
    features = {row["features"] for row in rows}
    required = {"", "std-surface", "cross-module-imports",
                "std-surface,cross-module-imports", "autodiff",
                "mlir-lowering", "cpu-buffers", "ffi-c"}
    missing = sorted(required - features)
    if missing:
        raise AssertionError(f"build_test cargo matrix missing feature rows: {missing}")
    # Cargo.lock is committed, so every CI row must resolve against it.
    unlocked = [row["command"] for row in rows if row["lock_modes"] != ["locked"]]
    if unlocked:
        raise AssertionError(f"build_test runs cargo test without --locked: {unlocked}")
    pair = cargo_test_matrix("cargo test --features x\ncargo test --locked --features x")
    if len(pair) != 1 or set(pair[0]["lock_modes"]) != {"locked", "unlocked"}:
        raise AssertionError(f"lock/no-lock forms of one argv were not normalized together: {pair}")
    # Adversarial parser contract: continuations and quoted arguments must
    # remain one invocation, while the lock/no-lock duplicate collapses only
    # after preserving its selector and strongest timeout.
    probe = "\n".join([
        "timeout 60 cargo test --verbose \\",
        '  --features "std-surface cross-module-imports" --test "fixture suite" -- "case one"',
        "timeout 90 cargo test --verbose --features 'std-surface cross-module-imports' --test 'fixture suite' -- 'case one'",
    ])
    probe_rows = cargo_test_matrix(probe)
    if len(probe_rows) != 1 or probe_rows[0]["features"] != "std-surface cross-module-imports":
        raise AssertionError(f"quoted/continued cargo command was not normalized: {probe_rows}")
    expected_argv = ["--verbose", "--features", "std-surface cross-module-imports",
                     "--test", "fixture suite", "--", "case one"]
    if probe_rows[0]["selector_argv"] != ["--test", "fixture suite", "--", "case one"]:
        raise AssertionError(f"selector was not preserved: {probe_rows}")
    if probe_rows[0]["argv"] != expected_argv or probe_rows[0]["timeout"] != "90":
        raise AssertionError(f"argv/timeout was not preserved: {probe_rows}")

    # Extraction is executable-shell scoped.  Prose and echo labels are not
    # commands, shell wrapper punctuation is not an argument, quoted whitespace
    # is data, and otherwise-distinct Cargo invocations must not be collapsed
    # merely because their feature and selector summaries happen to match.
    adversarial = "\n".join([
        'echo "cargo test --features fake"',
        'label="cargo test --features also-fake"',
        'out=$(timeout 7 cargo test --release --features real --lib "case  (one)")',
        'cargo test --features real --lib "case  (one)"',
    ])
    adversarial_rows = cargo_test_matrix(adversarial)
    if len(adversarial_rows) != 2:
        raise AssertionError(f"fake command accepted or distinct argv collapsed: {adversarial_rows}")
    if adversarial_rows[0]["argv"][-1] != "case  (one)":
        raise AssertionError(f"quoted argv or wrapper suffix was corrupted: {adversarial_rows}")
    if adversarial_rows[0]["timeout"] != "7" or "--release" not in adversarial_rows[0]["argv"]:
        raise AssertionError(f"timeout/release argv was not preserved: {adversarial_rows}")
    try:
        cargo_test_matrix('echo "$(cargo test --features hidden)"')
    except ValueError:
        pass
    else:
        raise AssertionError("executable cargo command substitution was mistaken for an echo label")
    boundary = {"argv": ["--features", "real", "--", "--locked", "--no-fail-fast"]}
    expected = ["--features", "real", "--no-fail-fast", "--locked", "--", "--locked", "--no-fail-fast"]
    if cargo_test_execution_argv(boundary, True) != expected:
        raise AssertionError("Cargo options were not inserted before the libtest separator")
    return 0


# --- Lockfile / cargo-call scanning for check_release_gating.py (rule R7) ---
ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = "Cargo.lock"
# A cargo (or cross) call: optional `+toolchain` / global flags, then the subcommand.
CARGO_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:cargo|cross)((?:\s+[+-]\S+)*)\s+([A-Za-z][A-Za-z0-9_-]*)"
    r"(?![A-Za-z0-9_-])"
)
# Subcommands that never resolve the dependency graph. Every other subcommand,
# plugins included, must be locked: a list of the RESOLVING ones missed `deny`.
NON_RESOLVING = frozenset({"fmt", "help", "version"})
LOCK_FLAG_RE = re.compile(r"(?<![A-Za-z0-9_-])--(?:locked|frozen)(?![A-Za-z0-9_-])")
LOCK_EXEMPT_RE = re.compile(r"^#\s*lockfile-exempt:\s*(\S.{19,})$")


def unlocked_cargo_commands(path: Path) -> tuple[list[str], list[str]]:
    """(unlocked resolving commands, stale exemptions) in one workflow's run blocks."""
    from gate_runner_wiring_lint import (  # local: keeps this module free of a top-level lint import
        join_continuations,
        run_block_lines,
        split_segments,
        strip_shell_comment,
    )

    unlocked: list[str] = []
    stale: list[str] = []
    exempt_at: int | None = None
    for lineno, cmd in join_continuations(run_block_lines(path.read_text(encoding="utf-8"))):
        if cmd.startswith("#"):
            if LOCK_EXEMPT_RE.match(cmd):
                exempt_at = lineno
            continue
        hits = []
        for seg in split_segments(strip_shell_comment(cmd)):
            for m in CARGO_RE.finditer(seg):
                # Only flags before a bare `--` are cargo's; after it they are libtest's.
                own = re.split(r"\s--(?:\s|$)", seg[m.start():], maxsplit=1)[0]
                if m.group(2) not in NON_RESOLVING and not LOCK_FLAG_RE.search(own):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {seg.strip()}")
                    break
        if exempt_at is not None and not hits:
            stale.append(f"{path.relative_to(ROOT)}:{exempt_at}")
        if exempt_at is None:
            unlocked.extend(hits)
        exempt_at = None
    return unlocked, stale


def lockfile_ignored(gitignore: Path) -> list[str]:
    """Non-negated .gitignore patterns that match the root Cargo.lock."""
    if not gitignore.is_file():
        return []
    hits = []
    for raw in gitignore.read_text(encoding="utf-8").splitlines():
        pat = raw.strip()
        if not pat or pat.startswith(("#", "!")):
            continue
        bare = pat.removeprefix("**/").removeprefix("/").rstrip("/")
        if fnmatch.fnmatchcase(LOCKFILE, bare):
            hits.append(pat)
    return hits


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: workflow_scan.py {--self-test|cargo_test_matrix|run_cargo_test_matrix}")
        if sys.argv[1] == "--self-test":
            raise SystemExit(_self_test())
        if sys.argv[1] == "cargo_test_matrix":
            jobs = workflow_jobs(root / ".github/workflows/ci.yml")
            for row in cargo_test_matrix(jobs.get("build_test", "")):
                print(json.dumps(row, separators=(",", ":")))
        elif sys.argv[1] == "run_cargo_test_matrix":
            raise SystemExit(run_cargo_test_matrix(root))
        else:
            raise ValueError(f"unknown workflow_scan command: {sys.argv[1]!r}")
    except (KeyError, OSError, ValueError) as exc:
        print(f"workflow_scan: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
