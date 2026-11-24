#!/usr/bin/env python3
"""Language conformance gate: the measured language status of this compiler.

The corpus in tests/lang_conformance/ holds small programs. ROWS.tsv gives each
program the verdict that the language specification (star-ga/mind-spec) demands
at three stages:

  check  `mindc check --no-fmt <file>` accepts or rejects (and, for a reject row
         that names one, the catalog error code of the first error);
  build  `mindc build <file> --backend native --out <image>` writes an image or
         refuses;
  run    the image exits with the status (and prints the stdout) the program
         defines; "-" means the program pins no value, and then the run must end
         by itself with an exit status 0..255 (no signal, no timeout).

A stage that ends by a signal, a timeout or an exit status that is not a
documented one is a crash: it differs from every expected verdict, so a crash,
a panic or a hang is never read as a rejection. The documented statuses: check
0 (accept) or 1 (compilation error); build 0 with an image (accept), 1
(compilation error) or 3 (refusal of the native backend, no image).

KNOWN_GAPS.tsv lists every row where the compiler of this tree differs from the
expected verdict, with the class of the difference and the measured first
diagnostic or refusal key. The gate is green only when the set of measured
differences equals that list exactly:
  * a new difference fails the gate;
  * a listed gap that no longer differs fails the gate (remove it in the same
    change, `--write-gaps` does it);
  * the number of listed gaps may never grow: it must not be above the count of
    any committed version of KNOWN_GAPS.tsv in the git history of HEAD (the
    gate reads that history, so an edit of the list alone can not raise it).

Classes: conformance-gap (the specification defines the program and check
rejects it, or a reject row gets another code), native-only-gap (check accepts,
the native backend refuses with a `lane.` key or its image runs wrong), loose
(the compiler accepts what the specification forbids), unkeyed-refusal (the
native backend refuses with no key).

docs/lang-conformance-status.md is GENERATED from the two data files
(`--write-status`); the gate run fails when the committed table differs from a
new generation. `--self-test` proves the three behaviours above and that a
wrong expected verdict is caught, on planted rows run through the real
compiler; it also runs this script itself as a child on planted corpora and
checks its exit status, so the exit path of the gate run is under test. The compiler is MINDC_BIN, else target/release/mindc of this
tree; there is no fallback to a `mindc` on PATH.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # the tree tracks no Python build products

from _selfhost_sources import core_limit_argv, core_limit_check, start_child  # noqa: E402

CORPUS = REPO / "tests" / "lang_conformance"
STATUS = REPO / "docs" / "lang-conformance-status.md"
CLASSES = ("conformance-gap", "native-only-gap", "loose", "unkeyed-refusal")
ROW_FIELDS = ("id", "check", "build", "exit", "stdout", "code", "spec", "construct")
GAP_FIELDS = ("id", "class", "key")
TIMEOUTS = {"check": 60, "build": 120, "run": 10}
CHECK_STATUS = {0: "accept", 1: "reject"}
BUILD_REFUSAL = (1, 3)  # compilation error, native backend refusal


class CorpusError(Exception):
    pass


def resolve_mindc() -> pathlib.Path | None:
    raw = os.environ.get("MINDC_BIN")
    path = pathlib.Path(raw) if raw else REPO / "target" / "release" / "mindc"
    return path if path.is_file() and os.access(path, os.X_OK) else None


def _tsv(path: pathlib.Path, fields: tuple[str, ...]) -> list[dict]:
    """Rows of a tab-separated file with a header line; blank lines and `#` lines are not rows."""
    if not path.is_file():
        raise CorpusError(f"{path.name}: file is missing")
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    if not body or tuple(body[0].split("\t")) != fields:
        raise CorpusError(f"{path.name}: header must be {' '.join(fields)}")
    rows = []
    for n, ln in enumerate(body[1:], 2):
        cells = ln.split("\t")
        if len(cells) != len(fields):
            raise CorpusError(f"{path.name}: data line {n} has {len(cells)} fields, not {len(fields)}")
        rows.append(dict(zip(fields, cells)))
    return rows


def load_rows(corpus: pathlib.Path) -> list[dict]:
    rows = _tsv(corpus / "ROWS.tsv", ROW_FIELDS)
    seen = set()
    for r in rows:
        rid = r["id"]
        if rid in seen:
            raise CorpusError(f"ROWS.tsv: row {rid} is given twice")
        seen.add(rid)
        if not re.fullmatch(r"[a-z0-9-]+/[A-Za-z0-9_]+", rid) or not (corpus / f"{rid}.mind").is_file():
            raise CorpusError(f"ROWS.tsv: row {rid} names no program {rid}.mind")
        if r["check"] not in ("accept", "reject") or r["build"] not in ("accept", "reject"):
            raise CorpusError(f"ROWS.tsv: row {rid}: check and build are accept or reject")
        if r["check"] == "reject" and r["build"] != "reject":
            raise CorpusError(f"ROWS.tsv: row {rid}: a reject row has build reject")
        if r["build"] == "reject" and (r["exit"] != "-" or r["stdout"] != "-"):
            raise CorpusError(f"ROWS.tsv: row {rid}: a row with build reject has no run value")
        if r["check"] == "accept" and r["code"] != "-":
            raise CorpusError(f"ROWS.tsv: row {rid}: an accept row names no error code")
        if r["exit"] != "-" and not (r["exit"].isdigit() and int(r["exit"]) <= 255):
            raise CorpusError(f"ROWS.tsv: row {rid}: exit is 0..255 or -")
        if r["stdout"] != "-" and not re.fullmatch(r"(?:[^\\]|\\[\\nt])*", r["stdout"]):  # the form of stdout_field
            raise CorpusError(f"ROWS.tsv: row {rid}: stdout is - or a text with the escapes \\\\, \\n and \\t")
        if r["code"] != "-" and not re.fullmatch(r"E\d{4}", r["code"]):
            raise CorpusError(f"ROWS.tsv: row {rid}: code is E#### or -")
    on_disk = {str(p.relative_to(corpus).with_suffix("")) for p in corpus.glob("*/*.mind")}
    if on_disk - seen:
        raise CorpusError(f"ROWS.tsv: programs with no row: {', '.join(sorted(on_disk - seen))}")
    first: dict[bytes, str] = {}  # a twin would count one gap two times
    for r in rows:
        text = (corpus / f"{r['id']}.mind").read_bytes()
        if text in first:
            raise CorpusError(f"ROWS.tsv: rows {first[text]} and {r['id']} have byte-equal programs")
        first[text] = r["id"]
    return rows


def load_gaps(corpus: pathlib.Path) -> tuple[dict[str, tuple[str, str]], int]:
    rows = _tsv(corpus / "KNOWN_GAPS.tsv", GAP_FIELDS)
    gaps = {}
    for g in rows:
        if g["id"] in gaps:
            raise CorpusError(f"KNOWN_GAPS.tsv: row {g['id']} is listed twice")
        if g["class"] not in CLASSES or not g["key"]:
            raise CorpusError(f"KNOWN_GAPS.tsv: row {g['id']}: class is one of {', '.join(CLASSES)}, key is not empty")
        gaps[g["id"]] = (g["class"], g["key"])
    return gaps, len(gaps)


def _git(repo: pathlib.Path, *args: str) -> tuple[int, str]:
    p = start_child(["git", "-C", str(repo), *args], None, capture_output=True, stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace")


def ratchet_error(repo: pathlib.Path, rel: str, count: int) -> str:
    """'' when `count` is not above the gap count of any committed version of `rel`, else why not."""
    rc, shallow = _git(repo, "rev-parse", "--is-shallow-repository")
    if rc != 0 or shallow.strip() != "false":
        return "the gap ratchet needs the full git history of this tree (no git checkout, or a shallow one)"
    rc, log = _git(repo, "log", "--format=%H", "--", rel)  # `rel` is relative to `repo`
    if rc != 0:
        return f"git log of {rel} gave exit {rc}"
    counts = []
    for rev in log.split():
        rc, text = _git(repo, "show", f"{rev}:./{rel}")
        if rc == 0:  # a commit that deleted the file has no version
            counts.append((sum(1 for ln in text.splitlines() if ln and not ln.startswith("#")
                               and ln != "\t".join(GAP_FIELDS)), rev[:12]))
    if counts and count > min(counts)[0]:
        low, rev = min(counts)
        return f"{count} gaps listed, commit {rev} listed {low}: the count may never grow"
    return ""


def _first_error(text: str) -> str:
    for ln in text.splitlines():
        if "error" in ln.lower():
            return ln.strip()
    return text.strip().splitlines()[0].strip() if text.strip() else "no message"


def check_key(text: str) -> tuple[str, str]:
    """(key, code) of the first error: the catalog code when it has one, else its tag, else its text."""
    first = _first_error(text)
    code = re.search(r"\bE\d{4}\b", first)
    if code:
        return code.group(0), code.group(0)
    tag = re.search(r"\[([A-Za-z_:-]+)\]", first)
    return (tag.group(1) if tag else re.sub(r"\s+", " ", first)[:80].rstrip()), "-"


def refusal_key(text: str) -> str:
    key = re.search(r"\((lane\.[A-Za-z0-9_.]+)\)", text)
    return key.group(1) if key else re.sub(r"\s+", " ", _first_error(text))[:80].rstrip()


def _start(argv: list[str], cwd: pathlib.Path, timeout: int) -> tuple[int | str, str, bytes]:
    env = dict(os.environ, LC_ALL="C")
    try:
        p = start_child(core_limit_argv(argv), None, cwd=cwd, capture_output=True, timeout=timeout, env=env,
                        stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "timeout", "", b""
    text = (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, text, p.stdout or b""


def _crash(stage: str, rc: int | str) -> str:
    return f"{stage}: timeout" if rc == "timeout" else f"{stage}: signal {-rc}" if rc < 0 else f"{stage}: exit {rc}"


def measure(mindc: pathlib.Path, program: pathlib.Path, work: pathlib.Path, timeouts: dict = TIMEOUTS) -> dict:
    """Run one program through the three stages in its own folder."""
    work.mkdir(parents=True)
    src = work / program.name
    shutil.copyfile(program, src)
    rc, text, _ = _start([str(mindc), "check", "--no-fmt", src.name], work, timeouts["check"])
    if rc in CHECK_STATUS:
        key, code = check_key(text) if rc else ("", "-")
        m = {"check": CHECK_STATUS[rc], "check_key": key, "code": code}
    else:
        m = {"check": "crash", "check_key": _crash("check", rc), "code": "-"}
    rc, text, _ = _start([str(mindc), "build", src.name, "--backend", "native", "--out", "image"], work,
                         timeouts["build"])
    image = work / "image"
    if rc == 0 and image.is_file():
        m["build"], m["build_key"] = "accept", ""
    elif rc in BUILD_REFUSAL and not image.exists():
        m["build"], m["build_key"] = "reject", refusal_key(text)
    else:  # a signal, a timeout, a panic, an undocumented status, or an image with a refusal status
        m["build"] = "crash"
        m["build_key"] = _crash("build", rc) + (", no image" if rc == 0 else ", image written" if image.exists() else "")
        return m
    if m["build"] == "accept":
        rc, _, out = _start([str(image)], work, timeouts["run"])
        m["exit"], m["stdout"] = rc, out.decode("utf-8", "backslashreplace")
    return m


def stdout_field(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def difference(row: dict, m: dict) -> tuple[str, str] | None:
    """(class, key) when the measured verdict differs from the expected one, else None."""
    if m["check"] == "crash":
        return "conformance-gap", m["check_key"]
    if row["check"] == "accept":
        if m["check"] == "reject":
            return "conformance-gap", m["check_key"]
        if m["build"] == "crash":
            return "native-only-gap", m["build_key"]
        if row["build"] == "reject":
            return None if m["build"] == "reject" else ("loose", "native build accepts")
        if m["build"] == "reject":
            key = m["build_key"]
            return ("native-only-gap" if key.startswith("lane.") else "unkeyed-refusal"), key
        ran = m["exit"]
        if not isinstance(ran, int) or not 0 <= ran <= 255:
            return "native-only-gap", _crash("run", ran)
        if row["exit"] != "-" and ran != int(row["exit"]):
            return "native-only-gap", f"run: exit {ran}, not {row['exit']}"
        if row["stdout"] != "-" and stdout_field(m["stdout"]) != row["stdout"]:
            return "native-only-gap", "run: stdout differs"
        return None
    if m["check"] == "accept":
        return "loose", "check accepts"
    if row["code"] != "-" and m["code"] != row["code"]:
        return "conformance-gap", f"{m['check_key']} for {row['code']}"
    if m["build"] == "crash":
        return "conformance-gap", m["build_key"]
    if m["build"] == "accept":
        return "loose", "native build accepts"
    return None


def measure_all(mindc: pathlib.Path, corpus: pathlib.Path, rows: list[dict], workers: int) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="lang-conformance-") as tmp:
        root = pathlib.Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {r["id"]: pool.submit(measure, mindc, corpus / f"{r['id']}.mind", root / f"w{n}")
                    for n, r in enumerate(rows)}
            return {rid: job.result() for rid, job in jobs.items()}


def judge(rows: list[dict], gaps: dict, measured: dict[str, dict]) -> tuple[list[tuple[bool, str]], dict]:
    """One (ok, text) per row, plus a line per listed gap that names no row; and the measured differences."""
    found = {}
    lines = []
    for r in sorted(rows, key=lambda r: r["id"]):
        got = difference(r, measured[r["id"]])
        if got:
            found[r["id"]] = got
        want = gaps.get(r["id"])
        if got == want:
            lines.append((True, f"{r['id']}: " + ("expected verdict" if got is None else f"listed {got[0]} ({got[1]})")))
        elif want is None:
            lines.append((False, f"{r['id']}: new difference {got[0]} ({got[1]}); fix it or list it"))
        elif got is None:
            lines.append((False, f"{r['id']}: listed {want[0]} no longer differs; remove it from KNOWN_GAPS.tsv"))
        else:
            lines.append((False, f"{r['id']}: listed {want[0]} ({want[1]}), measured {got[0]} ({got[1]})"))
    ids = {r["id"] for r in rows}
    lines += [(False, f"{gid}: listed gap names no row of ROWS.tsv") for gid in sorted(set(gaps) - ids)]
    return lines, found


def render_status(rows: list[dict], gaps: dict) -> str:
    fams = sorted({r["id"].split("/")[0] for r in rows})
    out = ["# Language conformance status", "",
           "Generated by `examples/mindc_mind/lang_conformance_gate.py --write-status` from",
           "`tests/lang_conformance/ROWS.tsv` and `tests/lang_conformance/KNOWN_GAPS.tsv`. Do not edit by hand:",
           "the gate fails when this file differs from a new generation. See `docs/lang-conformance.md`.", "",
           "A row is accepted end to end when it is not in the gap list: check, native build and run give the",
           "verdict that the specification demands.", "",
           "| family | rows | valid rows | reject rows | rows as expected | valid rows accepted end to end | share |",
           "|---|---|---|---|---|---|---|"]

    def line(name: str, part: list[dict]) -> str:
        valid = [r for r in part if r["check"] == "accept"]
        good = [r for r in valid if r["id"] not in gaps]
        share = f"{100 * len(good) / len(valid):.1f} %" if valid else "-"
        ok = sum(r["id"] not in gaps for r in part)
        return f"| {name} | {len(part)} | {len(valid)} | {len(part) - len(valid)} | {ok} | {len(good)} | {share} |"

    out += [line(f, [r for r in rows if r["id"].startswith(f + "/")]) for f in fams]
    out += [line("all", rows), "", "## Gaps by class", "", "| class | rows |", "|---|---|"]
    out += [f"| {c} | {sum(g[0] == c for g in gaps.values())} |" for c in CLASSES]
    by_id = {r["id"]: r for r in rows}
    for c in CLASSES:
        listed = sorted(gid for gid, g in gaps.items() if g[0] == c)
        out += ["", f"### {c}", ""]
        if not listed:
            out.append("None.")
            continue
        out += ["| row | measured key | specification reference |", "|---|---|---|"]
        for gid in listed:
            spec = by_id[gid]["spec"] if gid in by_id else "-"
            out.append(f"| `{gid}` | `{gaps[gid][1].replace('|', chr(92) + '|')}` | {spec.replace('|', chr(92) + '|')} |")
    return "\n".join(out) + "\n"


def write_gaps(path: pathlib.Path, gaps: dict) -> None:
    body = ["# Rows where the compiler of this tree differs from the expected verdict (see docs/lang-conformance.md).",
            "# The count may never grow: the gate compares it with every committed version of this file.",
            "\t".join(GAP_FIELDS)]
    body += [f"{gid}\t{c}\t{k}" for gid, (c, k) in sorted(gaps.items())]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def status_fresh(corpus: pathlib.Path, status: pathlib.Path) -> str:
    """'' when the committed table equals a new generation, else why not."""
    want = render_status(load_rows(corpus), load_gaps(corpus)[0])
    have = status.read_text(encoding="utf-8") if status.is_file() else None
    return "" if have == want else f"{status.name} is stale: run --write-status and commit it"


def _image_runner(script: bytes, timeout: int):
    """The runner core_limit_check proves the core limit on: a script image run as a row image is run."""
    with tempfile.TemporaryDirectory(prefix="lang-conformance-core-") as tmp:
        image = pathlib.Path(tmp) / "image"
        image.write_bytes(script)
        image.chmod(0o755)
        rc, _, out = _start([str(image)], pathlib.Path(tmp), timeout)
        return None if rc == "timeout" else (rc, out)


PLANTS = {"accept": "fn main() -> i64 {\n    return 3;\n}\n", "reject": "fn main() -> i64 {\n    return @;\n}\n"}
# Stand-in compilers of the stage cases. argv is `check --no-fmt p.mind` or `build p.mind --backend native --out image`.
# A stage that hangs sleeps 8 s: LIMITS give it 3 s and the other stages 30 s, so each stage limit is under test.
FAKES = {"segv": "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n", "hang": "import time\ntime.sleep(8)\n",
         # check rejects as documented, build panics with the panic status of the host compiler
         "panic": "import sys\nif sys.argv[1] == 'check':\n    print('p.mind:2:12: error: unexpected token [E1001]')\n"
                  "    sys.exit(1)\nprint(\"thread 'main' panicked\", file=sys.stderr)\nsys.exit(101)\n",
         "bhang": "import sys, time\nif sys.argv[1] == 'check':\n    print('p.mind:2:12: error: unexpected token [E1001]')\n"
                  "    sys.exit(1)\ntime.sleep(8)\n",
         "berr": "print('p.mind:2:12: error: unexpected token [E1001]')\nraise SystemExit(1)\n",  # both stages exit 1
         "check101": "print(\"thread 'main' panicked\")\nraise SystemExit(101)\n",  # an undocumented check status
         "noimage": "",  # check and build exit 0, and the build writes no image
         # a status next to the documented ones: check exits 2; check exits 1 and build exits 2 with no image
         "check2": "raise SystemExit(2)\n", "build2": "import sys\nraise SystemExit(1 if sys.argv[1] == 'check' else 2)\n",
         "lane": "import sys\nif sys.argv[1] == 'build':\n    print('error: no native  form (lane.plant.key_1)', file=sys.stderr)"
                 "\n    sys.exit(3)\n",
         "plain": "import sys\nif sys.argv[1] == 'build':\n    print('note: n\\nerror:  no   native form\\n')\n    sys.exit(3)\n"}
# The build writes an image (a script) and exits 0; IMAGES are the images of the run cases.
IMAGE = ("import pathlib, sys\nif sys.argv[1] == 'build':\n    out = pathlib.Path(sys.argv[-1])\n"
         "    out.write_text({image!r})\n    out.chmod(0o755)\n")
IMAGES = {"run-hup": "import os, signal\nsignal.signal(signal.SIGHUP, signal.SIG_DFL)\nos.kill(os.getpid(), signal.SIGHUP)\n",
          "run-hang": "import time\ntime.sleep(8)\n",  # ends inside the check and build limits, not in the run limit
          "run-print": "print('x\\\\y\\tz')\n", "run-255": "raise SystemExit(255)\n",
          "run-bytes": "import sys\nsys.stdout.buffer.write(b'\\xff\\n')\n"}  # a stdout that is not UTF-8
LIMITS = {"hang": (3, 30, 30), "bhang": (30, 3, 30), "run-hang": (30, 30, 3)}  # check, build, run


def _safe(fn, *args):
    """fn(*args), or the name of the exception it raised: a crash of the code under test is a red self-test case."""
    try:
        return fn(*args)
    except Exception as error:
        return f"raised {type(error).__name__}: {error}"


def _ratchet_cases(root: pathlib.Path) -> list[tuple[bool, str]]:
    """The ratchet on a planted history of gaps.tsv: counts 3, 2, deleted, 2, 3 (with `#` lines: the newest version is
    above an older one), flat.tsv: 3 each time; a shallow clone of it, a repo with no commit and a folder with no git."""
    repo, rel = root / "history", "gaps.tsv"
    repo.mkdir()
    ok = _git(repo, "init", "-q")[0] == 0
    for k, n in enumerate((3, 2, 0, 2, 3)):  # 0: a commit that deletes the file
        for name, m, tag in (("flat.tsv", 3, f"f{k}"), (rel, n, "g")):  # flat.tsv: other gaps, the same count
            lines = ["# a", "# b", "\t".join(GAP_FIELDS)] + [f"plant/{tag}{i}\tloose\tcheck accepts" for i in range(m)]
            if m:
                (repo / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok = ok and _git(repo, "add", "flat.tsv")[0] == 0
        ok = ok and _git(repo, "add", rel)[0] == 0 if n else ok and _git(repo, "rm", "-q", rel)[0] == 0
        ok = ok and _git(
            repo, "-c", "user.name=plant", "-c", "user.email=plant@example.invalid", "-c", "commit.gpgsign=false",
            "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "-m", f"plant {n}")[0] == 0
    (root / "no-git").mkdir()
    ok = ok and _git(root, "clone", "-q", "--depth", "1", f"file://{repo}", "shallow")[0] == 0
    ok = ok and _git(root, "init", "-q", "empty")[0] == 0
    return [(ok and _safe(ratchet_error, repo, rel, 2) == "", "a gap count not above any committed count passes the ratchet "
             "(a commit that deleted the list gives no count)"),
            (ok and _safe(ratchet_error, repo, rel, 3) != "", "the gap count of the newest committed version is refused when "
             "an older version is below it (`#` lines and the header are not gaps)"),
            (ok and _safe(ratchet_error, repo, "flat.tsv", 3) == "",
             "the same count passes when no committed version is below it"),
            (ok and _safe(ratchet_error, repo, "never.tsv", 9) == "", "a list with no committed version has no committed count"),
            (ok and _safe(ratchet_error, root / "shallow", rel, 1) != "", "the ratchet is refused in a shallow clone"),
            (ok and "git log" in _safe(ratchet_error, root / "empty", rel, 0), "the ratchet is refused when git log fails"),
            (_safe(ratchet_error, root / "no-git", rel, 0) != "", "the ratchet is refused with no git history")]


def _exit_path_cases(root: pathlib.Path, mindc: pathlib.Path, workers: int, rows: list[str],
                     gap_line: str) -> list[tuple[bool, str]]:
    """Run this script as a child on a planted corpus in its own git history and read its exit status."""
    repo = root / "exit-path"
    shutil.copytree(root / "plant", repo / "plant")
    head = "\t".join(ROW_FIELDS)
    status = root / "exit-path-status.md"

    def plant(rows_txt: list[str], gaps_txt: list[str]) -> None:
        (repo / "ROWS.tsv").write_text("\n".join([head, *rows_txt]) + "\n", encoding="utf-8")
        (repo / "KNOWN_GAPS.tsv").write_text("\n".join(["\t".join(GAP_FIELDS), *gaps_txt]) + "\n", encoding="utf-8")
        status.write_text(render_status(load_rows(repo), load_gaps(repo)[0]), encoding="utf-8")

    plant(rows, [gap_line])
    ok = _git(repo, "init", "-q")[0] == 0 and _git(repo, "add", ".")[0] == 0 and _git(
        repo, "-c", "user.name=plant", "-c", "user.email=plant@example.invalid", "-c", "commit.gpgsign=false",
        "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "-m", "plant")[0] == 0

    def child(*extra: str, tool: pathlib.Path = mindc, python: str = sys.executable) -> tuple[int | str, str]:
        argv = [str(python), "-I", str(pathlib.Path(__file__).resolve()), "--corpus", str(repo),
                "--status", str(status), "--workers", str(workers), *extra]
        env = dict(os.environ, MINDC_BIN=str(tool))
        try:
            p = start_child(argv, None, capture_output=True, stdin=subprocess.DEVNULL, env=env, timeout=600)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        return p.returncode, (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")

    results = []

    def case(label: str, want_rc: int, want_text: str) -> None:
        rc, text = child()
        results.append((ok and rc == want_rc and want_text in text, f"gate run exits {want_rc}: {label}"))

    case("differences equal to the list", 0, "[PASS] plant/gap: listed conformance-gap")
    rc, text = child(tool=root / "no-such-compiler")
    results.append((ok and rc == 1 and "no compiler" in text, "gate run exits 1: no compiler"))
    longpy = root / ("p" * 200) / ("y" * 80)  # the "#!" line of the probe image is then over the exec limit (255)
    longpy.parent.mkdir()
    longpy.symlink_to(sys.executable)
    rc, text = child(python=longpy)
    results.append((ok and rc == 1 and "core limit start check" in text, "gate run exits 1: the core limit check fails"))
    want = status.read_text(encoding="utf-8")
    status.write_text("old\n", encoding="utf-8")
    rc, _ = child("--write-status")
    results.append((rc == 0 and status.read_text(encoding="utf-8") == want, "--write-status exits 0 and writes the table"))
    (repo / "ROWS.tsv").write_text("\n".join([head, *rows, "plant/none\taccept\taccept\t-\t-\t-\tp\tp"]) + "\n",
                                   encoding="utf-8")
    rc, text = child("--write-status")
    results.append((rc == 1 and text.startswith("error: ROWS.tsv: row plant/none names no program"),
                    "--write-status exits 1 on a corpus it refuses"))
    plant(rows, [])
    case("a new difference", 1, "[FAIL] plant/gap: new difference")
    plant(rows, ["plant/none\tloose\tcheck accepts"])
    case("a listed gap that names no row", 1, "[FAIL] plant/none: listed gap names no row")
    plant(rows, [gap_line, "plant/accept\tloose\tcheck accepts"])
    case("a gap count above the committed count", 1, "the count may never grow")
    plant(rows, [gap_line])
    status.write_text(status.read_text(encoding="utf-8").replace("| 3 |", "| 4 |", 1), encoding="utf-8")
    case("a stale status table", 1, "is stale")
    plant(rows, [gap_line])
    (repo / "plant" / "extra.mind").write_text(PLANTS["accept"], encoding="utf-8")
    case("a program with no row", 1, "FAIL  corpus: ROWS.tsv: programs with no row: plant/extra")
    (repo / "plant" / "extra.mind").unlink()
    for gaps_txt, want_rc, label in (([], 1, "a new difference is never added"),
                                     (["plant/accept\tloose\tcheck accepts"], 1, "a closed gap is dropped"),
                                     (["plant/gap\tconformance-gap\tstale"], 0, "a changed key is updated")):
        plant(rows, gaps_txt)
        rc, _ = child("--write-gaps")
        listed = [ln for ln in (repo / "KNOWN_GAPS.tsv").read_text(encoding="utf-8").splitlines()
                  if ln and not ln.startswith("#")][1:]
        results.append((ok and rc == want_rc and listed == ([gap_line] if want_rc == 0 else []),
                        f"--write-gaps exits {want_rc}: {label}"))
    return results


def self_test(mindc: pathlib.Path, workers: int) -> int:
    results: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="lang-conformance-self-") as tmp:
        root = pathlib.Path(tmp)
        (root / "plant").mkdir()
        for name, text in PLANTS.items():
            (root / "plant" / f"{name}.mind").write_text(text, encoding="utf-8")
        (root / "plant" / "gap.mind").write_text("// planted gap\n" + PLANTS["reject"], encoding="utf-8")
        head = "\t".join(ROW_FIELDS)
        plain = ["plant/accept\taccept\taccept\t3\t\t-\tplanted\tplanted",
                 "plant/reject\treject\treject\t-\t-\t-\tplanted\tplanted",
                 "plant/gap\taccept\taccept\t-\t-\t-\tplanted\tplanted"]

        def corpus(rows: list[str], gaps: list[str]) -> pathlib.Path:
            (root / "ROWS.tsv").write_text("\n".join([head, *rows]) + "\n", encoding="utf-8")
            (root / "KNOWN_GAPS.tsv").write_text("\n".join(["\t".join(GAP_FIELDS), *gaps]) + "\n", encoding="utf-8")
            return root

        try:  # a crash of the loader or of measure is a red case, not a traceback
            rows = load_rows(corpus(plain, []))
            measured = measure_all(mindc, root, rows, workers)
        except Exception as error:
            print(f"[FAIL] self-test: the planted corpus loads and is measured: {type(error).__name__}: {error}")
            return 1
        key = measured["plant/gap"]["check_key"]
        gap_line = f"plant/gap\tconformance-gap\t{key}"
        by_id = {r["id"]: r for r in rows}

        def outcome(rows_txt: list[str], gaps_txt: list[str]) -> bool | str:
            """True when the gate would be green on this planted corpus."""
            c = corpus(rows_txt, gaps_txt)
            lines = _safe(lambda: judge(load_rows(c), load_gaps(c)[0], measured)[0])  # a refusal is a string too
            return lines if isinstance(lines, str) else all(ok for ok, _ in lines)

        def staged(fake: str) -> dict:
            """The measure of the planted reject program when the compiler is the stand-in `fake`."""
            body = FAKES[fake] if fake in FAKES else IMAGE.format(image=f"#!{sys.executable} -I\n{IMAGES.get(fake, '')}")
            tool = root / f"fake-{fake}"
            # image3 writes an image and exits with the refusal status 3
            tool.write_text(f"#!{sys.executable} -I\n{body}" + ("    raise SystemExit(3)\n" * (fake == "image3")),
                            encoding="utf-8")
            tool.chmod(0o755)
            limits = dict(zip(TIMEOUTS, LIMITS.get(fake, (3, 3, 3))))
            return _safe(measure, tool, root / "plant" / "reject.mind", root / "crash" / f"w-{fake}", limits)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fakes = [*FAKES, *IMAGES, "image3"]
            got_m = dict(zip(fakes, pool.map(staged, fakes)))
        rej, acc = by_id["plant/reject"], by_id["plant/accept"]
        run_row = dict(acc, exit="-", stdout="-")  # an accept row that pins no value
        code = measured["plant/reject"]["code"]

        def judged(row: dict, m: dict) -> tuple[str, str] | str | None:
            return _safe(difference, row, m)

        def crashed(fake: str, row: dict = rej) -> tuple[str, str] | str | None:
            return judged(row, got_m[fake])

        (root / "plant" / "twin.mind").write_text(PLANTS["accept"], encoding="utf-8")
        try:
            load_rows(corpus([*plain, "plant/twin\taccept\taccept\t3\t\t-\tplanted\tplanted"], [gap_line]))
            twin = ""
        except CorpusError as error:
            twin = str(error)
        (root / "plant" / "twin.mind").unlink()
        accept_row = plain[0]
        (root / "plant" / "sub").mkdir()
        (root / "plant" / "sub" / "x.mind").write_text("// not a row\n" + PLANTS["accept"], encoding="utf-8")
        (root / "head.tsv").write_text("id\tclass\n", encoding="utf-8")
        (root / "empty.tsv").write_text("# only a note\n", encoding="utf-8")

        def refused(fn, want: str) -> bool:
            try:
                fn()
            except Exception as error:  # a crash is not a refusal
                return isinstance(error, CorpusError) and want in str(error)
            return False

        def bad(row: str, want: str, gaps: list[str] | None = None) -> bool:
            """`row` replaces the planted row of its id (it is added when it is equal to it or names no planted row)."""
            same = [r for r in plain if r.split("\t")[0] == row.split("\t")[0] and r != row]
            rows_txt = [row if r in same else r for r in plain] + ([] if same else [row])
            return refused(lambda: load_gaps(corpus(rows_txt, gaps or [gap_line])) and load_rows(root), want)

        rej_row, field = plain[1], "\t-\t-\t-\tp\tp"
        refusals = [
            ("a row given twice", bad(accept_row, "given twice")),
            ("an id outside the id form", bad("plant/sub/x\taccept\taccept" + field, "names no program")),
            ("a row with no program", bad("plant/missing\taccept\taccept" + field, "names no program")),
            ("a check verdict that is not accept or reject", bad("plant/accept\tmaybe\taccept" + field, "accept or reject")),
            ("a build verdict that is not accept or reject", bad("plant/accept\taccept\tmaybe" + field, "accept or reject")),
            ("a reject row with build accept", bad(rej_row.replace("\treject\treject\t", "\treject\taccept\t"), "has build reject")),
            ("a build-reject row with an exit", bad("plant/accept\taccept\treject\t3\t-\t-\tp\tp", "no run value")),
            ("a build-reject row with a stdout", bad("plant/accept\taccept\treject\t-\tx\t-\tp\tp", "no run value")),
            ("an accept row with a code", bad(accept_row.replace("\t-\tplanted", "\tE1001\tplanted", 1), "no error code")),
            ("an exit above 255", bad(accept_row.replace("\t3\t", "\t256\t"), "exit is 0..255")),
            ("a stdout that no run prints (a lone backslash)", bad(accept_row.replace("\t3\t\t", "\t3\ta\\q\t"), "stdout is -")),
            ("an exit that is not a number", bad(accept_row.replace("\t3\t", "\tx\t"), "exit is 0..255")),
            ("a code outside the E#### form", bad(rej_row.replace("\t-\tplanted", "\tE12\tplanted", 1), "E#### or -")),
            ("a data line with a wrong field count", bad("plant/x\taccept", "has 2 fields")),
            ("a gap listed twice", bad("", "listed twice", [gap_line, gap_line])),
            ("a gap of an unknown class", bad("", "class is one of", ["plant/gap\tmaybe\tk"])),
            ("a gap with an empty key", bad("", "key is not empty", ["plant/gap\tloose\t"])),
            ("a missing file", refused(lambda: _tsv(root / "none.tsv", GAP_FIELDS), "file is missing")),
            ("a wrong header", refused(lambda: _tsv(root / "head.tsv", GAP_FIELDS), "header must be")),
            ("a file with no header", refused(lambda: _tsv(root / "empty.tsv", GAP_FIELDS), "header must be")),
        ]
        cases = [(f"the corpus is refused: {label}", got, True) for label, got in refusals] + [
            ("blank lines and `#` lines of a data file are not rows",
             outcome([plain[0], "", "# a note", *plain[1:]], [gap_line, "# a note"]), True),
            ("two rows with byte-equal programs are refused", "byte-equal programs" in twin, True),
            ("the planted accept row runs and exits 3", measured["plant/accept"].get("exit") == 3, True),
            ("the planted reject row is refused by check", measured["plant/reject"]["check"] == "reject", True),
            ("differences equal to the list give green", outcome(plain, [gap_line]), True),
            ("a new difference (gap not listed) gives red", outcome(plain, []), False),
            ("a listed gap that no longer differs gives red",
             outcome(plain, [gap_line, "plant/accept\tconformance-gap\tE1001"]), False),
            ("a listed gap of the wrong class gives red", outcome(plain, ["plant/gap\tnative-only-gap\t" + key]), False),
            ("a listed gap with the wrong key gives red", outcome(plain, ["plant/gap\tconformance-gap\tE0000"]), False),
            ("a wrong expected exit status is caught",
             outcome([accept_row.replace("\t3\t", "\t4\t"), *plain[1:]], [gap_line]), False),
            ("a wrong expected stdout is caught",
             outcome([accept_row.replace("\t3\t\t", "\t3\tplanted\\n\t"), *plain[1:]], [gap_line]), False),
            ("a wrong expected check verdict is caught",
             outcome([accept_row.replace("accept\taccept\t3\t\t", "reject\treject\t-\t-\t"), *plain[1:]], [gap_line]),
             False),
            ("a wrong expected native build verdict is caught",
             outcome([accept_row.replace("accept\taccept\t3\t\t", "accept\treject\t-\t-\t"), *plain[1:]], [gap_line]),
             False),
            ("a wrong expected error code is caught",
             outcome([accept_row, plain[1].replace("\t-\tplanted", "\tE9999\tplanted", 1), plain[2]], [gap_line]),
             False),
            ("a reject row whose native build accepts differs (loose)",
             judged(by_id["plant/reject"], dict(measured["plant/reject"], build="accept")),
             ("loose", "native build accepts")),
            ("a compiler killed by a signal on a reject row differs", crashed("segv"),
             ("conformance-gap", "check: signal 11")),
            ("a compiler that hangs on a reject row differs", crashed("hang"), ("conformance-gap", "check: timeout")),
            ("a build that panics (exit 101) on a reject row differs", crashed("panic"),
             ("conformance-gap", "build: exit 101")),
            ("a build that hangs on a reject row differs", crashed("bhang"), ("conformance-gap", "build: timeout")),
            ("a build that exits 1 (compilation error) on a reject row gives no difference", crashed("berr"), None),
            ("a check with an undocumented status (101) differs", crashed("check101"),
             ("conformance-gap", "check: exit 101")),
            ("a check or a build that exits 2 (next to a documented status) on a reject row differs",
             [crashed("check2"), crashed("build2")], [("conformance-gap", "check: exit 2"), ("conformance-gap", "build: exit 2")]),
            ("the documented statuses: check 0 accept and 1 reject, build 1 and 3 a refusal; every other status is a crash",
             (CHECK_STATUS, BUILD_REFUSAL), ({0: "accept", 1: "reject"}, (1, 3))),
            ("a build that exits 0 with no image differs", crashed("noimage", run_row),
             ("native-only-gap", "build: exit 0, no image")),
            ("a build that writes an image and exits 3 differs", crashed("image3", run_row),
             ("native-only-gap", "build: exit 3, image written")),
            ("a native refusal with a lane key is a native-only gap", crashed("lane", run_row),
             ("native-only-gap", "lane.plant.key_1")),
            ("a native refusal with no lane key is an unkeyed refusal", crashed("plain", run_row),
             ("unkeyed-refusal", "error: no native form")),
            ("an expected native refusal gives no difference", crashed("lane", dict(run_row, build="reject")), None),
            ("an image killed by signal 1 (status -1, next to the run range 0..255) differs (run key)",
             crashed("run-hup", run_row), ("native-only-gap", "run: signal 1")),
            ("an image that does not end in the run limit differs (run key)", crashed("run-hang", run_row),
             ("native-only-gap", "run: timeout")),
            ("an image that prints, on a row whose stdout is -, gives no difference", crashed("run-print", run_row), None),
            ("an image that exits 255 on a row that pins 255 gives no difference",
             crashed("run-255", dict(run_row, exit="255")), None),
            ("an exit of 255 loads and is judged", outcome([accept_row.replace("\t3\t", "\t255\t"), *plain[1:]], [gap_line]),
             False),
            ("a wrong exit and a wrong stdout are native-only gaps, a wrong code a conformance gap, with their keys",
             [judged(dict(acc, exit="4"), measured["plant/accept"]), judged(dict(acc, stdout="x"), measured["plant/accept"]),
              judged(dict(rej, code="E9999"), measured["plant/reject"])],
             [("native-only-gap", "run: exit 3, not 4"), ("native-only-gap", "run: stdout differs"),
              ("conformance-gap", f"{measured['plant/reject']['check_key']} for E9999")]),
            ("a native build that accepts on a build-reject row is loose",
             judged(dict(run_row, build="reject"), measured["plant/accept"]), ("loose", "native build accepts")),
            ("the stdout of an image is compared in its escaped form",
             crashed("run-print", dict(run_row, stdout="x\\\\y\\tz\\n")), None),
            ("a stdout that is not UTF-8 keeps each byte (backslash form)",
             crashed("run-bytes", dict(run_row, stdout="\\\\xff\\n")), None),
            ("an image that ends on a row that pins no value gives no difference",
             judged(run_row, measured["plant/accept"]), None),
            ("a reject row whose check accepts differs (loose)", judged(rej, measured["plant/accept"]),
             ("loose", "check accepts")),
            ("a reject row with its measured code gives no difference",
             (bool(re.fullmatch(r"E\d{4}", code)), judged(dict(rej, code=code), measured["plant/reject"])),
             (True, None)),
            ("the check key is the catalog code, else the tag, else the text (one space, 80 characters, no end space)",
             _safe(lambda: [check_key("a: warning: w\na: error: x [E1001]"),
                            check_key("note: n\nError: x [parse:a]\nerror: y [E2001]"), check_key("note:  a\n"),
                            check_key(""), check_key("error:\t" + "b" * 90), check_key("error: " + "b" * 72 + " c")]),
             [("E1001", "E1001"), ("parse:a", "-"), ("note: a", "-"), ("no message", "-"), ("error: " + "b" * 73, "-"),
              ("error: " + "b" * 72, "-")]),
            ("the refusal key is the lane key, else the first error text",
             _safe(lambda: [refusal_key("error: x (lane.a.b_1) y"), refusal_key("lane.c\nerror:  y\tz"),
                            refusal_key("error: " + "c" * 90), refusal_key("error: " + "c" * 72 + " d")]),
             ["lane.a.b_1", "error: y z", "error: " + "c" * 73, "error: " + "c" * 72]),
        ]
        for label, got, want in cases:
            results.append((got == want, label))
        results += _ratchet_cases(root)
        try:  # a crash of a planted gate run or of the loader is a red case, not a traceback
            results += _exit_path_cases(root, mindc, workers, plain, gap_line)
            status = root / "status.md"
            c = corpus(plain, [gap_line])
            status.write_text(render_status(load_rows(c), load_gaps(c)[0]), encoding="utf-8")
            fresh = status_fresh(c, status) == ""
            status.write_text(status.read_text(encoding="utf-8").replace("| 3 |", "| 4 |", 1), encoding="utf-8")
            stale = status_fresh(c, status) != ""
            missing = status_fresh(c, root / "no-status.md") != ""
            results.append((fresh and stale and missing,
                            "the freshness check passes a new table and refuses an edited or a missing one"))
        except Exception as error:
            results.append((False, f"the exit path and freshness cases run: {type(error).__name__}: {error}"))
        saved, found = os.environ.get("MINDC_BIN"), []
        for path in (root / "no-such-compiler", root / "plant", root / "status.md", mindc, None):
            os.environ.pop("MINDC_BIN") if path is None else os.environ.__setitem__("MINDC_BIN", str(path))
            found.append(_safe(resolve_mindc))
        default = REPO / "target" / "release" / "mindc"  # MINDC_BIN unset: the compiler of this tree, never PATH
        default = default if default.is_file() and os.access(default, os.X_OK) else None
        if saved is not None:  # the loop ends with MINDC_BIN unset
            os.environ["MINDC_BIN"] = saved
        results.append((found == [None, None, None, mindc, default], "the compiler is an executable file (not missing, "
                        "not a folder, not a file without the execute bit), else the one of this tree"))
    for ok, label in results:
        print(f"[{'PASS' if ok else 'FAIL'}] self-test: {label}")
    return 0 if all(ok for ok, _ in results) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="prove the gate on planted rows")
    ap.add_argument("--write-status", action="store_true", help="regenerate docs/lang-conformance-status.md")
    ap.add_argument("--write-gaps", action="store_true",
                    help="run, then drop closed gaps and update keys in KNOWN_GAPS.tsv; never adds a gap")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS, help=argparse.SUPPRESS)  # the self-test
    ap.add_argument("--status", type=pathlib.Path, default=STATUS, help=argparse.SUPPRESS)  # plants its own
    ns = ap.parse_args(argv)
    corpus, status = ns.corpus, ns.status
    if ns.write_status:
        try:
            status.write_text(render_status(load_rows(corpus), load_gaps(corpus)[0]), encoding="utf-8")
        except CorpusError as error:
            print(f"error: {error}")
            return 1
        print(f"wrote {status}")
        return 0
    mindc = resolve_mindc()
    if mindc is None:
        print("FAIL  no compiler: set MINDC_BIN or build target/release/mindc")
        return 1
    if not core_limit_check(_image_runner):
        return 1
    if ns.self_test:
        return self_test(mindc, ns.workers)
    try:
        rows, (gaps, _) = load_rows(corpus), load_gaps(corpus)
        stale = status_fresh(corpus, status)
    except CorpusError as error:
        print(f"FAIL  corpus: {error}")
        return 1
    grown = ratchet_error(corpus, "KNOWN_GAPS.tsv", len(gaps))
    if grown:
        print(f"FAIL  KNOWN_GAPS.tsv: {grown}")
        return 1
    print(f"ratchet: {len(gaps)} gaps listed, not above any committed count")
    measured = measure_all(mindc, corpus, rows, ns.workers)
    lines, found = judge(rows, gaps, measured)
    if ns.write_gaps:
        new = {gid: found[gid] for gid in gaps if gid in found}
        write_gaps(corpus / "KNOWN_GAPS.tsv", new)
        added = sorted(set(found) - set(gaps))
        print(f"wrote KNOWN_GAPS.tsv: {len(new)} gaps ({len(gaps) - len(new)} closed); "
              f"{len(added)} new differences not added: {' '.join(added) or 'none'}")
        return 0 if not added else 1
    for ok, text in lines:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    for c in CLASSES:
        print(f"class {c}: {sum(g[0] == c for g in found.values())} measured, "
              f"{sum(g[0] == c for g in gaps.values())} listed")
    valid = [r for r in rows if r["check"] == "accept"]
    print(f"rows {len(rows)}, valid {len(valid)}, valid rows accepted end to end "
          f"{sum(r['id'] not in found for r in valid)}")
    if stale:
        print(f"FAIL  {stale}")
        return 1
    return 0 if all(ok for ok, _ in lines) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
