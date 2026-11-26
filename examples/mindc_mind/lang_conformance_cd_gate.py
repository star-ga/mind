#!/usr/bin/env python3
"""Language conformance gate, check-driver column: the measured verdict of the pure-MIND check driver.

The corpus in tests/lang_conformance/ (see lang_conformance_gate.py) gives each program the check verdict
that the language specification demands: accept, or reject (with the catalog error code of the first
error when the row names one). This gate runs each program through the pure-MIND check driver
(examples/mindc_mind/check_driver.mind, built to a native ELF the way self_host_check_driver_smoke.py
builds it) and compares the measured verdict with the expected one. No compile path uses this front end
yet: the column measures it on its own.

A run of the driver gives: exit 0 = accept; exit 1 = reject, with its first diagnostic (the first
`ERROR:` line, else the first line) and the catalog code in it, if any. Any other end of the run (a
signal, a timeout, another exit status) is a crash and differs from every expected verdict.

CD_KNOWN_GAPS.tsv lists every row where the measured verdict differs, with its class and the measured
first diagnostic. The gate is green only when the set of measured differences equals that list exactly:
a new difference fails, a listed gap that no longer differs fails, and the number of listed gaps may
never be above the count of any committed version of the list in the git history of HEAD (a shallow
clone or a tree with no history is refused). Classes: accepts-forbidden (the driver accepts what the
specification forbids), refuses-defined (it refuses what the specification defines), other-code (a
reject row whose named code is not the measured one), crash.

docs/lang-conformance-cd-status.md is GENERATED from ROWS.tsv and CD_KNOWN_GAPS.tsv (`--write-status`);
the gate run fails when it is stale. `--write-gaps` drops closed gaps and updates keys, never adds one.
`--self-test` proves the gate on planted rows with stand-in drivers, and runs this script as a child.
It first builds the real check driver and runs it on one planted accept and one planted reject program,
so the self-test too fails (and asserts nothing) when the self-host .so is absent or cannot build it.
The driver is built from MINDC_SO (the self-host .so, as the smoke resolves it).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # the tree tracks no Python build products

from lang_conformance_gate import (  # noqa: E402
    GAP_FIELDS, ROW_FIELDS, CorpusError, _git, _image_runner, _ratchet_cases, _safe, _start, _tsv, load_rows,
    ratchet_error)
from _selfhost_sources import core_limit_check, start_child  # noqa: E402

CORPUS = REPO / "tests" / "lang_conformance"
GAPS = "CD_KNOWN_GAPS.tsv"
STATUS = REPO / "docs" / "lang-conformance-cd-status.md"
CLASSES = ("accepts-forbidden", "refuses-defined", "other-code", "crash")
TIMEOUT = 60
STD_ARG = "--std-src="  # the bundled std text, for the rules that resolve a name against std


def load_gaps(corpus: pathlib.Path) -> dict[str, tuple[str, str]]:
    gaps = {}
    for g in _tsv(corpus / GAPS, GAP_FIELDS):
        if g["id"] in gaps:
            raise CorpusError(f"{GAPS}: row {g['id']} is listed twice")
        if g["class"] not in CLASSES or not g["key"]:
            raise CorpusError(f"{GAPS}: row {g['id']}: class is one of {', '.join(CLASSES)}, key is not empty")
        gaps[g["id"]] = (g["class"], g["key"])
    return gaps


def first_diag(text: str) -> tuple[str, str]:
    """(key, code) of a reject: the first `ERROR:` line, else the first line; its catalog code, else -."""
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    errs = [ln for ln in lines if ln.startswith("ERROR:")]
    key = (errs or lines or ["no message"])[0][:120].rstrip()
    code = re.search(r"\bE\d{4}\b", key)
    return key, code.group(0) if code else "-"


def measure(driver: pathlib.Path, program: pathlib.Path, work: pathlib.Path, std: pathlib.Path | None,
            timeout: int = TIMEOUT) -> dict:
    """One run of the driver on one program, by path, in its own folder."""
    work.mkdir(parents=True)
    src = work / program.name
    shutil.copyfile(program, src)
    argv = [str(driver), src.name] + ([f"{STD_ARG}{std}"] if std else [])
    rc, text, _ = _start(argv, work, timeout)
    if rc == 0:
        return {"check": "accept", "key": "", "code": "-"}
    if rc == 1:
        key, code = first_diag(text)
        return {"check": "reject", "key": key, "code": code}
    end = "timeout" if rc == "timeout" else f"signal {-rc}" if rc < 0 else f"exit {rc}"
    return {"check": "crash", "key": end, "code": "-"}


def difference(row: dict, m: dict) -> tuple[str, str] | None:
    """(class, key) when the measured check verdict differs from the expected one, else None."""
    if m["check"] == "crash":
        return "crash", m["key"]
    if row["check"] == "accept":
        return ("refuses-defined", m["key"]) if m["check"] == "reject" else None
    if m["check"] == "accept":
        return "accepts-forbidden", "check accepts"
    if row["code"] != "-" and m["code"] != row["code"]:
        return "other-code", m["key"]
    return None


def measure_all(driver: pathlib.Path, corpus: pathlib.Path, rows: list[dict], workers: int,
                std: pathlib.Path | None) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="lang-conformance-cd-") as tmp:
        root = pathlib.Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {r["id"]: pool.submit(measure, driver, corpus / f"{r['id']}.mind", root / f"w{n}", std)
                    for n, r in enumerate(rows)}
            return {rid: job.result() for rid, job in jobs.items()}


def judge(rows: list[dict], gaps: dict, measured: dict[str, dict]) -> tuple[list[tuple[bool, str]], dict]:
    """One (ok, text) per row, plus a line per listed gap that names no row; and the measured differences."""
    found, lines = {}, []
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
            lines.append((False, f"{r['id']}: listed {want[0]} no longer differs; remove it from {GAPS}"))
        else:
            lines.append((False, f"{r['id']}: listed {want[0]} ({want[1]}), measured {got[0]} ({got[1]})"))
    lines += [(False, f"{gid}: listed gap names no row of ROWS.tsv") for gid in sorted(set(gaps) - {r["id"] for r in rows})]
    return lines, found


def counts(rows: list[dict], gaps: dict) -> tuple[int, int]:
    """(rows with the expected check verdict, rows with the expected verdict and code)."""
    verdict = sum(r["id"] not in gaps or gaps[r["id"]][0] == "other-code" for r in rows)
    return verdict, sum(r["id"] not in gaps for r in rows)


def render_status(rows: list[dict], gaps: dict) -> str:
    out = ["# Language conformance status: the pure-MIND check driver", "",
           "Generated by `examples/mindc_mind/lang_conformance_cd_gate.py --write-status` from",
           f"`tests/lang_conformance/ROWS.tsv` and `tests/lang_conformance/{GAPS}`. Do not edit by hand:",
           "the gate fails when this file differs from a new generation. See `docs/lang-conformance.md`.", "",
           "Each row is checked by the pure-MIND check driver (`examples/mindc_mind/check_driver.mind`), which",
           "no compile path uses yet. A row is as expected when it is not in the gap list.", "",
           "| family | rows | valid rows | reject rows | expected check verdict | expected verdict and code |",
           "|---|---|---|---|---|---|"]
    for name in [*sorted({r["id"].split("/")[0] for r in rows}), "all"]:
        part = [r for r in rows if name == "all" or r["id"].startswith(name + "/")]
        valid = sum(r["check"] == "accept" for r in part)
        verdict, code = counts(part, gaps)
        out.append(f"| {name} | {len(part)} | {valid} | {len(part) - valid} | {verdict} | {code} |")
    out += ["", "## Gaps by class", "", "| class | rows |", "|---|---|"]
    out += [f"| {c} | {sum(g[0] == c for g in gaps.values())} |" for c in CLASSES]
    for c in CLASSES:
        listed = sorted(gid for gid, g in gaps.items() if g[0] == c)
        out += ["", f"### {c}", ""] + (["None."] if not listed else ["| row | measured first diagnostic |", "|---|---|"])
        out += [f"| `{gid}` | `{gaps[gid][1].replace('|', chr(92) + '|')}` |" for gid in listed]
    return "\n".join(out) + "\n"


def write_gaps(path: pathlib.Path, gaps: dict) -> None:
    body = ["# Rows where the pure-MIND check driver differs from the expected check verdict (docs/lang-conformance.md).",
            "# The count may never grow: the gate compares it with every committed version of this file.",
            "\t".join(GAP_FIELDS)]
    body += [f"{gid}\t{c}\t{k}" for gid, (c, k) in sorted(gaps.items())]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def status_fresh(corpus: pathlib.Path, status: pathlib.Path) -> str:
    """'' when the committed table equals a new generation, else why not."""
    want = render_status(load_rows(corpus), load_gaps(corpus))
    have = status.read_text(encoding="utf-8") if status.is_file() else None
    return "" if have == want else f"{status.name} is stale: run --write-status and commit it"


def std_text(out: pathlib.Path, repo: pathlib.Path = REPO) -> pathlib.Path:
    """The bundled std modules of the host (IMPLICIT_STD_MODULES), concatenated, for the name rules."""
    rs = (repo / "src" / "project" / "stdlib" / "load.rs").read_text(encoding="utf-8")
    table = rs.split("IMPLICIT_STD_MODULES: &[&str] = &[", 1)[1].split("];", 1)[0]
    mods = re.findall(r'"std\.([a-z0-9_]+)"', table)
    if not mods:
        raise CorpusError("load.rs: IMPLICIT_STD_MODULES names no module")
    out.write_bytes(b"\n".join((repo / "std" / f"{m}.mind").read_bytes() for m in mods) + b"\n")
    return out


def build_driver(out: pathlib.Path) -> str:
    """'' when the check driver ELF is written to `out` (built as the smoke builds it), else why not."""
    try:
        import self_host_check_driver_smoke as smoke  # resolves MINDC_SO strictly, as the smoke does
    except SystemExit as error:
        return f"no self-host .so: {error}"
    if not smoke.SO.exists():
        return f"no self-host .so: {smoke.SO} is missing"
    try:  # a file that is not the self-host .so is a refusal, not a traceback
        lib = ctypes.CDLL(str(smoke.SO))
        lib.selftest_native_elf_hb.restype = ctypes.c_int64
        lib.selftest_native_elf_hb.argtypes = [ctypes.c_int64] * 4
        elf = smoke.build_check_elf(lib)
    except (OSError, AttributeError) as error:
        return f"the self-host .so {smoke.SO} cannot build the driver: {type(error).__name__}: {error}"
    if not elf:
        return "the check driver compiled to an empty ELF"
    out.write_bytes(elf)
    out.chmod(out.stat().st_mode | stat.S_IEXEC)
    return ""


# Stand-in drivers of the self-test (argv: driver <file> [--std-src=<path>]).
FAKES = {"acc": "print('fn main: () -> i64')\n",
         "rej": "print('let x: i64\\nERROR: E2001 array length at span [1, 2)\\nERROR: E2002 x at span [3, 4)')\n"
                "raise SystemExit(1)\n",
         "rejnc": "print('ERROR:  return type   mismatch at span [3, 4)')\nraise SystemExit(1)\n",
         "note": "print('\\nnote: cannot read source')\nraise SystemExit(1)\n",
         "empty": "raise SystemExit(1)\n",
         "segv": "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n",
         "hang": "import time\ntime.sleep(8)\n", "exit2": "raise SystemExit(2)\n",
         # rejects unless it gets the program by path in its folder and the std text when one is given
         "args": "import os, sys\nok = os.path.isfile(sys.argv[1]) and os.path.dirname(sys.argv[1]) == ''\n"
                 "std = [a for a in sys.argv[2:] if a.startswith('--std-src=')]\n"
                 "ok = ok and all(open(a[10:]).read() == 'STD' for a in std) and len(sys.argv) == 2 + len(std)\n"
                 "print('' if ok else 'ERROR: E9999 bad argv')\nraise SystemExit(0 if ok else 1)\n",
         "needstd": "import sys\nraise SystemExit(0 if any(a.startswith('--std-src=') for a in sys.argv) else 1)\n"}
ROWS = ["plant/acc\taccept\taccept\t3\t-\t-\tp\tp", "plant/rej\treject\treject\t-\t-\t-\tp\tp",
        "plant/code\treject\treject\t-\t-\tE2001\tp\tp", "plant/gap\taccept\taccept\t-\t-\t-\tp\tp"]
# The planted driver rejects plant/gap (its text has a marker) and accepts the rest.
ROWS_D = [dict(zip(ROW_FIELDS, r.split("\t"))) for r in ["plant/acc\taccept\taccept\t3\t-\t-\tp\tp",
          "plant/rej\treject\treject\t-\t-\tE2004\tp\tp", "plant/code\treject\treject\t-\t-\tE2001\tp\tp",
          "plant/gap\taccept\taccept\t-\t-\t-\tp\tp"]]
SPLIT = ("import sys\nt = open(sys.argv[1]).read()\nif 'GAP' in t or 'REJ' in t:\n"
         "    print('ERROR: E2001 planted at span [0, 1)')\n    raise SystemExit(1)\n")


def _fake(root: pathlib.Path, name: str, body: str) -> pathlib.Path:
    tool = root / f"fake-{name}"
    tool.write_text(f"#!{sys.executable} -I\n{body}", encoding="utf-8")
    tool.chmod(0o755)
    return tool


def _exit_path_cases(root: pathlib.Path, workers: int) -> list[tuple[bool, str]]:
    """Run this script as a child on a planted corpus in its own git history and read its exit status."""
    repo, status, driver = root / "exit-path", root / "exit-status.md", _fake(root, "split", SPLIT)
    (repo / "plant").mkdir(parents=True)
    for rid, text in (("acc", "fn main() -> i64 { return 3; }\n"), ("rej", "REJ\n"), ("code", "REJ 2\n"),
                      ("gap", "GAP\n")):
        (repo / "plant" / f"{rid}.mind").write_text(text, encoding="utf-8")
    gap = "plant/gap\trefuses-defined\tERROR: E2001 planted at span [0, 1)"

    def plant(gaps_txt: list[str]) -> None:
        (repo / "ROWS.tsv").write_text("\n".join(["\t".join(ROW_FIELDS), *ROWS]) + "\n", encoding="utf-8")
        (repo / GAPS).write_text("\n".join(["\t".join(GAP_FIELDS), *gaps_txt]) + "\n", encoding="utf-8")
        status.write_text(render_status(load_rows(repo), load_gaps(repo)), encoding="utf-8")

    plant([gap])
    ok = _git(repo, "init", "-q")[0] == 0 and _git(repo, "add", ".")[0] == 0 and _git(
        repo, "-c", "user.name=plant", "-c", "user.email=plant@example.invalid", "-c", "commit.gpgsign=false",
        "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "-m", "plant")[0] == 0

    def child(*extra: str, tool: pathlib.Path | None = driver, python: str = sys.executable) -> tuple[int | str, str]:
        argv = [str(python), "-I", str(pathlib.Path(__file__).resolve()), "--corpus", str(repo), "--status",
                str(status), "--workers", str(workers), *(["--driver", str(tool)] if tool else []), *extra]
        env = dict(os.environ, MINDC_SO=str(root / "no-such.so"), MINDC_SO_NOBUILD="1")
        try:
            p = start_child(argv, None, capture_output=True, stdin=subprocess.DEVNULL, env=env, timeout=600)
        except subprocess.TimeoutExpired:
            return "timeout", ""
        return p.returncode, (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")

    results = []

    def case(label: str, want_rc: int, want_text: str, *extra: str, **kw) -> None:
        rc, text = child(*extra, **kw)
        results.append((ok and rc == want_rc and want_text in text, f"gate run exits {want_rc}: {label}"))

    case("differences equal to the list", 0, "[PASS] plant/gap: listed refuses-defined")
    case("the counts are printed", 0, "rows 4, expected check verdict 3, expected verdict and code 3")
    case("the classes are counted", 0, "class refuses-defined: 1 measured, 1 listed\nclass other-code: 0 measured, 0 listed")
    longpy = root / ("p" * 200) / ("y" * 80)  # the "#!" line of the probe image is then over the exec limit (255)
    longpy.parent.mkdir()
    longpy.symlink_to(sys.executable)
    rc, text = child(python=longpy)
    results.append((ok and rc == 1 and "core limit start check" in text, "gate run exits 1: the core limit check fails"))
    (repo / "plant" / "extra.mind").write_text("fn main() -> i64 { return 4; }\n", encoding="utf-8")
    case("a program with no row", 1, "FAIL  corpus: ROWS.tsv: programs with no row: plant/extra")
    rc, text = child("--write-status", tool=None)
    results.append((rc == 1 and text.startswith("error: ROWS.tsv: programs with no row"),
                    "--write-status exits 1 on a corpus it refuses"))
    (repo / "plant" / "extra.mind").unlink()
    case("no self-host .so and no driver", 1, "FAIL  no check driver", tool=None)
    case("a driver that is not a file", 1, "FAIL  no check driver", tool=root / "no-such-driver")
    plant([])
    case("a new difference", 1, "[FAIL] plant/gap: new difference")
    plant([gap, "plant/none\tcrash\tsignal 11"])
    case("a gap count above the committed count", 1, "the count may never grow")
    plant([gap])
    status.write_text(status.read_text(encoding="utf-8").replace("| 4 |", "| 5 |", 1), encoding="utf-8")
    case("a stale status table", 1, "is stale")
    want = render_status(load_rows(repo), load_gaps(repo))
    rc, _ = child("--write-status", tool=None)
    results.append((rc == 0 and status.read_text(encoding="utf-8") == want, "--write-status exits 0 and writes the table"))
    for gaps_txt, want_rc, label, says in (([], 1, "a new difference is never added", "not added: plant/gap"),
                                           (["plant/acc\tcrash\tsignal 11"], 1, "a closed gap is dropped", "1 closed"),
                                           (["plant/gap\trefuses-defined\tstale"], 0, "a changed key is updated",
                                            "not added: none")):
        plant(gaps_txt)
        rc, text = child("--write-gaps")
        listed = [ln for ln in (repo / GAPS).read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")][1:]
        results.append((ok and rc == want_rc and says in text and listed == ([gap] if want_rc == 0 else []),
                        f"--write-gaps exits {want_rc}: {label}"))
    return results


# Programs of the real-driver cases: a well-typed function, and a bool returned where i64 is declared.
REAL = (("real-acc", "fn f() -> i64 {\n    let x: i64 = 5;\n    return x;\n}\n", "accept"),
        ("real-rej", "fn g() -> i64 {\n    return 1 < 2;\n}\n", "reject"))


def _real_driver_cases(driver: pathlib.Path, std: pathlib.Path, root: pathlib.Path) -> list[tuple[bool, str]]:
    """The real check driver, run as the corpus mode runs it: so the self-test needs the compiler."""
    results = []
    for name, text, want in REAL:
        prog = root / f"{name}.mind"
        prog.write_text(text, encoding="utf-8")
        m = _safe(measure, driver, prog, root / "w" / name, std)
        ok = isinstance(m, dict) and m["check"] == want and (want == "accept" or m["key"].startswith("ERROR:"))
        label = f"the real check driver gives {want} on {name}" + ("" if ok else f" (measured {m})")
        results.append((ok, label))
    return results


def self_test(workers: int, real: pathlib.Path, std: pathlib.Path) -> int:
    results: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="lang-conformance-cd-self-") as tmp:
        root = pathlib.Path(tmp)
        results += _real_driver_cases(real, std, root)
        prog = root / "p.mind"
        prog.write_text("fn main() -> i64 { return 3; }\n", encoding="utf-8")
        (root / "std.txt").write_text("STD", encoding="utf-8")
        fakes = {name: _fake(root, name, body) for name, body in FAKES.items()}

        def run(name: str, std: pathlib.Path | None = None) -> dict | str:
            return _safe(measure, fakes[name], prog, root / "w" / f"{name}-{std is not None}", std, 3)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            got = dict(zip(FAKES, pool.map(run, FAKES)))
        got["args-std"] = run("args", root / "std.txt")
        got["needstd-std"] = run("needstd", root / "std.txt")
        acc, rej = dict(zip(ROW_FIELDS, ROWS[0].split("\t"))), dict(zip(ROW_FIELDS, ROWS[1].split("\t")))
        code = dict(rej, id="plant/code", code="E2001")
        rej_m = {"check": "reject", "key": "ERROR: E2001 array length at span [1, 2)", "code": "E2001"}

        def judged(rows_txt: list[str], gaps: dict, measured: dict) -> bool | str:
            rows = [dict(zip(ROW_FIELDS, r.split("\t"))) for r in rows_txt]
            lines = _safe(lambda: judge(rows, gaps, measured)[0])
            return lines if isinstance(lines, str) else all(ok for ok, _ in lines)

        m4 = {"plant/acc": {"check": "accept", "key": "", "code": "-"}, "plant/rej": rej_m, "plant/code": rej_m,
              "plant/gap": rej_m}
        g4 = {"plant/gap": ("refuses-defined", rej_m["key"])}
        cases = [
            ("a driver that exits 0 accepts", got["acc"], {"check": "accept", "key": "", "code": "-"}),
            ("a driver that exits 1 rejects; the key is the first ERROR line, the code is in it", got["rej"], rej_m),
            ("a reject with no code: the key is the first ERROR line (one space), the code is -", got["rejnc"],
             {"check": "reject", "key": "ERROR: return type mismatch at span [3, 4)", "code": "-"}),
            ("a reject with no ERROR line: the key is its first line", got["note"],
             {"check": "reject", "key": "note: cannot read source", "code": "-"}),
            ("a reject with no text has the key 'no message'", got["empty"],
             {"check": "reject", "key": "no message", "code": "-"}),
            ("a driver killed by a signal is a crash", got["segv"], {"check": "crash", "key": "signal 11", "code": "-"}),
            ("a driver that does not end in the limit is a crash", got["hang"],
             {"check": "crash", "key": "timeout", "code": "-"}),
            ("a driver that exits 2 (next to a documented status) is a crash", got["exit2"],
             {"check": "crash", "key": "exit 2", "code": "-"}),
            ("the driver gets the program by path in its folder, and no other argument", got["args"]["check"], "accept"),
            ("the driver gets the std text as --std-src=<path> when one is given",
             [got["args-std"]["check"], got["needstd"]["check"], got["needstd-std"]["check"]],
             ["accept", "reject", "accept"]),
            ("a key is cut at 120 characters with no end space",
             _safe(first_diag, "ERROR: " + "a" * 112 + " b E1234"), ("ERROR: " + "a" * 112, "-")),
            ("the code is a whole E#### word", _safe(first_diag, "ERROR: XE12345 E123"), ("ERROR: XE12345 E123", "-")),
            ("accept row, accept: no difference", _safe(difference, acc, got["acc"]), None),
            ("accept row, reject: refuses what the specification defines", _safe(difference, acc, rej_m),
             ("refuses-defined", rej_m["key"])),
            ("reject row with no code, reject: no difference", _safe(difference, rej, rej_m), None),
            ("reject row, accept: accepts what the specification forbids", _safe(difference, rej, got["acc"]),
             ("accepts-forbidden", "check accepts")),
            ("reject row with its code: no difference", _safe(difference, code, rej_m), None),
            ("reject row with another code: other-code", _safe(difference, dict(rej, code="E2004"), rej_m),
             ("other-code", rej_m["key"])),
            ("reject row with a code, a reject with no code: other-code", _safe(difference, code, got["rejnc"]),
             ("other-code", "ERROR: return type mismatch at span [3, 4)")),
            ("a crash differs on an accept row and on a reject row",
             [_safe(difference, acc, got["segv"]), _safe(difference, rej, got["hang"])],
             [("crash", "signal 11"), ("crash", "timeout")]),
            ("differences equal to the list give green", judged(ROWS, g4, m4), True),
            ("a new difference (not listed) gives red", judged(ROWS, {}, m4), False),
            ("a listed gap that no longer differs gives red",
             judged(ROWS, {**g4, "plant/acc": ("crash", "signal 11")}, m4), False),
            ("a listed gap of the wrong class gives red", judged(ROWS, {"plant/gap": ("crash", rej_m["key"])}, m4), False),
            ("a listed gap with the wrong key gives red", judged(ROWS, {"plant/gap": ("refuses-defined", "x")}, m4), False),
            ("a listed gap that names no row gives red", judged(ROWS, {**g4, "plant/none": ("crash", "x")}, m4), False),
            ("a wrong expected check verdict is caught", judged([ROWS[0].replace("accept\taccept\t3", "reject\treject\t-"),
                                                                 *ROWS[1:]], g4, m4), False),
            ("a wrong expected code is caught", judged([*ROWS[:2], ROWS[2].replace("E2001", "E2004"), ROWS[3]], g4, m4),
             False),
            ("each judge line names what differs",
             _safe(lambda: [t for _, t in judge(ROWS_D, {"plant/acc": ("crash", "k"), "plant/gap": ("crash", "k"),
                                                        "plant/none": ("crash", "k")}, m4)[0]]),
             ["plant/acc: listed crash no longer differs; remove it from CD_KNOWN_GAPS.tsv",
              "plant/code: expected verdict",
              "plant/gap: listed crash (k), measured refuses-defined (ERROR: E2001 array length at span [1, 2))",
              "plant/rej: new difference other-code (ERROR: E2001 array length at span [1, 2)); fix it or list it",
              "plant/none: listed gap names no row of ROWS.tsv"]),
            ("the status table: one line per family and the total, the classes, the listed rows",
             _safe(lambda: [ln for ln in render_status([acc, rej, dict(acc, id="seed/a"), dict(code, id="seed/b"), dict(acc, id="seed/c")],
                                                       {"seed/a": ("refuses-defined", "a|b"), "plant/rej": ("other-code", "k")})
                            .splitlines() if ln.startswith("|") or ln.startswith("None")][2:]),
             ["| plant | 2 | 1 | 1 | 2 | 1 |", "| seed | 3 | 2 | 1 | 2 | 2 |", "| all | 5 | 3 | 2 | 4 | 3 |",
              "| class | rows |", "|---|---|", "| accepts-forbidden | 0 |", "| refuses-defined | 1 |", "| other-code | 1 |",
              "| crash | 0 |", "None.", "| row | measured first diagnostic |", "|---|---|", "| `seed/a` | `a\\|b` |",
              "| row | measured first diagnostic |", "|---|---|", "| `plant/rej` | `k` |", "None."]),
            ("the judge returns the measured differences", _safe(lambda: judge([acc, dict(acc, id="plant/gap")], {},
                                                                                 m4)[1]), {"plant/gap": g4["plant/gap"]}),
            ("the counts: expected verdict counts other-code rows, expected code does not",
             _safe(counts, [acc, rej, code, dict(acc, id="plant/gap")],
                   {"plant/rej": ("other-code", "k"), "plant/gap": ("crash", "k")}), (3, 2)),
        ]
        results += [(want == have, label) for label, have, want in cases]
        (root / "c").mkdir()
        for name, text in (("ROWS.tsv", "\t".join(ROW_FIELDS)), (GAPS, "\t".join(GAP_FIELDS))):
            (root / "c" / name).write_text(text + "\n", encoding="utf-8")

        def refused(lines: list[str] | None, want: str, head: str = "\t".join(GAP_FIELDS)) -> bool:
            if lines is None:
                (root / "c" / GAPS).unlink()
            else:
                (root / "c" / GAPS).write_text("\n".join([head, *lines]) + "\n", encoding="utf-8")
            try:
                load_gaps(root / "c")
            except CorpusError as error:
                return want in str(error)
            return False

        results += [(refused(["a/b\tcrash\tk", "a/b\tcrash\tk"], "listed twice"), "the list is refused: a gap listed twice"),
                    (refused(["a/b\tloose\tk"], "class is one of"), "the list is refused: a gap of an unknown class"),
                    (refused(["a/b\tcrash\t"], "key is not empty"), "the list is refused: a gap with an empty key"),
                    (not refused(["a/b\tcrash\tk", "# n", ""], "x"), "a list with notes and blank lines loads"),
                    (refused(["a/b\tcrash"], "has 2 fields"), "the list is refused: a data line with a wrong field count"),
                    (refused([], "header must be", "id\tclass"), "the list is refused: a wrong header"),
                    (refused([], "header must be", "# only a note"), "the list is refused: no header"),
                    (refused(None, "file is missing"), "the list is refused: a missing file")]
        results += _ratchet_cases(root)
        fake = root / "repo"
        (fake / "src" / "project" / "stdlib").mkdir(parents=True)
        (fake / "std").mkdir()
        for m in ("a", "b"):
            (fake / "std" / f"{m}.mind").write_text(f"// {m}\n", encoding="utf-8")
        load = fake / "src" / "project" / "stdlib" / "load.rs"
        load.write_text('X: &[&str] = &["std.c"];\nIMPLICIT_STD_MODULES: &[&str] = &[\n "std.b", "std.a",\n];\n"std.c"\n')
        two = _safe(lambda: std_text(root / "std2.mind", fake).read_text(encoding="utf-8"))
        load.write_text("IMPLICIT_STD_MODULES: &[&str] = &[\n];\n")
        none = _safe(std_text, root / "std0.mind", fake)
        results.append((two == "// b\n\n// a\n\n" and "names no module" in str(none),
                        "the std text joins the modules of IMPLICIT_STD_MODULES in order; a table with none is refused"))
        try:
            results += _exit_path_cases(root, workers)
            c, st = root / "exit-path", root / "status.md"
            st.write_text(render_status(load_rows(c), load_gaps(c)), encoding="utf-8")
            fresh = status_fresh(c, st) == ""
            st.write_text(st.read_text(encoding="utf-8").replace("| 4 |", "| 5 |", 1), encoding="utf-8")
            results.append((fresh and status_fresh(c, st) != "" and status_fresh(c, root / "none.md") != "",
                            "the freshness check passes a new table and refuses an edited or a missing one"))
        except Exception as error:
            results.append((False, f"the exit path and freshness cases run: {type(error).__name__}: {error}"))
    for ok, label in results:
        print(f"[{'PASS' if ok else 'FAIL'}] self-test: {label}")
    return 0 if all(ok for ok, _ in results) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="prove the gate on planted rows and stand-in drivers")
    ap.add_argument("--write-status", action="store_true", help="regenerate docs/lang-conformance-cd-status.md")
    ap.add_argument("--write-gaps", action="store_true", help=f"run, then drop closed gaps and update keys in {GAPS}")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--corpus", type=pathlib.Path, default=CORPUS, help=argparse.SUPPRESS)  # the self-test
    ap.add_argument("--status", type=pathlib.Path, default=STATUS, help=argparse.SUPPRESS)  # plants its own
    ap.add_argument("--driver", type=pathlib.Path, help=argparse.SUPPRESS)  # a stand-in driver of the self-test
    ns = ap.parse_args(argv)
    corpus, status = ns.corpus, ns.status
    if ns.write_status:
        try:
            status.write_text(render_status(load_rows(corpus), load_gaps(corpus)), encoding="utf-8")
        except CorpusError as error:
            print(f"error: {error}")
            return 1
        print(f"wrote {status}")
        return 0
    if not core_limit_check(_image_runner):
        return 1
    if ns.self_test:
        with tempfile.TemporaryDirectory(prefix="lang-conformance-cd-real-") as tmp:
            real = pathlib.Path(tmp) / "check.elf"
            why = build_driver(real)
            if why:  # no compiler: the self-test asserts nothing, as the corpus mode
                print(f"FAIL  no check driver: {why}")
                return 1
            return self_test(ns.workers, real, std_text(pathlib.Path(tmp) / "std.mind"))
    try:
        rows, gaps = load_rows(corpus), load_gaps(corpus)
        stale = status_fresh(corpus, status)
    except CorpusError as error:
        print(f"FAIL  corpus: {error}")
        return 1
    grown = ratchet_error(corpus, GAPS, len(gaps))
    if grown:
        print(f"FAIL  {GAPS}: {grown}")
        return 1
    print(f"ratchet: {len(gaps)} gaps listed, not above any committed count")
    with tempfile.TemporaryDirectory(prefix="lang-conformance-cd-driver-") as tmp:
        driver, std = ns.driver, None
        if driver is None:
            driver = pathlib.Path(tmp) / "check.elf"
            why = build_driver(driver)
            std = None if why else std_text(pathlib.Path(tmp) / "std.mind")
        else:
            why = "" if driver.is_file() else f"{driver} is not a file"
        if why:
            print(f"FAIL  no check driver: {why}")
            return 1
        measured = measure_all(driver, corpus, rows, ns.workers, std)
    lines, found = judge(rows, gaps, measured)
    if ns.write_gaps:
        new = {gid: found[gid] for gid in gaps if gid in found}
        write_gaps(corpus / GAPS, new)
        added = sorted(set(found) - set(gaps))
        print(f"wrote {GAPS}: {len(new)} gaps ({len(gaps) - len(new)} closed); "
              f"{len(added)} new differences not added: {' '.join(added) or 'none'}")
        return 0 if not added else 1
    for ok, text in lines:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    for c in CLASSES:
        print(f"class {c}: {sum(g[0] == c for g in found.values())} measured, {sum(g[0] == c for g in gaps.values())} listed")
    verdict, code = counts(rows, found)
    print(f"rows {len(rows)}, expected check verdict {verdict}, expected verdict and code {code}")
    if stale:
        print(f"FAIL  {stale}")
        return 1
    return 0 if all(ok for ok, _ in lines) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
