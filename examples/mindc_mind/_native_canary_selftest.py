"""Planted self-test of native_canary_gate.py, imported by its --self-test; not a gate by itself.

The first case builds a gated row with the real compiler and must give the committed row; when it does not
(no compiler, a file that is not the compiler), the self-test prints one FAIL line and asserts nothing more.
The other cases prove each red verdict: a planted reference, a planted kernel constant, stand-in compilers,
planted table cells, a planted table history, planted images with a source note, and child runs of the gate.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

import native_canary_gate as g
from lang_conformance_gate import CorpusError, _git, _ratchet_cases, _safe
from _selfhost_sources import start_child

REAL_ROW, KERNEL_ROW, REFUSED_ROW = "q16-arith-chain", "collatz", "u64-ops"
NOTE_PROGRAM, PLAIN_PROGRAM = "expressions/logic_and", "control/return_value"  # n_type 2 and n_type 1 today
STUB_SIZE = 1245  # the size of the launcher stub that an emit of no real image once wrote
# Stand-in compilers (argv: build p.mind --backend native --no-cache --out image); {image} is replaced.
WRITE = "import pathlib, sys\nout = pathlib.Path(sys.argv[-1])\nout.write_bytes({image})\nout.chmod(0o755)\n"
STANDINS = {"vary": "import time\n" + WRITE.replace("{image}", "b'#!/bin/sh\\n# ' + str(time.time_ns()).encode()"),
            "silent": WRITE.replace("{image}", "b'#!/bin/sh\\nexit 0\\n'"),
            "stub": WRITE.replace("{image}", f"b'#!/bin/sh\\nexit 0\\n' + b'#' * {STUB_SIZE - 18} + b'\\n'"),
            "refuse": "print('error: no native form (lane.plant.refuse)')\nraise SystemExit(3)\n",
            "segv": "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n"}


def note_elf(*types: int) -> bytes:
    """A minimal ELF64 header with one PT_NOTE that holds one MIND note per n_type."""
    notes = b"".join(struct.pack("<III", 5, 32, t) + b"MIND\x00\x00\x00\x00" + bytes(32) for t in types)
    ehdr = b"\x7fELF\x02\x01\x01" + bytes(9) + struct.pack("<HHIQQQIHHHHHH", 2, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    return ehdr + struct.pack("<IIQQQQQQ", 4, 4, 120, 0, 0, len(notes), len(notes), 4) + notes


def _standin(root: pathlib.Path, name: str, body: str) -> pathlib.Path:
    tool = root / f"standin-{name}"
    tool.write_text(f"#!{sys.executable} -I\n{body}", encoding="utf-8")
    tool.chmod(0o755)
    return tool


def _flip(text: str, at: int) -> str:
    return text[:at] + ("0" if text[at] != "0" else "1") + text[at + 1:]


def _commit(repo: pathlib.Path, msg: str) -> bool:
    return _git(repo, "add", "-A")[0] == 0 and _git(
        repo, "-c", "user.name=plant", "-c", "user.email=plant@example.invalid", "-c", "commit.gpgsign=false",
        "-c", "core.hooksPath=/dev/null", "commit", "-q", "--no-verify", "--allow-empty", "-m", msg)[0] == 0


def _table_text(rows: list[dict]) -> str:
    return "\n".join(["\t".join(g.FIELDS)] + ["\t".join(r[f] for f in g.FIELDS) for r in rows]) + "\n"


def _row_cases(ctx: g.Ctx, rows: dict, real_m: dict, root: pathlib.Path) -> list[tuple[bool, str]]:
    """The seven mutation proofs of the gated row, and the refused-row ratchet."""
    root.mkdir(parents=True)
    row, ref = rows[REAL_ROW], g.reference(ctx.fixtures, REAL_ROW)
    out = []

    def red(label: str, faults: list[str] | str, want: str) -> None:
        text = faults if isinstance(faults, str) else "; ".join(faults)
        out.append((want in text, f"{label} is red ({want})" + ("" if want in text else f": measured {text!r}")))

    fx = root / "fx"
    shutil.copytree(ctx.fixtures / REAL_ROW, fx / REAL_ROW)
    toml = fx / REAL_ROW / "reference_hashes.toml"
    clean = toml.read_text(encoding="utf-8")
    both = clean.replace(ref[0], _flip(ref[0], 7))
    toml.write_text(both, encoding="utf-8")
    red("1 a reference with one hex digit flipped", g.row_faults(row, real_m, g.reference(fx, REAL_ROW), ctx.seed),
        "!= committed reference")
    toml.write_text(clean.replace(f'avx2 = "{ref[0]}"', f'avx2 = "{_flip(ref[0], 7)}"'), encoding="utf-8")
    red("1b an avx2 line that differs from the neon line", g.row_faults(row, real_m, g.reference(fx, REAL_ROW),
                                                                        ctx.seed), "!= neon")
    planted = root / "repo2"
    shutil.copytree(ctx.repo / "tests" / "native_canaries", planted / "tests" / "native_canaries")
    src = (ctx.repo / "examples" / "collatz.mind").read_text(encoding="utf-8")
    (planted / "examples").mkdir()
    (planted / "examples" / "collatz.mind").write_text(src.replace("1000003", "1000004"), encoding="utf-8")
    ctx2 = dataclasses.replace(ctx, repo=planted, drivers=planted / "tests" / "native_canaries")
    m2 = g.measure_row(ctx2, rows[KERNEL_ROW], root / "m2")
    red("2 a kernel constant changed (1000003 -> 1000004), real compiler",
        g.row_faults(rows[KERNEL_ROW], m2, g.reference(ctx.fixtures, KERNEL_ROW), ctx.seed), "!= committed reference")
    stand = {n: dataclasses.replace(ctx, mindc=_standin(root, n, b)) for n, b in STANDINS.items()}
    got = {n: g.measure_row(c, row, root / f"s-{n}") for n, c in stand.items()}
    red("3 a compiler that writes other bytes on each build", g.row_faults(row, got["vary"], ref, ctx.seed),
        "image differs between builds")
    red("4 an image that prints nothing", g.row_faults(row, got["silent"], ref, ctx.seed), "empty output")
    stub_ok = got["stub"]["kind"] == "image" and (root / "s-stub" / "build0" / "w" / "image").stat().st_size == STUB_SIZE
    red(f"4b a {STUB_SIZE}-byte stub image", g.row_faults(row, got["stub"], ref, ctx.seed) if stub_ok else "no stub",
        "empty output")
    red("5 a compiler that refuses a gated row", g.row_faults(row, got["refuse"], ref, ctx.seed), "gated row refused")
    red("5b a compiler that crashes on a gated row", g.row_faults(row, got["segv"], ref, ctx.seed), "gated row crash")
    bad_image = dict(row, image_sha256=_flip(row["image_sha256"], 5))
    red("6 an image column with one hex digit flipped", g.row_faults(bad_image, real_m, ref, ctx.seed), "!= table image")
    bad_seed = dict(row, seed_sha256=_flip(row["seed_sha256"], 5))
    red("7 a seed column that is not the seed", g.row_faults(bad_seed, real_m, ref, ctx.seed), "seed changed")
    m_ref = g.measure_row(ctx, rows[REFUSED_ROW], root / "refused")
    ok, text = g.judge_row(rows[REFUSED_ROW], m_ref, (None, ""), ctx.seed)
    out.append((ok, f"a refused row with its key holds, real compiler ({text})"))
    red("a refused row whose key changed", g.judge_row(dict(rows[REFUSED_ROW], key="lane.plant.other"), m_ref,
                                                        (None, ""), ctx.seed)[1], "listed key lane.plant.other")
    builds_now = dict(row, status="refused", key="lane.plant.key", image_sha256="-", seed_sha256="-")
    red("a refused row that builds now", g.judge_row(builds_now, real_m, ref, ctx.seed)[1], "refused row builds now")
    return out


def _table_cases(ctx: g.Ctx, rows: dict, root: pathlib.Path) -> list[tuple[bool, str]]:
    """Inventory, table refusals, and the gated-set history on a planted git history."""
    root.mkdir(parents=True)
    every = list(rows.values())
    out = [(g.inventory_error(ctx.fixtures, every) == "", "the table names each fixture folder once"),
           ("fixtures with no row" in g.inventory_error(ctx.fixtures, every[1:]), "a fixture with no row is red"),
           ("rows with no fixture" in g.inventory_error(ctx.fixtures, every + [dict(every[0], id="plant-x")]),
            "a row with no fixture is red")]
    path = root / "t" / "table.tsv"
    path.parent.mkdir()
    gated, refused = rows[REAL_ROW], rows[REFUSED_ROW]
    for label, bad in (("a row given twice", [gated, gated]), ("a status that is not gated or refused",
                       [dict(gated, status="pending")]), ("a gated row with a key", [dict(gated, key="lane.x")]),
                       ("a refused row with no key", [dict(refused, key="-")]),
                       ("a gated row with no image", [dict(gated, image_sha256="-")])):
        path.write_text(_table_text(bad), encoding="utf-8")
        out.append((isinstance(_safe(g.load_table, path), str), f"the table is refused: {label}"))
    hist = root / "hist"
    hist.mkdir()
    table = hist / "table.tsv"
    ok = _git(hist, "init", "-q")[0] == 0
    table.write_text(_table_text([gated, refused]), encoding="utf-8")
    ok = ok and _commit(hist, "gated")
    now_refused = [dict(gated, status="refused", key="lane.x", image_sha256="-", seed_sha256="-"), refused]
    table.write_text(_table_text(now_refused), encoding="utf-8")
    ok = ok and _commit(hist, "refused")
    out += [(ok and "not gated now: q16-arith-chain" in g.gated_history_error(table, now_refused),
             "a fixture gated in a committed version and refused now is red"),
            (ok and g.gated_history_error(table, [gated, refused]) == "", "the gated set of every version held passes")]
    ok = ok and _git(root, "clone", "-q", "--depth", "1", f"file://{hist}", "shallow")[0] == 0
    out.append((ok and "full git history" in g.gated_history_error(root / "shallow" / "table.tsv", [gated]),
                "the gated-set history is refused in a shallow clone"))
    return out


def _note_cases(ctx: g.Ctx, root: pathlib.Path, workers: int) -> list[tuple[bool, str]]:
    """The source-note ratchet: the ELF note reader, planted images, the real list, the count ratchet."""
    root.mkdir(parents=True)
    out = [(g.note_types(note_elf(1)) == [1] and g.note_types(note_elf(2)) == [2], "the note reader gives n_type 1, 2"),
           (g.note_types(note_elf(1, 2)) == [1, 2] and g.note_types(b"#!/bin/sh\n") is None,
            "the note reader gives each MIND note; a file that is not an ELF64 image gives none")]
    prog = root / "plant.mind"
    prog.write_text("fn main() -> i64 {\n    return 2;\n}\n", encoding="utf-8")
    image = WRITE.replace("{image}", repr(note_elf(2)))
    tool = dataclasses.replace(ctx, mindc=_standin(root, "note2", image)).mindc
    m = g.measure_notes(tool, {"plant/x": prog}, workers)
    out += [(m == {"plant/x": "n_type 2"} and not all(ok for ok, _ in g.judge_notes(set(), m)) and any(
             "new source note plant/x" in t for _, t in g.judge_notes(set(), m)),
             "a planted image with a source note that is not listed is red"),
            (all(ok for ok, _ in g.judge_notes({"plant/x"}, m)), "the same image, listed, holds")]
    two = dataclasses.replace(ctx, mindc=_standin(root, "note12", WRITE.replace("{image}", repr(note_elf(1, 2))))).mindc
    m2 = g.measure_notes(two, {"plant/y": prog}, workers)
    out.append((not all(ok for ok, _ in g.judge_notes(set(), m2)), "an image with two MIND notes is red"))
    listed = g.load_notes(g.NOTES)
    progs = {p: g.CORPUS / f"{p}.mind" for p in sorted(listed | {PLAIN_PROGRAM})}
    real = g.measure_notes(ctx.mindc, progs, workers)
    lines = g.judge_notes(listed, real)
    out.append((all(ok for ok, _ in lines) and real[PLAIN_PROGRAM] == "n_type 1",
                f"the listed programs give n_type 2 and {PLAIN_PROGRAM} n_type 1, real compiler ({lines[-1][1]})"))
    less = g.judge_notes(listed - {NOTE_PROGRAM}, real)
    out.append((any(not ok and f"new source note {NOTE_PROGRAM}" in t for ok, t in less),
                f"a listed program removed from the list ({NOTE_PROGRAM}) is red"))
    more = g.judge_notes(listed | {PLAIN_PROGRAM}, real)
    out.append((any(not ok and "has no source note now" in t for ok, t in more), "a listed program with n_type 1 is red"))
    path = root / "notes.tsv"
    for label, body in (("a wrong class", f"{NOTE_PROGRAM}\tgap\t{g.NOTE_KEY}"),
                        ("a program listed twice", f"{NOTE_PROGRAM}\t{g.NOTE_CLASS}\t{g.NOTE_KEY}\n" * 2)):
        path.write_text("id\tclass\tkey\n" + body.rstrip("\n") + "\n", encoding="utf-8")
        out.append((isinstance(_safe(g.load_notes, path), str), f"the source-note list is refused: {label}"))
    return out + _ratchet_cases(root)


def _exit_cases(ctx: g.Ctx, rows: dict, root: pathlib.Path) -> list[tuple[bool, str]]:
    """This gate as a child on a planted table, fixture, corpus and list in their own git history."""
    rep = root / "exit"
    (rep / "fixtures").mkdir(parents=True)
    shutil.copytree(ctx.fixtures / REAL_ROW, rep / "fixtures" / REAL_ROW)
    corpus_rows = [ln for ln in (g.CORPUS / "ROWS.tsv").read_text(encoding="utf-8").splitlines()
                   if ln.split("\t")[0] in (NOTE_PROGRAM, PLAIN_PROGRAM) or ln.startswith("id\t")]
    for p in (NOTE_PROGRAM, PLAIN_PROGRAM):
        (rep / "corpus" / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(g.CORPUS / f"{p}.mind", rep / "corpus" / f"{p}.mind")
    (rep / "corpus" / "ROWS.tsv").write_text("\n".join(corpus_rows) + "\n", encoding="utf-8")
    table, notes, toml = rep / "table.tsv", rep / "notes.tsv", rep / "fixtures" / REAL_ROW / "reference_hashes.toml"
    note_row = f"{NOTE_PROGRAM}\t{g.NOTE_CLASS}\t{g.NOTE_KEY}"
    table.write_text(_table_text([rows[REAL_ROW]]), encoding="utf-8")
    notes.write_text(f"id\tclass\tkey\n{note_row}\n", encoding="utf-8")
    ok = _git(rep, "init", "-q")[0] == 0 and _commit(rep, "plant")

    def child(env_mindc: str = str(ctx.mindc)) -> tuple[int | str, str]:
        argv = [sys.executable, "-I", str(pathlib.Path(g.__file__).resolve()), "--table", str(table), "--notes",
                str(notes), "--corpus", str(rep / "corpus"), "--fixtures", str(rep / "fixtures"), "--workers", "2"]
        try:
            p = start_child(argv, None, capture_output=True, stdin=subprocess.DEVNULL, timeout=900,
                            env=dict(os.environ, MINDC_BIN=env_mindc))
        except subprocess.TimeoutExpired:
            return "timeout", ""
        return p.returncode, (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")

    out = []

    def case(label: str, want_rc: int, want: str, **kw) -> None:
        rc, text = child(**kw)
        out.append((ok and rc == want_rc and want in text, f"gate run exits {want_rc}: {label}"
                    + ("" if rc == want_rc and want in text else f" (exit {rc}: {text[-300:]!r})")))

    case("the planted rows hold", 0, f"[PASS] {REAL_ROW}: gated: image {rows[REAL_ROW]['image_sha256']}")
    clean = toml.read_text(encoding="utf-8")
    ref = g.reference(rep / "fixtures", REAL_ROW)[0] or "-" * 64
    toml.write_text(clean.replace(ref, _flip(ref, 3)), encoding="utf-8")
    case("a planted reference", 1, "!= committed reference")
    toml.write_text(clean, encoding="utf-8")
    notes.write_text("id\tclass\tkey\n", encoding="utf-8")
    case("a listed program removed from the list", 1, f"new source note {NOTE_PROGRAM}")
    notes.write_text(f"id\tclass\tkey\n{note_row}\n{PLAIN_PROGRAM}\t{g.NOTE_CLASS}\t{g.NOTE_KEY}\n", encoding="utf-8")
    case("a source-note list longer than a committed one", 1, "the count may never grow")
    notes.write_text(f"id\tclass\tkey\n{note_row}\n", encoding="utf-8")
    table.write_text(_table_text([dict(rows[REAL_ROW], status="refused", key="lane.plant.key", image_sha256="-",
                                       seed_sha256="-")]), encoding="utf-8")
    case("a gated row edited to refused", 1, "gated in a committed version and not gated now")
    table.write_text(_table_text([rows[REAL_ROW]]), encoding="utf-8")
    rc, text = child(env_mindc=str(root / "no-such-mindc"))
    out.append((rc == 1 and "FAIL  no compiler" in text and "[PASS]" not in text,
                "gate run exits 1 and asserts nothing: no compiler"))
    return out


def self_test(ctx: g.Ctx, workers: int) -> int:
    try:
        rows = {r["id"]: r for r in g.load_table(g.TABLE)}
    except CorpusError as error:
        print(f"FAIL  self-test: table: {error}")
        return 1
    with tempfile.TemporaryDirectory(prefix="native-canary-self-") as tmp:
        root = pathlib.Path(tmp)
        real_m = g.measure_row(ctx, rows[REAL_ROW], root / "real")
        ok, text = g.judge_row(rows[REAL_ROW], real_m, g.reference(ctx.fixtures, REAL_ROW), ctx.seed)
        if not ok:  # the compiler is absent or is not the compiler: nothing else is asserted
            print(f"FAIL  self-test: the compiler does not give the committed row: {text}")
            return 1
        results = [(ok, f"the real compiler gives the committed row ({text})")]
        for part in (lambda: _row_cases(ctx, rows, real_m, root / "rows"), lambda: _table_cases(ctx, rows, root / "tb"),
                     lambda: _note_cases(ctx, root / "notes", workers), lambda: _exit_cases(ctx, rows, root / "ex")):
            try:
                results += part()
            except Exception as error:  # a crash of the code under test is a red case
                results.append((False, f"a part of the self-test runs: {type(error).__name__}: {error}"))
    for ok, label in results:
        print(f"[{'PASS' if ok else 'FAIL'}] self-test: {label}")
    return 0 if all(ok for ok, _ in results) else 1
