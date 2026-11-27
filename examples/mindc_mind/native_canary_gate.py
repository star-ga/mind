#!/usr/bin/env python3
"""Native canary gate: the cross-substrate canaries built by the pure-MIND native backend.

Each fixture of tests/cross_substrate_identity/ has one row in testdata/native_canaries/NATIVE_CANARIES.tsv.
A row composes one program: the kernel text, cut by function name from the SAME source that the MLIR gate
compiles (a constant of tests/cross_substrate_identity.rs or an example file), then the helpers and the driver
`main` of tests/native_canaries/. The program is built with `mindc build --backend native` in BUILDS new folders.

  gated    every build writes an image, the images are byte-equal and equal to the image column of the row
           (for the seed of the seed column), RUNS runs of the image exit 0 with one non-empty output, and the
           sha256 of that output EQUALS the committed avx2 line AND neon line of the fixture's
           reference_hashes.toml. The gate never blesses an output reference: a native output that differs
           is a native fault and the gate is red.
  refused  the native backend refuses the program (exit 1 or 3, no image) with the key of the row. A changed
           key is red until the row is edited. A refused row that builds now is red: it leaves the refused
           set only by becoming gated (a driver that prints the reference output, an image column).

Ratchets: a fixture that is gated in any committed version of the table and is not gated now is red (the
gated set never shrinks); every fixture folder has exactly one row. The source-note ratchet builds every
program of the conformance corpus (tests/lang_conformance) natively and reads the MIND note of each image:
n_type 2 (the hash of the source image, written when the IR twin refuses a function) must be listed in
testdata/native_canaries/SOURCE_NOTES.tsv and n_type 1 must not; the count of that list may never be above
the count of any committed version of it.

`--bless` writes the image and seed columns of gated rows whose only fault is the image or the seed, and
prints one `BLESS <id> native-x86 <image> <seed>` line each; it never writes an output reference.
`--self-test` first builds one gated row with the real compiler (so with no compiler it fails and asserts
nothing), then proves each red verdict on planted rows, stand-in compilers and planted images.
The compiler is MINDC_BIN, else target/release/mindc of this tree; there is no fallback to a `mindc` on PATH.
Before any verdict it must build a probe program to an ELF image that exits 7: a file that is not the
compiler gives one FAIL line, exit 1, and no verdict.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import os
import pathlib
import re
import struct
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True  # the tree tracks no Python build products

from lang_conformance_gate import (  # noqa: E402
    GAP_FIELDS, CorpusError, _git, _image_runner, _start, _tsv, load_rows, ratchet_error, refusal_key, resolve_mindc)
from _selfhost_sources import FROZEN_SEED, core_limit_check  # noqa: E402

FIXTURES = REPO / "tests" / "cross_substrate_identity"
DRIVERS = REPO / "tests" / "native_canaries"
DATA = HERE / "testdata" / "native_canaries"
TABLE = DATA / "NATIVE_CANARIES.tsv"
NOTES = DATA / "SOURCE_NOTES.tsv"
CORPUS = REPO / "tests" / "lang_conformance"
FIELDS = ("id", "status", "source", "fns", "driver", "key", "image_sha256", "seed_sha256")
STATUSES = ("gated", "refused")
SUBSTRATES = ("avx2", "neon")
HELPERS = "xsi_native_helpers.mind"
BUILDS = 3
RUNS = 16  # DETERMINISM_RUNS of tests/cross_substrate_identity.rs
TIMEOUTS = {"build": 300, "run": 60}
NOTE_CLASS, NOTE_KEY = "source-note", "n_type 2"
HEX64 = re.compile(r"[0-9a-f]{64}")
IMPORT = re.compile(r"^import [A-Za-z0-9_.]+;[ \t]*$", re.M)


@dataclasses.dataclass(frozen=True)
class Ctx:
    """Where one run reads its inputs; the self-test plants its own."""
    mindc: pathlib.Path
    repo: pathlib.Path = REPO
    fixtures: pathlib.Path = FIXTURES
    drivers: pathlib.Path = DRIVERS
    seed: str = ""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_table(path: pathlib.Path) -> list[dict]:
    rows, seen = _tsv(path, FIELDS), set()
    for r in rows:
        rid = r["id"]
        if rid in seen:
            raise CorpusError(f"{path.name}: row {rid} is given twice")
        seen.add(rid)
        if r["status"] not in STATUSES:
            raise CorpusError(f"{path.name}: row {rid}: status is one of {', '.join(STATUSES)}")
        if r["fns"] != "*" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*)*", r["fns"]):
            raise CorpusError(f"{path.name}: row {rid}: fns is * or a comma list of function names")
        pinned = HEX64.fullmatch(r["image_sha256"]) and HEX64.fullmatch(r["seed_sha256"])
        if r["status"] == "gated" and (r["key"] != "-" or not pinned or r["driver"] == "-"):
            raise CorpusError(f"{path.name}: row {rid}: a gated row has a driver, key -, an image and a seed sha256")
        if r["status"] == "refused" and (r["key"] in ("", "-") or r["image_sha256"] != "-" or r["seed_sha256"] != "-"):
            raise CorpusError(f"{path.name}: row {rid}: a refused row has a key and image and seed -")
    return rows


def inventory_error(fixtures: pathlib.Path, rows: list[dict]) -> str:
    """'' when the rows name exactly the fixture folders (each with a reference_hashes.toml), else why not."""
    folders = {p.parent.name for p in fixtures.glob("*/reference_hashes.toml")}
    ids = {r["id"] for r in rows}
    if not folders:
        return f"no fixture folder under {fixtures}"
    if folders != ids:
        return f"fixtures with no row: {sorted(folders - ids)}; rows with no fixture: {sorted(ids - folders)}"
    return ""


def reference(fixtures: pathlib.Path, rid: str) -> tuple[str | None, str]:
    """(sha, '') from the `<substrate> = "<sha>"` lines (the rule of reference_hash() in the Rust gate)."""
    path = fixtures / rid / "reference_hashes.toml"
    if not path.is_file():
        return None, f"{path} is missing"
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            found.setdefault(k.strip(), v.strip().strip('"'))
    got = [found.get(s) for s in SUBSTRATES]
    if not all(g and HEX64.fullmatch(g) for g in got):
        return None, f"{path.name} has no sha256 line for each of {', '.join(SUBSTRATES)}"
    if len(set(got)) != 1:
        return None, f"{path.name}: avx2 {got[0][:12]} != neon {got[1][:12]}"
    return got[0], ""


def _rust_string(text: str, name: str) -> str:
    raw = re.search(r'const %s: &str = r#"\n(.*?)\n"#;' % re.escape(name), text, re.S)
    if raw:
        return raw.group(1) + "\n"
    lit = re.search(r'const %s: &str = "((?:[^"\\]|\\.)*)";' % re.escape(name), text, re.S)
    if not lit:
        raise CorpusError(f"no string constant {name}")
    body = re.sub(r"\\\n\s*", "", lit.group(1))  # a line continuation drops the newline and the indent
    esc = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
    return re.sub(r"\\(x[0-9a-fA-F]{2}|.)", lambda m: chr(int(m.group(1)[1:], 16)) if m.group(1)[0] == "x"
                  else esc[m.group(1)], body)


def source_text(repo: pathlib.Path, spec: str) -> str:
    """The text of `path` or of the string constant `path#NAME` of a Rust file."""
    rel, _, name = spec.partition("#")
    path = repo / rel
    if not path.is_file():
        raise CorpusError(f"source {rel} is missing")
    text = path.read_text(encoding="utf-8")
    return _rust_string(text, name) if name else text


def fn_text(src: str, name: str) -> str:
    """The text of one top-level fn: from its line to the next top-level fn or `//` line."""
    out, on = [], False
    for ln in src.split("\n"):
        if re.match(r"^(pub )?fn %s\(" % re.escape(name), ln):
            on = True
        elif on and (re.match(r"^(pub )?fn ", ln) or ln.startswith("//")):
            break
        if on:
            out.append(ln)
    if not out:
        raise CorpusError(f"no top-level fn {name}")
    return "\n".join(out).rstrip() + "\n"


def compose(ctx: Ctx, row: dict) -> str:
    """The program of one row: imports first, then kernel text, helpers and driver."""
    src = source_text(ctx.repo, row["source"])
    if row["driver"] == "-":
        return src  # the source is a whole program with its own main
    kernel = src if row["fns"] == "*" else "".join(fn_text(src, f) for f in row["fns"].split(","))
    kernel = re.sub(r"^(pub )?fn main\(", r"\1fn xsi_source_main(", kernel, flags=re.M)
    parts = [(ctx.drivers / HELPERS).read_text(encoding="utf-8"), kernel,
             (ctx.drivers / row["driver"]).read_text(encoding="utf-8")]
    imports = []
    for part in parts:
        imports += [m for m in IMPORT.findall(part) if m.strip() not in [i.strip() for i in imports]]
    body = [IMPORT.sub("", parts[1]), IMPORT.sub("", parts[0]), IMPORT.sub("", parts[2])]
    return "\n".join(i.strip() for i in imports) + "\n" + "\n".join(b.strip("\n") + "\n" for b in body)


def build(mindc: pathlib.Path, text: str | bytes, work: pathlib.Path) -> dict:
    """One native build of `text` in the new folder `work`."""
    work.mkdir(parents=True)
    (work / "p.mind").write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    argv = [str(mindc), "build", "p.mind", "--backend", "native", "--no-cache", "--out", "image"]
    rc, out, _ = _start(argv, work, TIMEOUTS["build"])
    image = work / "image"
    out = out.replace(str(work), "<work>")
    if rc == 0 and image.is_file():
        return {"kind": "image", "image": image, "sha": sha(image.read_bytes())}
    if rc in (1, 3) and not image.exists():
        return {"kind": "refused", "key": refusal_key(out)}
    end = "timeout" if rc == "timeout" else f"signal {-rc}" if rc < 0 else f"exit {rc}"
    return {"kind": "crash", "key": f"build: {end}" + (", image written" if image.exists() else ", no image")}


def measure_row(ctx: Ctx, row: dict, root: pathlib.Path) -> dict:
    """BUILDS builds in folders of other paths and depths; for an image, RUNS runs of the first one."""
    try:
        text = compose(ctx, row)
    except (CorpusError, OSError) as error:
        return {"kind": "crash", "key": f"compose: {error}"}
    builds = []
    for k in range(BUILDS):
        b = build(ctx.mindc, text, root / f"build{k}" / ("w" * (k + 1)))
        builds.append(b)
        if b["kind"] != "image":
            return b
    runs, first = [], b""
    for k in range(RUNS):
        rc, _, out = _start([str(builds[0]["image"])], root / "build0", TIMEOUTS["run"])
        runs.append((rc, sha(out), len(out)))
        first = first or out
    elf = builds[0]["image"].read_bytes()[:4] == b"\x7fELF"
    return {"kind": "image", "images": [b["sha"] for b in builds], "runs": runs, "elf": elf, "out": first}


def row_faults(row: dict, m: dict, ref: tuple[str | None, str], seed: str) -> list[str]:
    """Every fault of a gated row (empty: the row holds)."""
    if m["kind"] != "image":
        return [f"gated row {m['kind']} by native ({m['key']}): the gated set may not shrink"]
    faults = []
    if len(set(m["images"])) != 1:
        faults.append(f"image differs between builds ({', '.join(i[:12] for i in m['images'])})")
    if not m["elf"]:
        faults.append("the image is not an ELF file")
    if m["images"][0] != row["image_sha256"]:
        faults.append(f"image {m['images'][0][:16]} != table image {row['image_sha256'][:16]}")
    if seed != row["seed_sha256"]:
        faults.append(f"seed changed ({seed[:16]} != table seed {row['seed_sha256'][:16]}): re-bless the image column")
    if any(rc != 0 for rc, _, _ in m["runs"]):
        faults.append(f"a run did not exit 0 ({sorted({str(rc) for rc, _, _ in m['runs']})})")
    if any(n == 0 for _, _, n in m["runs"]):
        faults.append("empty output")
    if len({h for _, h, _ in m["runs"]}) != 1:
        faults.append(f"output differs between {RUNS} runs")
    if ref[0] is None:
        faults.append(f"no committed reference: {ref[1]}")
    elif m["runs"][0][1] != ref[0]:
        faults.append(f"native output {m['runs'][0][1][:16]} != committed reference {ref[0][:16]}")
    return faults


def judge_row(row: dict, m: dict, ref: tuple[str | None, str], seed: str) -> tuple[bool, str]:
    rid = row["id"]
    if row["status"] == "refused":
        if m["kind"] == "refused" and m["key"] == row["key"]:
            return True, f"{rid}: refused ({row['key']}), as listed"
        if m["kind"] == "refused":
            return False, f"{rid}: listed key {row['key']}, measured {m['key']}: edit the row"
        if m["kind"] == "image":
            return False, (f"{rid}: refused row builds now: give it a driver that prints the reference output and "
                           "gate it (the refused set only shrinks)")
        return False, f"{rid}: {m['key']}"
    faults = row_faults(row, m, ref, seed)
    if faults:
        return False, f"{rid}: " + "; ".join(faults)
    return True, (f"{rid}: gated: image {m['images'][0]} ({BUILDS} builds equal, = table), output {m['runs'][0][1]} "
                  f"({RUNS} runs equal) = committed reference ({' = '.join(SUBSTRATES)})")


def _versions(path: pathlib.Path) -> tuple[str, list[str]]:
    """('', texts of every committed version of `path`) or (why not, [])."""
    rc, shallow = _git(path.parent, "rev-parse", "--is-shallow-repository")
    if rc != 0 or shallow.strip() != "false":
        return "the ratchet needs the full git history of this tree (no git checkout, or a shallow one)", []
    rc, log = _git(path.parent, "log", "--format=%H", "--", path.name)
    if rc != 0:
        return f"git log of {path.name} gave exit {rc}", []
    texts = []
    for rev in log.split():
        rc, text = _git(path.parent, "show", f"{rev}:./{path.name}")
        if rc == 0:  # a commit that deleted the file has no version
            texts.append(text)
    return "", texts


def gated_history_error(path: pathlib.Path, rows: list[dict]) -> str:
    """'' when every fixture gated in a committed version of the table is gated now, else why not."""
    why, texts = _versions(path)
    if why:
        return why
    now = {r["id"] for r in rows if r["status"] == "gated"}
    ever = set()
    for text in texts:
        lines = [ln.split("\t") for ln in text.splitlines() if ln and not ln.startswith("#")]
        ever |= {c[0] for c in lines[1:] if len(c) == len(FIELDS) and c[1] == "gated"}
    lost = sorted(ever - now)
    return f"gated in a committed version and not gated now: {', '.join(lost)}" if lost else ""


def note_types(image: bytes) -> list[int] | None:
    """The n_type of each MIND note of an ELF64 little-endian image, None when it is not one."""
    if image[:4] != b"\x7fELF" or image[4:6] != b"\x02\x01" or len(image) < 64:
        return None
    phoff, = struct.unpack_from("<Q", image, 32)
    phentsize, phnum = struct.unpack_from("<HH", image, 54)
    types = []
    for k in range(phnum):
        at = phoff + k * phentsize
        if at + 56 > len(image):
            return None
        p_type, = struct.unpack_from("<I", image, at)
        off, = struct.unpack_from("<Q", image, at + 8)
        size, = struct.unpack_from("<Q", image, at + 32)
        if p_type != 4:  # PT_NOTE
            continue
        pos, end = off, off + size
        while pos + 12 <= end <= len(image):
            namesz, descsz, n_type = struct.unpack_from("<III", image, pos)
            name = image[pos + 12:pos + 12 + namesz]
            if name == b"MIND\x00":
                types.append(n_type)
            pos += 12 + (namesz + 3) // 4 * 4 + (descsz + 3) // 4 * 4
    return types


def measure_notes(mindc: pathlib.Path, programs: dict[str, pathlib.Path], workers: int) -> dict[str, str]:
    """For each program: 'refused', 'crash ...', or 'n_type <k>' of the one MIND note of its native image."""
    def one(item: tuple[int, str]) -> str:
        n, pid = item
        b = build(mindc, programs[pid].read_bytes(), root / f"n{n}")  # a lexical row may hold any bytes
        if b["kind"] != "image":
            return b["kind"] if b["kind"] == "refused" else f"crash {b['key']}"
        types = note_types(b["image"].read_bytes())
        return f"n_type {types[0]}" if types and len(types) == 1 else f"no single MIND note ({types})"

    with tempfile.TemporaryDirectory(prefix="native-canary-notes-") as tmp:
        root = pathlib.Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(programs, pool.map(one, enumerate(programs))))


def load_notes(path: pathlib.Path) -> set[str]:
    listed = set()
    for g in _tsv(path, GAP_FIELDS):
        if g["id"] in listed or g["class"] != NOTE_CLASS or g["key"] != NOTE_KEY:
            raise CorpusError(f"{path.name}: row {g['id']}: given once, class {NOTE_CLASS}, key {NOTE_KEY}")
        listed.add(g["id"])
    return listed


def judge_notes(listed: set[str], measured: dict[str, str]) -> list[tuple[bool, str]]:
    lines = []
    for pid in sorted(measured):
        got = measured[pid]
        if got in ("refused", "n_type 1") and pid not in listed:
            continue
        if got == NOTE_KEY and pid in listed:
            lines.append((True, f"source note: {pid} ({NOTE_KEY}), as listed"))
        elif got == NOTE_KEY:
            lines.append((False, f"source note: new source note {pid} ({NOTE_KEY}) is not listed: the count may only fall"))
        elif pid in listed:
            lines.append((False, f"source note: listed {pid} has no source note now ({got}): remove it from the list"))
        else:
            lines.append((False, f"source note: {pid}: {got}"))
    lines += [(False, f"source note: listed {pid} names no corpus program") for pid in sorted(listed - set(measured))]
    images = sum(v.startswith("n_type") for v in measured.values())
    notes = sum(v == NOTE_KEY for v in measured.values())
    lines.append((all(ok for ok, _ in lines), f"source notes: {notes} of {images} corpus images carry {NOTE_KEY}, "
                                             f"{len(listed)} listed"))
    return lines


PROBE = "fn main() -> i64 {\n    return 7;\n}\n"


def probe_error(mindc: pathlib.Path) -> str:
    """'' when `mindc` builds a probe program to an ELF image that exits 7, else why not (then nothing is asserted)."""
    with tempfile.TemporaryDirectory(prefix="native-canary-probe-") as tmp:
        b = build(mindc, PROBE, pathlib.Path(tmp) / "probe")
        if b["kind"] != "image" or b["image"].read_bytes()[:4] != b"\x7fELF":
            return f"{mindc} does not build a probe program to a native ELF image ({b.get('key', 'not an ELF file')})"
        rc, _, _ = _start([str(b["image"])], pathlib.Path(tmp), TIMEOUTS["run"])
        return "" if rc == 7 else f"the probe image of {mindc} exits {rc}, not 7"


def seed_sha(repo: pathlib.Path) -> tuple[str, str]:
    raw = os.environ.get("MINDC_NATIVE_ELF")
    path = pathlib.Path(raw) if raw else repo / FROZEN_SEED
    return (sha(path.read_bytes()), "") if path.is_file() else ("", f"no native seed at {path}")


def run_rows(ctx: Ctx, rows: list[dict], workers: int) -> list[tuple[dict, dict]]:
    with tempfile.TemporaryDirectory(prefix="native-canary-") as tmp:
        root = pathlib.Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            got = list(pool.map(lambda nr: measure_row(ctx, nr[1], root / f"r{nr[0]}"), enumerate(rows)))
    return list(zip(rows, got))


def write_table(path: pathlib.Path, rows: list[dict]) -> None:
    head = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    body = ["\t".join(FIELDS)] + ["\t".join(r[f] for f in FIELDS) for r in rows]
    path.write_text("\n".join(head + body) + "\n", encoding="utf-8")


def bless(ctx: Ctx, table: pathlib.Path, rows: list[dict], workers: int) -> int:
    """Write image and seed of gated rows whose only faults are those two; never an output reference."""
    out, bad = [], 0
    gated = [r for r in rows if r["status"] == "gated"]
    for row, m in run_rows(ctx, gated, workers):
        faults = row_faults(row, m, reference(ctx.fixtures, row["id"]), ctx.seed)
        other = [f for f in faults if not f.startswith(("image ", "seed changed"))
                 or f.startswith("image differs")]
        if other:
            bad += 1
            print(f"not blessed {row['id']}: " + "; ".join(other))
        elif faults:
            row.update(image_sha256=m["images"][0], seed_sha256=ctx.seed)
            out.append(f"BLESS {row['id']} native-x86 {m['images'][0]} {ctx.seed}")
    if not bad:
        write_table(table, rows)
    print("\n".join(out) or "no image to bless")
    return 1 if bad else 0


def gate(ctx: Ctx, table: pathlib.Path, notes: pathlib.Path, corpus: pathlib.Path, workers: int) -> int:
    try:
        rows = load_table(table)
        listed = load_notes(notes)
        programs = {r["id"]: corpus / f"{r['id']}.mind" for r in load_rows(corpus)}
    except CorpusError as error:
        print(f"FAIL  table: {error}")
        return 1
    lines = []
    why = inventory_error(ctx.fixtures, rows)
    lines.append((not why, f"inventory: {len(rows)} rows, one per fixture folder" + (f" ({why})" if why else "")))
    why = gated_history_error(table, rows)
    lines.append((not why, "gated set: no fixture gated in a committed version is refused now" + (f" ({why})" if why else "")))
    why = ratchet_error(notes.parent, notes.name, len(listed))
    lines.append((not why, f"source-note list: {len(listed)} listed, not above any committed count" + (f" ({why})" if why else "")))
    for row, m in run_rows(ctx, rows, workers):
        lines.append(judge_row(row, m, reference(ctx.fixtures, row["id"]), ctx.seed))
    lines += judge_notes(listed, measure_notes(ctx.mindc, programs, workers))
    for ok, text in lines:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
    gated = sum(r["status"] == "gated" for r in rows)
    print(f"rows {len(rows)}: gated {gated}, refused {len(rows) - gated}")
    return 0 if all(ok for ok, _ in lines) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="prove each red verdict; needs the compiler")
    ap.add_argument("--bless", action="store_true", help="write the image and seed columns of gated rows")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    for name, default in (("table", TABLE), ("notes", NOTES), ("corpus", CORPUS), ("fixtures", FIXTURES),
                          ("repo", REPO)):  # the self-test plants its own
        ap.add_argument(f"--{name}", type=pathlib.Path, default=default, help=argparse.SUPPRESS)
    ns = ap.parse_args(argv)
    mindc = resolve_mindc()
    if mindc is None:  # no compiler: nothing is asserted
        print("FAIL  no compiler: MINDC_BIN (else target/release/mindc of this tree) is not an executable file")
        return 1
    seed, why = seed_sha(ns.repo)
    if why:
        print(f"FAIL  {why}")
        return 1
    if not core_limit_check(_image_runner):
        return 1
    why = probe_error(mindc)
    if why:  # a file that is not the compiler: nothing is asserted
        print(f"FAIL  no compiler: {why}")
        return 1
    ctx = Ctx(mindc=mindc, repo=ns.repo, fixtures=ns.fixtures, drivers=ns.repo / "tests" / "native_canaries", seed=seed)
    if ns.self_test:
        from _native_canary_selftest import self_test
        return self_test(ctx, ns.workers)
    if ns.bless:
        try:
            rows = load_table(ns.table)
        except CorpusError as error:
            print(f"error: {error}")
            return 1
        return bless(ctx, ns.table, rows, ns.workers)
    return gate(ctx, ns.table, ns.notes, ns.corpus, ns.workers)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
