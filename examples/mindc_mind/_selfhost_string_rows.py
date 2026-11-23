"""Runner of the string-rule acceptance rows (tests/fixtures/native_lane_string_rows, README.md there):
`--string-rows`, `--string-rows-mindc`, `--string-rows-baseline`, `--string-rows-mutant` and the runner's self-test.
Split out of _selfhost_sources.py."""
from dataclasses import dataclass, field, replace
import collections
import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

from _selfhost_sources import _REPO, BOUNDED_LAYOUT, FROZEN_SEED, LAYOUT_MAGIC, capture_bundle, start_child


# String-rule acceptance rows (tests/fixtures/native_lane_string_rows/, README.md there): programs go straight to a
# compiler ELF (direct feed, no MINDSLF magic).
STRING_ROWS_DIR = "tests/fixtures/native_lane_string_rows"
ROW_COLUMNS = ("id", "verdict", "key", "line", "col", "exit", "profile", "status", "clause", "note")
ROW_STATUS_VERDICTS = {"run": ("admit", "refuse"), "carrier": ("admit", "refuse"), "by_run": ("by_run",), "capacity": ("pending",), "outside": ("admit",), "host": ("refuse",)}
PROFILES = ("default", "bounded")
ROW_PROFILES = {"both": PROFILES, "default": ("default",), "bounded": ("bounded",)}  # the profile column: the profiles a row runs on


@dataclass
class StringRows:
    rows: list
    programs: dict
    mutants: list
    keys: dict
    carriers: dict
    beyond: dict = field(default_factory=dict)  # {id: the program one nesting level past a ceiling row}
    ceiling_recipes: dict = field(default_factory=dict)  # {record: recipe parts with a `ceiling` count}, set per compiler


def _tsv_rows(path: pathlib.Path, columns: tuple[str, ...]) -> list[dict]:
    lines = [(n, ln) for n, ln in enumerate(path.read_bytes().decode("utf-8").split("\n"), 1) if ln and not ln.startswith("#")]
    if not lines or tuple(lines[0][1].split("\t")) != columns:
        raise ValueError(f"{path.name}: the header must be {chr(9).join(columns)}")
    out = []
    for n, line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise ValueError(f"{path.name}:{n}: {len(cells)} cells, want {len(columns)}")
        out.append(dict(zip(columns, cells)))
    return out


def _program_records(path: pathlib.Path) -> tuple[dict, dict, dict, dict]:
    """({id: program bytes}, {id: carrier module text}, {id: program bytes of `<id>@beyond`}, {record: parts of a recipe
    with a ceiling count}) of programs.txt; a record is `=== <id> | <form>`."""
    records: list[tuple[str, str, list[str]]] = []
    for line in path.read_bytes().decode("utf-8").split("\n")[:-1]:
        if line.startswith("=== "):
            name, _, form = line[4:].partition(" | ")
            records.append((name, form, []))
        elif records:
            records[-1][2].append(line)
    programs, carriers, recipes = {}, {}, {}
    for name, form, body in records:
        if name in programs or name in carriers:
            raise ValueError(f"programs.txt: duplicate record {name}")
        if form == "text":
            programs[name] = ("\n".join(body) + "\n").encode()
        elif form == "hex":
            programs[name] = bytes.fromhex("".join(body))
        elif form == "recipe":
            parts = [(c if re.fullmatch(r"ceiling([+-][0-9]+)?", c) else int(c), json.loads(t)) for c, t in (b.split(" ", 1) for b in body)]
            programs[name] = _expand(parts, nesting_ceiling(0))
            if any(isinstance(c, str) for c, _ in parts):
                recipes[name] = parts
        elif form == "carrier" and name.endswith("@carrier"):
            carriers[name[:-8]] = "\n".join(body)
        else:
            raise ValueError(f"programs.txt: record {name} has the unknown form {form!r}")
    beyond = {n[:-7]: programs.pop(n) for n in list(programs) if n.endswith("@beyond")}
    return programs, carriers, beyond, recipes


STACK_FLOOR = 402653184  # nb_own_stack_bytes: 384 MiB when the compiler can not read its own stack (not a native image)


def nesting_ceiling(stack: int) -> int:
    """The deepest parenthesis nesting of a receiver in a function body that a compiler with this stack admits: the budget
    (nb_depth_short, nb_stack_budget.mind) is (stack - 32 MiB) / 1280 B units, 4 for the body `{` and 16 for each `(`.
    A native image reads its own stack, of any size; stack 0 (not an ELF) is the fallback STACK_FLOOR."""
    return (((stack if stack > 0 else STACK_FLOOR) - 33554432) // 1280 - 4) // 16


def elf_stack(path: str) -> int:
    """The stack of a compiler ELF as the compiler reads its own (nb_own_stack_bytes): the memsz of the PT_LOAD with the
    highest vaddr; 0 for a file that is not ELF64."""
    d = pathlib.Path(path).read_bytes()
    if d[:5] != b"\x7fELF\x02":
        return 0
    phoff, (phentsize, phnum) = struct.unpack_from("<Q", d, 32)[0], struct.unpack_from("<HH", d, 54)
    loads = [struct.unpack_from("<IIQQQQQ", d, phoff + k * phentsize) for k in range(phnum)]
    return max(((p[3], p[6]) for p in loads if p[0] == 1), default=(0, 0))[1]


def _expand(parts: list, ceiling: int) -> bytes:
    """The bytes of a recipe: each count is a number, or `ceiling`, `ceiling+k`, `ceiling-k` (the nesting ceiling)."""
    return "".join(t * (c if isinstance(c, int) else ceiling + int(c[7:] or 0)) for c, t in parts).encode()


def at_ceiling(srows: StringRows, compiler: str) -> StringRows:
    """The rows with every ceiling count set to the nesting ceiling of `compiler`: a ceiling row sits at the ceiling of the
    RUNNING compiler (18022 on the 384 MiB stack of the bounded profile, 208076 on the 4 GiB stack of the default profile)."""
    c = nesting_ceiling(elf_stack(compiler))
    got = {n: _expand(parts, c) for n, parts in srows.ceiling_recipes.items()}
    return replace(srows, programs={**srows.programs, **{n: b for n, b in got.items() if n in srows.programs}},
                   beyond={**srows.beyond, **{n[:-7]: b for n, b in got.items() if n.endswith("@beyond")}})


def load_string_rows(root: pathlib.Path = _REPO, rows_dir: str = STRING_ROWS_DIR) -> StringRows:
    """The rows, programs, mutant map and key vocabulary; ValueError on any inconsistency."""
    base = pathlib.Path(root) / rows_dir
    keys = {}
    for r in _tsv_rows(base / "keys.tsv", ("key", "id")):
        if r["key"] in keys or not re.fullmatch(r"[a-z]+(\.[a-z0-9_]+)+", r["key"]) or not re.fullmatch(r"[0-9]+|-", r["id"]):
            raise ValueError(f"keys.tsv: bad or duplicate row {r['key']}")
        keys[r["key"]] = r["id"]
    rows = _tsv_rows(base / "rows.tsv", ROW_COLUMNS)
    programs, carriers, beyond, recipes = _program_records(base / "programs.txt")
    seen = set()
    for r in rows:
        i = r["id"]
        bad = [] if re.fullmatch(r"[a-z0-9_]+", i) and i not in seen else ["id is malformed or duplicated"]
        seen.add(i)
        if r["status"] not in ROW_STATUS_VERDICTS or r["verdict"] not in ROW_STATUS_VERDICTS.get(r["status"], ()):
            bad.append(f"status {r['status']!r} does not go with verdict {r['verdict']!r}")
        if r["profile"] not in ROW_PROFILES:
            bad.append(f"profile {r['profile']!r}")
        if r["verdict"] == "refuse" and r["key"] not in keys:
            bad.append(f"unknown key {r['key']!r}" if r["key"] else "a refuse row needs a key")
        if r["verdict"] in ("admit", "pending") and r["key"]:
            bad.append("only a refuse or by_run row names a key")
        if r["verdict"] == "by_run" and r["key"] and r["key"] not in keys:
            bad.append(f"unknown key {r['key']!r}")
        for col in ("line", "col", "exit"):
            if r[col] and not r[col].isdigit():
                bad.append(f"{col} {r[col]!r} is not a number")
        if (r["line"] or r["col"]) and r["verdict"] != "refuse" or r["col"] and not r["line"]:
            bad.append("line and col belong to a refuse row (col needs line)")
        if r["status"] == "host" and r["key"].startswith("closure.") and (r["line"] or r["col"] or not closure_symbol(r)):
            bad.append("a host row of a closure key states the symbol in its note (symbol `name`: ...) and no line")
        if r["exit"] and (r["verdict"] != "admit" or int(r["exit"]) > 255):
            bad.append("exit belongs to an admit row and is at most 255")
        if i not in programs:
            bad.append("no program record")
        if (r["status"] == "carrier" and i not in carriers) or (i in carriers and r["status"] not in ("carrier", "run")):
            bad.append("a carrier row needs exactly one @carrier record")
        if bad:
            raise ValueError(f"rows.tsv {i}: " + "; ".join(bad))
    orphans = sorted((set(programs) | set(carriers)) - seen)
    if orphans:
        raise ValueError(f"programs.txt records without a row: {orphans}")
    owners = {r["id"]: r for r in rows}
    stray = sorted(i for i in beyond if i not in owners or (owners[i]["verdict"], owners[i]["status"]) != ("admit", "run"))
    if stray:
        raise ValueError(f"programs.txt: a @beyond record belongs to an admit row of status run: {stray}")
    mutants = []
    for m in _tsv_rows(base / "mutants.tsv", ("arm", "clause", "off", "rows")):
        ids = m["rows"].split(",")
        unknown = [i for i in ids if i not in seen]
        if not ids or len(set(ids)) != len(ids) or unknown or m["arm"] in [x["arm"] for x in mutants]:
            raise ValueError(f"mutants.tsv {m['arm']}: empty, repeated or unknown rows {unknown}, or a repeated arm")
        mutants.append({**m, "rows": ids})
    text = " ".join((base / "README.md").read_text().split())
    stale_counts = [c for c in readme_counts(rows) if c not in text]
    if stale_counts:
        raise ValueError(f"README.md: these counts of the rows are not in the text (a stale count): {stale_counts}")
    return StringRows(rows, programs, mutants, keys, carriers, beyond, recipes)


def _by_count(c: collections.Counter, form: str) -> str:
    return ", ".join(form.format(k=k, n=n) for k, n in sorted(c.items(), key=lambda kn: (-kn[1], kn[0])))


def readme_counts(rows: list[dict]) -> list[str]:
    """The count sentences of README.md, made from the rows (whitespace folded): the README states each one."""
    st = collections.Counter(r["status"] for r in rows)
    of = {v: collections.Counter(r["status"] for r in rows if r["verdict"] == v) for v in ("admit", "refuse")}
    skip = collections.Counter({s: n for s, n in st.items() if s not in ("run", "host")})
    keys = collections.Counter(r["key"] for r in rows if r["verdict"] == "refuse")
    ids = {r["id"] for r in rows}
    twins = sum(r["id"].endswith("_host") and r["id"][:-5] in ids for r in rows)
    return [f"| all | {len(rows)} |", *(f"| `{v}` | {sum(c.values())} ({_by_count(c, '{n} `{k}`')}) |" for v, c in of.items()),
            f"| `by_run` | {st['by_run']} |", f"| `pending` (`capacity`) | {st['capacity']} |", f"| asserted directly (`run`) | {st['run']} |",
            f"| host path, not run without a mindc binary (`host`) | {st['host']} |", f"| skipped | {sum(skip.values())}: {_by_count(skip, '`{k}` {n}')} |",
            "`refuse` rows by key: " + _by_count(keys, "{k} {n}") + ".", f"`host path, not run: {st['host']}`", f"{twins} rows have such a twin"]


@dataclass
class Observation:
    rc: int
    elf: bool
    err: str
    image_exit: int | None = None


def names_key(text: str, key: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_.])" + re.escape(key) + r"(?![A-Za-z0-9_])", text) is not None


def position_ok(text: str, line: int, col: int | None) -> bool:
    """`line N` / `N:M`, or with a known column `line N, column M` / `line N col M`; a row with no column
    accepts `line N` or `:N`. The reporter's own format is not fixed yet: these are the renderings accepted."""
    if col is None:
        pats = (rf"\bline {line}(?![0-9])", rf"(?<![0-9]):{line}(?![0-9])")
    else:
        pats = (rf"\bline {line}, (?:col|column) {col}(?![0-9])", rf"\bline {line} col {col}(?![0-9])", rf"(?<![0-9]){line}:{col}(?![0-9])")
    return any(re.search(p, text) for p in pats)


def closure_symbol(row: dict) -> str:
    """The symbol that a host-path row of a closure key states in its note (symbol `name`: ...), else "".
    The closure check of the host reports by symbol, not by line: such a row asserts the key and the symbol."""
    if row.get("status") != "host" or not row["key"].startswith("closure."):
        return ""
    m = re.match(r"symbol `([A-Za-z_][A-Za-z0-9_]*)`: ", row.get("note", ""))
    return m.group(1) if m else ""


def judge_row(row: dict, obs: Observation) -> str:
    """"" when the compiler's behaviour is the row's verdict; else the reason (the row says what the rule must do)."""
    first = next(iter(obs.err.strip().splitlines()), "")[:120]
    if row["verdict"] == "admit":
        if obs.rc != 0 or not obs.elf:
            return f"admit row: compile rc={obs.rc} image={'yes' if obs.elf else 'no'}: {first}"
        if row["exit"] and obs.image_exit != int(row["exit"]):
            return f"image exit {obs.image_exit}, want {row['exit']}"
        return ""
    if obs.rc != 3 or obs.elf:
        return f"refuse row: compile rc={obs.rc} image={'yes' if obs.elf else 'no'}, want rc=3 and no image"
    named = [ln for ln in obs.err.splitlines() if names_key(ln, row["key"])]
    if not named:
        return f"stderr does not name {row['key']}: {first}"
    if symbol := closure_symbol(row):
        return "" if any(f"`{symbol}`" in ln for ln in named) else f"stderr names {row['key']} but not the symbol `{symbol}`"
    if row["line"] and not any(position_ok(ln, int(row["line"]), int(row["col"]) if row["col"] else None) for ln in named):
        return f"stderr names {row['key']} but not line {row['line']}" + (f" column {row['col']}" if row["col"] else "")
    return ""


def judge_beyond(obs: Observation) -> str:
    """"" when the program one nesting level past a ceiling row meets the compiler's stack budget refusal (rc 3, no
    image): the row then sits at the ceiling of the RUNNING compiler, else the reason."""
    if obs.rc != 3 or obs.elf or "stack budget" not in obs.err:
        first = next(iter(obs.err.strip().splitlines()), "")[:120]
        return f"one level past the ceiling: compile rc={obs.rc} image={'yes' if obs.elf else 'no'}, want the stack budget refusal: {first}"
    return ""


def printed_key(err: str, keys: dict) -> str:
    """The first key of the vocabulary that stderr names (earliest position), else none."""
    hits = [(m.start(), k) for k in keys for m in [re.search(r"(?<![A-Za-z0-9_.])" + re.escape(k) + r"(?![A-Za-z0-9_])", err)] if m]
    return min(hits)[1] if hits else "none"


def _carrier_bytes(text: str) -> bytes:
    """The carrier module appended to the std blob: comment lines (a leading #) are notes, not source."""
    kept = [ln for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    return ("\n".join(kept) + "\n").encode() if kept else b""


def _row_feed(std: bytes, program: bytes, profile: str) -> bytes:
    source = std + program
    head = struct.pack("<qq", len(std), len(source))
    return (struct.pack("<qqqq", LAYOUT_MAGIC, *BOUNDED_LAYOUT) if profile == "bounded" else b"") + head + source


def observe_rows(compiler: str, srows: StringRows, ids: list[str], profiles=PROFILES, run_images: bool = True,
                 jobs: int = 4, repo: pathlib.Path = _REPO, timeout: int = 600) -> dict:
    """{(id, profile): Observation}: each program to `compiler` on stdin; an admit row with an `exit` also runs its image."""
    from concurrent.futures import ThreadPoolExecutor
    from _exec_retry import exec_run
    std0 = capture_bundle(repo).std_prefix
    by_id = {r["id"]: r for r in srows.rows}
    work = [(i, p) for i in ids for p in profiles if p in ROW_PROFILES[by_id[i]["profile"]]]
    with tempfile.TemporaryDirectory() as td:
        def one(item):
            i, profile = item
            std = std0 + _carrier_bytes(srows.carriers.get(i, ""))
            try:
                r = start_child([compiler], _row_feed(std, srows.programs[i], profile), capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return item, Observation(124, False, f"compiler timeout ({timeout} s)")
            obs = Observation(r.returncode, r.stdout[:4] == b"\x7fELF", r.stderr.decode("utf-8", "replace"))
            if run_images and obs.elf and by_id[i]["verdict"] == "admit" and by_id[i]["exit"]:
                exe = pathlib.Path(td) / f"{i}.{profile}"
                exe.write_bytes(r.stdout)
                exe.chmod(0o755)
                try:
                    rc = exec_run([str(exe)], capture_output=True, stdin=subprocess.DEVNULL, timeout=20).returncode
                    obs.image_exit = rc if rc >= 0 else 128 - rc  # killed by a signal: the shell's 128 + signal
                except (subprocess.TimeoutExpired, OSError):  # a hang or an image that does not start
                    obs.image_exit = None
            return item, obs
        with ThreadPoolExecutor(jobs) as pool:
            return dict(pool.map(one, work))


def _label(compiler: str, repo: pathlib.Path = _REPO) -> str:
    p = pathlib.Path(compiler).resolve()
    return p.relative_to(repo).as_posix() if repo in p.parents else p.name


def _failures(srows: StringRows, ids: list[str], obs: dict) -> dict:
    by_id, bad = {r["id"]: r for r in srows.rows}, {}
    for (i, profile), o in sorted(obs.items(), key=lambda kv: ids.index(kv[0][0])):
        if why := judge_row(by_id[i], o):
            bad.setdefault(i, []).append(f"[{profile}] {why}")
    return bad


def _run_rows(srows: StringRows, only: set | None) -> tuple[list[str], list[dict]]:
    """(ids to run, skipped rows) of the selected rows; a row whose status is not `run` is counted, never run.
    A `host` row (its key belongs to the check of the host, not to the compiler) is in neither: see _host_rows."""
    chosen = [r for r in srows.rows if only is None or r["id"] in only]
    return [r["id"] for r in chosen if r["status"] == "run"], [r for r in chosen if r["status"] not in ("run", "host")]


def _host_rows(srows: StringRows, only: set | None = None) -> list[dict]:
    """The rows of status `host`: their key is named by the existing check of the host (`mindc build`), not by the
    compiler fed directly. They are run only when a mindc binary is given, and have their OWN count."""
    return [r for r in srows.rows if r["status"] == "host" and (only is None or r["id"] in only)]


def _skipped_text(skipped: list[dict]) -> str:
    counts = collections.Counter(r["status"] for r in skipped)
    return f"skipped={len(skipped)} (" + " ".join(f"{k}={counts[k]}" for k in ROW_STATUS_VERDICTS if k not in ("run", "host")) + ")"


def observe_host(mindc: str, compiler: str, srows: StringRows, rows: list[dict], profiles=PROFILES) -> dict:
    """{(id, profile): Observation} of the host path: `mindc build` of the program as a file, with the compiler
    ELF as the native compiler (MINDC_NATIVE_ELF). The host adds the std modules itself."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, LC_ALL="C", MINDC_NATIVE_ELF=compiler)
        for r in rows:
            src = pathlib.Path(td) / (r["id"] + ".mind")
            src.write_bytes(srows.programs[r["id"]])
            for profile in profiles:
                if profile not in ROW_PROFILES[r["profile"]]:
                    continue
                exe = pathlib.Path(td) / f"{r['id']}.{profile}"
                try:
                    p = subprocess.run([mindc, "build", str(src), "--target", "cpu", "--emit", "binary", "--no-cache", "--out", str(exe),
                                        "--backend", "native", "--native-profile", profile], env=env, capture_output=True, timeout=600, cwd=td)
                    out[(r["id"], profile)] = Observation(p.returncode, exe.is_file(), p.stderr.decode("utf-8", "replace"))
                except subprocess.TimeoutExpired:
                    out[(r["id"], profile)] = Observation(124, False, "mindc timeout (600 s)")
    return out


def _beyond_failures(compiler: str, srows: StringRows, ids: list[str], jobs: int = 4) -> dict:
    """{id: reasons} of the ceiling rows whose @beyond record does not meet the stack budget refusal of `compiler`."""
    bad = {}
    for (i, profile), o in sorted(observe_rows(compiler, replace(srows, programs=srows.beyond), ids, run_images=False, jobs=jobs).items()):
        if why := judge_beyond(o):
            bad.setdefault(i, []).append(f"[{profile}] {why}")
    return bad


def _seed_failures(seed: str, srows: StringRows, ids: list[str]) -> dict:
    """The rows of a ceiling record and their @beyond records on the default-profile seed (4 GiB stack, a deeper ceiling
    than the bounded compiler): one compile at a time, each holds about 3 GiB."""
    if not ids:
        return {}
    if not pathlib.Path(seed).is_file():
        return {i: [f"no default-profile seed at {seed}"] for i in ids}
    d = at_ceiling(srows, seed)
    bad = _failures(d, ids, observe_rows(seed, d, ids, jobs=1))
    for i, why in _beyond_failures(seed, d, [i for i in ids if i in d.beyond], jobs=1).items():
        bad.setdefault(i, []).extend(why)
    return bad


def string_rows_assert(compiler: str, srows: StringRows, mindc: str | None = None, only: set | None = None, seed: str | None = None) -> int:
    """Assert every runnable row (of `only`, when given): 0 all pass, 1 a failure, 2 nothing ran. The `host` rows run through
    `mindc build` when a mindc binary is given; else they are counted as "host path, not run", which is a failure.
    With `seed` (the default-profile compiler), the ceiling rows also run on it, at its own ceiling."""
    ids, skipped = _run_rows(srows, only)
    host = _host_rows(srows, only)
    bad = _failures(srows, ids, observe_rows(compiler, srows, ids))
    past = [i for i in ids if i in srows.beyond]
    for i, why in _beyond_failures(compiler, srows, past).items():
        bad.setdefault(i, []).extend(why)
    ceil_ids = [i for i in ids if i in srows.ceiling_recipes] if seed else []
    deep = _seed_failures(seed, srows, ceil_ids)
    host_bad = _failures(srows, [r["id"] for r in host], observe_host(mindc, compiler, srows, host)) if mindc else {}
    verdicts = {r["id"]: r["verdict"] for r in srows.rows}
    n = {v: (sum(verdicts[i] == v for i in ids), sum(verdicts[i] == v for i in bad)) for v in ("admit", "refuse")}
    for i, why in bad.items():
        print(f"  FAIL  {i}: {'; '.join(why)}")
    for i, why in host_bad.items():
        print(f"  FAIL  {i} (host path): {'; '.join(why)}")
    for i, why in deep.items():
        print(f"  FAIL  {i} (default-profile seed): {'; '.join(why)}")
    for i in ids:
        if i not in bad:
            print(f"  PASS  {i}")
    print(f"STRING-ROWS compiler={_label(compiler)} rows={len(ids) + len(skipped) + len(host)} run={len(ids)} {_skipped_text(skipped)} "
          f"admit: pass={n['admit'][0] - n['admit'][1]} fail={n['admit'][1]} refuse: pass={n['refuse'][0] - n['refuse'][1]} "
          f"fail={n['refuse'][1]} failing={len(bad)}")
    if mindc:
        print(f"STRING-ROWS host path (mindc build) rows={len(host)} pass={len(host) - len(host_bad)} fail={len(host_bad)}: "
              + " ".join(r["id"] for r in host))
    else:
        print(f"STRING-ROWS host path, not run: {len(host)} (no --string-rows-mindc): " + " ".join(r["id"] for r in host))
    if ceil_ids:
        print(f"STRING-ROWS ceiling rows on the default-profile seed {_label(seed)} ceiling="
              f"{nesting_ceiling(elf_stack(seed)) if pathlib.Path(seed).is_file() else 'none'}: run={len(ceil_ids)} pass={len(ceil_ids) - len(deep)} fail={len(deep)}")
    print(f"STRING-ROWS failing ids ({len(bad) + len(host_bad) + len(deep)}): " + " ".join([*bad, *host_bad, *(f"{i}@seed" for i in deep)]))
    return 1 if bad or host_bad or deep or (host and not mindc) else 0 if ids else 2


def string_rows_baseline(compiler: str, srows: StringRows, out: str | None) -> int:
    """Record, assert nothing: id, profile, compile exit status, key printed (or none), image written."""
    ids = [r["id"] for r in srows.rows]
    obs = observe_rows(compiler, srows, ids, run_images=False)
    table = [f"# string-rule acceptance rows: what compiler {_label(compiler)} does with every row; nothing is asserted",
             f"# compiler_sha256 {hashlib.sha256(pathlib.Path(compiler).read_bytes()).hexdigest()}",
             "# columns: id, profile, rc (compile exit status), key (first vocabulary key on stderr, else none), image (written: yes/no)",
             "id\tprofile\trc\tkey\timage"]
    table += [f"{i}\t{p}\t{o.rc}\t{printed_key(o.err, srows.keys)}\t{'yes' if o.elf else 'no'}" for (i, p), o in
              sorted(obs.items(), key=lambda kv: (ids.index(kv[0][0]), PROFILES.index(kv[0][1])))]
    pathlib.Path(out).write_text("\n".join(table) + "\n") if out else print("\n".join(table))
    rcs = collections.Counter(o.rc for o in obs.values())
    print(f"STRING-ROWS-BASELINE compiler={_label(compiler)} observed={len(obs)} " + " ".join(f"rc{k}={v}" for k, v in sorted(rcs.items()))
          + f" keys_printed={sum(printed_key(o.err, srows.keys) != 'none' for o in obs.values())}" + (f" table={out}" if out else ""))
    return 0 if obs else 2


def string_rows_mutant(compiler: str, srows: StringRows, arm: str) -> tuple[int, list]:
    """A compiler built with rule arm `arm` switched off must fail every runnable row of that arm."""
    mut = next((m for m in srows.mutants if m["arm"] == arm), None)
    if mut is None:
        print(f"BLOCKED: no mutant arm {arm!r}; arms: {' '.join(m['arm'] for m in srows.mutants)}")
        return 2, []
    ids, skipped = _run_rows(srows, set(mut["rows"]))
    bad = _failures(srows, ids, observe_rows(compiler, srows, ids))
    alive = [i for i in ids if i not in bad]
    print(f"STRING-ROWS-MUTANT arm={arm} compiler={_label(compiler)} rows={len(mut['rows'])} run={len(ids)} {_skipped_text(skipped)} "
          f"red={len(ids) - len(alive)} not_red={len(alive)}" + (": " + " ".join(alive) if alive else ""))
    return (1 if alive else 0 if ids else 2), alive


def add_string_rows_args(ap) -> None:
    ap.add_argument("--string-rows", metavar="STAGE1", help="assert the string-rule acceptance rows on a compiler ELF")
    ap.add_argument("--string-rows-mindc", metavar="MINDC", help="with --string-rows: run the `host` rows (keys of the host's own check) "
                    "through this mindc binary; without it they are counted as 'host path, not run'")
    ap.add_argument("--string-rows-baseline", metavar="STAGE1", help="record what a compiler does with every row (--out TABLE)")
    ap.add_argument("--string-rows-mutant", nargs=2, metavar=("ARM", "STAGE1"), help="rows of ARM must fail on a compiler with that arm off")
    ap.add_argument("--string-rows-selftest", metavar="STAGE1", help="the runner must be able to fail (STAGE1 has no string rule)")


def string_rows_main(args) -> int | None:
    """None when no string-rows mode was asked for."""
    mutant = args.string_rows_mutant or (None, None)
    compiler = args.string_rows or args.string_rows_baseline or args.string_rows_selftest or mutant[1]
    if not compiler:
        return None
    if not pathlib.Path(compiler).is_file():
        print(f"BLOCKED: no compiler ELF at {compiler}")
        return 2
    compiler = str(pathlib.Path(compiler).resolve())
    try:
        srows = at_ceiling(load_string_rows(), compiler)
    except (ValueError, OSError) as e:
        print(f"BLOCKED: the rows are not valid: {e}")
        return 2
    if args.string_rows_selftest:
        return string_rows_selftest(compiler)
    if args.string_rows_baseline:
        return string_rows_baseline(compiler, srows, args.out)
    mindc = args.string_rows_mindc
    if mindc and not pathlib.Path(mindc).is_file():
        print(f"BLOCKED: no mindc binary at {mindc}")
        return 2
    mindc = str(pathlib.Path(mindc).resolve()) if mindc else None
    seed = str((_REPO / FROZEN_SEED).resolve())
    return string_rows_mutant(compiler, srows, mutant[0])[0] if mutant[1] else string_rows_assert(compiler, srows, mindc, seed=None if seed == compiler else seed)


def _selftest_pure(check) -> StringRows:
    """Checks that need no compiler: the judgement of a row on known outputs; scratch copies of the rows with one planted
    defect each, rejected by name; the runner modes against stand-in compilers."""
    import argparse, contextlib, io, shutil  # noqa: E401
    srows = load_string_rows()
    base = _REPO / STRING_ROWS_DIR
    for c in _tsv_rows(base / "selftest_judge.tsv", ("name", "verdict", "exit", "key", "line", "col", "rc", "image", "stderr", "image_exit", "ok")):
        o = Observation(int(c["rc"]), c["image"] == "yes", c["stderr"].replace("\\n", "\n"), int(c["image_exit"]) if c["image_exit"] else None)
        check(f"judge: {c['name']}", (judge_row(c, o) == "") == (c["ok"] == "yes"), judge_row(c, o))
    clo = {"verdict": "refuse", "exit": "", "key": "closure.std_symbol_collision", "line": "", "col": "", "status": "host", "note": "symbol `f_x`: a note"}
    for name, o, want in (("key and symbol, no line", Observation(3, False, "`f_x` collides with `f_x` (closure.std_symbol_collision)"), True),
                          ("other symbol", Observation(3, False, "`f_y` collides (closure.std_symbol_collision)"), False),
                          ("longer symbol", Observation(3, False, "`f_xy` collides (closure.std_symbol_collision)"), False),
                          ("symbol on other stderr line", Observation(3, False, "`f_x`\n(closure.std_symbol_collision)"), False),
                          ("other key", Observation(3, False, "`f_x` (closure.missing_entry_root)"), False),
                          ("compiled", Observation(0, True, "`f_x` (closure.std_symbol_collision)"), False)):
        check(f"judge: closure host row: {name}", (judge_row(clo, o) == "") == want, judge_row(clo, o))
    # the rows files: the shipped ones load; a scratch copy with one planted defect is rejected by name
    ids, skipped = _run_rows(srows, None)
    budget = "error[backend-native]: program exceeds the native compiler's stack budget (too large or nested too deeply)"
    for name, o, want in (("the stack budget refusal", Observation(3, False, budget), True), ("admitted", Observation(0, True, ""), False),
                          ("refused for another reason", Observation(3, False, "error[backend-native]: unsupported construct"), False),
                          ("an image with rc 3", Observation(3, True, budget), False)):
        check(f"judge: one level past a ceiling row: {name}", (judge_beyond(o) == "") == want, judge_beyond(o))
    check("a ceiling row of status run has its @beyond record", any(i in srows.beyond for i in ids), str(sorted(srows.beyond)))
    check("the nesting ceiling of the bounded stack, the default stack, a 64 MiB stack and stack 0 (the fallback)",
          (nesting_ceiling(0x18000000), nesting_ceiling(0x100000000), nesting_ceiling(0x4000000), nesting_ceiling(0)) == (18022, 208076, 1638, 18022))
    n = collections.Counter(r["status"] for r in srows.rows)
    check("rows load; skipped rows are counted, only run rows run, a host row is neither", n["run"] > 0 and srows.mutants and len(ids) == n["run"]
          and len(ids) + len(skipped) + n["host"] == len(srows.rows) and _skipped_text(skipped) == (
        f"skipped={len(skipped)} (carrier={n['carrier']} by_run={n['by_run']} capacity={n['capacity']} outside={n['outside']})"), _skipped_text(skipped))
    refuse = next(r for r in srows.rows if r["verdict"] == "refuse" and r["status"] == "run" and r["line"])
    row = "\t".join(refuse.values())
    sub = {"id": refuse["id"], "key": refuse["key"], "line": refuse["line"], "row": row, "rowc": row.replace("\trun\t", "\tcarrier\t", 1),
           "adm": next(r["id"] for r in srows.rows if r["verdict"] == "admit" and r["status"] == "run"),
           "arm": srows.mutants[0]["arm"], "arm2": srows.mutants[1]["arm"], "r0": srows.mutants[0]["rows"][0]}
    for pl in _tsv_rows(base / "selftest_plants.tsv", ("name", "file", "old", "new", "want")):
        old, new = pl["old"], pl["new"]
        for k, v in {"\\t": "\t", "\\n": "\n", **{f"<{k}>": v for k, v in sub.items()}}.items():
            old, new = old.replace(k, v), new.replace(k, v)
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(base, pathlib.Path(td) / "d")
            f = pathlib.Path(td) / "d" / pl["file"]
            text = f.read_bytes().decode()
            f.write_bytes(text.replace(old, new, 1).encode())
            try:
                load_string_rows(root=pathlib.Path(td), rows_dir="d")
                check(f"rejects {pl['name']}", False, "loaded without error" if f.read_bytes().decode() != text else "the plant did not change the file")
            except ValueError as e:
                check(f"rejects {pl['name']}", pl["want"] in str(e), str(e)[:100])
    clo_row = next(r for r in srows.rows if closure_symbol(r))
    clo_head = f"{clo_row['id']}\trefuse\t{clo_row['key']}\t"
    for name, edit in (
            ("closure host row with a line", lambda t: t.replace(clo_head, clo_head + "1", 1)),
            ("closure host row without its symbol", lambda t: t.replace(f"\tsymbol `{closure_symbol(clo_row)}`: ", "\t", 1)),
    ):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(base, pathlib.Path(td) / "d")
            f = pathlib.Path(td) / "d" / "rows.tsv"
            text = f.read_bytes().decode()
            f.write_bytes(edit(text).encode())
            try:
                load_string_rows(root=pathlib.Path(td), rows_dir="d")
                check(f"rejects {name}", False, "loaded without error" if f.read_bytes().decode() != text else "the plant did not change the file")
            except ValueError as e:
                check(f"rejects {name}", "states the symbol" in str(e), str(e)[:100])
    # four stand-ins: a compiler that refuses and reports whether the feed began with the bounded layout header, one that
    # accepts (its image does not start), one that never finishes; and a host binary that names no key
    with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
        stand = [pathlib.Path(td) / n for n in ("refuse", "accept", "sleep", "host")]
        for f, body in zip(stand, (f"sys.stderr.write('layout=%d' % d.startswith({struct.pack('<qqqq', LAYOUT_MAGIC, *BOUNDED_LAYOUT)!r}))\nsys.exit(3)",
                                   "sys.stdout.buffer.write(b'\\x7fELF')", "import time\ntime.sleep(60)", None)):
            f.write_text(f"#!{sys.executable}\n" + (f"import sys\nd = sys.stdin.buffer.read()\n{body}\n" if body else ""))
            f.chmod(0o755)
        both = next(r["id"] for r in srows.rows if r["status"] == "run" and r["verdict"] == "admit" and not r["exit"] and r["profile"] == "both")
        dflt = next(r["id"] for r in srows.rows if r["profile"] == "default")
        bnd = next(r["id"] for r in srows.rows if r["profile"] == "bounded")
        got = {k: o.err for k, o in observe_rows(str(stand[0]), srows, [both, dflt, bnd]).items()}
        check("a row runs once per profile it names, the bounded feed starts with the layout header", got == {
            (both, "default"): "layout=0", (both, "bounded"): "layout=1", (dflt, "default"): "layout=0", (bnd, "bounded"): "layout=1"}, str(got))
        got = set(observe_host(str(stand[3]), str(stand[1]), srows, [r for r in srows.rows if r["id"] in (dflt, bnd)]))
        check("a host row runs once per profile it names", got == {(dflt, "default"), (bnd, "bounded")}, str(got))
        ex = next(r["id"] for r in srows.rows if r["status"] == "run" and r["verdict"] == "admit" and r["exit"] and r["profile"] == "both")
        check("a compiler that does not finish in time reads as rc 124 on each profile",
              {k: o.rc for k, o in observe_rows(str(stand[2]), srows, [both], timeout=1).items()} == {(both, "default"): 124, (both, "bounded"): 124})
        check("an image that does not start has no exit value",
              {k: o.image_exit for k, o in observe_rows(str(stand[1]), srows, [ex]).items()} == {(ex, "default"): None, (ex, "bounded"): None})
        check("--string-rows-baseline exits 2: no row", string_rows_baseline(str(stand[1]), replace(srows, rows=[]), None) == 2)
        for name, s, only, want in (("a failing row", 0, {both}, 1), ("every row passes", 1, {both}, 0), ("no row runs", 1, set(), 2)):
            rc = string_rows_assert(str(stand[s]), srows, only=only)
            check(f"--string-rows exits {want}: {name}", rc == want, f"rc={rc}")
        ceil = next(i for i in ids if i in srows.beyond)
        flat = replace(srows, rows=[{**r, "exit": ""} if r["id"] == ceil else r for r in srows.rows])
        for name, s, want in (("a ceiling row whose next level is admitted", flat, 1), ("the same row with no @beyond record", replace(flat, beyond={}), 0)):
            rc = string_rows_assert(str(stand[1]), s, only={ceil})
            check(f"--string-rows exits {want}: {name}", rc == want, f"rc={rc}")
        big = pathlib.Path(td) / "big"  # two PT_LOADs: the highest vaddr has the 4 GiB stack, the other a larger memsz
        big.write_bytes(b"\x7fELF\x02" + bytes(27) + struct.pack("<Q", 64) + bytes(14) + struct.pack("<HH", 56, 2) + bytes(6)
                        + struct.pack("<IIQQQQQQ", 1, 6, 0, 0x41001000, 0, 0, 0x100000000, 0) + struct.pack("<IIQQQQQQ", 1, 5, 0, 0x400000, 0, 0, 0x200000000, 0))
        deep = at_ceiling(srows, str(big)).beyond[ceil]
        check("a ceiling record follows the stack of the running compiler", b"(" * 208077 in deep and b"(" * 208078 not in deep, str(len(deep)))
        cs = pathlib.Path(td) / "ceil"  # admits a program, refuses one with the stack budget past the bounded ceiling
        cs.write_text(f"#!{sys.executable}\nimport sys\nd = sys.stdin.buffer.read()\nif b'(' * {nesting_ceiling(0) + 1} in d:\n"
                      "    sys.stderr.write('stack budget')\n    sys.exit(3)\nsys.stdout.buffer.write(b'\\x7fELF')\n")
        cs.chmod(0o755)
        for name, seed, want in (("the ceiling row on the compiler and on the default-profile seed", str(cs), 0),
                                 ("the default-profile seed admits one level past its ceiling", str(stand[1]), 1),
                                 ("no default-profile seed", "/no/such/elf", 1)):
            rc = string_rows_assert(str(cs), flat, only={ceil}, seed=seed)
            check(f"--string-rows exits {want}: {name}", rc == want, f"rc={rc}")
        nob = next(i for i in ids if i in srows.ceiling_recipes and i not in srows.beyond)
        flat2 = replace(srows, rows=[{**r, "exit": ""} if r["id"] == nob else r for r in srows.rows])
        for name, seed, want in (("a ceiling row with no @beyond record on the seed", str(cs), 0),
                                 ("a ceiling row with no @beyond record that the seed refuses", str(stand[0]), 1)):
            rc = string_rows_assert(str(cs), flat2, only={nob}, seed=seed)
            check(f"--string-rows exits {want}: {name}", rc == want, f"rc={rc}")
        hrow = next(r["id"] for r in srows.rows if r["status"] == "host")
        rc = string_rows_assert(str(stand[1]), srows, str(stand[3]), {both, hrow})
        check("--string-rows exits 1: only a host row fails", rc == 1, f"rc={rc}")
        rc = string_rows_assert(str(stand[1]), srows, None, {both, hrow})
        check("--string-rows exits 1: a host row is not run (no --string-rows-mindc)", rc == 1, f"rc={rc}")
        status = {r["id"]: r["status"] for r in srows.rows}
        for arm in (next(m["arm"] for m in srows.mutants if all(status[i] != "run" for i in m["rows"])), "no_such_arm"):
            check(f"--string-rows-mutant exits 2: {arm}", string_rows_mutant(str(stand[1]), srows, arm)[0] == 2)
        args = dict.fromkeys(("string_rows_mindc", "string_rows_baseline", "string_rows_selftest", "string_rows_mutant", "out"))
        check("a missing compiler exits 2", string_rows_main(argparse.Namespace(string_rows="/no/such/elf", **args)) == 2)
        # the CI command: --string-rows sets the rows to the ceiling of the compiler and runs the seed leg on the
        # default-profile seed (FROZEN_SEED), except when the compiler is that seed; string_rows_assert is a recorder here
        g, calls, keep = globals(), [], (globals()["string_rows_assert"], globals()["FROZEN_SEED"])
        g["string_rows_assert"] = lambda c, s, mindc=None, only=None, seed=None: calls.append((c, b"(" * 208076 in s.programs[ceil]
                                                                                              and b"(" * 208077 in s.beyond[ceil], seed)) or 0
        g["FROZEN_SEED"] = str(stand[1])
        try:
            for comp in (big, stand[1]):
                string_rows_main(argparse.Namespace(string_rows=str(comp), **args))
        finally:
            g["string_rows_assert"], g["FROZEN_SEED"] = keep
        want = [(str(big.resolve()), True, str(stand[1].resolve())), (str(stand[1].resolve()), False, None)]
        check("--string-rows: rows at the ceiling of the compiler, the seed leg on the default-profile seed", calls == want, str(calls)[:160])
    return srows


def _selftest_mutants(check) -> None:
    """The runner's own guards: with one of them switched off in a copy of this source, the checks above must go red."""
    import types
    src = pathlib.Path(__file__).read_text().split("def _selftest_mutants")[0]  # the mutated code only, never this table
    for name, old, new in (
            ("key vocabulary", 'if r["verdict"] == "refuse" and r["key"] not in keys:', "if False:"),
            ("refuse exit status", "if obs.rc != 3 or obs.elf:", "if obs.elf:"),
            ("image exit value", 'if row["exit"] and obs.image_exit != int(row["exit"]):', "if False:"),
            ("refused position", 'if row["line"] and not any(', "if False and not any("),
            ("closure symbol", "if symbol := closure_symbol(row):", "if (symbol := closure_symbol(row)) and False:"),
            ("closure row form", 'if r["status"] == "host" and r["key"].startswith("closure.") and (', "if False and ("),
            ("whole-token key", '+ r"(?![A-Za-z0-9_])", text) is not None', ", text) is not None"),
            ("orphan program", "if orphans:", "if False:"),
            ("beyond judgement", 'if obs.rc != 3 or obs.elf or "stack budget" not in obs.err:', "if obs.elf:"),
            ("beyond record owner", "    if stray:", "    if False:"),
            ("beyond run", "past = [i for i in ids if i in srows.beyond]", "past = []"),
            ("carrier record", 'if (r["status"] == "carrier" and i not in carriers) or (i in carriers and r["status"] not in ("carrier", "run")):', "if False:"),
            ("carrier record of a row that is not fed", ' or (i in carriers and r["status"] not in ("carrier", "run")):', ":"),
            ("README counts", "    if stale_counts:", "    if False:"),
            ("profile vocabulary", 'if r["profile"] not in ROW_PROFILES:', "if False:"),
            ("bounded-only rows", '"bounded": ("bounded",)}', '"bounded": PROFILES}'),
            ("host row profiles", 'if profile not in ROW_PROFILES[r["profile"]]:', "if False:"),
            ("skip count", '[r for r in chosen if r["status"] not in ("run", "host")]', "[]"),
            ("run only run rows", '[r["id"] for r in chosen if r["status"] == "run"]', '[r["id"] for r in chosen]'),
            ("host rows are not skipped rows", '[r for r in chosen if r["status"] not in ("run", "host")]', '[r for r in chosen if r["status"] != "run"]'),
            ("admit image", "if obs.rc != 0 or not obs.elf:", "if obs.rc != 0:"), ("refuse image", "if obs.rc != 3 or obs.elf:", "if obs.rc != 3:"),
            ("mode exit status", "return 1 if bad or host_bad or deep or (host and not mindc) else 0 if ids else 2", "return 0 if ids else 2"),
            ("host failure exit status", "return 1 if bad or host_bad or deep or (host and not mindc) else 0 if ids else 2",
             "return 1 if bad or deep or (host and not mindc) else 0 if ids else 2"),
            ("seed failure exit status", "return 1 if bad or host_bad or deep or (host and not mindc) else 0 if ids else 2",
             "return 1 if bad or host_bad or (host and not mindc) else 0 if ids else 2"),
            ("host rows not run exit status", "return 1 if bad or host_bad or deep or (host and not mindc) else 0 if ids else 2",
             "return 1 if bad or host_bad or deep else 0 if ids else 2"),
            ("seed run", "deep = _seed_failures(seed, srows, ceil_ids)", "deep = {}"),
            ("seed rows", "ceil_ids = [i for i in ids if i in srows.ceiling_recipes] if seed else []", "ceil_ids = [i for i in ids if i in srows.beyond] if seed else []"),
            ("ceiling of the running compiler", "c = nesting_ceiling(elf_stack(compiler))", "c = nesting_ceiling(0)"),
            ("stack below the fallback", "stack if stack > 0 else STACK_FLOOR", "max(stack, STACK_FLOOR)"),
            ("ceiling of the --string-rows compiler", "srows = at_ceiling(load_string_rows(), compiler)", "srows = load_string_rows()"),
            ("seed of --string-rows", "seed=None if seed == compiler else seed", "seed=None"),
            ("no seed leg on the seed itself", "seed=None if seed == compiler else seed", "seed=seed"),
            ("highest PT_LOAD", "max(((p[3], p[6]) for p", "max(((p[6], p[6]) for p"),
            ("profile list", "profiles=PROFILES, run_images", 'profiles=("default",), run_images'),
            ("default-only rows", 'p in ROW_PROFILES[by_id[i]["profile"]]]', "True]"),
            ("bounded header", 'if profile == "bounded" else b"")', 'if False else b"")'),
            ("mutant arm without a run row", "return (1 if alive else 0 if ids else 2), alive", "return (1 if alive else 0), alive"),
            ("mutant arm unknown", "        return 2, []", "        return 0, []"),
            ("baseline exit status", "return 0 if obs else 2", "return 0"), ("compiler timeout", "Observation(124, False,", "Observation(0, False,")):
        mod = sys.modules["_selftest_mutant"] = types.ModuleType("_selftest_mutant")
        mod.__file__ = __file__
        exec(compile(src.replace(old, new, 1), "_selftest_mutant", "exec"), mod.__dict__)
        red = []
        mod._selftest_pure(lambda n, ok, d="": red.append(n) if not ok else None)
        check(f"mutant {name}: turns a check red", bool(red), ", ".join(red[:2]) or "no check went red")


def string_rows_selftest(compiler: str) -> int:
    """The runner must be able to fail: the pure checks and the runner mutants need no compiler, the rest needs one WITHOUT the rule."""
    results: list[tuple[str, bool, str]] = []
    check = lambda name, ok, detail="": results.append((name, bool(ok), detail))  # noqa: E731

    srows = _selftest_pure(check)
    _selftest_mutants(check)
    # against the compiler: red on a refuse row, a planted wrong exit reported, a mutant that is not red is reported
    ids = [r["id"] for r in srows.rows if r["status"] == "run"]
    rr = next(r["id"] for r in srows.rows if r["id"] in ids and r["verdict"] == "refuse")
    bad = _failures(srows, [rr], observe_rows(compiler, srows, [rr]))
    check("rule absent: a refuse row is red", bool(bad), "; ".join(sum(bad.values(), [])))
    sig = _failures(srows, ["lim_alloc_over"], observe_rows(compiler, srows, ["lim_alloc_over"]))
    check("an image killed by a signal reads as 128 + signal", not sig, str(sig)[:100])
    exits = [r for r in srows.rows if r["id"] in ids and r["verdict"] == "admit" and r["exit"]][:12]
    obs = observe_rows(compiler, srows, [r["id"] for r in exits])
    good = [r for r in exits if not _failures(srows, [r["id"]], {k: v for k, v in obs.items() if k[0] == r["id"]})]
    check("control: an admit row with an exit passes", bool(good), f"{len(good)} of {len(exits)}")
    if good:
        planted = {**good[0], "exit": str((int(good[0]["exit"]) + 1) % 256)}
        why = [judge_row(planted, o) for (i, _), o in obs.items() if i == good[0]["id"]]
        check("planted wrong exit is reported", bool(why) and all("want " + planted["exit"] in w for w in why), "; ".join(why)[:100])
    by_id = {r["id"]: r for r in srows.rows}
    killed = next((m for m in srows.mutants if any(by_id[i]["status"] == "run" for i in m["rows"])
                   and all(by_id[i]["verdict"] == "refuse" for i in m["rows"] if by_id[i]["status"] == "run")), None)
    alive = next((m for m in srows.mutants if any(by_id[i]["status"] == "run" and by_id[i]["verdict"] == "admit" for i in m["rows"])), None)
    check("mutant arms exist for both checks", bool(killed and alive), "needs an arm of refuse rows only and one with an admit row")
    if killed and alive:
        check("mutant of refuse rows is red", string_rows_mutant(compiler, srows, killed["arm"])[0] == 0, killed["arm"])
        check("mutant with a green admit row is reported", string_rows_mutant(compiler, srows, alive["arm"])[0] == 1, alive["arm"])
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or not detail else f": {detail}"))
    failed = [n for n, ok, _ in results if not ok]
    print(f"STRING-ROWS-SELFTEST checks={len(results)} fail={len(failed)}")
    return 1 if failed else 0
