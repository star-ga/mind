"""Two-compiler identity of the native-lane smokes (`--identity-vs`), its partitions and their self-tests,
and the dispatch of the compiler-only modes of native_lane_value_shapes_smoke.py. Split out of
_selfhost_sources.py (source capture and the compile helpers stay there)."""
import ast
import collections
import errno
import os
import pathlib
import re
import struct
import subprocess
import tempfile
import time

from _selfhost_sources import COMPILER_UNIT_DRIVERS  # drivers that call compiler fns: compiler source, in self mode
from _selfhost_sources import _REPO, SELF_MAGIC, capture_bundle, core_limit_check, core_limit_one, direct_compile_cli, mint_stage2, self_mode_image, start_child
from _selfhost_string_rows import _tsv_rows, add_string_rows_args, load_string_rows, string_rows_main
from _w8_native import _libc_admitted, group_accrows_admitted, group_kernel_change, group_kernel_reason  # admission rows; kinds 87-89


def corpus_programs(path: pathlib.Path) -> list[tuple[str, str]]:
    """(id, source) of each `=== <id> | ...` record of a native-lane corpus file."""
    out: list[tuple[str, list[str]]] = []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("=== "):
            out.append((line[4:].split(" | ", 1)[0], []))
        elif out:
            out[-1][1].append(line)
    return [(name, "\n".join(body) + "\n") for name, body in out]


# Two programs the lane frame names (a raw line feed and an unknown escape inside a string literal): they ride in every
# identity run, so the rename rule below is exercised (the reference gives the generic text, the frame one lane line).
RENAMED_ROWS = (("frame/raw_newline", 'fn main() -> i64 {\n    print("ab\ncd");\n    return 0;\n}\n'),
                ("frame/escape", 'fn main() -> i64 {\n    print("a\\qb");\n    return 0;\n}\n'))
# The refusals the lane frame names: the reference prints the generic text, the candidate one lane line.
_RENAMED = re.compile(rb"^error\[native-lane\]: the program is outside candidate native-lane-v1 "
                      rb"\(lane\.string\.(?:raw_newline|escape|literal_capacity)\)[^\n]*\n$")


def _renamed_refusal(ref: tuple, cand: tuple) -> bool:
    return (ref[0] == cand[0] == 3 and ref[1] == cand[1] == b"" and b"unsupported construct" in ref[2]
            and _RENAMED.match(cand[2]) is not None)


def string_literals(source: str) -> int:
    """String literals of a source: a `"` outside a `//` comment opens one; an unescaped `"` or the
    end of the line closes it (the lexer ends a literal at its line). The ABI string of `extern "C"`
    is not a literal expression and is not counted. Counts from the text alone."""
    count = 0
    for line in source.splitlines():
        i = 0
        while i < len(line) and not line.startswith("//", i):
            if line[i] == '"':
                count += not re.search(r"\bextern\s*$", line[:i])  # `extern "C"` names an ABI: no handle
                i += 1
                while i < len(line) and line[i] != '"':
                    i += 2 if line[i] == "\\" else 1
            i += 1
    return count


def elf_note(image: bytes) -> bytes:
    """The bytes of the PT_NOTE segment (the 32-byte trace hash lives there); b"" when there is none."""
    phoff, phnum = struct.unpack_from("<Q", image, 0x20)[0], struct.unpack_from("<H", image, 0x38)[0]
    for i in range(phnum):
        p_type, _flags, p_off, _va, _pa, p_filesz = struct.unpack_from("<IIQQQQ", image, phoff + i * 56)
        if p_type == 4:
            return image[p_off:p_off + p_filesz]
    return b""


def run_image(image: bytes, timeout: int = 20, args: tuple = (), env: dict | None = None) -> tuple[int, bytes] | None:
    """(exit status, stdout) of an image run with `args` (default none), no stdin and the soft core limit 1 (core_limit_one:
    a trap starts no dump handler); None on a timeout. `env` is added to the parent environment (a row's own variables)."""
    penv = None if not env else {**os.environ, **{str(k): str(v) for k, v in env.items()}}
    with tempfile.TemporaryDirectory() as td:
        exe = pathlib.Path(td) / "image"
        exe.write_bytes(image)
        exe.chmod(0o755)
        for attempt in range(8):
            try:
                r = subprocess.run([str(exe), *args], stdin=subprocess.DEVNULL, capture_output=True,
                                   timeout=timeout, env=penv, preexec_fn=core_limit_one())
                return r.returncode, r.stdout
            except subprocess.TimeoutExpired:
                return None
            except OSError as e:  # ETXTBSY: another thread forked while the file was open for writing
                if e.errno != errno.ETXTBSY or attempt == 7:
                    raise
                time.sleep(0.2)
    return None


def _partition_compare(source: dict[str, str], pairs: list, std_literals: int, jobs: int, min_images: int) -> int:
    """String-literal identity. A literal-free program keeps the same image; a program with a literal
    gets a different image that runs to the same exit and stdout. A refusal must repeat.
    0 = all hold, 1 = a violation, 2 = nothing compared."""
    from concurrent.futures import ThreadPoolExecutor
    is_image = lambda r: r[0] == 0 and r[1][:4] == b"\x7fELF"
    bad, same, changed, refused = [], [], [], 0
    for name, ref, cand in pairs:
        lits = string_literals(source[name])
        if ref[0] < 0 or cand[0] < 0:
            bad.append(f"{name}: crash ref={ref[0]} cand={cand[0]}")
        elif not is_image(ref) and not is_image(cand):
            refused += 1
            if ref != cand:
                bad.append(f"{name}: refusal differs: ref rc={ref[0]} cand rc={cand[0]}")
        elif is_image(ref) != is_image(cand):
            bad.append(f"{name}: ref {'image' if is_image(ref) else 'refusal'}, cand {'image' if is_image(cand) else 'refusal'}")
        elif (ref[1] == cand[1]) == (lits == 0):
            (same if lits == 0 else changed).append((name, lits, ref[1], cand[1]))
        else:
            bad.append(f"{name}: {lits} string literal(s) in the text, image {'unchanged' if lits else 'changed'}")
    with ThreadPoolExecutor(jobs) as pool:
        runs = list(pool.map(lambda c: (run_image(c[2]), run_image(c[3])), changed))
    equal = unrun = 0
    for (name, lits, ref_image, cand_image), (ref_run, cand_run) in zip(changed, runs):
        if elf_note(ref_image) != elf_note(cand_image):
            bad.append(f"{name}: the PT_NOTE differs (the literal handle must not move the trace hash)")
            verdict = "NOTE DIFFERS"
        elif ref_run is None and cand_run is None:
            unrun += 1
            verdict = "unrun (both time out)"
        elif ref_run != cand_run:
            bad.append(f"{name}: ref run {ref_run if ref_run is None else ref_run[0]} cand run {cand_run if cand_run is None else cand_run[0]}, output differs")
            verdict = "RUN DIFFERS"
        else:
            equal += 1
            verdict = f"run equal exit={ref_run[0]} stdout={len(ref_run[1])}B"
        print(f"  DIFF  {name}: {lits} literal(s), image {len(ref_image)} -> {len(cand_image)} B, {verdict}")
    for name, _, image, _ in same:
        print(f"  SAME  {name}: no literal, image {len(image)} B")
    for b in bad:
        print("  FAIL  " + b)
    print(f"PARTITION programs={len(pairs)} literal_free_same={len(same)} literal_bearing_changed={len(changed)} "
          f"run_equal={equal} unrun={unrun} refusals_equal={refused} violations={len(bad)} std_blob_literals={std_literals}")
    return 1 if bad else 2 if len(same) + len(changed) < min_images or not changed else 0


def has_runtime_prologue(image: bytes) -> bool:
    """True when the image starts, at its ELF entry, with the allocator runtime blob of the default
    layout: `movabs r9, ST` (49 B9 <imm64>) with ST = arena base + 1 GiB - 4096. A BUMP image does not."""
    if image[:4] != b"\x7fELF":
        return False
    entry, phoff = struct.unpack_from("<Q", image, 0x18)[0], struct.unpack_from("<Q", image, 0x20)[0]
    for i in range(struct.unpack_from("<H", image, 0x38)[0]):
        p_type, _f, p_off, p_va, _pa, p_filesz = struct.unpack_from("<IIQQQQ", image, phoff + i * 56)
        if p_type == 1 and p_va <= entry < p_va + p_filesz:
            at = p_off + entry - p_va
            return image[at:at + 10] == b"\x49\xb9" + struct.pack("<Q", 0x100_0000 + (1 << 30) - 4096)
    return False


def _alloc_partition_compare(pairs: list, admitted: set[str], jobs: int, min_images: int, quiet: bool = False,
                             runner=None) -> tuple[int, str]:
    """Allocator identity: an image may differ only when the candidate starts with the runtime
    prologue and the reference does not (the program frees at a loop position, so the candidate
    chose RECLAIM). Every other image is byte-identical and every refusal repeats. A differing
    image runs to the reference exit and output, or, when the reference dies on a signal, to any
    exit of the candidate. Returns (0 | 1 | 2, census line)."""
    from concurrent.futures import ThreadPoolExecutor
    is_image = lambda r: r[0] == 0 and r[1][:4] == b"\x7fELF"
    bad, same, diff, refused = [], [], [], 0
    for name, ref, cand in pairs:
        if ref[0] < 0 or cand[0] < 0:
            bad.append(f"{name}: crash ref={ref[0]} cand={cand[0]}")
        elif not is_image(ref) and not is_image(cand):
            refused += 1
            if ref != cand:
                bad.append(f"{name}: refusal differs: ref rc={ref[0]} cand rc={cand[0]}")
        elif is_image(ref) != is_image(cand):
            bad.append(f"{name}: ref {'image' if is_image(ref) else 'refusal'}, cand {'image' if is_image(cand) else 'refusal'}")
        elif ref[1] == cand[1]:
            same.append(name)
            if has_runtime_prologue(cand[1]):
                bad.append(f"{name}: identical image that starts with the runtime prologue (an explicit-layout image)")
        elif has_runtime_prologue(cand[1]) and not has_runtime_prologue(ref[1]):
            diff.append((name, ref[1], cand[1]))
        else:
            bad.append(f"{name}: image differs ({len(ref[1])} -> {len(cand[1])} B) and the candidate does not start with the runtime prologue")
    with ThreadPoolExecutor(jobs) as pool:
        runs = list(pool.map(lambda d: ((runner or (lambda i: run_image(i, 180)))(d[1]), (runner or (lambda i: run_image(i, 180)))(d[2])), diff))
    equal = base_died = 0
    for (name, _ref, _cand), (ref_run, cand_run) in zip(diff, runs):
        if cand_run is None:
            bad.append(f"{name}: the candidate image times out")
            verdict = "CAND TIMEOUT"
        elif ref_run == cand_run:
            equal += 1
            verdict = f"run equal exit={cand_run[0]} stdout={len(cand_run[1])}B"
        elif ref_run is None or ref_run[0] < 0:
            base_died += 1
            verdict = f"reference died (rc {None if ref_run is None else ref_run[0]}), candidate exit={cand_run[0]}"
        else:
            bad.append(f"{name}: ref run exit={ref_run[0]}, cand run exit={cand_run[0]}")
            verdict = "RUN DIFFERS"
        if not quiet:
            print(f"  DIFF  {name}: prologue image, {verdict}")
    census = sorted(n for n, _, _ in diff if n in admitted)
    if not quiet:
        for b in bad:
            print("  FAIL  " + b)
        print(f"PARTITION programs={len(pairs)} identical={len(same)} prologue_changed={len(diff)} run_equal={equal} "
              f"reference_died={base_died} refusals_equal={refused} violations={len(bad)}")
        print(f"CENSUS admitted programs of the request corpus that select RECLAIM: {len(census)} {census}")
    return (1 if bad else 2 if len(same) + len(diff) < min_images else 0), f"{len(census)} {census}"


def _prologue_elf() -> bytes:
    """A synthetic image whose entry starts with the allocator runtime prologue (has_runtime_prologue)."""
    prologue = bytearray(0x200)
    prologue[:4] = b"\x7fELF"
    struct.pack_into("<Q", prologue, 0x18, 0x400078)           # e_entry
    struct.pack_into("<Q", prologue, 0x20, 0x40)               # e_phoff
    struct.pack_into("<H", prologue, 0x38, 1)                  # e_phnum
    struct.pack_into("<IIQQQQ", prologue, 0x40, 1, 5, 0, 0x400000, 0, 0x200)
    prologue[0x78:0x82] = b"\x49\xb9" + struct.pack("<Q", 0x100_0000 + (1 << 30) - 4096)
    return bytes(prologue)


def _alloc_partition_selftest() -> bool:
    """The partition rule must catch a planted stray difference and a prologue on an identical image."""
    elf = b"\x7fELF" + bytes(0x100)
    prologue = _prologue_elf()
    ok = lambda b: (0, b, b"")
    stray = [("t/stray", ok(elf), ok(elf + b"\x01"))]                     # a plain image that differs
    allowed = [("t/allowed", ok(elf), ok(bytes(prologue)))]                 # a prologue image
    same_prologue = [("t/same", ok(bytes(prologue)), ok(bytes(prologue)))]  # identical but explicit-layout
    fake = lambda image: (0, b"")  # the synthetic images are never executed
    return (_alloc_partition_compare(stray, set(), 1, 0, True, fake)[0] == 1
            and _alloc_partition_compare(same_prologue, set(), 1, 0, True, fake)[0] == 1
            and _alloc_partition_compare(allowed, set(), 1, 0, True, fake)[0] == 0)


_LANE_KEY = re.compile(rb"^error\[native-lane\]: [^\n]*\(lane\.[a-z0-9_.]+\)[^\n]*\n$")
_KEY_NAME = re.compile(r"lane\.[a-z0-9_.]*[a-z0-9_]")
_AT_POS = re.compile(r"at line (\d+), column (\d+)")
# The rows of the two frame programs above state their key here (their names say it).
FRAME_KEYS = {"frame/raw_newline": "lane.string.raw_newline", "frame/escape": "lane.string.escape"}


def _alloc_row_exits(root: pathlib.Path) -> tuple[dict[str, tuple[int, tuple]], dict[str, str]]:
    """(exits, classes) from the smoke's assignments only. classes are RECLAIM or BUMP of DEFAULT_ROWS."""
    tree = ast.parse((root / "examples/mindc_mind/self_host_native_alloc_bounded_smoke.py").read_text())
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(x, "id", "") in ("_R", "CASES", "DEFAULT_ROWS") for x in n.targets)]
    space: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "allocator rows", "exec"), space)
    bounded = {f"examples/mindc_mind/testdata/native_alloc_bounded/{n}.mind": (v[0], ()) for n, v in space["CASES"].items()}
    default = {f"examples/mindc_mind/testdata/native_alloc_default/{n}.mind": (v[1], v[3]) for n, v in space["DEFAULT_ROWS"].items()}
    classes = {f"examples/mindc_mind/testdata/native_alloc_default/{n}.mind": v[0] for n, v in space["DEFAULT_ROWS"].items()}
    return bounded | default, classes


def _add_keys(keys: dict, pos: dict, name: str, blob: str) -> None:
    """Record the lane keys of one row text, and the line and column when that text states them."""
    found = _KEY_NAME.findall(blob or "")
    if not found:
        return
    keys[name].update(found)
    at = _AT_POS.search(blob)
    if at:
        pair = (int(at.group(1)), int(at.group(2)))
        for key in found:
            pos[name].setdefault(key, set()).add(pair)


def _row_facts(root: pathlib.Path, source: dict[str, str]) -> dict:
    """Keys, positions, libc admissions, allocator exits and classes, and the std prefix, from row data."""
    mind = root / "examples/mindc_mind"
    keys: dict[str, set[str]] = collections.defaultdict(set)
    pos: dict[str, dict] = collections.defaultdict(dict)
    rows = ast.literal_eval((mind / "testdata/native_libc_rows/ROWS.txt").read_text())
    for r in rows:
        if r.get("fixture") and r.get("refusal"):
            _add_keys(keys, pos, f"examples/mindc_mind/{r['fixture']}", r["refusal"])
    for r in _tsv_rows(mind / "native_libc_abi_rows.tsv", ("name", "kind", "ret", "fixture", "code", "selfhost")):
        _add_keys(keys, pos, f"examples/mindc_mind/testdata/native_libc_binding/{r['fixture']}", r["selfhost"])
    for line in (root / "tests/fixtures/native_lane/request_corpus.txt").read_text().splitlines():
        if line.startswith("=== ") and " | refused=" in line:
            _add_keys(keys, pos, "corpus/" + line[4:].split(" | ")[0], line.split(" | ")[1])
    _add_keys(keys, pos, "examples/mindc_mind/testdata/native_libc_io/generated_source.mind",
              (mind / "testdata/native_libc_io/identity_row.txt").read_text())
    srows = load_string_rows(root)
    by_text = {srows.programs[r["id"]].decode().strip(): r["key"] for r in srows.rows if r["key"] and r["id"] in srows.programs}
    for name, text in source.items():
        keys[name].update(k for k in [by_text.get(text.strip()), FRAME_KEYS.get(name)] if k)
    libc, admit = _libc_admitted(rows + group_accrows_admitted())
    exits = {"corpus/" + line[4:].split(" | ")[0]: (int(line.split(" | ")[1][5:]), ())
             for line in (root / "tests/fixtures/native_lane/request_corpus.txt").read_text().splitlines()
             if line.startswith("=== ") and line.split(" | ")[1].startswith("exit=")}
    alloc, alloc_class = _alloc_row_exits(root)
    frozen = {n: {k: frozenset(v) for k, v in d.items()} for n, d in pos.items()}
    return {"libc": libc, "admit": admit, "keys": {n: frozenset(k) for n, k in keys.items()}, "pos": frozen,
            "alloc": exits | alloc, "alloc_class": alloc_class,
            "std": capture_bundle(root).std_prefix.decode(errors="replace")}


# The classes of the merged identity partition. Every program of a pair list is in exactly one.
MERGED_CLASSES = ("identical", "literal handle", "alloc prologue", "named refusal", "refusal equal",
                  "newly admitted", "refused by its row", "group kernel")
# A string-family key needs no row when the line the report names holds the construct of that key.
_STRING_CONSTRUCT = {"lane.string.result": re.compile(r"->\s*String\b"),
                     "lane.string.binding": re.compile(r"\blet\s+(?:mut\s+)?\w+\s*(?::\s*String\b|=\s*\")"),
                     "lane.string.const": re.compile(r"\bconst\s+\w+\s*:\s*String\b")}
_AT_LINE = re.compile(rb" at line (\d+), column \d+ of the composed user image")
_STD_FN = re.compile(rb"\) in fn `([A-Za-z0-9_]+)` in the std region\.")  # a std site: the line that declares the fn
_CALLS = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _strip_code(text: str) -> str:
    """Source with // comments and string literals removed, so a brace walk sees code."""
    out = []
    for line in text.splitlines():
        cut = line.split("//", 1)[0]
        out.append(re.sub(r'"(?:\\.|[^"\\])*"', '""', cut))
    return "\n".join(out)


def _fn_bodies(text: str) -> dict[str, str]:
    """{fn name: body} of `text`. A large text (the std prefix) is cached; user texts are not."""
    cache = _fn_bodies.__dict__.setdefault("_c", {})
    if len(text) > 10000 and text in cache:
        return cache[text]
    code, bodies = _strip_code(text), {}
    for m in re.finditer(r"(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        i = code.find("{", m.end())
        if i < 0:
            continue
        depth, j = 0, i
        while j < len(code):
            depth += (code[j] == "{") - (code[j] == "}")
            if depth == 0:
                bodies[m.group(1)] = code[i + 1:j]
                break
            j += 1
    if len(text) > 10000:
        cache[text] = bodies
    return bodies


def _hot_fragments(body: str) -> list[str]:
    """A while condition and body, and a for body. A for bound runs once, before the loop."""
    hot, i = [], 0
    while True:
        m = re.search(r"\b(while|for)\b", body[i:])
        if not m:
            return hot
        kind, start = m.group(1), i + m.end()
        brace = body.find("{", start)
        if brace < 0:
            return hot
        depth, j = 1, brace + 1
        while j < len(body) and depth:
            depth += (body[j] == "{") - (body[j] == "}")
            j += 1
        if kind == "while":
            hot.append(body[start:brace])
        hot.append(body[brace + 1:j - 1])
        i = j


def _reaches(user: str, std: str, target: str) -> bool:
    """True when `target` is reachable from main, or from every user fn when there is no main."""
    ub, sb = _fn_bodies(user), _fn_bodies(std) if std else {}
    stack, seen = (["main"] if "main" in ub else list(ub)), set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        body = ub.get(name, sb.get(name, ""))
        for call in _CALLS.findall(body):
            if call == target:
                return True
            if call not in seen and (call in ub or call in sb):
                stack.append(call)
    return False


def _frees_at_loop(user: str, std: str) -> bool:
    """True when __mind_free is at a loop position, or in a user or std fn reached from one."""
    ub = _fn_bodies(user)
    hot = [frag for body in ub.values() for frag in _hot_fragments(body)]
    if not hot:
        return False
    blob = "\n".join(hot)
    if "__mind_free" in blob:
        return True
    sb = _fn_bodies(std) if std else {}
    stack, seen = list(_CALLS.findall(blob)), set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        body = ub.get(name, sb.get(name))
        if body is None:
            continue
        if "__mind_free" in body:
            return True
        stack.extend(_CALLS.findall(body))
    return False


def _bump_directive(text: str) -> bool:
    """True when the leading comment block of the user region says `native_alloc = bump`."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("//"):
            return False
        if re.search(r"\bnative_alloc\s*=\s*bump\b", s):
            return True
    return False


def _prologue_ok(name: str, text: str, facts: dict) -> bool:
    """RECLAIM, or a free at a loop position when neither the row nor the leading directive says BUMP."""
    cls = (facts.get("alloc_class") or {}).get(name)
    if cls == "RECLAIM":
        return True
    if cls == "BUMP" or _bump_directive(text):
        return False
    return _frees_at_loop(text, facts.get("std", ""))


def _accepts_key(name: str, cand_key: str, err: bytes, stated: frozenset, facts: dict) -> bool:
    """The candidate key is one the row states, and at a position the row states when the row states one."""
    if cand_key not in stated:
        return False
    poses = (facts.get("pos") or {}).get(name, {}).get(cand_key)
    if not poses:
        return True
    at = _AT_POS.search(err.decode(errors="replace"))
    return at is not None and (int(at.group(1)), int(at.group(2))) in poses


def _construct_key(text: str, err: bytes, key: str, std: str = "") -> frozenset:
    """{key} when the named line holds the string construct, and a std site only when that fn is reached."""
    at, fn, lines = _AT_LINE.search(err), _STD_FN.search(err), text.splitlines()
    if fn and not at:
        called = fn[1].decode()
        decl = re.compile(r"\s*(?:pub\s+)?fn\s+" + re.escape(called) + r"\s*\(")
        lines, at = [ln for ln in std.splitlines() if decl.match(ln)][:1], [b"", b"1"]
        if not lines or not _reaches(text, std, called):
            return frozenset()
    if key not in _STRING_CONSTRUCT or not at or not 0 < int(at[1]) <= len(lines):
        return frozenset()
    return frozenset({key}) if _STRING_CONSTRUCT[key].search(lines[int(at[1]) - 1]) else frozenset()


def _merged_class_of(name: str, text: str, ref: tuple, cand: tuple, facts: dict) -> tuple[str | None, str]:
    """(class, why). None is a violation. A class accepts a difference only for the reason it names."""
    is_image = lambda r: r[0] == 0 and r[1][:4] == b"\x7fELF"
    lits = string_literals(text)
    cand_key = (_KEY_NAME.findall(cand[2].decode(errors="replace")) or [""])[0] if _LANE_KEY.match(cand[2]) else ""
    stated = facts["keys"].get(name, frozenset()) | _construct_key(text, cand[2], cand_key, facts.get("std", ""))
    if ref[0] < 0 or cand[0] < 0:
        return None, f"crash ref={ref[0]} cand={cand[0]}"
    if not is_image(ref) and not is_image(cand):
        if ref == cand:
            return "refusal equal", ""
        if ref[0] == cand[0] == 3 and ref[1] == cand[1] == b"" and cand_key:
            if _accepts_key(name, cand_key, cand[2], stated, facts):
                return "named refusal", cand[2].decode(errors="replace").strip()
            return None, (f"candidate key {cand_key}, its row states {sorted(stated) or 'no lane key'}; reference: "
                          + ref[2].decode(errors="replace").strip()[:70])
        return None, f"refusal differs: ref rc={ref[0]} cand rc={cand[0]}"
    if is_image(ref) and not is_image(cand):
        if cand[0] == 3 and cand[1] == b"" and _accepts_key(name, cand_key, cand[2], stated, facts):
            return "refused by its row", cand[2].decode(errors="replace").strip()
        return None, f"ref image, cand refusal {cand_key or cand[2][:60]!r}; its row states {sorted(stated) or 'no lane key'}"
    if not is_image(ref):
        if name in facts["libc"] and ref[0] == 3 and ref[1] == b"" and b"unsupported construct" in ref[2]:
            return "newly admitted", f"candidate image {len(cand[1])} B"
        return None, "ref refusal, cand image"
    ref_pro, cand_pro = has_runtime_prologue(ref[1]), has_runtime_prologue(cand[1])
    if ref[1] == cand[1]:
        if lits:
            return None, f"{lits} string literal(s) in the text, image unchanged"
        return ("identical", "") if not cand_pro else (None, "identical image that starts with the runtime prologue")
    if lits and cand_pro == ref_pro:
        return "literal handle", f"{lits} literal(s)"
    if not lits and cand_pro and not ref_pro and _prologue_ok(name, text, facts):
        return "alloc prologue", ""
    if not lits and cand_pro == ref_pro and group_kernel_change(text):
        return "group kernel", "kind 87 / 88 probe-skip contract"
    if lits:
        return None, f"{lits} literal(s) AND a prologue change: in two classes"
    return None, f"image differs ({len(ref[1])} -> {len(cand[1])} B) and the candidate does not start with the runtime prologue"


def _merged_partition(source: dict[str, str], pairs: list, facts: dict, jobs: int, min_images: int,
                      quiet: bool = False, runner=None) -> int:
    """One partition. Each differing program is in one class; images run equal (or the reference dies and
    the candidate exits as its row states); a newly admitted image runs under its row environment.
    0 = no violation, 1 = a violation, 2 = fewer than `min_images` images compared."""
    from concurrent.futures import ThreadPoolExecutor
    run = runner or (lambda image, args=(), env=None: run_image(image, 180, args, env))

    def row_exit(name: str, image: bytes, plain) -> bool:  # the candidate ends as its allocator row states
        want, args = facts["alloc"].get(name, (None, ()))
        got = run(image, args) if args else plain
        return want is not None and got is not None and got[0] == want
    members: dict[str, list] = {c: [] for c in MERGED_CLASSES}
    bad: list[str] = []
    for name, ref, cand in pairs:
        cls, why = _merged_class_of(name, source[name], ref, cand, facts)
        if cls is None:
            bad.append(f"{name}: {why}")
        else:
            members[cls].append((name, why, ref, cand))
    diff = [("literal handle", *m) for m in members["literal handle"]] + [("alloc prologue", *m) for m in members["alloc prologue"]]
    diff += [("group kernel", *m) for m in members["group kernel"]]  # both images run; exit status equal, stdout may differ
    with ThreadPoolExecutor(jobs) as pool:
        runs = list(pool.map(lambda m: (run(m[3][1]), run(m[4][1])), diff))
    equal = died = 0
    for (cls, name, _why, ref, cand), (ref_run, cand_run) in zip(diff, runs):
        if cls == "literal handle" and elf_note(ref[1]) != elf_note(cand[1]):
            bad.append(f"{name}: the PT_NOTE differs (the literal handle must not move the trace hash)")
            verdict = "NOTE DIFFERS"
        elif ref_run is None or cand_run is None:
            bad.append(f"{name}: an image did not finish (ref {'timeout' if ref_run is None else ref_run[0]}, "
                       f"cand {'timeout' if cand_run is None else cand_run[0]})")
            verdict = "UNRUN"
        elif ref_run == cand_run:
            equal += 1
            verdict = "run equal" if ref_run is None else f"run equal exit={ref_run[0]} stdout={len(ref_run[1])}B"
        elif cls == "group kernel" and cand_run[0] == ref_run[0] == 0:
            verdict = f"exit 0 both, stdout {len(ref_run[1])} -> {len(cand_run[1])} B (the value-shape row checks the bytes)"
        elif cls == "group kernel" and (reason := group_kernel_reason(name, ref_run, cand_run)):
            verdict = f"named reason: {reason}; ref exit={ref_run[0]}, cand exit={cand_run[0]} stdout={len(cand_run[1])} B = its row"
        elif ref_run[0] < 0 and row_exit(name, cand[1], cand_run):
            died += 1
            verdict = f"reference died (signal {-ref_run[0]}), candidate exit = its row {facts['alloc'][name]}"
        else:
            bad.append(f"{name}: ref run exit={ref_run[0]} stdout={len(ref_run[1])} B, cand run exit={cand_run[0]} stdout={len(cand_run[1])} B")
            verdict = "RUN DIFFERS"
        if not quiet:
            print(f"  DIFF  [{cls}] {name}: image {len(ref[1])} -> {len(cand[1])} B, {verdict}")
    for name, _why, _ref, cand in members["newly admitted"]:
        spec = (facts.get("admit") or {}).get(name)
        if not spec:
            bad.append(f"{name}: newly admitted with no exit in its row")
            continue
        for want, env, args, out in spec:  # each admission row: its arguments, its exit, its stdout when it states one
            got = run(cand[1], args, env)
            if got is None or got[0] != want or out is not None and got[1] != out:
                bad.append(f"{name}: admitted image {list(args)} exit {None if got is None else got[0]}, its row states {want}")
    if not quiet:
        for name, why, *_ in members["named refusal"]:
            print(f"  NAMED {name}: {why}")
        for name, why, *_ in members["newly admitted"]:
            print(f"  ADMIT {name}: {why}")
        for name, why, *_ in members["refused by its row"]:
            print(f"  ROWREF {name}: {why}")
        for b in bad:
            print("  FAIL  " + b)
        bad_names = {b.split(": ", 1)[0] for b in bad}
        for c in MERGED_CLASSES:
            for name, _why, *_rest in members[c]:
                if name not in bad_names:
                    print(f"  PASS  {name}")
        for c in MERGED_CLASSES:
            print(f"CLASS {c}: {len(members[c])}")
    violations = len(bad)
    images = len(members["identical"]) + len(diff)
    if not quiet:
        print(f"PARTITION programs={len(pairs)} run_equal={equal} reference_died={died} of {len(diff)} violations={violations}")
    return 1 if violations else 2 if images < min_images or not diff else 0


def _note_elf(note: bytes, mark: int) -> bytes:
    """Synthetic ELF whose only program header is a PT_NOTE. `mark` differs the image, not the note."""
    blob = bytearray(0x80 + len(note))
    blob[:4] = b"\x7fELF"
    struct.pack_into("<Q", blob, 0x20, 0x40)
    struct.pack_into("<H", blob, 0x38, 1)
    struct.pack_into("<IIQQQQ", blob, 0x40, 4, 4, 0x78, 0, 0, len(note))
    blob[0x78:0x78 + len(note)] = note
    blob[-1] = mark
    return bytes(blob)


def _row_shape_ok() -> bool:
    """expect_exit rows are admitted, refusal rows are not, and the identity row is the unknown-symbol site."""
    mind = _REPO / "examples/mindc_mind"
    table = ast.literal_eval((mind / "testdata/native_libc_rows/ROWS.txt").read_text())
    libc, _admit = _libc_admitted(table)
    refused = [f"examples/mindc_mind/{r['fixture']}" for r in table if r.get("refusal") and not r.get("front_door")]
    admitted = [f"examples/mindc_mind/{r['fixture']}" for r in table if "expect_exit" in r and not r.get("front_door")]
    if not refused or any(n in libc for n in refused) or any(n not in libc for n in admitted):
        return False
    name = "examples/mindc_mind/testdata/native_libc_io/generated_source.mind"
    keys: dict[str, set] = collections.defaultdict(set)
    pos: dict[str, dict] = collections.defaultdict(dict)
    _add_keys(keys, pos, name, (mind / "testdata/native_libc_io/identity_row.txt").read_text())
    return "lane.extern.unknown_symbol" in keys[name] and (1, 17) in pos[name].get("lane.extern.unknown_symbol", ())


def _merged_selftest() -> bool:
    """A clean set passes, and removing any class guard makes one planted case fail."""
    elf = lambda tag: b"\x7fELF" + bytes(0x100) + tag
    pro, ok = _prologue_elf(), lambda b: (0, b, b"")
    pro_eq, pro_b = bytearray(pro), bytearray(pro)
    pro_eq[0x100], pro_b[0x101] = 9, 3
    pro_eq, pro_b = bytes(pro_eq), bytes(pro_b)
    note_a, note_b = _note_elf(b"AAAA", 1), _note_elf(b"BBBB", 2)
    generic = (3, b"", b"error[backend-native]: unsupported construct; use --backend=mlir\n")
    code = (3, b"", b"error[E2039]: extern block: this calling convention has no ABI\n")
    lane = (3, b"", b"error[native-lane]: the program is outside candidate native-lane-v1 (lane.extern.arity) at line 2\n")
    other = (3, b"", b"error[E0001]: no\n")
    runs = {elf(b"a"): (0, b"x"), elf(b"b"): (0, b"x"), elf(b"c"): (0, b"y"), elf(b"d"): (-11, b""), pro: (33, b""),
            elf(b"e"): (33, b""), elf(b"L"): (33, b"x"), pro_eq: (0, b"x"), pro_b: (33, b""),
            note_a: (0, b"q"), note_b: (0, b"q"), elf(b"T"): None, elf(b"U"): None, elf(b"r1"): (1, b"\x01" + bytes(7)),
            elf(b"c0"): (0, bytes(8))}
    msg = b"error[native-lane]: the program is outside candidate native-lane-v1 ("
    keyed = lambda key, line: (3, b"", msg + key + b") in fn `f` at line " + line + b", column 1 of the composed user image.\n")
    at = lambda key, line, col: (3, b"", msg + key + b") at line " + line + b", column " + col + b" of the composed user image.\n")
    sres, ext = "fn f() -> String {\n    1\n}\n", keyed(b"lane.extern.type", b"1")
    std_site = lambda fn: (3, b"", msg + b"lane.string.result) in fn `" + fn + b"` in the std region. No artifact written.\n")
    std_g = "pub fn g(s: String) -> String {\n    1\n}\n"
    std_sha = "pub fn sha(n: i64) -> i64 {\n    let z: i64 = __mind_free(n);\n    return z;\n}\n"

    def fake(image, args=(), env=None):
        if args:
            return (len(args), b"")
        if env is not None:
            return (int(env.get("EXIT", "0")), b"")
        return runs[image] if image in runs else (0, b"")

    lit, plain = 'fn main() { print("a"); }\n', "fn main() { 1 }\n"
    free_loop = "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 2 {\n        let f: i64 = __mind_free(i);\n        i = i + 1;\n    }\n    return i;\n}\n"
    directed = "// mind: native_alloc = bump\n" + free_loop
    for_bound = "fn bound(n: i64) -> i64 {\n    let z: i64 = __mind_free(n);\n    return n;\n}\nfn main() -> i64 {\n    let mut a: i64 = 0;\n    for i in 0..bound(2) {\n        a = a + 1;\n    }\n    return a;\n}\n"
    sha_loop = "fn main() -> i64 {\n    let mut i: i64 = 0;\n    while i < 1 {\n        let r: i64 = sha(i);\n        i = i + 1;\n    }\n    return i;\n}\n"
    lit_loop = 'fn main() -> i64 {\n    while i < 2 {\n        let f: i64 = __mind_free(i);\n    }\n    print("z");\n    return 0;\n}\n'
    cond_free = "fn main() -> i64 {\n    while __mind_free(i) < 2 {\n        i = i + 1;\n    }\n    return i;\n}\n"
    via_g = "fn h() {\n    g()\n}\nfn main() {\n    h()\n}\n"
    rows = lambda keys=None, libc=(), alloc=None, admit=None, pos=None, alloc_class=None, std="": {"libc": frozenset(libc), "keys": keys or {}, "alloc": alloc or {}, "admit": admit or {}, "pos": pos or {}, "alloc_class": alloc_class or {}, "std": std}
    arity = {"named": frozenset({"lane.extern.arity"})}
    exf = rows({"ex": frozenset({"lane.extern.unknown_symbol"})}, pos={"ex": {"lane.extern.unknown_symbol": frozenset({(1, 17)})}})
    admit_ok, kb = rows(libc={"adm"}, admit={"adm": [(42, {"EXIT": "42"}, (), None)]}), "examples/mindc_mind/testdata/native_avx2/k3_bank.mind"
    k87, ref1, cand0 = "fn main() { __mind_group_bank_i64(0, 0, 0, 0, 35) }\n", ok(elf(b"r1")), ok(elf(b"c0"))

    def v(name, text, ref, cand, facts):
        return _merged_partition({name: text}, [(name, ref, cand)], facts, 1, 0, True, fake)
    clean = [("lit", lit, ok(elf(b"a")), ok(elf(b"b"))), ("same", plain, ok(elf(b"a")), ok(elf(b"a"))),
             ("named", plain, generic, lane), ("kept", plain, lane, lane)]
    pairs = [(n, r, c) for n, _, r, c in clean]
    good = _merged_partition({n: t for n, t, _, _ in clean}, pairs, rows(arity), 1, 0, True, fake) == 0
    died = lambda alloc, text=free_loop: v("al", text, ok(elf(b"d")), ok(pro), rows(alloc=alloc))
    cases = [
        ("lit", lit, ok(elf(b"a")), ok(elf(b"c")), rows(), 1), ("stray", plain, ok(elf(b"a")), ok(elf(b"b")), rows(), 1),
        ("lit", lit, ok(elf(b"a")), ok(elf(b"a")), rows(), 1), ("nt", lit, ok(note_a), ok(note_b), rows(), 1),
        ("to", lit, ok(elf(b"T")), ok(elf(b"U")), rows(), 1), ("ip", plain, ok(pro), ok(pro), rows(), 1),
        ("lp", lit, ok(elf(b"a")), ok(pro_eq), rows(), 1), ("rp", free_loop, ok(pro), ok(pro_b), rows(), 1),
        ("adm", plain, generic, ok(elf(b"a")), rows(), 1), ("adm", plain, generic, ok(elf(b"a")), admit_ok, 2),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(7, {"EXIT": "42"}, (), None)]}), 1),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(1, {}, ("8",), b""), (0, {}, (), None)]}), 2),
        ("adm", plain, generic, ok(elf(b"a")), rows(libc={"adm"}, admit={"adm": [(1, {}, ("8",), b"x")]}), 1),
        (kb, k87, ref1, cand0, rows(), 0), (kb[:-5] + "_x.mind", k87, ref1, cand0, rows(), 1), (kb, k87, cand0, ref1, rows(), 1),
        ("adm", plain, other, ok(elf(b"a")), admit_ok, 1), ("named", plain, generic, lane, rows(), 1),
        ("named", plain, code, lane, rows(arity), 2), ("named", plain, ok(elf(b"a")), lane, rows(), 1),
        ("named", plain, ok(elf(b"a")), lane, rows(arity), 2),
        ("ex", plain, generic, at(b"lane.extern.unknown_symbol", b"1", b"17"), exf, 2),
        ("ex", plain, generic, at(b"lane.extern.callconv", b"1", b"17"), exf, 1),
        ("ex", plain, generic, at(b"lane.extern.unknown_symbol", b"9", b"1"), exf, 1),
        ("ex", plain, ok(elf(b"a")), at(b"lane.extern.callconv", b"1", b"1"), exf, 1),
        ("lv", free_loop, ok(elf(b"L")), ok(pro), rows(alloc={"lv": (33, ())}), 1),
        ("al", lit, ok(elf(b"d")), ok(elf(b"e")), rows(alloc={"al": (33, ())}), 0),
        ("s", sres, generic, keyed(b"lane.string.result", b"1"), rows(), 2),
        ("s", sres, generic, keyed(b"lane.string.result", b"2"), rows(), 1),
        ("s", sres, generic, keyed(b"lane.string.binding", b"1"), rows(), 1),
        ("s", plain, generic, std_site(b"g"), {**rows(), "std": std_g}, 1),
        ("s", "fn main() {\n    g()\n}\n", generic, std_site(b"g"), {**rows(), "std": std_g}, 2),
        ("s", "fn other() {\n    g()\n}\nfn main() { 1 }\n", generic, std_site(b"g"), {**rows(), "std": std_g}, 1),
        ("s", plain, generic, std_site(b"g"), {**rows({"s": frozenset({"lane.string.result"})}), "std": std_g}, 2),
        ("s", plain, generic, std_site(b"h"), {**rows(), "std": std_g}, 1),
        ("corpus/w3_extern_x", plain, generic, ext, rows(), 1), ("w3_extern_x", plain, generic, ext, rows(), 1),
        ("nf", plain, ok(elf(b"a")), ok(pro_eq), rows(), 1),
        ("bd", directed, ok(elf(b"d")), ok(pro), rows(alloc={"bd": (33, ())}), 1),
        ("bp", free_loop, ok(elf(b"d")), ok(pro), rows(alloc={"bp": (33, ())}, alloc_class={"bp": "BUMP"}), 1),
        ("fb", for_bound, ok(elf(b"d")), ok(pro), rows(alloc={"fb": (33, ())}), 1),
        ("rk", plain, ok(elf(b"d")), ok(pro), rows(alloc={"rk": (33, ())}, alloc_class={"rk": "RECLAIM"}), 0),
        ("sh", sha_loop, ok(elf(b"d")), ok(pro), rows(alloc={"sh": (33, ())}, std=std_sha), 0),
        ("lf", lit_loop, ok(elf(b"a")), ok(pro_eq), rows(), 1),
        ("wc", cond_free, ok(elf(b"d")), ok(pro), rows(alloc={"wc": (33, ())}), 0),
        ("s", via_g, generic, std_site(b"g"), {**rows(), "std": std_g}, 2),
    ]
    return (good and _row_shape_ok() and all(v(*c[:5]) == c[5] for c in cases)
            and died({"al": (33, ())}) == 0 and died({"al": (2, ("a", "b"))}) == 0
            and died({"al": (34, ())}) == 1 and died({}) == 1)


def _identity_pairs(reference: str, candidate: str, extra, root: pathlib.Path, jobs: int, self_mode: bool):
    """(programs, std prefix, pairs): every program fed to both ELFs (direct_compile's stdin protocol);
    a pair is (name, (rc, stdout, stderr) of the reference, the same of the candidate)."""
    from concurrent.futures import ThreadPoolExecutor
    programs = [("corpus/" + n, s) for n, s in corpus_programs(root / "tests/fixtures/native_lane/request_corpus.txt")]
    programs += [(f.relative_to(root).as_posix(), f.read_text()) for f in sorted((root / "examples/mindc_mind/testdata").glob("native_*/*.mind"))]
    programs += [("smoke/" + n, s) for n, s in extra]
    programs += list(RENAMED_ROWS)
    std = capture_bundle(root).std_prefix

    def feed(compiler: str, source: str, prefix: bytes = b"", unit: bool = False) -> tuple[int, bytes, bytes]:
        data = std + source.encode()
        image = struct.pack("<qq", len(std), len(data)) + data
        r = start_child([compiler], self_mode_image(image, pathlib.Path(compiler)) if unit else prefix + image,
                           capture_output=True, timeout=600)
        return r.returncode, r.stdout, r.stderr

    cand_prefix = SELF_MAGIC if self_mode else b""  # a measurement of the magic, never a fixture default
    with ThreadPoolExecutor(jobs) as pool:  # a compiler-unit driver is compiler source: self mode, through the helper
        pairs = list(pool.map(lambda p: (p[0], feed(reference, p[1], unit=p[0] in COMPILER_UNIT_DRIVERS),
                                         feed(candidate, p[1], cand_prefix, p[0] in COMPILER_UNIT_DRIVERS)), programs))
    return programs, std, pairs


def identity_compare(reference: str, candidate: str | None, extra: list[tuple[str, str]] = (),
                     repo: pathlib.Path = _REPO, jobs: int = 4, min_images: int = 30, self_mode: bool = False) -> int:
    """Two-compiler identity of the request corpus, of testdata/native_*/*.mind and of `extra`.
    An image must repeat byte for byte; a refusal must repeat, except the generic text of a string
    refusal may become its lane key (TEXT); a crash fails. `self_mode` sends MINDSLF to the candidate
    only. 0 = hold, 1 = a difference, 2 = fewer than `min_images` images. `string-literal` uses
    `_partition_compare`, `alloc-prologue` uses `_alloc_partition_compare` after its planted check,
    and `merged` uses `_merged_partition` (0 means violations 0)."""
    if not candidate:
        print("BLOCKED: --identity-vs needs --compiler ELF (the candidate)")
        return 2
    if not core_limit_check(run_image):  # it printed its FAIL line
        return 1
    root = pathlib.Path(repo)
    programs, std, pairs = _identity_pairs(reference, candidate, extra, root, jobs, self_mode)
    if os.environ.get("MIND_IDENTITY_PARTITION") == "merged":
        if not _merged_selftest():
            print("FAIL  the merged partition rule did not catch its planted breaks")
            return 1
        return _merged_partition(dict(programs), pairs, _row_facts(root, dict(programs)), jobs, min_images)
    if os.environ.get("MIND_IDENTITY_PARTITION") == "string-literal":
        return _partition_compare(dict(programs), pairs, string_literals(std.decode()), jobs, min_images)
    if os.environ.get("MIND_IDENTITY_PARTITION") == "alloc-prologue":
        if not _alloc_partition_selftest():
            print("FAIL  the allocator partition rule did not catch its planted stray difference")
            return 1
        header = [line for line in (root / "tests/fixtures/native_lane/request_corpus.txt").read_text().splitlines() if line.startswith("=== ")]
        admitted = {"corpus/" + h[4:].split(" | ")[0] for h in header if not h.split(" | ")[1].startswith("refused=")}
        return _alloc_partition_compare(pairs, admitted, jobs, min_images)[0]
    bad = []
    renamed = []  # rows whose refusal TEXT changed to a lane key (the generic text became a named one)
    groups: dict[str, list[int]] = {}  # source group -> [programs, images_equal, refusals_equal]
    for name, ref, cand in pairs:
        group = groups.setdefault(name.split("/", 1)[0] if name.startswith(("corpus/", "smoke/", "frame/")) else "testdata", [0, 0, 0])
        group[0] += 1
        if ref[0] < 0 or cand[0] < 0:
            bad.append(f"{name}: crash ref={ref[0]} cand={cand[0]}")
        elif ref != cand and _renamed_refusal(ref, cand):
            renamed.append(f"{name}: {cand[2].decode(errors='replace').strip()}")
            group[2] += 1
        elif ref != cand:
            bad.append(f"{name}: ref rc={ref[0]} {len(ref[1])} B, cand rc={cand[0]} {len(cand[1])} B")
        else:
            group[1 if ref[0] == 0 and ref[1][:4] == b"\x7fELF" else 2] += 1
    for r in renamed:
        print("  TEXT  " + r)
    for b in bad:
        print("  FAIL  " + b)
    for g, (n, i, r) in sorted(groups.items()):
        print(f"  {g}: programs={n} images_equal={i} refusals_equal={r}")
    images = sum(g[1] for g in groups.values())
    print(f"IDENTITY programs={len(programs)} images_equal={images} "
          f"refusals_equal={sum(g[2] for g in groups.values())} differ={len(bad)} text_renamed={len(renamed)}")
    return 1 if bad else 2 if images < min_images else 0


def add_selfhost_args(ap) -> None:
    """The compiler-only modes of native_lane_value_shapes_smoke.py (no mindc binary needed)."""
    ap.add_argument("--mint-stage2", metavar="OUT")
    ap.add_argument("--assert-fixed-point", action="store_true")
    ap.add_argument("--direct-compile", metavar="PROG")
    ap.add_argument("--identity-vs", metavar="REF_ELF")
    ap.add_argument("--self-mode", action="store_true")
    ap.add_argument("--out")
    add_string_rows_args(ap)


def selfhost_modes_main(args, extra: list[tuple[str, str]]) -> int | None:
    """Run the compiler-only mode asked for; None when none was (`extra`: more identity programs)."""
    if args.mint_stage2:
        digest = mint_stage2(pathlib.Path(args.mint_stage2), assert_fixed_point=args.assert_fixed_point)
        print(f"MINT stage2 {args.mint_stage2} sha256={digest}" + (" (stage2 == stage3)" if args.assert_fixed_point else ""))
        return 0
    if args.identity_vs:
        return identity_compare(args.identity_vs, args.compiler, extra, self_mode=args.self_mode)
    if args.direct_compile:
        return direct_compile_cli(args.direct_compile, args.compiler, args.out)
    return string_rows_main(args)
