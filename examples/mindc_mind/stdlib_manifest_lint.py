#!/usr/bin/env python3
"""stdlib_manifest_lint.py — machine-checked contract for WHICH std/*.mind
modules each consumer links.

enforces: STDLIB-MANIFEST

This lint IS the test for the STDLIB-MANIFEST rule: delete the manifest read in
`native_bridge.rs::frozen_std_seed_modules` (or let any consumer's membership drift) and
check 3 / check 6 below go red. The `enforced-by:` sites are the readers.

WHY THIS EXISTS
---------------
"Which std modules go into the native backend's std blob" was asserted by
SEVENTEEN hand-copied literal lists — `STD_MODULES` in
`src/build/native_bridge.rs::run_native_backend_bridge` plus sixteen `_STDLIB_MODULES` /
`STD_MODULES` copies across `examples/mindc_mind/*.py` — with nothing executable
asserting they agreed. The bridge's own comment admitted it:

    "This list is the twin of self_host_standalone_driver_smoke.py::_STDLIB_MODULES"
    "deferred: unify both readers on one committed manifest"

A fourth, DIFFERENT list — `STDLIB_MIND_SOURCES` in `src/project/stdlib.rs`, 23
entries — governs general `use std.<m>` resolution and had drifted from the seed
set by two modules (`http`, `sha512`) with no check and no record of whether that
was deliberate.

Two hand-copied lists that must agree, with nothing asserting they do, is the most
common defect class in this tree. `testdata/stdlib_manifest.txt` is the one
committed contract; this lint recomputes every consumer's REAL membership from its
source and fails on drift, IN BOTH DIRECTIONS, with positive counts.

WHAT IS CHECKED
---------------
  1. manifest <-> std/*.mind on disk, BOTH directions (the manifest is exhaustive,
     like `profile_frozen_admits` — a new std module is a hard failure until
     someone records a deliberate in/out decision for each consumer).
  2. the seed blob composed FROM the manifest hashes to the golden recorded in the
     manifest header — a positive pin on the seed SET and its ORDER, so a reseed
     event cannot land as a quiet one-line edit.
  3. the Rust bridge reads the manifest and carries no literal list of its own.
  4. `native_bridge.rs::FROZEN_STD_SEED_COUNT` == `_stdlib_manifest.FROZEN_STD_SEED_COUNT`
     == the manifest's actual seed count.
  5. `STDLIB_MIND_SOURCES` <-> the manifest's `bundled` column, BOTH directions.
  6. every REMAINING hand-copied `_STDLIB_MODULES` / `STD_MODULES` literal in
     `examples/mindc_mind/*.py` equals the manifest seed list exactly.
  7. every count the manifest's comments state (`SEED_COUNT`, `BUNDLED_COUNT`,
     `TOTAL_COUNT`, and the `<n> modules are not `seed`` sentence) appears
     EXACTLY ONCE, in its canonical form, and equals the real row count. A
     three-way merge keeps both sides' count lines; this check refuses that,
     and refuses its own hostile variants of the real header on every run.
  9. every extern name declared in more than one std module has ONE form: the
     same parameter types, in the same order, and the same result type. The
     native compiler binds one row per name and refuses the second declaration
     of a name that carries other types, so two std modules in one image (the
     seed, or one program's std closure) must not disagree. A pointer is a
     pointer type in every module, never `i64` in one and `*mut u8` in another.
     Agreement is not enough: two modules can agree on a wrong `i64`. So every
     such name also has a row in `_C_POINTER_PARAMS`, the parameters its C
     prototype takes as pointers, and each declaration must spell them as
     pointer types. A shared name without a row is an error.

`--self-test` is this lint's own mutation proof (python3 scripts/run_gate.py
examples/mindc_mind/stdlib_manifest_lint.py --self-test). It builds a mini repo from
the lint's REAL file set, damages a copy of the REAL manifest (duplicated header,
wrong count), runs the lint's real entry point on it as a subprocess and requires a
non-zero exit with the count-header message. The in-process probes above call
`count_header_errors` directly, so deleting the one line in `main()` that feeds the
real manifest to it left them green; only a run through the entry point sees that.

Check 6 is the bridge to full migration: those fourteen smokes still carry a
literal, but the literal can no longer drift silently.

  deferred: migrate the remaining literal lists onto `_stdlib_manifest.seed_modules()`
  the way self_host_standalone_driver_smoke.py now does — upgrade path: replace each
  `_STDLIB_MODULES = [...]` with `import _stdlib_manifest` +
  `_STDLIB_MODULES = _stdlib_manifest.seed_modules()`, one smoke per change, each
  verified by re-running that smoke. Not done in one sweep because several of these
  smokes are byte-identity gates that must be individually re-run, and an unrunnable
  bulk edit is exactly the kind of change this lint exists to prevent. Until then
  check 6 holds the invariant.
"""

import ast
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import _stdlib_manifest  # noqa: E402
from _stdlib_manifest import count_header_errors, count_header_probes  # noqa: E402
from _process_starts import PROCESS_START_PLANTS, PROCESS_STARTS, process_start_errors  # noqa: E402
from _selfhost_sources import _start_child_rows  # noqa: E402

_BRIDGE_RS = _REPO / "src" / "build" / "native_bridge.rs"
_STDLIB_RS = _REPO / "src" / "project" / "stdlib.rs"
_NATIVE_IMAGE_RS = _REPO / "src" / "build" / "native_image.rs"
_STD_DIR = _REPO / "std"

# Smokes that legitimately consume the manifest rather than a literal.
_MIGRATED = {"_stdlib_manifest.py", "stdlib_manifest_lint.py"}

_LIST_RE = re.compile(r"^_?STD(?:LIB)?_MODULES\s*=\s*(\[[^\]]*\])", re.M)
MAIN_MIND_LINE_PIN = 67850  # exact: new code goes in a part; re-pin DOWN to the new `wc -l` in the commit that shortens main.mind
_SEED_PRIVATE_PREFIXES = {"arena": ("arena_",), "async": ("async_", "sched_", "snd_"), "blas": ("blas_",), "cli": ("cli_",),
    "fs": ("fs_",), "io": ("io_",), "io_canon": ("canon_",), "iouring": ("io_", "iouring_", "sys_io_uring_"), "map": ("map_",),
    "json": ("jv_", "jvsb_", "jvstr_", "jvpr_"), "net": ("net_",), "process": ("proc_",), "reactor": ("reactor_",),
    "regex": ("rx_",), "ring": ("ring_",), "sha256": ("sha_",), "string": ("string_",), "time": ("time_",),
    "toml": ("toml_", "tv_"), "tui": ("tui_",), "vec": ("vec_",)}


# The input magics of the stage driver (nb_lane_keys.mind): in text, and as the decimal value of the
# constant. Only the files below may hold them (the part that defines them, the driver that reads them,
# the one sender helper, its test, this lint, the docs). Anywhere else a copy would be a second sender.
_MAGIC_TEXT = (b"MINDSLF", b"MINDKEY", b"MINDHOOK", b"91844762919651661", b"97185056536021325",
               b"5426643247806957901")
_MAGIC_FILES = frozenset("examples/mindc_mind/" + n for n in (
    "nb_lane_keys.mind", "nb_lane_rules.mind", "selfhost_driver.mind", "_selfhost_sources.py",
    "_selfhost_identity.py", "_selfhost_string_rows.py", "self_host_native_diag_smoke.py",
    "native_lane_value_shapes_smoke.py", "stdlib_manifest_lint.py", "PARTS.md")
) | frozenset(("tests/native_lane_extern_support/mod.rs",))
_MAGIC_SUFFIXES = (".py", ".mind", ".rs", ".md", ".sh", ".yml", ".yaml", ".toml", ".tsv", ".txt", ".json")


def process_start_selftest(repo=_REPO) -> list[tuple[str, bool]]:
    """(label, ok) rows of the start rule: each plant of PROCESS_START_PLANTS (a file in a scratch checkout, in a table
    row when its column says so), a row with no start, the run-time rows, and the committed tree."""
    lines = (pathlib.Path(repo) / PROCESS_START_PLANTS).read_text(encoding="utf-8").splitlines()
    plants = [line.split("\t") for line in lines[1:] if line]
    rows = [("process start: the plant table has a header and four columns per row",
             lines[:1] == ["name\tflagged\trow\tform\tcode"] and all(len(p) == 5 for p in plants))]

    def plant_file(name: str) -> str:  # a shell plant names its file; a Python plant is <name>.py
        return name if name.endswith(".sh") else f"{name}.py"
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        table = ["file\tfunction\treason", "stale.py\tgone\ta planted row with no start", "empty.py\tg\t",
                 "row_no_stdin.py\tf\ta planted row given twice"]
        (root / "stale.py").write_text("def gone():\n    pass\n", encoding="utf-8")
        for name, _, row, _, code in (p for p in plants if len(p) == 5):
            (root / plant_file(name)).write_text(code.replace("\\n", "\n"), encoding="utf-8")
            table += [f"{plant_file(name)}\tf\ta planted row"] if row == "1" else []
        (root / "table.tsv").write_text("\n".join(table) + "\n", encoding="utf-8")
        for git in (["init", "-q"], ["add", "-A"]):
            subprocess.run(["git", *git], cwd=root, capture_output=True, check=True)
        found = process_start_errors(root, "table.tsv")
    for name, flagged, _, form, _ in (p for p in plants if len(p) == 5):
        hit = any(f.startswith(f"{plant_file(name)}:") for f in found)
        rows.append((f"process start: {form} {'is refused' if flagged == '1' else 'passes'}", hit == (flagged == "1")))
    rows.append(("process start: a table row whose fn starts nothing is refused", any("row stale.py gone" in f for f in found)))
    rows.append(("process start: a table row with no reason, and a row given twice, are refused",
                 sum("a row needs a file, a fn and a reason, once" in f for f in found) == 2))
    rows += _start_child_rows(pathlib.Path(repo))
    rows.append(("process start: the committed tree starts each child through start_child or a table row",
                 not process_start_errors(pathlib.Path(repo))))
    return rows


def magic_file_errors(root):
    """Every TRACKED text file of the checkout at `root` (`git ls-files`, the index) that holds an input magic and
    is not in _MAGIC_FILES. An untracked file (scratch output that quotes the magic) is no sender and is not read."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True)
    if listed.returncode != 0:
        return [f"self-mode magic: `git ls-files` failed in {root}: {listed.stderr.decode(errors='replace').strip()[:200]}"]
    errors = []
    for rel in sorted(p for p in listed.stdout.decode().split("\0") if p):
        path = pathlib.Path(root) / rel
        if not rel.endswith(_MAGIC_SUFFIXES) or rel in _MAGIC_FILES or path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes() if path.stat().st_size < 8_000_000 else b""
        hit = [m.decode() for m in _MAGIC_TEXT if m in data]
        if hit:
            errors.append(f"{rel} holds the self-mode input magic ({hit[0]}): only {sorted(_MAGIC_FILES)} may; "
                          "send it through _selfhost_sources.self_mode_image")
    return errors


def host_first_word_errors(text):
    """The first 8 bytes of the image the host composes (src/build/native_image.rs) are the layout magic or
    user_lo, the length of the std blob taken BEFORE the user region is appended: never an input magic.
    The magics are above 2^56 as numbers, so no feasible std blob length equals one."""
    errors = []
    v1 = re.search(r"let user_lo = combined\.len\(\) as i64;\n\s*combined\.extend_from_slice\(&user_region\);", text)
    first = re.search(r"Vec::with_capacity\(16 \+ combined\.len\(\)\);\n\s*bytes\.extend_from_slice\(&user_lo\.to_le_bytes\(\)\);", text)
    v2 = re.search(r"Vec::with_capacity\(32 \+ img\.bytes\.len\(\)\);\n\s*bytes\.extend_from_slice\(&LAYOUT_MAGIC\.to_le_bytes\(\)\);", text)
    if not (v1 and first and v2):
        errors.append("native_image.rs: the first word of the composed image is no longer user_lo (std blob length) or LAYOUT_MAGIC")
    for magic in (b"MINDSLF\x01", b"MINDKEY\x01", b"MINDHOOK"):
        if int.from_bytes(magic, "little") < 2**56 or int.from_bytes(magic, "little") == 97180662784477517:
            errors.append(f"input magic {magic!r} can collide with a length or the layout magic")
    return errors
# ---- 9. one extern declaration form per name across std ----------------------
_EXTERN_BLOCK_RE = re.compile(r'extern\s*"C"\s*\{(.*?)\}', re.S)
_EXTERN_DECL_RE = re.compile(r"(?:pub\s+)?(?:unsafe\s+|safe\s+)?fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^;]+?))?\s*;", re.S)


def _norm_type(text):
    """One spelling per type: single spaces, none after a `*`."""
    return re.sub(r"\*\s+", "*", re.sub(r"\s+", " ", text).strip())


def extern_decls(std_dir):
    """Every `extern "C"` declaration of every std module as (name, module, params, result).

    params is the tuple of parameter TYPES (names dropped, spelling normalised). Also
    returns parse errors: an `extern` keyword the block pattern did not consume, or
    a block with text the declaration pattern did not consume, is an error, so a new
    declaration syntax cannot slip past the comparison unseen.
    """
    decls, errors = [], []
    for path in sorted(pathlib.Path(std_dir).glob("*.mind")):
        text = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
        blocks = _EXTERN_BLOCK_RE.findall(text)
        if len(re.findall(r"\bextern\b", text)) != len(blocks):
            errors.append(f"std/{path.name}: an `extern` keyword is not an `extern \"C\" {{ ... }}` block")
        for body in blocks:
            if _EXTERN_DECL_RE.sub("", body).strip():
                errors.append(f"std/{path.name}: extern block text is not only `fn name(..) -> ret;` declarations")
            for name, params, result in _EXTERN_DECL_RE.findall(body):
                types = tuple(_norm_type(p.split(":", 1)[-1]) for p in params.split(",") if p.strip())
                decls.append((name, path.stem, types, _norm_type(result or "")))
    return decls, errors


# The parameters (0-based) that the C prototype of each shared extern name takes as pointers,
# from the Linux manual pages (section 2 and 3). () = no pointer parameter. `ioctl` is variadic:
# its third argument is the pointer both std callers pass. A name that more than one std module
# declares needs a row here, so its author states the C prototype once.
_C_POINTER_PARAMS = {
    "__errno_location": (), "_exit": (), "bind": (1,), "clock_gettime": (1,), "close": (),
    "connect": (1,), "dup2": (), "execvp": (0, 1), "fork": (), "getenv": (0,),
    "getsockname": (1, 2), "ioctl": (2,), "kill": (), "listen": (), "memcpy": (0, 1),
    "memset": (0,), "open": (0,), "pipe": (0,), "read": (1,), "sched_getaffinity": (2,),
    "sched_setaffinity": (2,), "send": (1,), "setenv": (0, 1), "setsockopt": (3,),
    "socket": (), "unsetenv": (0,), "waitpid": (1,), "write": (1,),
}


def extern_form_errors(std_dir):
    """Errors for every extern name that two std modules declare with different types, or with an
    integer where its C prototype takes a pointer (`_C_POINTER_PARAMS`).

    Returns (errors, number of declarations, number of names declared in more than one module).
    """
    decls, errors = extern_decls(std_dir)
    if not decls:
        errors.append(f"no extern declaration found under {std_dir} - refusing a vacuous pass")
    by_name = {}
    for name, mod, params, result in decls:
        by_name.setdefault(name, []).append((mod, params, result))
    shared = 0
    for name in sorted(by_name):
        rows = by_name[name]
        if len({m for m, _, _ in rows}) < 2:
            continue
        shared += 1
        if len({(p, r) for _, p, r in rows}) > 1:
            forms = "; ".join(f"std/{m}.mind ({', '.join(p)}) -> {r or 'void'}" for m, p, r in rows)
            errors.append(f"extern `{name}` has more than one declaration form across std: {forms}")
        if name not in _C_POINTER_PARAMS:
            errors.append(f"extern `{name}` is declared in more than one std module but has no row in "
                          f"_C_POINTER_PARAMS: add the pointer parameters of its C prototype")
    for name in sorted(_C_POINTER_PARAMS):
        if name not in by_name:
            errors.append(f"_C_POINTER_PARAMS has a row for `{name}`, which no std module declares")
            continue
        for mod, params, _ in by_name[name]:
            for i in _C_POINTER_PARAMS[name]:
                if i >= len(params) or not params[i].startswith("*"):
                    got = params[i] if i < len(params) else "(missing)"
                    errors.append(f"extern `{name}` in std/{mod}.mind: parameter {i} is `{got}`, "
                                  f"the C prototype takes a pointer there")
    return errors, len(decls), shared


def _both_ways(name_a, set_a, name_b, set_b, errors):
    """Report BOTH directions of a set mismatch, never just the one that fired."""
    for m in sorted(set_a - set_b):
        errors.append(f"{name_a} has '{m}' but {name_b} does not")
    for m in sorted(set_b - set_a):
        errors.append(f"{name_b} has '{m}' but {name_a} does not")


# ---- 7. header counts: one canonical line each, equal to the rows ------------
# Merge 701e90fd kept both sides' count lines (55/55 AND 45/45, and a stale
# "34 modules are not `seed`") and nothing parsed them. The canonical forms:
def _run_on_manifest(manifest_text, main_mind=None, std_edits=()):
    """Run a COPY of this lint, as a subprocess, in a mini repo holding `manifest_text`.

    `main_mind` (bytes) replaces the mini repo's copy of main.mind (the line-pin cases).
    `std_edits` is a sequence of (module, old, new): each replaces `old` by `new` once in the
    mini repo's copy of std/<module>.mind; an `old` that is absent or a replace that changes
    nothing is an error, never a silent no-op.
    """
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        mind = root / "examples" / "mindc_mind"
        (mind / "testdata").mkdir(parents=True)
        for name in ("stdlib_manifest_lint.py", "_stdlib_manifest.py", "_selfhost_sources.py", "_process_starts.py",
                     "main.mind"):  # main.mind: the line pin
            shutil.copyfile(_HERE / name, mind / name)
        table = (_REPO / PROCESS_STARTS).read_text(encoding="utf-8").splitlines()  # the start rows of the copied files
        (root / PROCESS_STARTS).write_text("\n".join(
            row for row in table if row.startswith("file\t") or (root / row.split("\t")[0]).is_file()) + "\n", encoding="utf-8")
        if main_mind is not None:
            (mind / "main.mind").write_bytes(main_mind)
        (mind / "testdata" / "stdlib_manifest.txt").write_text(
            manifest_text, encoding="utf-8")
        for src in (_BRIDGE_RS, _STDLIB_RS, _NATIVE_IMAGE_RS):
            dst = root / src.relative_to(_REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        (root / "std").mkdir()
        for src in sorted(_STD_DIR.glob("*.mind")):
            shutil.copyfile(src, root / "std" / src.name)
        for module, old, new in std_edits:
            target = root / "std" / f"{module}.mind"
            before = target.read_text(encoding="utf-8")
            assert old in before and old != new, f"std edit anchor not in std/{module}.mind: {old!r}"
            target.write_text(before.replace(old, new, 1), encoding="utf-8")
        for git in (["init", "-q"], ["add", "-A"]):  # a checkout: the magic scan reads the index
            subprocess.run(["git", *git], cwd=root, capture_output=True, check=True)
        p = subprocess.run(
            [sys.executable, str(mind / "stdlib_manifest_lint.py")],
            capture_output=True, text=True, cwd=root,
        )
        return p.returncode, p.stdout + p.stderr


def selftest():
    """The lint's real entry point must refuse a damaged copy of the real manifest.

    The first case is an UNDAMAGED copy that must pass, so the mini repo is proven
    one the lint can pass in and each red case is red because of its damage; every
    damage is proven to change the text (a replace that matched nothing is an
    error, not a pass).
    """
    real = _stdlib_manifest.MANIFEST_PATH.read_text(encoding="utf-8")

    def header(name):
        hits = re.findall(rf"^# {name} (\d+)$", real, re.M)
        assert len(hits) == 1, f"real manifest must state {name} exactly once: {hits}"
        return int(hits[0]), f"# {name} {hits[0]}"

    def damaged(old, new):
        assert old in real, f"damage anchor not in the real manifest: {old!r}"
        text = real.replace(old, new, 1)
        assert text != real, f"damage did not change the manifest: {old!r}"
        return text

    (seed, seed_line), (bundled, bundled_line), (_, total_line) = (
        header("SEED_COUNT"), header("BUNDLED_COUNT"), header("TOTAL_COUNT"))
    main = (_HERE / "main.mind").read_bytes()
    n_main = main.count(b"\n")
    assert n_main == MAIN_MIND_LINE_PIN, f"the self-test needs an exact pin: {n_main} lines vs {MAIN_MIND_LINE_PIN}"
    shorter = main[:main.rindex(b"\n", 0, len(main) - 1) + 1]  # last line dropped: the pin is now stale, one too high
    assert shorter.count(b"\n") == n_main - 1
    cases = [
        ("control: real manifest copy -> lint passes", real, 0,
         ["stdlib_manifest_lint: PASS", "extern name(s) declared in more than one std module agree"]),
        # THE SURVIVING MUTANT: a stale second header, reached only via the real read.
        ("duplicated BUNDLED_COUNT header (stale value) -> lint fails",
         damaged(bundled_line, f"{bundled_line}\n# BUNDLED_COUNT {bundled - 12}"), 1,
         ["stdlib_manifest_lint: FAIL", "manifest states BUNDLED_COUNT 2 times"]),
        ("duplicated TOTAL_COUNT header (same value) -> lint fails",
         damaged(total_line, f"{total_line}\n{total_line}"), 1,
         ["stdlib_manifest_lint: FAIL", "manifest states TOTAL_COUNT 2 times"]),
        ("wrong SEED_COUNT value -> lint fails",
         damaged(seed_line, f"# SEED_COUNT {seed + 1}"), 1,
         ["stdlib_manifest_lint: FAIL",
          f"SEED_COUNT says {seed + 1} but the rows give {seed}"]),
        ("main.mind one line longer than the pin -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", f"main.mind has {n_main + 1} lines > MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}"],
         main + b"\n"),
        # A stale HIGHER pin: a shorter main.mind with the old pin left in place.
        ("main.mind one line shorter than the pin (stale pin) -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", f"main.mind has {n_main - 1} lines < MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}",
          f"re-pin it down to {n_main - 1}"], shorter),
        # Extern declaration form (check 9): each mutant puts ONE std declaration back to a form
        # another module does not use. The edited modules are not seed modules, so the seed
        # blob pin stays green and the extern check is the only reason for the red.
        ("extern pointer parameter back to i64 (std.dns clock_gettime) -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", "extern `clock_gettime` has more than one declaration form"],
         [("dns", "fn clock_gettime(clk: i32, ts: *mut u8) -> i32;", "fn clock_gettime(clk: i32, ts: i64) -> i32;")]),
        ("extern result type changed (std.dns connect -> i64) -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", "extern `connect` has more than one declaration form"],
         [("dns", "fn connect(sockfd: i32, addr: *const u8, addrlen: i32) -> i32;", "fn connect(sockfd: i32, addr: *const u8, addrlen: i32) -> i64;")]),
        ("extern arity changed (std.procgroup_sys open without mode) -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", "extern `open` has more than one declaration form"],
         [("procgroup_sys", "fn open(pathname: *const u8, flags: i32, mode: i32) -> i32;", "fn open(pathname: *const u8, flags: i32) -> i32;")]),
        # Agreement on a wrong form: every module that declares the name spells the pointer `i64`,
        # so the form-agreement check is green and only the C prototype row can fail.
        ("extern pointer spelled i64 in every module that declares it (std.fs_checked, std.procgroup, "
         "std.procgroup_text memcpy src) -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", "extern `memcpy` in std/procgroup.mind: parameter 1 is `i64`, the C prototype takes a pointer there"],
         [(m, "fn memcpy(dst: *mut u8, src: *const u8, n: i64)", "fn memcpy(dst: *mut u8, src: i64, n: i64)")
          for m in ("fs_checked", "procgroup", "procgroup_text")]),
        ("extern name declared in two modules without a C prototype row -> lint fails", real, 1,
         ["stdlib_manifest_lint: FAIL", "extern `getppid` is declared in more than one std module but has no row in _C_POINTER_PARAMS"],
         [(m, "fn clock_gettime(", "fn getppid() -> i32;\n    unsafe fn clock_gettime(") for m in ("dns", "procgroup_sys")]),
    ]
    failures = 0
    for case in cases:
        label, text, want_rc, want_in_output = case[:4]
        extra = case[4] if len(case) > 4 else None  # bytes: main.mind replacement; list: std edits
        rc, out = _run_on_manifest(text, *((extra,) if isinstance(extra, bytes) else ()),
                                   **({"std_edits": extra} if isinstance(extra, list) else {}))
        missing = [s for s in want_in_output if s not in out]
        ok = (rc == 0 if want_rc == 0 else rc != 0) and not missing \
            and "Traceback" not in out
        # ONE VERDICT LINE PER CASE (scripts/gate_assert.py counts reported verdicts).
        print(f"[{'PASS' if ok else 'FAIL'}] {label} (expected "
              f"{'exit 0' if want_rc == 0 else 'non-zero exit'}, got {rc})")
        if not ok:
            failures += 1
            print(f"    missing={missing}")
            for line in out.strip().splitlines()[:12]:
                print(f"    {line}")
    # The self-mode magic: allowed in a named file, refused in any other text file (a planted copy).
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for rel, data in (("examples/mindc_mind/nb_lane_keys.mind", b"// MINDSLF\n"), ("tests/plant.py", b"X = 91844762919651661\n"),
                          ("tests/clean.py", b"X = 1\n"), ("tests/blob.elf", b"MINDSLF\x01"), ("scratch/quote.txt", b"MINDHOOK\n")):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
        subprocess.run(["git", "init", "-q"], cwd=root, capture_output=True, check=True)
        subprocess.run(["git", "add", "examples", "tests"], cwd=root, capture_output=True, check=True)  # scratch/ untracked
        got = magic_file_errors(root)
        for label, ok in (("self-mode magic: a planted copy outside the named files -> refused, the named file and a binary pass",
                           any(g.startswith("tests/plant.py ") for g in got)),
                          ("self-mode magic: an untracked file that quotes the magic (scratch output) is not scanned",
                           len(got) == 1 and not any(g.startswith("scratch/") for g in got)),
                          ("self-mode magic: the committed tree holds it only in the named files", not magic_file_errors(_REPO))):
            cases.append(label)
            failures += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    # The host's first word: the committed native_image.rs passes, a text whose first write is something else fails.
    real_ni = _NATIVE_IMAGE_RS.read_text(encoding="utf-8")
    for label, ok in (("host image: the first word is user_lo or the layout magic", not host_first_word_errors(real_ni)),
                      ("host image: a first write of another word is refused",
                       bool(host_first_word_errors(real_ni.replace("bytes.extend_from_slice(&user_lo.to_le_bytes());", "bytes.extend_from_slice(&src_len.to_le_bytes());", 1))))):
        cases.append(label)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    # Process starts: the rows of the rule (plants, the tracked tree) come from process_start_selftest. Deleting a call in main() must go red too: the probes call the functions
    # directly, so only the source of main() shows the entry point.
    lint_src = pathlib.Path(__file__).read_text(encoding="utf-8")
    main_fn = next((n for n in ast.parse(lint_src).body if isinstance(n, ast.FunctionDef) and n.name == "main"), None)

    def calls(name: str) -> bool:
        return main_fn is not None and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name for n in ast.walk(main_fn))

    for label, ok in (
            ("process start: main() still feeds the committed tree", calls("process_start_errors")),
            *process_start_selftest()):
        cases.append(label)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    # The SANCTIONED count-marker shape read by scripts/gate_assert.py.
    print(f"\nSDLC-GATE stdlib_manifest_lint_selftest ran={len(cases)} fail={failures}")
    return 1 if failures else 0


def main(argv):
    if "--self-test" in argv or "--selftest" in argv:
        return selftest()
    if "--print-seed-sha256" in argv:
        print(_stdlib_manifest.actual_seed_sha256())
        return 0

    errors = []
    # ---- 7. main.mind line pin (exact, so a stale higher pin cannot hide room); `--main-mind-pin` runs it alone ----
    main_lines = (_HERE / "main.mind").read_bytes().count(b"\n")
    if main_lines > MAIN_MIND_LINE_PIN:
        errors.append(f"main.mind has {main_lines} lines > MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}: "
                      "put new compiler code in a COMPILER_COMPONENTS part")
    elif main_lines < MAIN_MIND_LINE_PIN:
        errors.append(f"main.mind has {main_lines} lines < MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}: the pin is stale, "
                      f"re-pin it down to {main_lines} in the commit that shortened main.mind")
    if "--main-mind-pin" in argv:
        print(errors[0] if errors else f"main.mind pin PASS: {main_lines} == MAIN_MIND_LINE_PIN {MAIN_MIND_LINE_PIN}")
        return 1 if errors else 0

    errors += magic_file_errors(_REPO)
    errors += host_first_word_errors(_NATIVE_IMAGE_RS.read_text(encoding="utf-8"))
    errors += process_start_errors(_REPO)
    manifest_all = _stdlib_manifest.all_modules()
    manifest_seed = _stdlib_manifest.seed_modules()
    manifest_bundled = _stdlib_manifest.bundled_modules()

    if len(set(manifest_all)) != len(manifest_all):
        dupes = sorted({m for m in manifest_all if manifest_all.count(m) > 1})
        errors.append(f"manifest lists these modules more than once: {dupes}")

    # ---- 1. manifest <-> disk, BOTH directions -------------------------------
    disk = sorted(p.stem for p in _STD_DIR.glob("*.mind"))
    if not disk:
        errors.append(f"no std/*.mind found under {_STD_DIR} — refusing a vacuous pass")
    _both_ways("std/ on disk", set(disk), "the manifest", set(manifest_all), errors)

    # ---- 2. seed blob golden (pins the SET and the ORDER) --------------------
    declared = _stdlib_manifest.declared_seed_sha256()
    if declared is None:
        errors.append(
            "manifest header has no `# SEED_BLOB_SHA256 <64 hex>` line — the seed "
            "order is then unpinned and a reseed could land silently"
        )
    else:
        actual = _stdlib_manifest.actual_seed_sha256()
        if actual != declared:
            errors.append(
                f"SEED BLOB CHANGED: manifest declares {declared}, std/ + manifest "
                f"now compose {actual}. If this is intentional it is a WHOLE-CORPUS "
                f"RESEED EVENT (re-mint testdata/selfhost_loop/stage1.elf), not a "
                f"routine edit."
            )

    # ---- 3. the Rust bridge reads the manifest, keeps no literal ------------
    bridge_src = _BRIDGE_RS.read_text(encoding="utf-8")
    if "include_str!(\"../../examples/mindc_mind/testdata/stdlib_manifest.txt\")" not in bridge_src:
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} no longer include_str!s the manifest — "
            f"the bridge has gone back to a hand-copied list"
        )
    if re.search(r"const\s+STD_MODULES\s*:\s*\[&str", bridge_src):
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} reintroduced a literal `const STD_MODULES` "
            f"array — the manifest must be the only source of truth"
        )

    # ---- 4. frozen seed count agrees on both sides --------------------------
    m = re.search(r"const\s+FROZEN_STD_SEED_COUNT\s*:\s*usize\s*=\s*(\d+)", bridge_src)
    if not m:
        errors.append(
            f"{_BRIDGE_RS.relative_to(_REPO)} has no FROZEN_STD_SEED_COUNT tripwire"
        )
    else:
        rust_count = int(m.group(1))
        if rust_count != len(manifest_seed):
            errors.append(
                f"native_bridge.rs FROZEN_STD_SEED_COUNT={rust_count} but the manifest declares "
                f"{len(manifest_seed)} seed modules"
            )
        if rust_count != _stdlib_manifest.FROZEN_STD_SEED_COUNT:
            errors.append(
                f"native_bridge.rs FROZEN_STD_SEED_COUNT={rust_count} but "
                f"_stdlib_manifest.FROZEN_STD_SEED_COUNT="
                f"{_stdlib_manifest.FROZEN_STD_SEED_COUNT}"
            )

    # ---- 5. STDLIB_MIND_SOURCES <-> the `bundled` column, BOTH directions ---
    stdlib_src = _STDLIB_RS.read_text(encoding="utf-8")
    bundled_actual = re.findall(r'\(\s*"std\.([a-z0-9_]+)",\s*include_str!', stdlib_src)
    if not bundled_actual:
        errors.append(
            f"could not extract STDLIB_MIND_SOURCES from "
            f"{_STDLIB_RS.relative_to(_REPO)} — the lint would pass vacuously"
        )
    _both_ways(
        "STDLIB_MIND_SOURCES (src/project/stdlib.rs)", set(bundled_actual),
        "the manifest `bundled` column", set(manifest_bundled), errors,
    )

    # ---- 6. every remaining hand-copied literal equals the manifest seed ----
    n_literals = 0
    for py in sorted(_HERE.glob("*.py")):
        if py.name in _MIGRATED:
            continue
        found = _LIST_RE.findall(py.read_text(encoding="utf-8"))
        for raw in found:
            n_literals += 1
            try:
                lst = ast.literal_eval(raw)
            except (ValueError, SyntaxError) as exc:
                errors.append(f"{py.name}: unparseable module list ({exc})")
                continue
            if list(lst) != manifest_seed:
                errors.append(
                    f"{py.name}: hand-copied std list DRIFTED from the manifest "
                    f"(has {len(lst)} entries; first difference: "
                    f"{_first_diff(lst, manifest_seed)})"
                )

    # ---- 8. seed private fns carry a _SEED_PRIVATE_PREFIXES prefix (the flat image reserves every seed name)
    seen = {}  # private name -> the seed module that defines it
    for mod in manifest_seed:
        prefixes = _SEED_PRIVATE_PREFIXES.get(mod, ())
        for name in re.findall(r"^[ \t]*fn (\w+)", (_STD_DIR / f"{mod}.mind").read_text(encoding="utf-8"), re.M):
            if not name.startswith(prefixes):
                errors.append(f"std/{mod}.mind private fn `{name}` lacks a module prefix {prefixes}")
            if seen.setdefault(name, mod) != mod: errors.append(f"private fn `{name}` in std/{seen[name]}.mind and std/{mod}.mind")
    if not seen: errors.append("no private seed fn found: the prefix check would pass vacuously")

    # ---- 9. one extern declaration form per name across std ------------------
    extern_errors, n_extern_decls, n_extern_shared = extern_form_errors(_STD_DIR)
    errors.extend(extern_errors)

    # ---- 7. header counts: exactly one canonical line each, equal to rows ---
    counts = (len(manifest_all), len(manifest_seed), len(manifest_bundled))
    manifest_text = _stdlib_manifest.MANIFEST_PATH.read_text(encoding="utf-8")
    errors.extend(count_header_errors(manifest_text, *counts))
    probes = count_header_probes(manifest_text, *counts)
    for name, variant in probes.items():
        if not count_header_errors(variant, *counts):
            errors.append(
                f"header-count check accepted its hostile '{name}' variant of the "
                f"manifest — a guard in count_header_errors has been lost"
            )
    if errors:
        print("stdlib_manifest_lint: FAIL")
        for e in errors:
            print(f"  - {e}")
        print(
            f"\n{len(errors)} problem(s). The contract is "
            f"examples/mindc_mind/testdata/stdlib_manifest.txt; every std/*.mind must "
            f"appear there exactly once with a deliberate in/out decision per consumer."
        )
        return 1

    print(
        f"stdlib_manifest_lint: PASS — {len(manifest_all)} std modules, manifest<->disk "
        f"exact both ways; seed={len(manifest_seed)} (blob sha256 pinned) "
        f"bundled={len(bundled_actual)} unreachable={len(manifest_all) - len(manifest_bundled)}; "
        f"{n_literals} remaining hand-copied list(s) verified against the manifest; "
        f"{len(seen)} seed private fn(s) carry their module prefix, no name in two modules; "
        f"{n_extern_shared} extern name(s) declared in more than one std module agree in every "
        f"parameter and result type ({n_extern_decls} declarations); "
        f"header counts SEED/BUNDLED/TOTAL/non-seed stated once each and exact, "
        f"{len(probes)} hostile header variants refused"
    )
    return 0


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i}: {x!r} vs manifest {y!r}"
    return f"length {len(a)} vs manifest {len(b)}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
