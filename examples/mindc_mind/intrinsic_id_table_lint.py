#!/usr/bin/env python3
"""Native intrinsic id table contract for main.mind (pure text analysis).

The native emitter names each `__mind_*` intrinsic by a small integer id:
nb_intrinsic_id / nb_sys_intrinsic_id map a callee name to the id, and
nb_emit_intrinsic selects the machine code by `if kind == ID { .. }`. An id with
no arm falls through to the trailing `__mind_read` sequence (id 9).

Two lanes that each append ids from the same "next free" number merge as clean
text and still collide: one name then emits the other name's code, or an id
whose arm was displaced falls through to a read syscall. That happened once
(float bit intrinsics and the syscall block both claimed 20 and 21); git
reported no conflict. This lint makes that shape fail before anything is built:

  1. every id is returned for exactly ONE name across both mapping functions;
  2. every id except the fall-through id 9 has exactly ONE top-level
     `if kind == ID` arm in the dispatch function;
  3. every `if kind == ID` arm at any depth (below the stream-sentinel range
     >= 1008) names a mapped id.

Kinds 60..99 live in the compiler parts (COMPILER_COMPONENTS), so the lint reads the
composed unit (main.mind + parts): every part map `nb*_intrinsic_id` /
`nb*_sys_intrinsic_id` joins rule 1, its ids must lie in 60..99 (and main.mind's
outside it), and the top-level `if kind == ID` arms of the part `nb*_emit` /
`nb*_sys_emit` fns are the dispatch arms of those ids (rules 2 and 3).

The dispatch function is read out of main.mind, not hand-named: nb_emit_intrinsic,
or the function it forwards `(code, kind, ..)` to when it holds no arm itself (the
source-image context refactor moved every arm into nb_emit_intrinsic_ctx and left
nb_emit_intrinsic a one-line wrapper, after which this lint saw 0 arms for every
id). A dispatch arm is a statement of that function's body (4-space indent); the
nested `if ctx != 0` stream re-routes of ids 8 and 9 are not second arms.

Every run first mutates the text in memory (a duplicate id, a dropped arm, an orphan
arm) and requires each mutant to be rejected, so a lint that cannot fail is caught.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _selfhost_sources import compiler_library  # noqa: E402

MAP_FNS = ("nb_intrinsic_id", "nb_sys_intrinsic_id")
EMIT_FN = "nb_emit_intrinsic"
FALLTHROUGH_ID = 9
PART_KINDS = range(60, 100)
PART_MAP = re.compile(r"^pub fn (nb[a-z]_(?:sys_)?intrinsic_id)\(", re.M)
PART_EMIT = re.compile(r"^pub fn (nb[a-z]_(?:sys_)?emit)\(", re.M)


def fn_body(src: str, name: str) -> str:
    start = src.find(f"pub fn {name}(")
    if start < 0:
        raise SystemExit(f"FAIL  {name} not found in the composed compiler unit")
    end = src.find("\n}\n", start)
    return src[start:end]


def dispatch_fn(src: str) -> tuple[str, str]:
    """(name, body) of the function holding the `if kind == ID` dispatch."""
    name, seen = EMIT_FN, set()
    while True:
        body = fn_body(src, name)
        fwd = re.search(r"^    (\w+)\(code, kind\b", body, re.M)
        if re.search(r"\bif kind == \d+ \{", body) or fwd is None or name in seen:
            return name, body
        seen.add(name)
        name = fwd.group(1)


def returned_ids(src: str, fns) -> list[int]:
    return [int(x) for fn in fns for x in re.findall(r"\breturn (\d+);", fn_body(src, fn))]


def problems(src: str) -> list[str]:
    out: list[str] = []
    main_ids = returned_ids(src, MAP_FNS)
    part_ids = returned_ids(src, PART_MAP.findall(src))
    ids = collections.Counter(main_ids + part_ids)
    out += [f"main.mind map returns part kind {i} (60..99 belong to the parts)" for i in main_ids if i in PART_KINDS]
    out += [f"a part map returns {i}, outside the part kinds 60..99" for i in part_ids if i not in PART_KINDS]
    emit_fn, body = dispatch_fn(src)
    part_bodies = [fn_body(src, fn) for fn in PART_EMIT.findall(src)]
    if not part_bodies:
        out.append("no part emit fn found (the parts were not composed)")
    body += "".join(part_bodies)
    arms = collections.Counter(int(x) for x in re.findall(r"^    if kind == (\d+) \{", body, re.M))
    every_arm = collections.Counter(int(x) for x in re.findall(r"\bif kind == (\d+) \{", body))
    for i, n in sorted(ids.items()):
        if n != 1:
            out.append(f"id {i} is returned for {n} names (ids must be unique)")
    for i in sorted(ids):
        if i == FALLTHROUGH_ID:
            continue
        if arms[i] != 1:
            out.append(f"id {i} has {arms[i]} emit arms in {emit_fn} (want exactly 1)")
    for a in sorted(every_arm):
        if a < 1008 and a not in ids:
            out.append(f"emit arm `kind == {a}` names no mapped id")
    if not ids:
        out.append("no intrinsic ids found (the parser lost the table)")
    return out


def selftest(src: str) -> int:
    first_id = int(re.search(r"\breturn (\d+);", fn_body(src, "nb_sys_intrinsic_id")).group(1))
    mutants = {
        "duplicate id": src.replace(f"return {first_id};", "return 1;", 1),
        "dropped arm": src.replace(f"if kind == {first_id} {{", "if kind == 999 {", 1),
        "orphan arm": src.replace(f"if kind == {first_id} {{", f"if kind == 777 {{\n        return code;\n    }}\n    if kind == {first_id} {{", 1),
        # Two emitter lanes take the same next free part id: 60 in nb_x86_vex AND nb_shm.
        "duplicate id across two parts": _claim(_claim(src, "nbv", True), "nbs", False),
    }
    bad = 0
    for name, text in mutants.items():
        caught = bool(problems(text))
        print(f"  {'PASS' if caught else 'FAIL'}  selftest mutant rejected: {name}")
        bad += not caught
    return bad


def _claim(src: str, part: str, arm: bool) -> str:
    """`part` maps one name to id 60 (and, with `arm`, emits it): an emitter lane's shape."""
    stub = f"pub fn {part}_intrinsic_id(buf: i64, lo: i64, hi: i64) -> i64 {{\n"
    out = src.replace(stub, stub + "    if hi > lo {\n        return 60;\n    }\n", 1)
    if arm:
        emit = f"pub fn {part}_emit(code: String, kind: i64, ids: i64, dst_id: i64, ctx: i64) -> String {{\n"
        out = out.replace(emit, emit + "    if kind == 60 {\n        return code;\n    }\n", 1)
    if out.count("return 60;") <= src.count("return 60;"):
        raise SystemExit(f"FAIL  mutant anchor for {part}_intrinsic_id not found")
    return out


def main(argv: list[str]) -> int:
    src = compiler_library().decode("utf-8")
    bad = selftest(src)
    found = problems(src)
    for p in found:
        print(f"  FAIL  {p}")
    if not found:
        n = len(returned_ids(src, MAP_FNS))
        parts = len(PART_MAP.findall(src))
        print(f"  PASS  {n} intrinsic ids unique, each with exactly one emit arm (fall-through id {FALLTHROUGH_ID});"
              f" {len(returned_ids(src, PART_MAP.findall(src)))} part ids over {parts} part maps")
    return 1 if (found or bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
