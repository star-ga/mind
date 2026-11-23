"""Floor cases of the gate-assert count contract.

A gate whose green run states its own verdict count (the legs of the mic@3 flip
gate, the `VERDICTS` of a gate source) must be called with exactly that count as
its runner floor. tests/gate_assert_count_contract_test.py runs both cases with
its `check`; this module holds no runner.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import smoke_wiring_lint  # on sys.path: the contract test puts examples/mindc_mind there first

REPO = Path(__file__).resolve().parent.parent
Check = Callable[[str, object, object], None]


def case_flip_gate_floor_counts_every_leg(check: Check) -> None:
    """The whole-module mic@3 FLIP gate prints one verdict per leg on a green
    run (the flip, the record-binding rows, the refusal pins). Its runner call
    sites carry a measured floor, so each must demand every leg, or a dropped
    leg grades green. The leg count is READ from the gate source (one literal
    `PASS:` print per leg), so a leg added without a floor change turns this red."""
    gate = REPO / "examples" / "mindc_mind" / "mic3_flip_smoke.py"
    legs = 0
    for node in ast.walk(ast.parse(gate.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print" and node.args):
            continue
        arg = node.args[0]
        head = arg.values[0] if isinstance(arg, ast.JoinedStr) and arg.values else arg
        if isinstance(head, ast.Constant) and str(head.value).startswith("PASS:"):
            legs += 1
    check("flip gate: legs read from the gate source", legs >= 3, True)
    checked = 0
    for rel in (".github/workflows/ci.yml", "scripts/preflight.sh"):
        joined = (REPO / rel).read_text(encoding="utf-8").replace("\\\n", " ")
        for line in joined.splitlines():
            if "run_gate.py" not in line or "mic3_flip_smoke" not in line:
                continue
            checked += 1
            m = smoke_wiring_lint.MIN_ASSERTED_RE.search(line)
            check(f"{rel}: flip gate floor is its leg count", int(m.group(1)) if m else 1, legs)
    check("flip-gate call sites found", checked >= 2, True)


def _declared_verdicts(gate: Path) -> int | None:
    """The module-level `VERDICTS` of a gate source: the number of verdict lines
    of its green run, as a sum of its own row-table constants. None if absent."""
    consts: dict[str, int] = {}

    def value(node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return consts[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        raise ValueError(ast.dump(node))

    for node in ast.parse(gate.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = value(node.value)
            except (KeyError, ValueError):
                pass
    return consts.get("VERDICTS")


def case_declared_verdict_count_is_the_call_site_floor(check: Check) -> None:
    """A gate that declares `VERDICTS` (its rows count themselves at run time)
    states its own green count, so every runner call site of it must demand that
    number: a floor below it lets a dropped row grade green. The call sites are
    READ out of the wiring files, the count out of the gate source."""
    seen: dict[str, int] = {}
    for rel in (".github/workflows/ci.yml", "scripts/preflight.sh"):
        joined = (REPO / rel).read_text(encoding="utf-8").replace("\\\n", " ")
        for line in joined.splitlines():
            if "run_gate.py" not in line:
                continue
            tail = line.split("run_gate.py", 1)[1].split()
            gate = next((t for t in tail if t.endswith((".py", ".sh")) and not t.startswith("-")), "")
            if not gate.endswith(".py") or not (REPO / gate).is_file():
                continue
            want = _declared_verdicts(REPO / gate)
            if want is None:
                continue
            seen[gate] = seen.get(gate, 0) + 1
            m = smoke_wiring_lint.MIN_ASSERTED_RE.search(line)
            check(f"{rel}: {gate} floor is its declared VERDICTS", int(m.group(1)) if m else 1, want)
    diag = "examples/mindc_mind/self_host_native_diag_smoke.py"
    check("diag smoke declares its verdict count", _declared_verdicts(REPO / diag) is not None, True)
    check("diag smoke call sites found", seen.get(diag, 0) >= 1, True)
