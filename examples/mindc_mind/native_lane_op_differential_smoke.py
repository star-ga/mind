#!/usr/bin/env python3
"""Seeded three-path differential for the admitted native-lane scalar operators.

Each program is executed three ways and the exit status and stdout are compared:

  * native-lane ELF, `mindc build --backend native --native-profile bounded`
    with the marked compiler named by --compiler / MINDC_NATIVE_ELF;
  * MLIR ELF, `mindc build --backend mlir`;
  * the reference evaluator, `mind eval` of the same source plus a `main()`
    call. The evaluator prints the value of that call and then a lowered-IR
    dump; only the value is read. An evaluation error is not an exit status.

Operators: + - * / % << >> | & ^, comparisons used only as if-conditions,
if/else, and while with a loop-carried copy (`b = a`). Arithmetic goes through
a parameter helper so a constant folder cannot erase the operation.

Results of ordinary rows are kept in 0..255 so a process exit equals the full
value. A negative or out-of-range result is encoded as a small positive exit
(`if r == <model> { return YES } else { return NO }`). The model is wrapping
two's-complement arithmetic, toward-zero division, and a shift count masked
with `& 63`.

Division edges have per-path pins (expectation follows D1): compiled paths
require trap 79, except the narrow signed MIN / -1 and MIN % -1 rows, which
docs/errors.md ("Integer division trap") defines as MIN and 0 (the divisor-1
result, no trap) and which self_host_native_div_trap_smoke.py pins the same way
on both backends; the evaluator requires the measured named error, overflow
panic, or value. Narrow native types may refuse with lane.type.unsupported.
The imported division row builds a real helper module on both compiled paths.
The expression-only evaluator receives the same helper text joined with the
entry (only the import directive is removed); it has no module loader.

Exit: 0 every row matches; 1 any row does not; 2 BLOCKED (mindc, mind, the
marked compiler, or prlimit is missing). Prints one verdict line per row and
never a count marker.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import gate_assert  # noqa: E402

SEED = 20261021
MARK = b"\x48\xb8nblane01"
TRAP_EXIT = 79
# Exit of _narrow_min when the documented narrow MIN / -1 result (MIN, or 0 for
# `%`) is observed; 1 is a wrong value and TRAP_EXIT a trap the contract forbids.
NARROW_MIN_EXIT = 2
INT_MIN = -1 << 63
INT_MAX = (1 << 63) - 1
MIN_PROGRAMS = 64

# Used only when the variable is unset AND this path is an executable file.
# A matrix leg that installed a different major leaves the variable unset and
# does not have this path, so the build uses PATH.
_LLVM20 = {
    "MLIR_OPT": "/usr/lib/llvm-20/bin/mlir-opt",
    "MLIR_TRANSLATE": "/usr/lib/llvm-20/bin/mlir-translate",
    "CLANG": "/usr/lib/llvm-20/bin/clang",
}
_REFUSAL_RE = re.compile(r"outside candidate native-lane-v1 \(([^)]*)\)")

_OP_NAME = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod",
    "<<": "shl", ">>": "shr", "|": "or", "&": "and", "^": "xor",
    "<": "lt", ">": "gt", "<=": "le", ">=": "ge", "==": "eq", "!=": "ne",
}


class LCG:
    def __init__(self, seed: int) -> None:
        self.s = seed & 0x7FFFFFFF

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s

    def span(self, lo: int, hi: int) -> int:
        return lo + self.next() % (hi - lo + 1)


def wrap(v: int) -> int:
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


def lit(v: int) -> str:
    """A MIND expression for v with no negative literal token."""
    if v == INT_MIN:
        return "(0 - 9223372036854775807 - 1)"
    if v < 0:
        return f"(0 - {-v})"
    return str(v)


def model(op: str, a: int, b: int) -> int | None:
    """Independent reference. None means the operation traps on i64."""
    if op in ("/", "%"):
        if b == 0 or (a == INT_MIN and b == -1):
            return None
        q = abs(a) // abs(b)
        if (a < 0) != (b < 0):
            q = -q
        return wrap(q) if op == "/" else wrap(a - b * q)
    if op == "+":
        return wrap(a + b)
    if op == "-":
        return wrap(a - b)
    if op == "*":
        return wrap(a * b)
    if op == "|":
        return wrap(a | b)
    if op == "&":
        return wrap(a & b)
    if op == "^":
        return wrap(a ^ b)
    c = b & 63
    if op == "<<":
        return wrap(a << c)
    if op == ">>":
        return wrap(a >> c)
    raise ValueError(op)


def model_self_check() -> str | None:
    """Return a drift description, or None when the oracle matches known rows."""
    two62 = 1 << 62
    expect = [
        (model("+", 3, 4), 7),
        (model("-", 3, 10), -7),
        (model("*", two62, 4), 0),
        (model("/", 7, 2), 3),
        (model("/", -7, 2), -3),
        (model("%", -7, 2), -1),
        (model("/", 5, 0), None),
        (model("%", 5, 0), None),
        (model("/", INT_MIN, -1), None),
        (model("%", INT_MIN, -1), None),
        (model("+", INT_MAX, 1), INT_MIN),
        (model("<<", 1, 64), 1),
        (model("<<", 1, 63), INT_MIN),
        (model(">>", -8, 1), -4),
        (model(">>", 64, 2), 16),
        (model("|", 8, 1), 9),
        (model("&", 15, 7), 7),
        (model("^", 15, 7), 8),
    ]
    for got, want in expect:
        if got != want:
            return f"got {got} want {want}"
    return None


@dataclass
class Case:
    cid: str
    source: str
    expect_exit: int
    allow_native_refusal: bool = False
    follows_d1: bool = False
    # Per-path pin for edge rows (STEER R6): the evaluator's current defined
    # outcome, measured and printed; empty means the ordinary value pin.
    expect_eval: str = ""
    modules: dict[str, str] = field(default_factory=dict)


def _bin_direct(op: str, a: int, b: int) -> str:
    return (
        "fn bin(a: i64, b: i64) -> i64 {\n"
        f"    return a {op} b;\n"
        "}\n"
        "fn main() -> i64 {\n"
        f"    return bin({lit(a)}, {lit(b)});\n"
        "}\n"
    )


def _bin_encoded(op: str, a: int, b: int, want: int, yes: int, no: int) -> str:
    return (
        "fn bin(a: i64, b: i64) -> i64 {\n"
        f"    return a {op} b;\n"
        "}\n"
        "fn main() -> i64 {\n"
        f"    let r: i64 = bin({lit(a)}, {lit(b)});\n"
        f"    if r == {lit(want)} {{\n"
        f"        return {yes};\n"
        "    } else {\n"
        f"        return {no};\n"
        "    }\n"
        "}\n"
    )


def _cmp_source(op: str, a: int, b: int) -> str:
    return (
        "fn main() -> i64 {\n"
        f"    let a: i64 = {lit(a)};\n"
        f"    let b: i64 = {lit(b)};\n"
        f"    if a {op} b {{\n"
        "        return 11;\n"
        "    } else {\n"
        "        return 12;\n"
        "    }\n"
        "}\n"
    )


def _holds(op: str, a: int, b: int) -> bool:
    return {
        "<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b,
        "==": a == b, "!=": a != b,
    }[op]


def _trap_source(op: str, a: int, b: int, direct: bool) -> str:
    fn = "dv" if op == "/" else "md"
    expr = f"{lit(a)} {op} {lit(b)}" if direct else f"{fn}({lit(a)}, {lit(b)})"
    return (
        "fn dv(a: i64, b: i64) -> i64 {\n    return a / b;\n}\n"
        "fn md(a: i64, b: i64) -> i64 {\n    return a % b;\n}\n"
        "fn main() -> i64 {\n"
        f"    let q: i64 = {expr};\n"
        "    return 4 + q - q;\n"
        "}\n"
    )


def _narrow_zero(ty: str, op: str) -> str:
    fn = "dv" if op == "/" else "md"
    return (
        f"fn {fn}(a: {ty}, b: {ty}) -> {ty} {{\n    return a {op} b;\n}}\n"
        "fn main() -> i64 {\n"
        f"    let z: {ty} = {fn}(5, 0);\n"
        "    return 4 + (z as i64) - (z as i64);\n"
        "}\n"
    )


def _narrow_min(ty: str, op: str, min_expr: str) -> str:
    fn = "dv" if op == "/" else "md"
    cond = f"q != {min_expr}" if op == "/" else "q != 0"
    return (
        f"fn {fn}(a: {ty}, b: {ty}) -> {ty} {{\n    return a {op} b;\n}}\n"
        "fn main() -> i64 {\n"
        f"    let q: {ty} = {fn}({min_expr}, 0 - 1);\n"
        f"    if {cond} {{\n        return 1;\n    }}\n"
        "    return 2;\n"
        "}\n"
    )


def _draw_pair(rng: LCG, op: str) -> tuple[int, int]:
    if op == "-":
        a = rng.span(0, 200)
        b = rng.span(0, a)
        return a, b
    if op in ("/", "%"):
        return rng.span(0, 200), rng.span(1, 20)
    if op == "<<":
        return rng.span(0, 15), rng.span(0, 4)
    if op == ">>":
        return rng.span(0, 255), rng.span(0, 6)
    if op == "*":
        return rng.span(0, 15), rng.span(0, 15)
    return rng.span(0, 120), rng.span(0, 120)


def build_corpus(seed: int = SEED) -> list[Case]:
    rng = LCG(seed)
    cases: list[Case] = []
    seen: set[tuple[str, int, int]] = set()
    for op in ("+", "-", "*", "/", "%", "<<", ">>", "|", "&", "^"):
        got = 0
        guard = 0
        while got < 6:
            guard += 1
            if guard > 10000:
                raise RuntimeError(f"could not draw 6 pairs for {op}")
            a, b = _draw_pair(rng, op)
            if (op, a, b) in seen:
                continue
            result = model(op, a, b)
            if result is None or not 0 <= result <= 255:
                continue
            seen.add((op, a, b))
            cid = f"arith_{_OP_NAME[op]}_{got}"
            cases.append(Case(cid, _bin_direct(op, a, b), result))
            got += 1

    cases.append(Case(
        "reachable_div_in_import",
        "import helper;\nfn main() -> i64 { return imported_div(21, 3); }\n",
        7, modules={"helper.mind":
                    "pub fn imported_div(a: i64, b: i64) -> i64 { return a / b; }\n"},
    ))

    specials = [
        ("enc_sub_neg", "-", 3, 10, 21, 22),
        ("enc_div_neg", "/", -7, 2, 31, 32),
        ("enc_mod_neg", "%", -7, 2, 41, 42),
        ("enc_shr_neg", ">>", -8, 1, 51, 52),
        ("enc_add_wrap", "+", INT_MAX, 1, 71, 72),
        ("enc_shl_63", "<<", 1, 63, 61, 62),
    ]
    for cid, op, a, b, yes, no in specials:
        want = model(op, a, b)
        if want is None or 0 <= want <= 255:
            raise RuntimeError(f"{cid} is not an encoded out-of-range result")
        cases.append(Case(cid, _bin_encoded(op, a, b, want, yes, no), yes))
    shl64 = model("<<", 1, 64)
    mulw = model("*", 1 << 62, 4)
    if shl64 != 1 or mulw != 0:
        raise RuntimeError("direct wrap/mask rows drifted")
    cases.append(Case("dir_shl_64", _bin_direct("<<", 1, 64), 1))
    cases.append(Case("dir_mul_wrap", _bin_direct("*", 1 << 62, 4), 0))

    for op in ("<", ">", "<=", ">=", "==", "!="):
        # Three orderings, not three truth values: `==` is never true for a != b,
        # and `!=` is never false for a != b, so a true/false split does not fill.
        buckets: dict[str, tuple[int, int]] = {}
        guard = 0
        while len(buckets) < 3:
            guard += 1
            if guard > 10000:
                raise RuntimeError(f"could not draw comparisons for {op}")
            a = rng.span(-20, 40)
            b = rng.span(-20, 40)
            key = "eq" if a == b else ("lt" if a < b else "gt")
            buckets.setdefault(key, (a, b))
        for i, key in enumerate(("eq", "lt", "gt")):
            a, b = buckets[key]
            expect = 11 if _holds(op, a, b) else 12
            cases.append(Case(f"cmp_{_OP_NAME[op]}_{i}", _cmp_source(op, a, b), expect))

    cases.append(Case(
        "loop_copy_audit",
        "fn main() -> i64 {\n"
        "    let a: i64 = 7;\n"
        "    let mut b: i64 = 2;\n"
        "    let mut i: i64 = 0;\n"
        "    while i < 1 {\n"
        "        b = a;\n"
        "        i = i + 1;\n"
        "    }\n"
        "    return b;\n"
        "}\n",
        7,
    ))
    cases.append(Case(
        "loop_copy_zero",
        "fn main() -> i64 {\n"
        "    let a: i64 = 7;\n"
        "    let mut b: i64 = 2;\n"
        "    let mut i: i64 = 0;\n"
        "    while i < 0 {\n"
        "        b = a;\n"
        "        i = i + 1;\n"
        "    }\n"
        "    return b;\n"
        "}\n",
        2,
    ))
    for i in range(6):
        n = rng.span(1, 5)
        a0 = rng.span(0, 30)
        step = rng.span(1, 3)
        expect = a0 + step * (n - 1)
        src = (
            "fn main() -> i64 {\n"
            f"    let mut a: i64 = {a0};\n"
            "    let mut b: i64 = 0;\n"
            "    let mut i: i64 = 0;\n"
            f"    while i < {n} {{\n"
            "        b = a;\n"
            f"        a = a + {step};\n"
            "        i = i + 1;\n"
            "    }\n"
            "    return b;\n"
            "}\n"
        )
        cases.append(Case(f"while_copy_{i}", src, expect))

    seen_branch: set[int] = set()
    made = 0
    guard = 0
    while made < 4:
        guard += 1
        if guard > 10000:
            raise RuntimeError("could not draw if/else rows")
        lo = rng.span(0, 12)
        hi = rng.span(lo + 1, 20)
        x = rng.span(0, 20)
        if x < lo:
            expect = 3
        elif x >= hi:
            expect = 5
        else:
            expect = 4
        # The first three rows cover the three branches; the fourth may repeat.
        if made < 3 and expect in seen_branch:
            continue
        seen_branch.add(expect)
        src = (
            "fn main() -> i64 {\n"
            f"    let x: i64 = {x};\n"
            f"    if x < {lo} {{\n"
            "        return 3;\n"
            "    } else {\n"
            f"        if x >= {hi} {{\n"
            "            return 5;\n"
            "        } else {\n"
            "            return 4;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        cases.append(Case(f"ifelse_{made}", src, expect))
        made += 1

    for op, a, b in (
        ("/", 5, 0), ("%", 5, 0), ("/", INT_MIN, -1), ("%", INT_MIN, -1),
    ):
        tag = _OP_NAME[op]
        rhs = "zero" if b == 0 else "min"
        for direct in (False, True):
            form = "literal" if direct else "helper"
            # Per-path pin (STEER R6): native/mlir trap 79; the evaluator's
            # current defined outcome is a named error on x/0 and a panic on
            # INT_MIN/-1. Measured 2026-10-21; expectation follows D1.
            eval_pin = "zero" if b == 0 else ("div_overflow" if op == "/" else "mod_overflow")
            cases.append(Case(
                f"edge_i64_{tag}_{rhs}_{form}",
                _trap_source(op, a, b, direct),
                TRAP_EXIT,
                allow_native_refusal=False,
                follows_d1=True,
                expect_eval=eval_pin,
            ))

    narrow_min = {
        "i8": "(0 - 127 - 1)",
        "i16": "(0 - 32767 - 1)",
        "i32": "(0 - 2147483647 - 1)",
    }
    for ty, min_expr in narrow_min.items():
        for op in ("/", "%"):
            cases.append(Case(
                f"edge_{ty}_{_OP_NAME[op]}_zero",
                _narrow_zero(ty, op),
                TRAP_EXIT,
                allow_native_refusal=True,
                follows_d1=True,
                expect_eval="narrow_z" if ty == "i32" else "zero",
            ))
            cases.append(Case(
                f"edge_{ty}_{_OP_NAME[op]}_min",
                _narrow_min(ty, op, min_expr),
                NARROW_MIN_EXIT,
                allow_native_refusal=True,
                follows_d1=True,
                expect_eval="narrow_q" if ty == "i32" else "value2",
            ))
    for ty in ("u8", "u16", "u32"):
        for op in ("/", "%"):
            cases.append(Case(
                f"edge_{ty}_{_OP_NAME[op]}_zero",
                _narrow_zero(ty, op),
                TRAP_EXIT,
                allow_native_refusal=True,
                follows_d1=True,
                expect_eval="narrow_z" if ty == "u32" else "zero",
            ))

    ids = [c.cid for c in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case id")
    if len(cases) < MIN_PROGRAMS:
        raise RuntimeError(f"corpus {len(cases)} < {MIN_PROGRAMS}")
    return cases


@dataclass
class Obs:
    kind: str
    exit_code: int | None
    stdout: bytes
    note: str

    def brief(self) -> str:
        note = " ".join(self.note.split())[:120]
        if self.kind == "ran":
            return f"exit={self.exit_code} stdout={len(self.stdout)}B"
        if note:
            return f"{self.kind}:{note}"
        return self.kind


def _last(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def _tools(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    for key, path in _LLVM20.items():
        if out.get(key):
            continue
        if os.path.isfile(path) and os.access(path, os.X_OK):
            out[key] = path
    return out


def _build(mindc: str, src: pathlib.Path, out: pathlib.Path, native: bool,
           env: dict[str, str]) -> tuple[int, str]:
    project = (src.parent.parent / "Mind.toml").is_file()
    cmd = [mindc, "build"] + ([] if project else [str(src)])
    cmd += ["--target", "cpu", "--emit", "binary",
           "--no-cache", "--out", str(out)]
    cmd += (["--backend", "native", "--native-profile", "bounded"] if native
            else ["--backend", "mlir"])
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=180, cwd=src.parent.parent if project else src.parent)
    except subprocess.TimeoutExpired:
        return 124, "build timeout"
    return r.returncode, (r.stderr or "") + (r.stdout or "")


def _observe_build(rc: int, err: str, exe: pathlib.Path, env_run: dict[str, str]) -> Obs:
    if rc != 0:
        if exe.exists():
            return Obs("build_fail", rc, b"", "artifact exists after nonzero build")
        if "outside candidate native-lane-v1" in err and "No artifact written" in err:
            m = _REFUSAL_RE.search(err)
            return Obs("refused", None, b"", m.group(1) if m else "unnamed")
        return Obs("build_fail", rc, b"", _last(err) or f"build status {rc}")
    if not exe.exists():
        return Obs("build_fail", rc, b"", "build status 0 but no artifact")
    try:
        r = subprocess.run(
            ["/usr/bin/prlimit", "--as=2147483648", "--", str(exe)],
            capture_output=True, timeout=60, env=env_run,
        )
    except subprocess.TimeoutExpired:
        return Obs("timeout", None, b"", "run timeout")
    return Obs("ran", r.returncode, r.stdout, "")


def _observe_eval(mind: str, source: str, env: dict[str, str]) -> Obs:
    text = source if source.endswith("\n") else source + "\n"
    text += "main()\n"
    try:
        r = subprocess.run(
            ["/usr/bin/prlimit", "--as=2147483648", "--", mind, "eval", text],
            capture_output=True, text=True, timeout=60, env=env,
        )
    except subprocess.TimeoutExpired:
        return Obs("timeout", None, b"", "eval timeout")
    err = r.stderr or ""
    if r.returncode != 0:
        return Obs("abort", r.returncode, r.stdout.encode(), err.strip())
    if "Evaluation error" in err:
        return Obs("eval_error", r.returncode, r.stdout.encode(), err.strip())
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return Obs("empty", r.returncode, b"", _last(err))
    try:
        value = int(lines[0], 10)
    except ValueError:
        return Obs("unparsed", r.returncode, b"", lines[0][:120])
    if 0 <= value <= 255:
        return Obs("ran", value, b"", "")
    return Obs("out_of_range", None, b"", f"value={value}")


def _match(obs: Obs, case: Case, path: str) -> bool:
    # Select the evaluator pin BEFORE the common compiled value check. A
    # compiled trap, unrelated panic or loader failure is not this outcome.
    if path == "eval" and case.follows_d1:
        if obs.stdout != b"":
            return False
        pin = case.expect_eval
        if pin == "value2":
            return obs.kind == "ran" and obs.exit_code == 2
        if pin == "zero":
            return (obs.kind == "eval_error" and obs.exit_code == 0
                    and obs.note == "Evaluation error: division by zero")
        if pin in ("narrow_q", "narrow_z"):
            var = pin[-1]
            message = ("implicit narrowing Scalar[i64] -> Scalar[i32] loses data "
                       f"for `{var}`; use an explicit `as Scalar[i32]` cast")
            lines = obs.note.splitlines()
            return (obs.kind == "eval_error" and obs.exit_code == 0
                    and lines[:3] == ["Evaluation error: type error",
                         "[E2004] " + message, "error[type-check][E2004]: " + message])
        if pin in ("div_overflow", "mod_overflow"):
            message = ("attempt to divide with overflow" if pin == "div_overflow"
                       else "attempt to calculate the remainder with overflow")
            return (obs.kind == "abort" and obs.exit_code == 101
                    and message in obs.note.splitlines()
                    and re.search(r"thread 'main'(?: \(\d+\))? panicked at "
                                  r"(?:[^\n]*[/\\])?src/eval/mod.rs:\d+:\d+:",
                                  obs.note) is not None)
        return False
    if path == "native" and case.allow_native_refusal and obs.kind == "refused":
        return obs.note == "lane.type.unsupported"
    return obs.kind == "ran" and obs.exit_code == case.expect_exit and obs.stdout == b""


def _eval_source(case: Case) -> str:
    """Reference closure for this fixed one-helper fixture, not a module resolver."""
    if not case.modules:
        return case.source
    if set(case.modules) != {"helper.mind"} or not case.source.startswith("import helper;\n"):
        raise ValueError("unsupported reference closure")
    return case.modules["helper.mind"] + case.source.removeprefix("import helper;\n")


def self_test() -> int:
    from dataclasses import replace
    cases = {c.cid: c for c in build_corpus()}
    c = cases["edge_i64_div_min_helper"]
    panic = "thread 'main' (123) panicked at src/eval/mod.rs:2759:13:\nattempt to divide with overflow"
    good = Obs("abort", 101, b"", panic)
    checks = [("named_overflow", _match(good, c, "eval"))]
    for name, obs in [("segfault", Obs("abort", -11, b"", "segmentation fault")),
                      ("missing_library", Obs("abort", 127, b"", "missing library")),
                      ("unrelated_exit", Obs("ran", 79, b"", "")),
                      ("wrong_panic", replace(good, note="unrelated panic")),
                      ("wrong_code", replace(good, exit_code=127)),
                      ("stdout", replace(good, stdout=b"unexpected"))]:
        checks.append((name, not _match(obs, c, "eval")))
    checks.append(("mlir_cannot_use_eval_pin", not _match(good, c, "mlir")))
    # Row pins `code`; `code` accepted, every `bad` exit rejected, both compiled paths.
    def pinned(row, code, *bad):
        return row.expect_exit == code and all(
            _match(Obs("ran", code, b"", ""), row, p)
            and not any(_match(Obs("ran", x, b"", ""), row, p) for x in bad)
            for p in ("native", "mlir"))
    # i64 INT_MIN/-1 keeps the compiled-path trap (the narrow flip must not reach it).
    for cid in ("edge_i64_div_min_helper", "edge_i64_mod_min_literal"):
        checks.append((cid + "_compiled_trap", pinned(cases[cid], TRAP_EXIT, 4)))
    for ty in ("i8", "i16", "i32"):
        for op in ("div", "mod"):
            # Narrow MIN/-1 is defined (MIN, remainder 0): value accepted, trap and wrong
            # value (exit 1) rejected; positive control: the zero-divisor row still traps.
            row, zrow = cases[f"edge_{ty}_{op}_min"], cases[f"edge_{ty}_{op}_zero"]
            checks.append((row.cid, pinned(row, NARROW_MIN_EXIT, TRAP_EXIT, 1)))
            checks.append((zrow.cid, pinned(zrow, TRAP_EXIT, NARROW_MIN_EXIT)))
    for cid, good_obs in [
        ("edge_i64_div_zero_helper", Obs("eval_error", 0, b"", "Evaluation error: division by zero")),
        ("edge_i8_div_min", Obs("ran", 2, b"", "")),
    ]:
        checks.append((cid + "_eval_pin", _match(good_obs, cases[cid], "eval")))
        checks.append((cid + "_wrong_eval", not _match(Obs("eval_error", 0, b"", "unrelated error"), cases[cid], "eval")))
    row = cases["edge_i32_div_min"]
    checks.append(("unrelated_type_error", not _match(Obs("eval_error", 0, b"", "Evaluation error: type error\n[E2003] unsupported call"), row, "eval")))
    row = cases["reachable_div_in_import"]
    checks.append((row.cid, "a / b" in row.modules.get("helper.mind", "")
                   and "import helper;" in row.source and "imported_div(21, 3)" in row.source
                   and "a / b" in _eval_source(row)))
    for name, ok in checks:
        gate_assert.check(ok, "contract " + name)
    return 0 if all(ok for _, ok in checks) else 1


def _blocked(msg: str) -> int:
    print(f"BLOCKED: {msg}")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compiler", default=os.environ.get("MINDC_NATIVE_LANE_ELF"))
    ap.add_argument("--paths", default="native,mlir,eval",
                    help="comma-joined subset of {native,mlir,eval}; the "
                         "native-lane ELF is x86-only by construction")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print the corpus and exit without building")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    unknown = set(paths) - {"native", "mlir", "eval"}
    if unknown:
        return _blocked(f"unknown --paths entries: {sorted(unknown)}")
    if not paths:
        return _blocked("--paths selected nothing")
    if "native" not in paths:
        print("native leg: x86-only by construction, not run")
    drift = model_self_check()
    if drift is not None:
        print(f"model self-check failed: {drift}")
        return 1
    try:
        corpus = build_corpus()
    except RuntimeError as err:
        print(f"corpus builder failed: {err}")
        return 1
    traps = sum(1 for c in corpus if c.follows_d1)
    print(f"native_lane_op_differential seed={SEED} programs={len(corpus)} "
          f"trap_rows={traps}")
    if args.list:
        for c in corpus:
            flag = " trap" if c.follows_d1 else ""
            refuse = " refusal-ok" if c.allow_native_refusal else ""
            print(f"  {c.cid} expect_exit={c.expect_exit}{flag}{refuse}")
        return 0
    if not pathlib.Path("/usr/bin/prlimit").exists():
        return _blocked("prlimit not found at /usr/bin/prlimit")
    mindc = os.environ.get("MINDC_BIN", str(REPO / "target" / "release" / "mindc"))
    mindc = str(pathlib.Path(mindc).resolve())
    mind = os.environ.get("MIND_BIN", str(REPO / "target" / "release" / "mind"))
    mind = str(pathlib.Path(mind).resolve())
    compiler = str(pathlib.Path(args.compiler).resolve()) if args.compiler else ""
    if not pathlib.Path(mindc).is_file():
        return _blocked(f"mindc not found at {mindc}")
    if not pathlib.Path(mind).is_file():
        return _blocked(f"mind not found at {mind}")
    if not compiler or not pathlib.Path(compiler).is_file():
        return _blocked(
            "pass --compiler <marked native compiler> "
            "(the committed bounded seed is "
            "examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf)")
    if MARK not in pathlib.Path(compiler).read_bytes():
        return _blocked(f"{compiler} does not carry the native-lane-v1 mark")
    digest = hashlib.sha256(pathlib.Path(compiler).read_bytes()).hexdigest()
    print(f"compiler={compiler} sha256={digest}", flush=True)
    env = _tools(dict(os.environ))
    env["MINDC_NATIVE_ELF"] = compiler
    env["MINDC_STD_DIR"] = str(REPO / "std")
    env["LC_ALL"] = "C"
    env.pop("MINDC_SO", None)
    run_env = {"LC_ALL": "C", "PATH": env.get("PATH", "")}
    failed = 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lane-op-") as td:
        tmp = pathlib.Path(td)
        for index, case in enumerate(corpus):
            case_dir = tmp / case.cid
            case_dir.mkdir()
            source_dir = case_dir
            if case.modules:
                source_dir = case_dir / "src"
                source_dir.mkdir()
                (case_dir / "Mind.toml").write_text(
                    '[package]\nname = "import_div"\nversion = "0.1.0"\n'
                    '[build]\nentry = "src/main.mind"\n')
                subprocess.run(["git", "init", "-q", str(case_dir)], check=True)
            source = source_dir / "main.mind"
            source.write_text(case.source)
            for name, content in case.modules.items():
                (source_dir / name).write_text(content)
            native = _observe_build(*_build(
                mindc, source, case_dir / "native", True, env), case_dir / "native", run_env) if "native" in paths else Obs("skipped", None, b"", "not selected")
            mlir = _observe_build(*_build(
                mindc, source, case_dir / "mlir", False, env), case_dir / "mlir", run_env) if "mlir" in paths else Obs("skipped", None, b"", "not selected")
            ev = _observe_eval(mind, _eval_source(case), env) if "eval" in paths else Obs("skipped", None, b"", "not selected")
            ok = True
            if "native" in paths:
                ok = ok and _match(native, case, "native")
            if "mlir" in paths:
                ok = ok and _match(mlir, case, "mlir")
            if "eval" in paths:
                ok = ok and _match(ev, case, "eval")
            failed += not ok
            gate_assert.check(
                ok,
                f"{case.cid} expect_exit={case.expect_exit} "
                f"native={native.brief()} mlir={mlir.brief()} eval={ev.brief()}",
            )
            if case.follows_d1:
                print(f"  note {case.cid}: expectation follows D1; eval_pin={case.expect_eval}")
            if not ok:
                print(f"  repro {case.cid} expect_exit={case.expect_exit} "
                      f"native={native.brief()} mlir={mlir.brief()} eval={ev.brief()}")
                for name, content in {"main.mind": case.source, **case.modules}.items():
                    print(f"  file {name}: {content!r}")
                for line in case.source.splitlines():
                    print(f"  | {line}")
            print(f"progress index={index} id={case.cid} "
                  f"seconds={time.monotonic() - started:.1f}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
