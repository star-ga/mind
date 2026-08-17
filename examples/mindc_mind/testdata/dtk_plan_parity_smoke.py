"""
DTK slice-1 differential PARITY smoke (audit finding #1).

The `tests/regalloc_dtk_parity.rs` test checks the Rust `plan_module_k` against
HARDCODED expectations — Rust-vs-Rust. Nothing cross-checked the pure-MIND
`selftest_dtk_plan` export in examples/mindc_mind/main.mind against it. This
smoke closes that gap: it builds the self-host cdylib, runs `selftest_dtk_plan`
over a corpus, and diffs its serialized `[count | (id, reg_code)*count]` table,
FIELD-EXACT, against the Rust `plan_module_k(m, 5)` result (filtered to
non-params, top-2). It also pins the eligibility predicate from OUTSIDE (audit
finding #3: the Rust reference has NO eligibility mirror) by asserting
`selftest_dtk_plan` returns count==0 for every ineligible class.

Oracle independence (the gate is real, not a null gate): the Rust
`plan_module_k` computes the ranking on an IRModule via its OWN use-counting +
sort; the pure-MIND side scans SOURCE TOKENS independently. Only the SSA id
namespace (params first, then eval-order defs) is shared — that is the contract
being tested, not an algorithm copied between the two. The oracle tables come
from Rust (`dump_dtk_slice1_parity`, live via cargo, or the pinned fallback that
`tests/regalloc_dtk_parity.rs::parity_corpus_tables` gates), NEVER from a Python
re-implementation of the port.

Run:
  MINDC_SO=<self-host .so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/testdata/dtk_plan_parity_smoke.py

Env:
  MIND_DTK_SKIP_RUST_REGEN=1  -> skip the live cargo oracle regen; use the
                                 pinned (Rust-gated) fallback tables.
"""

import ctypes
import os
import pathlib
import re
import shutil
import subprocess
import sys

_HERE = pathlib.Path(__file__).parent  # examples/mindc_mind/testdata
_MIND_DIR = _HERE.parent  # examples/mindc_mind
_REPO = _MIND_DIR.parent.parent
sys.path.insert(0, str(_MIND_DIR))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()

# reg_code convention the pure-MIND side serializes: 0 = rbx (rank 0), 1 = r12.
# --------------------------------------------------------------------------
# ELIGIBLE corpus: (name, source). Each name has a matching IRModule in
# tests/regalloc_dtk_parity.rs::parity_corpus() with the SAME SSA numbering, so
# the Rust plan_module_k result is the oracle. The .mind SSA numbering seeds ids
# at n_params (params take 0..n_params-1), then int-lit/binop consume ids in
# eval order — the sources below are written to match their Rust twin exactly.
ELIGIBLE = [
    # single non-param def (a+b), used once -> id 2 -> rbx.
    ("add", "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n"),
    # value used twice: x=a+a (id1, used 2x by x+x), y=x+x (id2, used 1x).
    # ranking [1,2] -> (1,rbx),(2,r12).
    ("twice", "fn f(a: i64) -> i64 {\n    let x: i64 = a + a;\n    return x + x;\n}\n"),
    # pure tie: three non-param defs {2,3,4} each used once -> id-ASC tiebreak
    # -> top-2 (2,rbx),(3,r12).
    ("tie", "fn f(a: i64, b: i64) -> i64 {\n    let x: i64 = a + b;\n    let y: i64 = a + b;\n    return x + y;\n}\n"),
    # zero-use def: x=a+b (id2) is never read, still the only candidate -> rbx.
    ("zero_use", "fn f(a: i64, b: i64) -> i64 {\n    let x: i64 = a + b;\n    return a;\n}\n"),
    # single-def fn: return a+a (id1) -> rbx.
    ("single_def", "fn f(a: i64) -> i64 {\n    return a + a;\n}\n"),
]

# Pinned Rust oracle tables (fallback when the live cargo regen is skipped /
# unavailable). Gated by tests/regalloc_dtk_parity.rs::parity_corpus_tables, so
# a stale pin here fails that Rust test — it can never silently drift.
PINNED_ORACLE = {
    "add": [(2, 0)],
    "twice": [(1, 0), (2, 1)],
    "tie": [(2, 0), (3, 1)],
    "zero_use": [(2, 0)],
    "single_def": [(1, 0)],
}

# INELIGIBLE corpus: every class the .mind eligibility predicate must refuse
# (nb_dtk_plan / nb_dtk_scan fail-closed). selftest_dtk_plan MUST return
# count==0 for each — the only external pin on "should this fn be planned at
# all" (audit finding #3; Rust plan_module_k has no eligibility mirror).
INELIGIBLE = [
    ("call", "fn f(a: i64) -> i64 {\n    return dbl(a);\n}\nfn dbl(x: i64) -> i64 {\n    return x + x;\n}\n"),
    ("if", "fn f(a: i64, b: i64) -> i64 {\n    if a > b {\n        return a;\n    }\n    return b;\n}\n"),
    ("while", "fn f(a: i64) -> i64 {\n    let mut i: i64 = a;\n    while i > 0 {\n        i = i - 1;\n    }\n    return i;\n}\n"),
    ("float_param", "fn f(a: f64, b: i64) -> i64 {\n    return b;\n}\n"),
    ("narrow_param", "fn f(a: i32, b: i64) -> i64 {\n    return b;\n}\n"),
    ("div", "fn f(a: i64, b: i64) -> i64 {\n    return a / b;\n}\n"),
    ("shift", "fn f(a: i64, b: i64) -> i64 {\n    return a << b;\n}\n"),
    ("deref_assign", "fn f(a: i64) -> i64 {\n    let mut x: i64 = a;\n    let p: i64 = &x;\n    *p = 5;\n    return x;\n}\n"),
]

# deferred: the ENTRY-FN ineligible class is NOT exercised here because
# selftest_dtk_plan hardcodes is_entry=0 in its nb_dtk_plan call
# (examples/mindc_mind/main.mind). The nb_dtk_plan is_entry gate is therefore
# unreachable through this export. Upgrade path: thread an is_entry flag through
# selftest_dtk_plan (detect the first fn's name == "main", or add a 4th arg),
# then add an ("entry", "fn main() -> i64 { let x: i64 = 1 + 1; return x + x; }")
# fixture below asserting count==0.


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(SO))
    lib.selftest_dtk_plan.restype = ctypes.c_int64
    lib.selftest_dtk_plan.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    return lib


def mind_plan(lib: ctypes.CDLL, src: str) -> tuple[int, list[tuple[int, int]]]:
    """Run selftest_dtk_plan and decode [count | (id, reg_code)*count]."""
    data = src.encode()
    sbuf = ctypes.create_string_buffer(data, len(data))
    out = ctypes.create_string_buffer(4096)
    rd = lambda off: ctypes.cast(ctypes.addressof(out) + off, ctypes.POINTER(ctypes.c_int64))[0]
    count = lib.selftest_dtk_plan(
        ctypes.cast(sbuf, ctypes.c_void_p).value, len(data), ctypes.addressof(out)
    )
    pairs = [(rd(8 + i * 16), rd(8 + i * 16 + 8)) for i in range(count)]
    return count, pairs


def rust_oracle() -> tuple[dict[str, list[tuple[int, int]]], str]:
    """Return (name -> table, provenance). Prefer the LIVE Rust dump over the
    pinned fallback so the oracle can never silently go stale."""
    if os.environ.get("MIND_DTK_SKIP_RUST_REGEN"):
        return dict(PINNED_ORACLE), "pinned fallback (MIND_DTK_SKIP_RUST_REGEN set)"
    cargo = shutil.which("cargo")
    if cargo is None:
        return dict(PINNED_ORACLE), "pinned fallback (cargo not on PATH)"
    cmd = [
        cargo, "test", "--test", "regalloc_dtk_parity",
        "dump_dtk_slice1_parity", "--", "--ignored", "--nocapture",
    ]
    print("  [oracle] regenerating Rust plan_module_k tables (cargo test dump_dtk_slice1_parity) ...")
    try:
        proc = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, timeout=1800)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  [oracle] live regen did not run ({exc}) — using pinned fallback")
        return dict(PINNED_ORACLE), "pinned fallback (live regen failed)"
    out = proc.stdout + proc.stderr
    tables: dict[str, list[tuple[int, int]]] = {}
    for m in re.finditer(r"DTK (\w+): \[(.*?)\]", out):
        name = m.group(1)
        body = m.group(2).strip()
        pairs: list[tuple[int, int]] = []
        for pm in re.finditer(r"\((\d+),(\d+)\)", body):
            pairs.append((int(pm.group(1)), int(pm.group(2))))
        tables[name] = pairs
    if proc.returncode != 0 or not tables:
        tail = "\n".join(out.strip().splitlines()[-8:])
        print(f"  [oracle] live regen failed (rc={proc.returncode}) — using pinned fallback\n{tail}")
        return dict(PINNED_ORACLE), "pinned fallback (live regen failed)"
    print(f"  [oracle] derived {len(tables)} Rust plan_module_k tables live")
    return tables, "live Rust plan_module_k (cargo dump_dtk_slice1_parity)"


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0

    lib = load_lib()
    failures = 0

    oracle, prov = rust_oracle()
    print(f"[eligible corpus: pure-MIND selftest_dtk_plan vs Rust plan_module_k — {prov}]")
    for name, src in ELIGIBLE:
        exp = oracle.get(name)
        if exp is None:
            print(f"  FAIL  {name}: no Rust oracle table (regen incomplete)")
            failures += 1
            continue
        count, got = mind_plan(lib, src)
        ok = got == exp and count == len(exp)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: mind count={count} table={got}  rust={exp}")
        if not ok:
            print(f"        DIVERGENCE: mind {got} (count {count}) != rust {exp}")
            failures += 1

    print("\n[ineligible corpus: selftest_dtk_plan must fail-closed (count==0)]")
    for name, src in INELIGIBLE:
        count, got = mind_plan(lib, src)
        ok = count == 0 and got == []
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: count={count} (expect 0)")
        if not ok:
            print(f"        ELIGIBILITY LEAK: {name} returned count={count} table={got}, expected count==0")
            failures += 1

    if failures:
        print(f"\nFAIL  dtk_plan_parity_smoke: {failures} divergence(s)")
        return 1
    print(
        "\nALL PASS  (pure-MIND selftest_dtk_plan == Rust plan_module_k on every "
        "eligible fixture, field-exact; every ineligible class fails closed count==0; "
        "entry-fn class deferred — see marker)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
