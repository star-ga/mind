#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND classify_error_code router.

Ports the order-sensitive message/code router in src/type_checker/mod.rs
(`fn classify_error_code`, ~5622-5653): a type-error message can satisfy
several substring/prefix guards at once, and the FIRST matching branch wins,
selecting the emitted E-code.

The pure-MIND twin `selftest_tc_classify_error_code(c1..c9)` takes the nine
per-branch condition results in the exact Rust order and returns the winning
branch index (0..8, or 9 for the generic fallthrough). This Python oracle
recomputes BOTH sides from real diagnostic messages:

  * cond_flags(msg)  -> the nine 1/0 flags fed to the .mind export, each the
    exact Rust guard (contains / starts_with, two-part ANDs pre-combined).
  * rust_classify(msg) -> the same if-else chain re-derived here, returning the
    branch index; this is the independent oracle the .mind result must equal.

Order-sensitivity is the crux, so the corpus deliberately includes messages
that satisfy multiple branches (the `@` matmul msg also matches the generic
"inner"+"dimension" branch; the "elementwise broadcast" msg also matches the
bare "broadcast" branch) — the port must reproduce which branch wins.

Non-vacuous: every branch index 0..9 is hit at least once (>=1 "positive"
non-generic verdict AND >=1 generic fallthrough control), asserted below.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_shape_rules_smoke.py.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")

# Rust constant markers (verbatim substrings/prefixes from classify_error_code).
AT_MARK = "`@`"
INNER_DIM_PHRASE = "inner dimension"
ELEMENTWISE = "elementwise"
BROADCAST = "broadcast"
INNER = "inner"
DIMENSION = "dimension"
RANK_MISMATCH = "rank mismatch"
FN_PREFIX = "function `"
ARGS_GOT = "argument(s); got"
NONEXH_PREFIX = "non-exhaustive `match`"
ARM_PREFIX = "match arm type class mismatch"
FIXED_BYTES = "fixed-size `bytes[N]` buffer handle"


def cond_flags(msg):
    """The nine ordered branch guards from classify_error_code, as 1/0 ints.

    Two-part branches (c1,c2,c6) pre-combine their AND exactly as the Rust
    source does; the rest are a single contains/starts_with."""
    c1 = 1 if (AT_MARK in msg and INNER_DIM_PHRASE in msg) else 0
    c2 = 1 if (ELEMENTWISE in msg and BROADCAST in msg) else 0
    c3 = 1 if (INNER in msg and DIMENSION in msg) else 0
    c4 = 1 if BROADCAST in msg else 0
    c5 = 1 if RANK_MISMATCH in msg else 0
    c6 = 1 if (msg.startswith(FN_PREFIX) and ARGS_GOT in msg) else 0
    c7 = 1 if msg.startswith(NONEXH_PREFIX) else 0
    c8 = 1 if msg.startswith(ARM_PREFIX) else 0
    c9 = 1 if FIXED_BYTES in msg else 0
    return (c1, c2, c3, c4, c5, c6, c7, c8, c9)


def rust_classify(msg):
    """Independent re-derivation of the Rust if-else chain -> branch index."""
    if AT_MARK in msg and INNER_DIM_PHRASE in msg:
        return 0
    elif ELEMENTWISE in msg and BROADCAST in msg:
        return 1
    elif INNER in msg and DIMENSION in msg:
        return 2
    elif BROADCAST in msg:
        return 3
    elif RANK_MISMATCH in msg:
        return 4
    elif msg.startswith(FN_PREFIX) and ARGS_GOT in msg:
        return 5
    elif msg.startswith(NONEXH_PREFIX):
        return 6
    elif msg.startswith(ARM_PREFIX):
        return 7
    elif FIXED_BYTES in msg:
        return 8
    else:
        return 9


# Branch index -> the actual E-code / rule-id the Rust router emits (for logs).
CODE = {
    0: "shape::matmul_mismatch",
    1: "shape::broadcast_mismatch",
    2: "E2103",
    3: "E2101",
    4: "E2102",
    5: "E2005",
    6: "match::non_exhaustive",
    7: "match::arm_mismatch",
    8: "E2006",
    9: "E2001",
}

# Corpus of real-shaped diagnostic messages. `note` flags the order-sensitive
# overlaps so a regression in branch ordering is obvious in the log.
CASES = [
    # --- one clean representative per branch ---
    ("`@` operator: lhs inner dimension 3 does not match rhs outer dimension 4",
     "c1: @ matmul (also matches c3 inner+dimension -> must NOT be 2)"),
    ("elementwise `+`: cannot broadcast shapes [2,3] and [4,5]",
     "c2: elementwise broadcast (also matches c4 broadcast -> must NOT be 3)"),
    ("tensor.matmul: inner dimension 3 != 4",
     "c3: legacy matmul inner/dimension, no `@` -> E2103"),
    ("cannot broadcast shapes [2] and [3]",
     "c4: bare broadcast -> E2101"),
    ("shape rank mismatch: 2 vs 3",
     "c5: rank mismatch -> E2102"),
    ("function `foo` expects 2 argument(s); got 3",
     "c6: arity, starts function` + argument(s); got -> E2005"),
    ("non-exhaustive `match`: variant `Baz` not covered",
     "c7: non-exhaustive match -> match::non_exhaustive"),
    ("match arm type class mismatch: int tag vs float payload",
     "c8: arm class mismatch -> match::arm_mismatch"),
    ("fixed-size `bytes[N]` buffer handle cannot flow into `bytes`",
     "c9: fixed buffer into vec -> E2006"),
    ("type mismatch: expected `i32`, found `f32`",
     "generic fallthrough -> E2001"),
    # --- extra order-sensitivity / faithfulness controls ---
    ("`@` operator inner dimension broadcast rank mismatch soup",
     "c1 wins over c2/c3/c4/c5 all present -> 0"),
    ("elementwise `*`: inner dimension broadcast rank mismatch",
     "c2 wins over c3/c4/c5 (no `@`) -> 1"),
    ("inner dimension differ; also a rank mismatch here",
     "c3 wins over c5 -> 2"),
    ("broadcast failed and rank mismatch present",
     "c4 wins over c5 -> 3"),
    ("closure expects 1 argument(s); got 2",
     "starts_with faithfulness: NOT function` -> fallthrough 9, not 5"),
    ("the value is non-exhaustive `match` somewhere",
     "starts_with faithfulness: prefix not at start -> fallthrough 9, not 6"),
    ("function `bar` has a problem but no arity phrase",
     "c6 needs BOTH parts: function` without argument(s);got -> 9"),
    ("a match arm type class mismatch mentioned mid-sentence",
     "starts_with faithfulness: not at start -> 9, not 7"),
    ("plain unrelated error about ownership",
     "generic fallthrough -> E2001"),
]


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    cmd = [mindc, MAIN_MIND, "--emit-shared", out]
    print("BUILD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        sys.exit(1)
    return out, True


def main():
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)
    fn = lib.selftest_tc_classify_error_code
    fn.argtypes = [ctypes.c_int64] * 9
    fn.restype = ctypes.c_int64

    total = fails = 0
    seen = set()
    for msg, note in CASES:
        flags = cond_flags(msg)
        got = fn(*flags)
        exp = rust_classify(msg)
        # sanity: flags-derived winner must equal the independent oracle
        total += 1
        seen.add(exp)
        mark = "ok " if got == exp else "DIFF"
        if got != exp:
            fails += 1
        print(f"  {mark} idx got={got}({CODE[got]}) exp={exp}({CODE[exp]}) "
              f"flags={flags} | {note}")

    # Non-vacuous gate: every branch (incl. generic fallthrough 9) exercised,
    # so >=1 positive (non-9) AND >=1 negative control (==9) are guaranteed.
    positives = sum(1 for i in seen if i != 9)
    negatives = 1 if 9 in seen else 0
    print(f"classify: cases={total} distinct_branches={sorted(seen)} "
          f"positives={positives} generic_controls={negatives} fails={fails}")
    if positives < 1:
        print("FAIL: vacuous (no non-generic branch exercised)")
        sys.exit(1)
    if negatives < 1:
        print("FAIL: vacuous (no generic fallthrough control)")
        sys.exit(1)
    if seen != set(range(10)):
        print(f"FAIL: incomplete branch coverage, missing "
              f"{sorted(set(range(10)) - seen)}")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND classify_error_code diverges from Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()