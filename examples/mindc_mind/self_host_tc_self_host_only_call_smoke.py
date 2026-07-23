#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2024 self-host-only-call rule.

Ports the resolve.rs advisory (SELF_HOST_ONLY_CALL_CODE, "E2024"): a callee
that `starts_with("__mind_")` but is NOT one of the registered, arity-checked
`STD_SURFACE_INTRINSICS` entries in src/type_checker/mod.rs gets flagged as a
Warning — the Rust/MLIR backends cannot emit that call.

The pure-MIND twin `selftest_tc_self_host_only_call(w0, w1, w2, w3, len)`
takes the callee name's first 32 bytes packed little-endian into four i64
words plus the TRUE byte length (for len <= 32 the tuple is a bijection with
the name; for len > 32 length inequality alone rejects every table entry —
the longest registered name is 31 bytes — so the encoding is exact over the
whole identifier space).

Oracle construction (machine-checked, no hand table):
  1. The STD_SURFACE_INTRINSICS table (name, arity) is PARSED from the Rust
     source src/type_checker/mod.rs at run time — table drift fails loud.
  2. Every case's expected verdict is recomputed from the exact Rust rule
     (`name.startswith("__mind_") and name not in table`).
  3. Every case is ALSO driven through the LIVE `mindc check` oracle: a
     triggering source calling the name (at registered arity when known) is
     checked and the presence of "E2024" in the output is asserted to equal
     the rule verdict — guarding both the parsed table and the rule itself.

Corpus: all 33 registered entries (negative), per-entry mutations (suffix /
truncation -> positive), unregistered `__mind_*` names incl. the bare prefix
and a >32-byte name, and non-prefixed controls.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_classify_error_code_smoke.py.
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAIN_MIND = os.path.join(HERE, "main.mind")
TYPE_CHECKER = os.path.join(ROOT, "src", "type_checker", "mod.rs")

PREFIX = "__mind_"


def parse_table():
    """Parse STD_SURFACE_INTRINSICS from the Rust source (name -> arity)."""
    src = open(TYPE_CHECKER).read()
    m = re.search(
        r"const STD_SURFACE_INTRINSICS: &\[\(&str, usize\)\] = &\[(.*?)\n\];",
        src,
        re.S,
    )
    if not m:
        print("FAIL: STD_SURFACE_INTRINSICS table not found in", TYPE_CHECKER)
        sys.exit(1)
    entries = re.findall(r'\("([^"]+)",\s*(\d+)\)', m.group(1))
    if len(entries) < 30:
        print(f"FAIL: implausibly small parsed table ({len(entries)} entries)")
        sys.exit(1)
    return {name: int(arity) for name, arity in entries}


def rust_rule(name, table):
    """The exact resolve.rs decision: prefixed AND unregistered -> 1."""
    return 1 if (name.startswith(PREFIX) and name not in table) else 0


def enc(name):
    """Pack first 32 bytes LE into 4 i64 words + TRUE length."""
    b = name.encode()
    words = [
        int.from_bytes(b[i : i + 8].ljust(8, b"\0"), "little")
        for i in (0, 8, 16, 24)
    ]
    return words + [len(b)]


def live_oracle(mindc, name, table, workdir):
    """Run `mindc check` on a source calling `name`; 1 iff E2024 is emitted."""
    arity = table.get(name, 1)
    args = ", ".join(["1"] * arity) if arity else ""
    src = f"fn main() -> i64 {{\n    let p: i64 = {name}({args});\n    return 0;\n}}\n"
    path = os.path.join(workdir, "case.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run(
        [mindc, "check", path], capture_output=True, text=True
    )
    out = r.stdout + r.stderr
    return 1 if "E2024" in out else 0


def build_cases(table):
    cases = []
    # Every registered entry: must be 0 (no E2024).
    for name in sorted(table):
        cases.append((name, "registered entry"))
    # Per-entry mutations: suffix + last-char truncation (positives for the
    # __mind_-prefixed ones; the truncation of a prefixed name stays prefixed
    # unless it collides with another registered entry).
    for name in sorted(table):
        if name.startswith(PREFIX):
            cases.append((name + "_x", "registered + suffix"))
            cases.append((name[:-1], "registered truncated by 1"))
    # Unregistered prefixed names.
    cases.append(("__mind_", "bare prefix, len 7"))
    cases.append(("__mind_bogus_thing", "unregistered prefixed"))
    cases.append(("__mind_alloc2", "registered + digit"))
    cases.append(
        (
            "__mind_blas_matmul_rmajor_f32_v_extended_long",
            "prefixed, len > 32 (word truncation path)",
        )
    )
    # Non-prefixed controls (never E2024 regardless of resolvability).
    cases.append(("main", "plain fn name"))
    cases.append(("mind_alloc", "missing leading underscores"))
    cases.append(("_mind_alloc", "single leading underscore"))
    cases.append(("x__mind_", "prefix not at start"))
    cases.append(("__mind", "len 6, one short of the prefix"))
    cases.append(("byte", "registered non-prefixed entry"))
    return cases


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
    table = parse_table()
    print(f"table: {len(table)} STD_SURFACE_INTRINSICS entries parsed from mod.rs")
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)
    fn = lib.selftest_tc_self_host_only_call
    fn.argtypes = [ctypes.c_int64] * 5
    fn.restype = ctypes.c_int64

    mindc = os.environ.get("MINDC_BIN", "mindc")
    cases = build_cases(table)
    total = fails = positives = negatives = 0
    with tempfile.TemporaryDirectory() as workdir:
        for name, note in cases:
            w = enc(name)
            got = fn(*w)
            exp = rust_rule(name, table)
            live = live_oracle(mindc, name, table, workdir)
            total += 1
            ok = got == exp == live
            if exp == 1:
                positives += 1
            else:
                negatives += 1
            mark = "ok " if ok else "DIFF"
            if not ok:
                fails += 1
            print(
                f"  {mark} got={got} rule={exp} live={live} len={w[4]:>2} "
                f"{name!r} | {note}"
            )

    print(
        f"self_host_only_call: cases={total} positives={positives} "
        f"negatives={negatives} fails={fails}"
    )
    if positives < 1 or negatives < 1:
        print("FAIL: vacuous corpus")
        sys.exit(1)
    if fails:
        print("FAIL: pure-MIND E2024 core diverges from the Rust oracle")
        sys.exit(1)
    print("ALL PASS")
    if built:
        try:
            os.unlink(so)
        except OSError:
            pass


if __name__ == "__main__":
    main()
