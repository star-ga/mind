#!/usr/bin/env python3
"""CPU-as-oracle smoke for the pure-MIND E2024 self-host-only-call rule.

Ports the resolve.rs advisory (SELF_HOST_ONLY_CALL_CODE, "E2024"): a callee
that `starts_with("__mind_")` but is registered in NO profile of
`STD_SURFACE_INTRINSICS` in src/intrinsics.rs gets flagged as a Warning. A
FrozenNative-only row is registered (arity-checked in any profile): no E2024.

The legacy pure-MIND twin `selftest_tc_self_host_only_call(w0, w1, w2, w3,
len)` remains exported for its five-word ABI. The authoritative twin,
`selftest_tc_self_host_only_call_span(buf, len)`, compares the complete source
byte span so a registered name longer than 32 bytes cannot alias a same-prefix,
same-length mutation.

Oracle construction (machine-checked, no hand table):
  1. Every STD_SURFACE_INTRINSICS row (Rust/MLIR-selectable and FrozenNative-
     only) is PARSED from the Rust source src/intrinsics.rs at run time — table
     drift fails loud.
  2. Every case's expected verdict is recomputed from the exact Rust rule
     (`name.startswith("__mind_") and name not in declared`).
  3. Every case is ALSO driven through the LIVE `mindc check` oracle: a
     triggering source calling the name (at registered arity when known) is
     checked and the presence of "E2024" in the output is asserted to equal
     the rule verdict — guarding both the parsed table and the rule itself.

Corpus: every registered entry (negative), per-entry mutations (suffix /
truncation -> positive), a same-prefix/same-length different-tail negative,
unregistered `__mind_*` names incl. the bare prefix and a >32-byte name, and
non-prefixed controls.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_tc_classify_error_code_smoke.py.
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

# The compiler under test is resolved through the SHARED fail-closed resolver,
# never a bare-name PATH lookup: a PATH `mindc` is another checkout's binary and
# lets this gate report green about a tree that never built one.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _selfhost_so import resolve_mindc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAIN_MIND = os.path.join(HERE, "main.mind")
TYPE_CHECKER = os.path.join(ROOT, "src", "intrinsics.rs")

# Count each three-way case through the shared gate contract.  The local `fails`
# tally remains the control-flow guard, while these verdicts prevent an empty
# or unobserved corpus from publishing one green recap.
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gate_assert import check  # noqa: E402

PREFIX = "__mind_"
LONG_REGISTRY_NAME = "__mind_nerve_blas_matmul_score_q16_i64"
# R-01b register bank rows (seed ids 70-83) and the R-02e epilogue rows (94-99):
# FrozenNative-only, so in E2024's domain.
BANK_ROWS = {f"__mind_ymm_{op}" for op in (
    "vmovdqu_ld", "vmovdqu_st", "vpaddd_i32x8_m", "vpsubd_i32x8_m", "vpaddb_i8x32_m", "vpshufb_i8x32_m",
    "vpaddd_i32x8_r", "vpsubd_i32x8_r", "vpaddb_i8x32_r", "vpshufb_i8x32_r", "vpaddq_i64x4_m",
    "vpsubq_i64x4_m", "vpaddq_i64x4_r", "vpsubq_i64x4_r", "vpmuludq_i64x4_r", "vpand_i64x4_r",
    "vpor_i64x4_r", "vpcmpgtq_i64x4_r", "vpsrlq_i64x4_r", "vpsllq_i64x4_r")}


def parse_registry_source(src):
    """Parse declared and Rust/MLIR-selectable intrinsic tables.

    The registry used to be a tuple table, but its current declarations carry
    contract metadata in constructors.  Parse those declarations directly so
    adding or changing a constructor changes this oracle's input.  Legacy,
    `frozen_value`, `frozen_discard`, and `frozen_io` rows are Rust/MLIR-
    selectable; `frozen_native_*` rows are deliberately native-only and
    therefore remain in E2024's domain.
    """
    m = re.search(
        r"const STD_SURFACE_INTRINSICS: &\[IntrinsicSpec\] = &\[(.*?)\n\];",
        src,
        re.S,
    )
    if not m:
        raise ValueError("STD_SURFACE_INTRINSICS IntrinsicSpec declarations not found")
    body = m.group(1)
    entries = []
    entries.extend(
        ("legacy", name, int(arity))
        for name, arity in re.findall(
            r'IntrinsicSpec::legacy\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*,',
            body,
        )
    )
    arities = {"NO_I64": 0, "ONE_I64": 1, "TWO_I64": 2, "THREE_I64": 3, "FOUR_I64": 4,
               "FIVE_I64": 5, "SIX_I64": 6}
    entries.extend(
        (kind, name, arities[arity])
        for kind, name, arity in re.findall(
            r"IntrinsicSpec::(frozen_native_value|frozen_value|frozen_native_only|"
            r'frozen_discard|frozen_io)\s*\(\s*"([^"]+)"\s*,\s*"[^"]+"\s*,\s*'
            r"(NO_I64|ONE_I64|TWO_I64|THREE_I64|FOUR_I64|FIVE_I64|SIX_I64)\s*,",
            body,
            re.S,
        )
    )
    # Struct-literal rows (a Det::World FrozenNative row has no constructor):
    # native-only iff its profile list does not name RustMlir.
    literals = struct_literal_rows(body)
    entries.extend(
        ("frozen_native_only" if row["native_only"] else "literal", row["name"], row["arity"])
        for row in literals
    )
    declared_kinds = re.findall(
        r"(?m)^\s*IntrinsicSpec::([A-Za-z0-9_]+)\s*\(", body
    ) + ["frozen_native_only"] * len(literals)
    supported_kinds = {
        "legacy",
        "frozen_native_value",
        "frozen_value",
        "frozen_native_only",
        "frozen_discard",
        "frozen_io",
    }
    if any(kind not in supported_kinds for kind in declared_kinds):
        raise ValueError(f"unsupported IntrinsicSpec constructor in registry: {declared_kinds!r}")
    if len(entries) != len(declared_kinds):
        raise ValueError(
            f"registry parser consumed {len(entries)} of {len(declared_kinds)} declarations"
        )
    if len(entries) < 30:
        raise ValueError(f"implausibly small parsed table ({len(entries)} entries)")
    declared = {name: int(arity) for _, name, arity in entries}
    if len(declared) != len(entries):
        raise ValueError("duplicate names in STD_SURFACE_INTRINSICS")
    rust_mlir = {
        name: int(arity)
        for kind, name, arity in entries
        if kind not in {"frozen_native_value", "frozen_native_only"}
    }
    if len(rust_mlir) < 30:
        raise ValueError(
            f"implausibly small Rust/MLIR-selectable table ({len(rust_mlir)} entries)"
        )
    long_names = sorted(name for name in declared if len(name.encode()) > 32)
    if long_names != [LONG_REGISTRY_NAME]:
        raise ValueError(
            "FAIL: unsupported >32-byte registry rows; regenerate the full-span "
            f"twin for {long_names!r}"
        )
    return rust_mlir, declared


def struct_literal_rows(body):
    """Rows written as `IntrinsicSpec { physical_name, arity, det, contract }`."""
    rows = []
    for m in re.finditer(r"(?m)^\s*IntrinsicSpec\s*\{(.*?)^    \},", body, re.S):
        text = m.group(1)
        name = re.search(r'physical_name:\s*"([^"]+)"', text)
        arity = re.search(r"arity:\s*(\d+)", text)
        if not name or not arity:
            raise ValueError(f"unparsed IntrinsicSpec literal: {text[:80]!r}")
        seed = re.search(r"seed_id:\s*(\d+)", text)
        rows.append({
            "name": name.group(1),
            "arity": int(arity.group(1)),
            "seed": int(seed.group(1)) if seed else None,
            "world": "Det::World" in text,
            "native_only": "RustMlir" not in text and "FROZEN_AND_RUST_MLIR" not in text,
        })
    return rows


def seed_rows(src):
    """{name: (seed_id, det_world)} for every FrozenNative row with a seed id."""
    body = re.search(
        r"const STD_SURFACE_INTRINSICS: &\[IntrinsicSpec\] = &\[(.*?)\n\];", src, re.S
    ).group(1)
    rows = {
        name: (int(seed), False)
        for name, seed in re.findall(
            r'IntrinsicSpec::frozen_\w+\s*\(\s*"([^"]+)"[^;]*?,\s*(\d+)\s*,?\s*\)', body, re.S
        )
    }
    for row in struct_literal_rows(body):
        if row["seed"] is not None:
            rows[row["name"]] = (row["seed"], row["world"])
    return rows


def parse_table_source(src):
    """Parse the exact table whose members suppress E2024."""
    return parse_registry_source(src)[0]


def registry_source():
    with open(TYPE_CHECKER, encoding="utf-8") as registry:
        return registry.read()


def parse_table():
    """Parse Rust/MLIR-selectable declarations from the Rust source."""
    try:
        return parse_table_source(registry_source())
    except ValueError as error:
        print(f"FAIL: {error} in {TYPE_CHECKER}")
        sys.exit(1)


def registry_mutation_control(source, declared, span_fn):
    """Prove a changed declaration is observed by this oracle.

    This is an in-memory mutation: the compiler under test remains untouched,
    while a changed physical name must change the rule verdict for the real
    registered call.  If the parser silently falls back to a stale list, this
    control fails.
    """
    declaration = 'IntrinsicSpec::legacy("__mind_blas_dot_f32", 3, Det::Pure)'
    mutated_source = source.replace(
        declaration,
        'IntrinsicSpec::legacy("__mind_blas_dot_f32_mutated", 3, Det::Pure)',
        1,
    )
    if mutated_source == source:
        raise SystemExit("FAIL: registry mutation did not find selectable declaration")
    try:
        mutated = parse_registry_source(mutated_source)[1]
    except ValueError as error:
        raise SystemExit(f"FAIL: registry mutation cannot be parsed: {error}") from error
    name = "__mind_blas_dot_f32"
    name_bytes = name.encode()
    name_buf = ctypes.create_string_buffer(name_bytes, len(name_bytes))
    actual = span_fn(ctypes.addressof(name_buf), len(name_bytes))
    observed = check(
        actual == rust_rule(name, declared) and actual != rust_rule(name, mutated),
        "changed IntrinsicSpec declaration changes the self-host registry verdict",
    )
    if not observed:
        raise SystemExit("FAIL: self-host registry parser is stale under declaration mutation")


def rust_rule(name, declared):
    """The exact resolve.rs decision: prefixed AND registered in no profile -> 1."""
    return 1 if (name.startswith(PREFIX) and name not in declared) else 0


def enc(name):
    """Pack first 32 bytes LE into 4 i64 words + TRUE length."""
    b = name.encode()
    words = [
        int.from_bytes(b[i : i + 8].ljust(8, b"\0"), "little")
        for i in (0, 8, 16, 24)
    ]
    return words + [len(b)]


def live_oracle(mindc, name, declared, workdir):
    """Run `mindc check` on a source calling `name`; 1 iff E2024 is emitted."""
    arity = declared.get(name, 1)
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


def build_cases(table, declared):
    cases = []
    # Every Rust/MLIR-selectable entry: must be 0 (no E2024).
    for name in sorted(table):
        cases.append((name, "Rust/MLIR-selectable entry"))
    # FrozenNative-only rows are registered (any-profile arity): also 0.
    for name in sorted(set(declared) - set(table)):
        cases.append((name, "declared FrozenNative-only entry"))
    # Per-entry mutations: suffix + last-char truncation (positives for the
    # __mind_-prefixed ones; the truncation of a prefixed name stays prefixed
    # unless it collides with another registered entry).
    for name in sorted(declared):
        if name.startswith(PREFIX):
            cases.append((name + "_x", "registered + suffix"))
            cases.append((name[:-1], "registered truncated by 1"))
    # Unregistered prefixed names.
    cases.append(("__mind_", "bare prefix, len 7"))
    cases.append(("__mind_bogus_thing", "unregistered prefixed"))
    cases.append(("__mind_alloc2", "registered + digit"))
    cases.append(("__mind_vpadd_i32x8", "misspelled native-only row"))
    cases.append(
        (
            "__mind_blas_matmul_rmajor_f32_v_extended_long",
            "unregistered prefixed name, len > 32",
        )
    )
    cases.append(
        (
            "__mind_nerve_blas_matmul_score_q17_i64",
            "same first 32 bytes and length as registered long name, tail differs",
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


# Seed ids >= 60 are the native-only rows of the combined train (kind ledger:
# Rust seed_id == self-host intrinsic kind). Each must exist on all four sides.
MARKED_SEED_MIN = 60
TYPES_RS = os.path.join(ROOT, "src", "ir", "native_lane_profile", "types.rs")


def bijection(marked):
    """Registry seed id >= 60 <-> `kind == N` arm in nb_*.mind <-> types.rs arm."""
    parts = {
        f: open(os.path.join(HERE, f), encoding="utf-8").read()
        for f in sorted(os.listdir(HERE))
        if f.startswith("nb") and f.endswith(".mind") and f != "nb_part_dispatch.mind"
    }
    text = "\n".join(parts.values())
    with open(TYPES_RS, encoding="utf-8") as f:
        types_rs = f.read()
    seeds = {seed: name for name, (seed, _) in marked.items()}
    bad = [
        f"{name} (seed {seed}): no `kind == {seed}` arm in examples/mindc_mind/nb_*.mind"
        for name, (seed, _) in sorted(marked.items())
        if not re.search(rf"\bkind == {seed}\b", text)
    ]
    bad += [f"{name}: no lane-v1 arm in {TYPES_RS}" for name in sorted(marked)
            if f'"{name}"' not in types_rs]
    for kind in sorted({int(k) for k in re.findall(r"\bkind == (\d+)\b", text)}):
        if kind >= MARKED_SEED_MIN and kind not in seeds:
            bad.append(f"kind {kind}: self-host arm has no registry row (seed id {kind})")
    ok = check(not bad, f"bijection over {len(marked)} seed>=60 rows: " + "; ".join(bad))
    if not ok:
        raise SystemExit("FAIL: native-only intrinsic bijection")


def cli_probes(mindc, marked, declared, workdir):
    """--emit-mic3 twice + --emit-evidence on each seed>=60 row, plus a misspelling.

    Every registered native-only row must exit 0 on all three runs, intern the
    name in the mic@3 bytes, and give two byte-identical --emit-mic3 runs
    (trace_hash = sha256 of those bytes). The misspelled control must warn E2024
    and every path must refuse with nothing written.
    """
    fails = emitted = 0
    control = "__mind_vpadd_i32x8"
    for name, (_, world) in sorted(marked.items()) + [(control, (0, False))]:
        args = ", ".join(["1"] * declared.get(name, 2))
        src = os.path.join(workdir, "probe.mind")
        with open(src, "w") as f:
            f.write(f"fn main() -> i64 {{\n    let v: i64 = {name}({args});\n    return v;\n}}\n")
        e2024 = live_oracle(mindc, name, declared, workdir)
        outs = []
        nondet = ["--allow-nondeterministic"] if world else []
        for flag, extra in [("--emit-mic3", []), ("--emit-mic3", []), ("--emit-evidence", nondet)]:
            out = os.path.join(workdir, f"probe{len(outs)}.bin")
            if os.path.exists(out):
                os.unlink(out)
            r = subprocess.run([mindc, src, flag, out, *extra], capture_output=True)
            outs.append((r.returncode, open(out, "rb").read() if os.path.exists(out) else None))
        if name == control:
            good = e2024 == 1 and all(rc != 0 and b is None for rc, b in outs)
            what = "E2024=1, every artifact path refuses, nothing written"
        else:
            good = e2024 == 0 and outs[0] == outs[1] and all(
                rc == 0 and b and name.encode() in b for rc, b in outs
            )
            what = f"E2024={e2024}, rc 0, name interned, two --emit-mic3 byte-identical"
            emitted += good
        if not check(good, f"cli {name}: {what} (rcs {[rc for rc, _ in outs]})"):
            fails += 1
    print(f"native-only rows carried by --emit-mic3/--emit-evidence: {emitted} of {len(marked)}")
    if not check(len(marked) > 0 and emitted == len(marked), "every seed>=60 row carried"):
        fails += 1
    return fails


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so, False
    mindc = resolve_mindc()
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
    source = registry_source()
    table, declared = parse_registry_source(source)
    native_only = set(declared) - set(table)
    marked = {n: sw for n, sw in seed_rows(source).items() if sw[0] >= MARKED_SEED_MIN}
    # argc/argv carry both profiles (the MLIR path calls the runtime pair), so
    # the native-only partition is exactly the marked rows and the bank rows.
    if native_only != set(marked) | BANK_ROWS or not {"__mind_argc", "__mind_argv"} <= set(table):
        raise SystemExit(
            "FAIL: unexpected FrozenNative-only E2024 partition: "
            f"{sorted(native_only)!r}"
        )
    bijection(marked)
    print(
        f"table: {len(table)} Rust/MLIR-selectable of {len(declared)} "
        "STD_SURFACE_INTRINSICS entries parsed from intrinsics.rs"
    )
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
    span_fn = lib.selftest_tc_self_host_only_call_span
    span_fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    span_fn.restype = ctypes.c_int64
    # Keep the historical five-word ABI live while the span ABI handles the
    # complete identifier.  This is a symbol/behavior probe, not the oracle.
    if fn(*enc("__mind_alloc")) != 0 or fn(*enc("__mind_argc")) != 0:
        print("FAIL: legacy selftest_tc_self_host_only_call ABI drifted")
        sys.exit(1)
    registry_mutation_control(source, declared, span_fn)

    mindc = resolve_mindc()
    cases = build_cases(table, declared)
    total = fails = positives = negatives = 0
    with tempfile.TemporaryDirectory() as workdir:
        for name, note in cases:
            w = enc(name)
            name_bytes = name.encode()
            name_buf = ctypes.create_string_buffer(name_bytes, len(name_bytes))
            got = span_fn(ctypes.addressof(name_buf), len(name_bytes))
            exp = rust_rule(name, declared)
            live = live_oracle(mindc, name, declared, workdir)
            total += 1
            ok = check(
                got == exp == live,
                f"{name!r}: got={got} rule={exp} live={live} ({note})",
            )
            if exp == 1:
                positives += 1
            else:
                negatives += 1
            if not ok:
                fails += 1
            if not ok:
                print(
                    f"  DIFF got={got} rule={exp} live={live} len={w[4]:>2} "
                    f"{name!r} | {note}"
                )

        fails += cli_probes(mindc, marked, declared, workdir)
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
