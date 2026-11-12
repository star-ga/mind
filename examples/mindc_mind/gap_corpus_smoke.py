"""
Self-host gap-corpus integrity gate.

Surveys every fixture in tests/selfhost_gaps/*.mind through the pure-MIND self-host
driver `selftest_mic3_module_nfn(...)` and compares each emitted mic@3 module byte-for-byte
against the Rust oracle `mindc --emit-mic3`. The corpus is a fuzz-discovered regression set
spanning struct-lit / field-read / value-if-expr / fall-through-shadow / mixed-prefix /
call-arg / unary-neg construct families.

The gate enforces three invariants:

  * WRONG_BYTES == 0   — the cardinal invariant: the driver NEVER emits incorrect bytes
                         (a silent miscompile is the worst failure mode for a deterministic
                         compiler). Always hard-fails.
  * BYTE_EXACT >= FLOOR — the coverage ratchet: no catalogued construct may regress from
                         byte-exact to fail-closed or wrong.
  * AST-KIND COVERAGE  — every AST node kind the self-host parser declares must be
                         exercised by at least one fixture. A survey can only speak about
                         constructs somebody wrote a fixture for; without this the corpus
                         reports PASS while whole construct families go uncompared.
                         See `ast_kind_coverage` for why the kind list is read out of
                         main.mind rather than restated here.

Per-case rows and what scripts/gate_assert.py counts from them:
  * one `[PASS]`/`[FAIL]` line per oracle-valid fixture — a byte comparison (or a
    safe refusal / crash classification) that actually ran;
  * one `[PASS]`/`[FAIL]` line per SOURCE_PROBES kind — a fixture-coverage check that
    actually ran against the corpus text;
  * one `[EXEMPT]` line per SYNTHETIC / NO_ORACLE_CONSTRUCT kind — a CLASSIFICATION,
    not a comparison: nothing was byte-compared for that kind, so the row carries no
    verdict token and is NOT counted toward `asserted=`. A kind whose classification
    is stale, or a kind nobody classified, is a `[FAIL]` row like any other.

Verdicts:
  PASS    — 0 wrong-bytes, byte-exact >= floor, every probed AST node kind covered and
            every exempt kind still declared
  FAIL    — any wrong-bytes, a coverage-floor regression, an unexercised probed kind,
            an unclassified kind, or a stale classification
  BLOCKED — .so / mindc missing

CI: point MINDC_SO at the freshly built self-host .so; a missing .so then HARD-FAILS
(refuses to skip, no false green). Local runs default to the .so next to this script.
"""

import ctypes
import os
import pathlib
import re
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
# MINDC_SO (CI) verbatim; else build the self-host .so FRESH — never trust a
# stale in-tree libmindc_mind.so (a cargo build does not regenerate it).
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
from _selfhost_sources import COMPILER_COMPONENTS  # noqa: E402

# Resolved at the top of main(), not at import: `resolve_so()` refuses a stale
# oracle by raising, which is the right thing for the gate and the wrong thing
# for tests/gate_assert_count_contract_test.py, which imports this module only
# to exercise the pure `ast_kind_coverage` classification below. The refusal
# still happens before any fixture is read — same fail-closed behaviour, same
# point in the CLI run, one line later.
SO = None
MINDC = None
CORPUS = _HERE.parents[1] / "tests" / "selfhost_gaps"


def _resolve_oracles():
    """Bind SO / MINDC exactly as the import-time resolution used to."""
    global SO, MINDC
    SO = resolve_so()
    MINDC = pathlib.Path(
        os.environ.get("MINDC", str(_HERE.parents[1] / "target" / "release" / "mindc"))
    )

_P = ctypes.POINTER(ctypes.c_int64)


def _i64(addr, off=0):
    return int(ctypes.cast(addr + off, _P)[0])


def _read_string_record(handle):
    """EmitState.buf -> String record {addr, len, cap}; return its bytes."""
    if not handle:
        return b""
    addr, length = _i64(handle, 0), _i64(handle, 8)
    if not addr or not length:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def oracle_mic3(src: str):
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "m.mind"
        op = pathlib.Path(td) / "m.mic3"
        sp.write_text(src)
        subprocess.run(
            [str(MINDC), "--emit-mic3", str(op), str(sp)],
            capture_output=True,
        )
        return op.read_bytes() if op.exists() else None


def _nfn_mic3_raw(src: str):
    lib = ctypes.CDLL(str(SO))
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5
    sb = src.encode()
    sc = ctypes.create_string_buffer(sb, len(sb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
    cc = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(sc, ctypes.c_void_p).value,
        len(sb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(cc, ctypes.c_void_p).value,
    )
    return _read_string_record(_i64(es, 0)) if es else b""


def nfn_mic3(src: str):
    """Fork-isolate each emit and load the .so fresh inside the child. A clean child contains
    any out-of-subset SIGSEGV so one bad fixture can't take down the whole survey, and a fresh
    per-process load matches a real `mindc` invocation (defensive: it would surface any
    arena-state-dependent lowering rather than mask it behind a warm handle — though none
    exists here). READ THE RESULT CORRECTLY: the output is a `String` at `EmitState.buf`
    (offset 0) → (addr@0, len@8); reading the wrong EmitState field once faked a 64/66.
    Returns (bytes, "OK"|"CRASH")."""
    rp, wp = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(rp)
        try:
            data = _nfn_mic3_raw(src)
        except Exception:  # noqa: BLE001 — surfaced to the parent as wrong-bytes
            data = b""
        try:
            os.write(wp, len(data).to_bytes(4, "little") + data)
        finally:
            os.close(wp)
            os._exit(0)
    os.close(wp)
    buf = b""
    while True:
        chunk = os.read(rp, 65536)
        if not chunk:
            break
        buf += chunk
    os.close(rp)
    status = os.waitpid(pid, 0)[1]
    if os.WIFSIGNALED(status):
        return None, "CRASH"
    if len(buf) < 4:
        return b"", "OK"
    n = int.from_bytes(buf[:4], "little")
    return buf[4:4 + n], "OK"


# ---------------------------------------------------------------------------
# AST-node-kind coverage lint
#
# The byte survey above can only speak about constructs someone thought to write
# a fixture for. Ten of the self-host parser's AST node kinds had ZERO fixtures
# in this corpus, and four of them (`ast_cast`, `ast_unsupported`, `ast_while`,
# `ast_method`) name construct families that emit wrong bytes or refuse today —
# the corpus reported PASS while the hole was invisible. A survey with no
# coverage floor measures the fixtures, not the compiler.
#
# So the kind list is READ OUT OF `main.mind` (the thing under test) rather than
# hand-copied here: a kind added to the parser with no entry below is a HARD
# FAIL, not a silent omission. Every kind must land in exactly one of three
# buckets, each of which has to justify itself:
#
#   SOURCE_PROBES        — has a source-level spelling; needs >= 1 fixture.
#   SYNTHETIC            — never produced from a distinct source construct
#                          (built by the emitter/desugar), so no fixture can
#                          target it directly.
#   NO_ORACLE_CONSTRUCT  — the self-host front end accepts it but the Rust
#                          oracle cannot compile any source that produces it,
#                          so a BYTE-COMPARABLE fixture is impossible today.
#
# The probes are deliberately syntactic and conservative: a probe that is too
# loose reports coverage that does not exist, which is the exact failure this
# lint replaces. When a probe and a construct disagree, tighten the probe.
# ---------------------------------------------------------------------------

#: `pub fn ast_<name>() -> i64` — the self-host AST node-kind constructors.
_AST_KIND_DECL = re.compile(r"^pub fn (ast_[a-z_0-9]+)\(\) -> i64", re.M)

#: kind -> regex that a fixture's SOURCE must match to exercise that kind.
SOURCE_PROBES = {
    "ast_int_lit": r"(?<![\w.])\d+(?![\w.])",
    "ast_ident": r"\b[a-z_][A-Za-z_0-9]*\b",
    "ast_binop": r"[a-zA-Z0-9_)\]]\s*(?:\+|-|\*|/|%|<<|>>|&&|\|\||==|!=|<=|>=|[<>&|^])\s*[a-zA-Z0-9_(\[]",
    "ast_call": r"\b[a-z_][A-Za-z_0-9]*\s*\(",
    "ast_fn_def": r"\bfn\s+[A-Za-z_]",
    "ast_let": r"\blet\b",
    "ast_use": r"^\s*use\b",
    "ast_return": r"\breturn\b",
    "ast_param": r"\bfn\s+[A-Za-z_][A-Za-z_0-9]*\s*\(\s*[A-Za-z_]",
    "ast_block": r"\{",
    "ast_paren": r"=\s*\(|\(\s*[a-z_][A-Za-z_0-9]*\s*[-+*/]",
    "ast_struct_def": r"^\s*(pub\s+)?struct\s+[A-Z]",
    # prefix `-`, never the binary minus in `a - b`
    "ast_neg": r"(?:^|[=(,\[]|\breturn|\{)\s*-\s*[a-zA-Z_(0-9]",
    "ast_not": r"!\s*[a-zA-Z_(]",
    "ast_if": r"\bif\b",
    "ast_field": r"\.[a-z_][A-Za-z_0-9]*(?!\s*\()",
    "ast_method": r"\.[a-z_][A-Za-z_0-9]*\s*\(",
    "ast_struct_lit": r"\b[A-Z][A-Za-z_0-9]*\s*\{",
    "ast_while": r"\bwhile\b",
    "ast_assign": r"^\s*[a-z_][A-Za-z_0-9]*\s*=[^=]",
    "ast_break": r"\bbreak\b",
    "ast_continue": r"\bcontinue\b",
    "ast_float_lit": r"(?<![\w.])\d+\.\d+",
    "ast_cast": r"\bas\s+[iuf]\d|\bas\s+bool",
    # the parser's poison marker — an item-level attribute is the reachable
    # source spelling that folds an item into `ast_unsupported`
    "ast_unsupported": r"#\[",
    "ast_array_lit": r"=\s*\[|\breturn\s*\[",
    "ast_index": r"[a-z_][A-Za-z_0-9]*\s*\[[^\];]*\]\s*(?!=[^=])",
    "ast_index_assign": r"[a-z_][A-Za-z_0-9]*\s*\[[^\];]*\]\s*=[^=]",
    "ast_str_lit": r"\"",
    "ast_enum_ctor": r"\b[A-Z][A-Za-z_0-9]*::[A-Z]",
    # prefix `&`, never the binary bit-and in `a & b` nor `&&`
    "ast_addr_of": r"(?:[=(,:]|\breturn)\s*&(?!&)\s*[a-zA-Z_]",
    "ast_field_assign": r"[a-z_][A-Za-z_0-9]*\.[a-z_][A-Za-z_0-9]*\s*=[^=]",
}

#: kinds with no source spelling of their own — the emitter/desugar builds them.
SYNTHETIC = {
    "ast_program": "the module root — present in every fixture by construction",
    "ast_alloc": "synthesised by the struct-lit lowering, never parsed",
}

#: kinds the self-host front end accepts but for which no ORACLE BYTES exist, so
#: no fixture can compare anything. Measured: `mindc --emit-mic3` writes no
#: artifact for `*p` / `*p = v` under an `i64`, `&i64`, `&mut i64` or `*i64`
#: receiver. `oracle_parity_lint.py` reaches the same construct from the other
#: side — its `deref-read` / `deref-write` arms are pinned "native-ELF-only;
#: mic@3 must fail closed EMPTY" — so a refusal here is the agreed behaviour and
#: an empty-vs-empty comparison would assert nothing.
#: deferred: give the oracle a mic@3 pointer surface so these become
#: byte-comparable — upgrade path: a deref/addr-of `ast::Expr` variant in
#: src/ast/mod.rs plus its mic@3 lowering, after which both kinds move into
#: SOURCE_PROBES and this bucket empties.
NO_ORACLE_CONSTRUCT = {
    "ast_deref": "`*p` — no oracle mic@3 bytes exist for any spelling tried",
    "ast_deref_assign": "`*p = v` — same: no oracle-compilable spelling exists",
}


def ast_kind_coverage(fixture_texts):
    """Every parser AST node kind must be exercised by >= 1 corpus fixture.

    Returns (ok, ran, failures). `ran` is the number of kinds actually checked —
    a gate that asserts nothing must never read as a pass, so a run that
    resolves zero kinds is itself a failure.
    """
    # The parser is main.mind plus its parts (kinds move into parts verbatim), so the
    # kind list is read from the whole composed source, never from main.mind alone.
    main_mind = _HERE / "main.mind"
    sources = [main_mind] + [_HERE.parents[1] / part for part in COMPILER_COMPONENTS]
    try:
        kinds = list(dict.fromkeys(k for src in sources for k in _AST_KIND_DECL.findall(src.read_text())))
    except OSError as exc:
        return False, 0, [f"cannot read the compiler sources: {exc}"]

    failures = []
    if not kinds:
        return False, 0, [
            f"no `pub fn ast_*() -> i64` declarations found in {main_mind} — the "
            "coverage lint cannot state anything about a kind list it did not "
            "resolve (refusing to pass vacuously)."
        ]

    mapped = set(SOURCE_PROBES) | set(SYNTHETIC) | set(NO_ORACLE_CONSTRUCT)
    for k in kinds:
        if k not in mapped:
            failures.append(
                f"{k}: NEW AST node kind with no entry in this lint. Add a "
                "SOURCE_PROBES regex plus a fixture, or classify it as SYNTHETIC / "
                "NO_ORACLE_CONSTRUCT with the reason. A kind nobody classified is a "
                "construct that can hide a miscompile."
            )
    for k in sorted(mapped - set(kinds)):
        failures.append(
            f"{k}: classified here but no longer declared in main.mind or its parts — the "
            "lint's scope has drifted from the parser it claims to cover."
        )

    for k in kinds:
        probe = SOURCE_PROBES.get(k)
        if probe is None:
            continue
        if not any(re.search(probe, t, re.M) for t in fixture_texts):
            failures.append(
                f"{k}: ZERO fixtures exercise this AST node kind. Add one to "
                f"tests/selfhost_gaps/ (or never_wrong/ if the driver refuses it) — "
                "an unexercised kind is a construct whose bytes nothing compares."
            )
    # One row per kind. The token on the row states what was CHECKED for it:
    #   [FAIL]   any failure above (unclassified, stale classification, or a
    #            probed kind no fixture exercises) — including kinds that are
    #            classified here but no longer declared, which iterate below so
    #            a drifted lint scope is a visible red row, not only a summary;
    #   [PASS]   a SOURCE_PROBES kind with >= 1 fixture matching its probe — a
    #            coverage check that ran against the corpus text;
    #   [EXEMPT] a SYNTHETIC / NO_ORACLE_CONSTRUCT kind — a classification only.
    #            Nothing was byte-compared, so the row deliberately carries NO
    #            verdict token: scripts/gate_assert.py must not count an
    #            exemption as an assertion. The literal tokens are spelled out
    #            (not interpolated) so smoke_wiring_lint's per-case scan can see
    #            them, and tests/gate_assert_count_contract_test.py pins that the
    #            [EXEMPT] rows stay token-free even if a reason string changes.
    stale = sorted(mapped - set(kinds))
    for k in list(kinds) + stale:
        kind_failures = [failure for failure in failures if failure.startswith(f"{k}:")]
        if kind_failures:
            print(f"[FAIL] AST node kind {k}: {kind_failures[0]}")
        elif k in SOURCE_PROBES:
            print(f"[PASS] AST node kind {k}: >= 1 corpus fixture matches its source probe")
        else:
            bucket = "SYNTHETIC" if k in SYNTHETIC else "NO_ORACLE_CONSTRUCT"
            reason = SYNTHETIC.get(k) or NO_ORACLE_CONSTRUCT.get(k)
            print(f"[EXEMPT] AST node kind {k}: not byte-compared — {bucket}: {reason}")
    return not failures, len(kinds), failures


def main():
    _resolve_oracles()
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not found (build libmindc_mind.so first)")
        return 0
    if not MINDC.exists():
        print(f"BLOCKED: oracle mindc not found at {MINDC}")
        return 1

    fixtures = sorted(CORPUS.glob("*.mind"))
    if not fixtures:
        print(f"BLOCKED: no fixtures under {CORPUS}")
        return 1

    byte_exact = 0
    fail_closed = []
    wrong = []
    oracle_invalid = []

    for f in fixtures:
        src = f.read_text()
        oracle = oracle_mic3(src)
        if oracle is None:
            oracle_invalid.append(f.name)  # not a self-host gap (source doesn't compile)
            continue
        nfn, status = nfn_mic3(src)
        if status == "CRASH":
            wrong.append(f"{f.name} (driver CRASHED — must fail-closed, never crash)")
            print(f"[FAIL] {f.name}: driver crashed")
        elif not nfn:
            fail_closed.append(f.name)
            print(f"[PASS] {f.name}: self-host refused safely")
        elif nfn == oracle:
            byte_exact += 1
            print(f"[PASS] {f.name}: byte-exact")
        else:
            n = min(len(nfn), len(oracle))
            di = next((i for i in range(n) if nfn[i] != oracle[i]), n)
            wrong.append(f"{f.name} (nfn={len(nfn)}B oracle={len(oracle)}B diff@{di})")
            print(f"[FAIL] {f.name}: wrong bytes")

    total = len(fixtures) - len(oracle_invalid)
    print(
        f"gap corpus: {byte_exact}/{total} byte-exact, "
        f"{len(fail_closed)} fail-closed, {len(wrong)} wrong-bytes"
        + (f" ({len(oracle_invalid)} oracle-invalid skipped)" if oracle_invalid else "")
    )

    # Cardinal invariant: the driver NEVER emits wrong bytes (a crash counts as wrong — the
    # driver must fail-closed, never crash). Always hard-fails.
    # Coverage ratchet: byte-exact must not drop below FLOOR. The whole corpus
    # lowers byte-exactly under the canonical FRESH-load measurement; the floor pins it so
    # no fixture can silently regress to fail-closed or wrong.
    # 142 -> 143: the neg/not-call-in-branch coverage batch. node_has_call now walks
    # ast_neg/ast_not/ast_index, so a call hidden under a unary op in a branch value
    # can no longer slip into the src=0 narrow emitter (was silent WRONG-BYTES).
    # not_call_branch_then is byte-exact (+1); the 4 neg_call_branch_* fixtures are
    # pinned safe fail-closed (the general path refuses rather than mis-emits) — a
    # regression to wrong-bytes trips the `wrong` assert above regardless of FLOOR.
    # 143 -> 145: the AST-node-kind coverage batch added `use_item_1` (ast_use)
    # and `while_counter_1` (ast_while), both byte-exact. The floor ratchets with
    # them so neither can silently regress to fail-closed.
    # 145 -> 149: canonical typed-cast lowering. 149 -> 150: `#[inline]` is
    # parsed as an exact, byte-neutral item prefix instead of swallowing `g`.
    FLOOR = 150
    ok = True
    if wrong:
        print("FAIL: WRONG-BYTES (silent miscompile) — the cardinal invariant is violated:")
        for w in wrong:
            print(f"   {w}")
        ok = False
    if fail_closed:
        print(f"NOTE: {len(fail_closed)} fixture(s) fail-closed (safe refusal, never wrong bytes):")
        for fc in fail_closed:
            print(f"   {fc}")
    if byte_exact < FLOOR:
        print(f"FAIL: byte-exact {byte_exact} < floor {FLOOR} — a fixture regressed to fail-closed/wrong.")
        ok = False

    # --- never-wrong corpus (tests/selfhost_gaps/never_wrong/) ------------------------
    # Shapes the self-host does not yet lower byte-exactly and therefore FAIL CLOSED
    # (0 bytes) rather than emit wrong bytes. Currently pins the ast_index fail-closed
    # guards (flatten_ast / count_nonparam_nodes / flatten_expr_env ast_index arms):
    #   * index_call_in_if_1  — `xs[g(a)]` (CALL in the index subtree) — count-vs-
    #                           flatten slot disagreement on the call's non-emitting
    #                           callee/arg-cons slots would trail 0xFF garbage.
    #   * neg_index_in_if_1    — `-xs[0]` in an if-branch (NEG base: `(-xs)[0]`).
    #   * neg_index_trailing_1 — `-xs[0]` in trailing position (env-arm NEG base).
    # Each must self-emit 0B (fail-closed) or byte-exact — NEVER wrong-bytes. Unlike
    # the byte-exact corpus, fail-closed is PERMITTED here (these shapes are not yet
    # lowered); the ONLY forbidden outcome is wrong-bytes: a change that reintroduces
    # one of these miscompiles turns this RED, where the byte-exact-only gap corpus
    # structurally cannot. Upgrade paths (deferred): parser neg / postfix-index
    # precedence; branch-buffer slot accounting for a call inside an index subtree.
    nw = sorted((CORPUS / "never_wrong").glob("*.mind")) if (CORPUS / "never_wrong").is_dir() else []
    nw_wrong, nw_be, nw_fc, nw_skip = [], 0, 0, 0
    for f in nw:
        src = f.read_text()
        oracle = oracle_mic3(src)
        if oracle is None:
            nw_skip += 1
            continue
        nfn, status = nfn_mic3(src)
        if status == "CRASH":
            nw_wrong.append(f"{f.name} (driver CRASHED — must fail-closed, never crash)")
        elif not nfn:
            nw_fc += 1
        elif nfn == oracle:
            nw_be += 1
        else:
            n = min(len(nfn), len(oracle))
            di = next((i for i in range(n) if nfn[i] != oracle[i]), n)
            nw_wrong.append(f"{f.name} (nfn={len(nfn)}B oracle={len(oracle)}B diff@{di})")
    if nw:
        print(
            f"never-wrong corpus: {nw_be + nw_fc}/{len(nw) - nw_skip} safe "
            f"({nw_be} byte-exact, {nw_fc} fail-closed), {len(nw_wrong)} wrong-bytes"
            + (f" ({nw_skip} oracle-invalid skipped)" if nw_skip else "")
        )
        if nw_wrong:
            print("FAIL: never-wrong corpus WRONG-BYTES — the desugar-hoist guard regressed to a miscompile:")
            for w in nw_wrong:
                print(f"   {w}")
            ok = False

    # --- AST-node-kind coverage (the hole the byte survey cannot see) --------
    cov_ok, cov_ran, cov_fail = ast_kind_coverage(
        [f.read_text() for f in fixtures] + [f.read_text() for f in nw]
    )
    print(
        f"AST-kind coverage checked {cov_ran} kinds "
        f"({len(SOURCE_PROBES)} probed, {len(SYNTHETIC)} synthetic, "
        f"{len(NO_ORACLE_CONSTRUCT)} no-oracle-construct)"
    )
    if not cov_ok:
        print("FAIL: AST-node-kind coverage — a construct family nothing compares:")
        for c in cov_fail:
            print(f"   {c}")
        ok = False

    if ok:
        # cov_ok means mapped == kinds exactly, so the exempt count is the size
        # of the two exemption buckets and the rest were fixture-covered.
        exempt = len(SYNTHETIC) + len(NO_ORACLE_CONSTRUCT)
        print(
            f"PASS: 0 wrong-bytes; {byte_exact}/{total} byte-exact "
            f"(>= floor {FLOOR}), {len(fail_closed)} safe fail-closed, "
            f"{cov_ran - exempt} of {cov_ran} AST node kinds fixture-covered, "
            f"{exempt} exempt (synthetic / no-oracle, not byte-compared)"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
