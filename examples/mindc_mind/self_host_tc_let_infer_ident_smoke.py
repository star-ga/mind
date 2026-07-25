#!/usr/bin/env python3
"""self_host_tc_let_infer_ident_smoke — three-leg gate for the T2 port.

Rule: `let NAME: ANN = <IDENT>` implicit integer NARROWING (E2004,
NARROWING_CODE, src/type_checker/mod.rs:139 + is_implicit_narrowing:267, fired
at mod.rs:3995). The RHS type is `infer_expr`'s Ident arm (mod.rs:831) — the
variable's declared type from the typed env. E2004 fires iff BOTH the
destination annotation and the source variable are int scalars AND the
destination is STRICTLY narrower. The Rust oracle knows exactly two int widths
(ScalarI32=32, ScalarI64=64; `u32` -> ScalarI32, mod.rs:2659), so the only
firing shape is an i32/u32 destination with an i64 source.

# oracle-legs: port=selftest_tc_let_infer_ident leg2=rust-width-table live=mindc-check position_sentinel=-3

Three legs, agreement REQUIRED on every scored case:
  leg-1 PORT  — selftest_tc_let_infer_ident via ctypes (1 fire / 0 clean / -3 decline)
  leg-2 RUST  — the declared-WIDTH table (valuetype_from_ann -> int_scalar_bits
                -> `to_bits < from_bits`). Keyed ONLY by the two annotation
                NAMES written in the fixture; it never tokenises, never looks at
                positions, never calls the .so, and emits only {0,1} (the
                position sentinel -3 is port-only) — OIA (Rule 3a).
  leg-3 LIVE  — `mindc check`: is E2004 reported at the RHS ident's line:col?

Plus a NO-OVER-FIRE block over the shapes the port deliberately declines
(params, shadowing, out-of-scope bindings, opaque annotations, compound RHS):
the port must never emit a false E2004; an under-fire vs live is the safe
direction and is printed for the record.

Env: MINDC_SO (port .so, required) · MINDC_BIN (mindc, default `mindc`).
Exit 0 = ALL PASS; nonzero = a divergence (SEV-1).
"""
import ctypes
import os
import re
import subprocess
import sys
import tempfile

MINDC = os.environ.get("MINDC_BIN", "mindc")
SO = os.environ.get("MINDC_SO")
E2004_RE = re.compile(r":(\d+):(\d+): error: .*\[type_check::E2004\]")

# ── leg-2: the independent Rust WIDTH table (NOT derived from the port) ─────
# valuetype_from_ann (mod.rs:2640) -> int_scalar_bits (mod.rs:249). Only the
# annotation names that map to a concrete int ValueType carry a width; every
# other annotation is either a non-int scalar or TypeAnn::Named (None).
_INT_BITS = {"i32": 32, "u32": 32, "i64": 64}


def leg2_rust(dst_ann, src_ann):
    """E2004 fires iff both anns are int scalars and dst is strictly narrower."""
    to_bits = _INT_BITS.get(dst_ann)
    from_bits = _INT_BITS.get(src_ann)
    if to_bits is None or from_bits is None:
        return 0
    return 1 if to_bits < from_bits else 0


# ── the scored matrix: declared source ann × declared destination ann ───────
SEED = {"i32": "5", "u32": "5", "i64": "5", "f32": "1.5", "f64": "1.5",
        "bool": "true"}
ANNS = ["i32", "i64", "u32", "f32", "f64", "bool"]


def fixture(src_ann, dst_ann, mut=False):
    head = "let mut a" if mut else "let a"
    return (f"fn m() -> i64 {{\n    {head}: {src_ann} = {SEED[src_ann]}\n"
            f"    let b: {dst_ann} = a\n    return 0\n}}\n")


def build_cases():
    cases = []
    for src_ann in ANNS:
        for dst_ann in ANNS:
            src = fixture(src_ann, dst_ann)
            cases.append((f"{src_ann}->{dst_ann}", src_ann, dst_ann, src,
                          src.rindex("= a") + 2))
    # `let mut` source-head variant across the same destination axis.
    for dst_ann in ANNS:
        src = fixture("i64", dst_ann, mut=True)
        cases.append((f"mut i64->{dst_ann}", "i64", dst_ann, src,
                      src.rindex("= a") + 2))
    # ESCAPED-QUOTE-before-let (task #244): a string / char literal with an
    # escaped quote (`"a\"b"` / `'\''`) in a statement BEFORE the narrowing let.
    # The self-host lexer tokenizes byte-by-byte; tc_dn_skip_str now honors
    # `\`-escapes so the escaped quote no longer terminates the literal and
    # desyncs the following let. Narrowing i64->i32 so all three legs FIRE.
    # Before the fix the port under-fired here (declined, praw=-3) — the RED.
    esc_dq = ('fn m() -> i64 {\n    let s = "a\\"b"\n'
              "    let a: i64 = 5\n    let b: i32 = a\n    return 0\n}\n")
    cases.append(("esc-dquote i64->i32", "i64", "i32", esc_dq,
                  esc_dq.rindex("= a") + 2))
    esc_sq = ("fn m() -> i64 {\n    let c = '\\''\n"
              "    let a: i64 = 5\n    let b: i32 = a\n    return 0\n}\n")
    cases.append(("esc-squote i64->i32", "i64", "i32", esc_sq,
                  esc_sq.rindex("= a") + 2))
    return cases


# ── shapes the port DECLINES (fail-closed) — invariant: never over-fire ─────
# (label, source, needle-for-position). `live_note` is informational only.
def decline_cases():
    out = []
    out.append(("i8 src", "fn m() -> i64 {\n    let a: i8 = 5\n"
                "    let b: i32 = a\n    return 0\n}\n"))
    out.append(("usize src", "fn m() -> i64 {\n    let a: usize = 5\n"
                "    let b: i32 = a\n    return 0\n}\n"))
    out.append(("Widget dst", "fn m() -> i64 {\n    let a: i64 = 5\n"
                "    let b: Widget = a\n    return 0\n}\n"))
    out.append(("param src", "fn m(a: i64) -> i64 {\n"
                "    let b: i32 = a\n    return 0\n}\n"))
    out.append(("shadowed src", "fn m() -> i64 {\n    let a: i64 = 5\n"
                "    let a: i32 = 1\n    let b: i32 = a\n    return 0\n}\n"))
    out.append(("dead-scope src", "fn m() -> i64 {\n    if true {\n"
                "        let a: i64 = 5\n    }\n    let b: i32 = a\n"
                "    return 0\n}\n"))
    out.append(("compound rhs", "fn m() -> i64 {\n    let a: i64 = 5\n"
                "    let b: i32 = a + 1\n    return 0\n}\n"))
    out.append(("unannotated src", "fn m() -> i64 {\n    let a = 5\n"
                "    let b: i32 = a\n    return 0\n}\n"))
    out.append(("no annotation dst", "fn m() -> i64 {\n    let a: i64 = 5\n"
                "    let b = a\n    return 0\n}\n"))
    out.append(("use before decl", "fn m() -> i64 {\n    let b: i32 = a\n"
                "    let a: i64 = 5\n    return 0\n}\n"))
    # NESTED-BLOCK narrowing — live's is_implicit_narrowing (mod.rs:3995) runs
    # ONLY over the fn body's TOP-LEVEL Node::Let statements and never recurses
    # into a branch body, so a narrowing `let` nested in any if/while/match/
    # block emits NO E2004. The port MUST decline (top-level-only guard); over-
    # firing here is the fail-OPEN class the coordinator's adversarial probe
    # caught. Both-in-if / src-outer-rhs-in-if / 2-deep / while-body / match-arm.
    out.append(("both-in-if", "fn m() -> i64 {\n    if 1 == 1 {\n"
                "        let a: i64 = 5\n        let b: i32 = a\n    }\n"
                "    0\n}\n"))
    out.append(("src-outer/rhs-in-if", "fn m() -> i64 {\n    let a: i64 = 5\n"
                "    if 1 == 1 {\n        let b: i32 = a\n    }\n    0\n}\n"))
    out.append(("if-2-deep", "fn m() -> i64 {\n    if 1 == 1 {\n"
                "        if 1 == 1 {\n            let a: i64 = 5\n"
                "            let b: i32 = a\n        }\n    }\n    0\n}\n"))
    out.append(("while-body", "fn m() -> i64 {\n    while 1 == 1 {\n"
                "        let a: i64 = 5\n        let b: i32 = a\n    }\n"
                "    0\n}\n"))
    out.append(("if-else-narrow", "fn m() -> i64 {\n    if 1 == 1 {\n"
                "        let a: i64 = 5\n        let b: i32 = a\n"
                "    } else {\n        let c: i64 = 5\n        let d: i32 = c\n"
                "    }\n    0\n}\n"))
    out.append(("match-arm-narrow", "fn m() -> i64 {\n    match 1 {\n"
                "        _ => {\n            let a: i64 = 5\n"
                "            let b: i32 = a\n        }\n    }\n    0\n}\n"))
    # for the last two the RHS ident is `c`/`d`; use a needle that resolves the
    # correct RHS position per-case below.
    cases = []
    for label, s in out:
        needle = "= c" if label == "if-else-narrow" else "= a"
        cases.append((label, s, s.rindex(needle) + 2))
    return cases


def line_col(src, pos):
    return src.count("\n", 0, pos) + 1, pos - (src.rfind("\n", 0, pos) + 1) + 1


def leg3_live(mindc, workdir, src, pos):
    path = os.path.join(workdir, "c.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    want = line_col(src, pos)
    for m in E2004_RE.finditer(r.stdout + r.stderr):
        if (int(m.group(1)), int(m.group(2))) == want:
            return 1
    return 0


def main():
    if not SO:
        print("INFRA FAIL: MINDC_SO not set")
        sys.exit(2)
    st = os.stat(SO)
    if st.st_size < 4096:
        print(f"INFRA FAIL: .so too small ({st.st_size} bytes — stub?)")
        sys.exit(2)
    lib = ctypes.CDLL(SO)
    port = lib.selftest_tc_let_infer_ident
    port.argtypes = [ctypes.c_int64] * 5
    port.restype = ctypes.c_int64
    std = ctypes.create_string_buffer(b"", 0)
    sp = ctypes.cast(std, ctypes.c_void_p).value

    def call_port(src, pos):
        data = src.encode()
        b = ctypes.create_string_buffer(data, len(data))
        return port(ctypes.cast(b, ctypes.c_void_p).value, len(data), pos, sp, 0)

    cases = build_cases()
    declines = decline_cases()
    fails = 0
    with tempfile.TemporaryDirectory() as workdir:
        for label, src_ann, dst_ann, src, pos in cases:
            praw = call_port(src, pos)
            if praw not in (0, 1, -3):
                print(f"FAIL  {label:16s} port returned illegal {praw}")
                fails += 1
                continue
            pfire = 1 if praw == 1 else 0
            l2 = leg2_rust(dst_ann, src_ann)
            l3 = leg3_live(MINDC, workdir, src, pos)
            if pfire == l2 == l3 and praw != -3:
                verd = "fire " if l3 else "clean"
                print(f"PASS  {label:16s} {verd} (port_raw={praw} leg2={l2} live={l3})")
            else:
                why = "DECLINED in-domain" if praw == -3 else "three-leg divergence"
                print(f"FAIL  {label:16s} {why}: port={pfire}(raw={praw}) "
                      f"leg2={l2} live={l3}")
                fails += 1

        for label, src, pos in declines:
            praw = call_port(src, pos)
            pfire = 1 if praw == 1 else 0
            l3 = leg3_live(MINDC, workdir, src, pos)
            if pfire == 0:
                note = "under-fire(safe)" if l3 == 1 else "agree-clean"
                print(f"PASS  {label:16s} no-over-fire (port_raw={praw} "
                      f"live={l3}) {note}")
            else:
                print(f"FAIL  {label:16s} OVER-FIRE — false E2004 "
                      f"(port_raw={praw} live={l3})")
                fails += 1

    n = len(cases) + len(declines)
    if fails:
        print(f"\n{fails}/{n} FAILED — three-leg divergence / over-fire (SEV-1)")
        sys.exit(1)
    print(f"\nALL PASS ({n}/{n}) — port == rust-width-table == live mindc on "
          f"{len(cases)} scored ann-pairs; no over-fire on {len(declines)} "
          f"declined shapes")


if __name__ == "__main__":
    main()
