#!/usr/bin/env python3
"""self_host_tc_let_infer_smoke — three-leg gate for the T1 type-inference port.

Rule: `let NAME: ANN = <literal>` scalar-class mismatch. The pure-MIND
`infer_expr_tag` leaf map (examples/mindc_mind/main.mind) tags the RHS literal;
the annotation's scalar class and the RHS tag's class are compared. The ONLY
observable diagnostic for a LITERAL RHS is E2015 (LET_CLASS_MISMATCH_CODE, RFC
0011 no-implicit-int<->float) — E2004 narrowing is unreachable from a literal
(int literals are the narrowest int width; narrowing needs an i64 SOURCE = T2).

# oracle-legs: port=selftest_tc_let_infer_lit leg2=rust-truth-table live=mindc-check position_sentinel=-3

Three legs, agreement REQUIRED on every case:
  leg-1 PORT  — selftest_tc_let_infer_lit via ctypes (1 fire / 0 clean / -3 decline)
  leg-2 RUST  — the mod.rs truth table (scalar_class_of_ann + confident_scalar_class),
                hardcoded by ANN NAME + RHS KIND. Derived INDEPENDENTLY of the
                port's byte-comparison tokeniser; NEVER calls the .so; emits only
                {0,1} (the position sentinel -3 is port-only) — OIA (Rule 3a).
  leg-3 LIVE  — `mindc check`: is E2015 reported at the RHS literal's line:col?

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
E2015_RE = re.compile(r":(\d+):(\d+): error: .*\[type_check::E2015\]")

# ── leg-2: the independent Rust truth table (NOT derived from the port) ──────
# scalar_class_of_ann (mod.rs:2938): the Int + Float scalar annotation names.
_INT_ANNS = {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
             "isize", "usize", "bool"}
_FLOAT_ANNS = {"f32", "f64"}
# confident_scalar_class (mod.rs:2976) for leaf literals: Int lit (incl.
# true/false which fold to Int(1/0)); Float lit -> Float; Str -> None.
_RHS_CLASS = {"int": "Int", "float": "Float", "bool": "Int", "str": "None"}


def ann_class(name):
    if name in _INT_ANNS:
        return "Int"
    if name in _FLOAT_ANNS:
        return "Float"
    return "None"


def leg2_rust(ann_name, rhs_kind):
    """E2015 fires iff both classes are scalar (Int/Float) and DIFFER."""
    ac = ann_class(ann_name)
    rc = _RHS_CLASS[rhs_kind]
    if ac in ("Int", "Float") and rc in ("Int", "Float") and ac != rc:
        return 1
    return 0


# ── the case matrix (ann name, rhs kind, rhs text) ──────────────────────────
ANNS = ["i32", "i64", "u32", "bool", "i8", "i16", "u8", "u16", "u64",
        "isize", "usize", "f32", "f64", "String", "Vec", "Widget"]
RHS = [("int", "42"), ("float", "3.5"), ("str", '"hi"'),
       ("bool", "true"), ("bool", "false"),
       # `1.5e-3` lexes as tk_float (dotted mantissa) → correctly Float on both
       # sides; the exponent-WITHOUT-dot forms (1e5) are the deferred-shape block.
       ("float", "1.5e-3")]

# Shapes infer_expr_tag CANNOT yet confidently classify — the invariant is that
# it must NEVER over-fire (emit a false E2015 on valid code). The self-host
# number lexer mis-splits exponent (`1e5`) and underscore (`1_000.0`) forms:
# this was the HIGH false-positive the blind review caught — `let x: f64 = 1e5`
# used to fire E2015 on valid code — now fail-closed to a decline. Neg / Paren
# RHS are slice-T2 scope. PASS iff the port does NOT fire (pfire == 0); the live
# verdict is printed for the record (an under-fire, where live fires and the
# port declines, is the SAFE direction — a wrong "not yet", never a wrong verdict).
# Real fix for the numeric forms = exponent + `_` in the self-host number lexer
# (a byte-identity-gated mind-self-host task).
DEFERRED_NO_OVERFIRE = [
    ("f64", "1e5"), ("f32", "1e5"), ("f64", "2E3"), ("i32", "1e5"),
    ("i32", "2E3"), ("f64", "1_000.0"), ("i32", "1_000.0"), ("i32", "1_000"),
    ("f64", "1_000"), ("i32", "-3.5"), ("f64", "-5"), ("i32", "(3.5)"),
]


def build_cases():
    cases = []
    for ann in ANNS:
        for kind, text in RHS:
            src = f"fn main() -> i64 {{\n    let z: {ann} = {text}\n    return 0\n}}\n"
            cases.append((f"{ann}={text}", ann, kind, src, src.index(text)))
    # `let mut` head variant (float RHS: fires for int anns, clean for float).
    for ann in ("i32", "f64", "usize"):
        src = f"fn main() -> i64 {{\n    let mut z: {ann} = 3.5\n    return 0\n}}\n"
        cases.append((f"mut {ann}=3.5", ann, "float", src, src.index("3.5")))
    return cases


def line_col(src, pos):
    return src.count("\n", 0, pos) + 1, pos - (src.rfind("\n", 0, pos) + 1) + 1


def leg3_live(mindc, workdir, src, pos):
    path = os.path.join(workdir, "c.mind")
    with open(path, "w") as f:
        f.write(src)
    r = subprocess.run([mindc, "check", path], capture_output=True, text=True)
    want = line_col(src, pos)
    for m in E2015_RE.finditer(r.stdout + r.stderr):
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
    port = lib.selftest_tc_let_infer_lit
    port.argtypes = [ctypes.c_int64] * 5
    port.restype = ctypes.c_int64
    std = ctypes.create_string_buffer(b"", 0)
    sp = ctypes.cast(std, ctypes.c_void_p).value

    def call_port(src, pos):
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        return port(ctypes.cast(b, ctypes.c_void_p).value, len(src.encode()), pos, sp, 0)

    cases = build_cases()
    fails = 0
    with tempfile.TemporaryDirectory() as workdir:
        for label, ann, kind, src, pos in cases:
            praw = call_port(src, pos)
            if praw not in (0, 1, -3):
                print(f"FAIL  {label:16s} port returned illegal {praw}")
                fails += 1
                continue
            pfire = 1 if praw == 1 else 0
            l2 = leg2_rust(ann, kind)
            l3 = leg3_live(MINDC, workdir, src, pos)
            if pfire == l2 == l3:
                verd = "fire " if l3 else "clean"
                print(f"PASS  {label:16s} {verd} (port_raw={praw} leg2={l2} live={l3})")
            else:
                print(f"FAIL  {label:16s} port={pfire}(raw={praw}) leg2={l2} live={l3}")
                fails += 1

        # ── no-over-fire invariant on deferred shapes ───────────────────────
        # The port must NEVER emit a false positive (pfire==1 where the code is
        # valid). It is allowed to DECLINE (pfire==0) an uncovered shape; an
        # under-fire vs a live E2015 is the safe direction and is documented.
        for ann, text in DEFERRED_NO_OVERFIRE:
            src = f"fn main() -> i64 {{\n    let z: {ann} = {text}\n    return 0\n}}\n"
            pos = src.index(text)
            praw = call_port(src, pos)
            pfire = 1 if praw == 1 else 0
            l3 = leg3_live(MINDC, workdir, src, pos)
            label = f"{ann}={text}"
            if pfire == 0:
                note = "under-fire(safe)" if l3 == 1 else "agree-clean"
                print(f"PASS  {label:16s} no-over-fire (port_raw={praw} live={l3}) {note}")
            else:
                print(f"FAIL  {label:16s} OVER-FIRE — false E2015 on valid code "
                      f"(port_raw={praw} live={l3})")
                fails += 1

    n = len(cases) + len(DEFERRED_NO_OVERFIRE)
    if fails:
        print(f"\n{fails}/{n} FAILED — three-leg divergence / over-fire (SEV-1)")
        sys.exit(1)
    print(f"\nALL PASS ({n}/{n}) — port == rust-truth-table == live mindc; "
          f"no over-fire on {len(DEFERRED_NO_OVERFIRE)} deferred shapes")


if __name__ == "__main__":
    main()
