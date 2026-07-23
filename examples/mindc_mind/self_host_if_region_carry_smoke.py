#!/usr/bin/env python3
"""Native-ELF smoke: i64 loop-carry through BRANCHED regions (Sub-step C).

A loop-carried i64 var mutated inside a branched region inside a `while`/`for`
now lowers CORRECTLY through the pure-MIND nb native path. Two fixes compose:

  1. nb_if_carry_promote promotes a merged OUTER var whenever it is ASSIGNED
     anywhere in either branch (any depth, nb_block_writes_rec) with
     post_id = its if-merge phi slot (merge_base + j) — the region-exit
     incarnation (lower.rs last_region_exit_rebindings). The earlier XOR
     single-branch filter under-promoted both-branch-same-var, nested-if and
     if-wrapped-while writes, silently dropping the carry across the loop
     back-edge (fail-OPEN: while{ if{x+=1}else{x+=2} } returned 7 want 11,
     while{ if{ if{x+=1} } } returned 1 want 3, while{ if{ while{x+=1} } }
     returned 2 want 6).
  2. nb_while_live_writes descends into nested if branches and nested while
     bodies (was top-level-assign-only), so a var assigned only inside a
     region nested in a branch-while is merged by the ENCLOSING if — the
     straight-line if{ while{ if{x+=1} } } returned 0 want 3 before.

Count and emit stay in lockstep by construction: both changes live in the
SHARED Mode traversal (Sub-step A) — nb_merged_names/nb_branch_writes feed
both nb_merge_ids_count and the emit merge table, and nb_if_carry_promote is
reached from the single nb_while_carry pre-walk that the count delegates to.

Every fixture is emitted via selftest_native_elf_h, RUN, and its exit code
compared to an INDEPENDENT Python-computed reference (no hand-authored
tables). A 0-byte emit or wrong exit is a FAIL. Guarded on the full case
count so it cannot pass vacuously.

Env: MINDC_SO (prebuilt .so) or MINDC_BIN (default mindc).
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")


def _loop_ref(x0, iters, then_pred, then_d, else_d):
    """Reference: while a<iters { if then_pred(a) {x+=then_d} else {x+=else_d} ; a+=1 }"""
    x = x0
    for a in range(iters):
        x += then_d if then_pred(a) else else_d
    return x


# (label, python_ref_value, src) — LOOP-CARRY-THROUGH-BRANCH shapes (i64).
CASES = [
    ("both-branch-same-var while{ if{x+=1}else{x+=2} } x=5 4it",
     _loop_ref(5, 4, lambda a: a < 2, 1, 2),
     "fn main() -> i64 {\n    let mut x: i64 = 5;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { x = x + 2; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("both-branch bare-= while{ if{x=7}else{x=9} } 3it", 9,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 1 { x = 7; } else { x = 9; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("single-branch control while{ if{x+=1} } f(10) 3it", 13,
     "fn f(x: i64) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { x = x + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("selective bare-= while{ if(a%2==0){x=a} }", 2,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a % 2 == 0 { x = a; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("if-in-if while{ if{ if{x+=1} } } 3it", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { if a < 20 { x = x + 1; } }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("if-in-if inner-else while{ if{ if{x+=1}else{x+=3} } } 4it",
     _loop_ref(0, 4, lambda a: a < 2, 1, 3),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 10 { if a < 2 { x = x + 1; } else { x = x + 3; } }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("if-wrapped inner while (Sub-step C) while{ if{ while{x+=1} } } 3x2", 6,
     "fn f(x: i64) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut d: i64 = 0;\n        if a < 10 {\n            while d < 2 { x = x + 1; d = d + 1; }\n        }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(0); }\n"),
    ("selective if-wrapped inner while while{ if(a%2==0){ while{x+=1} } } 2x3", 6,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        let mut d: i64 = 0;\n        if a % 2 == 0 {\n            while d < 3 { x = x + 1; d = d + 1; }\n        }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("outer if wraps loop if{ while{ if{x+=1} } } 3it", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    if 1 < 2 {\n        while a < 3 {\n            if a < 10 { x = x + 1; }\n            a = a + 1;\n        }\n    }\n    return x;\n}\n"),
    ("two-diff-vars while{ if{x+=1}else{y+=1} } -> x*10+y", 22,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut y: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { y = y + 1; }\n        a = a + 1;\n    }\n    return x * 10 + y;\n}\n"),
    ("mixed both+single while{ if{x+=1;y+=5}else{x+=2} } -> x*100+y",
     (lambda: (lambda x, y: x * 100 + y)(
         _loop_ref(0, 4, lambda a: a < 2, 1, 2),
         _loop_ref(0, 4, lambda a: a < 2, 5, 0)))(),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut y: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; y = y + 5; } else { x = x + 2; }\n        a = a + 1;\n    }\n    return x * 100 + y;\n}\n"),
    ("for single-branch for{ if{x+=1} } 3it", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    for a in 0..3 {\n        if a < 10 { x = x + 1; }\n    }\n    return x;\n}\n"),
    ("for both-branch for{ if{x+=1}else{x+=2} } x=5 4it",
     _loop_ref(5, 4, lambda a: a < 2, 1, 2),
     "fn main() -> i64 {\n    let mut x: i64 = 5;\n    for a in 0..4 {\n        if a < 2 { x = x + 1; } else { x = x + 2; }\n    }\n    return x;\n}\n"),
    ("else-only write while{ if{z+=1}else{x+=3} }",
     _loop_ref(0, 4, lambda a: a < 2, 0, 3),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    let mut z: i64 = 0;\n    while a < 4 {\n        if a < 2 { z = z + 1; } else { x = x + 3; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("sibling-nested write while{ if{x+=1}else{ if{x+=5} } }",
     _loop_ref(0, 4, lambda a: a < 2, 1, 5),
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 4 {\n        if a < 2 { x = x + 1; } else { if a < 10 { x = x + 5; } }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("depth-3 nest while{ if{if{if{x+=2}}} } 3it", 6,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a < 10 { if a < 20 { if a < 30 { x = x + 2; } } }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("direct while-in-while control 3x4", 12,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        let mut d: i64 = 0;\n        while d < 4 { x = x + 1; d = d + 1; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("never-taken branch control while{ if(a>100){x+=1} } f(10)", 10,
     "fn f(x: i64) -> i64 {\n    let mut a: i64 = 0;\n    while a < 3 {\n        if a > 100 { x = x + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("param both-branch while{ if(a%2==0){x+=2}else{x+=1} } f(3) 5it",
     _loop_ref(3, 5, lambda a: a % 2 == 0, 2, 1),
     "fn f(x: i64) -> i64 {\n    let mut a: i64 = 0;\n    while a < 5 {\n        if a % 2 == 0 { x = x + 2; } else { x = x + 1; }\n        a = a + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(3); }\n"),
    ("top-level + branch mix while{ x+=1; if{x+=2} } 3it", 9,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 3 {\n        x = x + 1;\n        if a < 10 { x = x + 2; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
]

# FAIL-CLOSED refusals (MUST emit 0B — a wrong-value ELF here is a FAIL):
#   * branch-exit carry bypass: a `break`/`continue` inside an if-branch that ALSO
#     assigns a variable bypasses the branch's merge-phi stores when taken, silently
#     DROPPING the writes (was fail-OPEN: exit 2 want 102 / 0 want 100 / 4 want 54).
#     Guarded by nb_fns_branch_exit_carry. `return`-in-branch and exit-ONLY branches
#     are permitted — both proven value-correct (see CASES / CONTROLS).
#   * field-store statement `p.x = 5;`: bare `=` is an infix comparison in the Pratt
#     table, so the store used to parse as a DEAD `(p.x == 5)` comparison and drop
#     (was fail-OPEN even at top level: 0 want 5). parse_field_assign_or_expr now
#     poisons it ast_unsupported -> every emitter refuses 0B.
REFUSE = [
    ("break after write, same branch: while{ if a==2 {x+=100; break;} else {x+=1} }",
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 10 {\n        if a == 2 { x = x + 100; break; } else { x = x + 1; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("bare-= + break: while{ if a==2 {x=100; break;} }",
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 10 {\n        if a == 2 { x = 100; break; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("continue after write: while{ a+=1; if a==3 {x+=50; continue;} x+=1 }",
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 5 {\n        a = a + 1;\n        if a == 3 { x = x + 50; continue; }\n        x = x + 1;\n    }\n    return x;\n}\n"),
    ("break at nested-if depth in a writing branch: while{ if{ x+=1; if{ break; } } }",
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 10 {\n        if a < 10 { x = x + 1; if x > 2 { break; } }\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("field store in straight-line if: if 1<2 { p.x = 5; }",
     "struct P {\n    x: i64,\n    y: i64,\n}\nfn main() -> i64 {\n    let p: P = P { x: 0, y: 9 };\n    if 1 < 2 { p.x = 5; }\n    return p.x;\n}\n"),
    ("field store top-level (no region): p.x = 5;",
     "struct P {\n    x: i64,\n    y: i64,\n}\nfn main() -> i64 {\n    let p: P = P { x: 0, y: 9 };\n    p.x = 5;\n    return p.x;\n}\n"),
    ("field store in while: while{ p.x = p.x + 1 }",
     "struct P {\n    x: i64,\n    y: i64,\n}\nfn main() -> i64 {\n    let p: P = P { x: 0, y: 9 };\n    let mut a: i64 = 0;\n    while a < 3 {\n        p.x = p.x + 1;\n        a = a + 1;\n    }\n    return p.x;\n}\n"),
]

# Exit-in-branch shapes that MUST STAY value-correct (the guard is per-branch —
# an exit-ONLY branch or a write+RETURN branch is sound and must not be refused;
# `return` computes from the live env at the return site, and main.mind's
# early-return style depends on it).
EXIT_CONTROLS = [
    ("write+RETURN in branch (sound, permitted): while{ if a==2 {x+=100; return x;} x+=1 }", 102,
     "fn f() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 10 {\n        if a == 2 { x = x + 100; return x; }\n        x = x + 1;\n        a = a + 1;\n    }\n    return 0 - 1;\n}\nfn main() -> i64 { return f(); }\n"),
    ("exit-ONLY branch, sibling writes (sound, permitted): while{ if a==2 {break;} x+=1 }", 2,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 10 {\n        if a == 2 { break; }\n        x = x + 1;\n        a = a + 1;\n    }\n    return x;\n}\n"),
    ("continue-ONLY branch (sound, permitted): while{ a+=1; if a%2==1 {continue;} x+=1 }", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 6 {\n        a = a + 1;\n        if a % 2 == 1 { continue; }\n        x = x + 1;\n    }\n    return x;\n}\n"),
    ("break in exit-only branch, else writes (sound, permitted)", 4,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    while a < 100 {\n        if x >= 4 { break; } else { x = x + 2; }\n        a = a + 1;\n    }\n    return x;\n}\n"),
]

# Straight-line (NO enclosing loop) region-merge shapes — the nb_while_live_writes
# recursion fix (a while inside an if branch whose writes are region-nested).
STRAIGHT = [
    ("SL if{ if{x+=1} }", 1,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    if 1 < 2 { if 2 < 3 { x = x + 1; } }\n    return x;\n}\n"),
    ("SL if{ if{x+=1}else{x+=3} }", 1,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    if 1 < 2 { if 2 < 3 { x = x + 1; } else { x = x + 3; } }\n    return x;\n}\n"),
    ("SL if{ while{x+=1} }", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut d: i64 = 0;\n    if 1 < 2 { while d < 3 { x = x + 1; d = d + 1; } }\n    return x;\n}\n"),
    ("SL if{ while{ if{x+=1} } } (was 0 want 3)", 3,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    if 1 < 2 { while a < 3 { if a < 10 { x = x + 1; } a = a + 1; } }\n    return x;\n}\n"),
    ("SL if{ if{x+=1} } outer not taken", 0,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    if 1 > 2 { if 2 < 3 { x = x + 1; } }\n    return x;\n}\n"),
    ("SL if{ while{ while{x+=1} } } (was 0 want 6)", 6,
     "fn main() -> i64 {\n    let mut x: i64 = 0;\n    let mut a: i64 = 0;\n    if 1 < 2 { while a < 2 { let mut d: i64 = 0; while d < 3 { x = x + 1; d = d + 1; } a = a + 1; } }\n    return x;\n}\n"),
]


def build_so():
    so = os.environ.get("MINDC_SO")
    if so:
        return so
    mindc = os.environ.get("MINDC_BIN", "mindc")
    out = tempfile.NamedTemporaryFile(suffix=".so", delete=False).name
    r = subprocess.run([mindc, MAIN_MIND, "--emit-shared", out], capture_output=True, text=True)
    if r.returncode != 0:
        print("BUILD FAILED rc=", r.returncode)
        print(r.stderr[-3000:])
        sys.exit(1)
    return out


def main() -> int:
    so = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        return 1
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_h"):
        print("FAIL: selftest_native_elf_h absent")
        return 1
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]

    def emit(src: str) -> bytes:
        b = ctypes.create_string_buffer(src.encode(), len(src.encode()))
        h = ctypes.create_string_buffer(b"\x00" * 32, 32)
        es = lib.selftest_native_elf_h(
            ctypes.cast(b, ctypes.c_void_p).value, len(src.encode()),
            ctypes.cast(h, ctypes.c_void_p).value,
        )
        sh = rd(es, 0)
        ln = rd(sh, 8)
        return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""

    all_ok = True
    loop_ok = 0
    sl_ok = 0
    exitc_ok = 0
    refused = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        def run_elf(elf: bytes) -> int:
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            return subprocess.run([str(p)], timeout=10).returncode

        for group, cases in (("loop-carry", CASES), ("straight-line", STRAIGHT),
                             ("exit-control", EXIT_CONTROLS)):
            for label, exp, src in cases:
                elf = emit(src)
                if not elf:
                    print(f"  FAIL  {group} REFUSED (want emit+run): {label} (emit 0B)")
                    all_ok = False
                    continue
                want = exp & 0xFF
                rc = run_elf(elf)
                ok = rc == want
                all_ok = all_ok and ok
                if ok:
                    if group == "loop-carry":
                        loop_ok += 1
                    elif group == "straight-line":
                        sl_ok += 1
                    else:
                        exitc_ok += 1
                print(f"  {'PASS' if ok else 'FAIL'}  {group}: {label} "
                      f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, src in REFUSE:
            elf = emit(src)
            if elf:
                rc = run_elf(elf)
                print(f"  FAIL  fail-OPEN (want 0B refusal, got a RUNNING ELF exit {rc}): {label}")
                all_ok = False
                continue
            refused += 1
            print(f"  PASS  fail-closed 0B: {label}")

    if (loop_ok < len(CASES) or sl_ok < len(STRAIGHT)
            or exitc_ok < len(EXIT_CONTROLS) or refused < len(REFUSE)):
        all_ok = False
    if all_ok:
        print(f"ALL PASS  i64 loop-carry through branched regions ({loop_ok} loop-carry + "
              f"{sl_ok} straight-line + {exitc_ok} exit-control shapes) emit + run to the "
              "independent Python reference — both-branch-same-var, if-in-if (depth 3), "
              "if-wrapped inner while (Sub-step C), outer-if-wrapped loop, for-loop, multi-var, "
              f"write+return / exit-only branches — and {refused} unsound shapes (branch-exit "
              "carry bypass, field-store statements) REFUSE 0B fail-closed; no fail-OPEN drop")
        return 0
    print("FAILURES above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
