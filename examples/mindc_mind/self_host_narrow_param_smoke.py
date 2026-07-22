#!/usr/bin/env python3
"""Native-ELF smoke for narrow-width (i8/i16/i32) function PARAMETERS carried by a loop.

A narrow param REASSIGNED inside a TOP-LEVEL `while` now lowers CORRECTLY by the
pure-MIND nb native path: nb_while_carry mints the same fresh width-wrap slot the
emit does for a narrow carried assign (post_id = the wrap slot, next_id advanced
+1), so the carried var's working slot and every later carried var stay
frame-consistent with the emit + the frame count (nb_count_stmt's assign arm). The
value re-wraps two's-complement each iteration and post-loop reads resolve to the
live slot — no stale value, no hang, no frame undercount. Previously ALL narrow
param + loop shapes fail-closed; this proves the top-level-carry sub-shape now
EMITS and RUNS to the value an INDEPENDENT Python reference computes.

Two sub-shapes remain genuinely broken and MUST still fail closed (empty ELF):
  * a narrow param REASSIGNED by a while NESTED inside an `if` branch — needs the
    F2 last_region_exit_rebindings threading nb_while_carry does not yet record.
  * a read-only narrow param used post-loop via a `(x as i64)`-cast fed into a
    binop — a SEPARATE, still-open cast-in-binop lowering bug (`as` is not an infix
    operator in the pratt parser, so the cast mis-lowers and the trailing binop
    operand is emitted as dead code after `ret`). Loop-INDEPENDENT; refused here.

Asserts: (1) the top-level-carry shapes EMIT and run to the Python-reference exit
(incl. two's-complement overflow wrap); (2) the two broken sub-shapes emit an EMPTY
ELF (refused — loud, no silent miscompile); (3) i64 + narrow-no-loop controls still
emit + run (no over-rejection). Guarded on >=1 of each so it cannot pass vacuously.

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


def wrap(v: int, bits: int) -> int:
    """Independent two's-complement wrap-to-width reference (no compiler involved)."""
    m = 1 << bits
    v &= m - 1
    if v >= (m >> 1):
        v -= m
    return v


def _carry_ref(start: int, iters: int, bits: int) -> int:
    """Reference model of `x: iN` reassigned `x = x + 1` `iters` times, then read."""
    x = start
    for _ in range(iters):
        x = wrap(x + 1, bits)
    return wrap(x, bits)


# EMIT + RUN correct: a narrow param reassigned by a TOP-LEVEL while now carries
# correctly. `exp` is computed by the independent Python reference above, then the
# process exit is the low byte of it.
CARRY = [
    ("i8 param reassigned in top-level loop, f(10) +3",
     _carry_ref(10, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i16 param reassigned in top-level loop, f(100) +5",
     _carry_ref(100, 5, 16),
     "fn f(x: i16) -> i64 {\n    let mut c: i64 = 0;\n    while c < 5 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i8 param OVERFLOW two's-complement wrap, f(126) +3 -> -127",
     _carry_ref(126, 3, 8),
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(126); }\n"),
]
# REFUSE: the two genuinely-broken sub-shapes must emit an EMPTY ELF.
REFUSE = [
    ("i8 param reassigned in an IF-NESTED loop (F2 nested-region carry, unrecorded)",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if c < 1 {\n        while c < 4 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 read-only param, post-loop (x as i64)+c cast-in-binop (separate cast bug)",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        c = c + 1;\n    }\n    return (x as i64) + c;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# WORK controls: i64 loops are unaffected; a narrow param with NO loop lowers via the
# entry width-wrap driver (must NOT be over-rejected).
I64_CONTROLS = [
    ("i64 param reassigned in loop", 13,
     "fn f(x: i64) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i64 add", 5,
     "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\nfn main() -> i64 { return add(2, 3); }\n"),
    ("narrow param NO loop (read/return works via wrap driver — must NOT be rejected)", 5,
     "fn pw(x: i32) -> i64 {\n    return x;\n}\nfn main() -> i64 { return pw(5); }\n"),
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
    carried = 0
    refused = 0
    ran = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        def run_elf(elf: bytes) -> int:
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            return subprocess.run([str(p)], timeout=10).returncode

        for label, exp, src in CARRY:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  narrow carry OVER-REJECTED: {label} (emit 0B, want run)")
                all_ok = False
                continue
            want = exp & 0xFF
            rc = run_elf(elf)
            ok = rc == want
            all_ok = all_ok and ok
            carried += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  narrow param carried by loop: {label} "
                  f"-> exit {rc} (python-ref {exp} -> byte {want})")

        for label, src in REFUSE:
            elf = emit(src)
            ok = len(elf) == 0
            all_ok = all_ok and ok
            refused += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  still-broken sub-shape refused: {label} "
                  f"(emit {len(elf)}B, want 0 — fail-closed, NOT run)")

        for label, exp, src in I64_CONTROLS:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  control OVER-REJECTED: {label} (emit 0B)")
                all_ok = False
                continue
            rc = run_elf(elf)
            ok = rc == exp
            all_ok = all_ok and ok
            ran += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  control still works: {label} -> exit {rc} (want {exp})")

    if carried < 1:
        print("FAIL: vacuous (no narrow-param carry ran)")
        return 1
    if refused < 1:
        print("FAIL: vacuous (no broken sub-shape refused)")
        return 1
    if ran < 1:
        print("FAIL: vacuous (no i64 control ran)")
        return 1
    if all_ok:
        print("ALL PASS  narrow-width params carried by a top-level loop emit + run "
              "correct (two's-complement wrap, no stale/hang) while the nested-loop + "
              "cast-in-binop sub-shapes stay fail-closed and i64 fns are unaffected")
        return 0
    print("FAIL  narrow-param carry smoke mis-behaved")
    return 1


if __name__ == "__main__":
    sys.exit(main())
