#!/usr/bin/env python3
"""Fail-closed guard smoke for narrow-width (i8/i16/i32) function PARAMETERS.

Narrow-width params are not yet correctly lowered by the pure-MIND nb native path:
the SysV spill + in-place width-wrap desync the frame/slot count, so a narrow param
SILENTLY MISCOMPILES — `fn f(x: i8){ while c<3 {x=x+1; c=c+1;} return x }` HANGS
(the carried loop var never updates), `fn f(x: i8){ return (x as i64)+1 }` drops the
`+1`, and `f(200)` fails to wrap to -56. Rather than emit a hanging / wrong-valued
ELF, selftest_native_elf now FAILS CLOSED (empty EmitState) on any program whose
body-having fns declare a narrow param — a loud refusal until narrow-param support
lands (roadmap C2).

This asserts: (1) several narrow-param shapes emit an EMPTY ELF (refused — no run,
so the hang can't stall the test); (2) i64-only controls still emit + run correctly
(no over-rejection). Guarded on >=1 of each so it cannot pass vacuously.

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

# REFUSE: a narrow param + a while loop is the broken shape (hang / stale carried
# value) -> must emit EMPTY. Includes a reassigned-in-loop param (hangs) AND a
# read-only-with-loop param (stale value) — both mis-lower, both refused.
NARROW = [
    ("i8 param reassigned in loop (hangs unguarded)",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i16 param reassigned in loop",
     "fn f(x: i16) -> i64 {\n    let mut c: i64 = 0;\n    while c < 5 {\n        x = x + 1;\n        c = c + 1;\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(100); }\n"),
    ("i8 param read-only but a loop present (stale carried value unguarded)",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    while c < 3 {\n        c = c + 1;\n    }\n    return (x as i64) + c;\n}\nfn main() -> i64 { return f(10); }\n"),
    ("i8 param reassigned in an if-nested loop",
     "fn f(x: i8) -> i64 {\n    let mut c: i64 = 0;\n    if c < 1 {\n        while c < 4 {\n            x = x + 1;\n            c = c + 1;\n        }\n    }\n    return x as i64;\n}\nfn main() -> i64 { return f(10); }\n"),
]
# WORK: i64 loops are unaffected; a narrow param with NO loop lowers correctly via
# the entry width-wrap driver (must NOT be over-rejected).
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
    refused = 0
    for label, src in NARROW:
        elf = emit(src)
        ok = len(elf) == 0
        all_ok = all_ok and ok
        refused += 1 if ok else 0
        print(f"  {'PASS' if ok else 'FAIL'}  narrow-param refused: {label} "
              f"(emit {len(elf)}B, want 0 — fail-closed, NOT run)")
    ran = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for label, exp, src in I64_CONTROLS:
            elf = emit(src)
            if not elf:
                print(f"  FAIL  i64 control OVER-REJECTED: {label} (emit 0B)")
                all_ok = False
                continue
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            rc = subprocess.run([str(p)], timeout=10).returncode
            ok = rc == exp
            all_ok = all_ok and ok
            ran += 1 if ok else 0
            print(f"  {'PASS' if ok else 'FAIL'}  i64 control still works: {label} -> exit {rc} (want {exp})")
    if refused < 1:
        print("FAIL: vacuous (no narrow-param refused)")
        return 1
    if ran < 1:
        print("FAIL: vacuous (no i64 control ran)")
        return 1
    if all_ok:
        print("ALL PASS  narrow-width params fail closed (loud refusal, no silent "
              "miscompile / hang) while i64 fns are unaffected")
        return 0
    print("FAIL  narrow-param guard mis-behaved")
    return 1


if __name__ == "__main__":
    sys.exit(main())
