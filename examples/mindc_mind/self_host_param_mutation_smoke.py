#!/usr/bin/env python3
"""CPU-as-oracle smoke for the param-mutation fix (nb_expr ident arm: consult the
let-env BEFORE resolve_param).

A REASSIGNED parameter (loop-carried `x = x + 1`, or an if-branch merge) is bound
into the let-env at its live post-slot by nb_lets_bind; its parameter home slot is
then STALE. The ident arm previously resolved via resolve_param FIRST, so every
read of a reassigned param returned the stale home slot and SILENTLY DROPPED the
mutation — a wrong value with no fail-closed signal (verified: `fn f(x){ while
c<3 { x=x+1 } return x }` for f(10) emitted a clean ELF that exited 10, not 13).

This feeds that shape for several (arg, iters) pairs through the pure-MIND
self-host native-ELF entry (selftest_native_elf_h), runs the emitted ELF, and
asserts exit == arg + iters — the mathematically correct mutated value, an oracle
independent of ANY compiler (the real mindc MLIR path agrees: it returns arg+iters
too). The stale-home-slot bug returns `arg`, so the exact-value check over five
distinct (arg, iters) pairs is non-vacuous and catches a regression in either
direction.

Env: MINDC_SO (prebuilt .so, skips the build) or MINDC_BIN (default mindc).
Template: self_host_native_tensor_matmul_smoke.py.
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
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit(1)
    return out, True


def pm_src(arg: int, iters: int) -> bytes:
    """A fn whose i64 param is reassigned `iters` times in a while loop, then read.
    Correct result = arg + iters; the stale-home-slot bug returns arg."""
    return (
        f"fn f(x: i64) -> i64 {{\n"
        f"    let mut c: i64 = 0;\n"
        f"    while c < {iters} {{\n"
        f"        x = x + 1;\n"
        f"        c = c + 1;\n"
        f"    }}\n"
        f"    return x;\n"
        f"}}\n"
        f"fn main() -> i64 {{\n"
        f"    return f({arg});\n"
        f"}}\n"
    ).encode()


def main() -> int:
    so, built = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        sys.exit(1)
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_h"):
        print("FAIL: selftest_native_elf_h absent (self-host native entry missing)")
        return 1
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]

    def mind_elf(src: bytes) -> bytes:
        b = ctypes.create_string_buffer(src, len(src))
        h = ctypes.create_string_buffer(b"\x00" * 32, 32)
        es = lib.selftest_native_elf_h(
            ctypes.cast(b, ctypes.c_void_p).value, len(src),
            ctypes.cast(h, ctypes.c_void_p).value,
        )
        sh = rd(es, 0)
        ln = rd(sh, 8)
        return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""

    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for arg, iters in [(10, 3), (20, 3), (0, 3), (7, 5), (100, 1)]:
            elf = mind_elf(pm_src(arg, iters))
            if not elf:
                print(f"  FAIL  f({arg}) [iters={iters}] fail-closed (empty ELF)")
                all_ok = False
                continue
            p = tmp / "m.elf"
            p.write_bytes(elf)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            rc = subprocess.run([str(p)]).returncode
            exp = arg + iters
            ok = rc == exp
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  reassigned param f({arg}) after "
                f"{iters}x `x = x + 1` -> exit {rc} (want {exp}; the stale-home-slot "
                f"bug would return {arg})"
            )
    if all_ok:
        print(
            "ALL PASS  a reassigned parameter reads its live carried slot (let-env "
            "before resolve_param), not the stale home slot — no silent mutation drop"
        )
        if built:
            try:
                os.unlink(so)
            except OSError:
                pass
        return 0
    print("FAIL  a reassigned parameter read a stale slot (silent wrong value)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
