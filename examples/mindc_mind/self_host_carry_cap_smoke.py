#!/usr/bin/env python3
"""Cap-guard smoke for the self-host loop-carry / loop-frame scratch tables.

nb_carry_record and nb_loops_push write into FIXED 256-entry buffers
(__mind_alloc(256 * 4 * 8) and __mind_alloc(256 * 3 * 8)). A 257th distinct
carried var / nested loop frame would store one slot past the end and corrupt the
adjacent arena allocation — the compiler's own state — a silent miscompile. Both
now cap-guard: at n >= 256 they refuse (fail-closed -1) instead of overflowing.

This calls each recorder ctypes-directly 300 times and asserts it records exactly
256 (return 0) then refuses the remaining 44 (return -1), with the count cell
staying at 256 (no overflow). Non-vacuous by construction: without the guard the
count runs to 300 (a real overflow), so the exact 256 cap is the discriminator.

Env: MINDC_SO (prebuilt .so) or MINDC_BIN (default mindc).
"""
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_MIND = os.path.join(HERE, "main.mind")
N = 300
CAP = 256


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


def cbuf(nbytes, init_zero=False):
    b = ctypes.create_string_buffer(nbytes)
    if init_zero:
        ctypes.memset(b, 0, nbytes)
    return b, ctypes.cast(b, ctypes.c_void_p).value


def check_carry(lib):
    lib.nb_carry_record.restype = ctypes.c_int64
    lib.nb_carry_record.argtypes = [ctypes.c_int64] * 8
    rd = lambda a: ctypes.cast(a, ctypes.POINTER(ctypes.c_int64))[0]
    # + slack so that (if the guard were absent) an overflow lands in owned memory
    # rather than segfaulting the test — we assert the guard, not the crash.
    _c, carry = cbuf(CAP * 4 * 8 + 4096)
    cbo, ccell = cbuf(8, True)
    names = b"".join(f"{i:03d}".encode() for i in range(N))
    _b, buf = cbuf(len(names))
    ctypes.memmove(_b, names, len(names))
    _l, lets = cbuf(8)
    _lc, lcell = cbuf(8, True)
    rets = [lib.nb_carry_record(carry, ccell, buf, i * 3, i * 3 + 3, i, lets, lcell) for i in range(N)]
    final = rd(ccell)
    recorded = sum(1 for r in rets if r == 0)
    refused = sum(1 for r in rets if r == -1)
    ok = final == CAP and recorded == CAP and refused == N - CAP and rets[CAP - 1] == 0 and rets[CAP] == -1
    print(f"  {'PASS' if ok else 'FAIL'}  nb_carry_record: recorded={recorded} refused={refused} "
          f"final_count={final} (cap {CAP}, no overflow)")
    return ok


def check_loops(lib):
    lib.nb_loops_push.restype = ctypes.c_int64
    lib.nb_loops_push.argtypes = [ctypes.c_int64] * 4
    rd = lambda a: ctypes.cast(a, ctypes.POINTER(ctypes.c_int64))[0]
    _l, loops = cbuf(CAP * 3 * 8 + 4096)
    _lc, lpcell = cbuf(8, True)
    _bb, breaks = cbuf(8)
    rets = [lib.nb_loops_push(loops, lpcell, i, breaks) for i in range(N)]
    final = rd(lpcell)
    recorded = sum(1 for r in rets if r == 0)
    refused = sum(1 for r in rets if r == -1)
    ok = final == CAP and recorded == CAP and refused == N - CAP and rets[CAP - 1] == 0 and rets[CAP] == -1
    print(f"  {'PASS' if ok else 'FAIL'}  nb_loops_push:  recorded={recorded} refused={refused} "
          f"final_count={final} (cap {CAP}, no overflow)")
    return ok


def main() -> int:
    so = build_so()
    st = os.stat(so)
    print(f"SO: {so} ({st.st_size} bytes)")
    if st.st_size < 4096:
        print("FAIL: .so too small (stub?)")
        return 1
    lib = ctypes.CDLL(so)
    for sym in ("nb_carry_record", "nb_loops_push"):
        if not hasattr(lib, sym):
            print(f"FAIL: {sym} absent (cap-guard build missing)")
            return 1
    ok = check_carry(lib) & check_loops(lib)
    if ok:
        print("ALL PASS  carry / loop-frame scratch tables fail closed at the 256 cap — "
              "no silent buffer overflow into the compiler's own arena state")
        return 0
    print("FAIL  a cap guard did not fire — potential 256-cap buffer overflow")
    return 1


if __name__ == "__main__":
    sys.exit(main())
