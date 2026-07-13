#!/usr/bin/env python3
"""RI-B2-S2 STEP A (#108) — de-risk the C1 "emit 8 LE bytes to stdout + hash" gate.

`selftest_native_elf_write8(val)` emits a runnable x86-64 ET_EXEC that writes the
8 LITTLE-ENDIAN bytes of a KNOWN i64 to stdout (fd 1) via the raw write(2)
syscall, then exit(0). This proves the exact mechanism Step B's byte-identity
canary relies on:
  (a) __mind_write emits a buffer to fd 1 correctly,
  (b) the on-wire byte order is little-endian,
  (c) sha256-over-stdout is deterministic (run twice -> same hash).

The oracle is Python: sha256(the 8 captured bytes) must equal
sha256(struct.pack('<q', val)) computed independently here.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_write8_smoke.py
"""
import ctypes
import hashlib
import os
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"


def mind_write8_elf(lib, val: int) -> bytes:
    fn = lib.selftest_native_elf_write8
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(val)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf_capture(elf: bytes, tmp: pathlib.Path) -> bytes:
    p = tmp / "mind_write8.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)], capture_output=True).stdout


def main() -> int:
    so = os.environ.get("MINDC_SO", str(_DEFAULT_SO))
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf_write8"):
        print("FAIL  selftest_native_elf_write8: symbol absent (RI-B2-S2 Step A not built)")
        return 1

    val = 39  # reuse the S1 dot a=[3,4],b=[5,6] = 39 (a known integer)
    expect = hashlib.sha256(struct.pack("<q", val)).hexdigest()
    all_ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_write8_elf(lib, val)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  write8({val}): not a runnable ELF (len={len(elf)})")
            return 1
        out1 = run_elf_capture(elf, tmp)
        out2 = run_elf_capture(elf, tmp)  # determinism
        h1 = hashlib.sha256(out1).hexdigest()
        h2 = hashlib.sha256(out2).hexdigest()

        ok_len = len(out1) == 8
        ok_le = out1 == struct.pack("<q", val)
        ok_det = h1 == h2
        ok_hash = h1 == expect
        all_ok = ok_len and ok_le and ok_det and ok_hash
        print(f"  {'PASS' if ok_len else 'FAIL'}  emitted exactly 8 bytes (got {len(out1)})")
        print(f"  {'PASS' if ok_le else 'FAIL'}  bytes are LITTLE-ENDIAN i64 of {val}: "
              f"{out1.hex()} vs {struct.pack('<q', val).hex()}")
        print(f"  {'PASS' if ok_det else 'FAIL'}  sha256-over-stdout deterministic "
              f"(run1 {h1[:12]}... == run2 {h2[:12]}...)")
        print(f"  {'PASS' if ok_hash else 'FAIL'}  sha256(stdout) == "
              f"sha256(struct.pack('<q', {val})) = {expect[:12]}...")
    if all_ok:
        print("ALL PASS  C1 emit-8-LE-bytes-to-stdout + sha256 harness proven on a known "
              "i64 (RI-B2-S2 Step A #108) — __mind_write + LE order + deterministic hash")
        return 0
    print("FAIL  RI-B2-S2 Step A write8-to-stdout harness")
    return 1


if __name__ == "__main__":
    sys.exit(main())
