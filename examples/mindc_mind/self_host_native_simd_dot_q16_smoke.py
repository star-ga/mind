#!/usr/bin/env python3
"""RI-B2-S10 (#108) — native-ELF PACKED-SIMD Q16.16 DOT-PRODUCT, byte-identity rung.

Opens the SIMD tier: proves the pure-MIND native-ELF backend emits VECTOR SIMD
(SSE2/SSE4.1, 128-bit) whose i64 result is byte-identical to the SAME MLIR canary
dot-l2-q16 the scalar S4 rung hit. Deterministic SIMD, zero LLVM.

Two gates:
  STEP A  selftest_native_elf_simd_probe() — runs the full packed-SIMD arithmetic
    core (movdqu / pmuldq / asr16 emulation / paddq / horizontal reduce / movq /
    movsxd) on a KNOWN 2-element input and writes the 8 LE bytes of the reduced,
    `as i32`-narrowed result. Python recomputes ((a0*b0)>>16 + (a1*b1)>>16) as i32
    and asserts byte-equality. De-risks the SIMD encoders before the loop.
  STEP B  selftest_native_elf_simd_dot_q16(n) — the full dot: EXACT S4 Q16 LCG
    generation, 2-wide packed-SIMD accumulate, horizontal reduce, narrow-once-i32,
    write 8 LE bytes. sha256(the 8 bytes) == the committed canary dot-l2-q16.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_simd_dot_q16_smoke.py
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

# Committed cross-substrate reference: canary dot-l2-q16 = sha256 of the 8 LE bytes
# of the i64 dot result (tests/cross_substrate_identity.rs dot_l2_q16_reproducibility_gate).
CANARY = "1d7f272b85e5f0fd7cf473086fb1da558a723134ff02ef30a4323eb757209823"
LENGTH = 65536

# Step A known input (matches selftest_native_elf_simd_probe bakes).
PROBE_A = (100000, -30000)
PROBE_B = (-50000, -7000)


def _probe_ref() -> bytes:
    p0 = (PROBE_A[0] * PROBE_B[0]) >> 16  # python >> is arithmetic for signed ints
    p1 = (PROBE_A[1] * PROBE_B[1]) >> 16
    acc = p0 + p1
    r = acc & 0xFFFFFFFF
    r = r - (1 << 32) if r >= (1 << 31) else r
    return struct.pack("<q", r)


def _emit(fn) -> bytes:
    fn.restype = ctypes.c_int64
    es = fn()
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def _emit_n(fn, n: int) -> bytes:
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64]
    es = fn(n)
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def _run(elf: bytes, tmp: pathlib.Path, name: str) -> bytes:
    p = tmp / name
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
    for sym in ("selftest_native_elf_simd_probe", "selftest_native_elf_simd_dot_q16"):
        if not hasattr(lib, sym):
            print(f"FAIL  {sym}: symbol absent (RI-B2-S10 not built)")
            return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)

        # ---- STEP A: SIMD arithmetic core on a known 2-element input ----
        elf_a = _emit(lib.selftest_native_elf_simd_probe)
        if not (len(elf_a) > 120 and elf_a[:4] == b"\x7fELF"):
            print(f"  FAIL  probe: not a runnable ELF (len={len(elf_a)})")
            return 1
        out_a = _run(elf_a, tmp, "simd_probe.elf")
        ref_a = _probe_ref()
        if len(out_a) != 8:
            print(f"  FAIL  Step A: expected 8 stdout bytes, got {len(out_a)}: {out_a.hex()}")
            return 1
        va = struct.unpack("<q", out_a)[0]
        vr = struct.unpack("<q", ref_a)[0]
        print(f"  Step A native i64  = {va}  (LE8 {out_a.hex()})")
        print(f"  Step A python ref  = {vr}  (LE8 {ref_a.hex()})")
        if out_a != ref_a:
            print("FAIL  Step A: packed-SIMD arithmetic core != python reference "
                  "(movdqu/pmuldq/asr16/paddq/reduce mismatch — do NOT proceed).")
            return 1
        print("  Step A PASS  SIMD encoders (movdqu/pmuldq/asr16/paddq/pshufd/movq) round-trip.")

        # ---- STEP B: full packed-SIMD Q16 dot, byte-identity to canary ----
        elf_b = _emit_n(lib.selftest_native_elf_simd_dot_q16, LENGTH)
        if not (len(elf_b) > 120 and elf_b[:4] == b"\x7fELF"):
            print(f"  FAIL  simd_dot_q16(n={LENGTH}): not a runnable ELF (len={len(elf_b)})")
            return 1
        out_b = _run(elf_b, tmp, "simd_dot_q16.elf")
        if len(out_b) != 8:
            print(f"  FAIL  Step B: expected 8 stdout bytes, got {len(out_b)}: {out_b.hex()}")
            return 1
        val = struct.unpack("<q", out_b)[0]
        got = hashlib.sha256(out_b).hexdigest()
        print(f"  Step B native i64  = {val}  (LE8 {out_b.hex()})")
        print(f"  Step B native sha  = {got}")
        print(f"  canary dot-l2-q16  = {CANARY}")
        if got == CANARY:
            print("ALL PASS  native-ELF PACKED-SIMD (SSE2/SSE4.1, 128-bit) Q16.16 dot is "
                  "BYTE-IDENTICAL to the MLIR canary dot-l2-q16 — 2-wide pmuldq lane "
                  "accumulate, per-product arithmetic >>16 (SSE psraq emulation), "
                  "horizontal reduce, narrow-once-to-i32, zero MLIR/LLVM. Native-ELF "
                  "SIMD == scalar == MLIR (RI-B2-S10 #108, opens the SIMD tier).")
            return 0
        print("FAIL  native-ELF SIMD Q16 dot hash != canary dot-l2-q16 — the lane "
              "widening or the per-product arithmetic >>16 differs; report the native "
              "value + hash above and STOP (do NOT guess).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
