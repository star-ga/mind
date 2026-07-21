#!/usr/bin/env python3
"""Roadmap C2 narrow-int native-ELF rung — user-reachable i8/i16/i32 truncating
stores + zero-extend loads, lowered THROUGH the general nb_expr / nb_emit_intrinsic
path (not a bespoke kernel driver).

Before this rung only `__mind_store_i8` / `__mind_load_i8` (byte, zero-extend) were
wired into the intrinsic name-classifier + dispatch; the i16/i32 primitives existed
but were reachable only inside the hand-written tensor kernels (fixed frame disps).
This rung adds the four user-reachable names `__mind_store_i16`, `__mind_load_i16`,
`__mind_store_i32`, `__mind_load_i32` (classifier arms -> kinds 13/14/15/16) plus two
new encoders (`nb_emit_store_i32` = 89 08, `nb_emit_load_i16_zx` = 48 0F B7 00,
`nb_emit_load_i32_zx` = 8B 00), reusing the existing `nb_emit_store_i16` (66 89 08).

No frozen byte-oracle exists for this construct (the deleted Rust `src/native`
backend had no user-level narrow store/load smoke; and by design the low N bits of a
truncated value are identical whether or not truncation happens, so a byte-oracle
would be circular). The oracle is EXECUTION CORRECTNESS on the CPU: a real user MIND
`main` compiled to native ELF via `selftest_native_elf` exercises every new op and
returns 42 IFF all narrow ops behave correctly. The check is non-fakeable:

  * truncating STORE — a value with bits above the store width is written into a
    pre-zeroed slot; a NEIGHBOR byte just past the width is read back and MUST be 0
    (a wrong-width / non-truncating store would clobber it with the high byte 0x11).
  * store LOW byte — the expected low byte is read back and differenced (0 iff the
    store actually wrote).
  * zero-extend LOAD — a full 8-byte value with distinct nonzero bytes across all 64
    bits is stored, then narrow-loaded and right-shifted by the load width; the high
    part MUST be 0 (a wrong-width load would surface nonzero high bytes).

`err` sums all nine independently-zero-iff-correct terms; `main` returns err + 42 so
a correct build exits 42 and ANY miscompiled narrow op shifts the exit code. The CPU
is the reference — zero MLIR/LLVM in the native path.

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_narrowint_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

# --- fixture constants (documented in hex, embedded as decimal in the source) ---
SW16 = 0x11AABB          # i16 store: low16=0xAABB, byte>=16 (0x11) must not leak
SW32 = 0x1199887766      # i32 store: low32=0x99887766, byte>=32 (0x11) must not leak
SW8 = 0x1122             # i8  store: low8=0x22,    byte>=8  (0x11) must not leak
LV = 0x0011223344556677  # load fixture: distinct nonzero bytes across 64 bits

# expected written low bytes (little-endian)
LB16 = SW16 & 0xFF       # 0xBB = 187
LB32 = SW32 & 0xFF       # 0x66 = 102
LB8 = SW8 & 0xFF         # 0x22 = 34

MARKER = 42

SRC = f"""fn main() -> i64 {{
    let p: i64 = __mind_alloc(64);
    __mind_store_i64(p, 0);
    __mind_store_i64(p + 8, 0);
    __mind_store_i64(p + 16, 0);
    __mind_store_i64(p + 24, {LV});

    __mind_store_i16(p, {SW16});
    let sn16: i64 = __mind_load_i8(p + 2);
    let a16: i64 = __mind_load_i8(p);
    let e16: i64 = {LB16} - a16;

    __mind_store_i32(p + 8, {SW32});
    let sn32: i64 = __mind_load_i8(p + 12);
    let a32: i64 = __mind_load_i8(p + 8);
    let e32: i64 = {LB32} - a32;

    __mind_store_i8(p + 16, {SW8});
    let sn8: i64 = __mind_load_i8(p + 17);
    let a8: i64 = __mind_load_i8(p + 16);
    let e8: i64 = {LB8} - a8;

    let l16: i64 = __mind_load_i16(p + 24);
    let l16hi: i64 = l16 >> 16;
    let l32: i64 = __mind_load_i32(p + 24);
    let l32hi: i64 = l32 >> 32;
    let l8: i64 = __mind_load_i8(p + 24);
    let l8hi: i64 = l8 >> 8;

    let err: i64 = sn16 + sn32 + sn8 + e16 + e32 + e8 + l16hi + l32hi + l8hi;
    err + {MARKER}
}}
"""


def mind_narrowint_elf(lib, src: str) -> bytes:
    fn = lib.selftest_native_elf
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64]
    enc = src.encode()
    buf = ctypes.create_string_buffer(enc, len(enc))
    es = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(enc))
    rd = lambda addr, o=0: ctypes.cast(addr + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf: String handle (addr/len/cap)
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def run_elf(elf: bytes, tmp: pathlib.Path) -> int:
    p = tmp / "mind_narrowint.elf"
    p.write_bytes(elf)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return subprocess.run([str(p)]).returncode


def main() -> int:
    so = str(resolve_so())  # MINDC_SO verbatim, else fresh build (never stale)
    if not os.path.exists(so):
        if os.environ.get("MINDC_SO"):
            print(f"FAIL  MINDC_SO set but missing: {so!r}")
            return 1
        print(f"SKIP  {so} not built")
        return 0
    lib = ctypes.CDLL(so)
    if not hasattr(lib, "selftest_native_elf"):
        print("FAIL  selftest_native_elf: symbol absent")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        elf = mind_narrowint_elf(lib, SRC)
        if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
            print(f"  FAIL  narrowint: not a runnable ELF (len={len(elf)})")
            return 1
        got = run_elf(elf, tmp)
        ok = got == MARKER
        print(
            f"  {'PASS' if ok else 'FAIL'}  narrow-int store/load exit={got} "
            f"expected={MARKER}  (elf {len(elf)}B, i8/i16/i32 trunc-store + "
            f"zero-extend load via nb_emit_intrinsic, zero MLIR/LLVM)"
        )
        if ok:
            print(
                "ALL PASS  user-reachable __mind_{store,load}_{i16,i32} (+ existing i8) "
                "lower to native x86-64 through the general nb_expr / nb_emit_intrinsic "
                "path — truncating byte/word/dword stores do not clobber neighbours and "
                "narrow loads zero-extend correctly (roadmap C2 narrow-int rung)"
            )
            return 0
        print(
            "FAIL  narrow-int rung: exit != 42 means a truncating store clobbered a "
            "neighbour byte, a store wrote nothing, or a narrow load did not truncate/"
            "zero-extend — do NOT guess (report the native exit above)."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
