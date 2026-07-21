#!/usr/bin/env python3
"""Roadmap C2 declared-width AUTO-WRAP driver — narrow-int (i8/i16/i32) `let` and
assignment arithmetic wraps two's-complement at the DECLARED width automatically,
with NO explicit __mind_wrap_* intrinsic in the source.

The prior C2 rung landed the mechanical wrap primitives (nb_movsx_rax_al /
nb_movsx_rax_ax / nb_movsxd_rax_eax) behind the user-reachable __mind_wrap_{i8,i16,i32}
intrinsics. This rung lands the DRIVER: a per-binding declared bit-width recorded in
the native let-env (5th entry word, 64 default), read back at the `let` binding and at
each reassignment. A narrow-declared value's full-64-bit arithmetic result is wrapped
(movsx/movsxd into a fresh frame slot, mirrored in the nb_count_stmt frame sizing)
before the name binds — so `let x: i32 = big + big` wraps at 32 bits by itself. The
width defaults to 64 for every unannotated/i64/float/struct binding, emitting NOTHING —
which is what keeps the all-i64 self-compile byte-identical (main.mind declares no
narrow lets).

No frozen byte-oracle exists for this construct (the deleted Rust src/native backend
had no narrow-int codegen). The oracle is EXECUTION CORRECTNESS on the CPU: a real user
MIND `main` compiled to native ELF via `selftest_native_elf` exercises the driver and
returns 42 IFF every wrap fires correctly. Non-fakeable: every `e*` term is exactly 0
iff its wrap is correct, and `main` compares the FULL 64-bit `err` against 0 BEFORE the
exit-code mod-256 truncation — a non-wrapping lowering leaves deltas of 2^32 / 2^16 /
2^8 (all multiples of 256, which a mod-256 sum check would mask) and still exits 43:

  * auto-wrap at overflow — `let x: i32 = a + 1` with a = INT32_MAX must yield
    INT32_MIN (the 64-bit register sum is 2147483648; the driver movsxd-wraps it at
    the binding). Same for i16 (32767+1 -> -32768) and i8 (127+1 -> -128).
  * truncation at binding — `let t: i16 = 70000` binds low16(70000) = 4464.
  * in-range identity     — `let p: i32 = 100` round-trips unchanged.
  * assignment re-wrap    — `w = w + 100` on an i8-declared w (100+100=200) wraps to
    -56 at the REASSIGNMENT, proving the width survives the let, the while-loop
    carry rebind, AND drives the assign arm. (The reassignment sits inside a
    `while` because a straight-line top-level assign statement is a PRE-EXISTING
    fail-closed gap in the no-feed trace-hash path — selftest_native_elf returns
    the empty EmitState for `let w: i64 = 100; w = w + 100;` on an UNMODIFIED
    tree too, verified during this rung. Not a driver limitation.)

Usage:
  MINDC_SO=/path/to.so python3 self_host_native_autowrap_smoke.py
"""
import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

MARKER = 42

# Every `e*` term below is algebraically zero IFF the auto-wrap is correct.
#   e32 = x + 2^31    where x: i32 = INT32_MAX + 1        -> (-2^31) + 2^31 = 0
#   e16 = y + 2^15    where y: i16 = INT16_MAX + 1        -> (-2^15) + 2^15 = 0
#   e8  = z + 2^7     where z: i8  = INT8_MAX + 1         -> (-128)  + 128  = 0
#   ep16= t - 4464    where t: i16 = 70000 (low16 0x1170) -> 4464 - 4464 = 0
#   ep  = p - 100     where p: i32 = 100 (in-range)       -> 0
#   ew  = w + 56      where w: i8 reassigned w+100 (200) in-loop -> (-56) + 56 = 0
SRC = f"""fn main() -> i64 {{
    let a: i32 = 2147483647;
    let x: i32 = a + 1;
    let e32: i64 = x + 2147483648;
    let b: i16 = 32767;
    let y: i16 = b + 1;
    let e16: i64 = y + 32768;
    let c: i8 = 127;
    let z: i8 = c + 1;
    let e8: i64 = z + 128;
    let t: i16 = 70000;
    let ep16: i64 = t - 4464;
    let p: i32 = 100;
    let ep: i64 = p - 100;
    let w: i8 = 100;
    let i: i64 = 0;
    while i < 1 {{
        w = w + 100;
        i = i + 1;
    }}
    let ew: i64 = w + 56;
    let err: i64 = e32 + e16 + e8 + ep16 + ep + ew;
    if err == 0 {{
        {MARKER}
    }} else {{
        43
    }}
}}
"""


# C2 param/return width driver: narrow-int PARAMS wrap at fn entry (in place in the
# param's home slot, after the SysV spill) and narrow-int RETURN types wrap rax at the
# declared width before the epilogue — with NO explicit __mind_wrap_* in the source.
# Discrimination is a FULL-64-bit err==0 compare BEFORE exit truncation (every wrong
# delta is a multiple of 256, which a mod-256 exit-sum check would mask):
#   e1 = wadd(INT32_MAX) + 2^31 where wadd(x: i32) -> i32 { x + 1 } — the 64-bit sum
#        2^31 must movsxd-wrap to INT32_MIN at the RETURN (delta 2^32 if not).
#   e2 = pw32(2^32)  where pw32(x: i32) -> i64 { x } — 2^32 must wrap to 0 at ENTRY;
#        the i64 return adds no wrap, so a missing PARAM wrap leaks the full 2^32.
#   e3 = pw16(65536) where pw16(x: i16) -> i64 { x } — the i16 twin (delta 2^16).
#   e4 = ret8(200) + 56 where ret8(v: i64) -> i8 { v } — the full-width param passes
#        200 through untouched; the i8 RETURN must wrap it to -56 (delta 2^8).
SRC_PARAM_RET = f"""fn wadd(x: i32) -> i32 {{
    x + 1
}}
fn pw32(x: i32) -> i64 {{
    x
}}
fn pw16(x: i16) -> i64 {{
    x
}}
fn ret8(v: i64) -> i8 {{
    v
}}
fn main() -> i64 {{
    let r: i64 = wadd(2147483647);
    let e1: i64 = r + 2147483648;
    let e2: i64 = pw32(4294967296);
    let e3: i64 = pw16(65536);
    let t: i64 = ret8(200);
    let e4: i64 = t + 56;
    let err: i64 = e1 + e2 + e3 + e4;
    if err == 0 {{
        {MARKER}
    }} else {{
        43
    }}
}}
"""

# The once-latent count/emit if-merge width asymmetry, now FIXED and emittable: a
# narrow-declared w is assigned in BOTH branches of an `if` (so the merge machinery
# rebinds it), then REASSIGNED after the if. nb_count_bind_merged now inherits w's
# true declared width (8) into the count-lets — mirroring nb_rebind_merges — so the
# post-if `w = w + 0` counts the SAME +1 wrap slot the emit side emits (previously
# count +0 vs emit +1: a one-slot frame undercount). Loop once: w=100 -> +100 wraps
# to -56 (then-branch), merge, `w + 0` re-wraps -56, exit-carry; ew = w + 56 == 0.
SRC_MERGE_ASSIGN = f"""fn main() -> i64 {{
    let w: i8 = 100;
    let i: i64 = 0;
    while i < 1 {{
        if i < 1 {{
            w = w + 100;
        }} else {{
            w = w + 1;
        }}
        w = w + 0;
        i = i + 1;
    }}
    let ew: i64 = w + 56;
    if ew == 0 {{
        {MARKER}
    }} else {{
        43
    }}
}}
"""


def mind_autowrap_elf(lib, src: str) -> bytes:
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
    p = tmp / "mind_autowrap.elf"
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

    cases = [
        (
            "let+assign",
            SRC,
            "i8/i16/i32 let+assign wrap with NO explicit intrinsic",
        ),
        (
            "param+return",
            SRC_PARAM_RET,
            "narrow-int PARAM wraps at entry, narrow-int RETURN wraps rax at the "
            "declared width",
        ),
        (
            "if-merge-narrow-assign",
            SRC_MERGE_ASSIGN,
            "post-if reassign of an if-merged narrow var (count inherits the true "
            "width — the fixed count/emit symmetry)",
        ),
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        all_ok = True
        for name, src, what in cases:
            elf = mind_autowrap_elf(lib, src)
            if not (len(elf) > 120 and elf[:4] == b"\x7fELF"):
                print(f"  FAIL  {name}: not a runnable ELF (len={len(elf)})")
                all_ok = False
                continue
            got = run_elf(elf, tmp)
            ok = got == MARKER
            all_ok = all_ok and ok
            print(
                f"  {'PASS' if ok else 'FAIL'}  {name} AUTO-wrap exit={got} "
                f"expected={MARKER}  (elf {len(elf)}B, {what}, zero MLIR/LLVM)"
            )
        if all_ok:
            print(
                "ALL PASS  per-binding declared width (native let-env word 5, default "
                "64) auto-emits the movsx/movsxd wrap at narrow `let` bindings, "
                "reassignments, fn ENTRY (narrow params, in place in the home slot), "
                "and fn RETURN (narrow return types, in rax before the epilogue) — "
                "with no __mind_wrap_* call (roadmap C2 declared-width driver rungs)"
            )
            return 0
        print(
            "FAIL  declared-width driver: exit != 42 means a narrow let/assign/param/"
            "return did not auto-wrap (i64-widened), wrapped at the wrong width, or "
            "corrupted an in-range value — do NOT guess (report the native exit above)."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
