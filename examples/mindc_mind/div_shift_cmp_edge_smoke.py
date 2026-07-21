"""
C3 division / shift / compare EDGE-CORRECTNESS smoke (native-ELF, CPU-as-oracle).

The signed C3 emitter arms already exist byte-for-byte in main.mind:
  * div/mod : nb_div_guarded (cqo + idiv rcx) with the branchless zero-guard
              (x/0 == 0, x%0 == 0) and the INT_MIN/-1 guard (no #DE trap).
  * shl/sar : inline in nb_arith_rax_mem (`shl rax,cl` D3 /4 ; `sar rax,cl` D3 /7
              -- `>>` is ARITHMETIC, sign-extending).
  * compare : nb_setcc_opcode + nb_cmp_rax_mem emit the SIGNED setcc family
              (setl/setle/setg/setge/sete/setne).

`mod_operator_smoke.py` only pins positive `17 % 5 == 2`. Nothing exercised the
DEFINED EDGE VALUES at runtime: negative operands (x86 idiv truncates toward
zero, remainder takes the sign of the dividend), INT_MIN / -1 (must NOT raise
#DE / SIGFPE), x/0 and x%0 (guarded to 0), arithmetic vs logical `>>`, and each
of the six signed compare predicates on negative operands (proving setl, not the
unsigned setb -- (-3) < 5 is TRUE signed, FALSE unsigned).

This adds NO emitter surface. It compiles each snippet with the pure-MIND
native-ELF backend (selftest_native_elf_h), RUNS the emitted ELF, and asserts the
process exit code equals the value MIND defines for that edge. A SIGFPE from an
unguarded INT_MIN/-1 idiv shows up as a negative returncode and fails LOUD.

All operands are routed through `let` bindings so the value lands in a stack slot
and the mem-operand `nb_arith_rax_mem` / `nb_div_guarded` / `nb_cmp_rax_mem` arm
actually runs (not a compile-time-folded constant).

Run:  MINDC_SO=.../libmindc_mind.so python3 examples/mindc_mind/div_shift_cmp_edge_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))


def _read_es_buf(es: int) -> bytes:
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    ln = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""


# (name, MIND source, expected process exit code (0..255), why)
# Exit codes are the low byte of the returned i64. All expected values are
# non-negative so a signal death (e.g. SIGFPE = returncode -8) is distinguishable.
CASES = [
    # --- signed division / remainder edges (nb_div_guarded, cqo + idiv) ---
    ("div_neg_dividend",
     "fn main() -> i64 { let a: i64 = 0 - 17; let b: i64 = 5; return a / b; }",
     253,  # -17 / 5 = -3 (trunc toward zero); (-3) & 0xFF = 253
     "-17/5 == -3 (idiv truncates toward zero)"),
    ("mod_neg_dividend",
     "fn main() -> i64 { let a: i64 = 0 - 17; let b: i64 = 5; return a % b; }",
     254,  # -17 % 5 = -2 (remainder takes sign of dividend); (-2)&0xFF = 254
     "-17%5 == -2 (remainder sign follows dividend)"),
    ("mod_neg_divisor",
     "fn main() -> i64 { let a: i64 = 17; let b: i64 = 0 - 5; return a % b; }",
     2,   # 17 % -5 = 2
     "17%-5 == 2"),
    ("div_both_neg",
     "fn main() -> i64 { let a: i64 = 0 - 17; let b: i64 = 0 - 5; return a / b; }",
     3,   # -17 / -5 = 3
     "-17/-5 == 3"),
    ("div_by_zero",
     "fn main() -> i64 { let z: i64 = 5 - 5; return 7 / z; }",
     0,   # guarded: x/0 == 0
     "7/0 == 0 (zero-guard, no trap)"),
    ("mod_by_zero",
     "fn main() -> i64 { let z: i64 = 5 - 5; return 7 % z; }",
     0,   # guarded: x%0 == 0
     "7%0 == 0 (zero-guard, no trap)"),
    ("intmin_div_neg1",
     "fn main() -> i64 { let a: i64 = 1 << 63; let b: i64 = 0 - 1; "
     "let q: i64 = a / b; if q == a { return 42; } return 7; }",
     42,  # INT_MIN / -1 == INT_MIN (guard prevents #DE); compare proves it
     "INT_MIN/-1 == INT_MIN, NO SIGFPE"),
    # --- shifts (shl D3/4, sar D3/7 -- arithmetic >>) ---
    ("shl_basic",
     "fn main() -> i64 { let a: i64 = 1; let n: i64 = 4; return a << n; }",
     16,  # 1 << 4
     "1<<4 == 16"),
    ("shr_arithmetic",
     "fn main() -> i64 { let a: i64 = 0 - 256; let n: i64 = 2; "
     "let r: i64 = a >> n; if r == (0 - 64) { return 64; } return 9; }",
     64,  # -256 >> 2 == -64 (arithmetic/sign-extending, NOT logical)
     "-256>>2 == -64 (sar, sign-extending)"),
    ("shr_positive",
     "fn main() -> i64 { let a: i64 = 100; let n: i64 = 3; return a >> n; }",
     12,  # 100 >> 3
     "100>>3 == 12"),
    # --- six SIGNED compare predicates on negative operands ---
    # (-3) < 5 is TRUE signed but FALSE unsigned -> proves setl, not setb.
    ("cmp_lt_signed",
     "fn main() -> i64 { let a: i64 = 0 - 3; let b: i64 = 5; return a < b; }",
     1, "(-3)<5 == true (signed setl)"),
    ("cmp_ge_signed",
     "fn main() -> i64 { let a: i64 = 0 - 3; let b: i64 = 5; return a >= b; }",
     0, "(-3)>=5 == false (signed setge)"),
    ("cmp_gt_signed",
     "fn main() -> i64 { let a: i64 = 5; let b: i64 = 0 - 3; return a > b; }",
     1, "5>(-3) == true (signed setg, unsigned would be false)"),
    ("cmp_le_eq",
     "fn main() -> i64 { let a: i64 = 0 - 3; let b: i64 = 0 - 3; return a <= b; }",
     1, "(-3)<=(-3) == true (setle)"),
    ("cmp_eq",
     "fn main() -> i64 { let a: i64 = 0 - 5; let b: i64 = 0 - 5; return a == b; }",
     1, "(-5)==(-5) == true (sete)"),
    ("cmp_ne",
     "fn main() -> i64 { let a: i64 = 0 - 5; let b: i64 = 5; return a != b; }",
     1, "(-5)!=5 == true (setne)"),
]


def run_case(lib, name, src, want, why) -> int:
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    raw = src.encode()
    src_buf = ctypes.create_string_buffer(raw, len(raw))
    hash_buf = ctypes.create_string_buffer(bytes(32), 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(raw),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    elf = _read_es_buf(es)
    if not elf:
        print(f"  FAIL  {name}: pure-MIND emitter returned an empty ELF (failed closed)")
        return 1
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / f"{name}.elf"
        p.write_bytes(elf)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        code = subprocess.run([str(p)]).returncode
    if code == want:
        print(f"  PASS  {name}: exit {code} -- {why}")
        return 0
    if code < 0:
        print(f"  FAIL  {name}: died by signal {-code} (expected exit {want}) -- {why}")
        return 1
    print(f"  FAIL  {name}: exit {code}, expected {want} -- {why}")
    return 1


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set -- refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0
    lib = ctypes.CDLL(str(SO))
    print("[native C3 edge-correctness: div/mod guards, sar/shl, signed setcc]")
    rc = 0
    for name, src, want, why in CASES:
        rc |= run_case(lib, name, src, want, why)
    if rc == 0:
        print(f"\nALL PASS  ({len(CASES)} native C3 edge cases compute MIND-defined values)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
