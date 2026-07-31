"""
Self-host ARRAY smoke — the pure-MIND mic@3 note path emits fixed-i64 constant
arrays (OP_CONST_ARRAY 0x19 / OP_ARRAY_LOAD 0x1A) BYTE-IDENTICAL to the Rust
oracle (`mindc --emit-mic3`), for array construction + index read.

Covers the note-builder-coverage gap #40/#111: `let a = [10,20,30]; a[1]`
FAIL-CLOSED (nb_len=0) in the `_u` note path because the oracle uses the compact
array opcodes, not the __mind_alloc+store+load expansion.

Byte-identity vs the live oracle IS the correctness proof (the oracle mic@3 is
the CPU semantics reference). Fail-closed, no false green:
  * oracle mindc missing/errors -> BLOCKED (exit 2)
  * MIND emits EMPTY buf (still fail-closed) -> DRIFT (exit 1)
  * MIND bytes != oracle bytes -> DRIFT (exit 1)

Run:
  MINDC_SO=/path/libmindc_mind.so MINDC_BIN=./target/release/mindc \
  python3 examples/mindc_mind/self_host_array_smoke.py
"""

import ctypes
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

_DEFAULT_MINDC = _HERE.parents[1] / "target" / "release" / "mindc"
_Int64Ptr = ctypes.POINTER(ctypes.c_int64)

EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_BLOCKED = 2

# (id, source). Every fixture is a self-contained module so the oracle type-checks
# it standalone and regenerates the golden live.
CASES = [
    ("arr3_mid", "fn main() -> i64 {\n    let a: [i64; 3] = [10, 20, 30];\n    return a[1];\n}\n"),
    ("arr2_zero", "fn main() -> i64 {\n    let a: [i64; 2] = [7, 9];\n    return a[0];\n}\n"),
    ("arr3_last", "fn main() -> i64 {\n    let a: [i64; 3] = [10, 20, 30];\n    return a[2];\n}\n"),
    ("arr_var_idx", "fn main() -> i64 {\n    let a: [i64; 3] = [10, 20, 30];\n    let i: i64 = 1;\n    return a[i];\n}\n"),
    ("arr4_sum", "fn main() -> i64 {\n    let a: [i64; 4] = [1, 2, 3, 4];\n    return a[0] + a[3];\n}\n"),
]


def read_i64_at(addr: int, off: int = 0) -> int:
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def read_string_handle(handle: int) -> bytes:
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length <= 0:
        return b""
    return ctypes.string_at(addr, length)


def oracle_mic3(mindc, src):
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "arm.mind"
        outp = pathlib.Path(td) / "arm.mic3"
        srcp.write_text(src)
        r = subprocess.run([str(mindc), "--emit-mic3", str(outp), str(srcp)], capture_output=True)
        if r.returncode != 0 or not outp.exists():
            return None
        return outp.read_bytes()


def mind_mic3(fn, src):
    srcb = src.encode()
    srcc = ctypes.create_string_buffer(srcb, len(srcb))
    strbuf = ctypes.create_string_buffer(1 << 16)
    offs = (ctypes.c_int64 * 4096)()
    ccell = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(srcc, ctypes.c_void_p).value, len(srcb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(ccell, ctypes.c_void_p).value,
    )
    return read_string_handle(read_i64_at(es, 0)) if es else b""


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    so = str(resolve_so())
    if not pathlib.Path(so).exists():
        print(f"[BLOCKED] .so not found: {so}")
        return EXIT_BLOCKED
    mindc = pathlib.Path(os.environ.get("MINDC_BIN", str(_DEFAULT_MINDC)))
    if not mindc.exists():
        print(f"[BLOCKED] oracle mindc not found: {mindc}")
        return EXIT_BLOCKED
    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    print("self-host ARRAY smoke — MIND mic@3 vs `mindc --emit-mic3`")
    print(f"  .so     : {so}")
    print(f"  oracle  : {mindc}")
    print(f"  cases   : {len(CASES)}\n")

    failed = 0
    for cid, src in CASES:
        want = oracle_mic3(mindc, src)
        if want is None:
            print(f"  [BLOCKED] {cid:<14} oracle --emit-mic3 failed")
            return EXIT_BLOCKED
        try:
            got = mind_mic3(fn, src)
        except Exception as exc:  # noqa: BLE001
            print(f"  [DRIFT  ] {cid:<14} MIND driver raised {exc!r}")
            failed += 1
            continue
        if not got:
            print(f"  [DRIFT  ] {cid:<14} MIND emitted EMPTY buf (nb_len=0, still fail-closed); oracle_len={len(want)}")
            failed += 1
            continue
        if got == want:
            h = hashlib.sha256(got).hexdigest()[:16]
            print(f"  [OK     ] {cid:<14} {len(got):>4}B  sha256={h}…  nb_len==oracle_len")
        else:
            off = first_diff(got, want)
            print(f"  [DRIFT  ] {cid:<14} PARITY VIOLATION at offset {off}")
            print(f"            mind_len={len(got)} oracle_len={len(want)}")
            print(f"            mind  ={got.hex()}")
            print(f"            oracle={want.hex()}")
            failed += 1

    print()
    if failed:
        print(f"ARRAY SMOKE: {failed}/{len(CASES)} DRIFT")
        return EXIT_DRIFT
    print(f"ALL PASS — {len(CASES)}/{len(CASES)} byte-identical (0 diff)")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
