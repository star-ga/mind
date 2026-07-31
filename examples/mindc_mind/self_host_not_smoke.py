"""
Self-host unary logical-NOT smoke — the pure-MIND mic@3 note path emits `!x`
BYTE-IDENTICAL to the Rust oracle (`mindc --emit-mic3`).

Covers the note-builder-coverage gap #40/#111: `!x` FAIL-CLOSED (nb_len=0) in the
`_u` note path (flatten_expr_env / count_nonparam_nodes_lv lacked the ast_not arm)
even though `!x` RUNS correct natively. `!x` desugars to `x == 0` — exactly how
src/eval/lower.rs Node::Not lowers it — so byte-identity vs the live oracle IS the
correctness proof (the Eq lowering is already keystone-stable).

Fail-closed, no false green:
  * oracle mindc missing/errors -> BLOCKED (exit 2)
  * MIND emits EMPTY buf (still fail-closed) -> DRIFT (exit 1)
  * MIND bytes != oracle bytes -> DRIFT (exit 1)

Run:
  MINDC_SO=/path/libmindc_mind.so MINDC_BIN=./target/release/mindc \
  python3 examples/mindc_mind/self_host_not_smoke.py
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
    ("not_letbool", "fn main() -> i64 {\n    let b: bool = false;\n    if !b { return 7; }\n    return 0;\n}\n"),
    ("not_param", "fn f(b: bool) -> i64 {\n    if !b { return 1; }\n    return 0;\n}\n"),
    ("not_cmp", "fn main() -> i64 {\n    let x: i64 = 3;\n    if !(x == 4) { return 1; }\n    return 0;\n}\n"),
    ("not_in_let", "fn main() -> i64 {\n    let b: bool = false;\n    let c: bool = !b;\n    if c { return 5; }\n    return 0;\n}\n"),
    ("not_intzero", "fn main() -> i64 {\n    let x: i64 = 0;\n    if !x { return 3; }\n    return 0;\n}\n"),
    ("not_ret_expr", "fn f(b: bool) -> bool {\n    return !b;\n}\nfn main() -> i64 {\n    if f(false) { return 4; }\n    return 0;\n}\n"),
    ("double_not", "fn main() -> i64 {\n    let b: bool = true;\n    if !!b { return 6; }\n    return 0;\n}\n"),
    ("not_lit_true", "fn main() -> i64 {\n    if !true { return 1; }\n    return 8;\n}\n"),
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
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
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

    print("self-host NOT smoke — MIND mic@3 vs `mindc --emit-mic3`")
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
            failed += 1

    print()
    if failed:
        print(f"[DRIFT] {failed}/{len(CASES)} case(s) not byte-identical to the oracle")
        return EXIT_DRIFT
    print(f"[PASS] all {len(CASES)} `!x` cases byte-identical to `mindc --emit-mic3`")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
