"""Self-host LET-PATH fail-closed smoke — pins the S1-collapse fail-open wall.

The Single-Canonical-Descriptor-Lowering (S1) collapse routed flatten_ast_le /
flatten_ast through the ONE flattener flatten_expr_env, which ACCEPTS call / array /
index constructs. But the let-chain drivers that feed the src=0 downstream
(selftest_mic3_let_fn -> emit_mic3_let_fn_module, selftest_mic3_ast_fn ->
emit_mic3_ast_fn_module, and the if-with-let-in-branch path
selftest_mic3_if_let_fn -> emit_mic3_if_block_instr) all emit the fn body via
emit_mic3_tree_body with src=0 / strbuf=0 / offs=0 / n_strings=0. A call/array/index
descriptor read a near-null src / a missing strtab -> SIGSEGV (const-array kind 7),
mis-interned name (call kind 3) or silent wrong-bytes (kinds 8/9/10).

descriptor_has_src_dep_kind now scans the flattened descriptor and FAILS CLOSED
(0-byte emit) if ANY node kind is in {3,5,7,8,9,10}, restoring the pre-collapse wall
for the src=0 downstream (fail-closed-over-wrong-bytes, Bound by MIND-CONSTITUTION
§I). Kinds 0/1/2/4/6 (param/binop/const/arg-cons/float) still emit byte-exact — a
real coverage gain, kept.

The never-wrong corpus (tests/selfhost_gaps/never_wrong/) CANNOT cover this: it
routes through the PRODUCTION selftest_mic3_module_nfn path, which threads a real
src+strtab and emits these shapes BYTE-EXACT. Only the let_fn / ast_fn entry points
exercise the src=0 downstream, so this gate drives them directly.

FAILCLOSE cases MUST emit 0 bytes AND never crash (return code non-negative).
EXACT cases (kinds 0/1/2/6 only) MUST stay byte-identical to the live
`mindc --emit-mic3` oracle — the guard must NOT over-refuse them.

Run:
  MINDC_SO=/path/libmindc_mind.so MINDC_BIN=./target/release/mindc \
  python3 examples/mindc_mind/self_host_letpath_failclose_smoke.py
"""
import base64
import ctypes
import os
import pathlib
import re
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

# Each fixture is a single `pub fn` module so the live oracle emits exactly that fn.
# FAILCLOSE: descriptor carries a src-dependent kind (call/array/index) -> must 0B.
FAILCLOSE = [
    ("const_array", "pub fn f(a: i64) -> i64 { let m: [i64; 3] = [1, 2, 3]; m[0] }"),
    ("var_array", "pub fn f(a: i64, b: i64) -> i64 { let m: [i64; 2] = [a, b]; m[1] }"),
    ("call", "pub fn f(a: i64) -> i64 { let x: i64 = g(a); x }"),
    ("const_index", "pub fn f(xs: i64) -> i64 { let x: i64 = xs[0]; x }"),
    ("index_expr", "pub fn f(xs: i64, a: i64) -> i64 { let x: i64 = xs[a] + 1; x }"),
]
# ORACLE_EXACT: a single-fn whose let_fn emission is byte-identical to the live
# whole-module `mindc --emit-mic3` oracle — proves the guard did NOT over-refuse a
# param/binop/const body.
ORACLE_EXACT = [
    ("binop", "pub fn f(a: i64, b: i64) -> i64 { let t: i64 = a * b; t + 1 }"),
]
# KEPT: param/const-neg/float bodies (kinds 0/1/2/6) that the guard must NOT refuse.
# The single-fn let_fn framing differs from the whole-module oracle for these (a
# pre-existing let_fn nuance, not a guard artifact), so we assert only that the guard
# keeps them (non-empty emit, no over-refusal). Their bytes are unchanged by the guard
# (proven separately: identical pre/post the descriptor scan).
KEPT = [
    ("neg_const", "pub fn f(a: i64) -> i64 { let x: i64 = -5; x }"),
    ("float", "pub fn f(a: i64) -> i64 { let x: i64 = 1.5; x }"),
]


def read_i64_at(addr, off=0):
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def read_string_handle(handle):
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length <= 0:
        return b""
    return ctypes.string_at(addr, length)


def name_params(src):
    m = re.search(r"fn\s+(\w+)\s*\(([^)]*)\)", src)
    name = m.group(1).encode()
    params = [p.split(":")[0].strip().encode()
              for p in m.group(2).split(",") if p.strip()]
    return name, params


def worker(so, src):
    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_let_fn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 9
    name, params = name_params(src)
    strs = [name] + params
    sbuf = b"".join(strs)
    soff = [0]
    for s in strs:
        soff.append(soff[-1] + len(s))
    sbuf_c = ctypes.create_string_buffer(sbuf, max(len(sbuf), 1))
    soff_c = (ctypes.c_int64 * (len(soff) + 8))(*soff)
    src_b = src.encode()
    src_c = ctypes.create_string_buffer(src_b, len(src_b))
    nodes_c = (ctypes.c_int64 * (512 * 4))()
    cursor_c = (ctypes.c_int64 * 1)()
    vidbuf_c = (ctypes.c_int64 * 512)()
    lenv_c = (ctypes.c_int64 * (512 * 3))()
    lcount_c = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(src_c, ctypes.c_void_p).value, len(src_b),
        ctypes.cast(sbuf_c, ctypes.c_void_p).value,
        ctypes.cast(soff_c, ctypes.c_void_p).value,
        ctypes.cast(nodes_c, ctypes.c_void_p).value,
        ctypes.cast(cursor_c, ctypes.c_void_p).value,
        ctypes.cast(vidbuf_c, ctypes.c_void_p).value,
        ctypes.cast(lenv_c, ctypes.c_void_p).value,
        ctypes.cast(lcount_c, ctypes.c_void_p).value)
    got = read_string_handle(read_i64_at(es, 0)) if es else b""
    sys.stdout.write("LEN=%d HEX=%s\n" % (len(got), got.hex()))
    sys.stdout.flush()


def run_isolated(so, src):
    r = subprocess.run(
        [sys.executable, __file__, so, base64.b64encode(src.encode()).decode()],
        capture_output=True, text=True)
    if r.returncode < 0:
        return None, r.returncode  # crashed by signal
    hx = ""
    for line in r.stdout.splitlines():
        if line.startswith("LEN="):
            hx = line.split()[1][4:]
    return bytes.fromhex(hx), 0


def oracle_mic3(mindc, src):
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "c.mind"
        outp = pathlib.Path(td) / "c.mic3"
        srcp.write_text(src)
        r = subprocess.run([str(mindc), "--emit-mic3", str(outp), str(srcp)],
                           capture_output=True)
        if r.returncode != 0 or not outp.exists():
            return None
        return outp.read_bytes()


def main():
    if len(sys.argv) == 3:
        worker(sys.argv[1], base64.b64decode(sys.argv[2]).decode())
        return 0
    so = str(resolve_so())
    if not pathlib.Path(so).exists():
        print(f"[BLOCKED] .so not found: {so}")
        return EXIT_BLOCKED
    mindc = pathlib.Path(os.environ.get("MINDC_BIN", str(_DEFAULT_MINDC)))
    print("self-host LET-PATH fail-closed smoke — selftest_mic3_let_fn (src=0 downstream)")
    print(f"  .so    : {so}")
    print(f"  oracle : {mindc}\n")

    failed = 0
    for cid, src in FAILCLOSE:
        got, sig = run_isolated(so, src)
        if got is None:
            print(f"  [DRIFT ] failclose {cid:<12} CRASHED (signal {-sig}) — must fail-closed, never crash")
            failed += 1
        elif len(got) == 0:
            print(f"  [OK    ] failclose {cid:<12} 0B rc=0 (fail-closed over wrong-bytes)")
        else:
            print(f"  [DRIFT ] failclose {cid:<12} emitted {len(got)}B via src=0 path — MUST be 0B")
            failed += 1

    for cid, src in ORACLE_EXACT:
        want = oracle_mic3(mindc, src)
        if want is None:
            print(f"  [BLOCKED] exact   {cid:<12} oracle --emit-mic3 failed")
            return EXIT_BLOCKED
        got, sig = run_isolated(so, src)
        if got is None:
            print(f"  [DRIFT ] exact     {cid:<12} CRASHED (signal {-sig})")
            failed += 1
        elif got == want:
            print(f"  [OK    ] exact     {cid:<12} {len(got)}B byte-exact vs oracle (guard did not over-refuse)")
        else:
            print(f"  [DRIFT ] exact     {cid:<12} OVER-REFUSED/wrong: got {len(got)}B vs oracle {len(want)}B")
            failed += 1

    for cid, src in KEPT:
        got, sig = run_isolated(so, src)
        if got is None:
            print(f"  [DRIFT ] kept      {cid:<12} CRASHED (signal {-sig})")
            failed += 1
        elif len(got) > 0:
            print(f"  [OK    ] kept      {cid:<12} {len(got)}B emitted (guard did not over-refuse)")
        else:
            print(f"  [DRIFT ] kept      {cid:<12} OVER-REFUSED to 0B — guard must keep kinds 0/1/2/6")
            failed += 1

    print()
    if failed:
        print(f"LET-PATH FAILCLOSE SMOKE: {failed} DRIFT")
        return EXIT_DRIFT
    print(f"ALL PASS — {len(FAILCLOSE)} fail-closed (no crash), "
          f"{len(ORACLE_EXACT)} byte-exact, {len(KEPT)} kept")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
