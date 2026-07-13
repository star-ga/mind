"""
Oracle-parity linter — the self-host CORRECTNESS gate.

For EVERY self-host arm of the pure-MIND bootstrap compiler, assert that the
mic@3 bytes emitted by MIND are BYTE-IDENTICAL to the bytes the Rust front-end
emits for the same source (`mindc --emit-mic3`). Any drift fails LOUD with the
SHA-256 of each side and the exact first-diff offset — so a divergence between
the MIND emitter and its Rust oracle can never land silently.

Relationship to the other self-host gates (NOT a duplicate of any):

  * cross_substrate_identity (Rust test)  — proves one Q16.16 workload hashes the
    same on avx2 vs neon. That is SUBSTRATE identity, a different axis entirely.
  * mic3_flip_smoke.py                     — proves the WHOLE main.mind module is
    byte-identical vs the oracle. When it goes red it reports one offset on an
    ~N-KB stream; it does NOT tell you WHICH construct regressed.
  * THIS linter                            — one representative module PER ARM
    (the shape-cases of `emit_mic3_module_fndef`: const / arith / call-2-lit /
    expr-tree / if-expr / if-both-return / let-chain / if-stmt-return /
    nested-if-return / general-seq / bitwise / call-with-expr-args / match-desugar).
    On drift it LOCALISES the failing arm and prints hash evidence, so a
    regression is diagnosed at the construct level, not the byte level.

Each arm is a SELF-CONTAINED module (no undefined external callee) so the Rust
type-checker accepts it standalone and the oracle regenerates it live from the
exact same source the MIND driver consumes — a self-golden that is never
cross-checked against real mindc is not acceptable (Bound by MIND-CONSTITUTION
Article I).

The MIND side is driven through `selftest_mic3_module_nfn` — the general N-fn
module emitter that loops every top-level fn and fails CLOSED (empty buf) on any
unmatched body shape. It does NOT touch the canary (`mindc_compile` ->
`emit_fn_def` stays stubbed), so the mic@1 fixed point is unaffected.

Determinism: fixed ordered corpus, no wall-clock, no RNG. Same inputs -> same
verdict, every run.

Fail-closed contract (no false green):
  * MINDC_SO set but the .so is missing        -> hard fail (exit 2)
  * the oracle mindc binary is missing/errors   -> hard fail (exit 2)
  * any arm emits an EMPTY buf                   -> hard fail (exit 1)
  * any arm's MIND bytes != oracle bytes         -> hard fail (exit 1)
Only a purely-local run with NEITHER MINDC_SO nor the default .so present may
SKIP (exit 0) — CI always sets MINDC_SO, so CI can never skip.

Run:
  MINDC_SO=/path/to/libmindc_mind.so \
  MINDC=./target/release/mindc \
  python3 examples/mindc_mind/oracle_parity_lint.py
"""

import ctypes
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"  # legacy in-tree path (fallback only)
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402
_DEFAULT_MINDC = _HERE.parents[1] / "target" / "release" / "mindc"
_Int64Ptr = ctypes.POINTER(ctypes.c_int64)

# Exit codes: 0 = PASS (or legitimate local skip), 1 = parity drift / empty buf,
# 2 = BLOCKED (fail-closed: toolchain/artifact absent under enforcement).
EXIT_PASS = 0
EXIT_DRIFT = 1
EXIT_BLOCKED = 2


def read_i64_at(addr: int, off: int = 0) -> int:
    return int(ctypes.cast(addr + off, _Int64Ptr)[0])


def read_string_handle(handle: int) -> bytes:
    """MIND String record laid out as [+0] addr, [+8] len, [+16] cap."""
    if handle == 0:
        return b""
    addr = read_i64_at(handle, 0)
    length = read_i64_at(handle, 8)
    if addr == 0 or length == 0:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


# Each entry is (arm_id, human_label, source). arm_id maps to the shape-case in
# emit_mic3_module_fndef so a red arm names the construct that regressed. Every
# arm is a supported (byte-exact) shape — the linter asserts equality, not
# fail-closed, for all of them.
ARMS = [
    ("a  const-return",
     "zero-arg const return (tk_* family)",
     "pub fn tk_eof() -> i64 { 0 }"),
    ("b  2param-arith",
     "two-param direct binop",
     "pub fn add(a: i64, b: i64) -> i64 { a + b }"),
    ("b  let-arith",
     "single-let arith (let t = pa OP pb; t)",
     "pub fn f(a: i64, b: i64) -> i64 { let t: i64 = a + b; t }"),
    ("c  call-2-lit",
     "zero-arg call with two int-literal args",
     "fn g(a: i64, b: i64) -> i64 { a + b }\n"
     "pub fn f() -> i64 { g(2, 3) }"),
    ("d  expr-tree",
     "arbitrary param/const/binop expression tree",
     "pub fn f(a: i64, b: i64, c: i64, d: i64) -> i64 { a * b + c * d }"),
    ("d  expr-tree-const-left",
     "expr tree with a const LEFT operand",
     "pub fn m(a: i64) -> i64 { 2 * a + 1 }"),
    ("e  if-expr-value",
     "single if-expression as a value",
     "pub fn s(b: i64) -> i64 { if b == 1 { 1 } else { 0 } }"),
    ("e  if-expr-binop-branch",
     "if-expression with a BINOP branch",
     "pub fn s(a: i64, b: i64) -> i64 { if a { b + a } else { a } }"),
    ("e2 if-both-return",
     "both branches diverge (if C { return A } else { return B })",
     "pub fn s(b: i64) -> i64 { if b == 1 { return 1; } else { return 0; } }"),
    ("f  let-chain",
     "multi-statement let chain",
     "pub fn f(a: i64) -> i64 { let x: i64 = a + 1; let y: i64 = x + 2; y }"),
    ("f  let-chain-3",
     "3-let chain with a final ref",
     "pub fn h(x: i64) -> i64 { let a: i64 = x + 1; let b: i64 = a + 2; "
     "let c: i64 = b + 3; c }"),
    ("g  if-stmt-return",
     "if-statement early return (else-less `if C { return E; } TRAIL`)",
     "pub fn s(b: i64) -> i64 { if b == 1 { return 1; } 0 }"),
    ("g2 nested-if-return",
     "nested if-statement early return then trailing value",
     "pub fn s(a: i64, b: i64) -> i64 { if a == 1 { if b == 1 { return 2; } } 0 }"),
    ("h  general-seq",
     "general statement sequence (let + if-return + let + trailing)",
     "pub fn foo(a: i64, b: i64) -> i64 { let t: i64 = a + b; "
     "if t == 0 { return 0; } let u: i64 = t * 2; u }"),
    ("d  bitand-mask",
     "& mask operator",
     "pub fn f(x: i64) -> i64 { x & 255 }"),
    ("d  bitor",
     "| bitor operator",
     "pub fn f(a: i64, b: i64) -> i64 { a | b }"),
    ("d  bitxor",
     "^ bitxor operator",
     "pub fn f(a: i64, b: i64) -> i64 { a ^ b }"),
    ("d  shl",
     "<< left shift",
     "pub fn f(a: i64, b: i64) -> i64 { a << b }"),
    ("d  shr",
     ">> right shift",
     "pub fn f(a: i64, b: i64) -> i64 { a >> b }"),
    ("d  bit-precedence",
     "bitwise precedence mix (a & b | c)",
     "pub fn f(a: i64, b: i64, c: i64) -> i64 { a & b | c }"),
    ("c  call-expr-args",
     "call with expression args",
     "fn add(a: i64, b: i64) -> i64 { a + b }\n"
     "pub fn f(x: i64, y: i64) -> i64 { add(x + 1, y * 2) }"),
    ("c  call-in-let-chain",
     "let-ref flowing into a call arg",
     "pub fn f(toks: i64, k: i64) -> i64 { let a: i64 = g(toks); "
     "let b: i64 = h(a, k); b }\n"
     "fn g(x: i64) -> i64 { x }\n"
     "fn h(x: i64, y: i64) -> i64 { x + y }"),
    ("e  match-desugar",
     "2-arm match desugars to if (zero new emit)",
     "pub fn pick(c: i64) -> i64 { match c { 0 => 10, _ => 20 } }"),
]


def oracle_mic3(mindc: pathlib.Path, src: str):
    """Regenerate the Rust-oracle mic@3 bytes for `src`, live. None on failure."""
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "arm.mind"
        outp = pathlib.Path(td) / "arm.mic3"
        srcp.write_text(src)
        r = subprocess.run(
            [str(mindc), "--emit-mic3", str(outp), str(srcp)],
            capture_output=True,
        )
        if r.returncode != 0 or not outp.exists():
            return None
        return outp.read_bytes()


def mind_mic3(fn, src: str) -> bytes:
    """Emit the mic@3 bytes MIND produces for `src` via selftest_mic3_module_nfn."""
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


def first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), b and len(b) or 0)
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main() -> int:
    so = str(resolve_so())  # MINDC_SO verbatim, else fresh build (never stale)
    so_forced = "MINDC_SO" in os.environ
    if not pathlib.Path(so).exists():
        if so_forced:
            print(f"BLOCKED: MINDC_SO set but .so missing: {so} (refusing to skip)")
            return EXIT_BLOCKED
        print(f"SKIP: {so} not found (opt-in local build artifact; CI sets MINDC_SO)")
        return EXIT_PASS

    mindc = pathlib.Path(os.environ.get("MINDC", str(_DEFAULT_MINDC)))
    if not mindc.exists():
        print(f"BLOCKED: oracle mindc not found at {mindc} (refusing to skip)")
        return EXIT_BLOCKED

    lib = ctypes.CDLL(so)
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5

    print("oracle-parity linter — per-arm MIND mic@3 vs `mindc --emit-mic3`")
    print(f"  .so     : {so}")
    print(f"  oracle  : {mindc} --emit-mic3")
    print(f"  arms    : {len(ARMS)}\n")

    drifted = 0
    blocked = 0
    for arm_id, label, src in ARMS:
        oracle = oracle_mic3(mindc, src)
        if oracle is None:
            print(f"  [BLOCKED] {arm_id:<22} oracle mindc --emit-mic3 failed for: {label}")
            blocked += 1
            continue
        try:
            got = mind_mic3(fn, src)
        except Exception as exc:  # noqa: BLE001 — surface any FFI/driver failure
            print(f"  [BLOCKED] {arm_id:<22} MIND driver raised {exc!r} for: {label}")
            blocked += 1
            continue

        if not got:
            # A supported arm must emit real bytes; an empty buf is fail-closed
            # in the driver but a REGRESSION here (this arm used to be byte-exact).
            print(f"  [DRIFT  ] {arm_id:<22} MIND emitted EMPTY buf (arm regressed): {label}")
            print(f"             oracle: {len(oracle)}B  sha256={hashlib.sha256(oracle).hexdigest()}")
            drifted += 1
            continue

        if got == oracle:
            h = hashlib.sha256(got).hexdigest()
            print(f"  [OK     ] {arm_id:<22} {len(got):>4}B  sha256={h[:16]}…  {label}")
            continue

        # Drift — fail loud with full hash evidence + first-diff window.
        drifted += 1
        di = first_diff(got, oracle)
        lo = max(0, di - 8)
        print(f"  [DRIFT  ] {arm_id:<22} PARITY VIOLATION: {label}")
        print(f"             MIND  : {len(got):>4}B  sha256={hashlib.sha256(got).hexdigest()}")
        print(f"             oracle: {len(oracle):>4}B  sha256={hashlib.sha256(oracle).hexdigest()}")
        print(f"             first diff @ byte {di}")
        print(f"             MIND  [{lo}:{di + 8}] = {list(got[lo:di + 8])}")
        print(f"             oracle[{lo}:{di + 8}] = {list(oracle[lo:di + 8])}")

    ok = len(ARMS) - drifted - blocked
    print(f"\n  parity {ok}/{len(ARMS)} arms byte-identical vs oracle"
          f"  |  drift {drifted}  |  blocked {blocked}")

    if blocked:
        # Any BLOCKED arm means the oracle or driver could not run — never a
        # green. Fail-closed regardless of drift count.
        print(f"\nBLOCKED: {blocked} arm(s) could not be oracle-checked (fail-closed)")
        return EXIT_BLOCKED
    if drifted:
        print(f"\nFAIL: {drifted} arm(s) drifted from the Rust oracle")
        return EXIT_DRIFT
    print("\nPASS: every self-host arm is byte-identical to its Rust oracle")
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
