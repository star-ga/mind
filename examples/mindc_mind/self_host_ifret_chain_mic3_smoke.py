"""
Self-host mic@3 note-emit gate for the return-terminated if / else-if / N-arm /
match STATEMENT chain (task #274, #40/#111).

Before this port, the pure-MIND mic@3 emitter (`emit_mic3_module_fndef`) took only
the 2-arm `if C { return A } else { return B }` shape; a THIRD arm (`else if …`, or
the equivalent N-arm `match`) fell through and FAILED CLOSED (nb_len == 0). The Rust
`--emit-mic3` oracle emits the else-region as a nested OP_IF (0x1C), so the pure-MIND
emitter now recurses into the else-region for a nested if-node
(`emit_mic3_ifret_chain_instr`), threading the SSA-vid base through each level to
match the oracle byte-for-byte.

`else if` parses to `else { if … }` (parse_if_else_tail) and a statement-form
`match x { P => { return V; } … _ => { … } }` desugars to the SAME else-if chain
(parse_match), so ONE recursive emit path closes all three constructs.

Two gates per fixture:
  (1) EXECUTION-CORRECTNESS — compile with the live Rust `mindc build --emit=binary`
      (CPU oracle) and RUN it; the process must exit with the arm the scrutinee
      selects, every arm incl. the wildcard, for 2/3/4-arm, else-if AND match.
  (2) mic@3 NOTE BYTE-IDENTITY — the pure-MIND `selftest_mic3_module_nfn(<fn src>)`
      must equal `mindc --emit-mic3 <fn src>` byte-for-byte (the actual gap closure;
      nb_len must equal oracle_len and every byte must match).

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_ifret_chain_mic3_smoke.py
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from _selfhost_so import resolve_so  # noqa: E402

SO = resolve_so()
MINDC = pathlib.Path(
    os.environ.get("MINDC_BIN", str(_HERE.parents[1] / "target" / "release" / "mindc"))
)

_P = ctypes.POINTER(ctypes.c_int64)


def _i64(addr, off=0):
    return int(ctypes.cast(addr + off, _P)[0])


def _read_string_record(handle):
    if not handle:
        return b""
    addr, length = _i64(handle, 0), _i64(handle, 8)
    if not addr or not length:
        return b""
    p = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int8))
    return bytes(int(p[i]) & 0xFF for i in range(length))


def oracle_mic3(src):
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "m.mind"
        op = pathlib.Path(td) / "m.mic3"
        sp.write_text(src)
        subprocess.run([str(MINDC), "--emit-mic3", str(op), str(sp)], capture_output=True)
        return op.read_bytes() if op.exists() else None


def nfn_mic3(src):
    lib = ctypes.CDLL(str(SO))
    fn = lib.selftest_mic3_module_nfn
    fn.restype = ctypes.c_void_p
    fn.argtypes = [ctypes.c_int64] * 5
    sb = src.encode()
    sc = ctypes.create_string_buffer(sb, len(sb))
    strbuf = ctypes.create_string_buffer(1 << 18)
    offs = (ctypes.c_int64 * 8192)()
    cc = (ctypes.c_int64 * 1)()
    es = fn(
        ctypes.cast(sc, ctypes.c_void_p).value,
        len(sb),
        ctypes.cast(strbuf, ctypes.c_void_p).value,
        ctypes.cast(offs, ctypes.c_void_p).value,
        ctypes.cast(cc, ctypes.c_void_p).value,
    )
    return _read_string_record(_i64(es, 0)) if es else b""


def run_binary(src):
    """Compile with live mindc build --emit=binary and return the process exit code."""
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "p.mind"
        op = pathlib.Path(td) / "p.bin"
        sp.write_text(src)
        r = subprocess.run(
            [str(MINDC), "build", str(sp), "--release", "--emit=binary", f"--out={op}"],
            capture_output=True,
        )
        if r.returncode != 0 or not op.exists():
            return None
        os.chmod(op, 0o755)
        return subprocess.run([str(op)], capture_output=True).returncode


# ── chain builders (statement form; each arm returns its int) ──────────────────
def elseif_stmt(arms, wild):
    head = arms[0]
    s = f"    if x == {head[0]} {{ return {head[1]}; }}"
    for p, v in arms[1:]:
        s += f" else if x == {p} {{ return {v}; }}"
    s += f" else {{ return {wild}; }}"
    return f"fn classify(x: i64) -> i64 {{\n{s}\n}}\n"


def match_stmt(arms, wild):
    body = " ".join(f"{p} => {{ return {v}; }}" for p, v in arms) + f" _ => {{ return {wild}; }}"
    return f"fn classify(x: i64) -> i64 {{\n    match x {{ {body} }}\n}}\n"


def match_expr(arms, wild):
    body = ", ".join(f"{p} => {v}" for p, v in arms) + f", _ => {wild}"
    return (
        f"fn classify(x: i64) -> i64 {{\n"
        f"    let r: i64 = match x {{ {body} }};\n    return r;\n}}\n"
    )


def _prog(fn_src, x):
    return fn_src + f"fn main() -> i64 {{ return classify({x}); }}\n"


ARMS = {
    2: [(1, 10)],
    3: [(1, 10), (2, 20)],
    4: [(1, 10), (2, 20), (3, 30)],
}
WILD = {2: 30, 3: 30, 4: 40}


def main():
    if not SO.exists():
        print(f"BLOCKED: {SO} not found")
        return 1
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 1

    fails = 0
    for form, build in (("elseif", elseif_stmt), ("match_stmt", match_stmt), ("match_expr", match_expr)):
        for n in (2, 3, 4):
            arms, wild = ARMS[n], WILD[n]
            fn_src = build(arms, wild)
            # mic@3 note byte-identity (single-fn source)
            o = oracle_mic3(fn_src)
            m = nfn_mic3(fn_src)
            ol = len(o) if o else -1
            ml = len(m) if m else 0
            byte_ok = (m == o and o is not None)
            # execution-correctness: probe every arm incl. wildcard
            exec_ok = True
            probes = [(p, v) for p, v in arms] + [(999, wild)]
            got = []
            for x, want in probes:
                rc = run_binary(_prog(fn_src, x))
                got.append((x, rc, want))
                if rc != (want & 0xFF):
                    exec_ok = False
            ok = byte_ok and exec_ok
            status = "PASS" if ok else "FAIL"
            print(
                f"  {status}  {form}-{n}arm  mic3 nb_len={ml} oracle_len={ol} byte_id={byte_ok}"
                f"  exec={['x%d->rc%s(want%d)' % g for g in got]}"
            )
            if not ok:
                fails += 1
                if not byte_ok and o and m:
                    k = min(len(o), len(m))
                    di = next((i for i in range(k) if o[i] != m[i]), k)
                    lo = max(0, di - 6)
                    print(f"       first diff @ {di}  nb={list(m[lo:di+10])}  oracle={list(o[lo:di+10])}")

    if fails:
        print(f"FAIL: {fails} chain fixture(s) diverged")
        return 1
    print("ALL PASS  (else-if / N-arm / match statement+expr chains: mic@3 note byte-identical to --emit-mic3 AND run-correct via mindc build --emit=binary)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
