"""
Self-host regression smoke for FIX #172 — the whole-module mic@3 NOTE sub-emitter
now emits a value-position `if` whose branches are FLOAT literals (a float
value-if), instead of failing closed (0 bytes).

Background
----------
`selftest_native_elf_u(src, len, user_lo)` self-computes the 32-byte PT_NOTE =
mini_sha256(emit_mic3(seeded+pruned IR)) entirely in pure MIND, via
nb_trace_hash -> nb_build_mic3. Before #172 the mic@3 emitter had NO float
support at all (no OP_CONST_F64), so a float value-if fn body poisoned
nb_build_mic3 (fail-closed) -> nb_trace_hash returned 0 -> the whole `_u` entry
returned the empty EmitState (0 bytes). An integer value-if in the same position
already emitted fine, so the gap was specifically the float const leaf.

The fix adds a float-const-leaf descriptor (kind=6) to the mic@3 flatten/count
walkers (flatten_ast, flatten_ast_lv, count_nonparam_nodes, count_nonparam_nodes_lv)
and OP_CONST_F64 emission (emit_mic3_const_f64) in emit_mic3_tree_body. Only DYADIC
float literals are accepted (nb_float_lit_is_dyadic); a non-dyadic / exponent /
overflow literal still fails closed (nb_float_lit_bits is exact only for dyadics —
emitting would be a silent miscompile).

What this gates
---------------
1. BYTE-IDENTITY: nb_build_mic3 (selftest_nb_mic3) == Rust `--emit-mic3` for float
   value-if programs (this is the note's oracle — a single wrong mic@3 byte flips
   the SHA-256 note). This is the load-bearing proof.
2. EXECUTION: the `_u` whole-module native ELF emits (>0 bytes) and RUNS.
3. FAIL-CLOSED: a non-dyadic / exponent float branch still refuses (0 bytes) —
   never a running wrong-value artifact.

Run:  MINDC_SO=<.so> MINDC_BIN=./target/release/mindc \
      python3 examples/mindc_mind/self_host_mic3_float_valueif_smoke.py
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

SO = resolve_so()
MINDC = os.environ.get("MINDC_BIN", str(_HERE.parents[1] / "target" / "release" / "mindc"))

_rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]


def _load():
    lib = ctypes.CDLL(str(SO))
    for nm in ("selftest_nb_mic3", "selftest_native_elf_u"):
        getattr(lib, nm).restype = ctypes.c_int64
        getattr(lib, nm).argtypes = [ctypes.c_int64] * 3
    return lib


def _es_bytes(es):
    sh = _rd(es, 0)
    n = _rd(sh, 8)
    return ctypes.string_at(_rd(sh, 0), n) if n > 0 else b""


def nb_mic3(lib, src: bytes) -> bytes:
    sb = ctypes.create_string_buffer(src, len(src))
    return _es_bytes(lib.selftest_nb_mic3(ctypes.cast(sb, ctypes.c_void_p).value, len(src), 0))


def elf_u(lib, src: bytes) -> bytes:
    sb = ctypes.create_string_buffer(src, len(src))
    return _es_bytes(lib.selftest_native_elf_u(ctypes.cast(sb, ctypes.c_void_p).value, len(src), 0))


def oracle_mic3(src: bytes) -> bytes:
    f = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".mind")
    f.write(src)
    f.close()
    out = f.name + ".mic3"
    p = subprocess.run([MINDC, f.name, "--emit-mic3", out], capture_output=True, text=True)
    data = pathlib.Path(out).read_bytes() if os.path.exists(out) else b""
    for pth in (f.name, out):
        if os.path.exists(pth):
            os.unlink(pth)
    if p.returncode != 0:
        raise RuntimeError(f"oracle --emit-mic3 failed rc={p.returncode}: {p.stderr[-200:]}")
    return data


def run_elf(elf: bytes) -> int:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".elf")
    f.write(elf)
    f.close()
    os.chmod(f.name, os.stat(f.name).st_mode | stat.S_IEXEC)
    rc = subprocess.run([f.name]).returncode
    os.unlink(f.name)
    return rc


# (name, source, expected_exit) — float value-if in LET-INIT position at whole-module
# scope, each with a DYADIC branch pair. main returns the value so we can check it runs.
BYTEID_CASES = [
    # f64 helper: value-if bound then returned; main trivial. Pure value-if, no cast.
    (
        "vif_f64_helper",
        b"fn pick(c: i64) -> f64 {\n"
        b"    let w: f64 = if c > 0 { 1.0 } else { 0.0 };\n"
        b"    return w;\n"
        b"}\n"
        b"fn main() -> i64 {\n"
        b"    return 0;\n"
        b"}\n",
        0,
    ),
    # non-trivial dyadic branch values (2.5 / 4.75) — exercises the 8-byte LE emit.
    (
        "vif_f64_dyadic",
        b"fn pick(c: i64) -> f64 {\n"
        b"    let w: f64 = if c > 0 { 2.5 } else { 4.75 };\n"
        b"    return w;\n"
        b"}\n"
        b"fn main() -> i64 {\n"
        b"    return 0;\n"
        b"}\n",
        0,
    ),
]

# Non-dyadic / exponent float branch MUST fail closed (0 bytes) — nb_float_lit_bits is
# inexact, so emitting would be a silent miscompile.
FAILCLOSED_CASES = [
    (
        "vif_nondyadic",
        b"fn pick(c: i64) -> f64 {\n"
        b"    let w: f64 = if c > 0 { 0.1 } else { 0.0 };\n"
        b"    return w;\n"
        b"}\n"
        b"fn main() -> i64 {\n"
        b"    return 0;\n"
        b"}\n",
    ),
    (
        "vif_exponent",
        b"fn pick(c: i64) -> f64 {\n"
        b"    let w: f64 = if c > 0 { 1.5e3 } else { 0.0 };\n"
        b"    return w;\n"
        b"}\n"
        b"fn main() -> i64 {\n"
        b"    return 0;\n"
        b"}\n",
    ),
]


def main() -> int:
    if not SO or not pathlib.Path(SO).exists():
        print(f"BLOCKED: self-host .so not found ({SO})")
        return 2
    lib = _load()
    rc = 0

    for name, src, want_exit in BYTEID_CASES:
        got = nb_mic3(lib, src)
        want = oracle_mic3(src)
        if len(got) == 0:
            print(f"  FAIL  {name}: nb_build_mic3 fail-closed (0 bytes) — #172 regression")
            rc = 1
            continue
        if got != want:
            print(f"  FAIL  {name}: mic@3 note {len(got)}B != Rust oracle {len(want)}B")
            for i, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    print(f"        first diverge @ {i}: nb={a:#04x} oracle={b:#04x}")
                    break
            rc = 1
            continue
        elf = elf_u(lib, src)
        if len(elf) == 0:
            print(f"  FAIL  {name}: mic@3 byte-identical but _u ELF fail-closed")
            rc = 1
            continue
        ec = run_elf(elf)
        if ec != want_exit:
            print(f"  FAIL  {name}: _u ELF ran exit {ec} != {want_exit}")
            rc = 1
            continue
        print(f"  PASS  {name}: mic@3 note == Rust --emit-mic3 ({len(got)}B) AND _u ELF runs exit {ec}")

    for name, src in FAILCLOSED_CASES:
        got = nb_mic3(lib, src)
        if len(got) != 0:
            print(f"  FAIL  {name}: non-dyadic float NOT fail-closed ({len(got)} bytes) — silent-miscompile risk")
            rc = 1
            continue
        print(f"  PASS  {name}: non-dyadic/exponent float branch fail-closed (0 bytes) — no silent miscompile")

    print("ALL PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
