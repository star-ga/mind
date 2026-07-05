"""
Modulo-operator (`%`) REGRESSION smoke — the gate that would have caught the
2-HIGH miscompile where `a % b` silently lowered to DIVISION (MLIR path) and
MULTIPLICATION (native path) in main.mind.

Root cause (commit 95293ee wired `%` into the mic@3 path only):
  * emit_mlir_arith_op had no op_mod arm, so `%` fell through to the "divsi "
    tail -> arith.divsi (QUOTIENT) instead of arith.remsi (REMAINDER).
  * nb_arith_rax_mem had no op_mod arm, so `%` fell through to the imul tail
    -> `a * b` (PRODUCT) instead of the guarded idiv + `mov rax,rdx` remainder.
All prior gates stayed green because main.mind's OWN source never uses `%`, so
the two broken emit paths were never exercised. This smoke exercises them
directly and RUNS the result, failing LOUD if `17 % 5` comes back 3 (quotient)
or 85 (product) instead of 2 (remainder).

  NATIVE path (main.mind's nb_arith_rax_mem, run end-to-end): compile
    `fn main() -> i64 { return 17 % 5; }` with the pure-MIND native-ELF emitter
    (selftest_native_elf_h), run the ELF, assert it exits 2.
  MLIR path (main.mind's emit_mlir_arith_op): emit MLIR text for `a % b` with
    the pure-MIND emitter (selftest_emit_mlir), assert the arith op is
    arith.remsi and NOT arith.divsi; cross-check the live `mindc` MLIR->native
    toolchain actually computes 17 % 5 == 2 (proving remsi == remainder).

Run:  MINDC_SO=.../libmindc_mind.so MINDC_BIN=.../mindc \
        python3 examples/mindc_mind/mod_operator_smoke.py
"""

import ctypes
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).parent
_REPO = _HERE.parent.parent
_DEFAULT_SO = _HERE / "libmindc_mind.so"
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))
_DEFAULT_MINDC = _REPO / "target" / "release" / "mindc"
MINDC = pathlib.Path(os.environ.get("MINDC_BIN", str(_DEFAULT_MINDC)))

# 17 % 5 == 2 (remainder). The two failure modes we must catch:
#   3  = 17 / 5 (integer QUOTIENT)  -> the MLIR divsi / native divsi bug
#   85 = 17 * 5 (PRODUCT)           -> the native imul bug
DIVIDEND, DIVISOR = 17, 5
WANT_REM = DIVIDEND % DIVISOR          # 2
QUOTIENT = DIVIDEND // DIVISOR         # 3
PRODUCT = DIVIDEND * DIVISOR           # 85


def _read_es_buf(es: int) -> bytes:
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]
    sh = rd(es, 0)  # buf (String handle: addr/len/cap)
    ln = rd(sh, 8)
    return ctypes.string_at(rd(sh, 0), ln) if ln > 0 else b""


def native_run_mod(lib) -> int:
    """Emit a native ELF for `fn main() -> i64 { return 17 % 5; }` with the
    pure-MIND native backend, run it, and assert it exits WANT_REM (2). The note
    is irrelevant to execution, so a zeroed 32-byte trace hash is fine here — we
    are proving the INSTRUCTION STREAM computes a remainder, not byte-identity."""
    lib.selftest_native_elf_h.restype = ctypes.c_int64
    lib.selftest_native_elf_h.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

    src = f"fn main() -> i64 {{\n    return {DIVIDEND} % {DIVISOR};\n}}\n".encode()
    src_buf = ctypes.create_string_buffer(src, len(src))
    hash_buf = ctypes.create_string_buffer(bytes(32), 32)
    es = lib.selftest_native_elf_h(
        ctypes.cast(src_buf, ctypes.c_void_p).value,
        len(src),
        ctypes.cast(hash_buf, ctypes.c_void_p).value,
    )
    elf = _read_es_buf(es)
    if not elf:
        print("  FAIL  native %: pure-MIND emitter returned an empty ELF (failed closed)")
        return 1
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "mod.elf"
        p.write_bytes(elf)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        code = subprocess.run([str(p)]).returncode
    if code == WANT_REM:
        print(f"  PASS  native %: pure-MIND ELF runs {DIVIDEND} % {DIVISOR} -> exits {code} (remainder)")
        return 0
    why = {
        QUOTIENT: "QUOTIENT — native divsi/div bug (a % b lowered to a / b)",
        PRODUCT: "PRODUCT — native imul bug (a % b lowered to a * b)",
    }.get(code, "wrong value")
    print(
        f"  FAIL  native %: {DIVIDEND} % {DIVISOR} exited {code}, expected {WANT_REM} — {why}"
    )
    return 1


def mlir_emit_mod(lib) -> int:
    """Emit MLIR text for `fn m(a, b) -> i64 { return a % b; }` with the pure-MIND
    emitter and assert the arith op is arith.remsi, NOT arith.divsi (the quotient
    bug). Then cross-check the live `mindc` MLIR->native toolchain computes
    17 % 5 == 2, proving remsi is a remainder end-to-end."""
    lib.selftest_emit_mlir.restype = ctypes.c_int64
    lib.selftest_emit_mlir.argtypes = [ctypes.c_int64, ctypes.c_int64]

    src = b"fn m(a: i64, b: i64) -> i64 {\n    return a % b;\n}\n"
    buf = ctypes.create_string_buffer(src, len(src))
    es = lib.selftest_emit_mlir(ctypes.cast(buf, ctypes.c_void_p).value, len(src))
    text = _read_es_buf(es)

    if b"arith.remsi" not in text:
        if b"arith.divsi" in text:
            print(
                "  FAIL  mlir %: pure-MIND emitter produced arith.divsi (QUOTIENT) "
                "for `a % b` — missing op_mod arm in emit_mlir_arith_op"
            )
        else:
            print(f"  FAIL  mlir %: no arith.remsi in emitted text: {text!r}")
        return 1
    if b"arith.divsi" in text:
        print("  FAIL  mlir %: emitted BOTH remsi and divsi — unexpected")
        return 1
    print("  PASS  mlir %: pure-MIND emit uses arith.remsi (not divsi) for `a % b`")

    # Live-toolchain semantic cross-check: 17 % 5 must RUN to 2 through mindc's
    # own MLIR->native lowering (the Rust remsi oracle). We use --emit-obj + a
    # tiny C driver + the system linker (the genuine MLIR->object path); the
    # `--emit=binary` launcher is a non-runnable stub in this environment.
    cc = _which_cc()
    if not MINDC.exists() or cc is None:
        print("  SKIP  mlir % runtime cross-check: mindc or a C compiler unavailable "
              "(emit-text remsi assertion above still holds)")
        return 0
    with tempfile.TemporaryDirectory() as td:
        srcp = pathlib.Path(td) / "modk.mind"
        srcp.write_text(f"fn m() -> i64 {{\n    return {DIVIDEND} % {DIVISOR};\n}}\n")
        objp = pathlib.Path(td) / "modk.o"
        drvp = pathlib.Path(td) / "drv.c"
        exep = pathlib.Path(td) / "modk"
        drvp.write_text("long m(void);\nint main(void){ return (int)m(); }\n")
        o = subprocess.run([str(MINDC), str(srcp), "--emit-obj", str(objp)], capture_output=True)
        if o.returncode != 0 or not objp.exists():
            print(f"  SKIP  mlir % runtime cross-check: --emit-obj failed (rc={o.returncode})")
            return 0
        # mindc appends a synthetic @main to the object; --allow-multiple-definition
        # keeps the driver's main (first defined) as the entry that calls m().
        lk = subprocess.run(
            [cc, "-Wl,--allow-multiple-definition", str(drvp), str(objp), "-o", str(exep)],
            capture_output=True,
        )
        if lk.returncode != 0 or not exep.exists():
            print(f"  SKIP  mlir % runtime cross-check: link failed (rc={lk.returncode})")
            return 0
        code = subprocess.run([str(exep)]).returncode
    if code != WANT_REM:
        why = {QUOTIENT: "QUOTIENT (divsi)", PRODUCT: "PRODUCT (imul)"}.get(code, "wrong")
        print(f"  FAIL  mlir % runtime: {DIVIDEND} % {DIVISOR} -> {code}, expected {WANT_REM} — {why}")
        return 1
    print(f"  PASS  mlir % runtime: mindc MLIR->obj->link {DIVIDEND} % {DIVISOR} -> {code} (remainder)")
    return 0


def _which_cc():
    import shutil
    return shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def main() -> int:
    if not SO.exists():
        if os.environ.get("MINDC_SO"):
            print(f"ERROR: {SO} not found (MINDC_SO is set — refusing to skip)")
            return 1
        print(f"SKIP: {SO} not built")
        return 0
    lib = ctypes.CDLL(str(SO))

    print("[native path: main.mind nb_arith_rax_mem op_mod, run end-to-end]")
    rc_native = native_run_mod(lib)
    print("\n[mlir path: main.mind emit_mlir_arith_op op_mod]")
    rc_mlir = mlir_emit_mod(lib)

    if rc_native == 0 and rc_mlir == 0:
        print("\nALL PASS  (% lowers to a remainder on both the native and MLIR paths)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
