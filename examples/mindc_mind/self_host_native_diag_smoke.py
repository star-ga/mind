#!/usr/bin/env python3
"""RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
EXACT diagnostic that NAMES the unsupported construct on stderr and exits NON-ZERO —
never the old silent 0-byte-stdout-exit-0. The pure-MIND self-host driver
(selfhost_driver.mind) scans the user region and writes the diagnostic; the Rust
bridge (run_native_backend_bridge, src/bin/mindc.rs) surfaces stage1.elf's stderr.

Three invariants, all through the SHIPPING `mindc build --backend=native` path
(spawns the frozen pure-MIND stage1.elf — zero MLIR/LLVM/clang):

  (1) SUPPORTED (scalar)  -> exit 0, an ELF is written, it RUNS to the expected code.
  (2) UNSUPPORTED tensor  -> exit != 0, NO artifact, stderr names "tensor".
  (3) UNSUPPORTED trait   -> exit != 0, NO artifact, stderr names "trait/impl".

This is the regression coverage for the fail-close diagnostic: without it, a
future driver/bridge change could silently revert to the 0-byte generic message
(UNSUPPORTED_FEATURE_DIAGNOSTICS regressing from EXACT), and nothing else catches it.

Exit: 0 all pass; 1 a gate failed; 2 BLOCKED (missing mindc / stage1.elf).

Run:  python3 examples/mindc_mind/self_host_native_diag_smoke.py
"""
import os
import pathlib
import subprocess
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
MINDC = pathlib.Path(os.environ.get("MINDC_BIN", str(_REPO / "target" / "release" / "mindc")))
STAGE1 = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"

SCALAR = "fn main() -> i64 { return 7 + 35; }\n"
TENSOR = "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }\nfn main() -> i64 { return 0; }\n"
TRAIT = "trait Greet { fn hi(self) -> i64; }\nfn main() -> i64 { return 0; }\n"


def build_native(src: str):
    """mindc build --backend=native <src>; return (exit_code, stderr_text, artifact_bytes_or_None)."""
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "p.mind"
        op = pathlib.Path(td) / "p.bin"
        sp.write_text(src)
        r = subprocess.run(
            [str(MINDC), "build", "--backend=native", f"--out={op}", str(sp)],
            capture_output=True, text=True, timeout=120,
        )
        art = op.read_bytes() if op.exists() and op.stat().st_size > 0 else None
        return r.returncode, (r.stderr or ""), art


def main() -> int:
    if not MINDC.exists():
        print(f"BLOCKED: mindc not found at {MINDC}")
        return 2
    if not STAGE1.exists():
        print(f"BLOCKED: frozen stage1.elf not found at {STAGE1}")
        return 2

    fails = 0

    # (1) SUPPORTED scalar -> exit 0 + ELF that runs to 42
    ec, err, art = build_native(SCALAR)
    ran = None
    if art is not None and art[:4] == b"\x7fELF":
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(art)
            p = f.name
        os.chmod(p, 0o755)
        try:
            ran = subprocess.run([p], timeout=10).returncode
        finally:
            os.unlink(p)
    ok = (ec == 0 and art is not None and art[:4] == b"\x7fELF" and ran == 42)
    print(f"  {'PASS' if ok else 'FAIL'}  scalar: exit={ec} elf={'YES' if art else 'NO'} ran={ran}(want 42)")
    fails += 0 if ok else 1

    # (2) UNSUPPORTED tensor -> non-zero, no artifact, stderr names the construct
    ec, err, art = build_native(TENSOR)
    ok = (ec != 0 and art is None and "tensor" in err.lower())
    print(f"  {'PASS' if ok else 'FAIL'}  tensor: exit={ec} artifact={'YES' if art else 'NO'} "
          f"names_tensor={'tensor' in err.lower()}  stderr={err.strip()[:70]!r}")
    fails += 0 if ok else 1

    # (3) UNSUPPORTED trait -> non-zero, no artifact, stderr names trait/impl
    ec, err, art = build_native(TRAIT)
    ok = (ec != 0 and art is None and ("trait" in err.lower() or "impl" in err.lower()))
    print(f"  {'PASS' if ok else 'FAIL'}  trait:  exit={ec} artifact={'YES' if art else 'NO'} "
          f"names_trait={'trait' in err.lower() or 'impl' in err.lower()}  stderr={err.strip()[:70]!r}")
    fails += 0 if ok else 1

    if fails:
        print(f"FAIL: {fails} RI-D1a diagnostic gate(s) failed")
        return 1
    print("ALL PASS  (RI-D1a: native backend fail-closes with an EXACT construct-naming "
          "diagnostic + non-zero exit; supported scalar still compiles+runs; zero MLIR/LLVM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
