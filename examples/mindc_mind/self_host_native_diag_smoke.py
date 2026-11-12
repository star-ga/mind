#!/usr/bin/env python3
"""RI-D1a gate (task #313): `mindc build --backend=native` must FAIL-CLOSED with an
EXACT diagnostic that NAMES the unsupported construct on stderr and exits NON-ZERO —
never the old silent 0-byte-stdout-exit-0. The pure-MIND self-host driver
(selfhost_driver.mind) scans the user region and writes the diagnostic; the Rust
bridge (run_native_backend_bridge, src/bin/mindc.rs) surfaces stage1.elf's stderr.

Four invariants, all through the SHIPPING `mindc build --backend=native` path
(spawns the frozen pure-MIND stage1.elf — zero MLIR/LLVM/clang):

  (1) SUPPORTED (scalar)  -> exit 0, an ELF is written, it RUNS to the expected code.
  (2) UNSUPPORTED tensor  -> exit != 0, NO artifact, stderr names "tensor".
  (3) UNSUPPORTED trait   -> exit != 0, NO artifact, the parser diagnostic names
      the construct and carries file/source/caret context.
  (4) MALFORMED source    -> exit != 0, NO artifact, the same structured parser
      diagnostic carries file/source/caret context.
  (5) SECOND FENCE        -> a whole-array binop fed straight to the pure-MIND
      compiler (no Rust fence in front) emits no ELF, while an element-read
      program fed the same way runs to its value. The compiler is --stage1, or
      one hop of the self-host loop: the committed seed compiling THIS tree.
      The case prints the sha256 of the compiler it fed and of the committed
      seed, so a mutation run shows it ran a compiler other than the seed.

  --stage1 PATH  the native compiler for every case: (1)-(4) run mindc with
                 MINDC_NATIVE_ELF=PATH, (5) feeds PATH directly.

This is the regression coverage for the fail-close diagnostic: without it, a
future driver/bridge change could silently revert to the 0-byte generic message
(UNSUPPORTED_FEATURE_DIAGNOSTICS regressing from EXACT), and nothing else catches it.

Exit: 0 all pass; 1 a gate failed; 2 BLOCKED (missing mindc / stage1.elf).

Run:  python3 examples/mindc_mind/self_host_native_diag_smoke.py
"""
import argparse
import hashlib
import os
import pathlib
import struct
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
MALFORMED = "fn main( -> i64 { return 0; }\n"
WHOLE_ARRAY = "fn main() -> i64 { let a = [8, 6]; let b = [2, 3]; let c = a + b; return c[0]; }\n"
ELEMENTS = "fn main() -> i64 { let a = [5, 6]; return a[1] + a[0]; }\n"


def tree_stage1() -> bytes:
    """One hop of the self-host loop: the committed seed compiles this tree."""
    sys.path.insert(0, str(_HERE))
    import self_host_loop_smoke as loop  # noqa: E402
    _combined, image, _user_lo = loop.build_seed()
    return loop.run_elf(STAGE1, image)


def compile_direct(stage1: pathlib.Path, src: str) -> bytes:
    """Feed an unseeded image straight to the compiler; return its stdout."""
    body = src.encode()
    image = struct.pack("<qq", 0, len(body)) + body
    r = subprocess.run([str(stage1)], input=image, capture_output=True, timeout=600)
    return r.stdout


def run_elf_bytes(elf: bytes):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(elf)
        p = f.name
    os.chmod(p, 0o755)
    try:
        return subprocess.run([p], timeout=10).returncode
    finally:
        os.unlink(p)


def build_native(src: str):
    """mindc build --backend=native <src>; return (exit_code, stderr_text, artifact_bytes_or_None)."""
    with tempfile.TemporaryDirectory() as td:
        sp = pathlib.Path(td) / "p.mind"
        op = pathlib.Path(td) / "p.bin"
        sp.write_text(src)
        r = subprocess.run(
            [str(MINDC), "build", "--backend=native", f"--out={op}", str(sp)],
            capture_output=True, text=True, timeout=120, env=ENV,
        )
        art = op.read_bytes() if op.exists() and op.stat().st_size > 0 else None
        return r.returncode, (r.stderr or ""), art


ENV = dict(os.environ)


def main() -> int:
    global STAGE1
    ap = argparse.ArgumentParser(description="RI-D1a native diagnostic gate")
    ap.add_argument("--stage1", type=pathlib.Path, default=None,
                    help="native compiler for every case (default: the committed seed)")
    args = ap.parse_args()
    if args.stage1 is not None:
        STAGE1 = args.stage1.resolve()
        ENV["MINDC_NATIVE_ELF"] = str(STAGE1)
    print(f"  stage1 = {STAGE1}")
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
    print(f"  {'PASS' if ok else 'FAIL'}  scalar: exit={ec} "
          f"elf={'YES' if art else 'NO'} result={ran}(want 42)")
    fails += 0 if ok else 1

    # (2) UNSUPPORTED tensor -> non-zero, no artifact, stderr names the construct
    ec, err, art = build_native(TENSOR)
    ok = (ec != 0 and art is None and "tensor" in err.lower())
    print(f"  {'PASS' if ok else 'FAIL'}  tensor: exit={ec} artifact={'YES' if art else 'NO'} "
          f"names_tensor={'tensor' in err.lower()}  stderr={err.strip()[:70]!r}")
    fails += 0 if ok else 1

    # (3) UNSUPPORTED trait -> non-zero, no artifact, exact parser context
    ec, err, art = build_native(TRAIT)
    trait_context = all(part in err for part in (
        "error[parse][E1001]", "p.mind:1:", TRAIT.splitlines()[0], "^"
    ))
    ok = (ec != 0 and art is None and "trait" in err.lower() and trait_context)
    print(f"  {'PASS' if ok else 'FAIL'}  trait:  exit={ec} artifact={'YES' if art else 'NO'} "
          f"names_trait={'trait' in err.lower()} context={trait_context}  "
          f"stderr={err.strip()[:100]!r}")
    fails += 0 if ok else 1

    # (4) MALFORMED source -> structured parse diagnostic with source context
    ec, err, art = build_native(MALFORMED)
    malformed_context = all(part in err for part in (
        "error[parse][E1001]", "p.mind:1:", MALFORMED.strip(), "^"
    ))
    ok = (ec != 0 and art is None and malformed_context)
    print(f"  {'PASS' if ok else 'FAIL'}  malformed: exit={ec} "
          f"artifact={'YES' if art else 'NO'} context={malformed_context}  "
          f"stderr={err.strip()[:100]!r}")
    fails += 0 if ok else 1

    # (5) SECOND FENCE -> the pure-MIND compiler itself refuses an array operand
    with tempfile.TemporaryDirectory() as td:
        if args.stage1 is not None:
            direct = STAGE1
        else:
            direct = pathlib.Path(td) / "stage1.elf"
            direct.write_bytes(tree_stage1())
            direct.chmod(0o755)
        fed_sha = hashlib.sha256(direct.read_bytes()).hexdigest()
        refused = compile_direct(direct, WHOLE_ARRAY)
        control = compile_direct(direct, ELEMENTS)
        got = run_elf_bytes(control) if control[:4] == b"\x7fELF" else None
    wrote = refused[:4] == b"\x7fELF"
    ok = not wrote and got == 11
    print(f"  {'PASS' if ok else 'FAIL'}  second fence: whole-array binop "
          f"elf={'YES' if wrote else 'NO'}(want NO)  element read result={got}(want 11)")
    seed = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
    seed_sha = hashlib.sha256(seed.read_bytes()).hexdigest()
    print(f"        compiler sha256={fed_sha}  committed seed sha256={seed_sha}")
    fails += 0 if ok else 1

    if fails:
        print(f"FAIL: {fails} RI-D1a diagnostic gate(s) failed")
        return 1
    print("ALL PASS  (RI-D1a: native backend fail-closes with an EXACT construct-naming "
          "diagnostic + non-zero exit; supported scalar still compiles+runs; zero MLIR/LLVM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
