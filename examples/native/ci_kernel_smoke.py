"""
Executable miscompile gate — compile a tiny deterministic MIND kernel, RUN it,
and ASSERT a known output.

Unlike the byte-identity keystone (which proves two builds agree) this gate
proves the compiled code actually *computes the right answer* — it catches a
silent miscompile that is nonetheless self-consistent (e.g. a wrong operator
lowering that reproduces byte-identically on every substrate but returns the
wrong integer).

The kernel (examples/native/ci_kernel.mind) sums i*i for i in 0..10, exercising
`while`, `let mut` re-assignment, integer `*`, `+`, and `<` comparison — the
executable-subset arithmetic that lowers straight through the frozen frontend.
The one true answer is 0+1+4+9+16+25+36+49+64+81 = 285.

Run:
  cargo build --release --no-default-features \
      --features "mlir-build std-surface cross-module-imports" --bin mindc
  python3 examples/native/ci_kernel_smoke.py

The gate copies the exact kernel bytes into an owned temporary directory,
builds that standalone source with `--no-cache`, and loads the resulting
artifact. `MINDC` may point at the compiler under test; `MINDC_SO` is ignored so
an artifact supplied by a caller can neither satisfy the gate nor be removed.
`MIND_GATE_TIMEOUT` bounds the compiler subprocess (900 seconds by default).
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
KERNEL_SRC = REPO / "examples" / "native" / "ci_kernel.mind"
MANIFEST = REPO / "Mind.toml"
BUILD_TIMEOUT = int(os.environ.get("MIND_GATE_TIMEOUT", "900"))

EXPECTED = 285  # sum(i*i for i in range(10))


def resolve_mindc() -> pathlib.Path | None:
    """Resolve the compiler that this gate must exercise."""
    override = os.environ.get("MINDC")
    if override:
        candidate = pathlib.Path(override)
        return (candidate if candidate.is_file() and os.access(candidate, os.X_OK)
                else None)
    candidate = REPO / "target" / "release" / "mindc"
    return (candidate if candidate.is_file() and os.access(candidate, os.X_OK)
            else None)


def main() -> int:
    """Compile and execute the kernel, with every check counted by assertions.

    ``scripts/run_gate.py`` routes Python gates through ``gate_assert.py``,
    which counts these assertions and publishes the contract line.  Keeping
    the checks as real asserts prevents this gate from manufacturing its own
    assertion count.
    """
    mindc = resolve_mindc()
    # A gate that cannot exercise the compiler has not shown the absence of a
    # miscompile.  It must fail, never quietly skip.
    assert mindc is not None, (
        "mindc is not built — build it with `cargo build --release "
        '--no-default-features --features "mlir-build std-surface '
        'cross-module-imports" --bin mindc`'
    )
    assert KERNEL_SRC.is_file(), f"kernel source missing at {KERNEL_SRC}"
    assert MANIFEST.is_file(), f"project manifest missing at {MANIFEST}"
    manifest_before = MANIFEST.read_bytes()
    manifest_mtime_before = MANIFEST.stat().st_mtime_ns

    # Do not honor MINDC_SO here.  Other gates use it for an input oracle, but
    # this gate's artifact must be owned by this invocation.  A unique temp
    # directory also leaves any caller-provided .so untouched on every path.
    with tempfile.TemporaryDirectory(prefix="ci_kernel_gate_") as td:
        source = pathlib.Path(td) / "ci_kernel.mind"
        so = pathlib.Path(td) / "libci_kernel.so"
        source_bytes = KERNEL_SRC.read_bytes()
        source.write_bytes(source_bytes)
        assert source.read_bytes() == source_bytes, "standalone kernel copy changed"

        proc = None
        timed_out = None
        try:
            proc = subprocess.run(
                [
                    str(mindc), "build", "--backend", "mlir", str(source), "--no-cache",
                    "--release", "--emit", "cdylib", "--out", str(so),
                ],
                cwd=pathlib.Path(td),
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = exc

        manifest_after = MANIFEST.read_bytes()
        manifest_mtime_after = MANIFEST.stat().st_mtime_ns
        assert manifest_after == manifest_before, "Mind.toml bytes changed during gate"
        assert manifest_mtime_after == manifest_mtime_before, "Mind.toml mtime changed during gate"
        assert timed_out is None, f"mindc build timed out after {BUILD_TIMEOUT}s"
        assert proc is not None, "mindc build produced no subprocess result"
        assert proc.returncode == 0, (
            f"mindc build returned {proc.returncode}\n"
            f"{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
        )
        assert so.is_file(), (
            f"mindc build reported success but produced no artifact at {so}"
        )

        lib = ctypes.CDLL(str(so))
        lib.kernel.restype = ctypes.c_int64
        lib.kernel.argtypes = []

        got = lib.kernel()
        assert got == EXPECTED, (
            f"kernel() = {got}, expected {EXPECTED} (silent miscompile)"
        )

        print(f"PASS: kernel() = {got} (== {EXPECTED}), compiled this run by {mindc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
