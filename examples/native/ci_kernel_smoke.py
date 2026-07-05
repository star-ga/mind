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
  mindc build examples/native/ci_kernel.mind --release --emit cdylib \
      --out /tmp/libci_kernel.so
  MINDC_SO=/tmp/libci_kernel.so python3 examples/native/ci_kernel_smoke.py
"""

import ctypes
import os
import pathlib
import sys

_DEFAULT_SO = pathlib.Path("/tmp/libci_kernel.so")
SO = pathlib.Path(os.environ.get("MINDC_SO", str(_DEFAULT_SO)))

EXPECTED = 285  # sum(i*i for i in range(10))


def main() -> int:
    if not SO.exists():
        print(f"FAIL: kernel cdylib not found at {SO}", file=sys.stderr)
        return 1

    lib = ctypes.CDLL(str(SO))
    lib.kernel.restype = ctypes.c_int64
    lib.kernel.argtypes = []

    got = lib.kernel()
    if got != EXPECTED:
        print(f"FAIL: kernel() = {got}, expected {EXPECTED} (silent miscompile)",
              file=sys.stderr)
        return 1

    print(f"PASS: kernel() = {got} (== {EXPECTED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
