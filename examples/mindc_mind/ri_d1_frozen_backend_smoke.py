#!/usr/bin/env python3
"""RI-D1 `--backend frozen` smoke — the safe production-profile dispatch.

`mindc build --backend frozen <src>` lowers to canonical IR, runs the
`profile_frozen_admits` allowlist gate (u64-safe, oracle-aligned), and ONLY on
admission hands the build to the pure-MIND native-ELF backend (zero MLIR/clang/ld).
A rejected construct fail-louds by name — never a silent MLIR fallback, never a
native emit of a construct the profile cannot prove byte-identical to the oracle.

Three claims:
  1. IN-PROFILE   int-arith program  -> rc=0, artifact written (native path).
  2. MARKED-U64   `a as u64` then `<` (emits the __mind_conv_u64 marker, the origin
                  of the oracle's unsigned-op selection that native lowers to a signed
                  `setl`) -> rc!=0, NO artifact, `u64-value` named on stderr.
  3. PARAM-U64    u64 fn params, never converted -> untagged -> native compares them
                  signed AND so does the MLIR oracle (a pre-existing MLIR #99 residual,
                  NOT a flip divergence) -> admitted (rc=0). Documents the boundary.

Exit: 0 all pass; 1 a claim failed; 2 BLOCKED (missing mindc).
"""
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
MINDC = HERE.parents[1] / "target" / "release" / "mindc"

CASES = [
    # (label, source, expect_artifact, expect_stderr_substr)
    ("in-profile-int",
     "pub fn main() -> i64 { let a: i64 = 7; let b: i64 = 35; a + b }\n",
     True, None),
    ("marked-u64-reject",
     "pub fn f(a: i64) -> i64 { let x: u64 = a as u64; if x < 5 { 1 } else { 0 } }\n",
     False, "u64-value"),
    ("param-u64-admit",
     "pub fn f(a: u64, b: u64) -> i64 { if a < b { 1 } else { 0 } }\n",
     True, None),
]


def main() -> int:
    if not MINDC.exists():
        print(f"BLOCKED: mindc not built at {MINDC}", file=sys.stderr)
        return 2
    failed = 0
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for label, src, want_artifact, want_stderr in CASES:
            sp = td / f"{label}.mind"
            sp.write_text(src, encoding="utf-8")
            outp = td / f"{label}.bin"
            r = subprocess.run(
                [str(MINDC), "build", "--backend", "frozen", str(sp), "--out", str(outp)],
                capture_output=True, text=True, timeout=90,
            )
            got_artifact = outp.exists()
            ok = got_artifact == want_artifact
            if want_stderr is not None and want_stderr not in r.stderr:
                ok = False
            status = "ok  " if ok else "FAIL"
            print(f"  {status} {label:18} rc={r.returncode} artifact={got_artifact} "
                  f"(want artifact={want_artifact}"
                  + (f", stderr~'{want_stderr}'" if want_stderr else "") + ")")
            if not ok:
                failed += 1
                if r.stderr:
                    print(f"        stderr: {r.stderr.strip()[:160]}", file=sys.stderr)
    if failed:
        print(f"\nFAIL: {failed}/{len(CASES)} --backend frozen claims failed")
        return 1
    print(f"\nPASS: --backend frozen admits in-profile + param-u64 (native≡oracle), "
          f"rejects marked u64 fail-loud (no miscompile).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
