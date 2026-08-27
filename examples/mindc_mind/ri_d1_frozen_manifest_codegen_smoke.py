#!/usr/bin/env python3
"""RI-D1 manifest `codegen` named-profile-default smoke — the RI-D1 flip slice.

Proves that a project can build native-by-default (zero MLIR/clang/ld) for a
named production profile WITHOUT typing `--backend frozen` every time, by
declaring `[targets.<name>].codegen = "frozen"` in `Mind.toml`. This is the
slice that flips RI_DEPENDENCY_MATRIX rows 3-7 PARTIAL -> PASS.

Three claims (over one on-disk mini-project with `[targets.prod] codegen =
"frozen"`):

  A. MANIFEST-FROZEN  `mindc build --target prod <src>` (NO --backend) routes
                      through the frozen native dispatch: rc=0, an ELF artifact,
                      and — proven by strace — the ONLY execve'd programs are
                      {mindc, stage1.elf}. Zero mlir-opt / mlir-translate / clang
                      / ld. The manifest field alone selected the native backend.

  B. CLI-PRECEDENCE   `mindc build --target prod --backend mlir <src>` on the
                      SAME project overrides the manifest `codegen = "frozen"`:
                      strace shows the MLIR toolchain (mlir-opt / mlir-translate
                      / clang) IS invoked and stage1.elf is NOT — an explicit
                      `--backend` always wins over the manifest.

  C. MANIFEST-FAILLOUD  An out-of-profile construct (a u64 order-compare, which
                      the native emitter would silently miscompile to a signed
                      `setl`) under the manifest-selected frozen profile
                      fail-louds by name (`backend-frozen` / `u64-value`), writes
                      NO artifact, and never silently falls back to MLIR — exactly
                      as an explicit `--backend frozen` does.

Exit: 0 all pass; 1 a claim failed; 2 BLOCKED (missing mindc / strace / stage1.elf).
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
MINDC = REPO / "target" / "release" / "mindc"
STD_DIR = REPO / "std"
NATIVE_ELF = REPO / "examples" / "mindc_mind" / "testdata" / "selfhost_loop" / "stage1.elf"

# Programs that mean "MLIR / native-toolchain path was taken" (must be ABSENT on
# the frozen route, PRESENT on the --backend mlir override route).
MLIR_TOOLS = {"mlir-opt", "mlir-translate", "clang", "clang-20", "cc", "gcc", "ld", "lld"}
# The only programs the frozen native route is allowed to execve.
FROZEN_ALLOWED = {"mindc", "stage1.elf"}

MANIFEST = """\
[package]
name = "prodapp"
version = "0.1.0"

[build]
entry = "main.mind"
output = "prodapp"

[targets.prod]
backend = "cpu"
codegen = "frozen"
"""

IN_PROFILE_SRC = "pub fn main() -> i64 { let a: i64 = 7; let b: i64 = 35; a + b }\n"
OUT_OF_PROFILE_SRC = (
    "pub fn f(a: i64) -> i64 { let x: u64 = a as u64; if x < 5 { 1 } else { 0 } }\n"
)

EXECVE_RE = re.compile(r'execve\("([^"]+)"')


def _run_traced(args, cwd, env):
    """Run `strace -f -e trace=execve mindc <args>`; return (rc, stderr, execve_basenames)."""
    with tempfile.NamedTemporaryFile("w+", suffix=".strace", delete=False) as tf:
        trace_path = tf.name
    try:
        r = subprocess.run(
            ["strace", "-f", "-qq", "-e", "trace=execve", "-o", trace_path,
             str(MINDC), *args],
            cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
        )
        basenames = set()
        with open(trace_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = EXECVE_RE.search(line)
                if m:
                    basenames.add(os.path.basename(m.group(1)))
        return r.returncode, r.stderr, basenames
    finally:
        try:
            os.unlink(trace_path)
        except OSError:
            pass


def main() -> int:
    if not MINDC.exists():
        print(f"BLOCKED: mindc not built at {MINDC}", file=sys.stderr)
        return 2
    if shutil.which("strace") is None:
        print("BLOCKED: strace not available (execve subset assertion needs it)",
              file=sys.stderr)
        return 2
    if not NATIVE_ELF.exists():
        print(f"BLOCKED: frozen native compiler ELF missing at {NATIVE_ELF}",
              file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["MINDC_STD_DIR"] = str(STD_DIR)
    env["MINDC_NATIVE_ELF"] = str(NATIVE_ELF)

    failed = 0
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "Mind.toml").write_text(MANIFEST, encoding="utf-8")
        src = td / "main.mind"
        src.write_text(IN_PROFILE_SRC, encoding="utf-8")

        # --- Claim A: manifest codegen="frozen" routes native, zero MLIR ---
        outa = td / "a.bin"
        rc, err, execs = _run_traced(
            ["build", "--target", "prod", str(src), "--out", str(outa)], td, env)
        stray = execs & MLIR_TOOLS
        subset_ok = execs.issubset(FROZEN_ALLOWED)
        is_elf = outa.exists() and outa.read_bytes()[:4] == b"\x7fELF"
        okA = rc == 0 and is_elf and subset_ok and not stray
        print(f"  {'ok  ' if okA else 'FAIL'} A manifest-frozen      rc={rc} "
              f"elf={is_elf} execve={sorted(execs)}")
        if not okA:
            failed += 1
            print(f"        stray-mlir-tools={sorted(stray)} subset_ok={subset_ok} "
                  f"stderr={err.strip()[:200]}", file=sys.stderr)

        # --- Claim B: explicit --backend mlir overrides manifest codegen ---
        outb = td / "b.bin"
        rc, err, execs = _run_traced(
            ["build", "--target", "prod", "--backend", "mlir", str(src),
             "--out", str(outb)], td, env)
        used_mlir = bool(execs & MLIR_TOOLS)
        no_frozen = "stage1.elf" not in execs and "pure-MIND native-ELF" not in err
        okB = used_mlir and no_frozen
        print(f"  {'ok  ' if okB else 'FAIL'} B cli-precedence-mlir  rc={rc} "
              f"used_mlir={used_mlir} no_frozen={no_frozen} execve={sorted(execs)}")
        if not okB:
            failed += 1
            print(f"        stderr={err.strip()[:200]}", file=sys.stderr)

        # --- Claim C: out-of-profile construct under manifest frozen fail-louds ---
        src.write_text(OUT_OF_PROFILE_SRC, encoding="utf-8")
        outc = td / "c.bin"
        r = subprocess.run(
            [str(MINDC), "build", "--target", "prod", str(src), "--out", str(outc)],
            cwd=str(td), env=env, capture_output=True, text=True, timeout=90)
        no_artifact = not outc.exists()
        named = "backend-frozen" in r.stderr and "u64-value" in r.stderr
        okC = r.returncode != 0 and no_artifact and named
        print(f"  {'ok  ' if okC else 'FAIL'} C manifest-fail-loud   rc={r.returncode} "
              f"no_artifact={no_artifact} named={named}")
        if not okC:
            failed += 1
            print(f"        stderr={r.stderr.strip()[:200]}", file=sys.stderr)

    if failed:
        print(f"\nFAIL: {failed}/3 manifest-codegen claims failed")
        return 1
    print("\nPASS: [targets.prod] codegen=\"frozen\" builds native-by-default "
          "(execve ⊆ {mindc, stage1.elf}); explicit --backend mlir overrides it; "
          "out-of-profile fail-louds — RI-D1 rows 3-7 flip PARTIAL→PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
