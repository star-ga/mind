"""Native stage1 census gate: top-level consts (A8), pointer casts (A2) and a
store intrinsic used as a value (the MIND-Lab `w_set` shape).

What it proves, per fixture in testdata/native_misc/:
  * A stage1 compiler is built from THIS tree's main.mind by running the FROZEN
    pure-MIND seed (testdata/selfhost_loop/stage1.elf) on
    [std seed blob + main.mind + selfhost_driver.mind] — no Rust, no MLIR in that
    step. Each fixture is then fed to that stage1 (stdin protocol
    [user_lo][src_len][std blob + fixture], see selfhost_driver.mind).
  * `run` fixtures: the native ELF must build, run, and produce the HAND-COMPUTED
    exit code and stdout (written in the CASES table below, derived from the
    fixture text, never from either compiler), AND the MLIR build of the same
    file (`mindc build`) must produce the same exit code and stdout. Two
    independent oracles: arithmetic by hand, and the C-runtime/MLIR backend.
  * `refuse` fixtures: stage1 must fail CLOSED (non-zero exit, 0 bytes of ELF).
    The MLIR leg is still run and must match the hand-computed outcome, so each
    refusal is pinned against what the program would have meant.

Env:
  MISC_MAIN_MIND  main.mind to build stage1 from (default: this tree's). Used to
                  show the gate RED on a base-tree copy or on a mutant.
  MINDC_BIN       mindc for the MLIR oracle leg (default: the shared resolver).

Run:  python3 examples/mindc_mind/self_host_native_misc_census_smoke.py
"""
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _stdlib_manifest import seed_blob  # noqa: E402

FROZEN_SEED = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
DRIVER = HERE / "selfhost_driver.mind"
FIX = HERE / "testdata" / "native_misc"
CACHE = pathlib.Path(tempfile.gettempdir()) / "native_misc_stage1_cache"

# (fixture, kind, native/hand want, MLIR want). A want is (exit, stdout) for a
# program that runs; "build-fail" for a program the MLIR front-end rejects.
RUN, REFUSE = "run", "refuse"
CASES = [
    # --- A8: top-level consts, `;` optional ---------------------------------
    ("a8_const_nosemi", RUN, (42, b"OK\n"), (42, b"OK\n")),
    ("a8_const_mixed", RUN, (7, b""), (7, b"")),
    ("a8_refuse_constexpr", REFUSE, None, (3, b"")),
    ("a8_refuse_float_const", REFUSE, None, "build-fail"),
    # --- A2: pointer casts ---------------------------------------------------
    ("a2_ptrcast_ok", RUN, (59, b"ptr\n"), (59, b"ptr\n")),
    ("a2_refuse_src_i32let", REFUSE, None, (6, b"")),
    ("a2_refuse_src_u32let", REFUSE, None, (252, b"")),
    ("a2_refuse_src_f64let", REFUSE, None, (4, b"")),
    ("a2_refuse_src_unannotated", REFUSE, None, (45, b"")),
    ("a2_refuse_src_cmp", REFUSE, None, (2, b"")),
    ("a2_refuse_src_narrowcall", REFUSE, None, (8, b"")),
    ("a2_refuse_src_narrowparam", REFUSE, None, (5, b"")),
    ("a2_refuse_ptr_to_i32", REFUSE, None, (5, b"")),
    ("a2_refuse_ptr_to_u8", REFUSE, None, (44, b"")),
    ("a2_refuse_ptr_let_annotation", REFUSE, None, (10, b"")),
    # --- store intrinsic used as a value (returns 0) -------------------------
    ("st_tail_w_set", RUN, (40, b""), (40, b"")),
    ("st_let_init", RUN, (40, b""), (40, b"")),
    ("st_return_i32", RUN, (40, b""), (40, b"")),
    ("st_valueif_i8", RUN, (40, b""), (40, b"")),
    ("st_operand_i16", RUN, (40, b""), (40, b"")),
    ("st_tail_valueif_main", RUN, (40, b""), (40, b"")),
    ("st_tail_chain", RUN, (40, b""), (40, b"")),
    ("st_main_tail", RUN, (0, b""), (0, b"")),
    ("st_cond_use", RUN, (64, b""), (64, b"")),
]


def main_mind() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MISC_MAIN_MIND", str(HERE / "main.mind")))


def _write_exec(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    os.chmod(path, 0o755)
    return path


def build_stage1() -> pathlib.Path:
    """stage1 = the frozen seed run on the chosen main.mind (cached by input sha)."""
    blob = seed_blob()
    combined = blob + main_mind().read_bytes() + b"\n" + DRIVER.read_bytes()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    key = hashlib.sha256(FROZEN_SEED.read_bytes() + img).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"stage1_{key}.elf"
    if out.exists():
        return out
    r = subprocess.run([str(FROZEN_SEED)], input=img, capture_output=True, timeout=1500)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise SystemExit(f"frozen seed failed to build stage1: rc={r.returncode} "
                         f"stderr={r.stderr[-400:]!r}")
    tmp = CACHE / f".{key}.{os.getpid()}"
    _write_exec(tmp, r.stdout)
    os.replace(tmp, out)
    return out


def native_compile(stage1: pathlib.Path, src: bytes) -> tuple[int, bytes, str]:
    blob = seed_blob()
    combined = blob + src
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    r = subprocess.run([str(stage1)], input=img, capture_output=True, timeout=900)
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")[-300:]


def run_exe(path: pathlib.Path) -> tuple[int, bytes]:
    r = subprocess.run([str(path)], capture_output=True, timeout=120)
    return r.returncode, r.stdout


def mindc_bin() -> str:
    if os.environ.get("MINDC_BIN"):
        return os.environ["MINDC_BIN"]
    from _selfhost_so import resolve_mindc
    return resolve_mindc()


def mlir_outcome(src: pathlib.Path, work: pathlib.Path, name: str):
    # Build a COPY outside the repo: `mindc build` on a file inside the tree
    # would pick up the repository's Mind.toml and emit a cdylib instead.
    copy = work / f"{name}.mind"
    copy.write_bytes(src.read_bytes())
    out = work / f"{name}.mlir.bin"
    b = subprocess.run([mindc_bin(), "build", str(copy), "--release", "--out", str(out)],
                       capture_output=True, timeout=900, cwd=str(work))
    if b.returncode != 0 or not out.exists():
        return "build-fail"
    return run_exe(out)


def main() -> int:
    if not FROZEN_SEED.exists():
        print(f"[FAIL] frozen seed missing: {FROZEN_SEED}")
        return 1
    stage1 = build_stage1()
    print(f"stage1 from {main_mind()} -> {stage1.name}")
    failures = []
    with tempfile.TemporaryDirectory(prefix="native_misc_") as tdn:
        work = pathlib.Path(tdn)
        for name, kind, want, want_mlir in CASES:
            src = FIX / f"{name}.mind"
            rc, out, err = native_compile(stage1, src.read_bytes())
            is_elf = rc == 0 and out[:4] == b"\x7fELF"
            if kind == RUN:
                if not is_elf:
                    got = f"compile rc={rc} {len(out)}B err={err.strip()[-120:]!r}"
                    ok = False
                else:
                    exe = _write_exec(work / f"{name}.native", out)
                    got_run = run_exe(exe)
                    got = f"run={got_run!r}"
                    ok = got_run == want
                print(f"[{'PASS' if ok else 'FAIL'}] native {name}: {got} want={want!r}")
            else:
                ok = rc != 0 and len(out) == 0
                print(f"[{'PASS' if ok else 'FAIL'}] native refuse {name}: "
                      f"rc={rc} {len(out)}B (want non-zero rc, 0B)")
            if not ok:
                failures.append(f"native:{name}")
            got_mlir = mlir_outcome(src, work, name)
            mok = got_mlir == want_mlir
            print(f"[{'PASS' if mok else 'FAIL'}] mlir {name}: got={got_mlir!r} want={want_mlir!r}")
            if not mok:
                failures.append(f"mlir:{name}")
    if failures:
        print(f"FAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print(f"ALL PASS ({len(CASES)} fixtures: native vs hand-computed, MLIR vs hand-computed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
