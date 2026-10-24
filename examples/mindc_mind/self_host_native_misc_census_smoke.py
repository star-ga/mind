"""Native stage1 census gate: top-level int consts (A8; string consts refuse), pointer casts (A2) and a
store intrinsic used as a value (a `w_set`-style store-tail helper).

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
  * `run` fixtures also pin the TWIN invariant: the native ELF's 32-byte trace
    note (the pure-MIND mic@3 twin) must equal the Rust emit_mic3 reference note
    of the same seeded program (mindfuzz_self_host.RustRef, `cargo test
    _ref_mic3_dump`) — a construct the native path runs correctly but records
    differently from mindc-Rust is a divergence, not a pass.
  * `refuse` fixtures: stage1 must fail CLOSED (non-zero exit, 0 bytes of ELF).
    The MLIR leg is still run and must match the hand-computed outcome, so each
    refusal is pinned against what the program would have meant.

Env:
  MISC_MAIN_MIND  main.mind to build stage1 from (default: this tree's). Used to
                  show the gate RED on a base-tree copy or on a mutant.
  MINDC_BIN       mindc for the MLIR oracle leg (default: the shared resolver).
  MISC_JOBS       fixtures checked concurrently (default 3).
  MISC_ONLY       comma-separated fixture names to check (default: all).

Run:  python3 examples/mindc_mind/self_host_native_misc_census_smoke.py
"""
import concurrent.futures
import errno
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _stdlib_manifest import seed_blob  # noqa: E402
from mindfuzz_self_host import RustRef, note_of_elf  # noqa: E402

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
    ("a8_const_narrow_u8", RUN, (150, b""), (150, b"")),
    ("a8_const_narrow_types", RUN, (181, b""), (181, b"")),
    ("a8_refuse_u64_const_div", REFUSE, None, (4, b"")),
    ("a8_const_usize", RUN, (8, b""), (8, b"")),
    ("a8_refuse_bool_const", REFUSE, None, "build-fail"),
    ("a8_refuse_const_assign", REFUSE, None, (7, b"")),
    ("a8_refuse_const_compound", REFUSE, None, (8, b"")),
    ("a8_str_const", REFUSE, None, (22, b"hello hi\n")),
    ("a8_str_refuse_i64type", REFUSE, None, (3, b"")),
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
    # one read position each, so a walker that skips that position goes red
    ("st_pos_if_cond", RUN, (15, b""), (15, b"")),
    ("st_pos_while_cond", RUN, (37, b""), (37, b"")),
    ("st_pos_assign", RUN, (8, b""), (8, b"")),
    ("st_pos_call_arg", RUN, (8, b""), (8, b"")),
]


def selected_cases() -> list:
    only = [n for n in os.environ.get("MISC_ONLY", "").split(",") if n]
    if not only:
        return CASES
    known = {c[0] for c in CASES}
    unknown = [n for n in only if n not in known]
    if unknown:
        raise SystemExit(f"MISC_ONLY names unknown fixtures: {unknown}")
    return [c for c in CASES if c[0] in only]


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
    # A sibling worker's fork can briefly inherit the write descriptor of an
    # executable this worker just wrote; exec then fails with ETXTBSY until that
    # child execs. Retry that one error only.
    for attempt in range(50):
        try:
            r = subprocess.run([str(path)], capture_output=True, timeout=120)
            return r.returncode, r.stdout
        except OSError as e:
            if e.errno != errno.ETXTBSY or attempt == 49:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


def mindc_bin() -> str:
    if os.environ.get("MINDC_BIN"):
        return os.environ["MINDC_BIN"]
    from _selfhost_so import resolve_mindc
    return resolve_mindc()


def mlir_outcome(src: pathlib.Path, work: pathlib.Path, name: str):
    # Build a COPY outside the repo: `mindc build` on a file inside the tree
    # would pick up the repository's Mind.toml and emit a cdylib instead. Each
    # fixture gets its own directory so concurrent builds never share the
    # project-local build products (target/, the build lock).
    work = work / f"mlir_{name}"
    work.mkdir()
    copy = work / f"{name}.mind"
    copy.write_bytes(src.read_bytes())
    out = work / f"{name}.mlir.bin"
    b = subprocess.run([mindc_bin(), "build", str(copy), "--release", "--out", str(out)],
                       capture_output=True, timeout=900, cwd=str(work))
    if b.returncode != 0 or not out.exists():
        return "build-fail"
    return run_exe(out)


def check_case(stage1: pathlib.Path, rust: RustRef, work: pathlib.Path, case) -> tuple[list[str], list[str]]:
    """Run every leg of ONE fixture; return (report lines, failure tags). Pure
    per-fixture state (its own scratch names under `work`), so fixtures can run
    concurrently and still print in CASES order."""
    name, kind, want, want_mlir = case
    lines: list[str] = []
    failures: list[str] = []
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
            refd = work / f"ref_{name}"
            refd.mkdir()
            note = note_of_elf(out)
            ref = rust.seeded_note(src.read_text(), refd)
            nok = note is not None and ref is not None and note == ref
            lines.append(f"[{'PASS' if nok else 'FAIL'}] note {name}: "
                         f"native={note.hex()[:16] if note else None} "
                         f"rust={ref.hex()[:16] if ref else None}")
            if not nok:
                failures.append(f"note:{name}")
        lines.append(f"[{'PASS' if ok else 'FAIL'}] native {name}: {got} want={want!r}")
    else:
        ok = rc != 0 and len(out) == 0
        lines.append(f"[{'PASS' if ok else 'FAIL'}] native refuse {name}: "
                     f"rc={rc} {len(out)}B (want non-zero rc, 0B)")
    if not ok:
        failures.append(f"native:{name}")
    got_mlir = mlir_outcome(src, work, name)
    mok = got_mlir == want_mlir
    lines.append(f"[{'PASS' if mok else 'FAIL'}] mlir {name}: got={got_mlir!r} want={want_mlir!r}")
    if not mok:
        failures.append(f"mlir:{name}")
    return lines, failures


def main() -> int:
    if not FROZEN_SEED.exists():
        print(f"[FAIL] frozen seed missing: {FROZEN_SEED}")
        return 1
    stage1 = build_stage1()
    print(f"stage1 from {main_mind()} -> {stage1.name}")
    failures = []
    rust = RustRef(HERE.parent.parent)
    # Fixtures are independent subprocess work (stage1 compile, run, MLIR build,
    # Rust reference note), so a small pool keeps the gate inside run_gate's
    # default per-gate timeout on a loaded runner. Output stays in CASES order.
    cases = selected_cases()
    jobs = max(1, int(os.environ.get("MISC_JOBS", "3")))
    with tempfile.TemporaryDirectory(prefix="native_misc_") as tdn:
        work = pathlib.Path(tdn)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(check_case, stage1, rust, work, c) for c in cases]
            for fut in futs:
                lines, fails = fut.result()
                for line in lines:
                    print(line, flush=True)
                failures.extend(fails)
    if failures:
        print(f"FAIL ({len(failures)}): {', '.join(failures)}")
        return 1
    print(f"ALL PASS ({len(cases)} fixtures: native vs hand-computed, MLIR vs hand-computed, "
          f"native trace note vs Rust emit_mic3 note)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
