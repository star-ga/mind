"""Bounded-profile allocator gate (W11c): grow-on-demand arena, C-runtime
alloc/realloc/free parity, and a deterministic out-of-memory exit, all under
RLIMIT_AS = 2 GiB.

The bounded profile (v2 stdin header: 768 MiB arena + 1 MiB guard + 384 MiB
stack) loads under a 2 GiB address-space limit, but its bump arena used to be
a hard 768 MiB ceiling with a no-op free. This gate builds the CURRENT tree's
compiler in the bounded profile, entirely natively:

  B1 = frozen default seed (testdata/selfhost_loop/stage1.elf) + v2 + tree source
  B2 = B1 + v2 + tree source           (under the cap; emitted by the tree's emitter)
  B3 = B2 + v2 + tree source           (under the cap; B3 == B2 required)

so B2 is the tree's emitter running in an image that uses the tree's allocator
(free-list reuse and overflow growth included); B3 == B2 proves that allocator
does not change what the compiler emits. B2 then compiles every fixture in
testdata/native_alloc_bounded/ under the cap and each emitted program runs
under the cap. Expected exit codes and stdout come from hand computation (in
each fixture's header comment, restated in CASES). When MINDC_BIN is set each
portable fixture is also built through the MLIR backend and run under the same
cap, as an emitter-independent oracle.

Peak VmPeak (sampled from /proc/<pid>/status) and ru_maxrss (wait4) are
printed for every compile and run.

Flags:
  --compiler PATH  skip the derivation and use PATH as the bounded compiler
                   (e.g. the pinned testdata/selfhost_loop_bounded/stage1.elf)
  --keep DIR       copy the derived B1/B2/B3 compilers into DIR
Exit 0 = ALL PASS, 1 = a FAIL, 2 = BLOCKED (missing fixture/seed).
"""

import hashlib
import os
import pathlib
import resource
import struct
import subprocess
import sys
import tempfile
import threading
import time

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from self_host_loop_smoke import build_seed, is_static_elf  # noqa: E402

AS_CAP = 2 * 1024**3
LAYOUT_MAGIC = int.from_bytes(b"MINDLAY\x01", "little")
BOUNDED = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard
DEFAULT_SEED = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
FIX = _HERE / "testdata" / "native_alloc_bounded"
OOM_DIAG = b"error: native arena exhausted\n"
# name -> (exit, stdout, portable: also checked against the MLIR backend)
CASES = {
    "r18_free_loop": (33, b"", True),
    "alloc_nonpos": (40, b"np\n", True),
    "grow_big": (30, b"big\n", True),
    "vec_grow": (7, b"vec\n", True),
    "vec_std": (9, b"std\n", True),
    "oom_exit": (70, b"", False),
}
FAILS: list[str] = []


def report(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(argv, stdin: bytes, *, cap=True, timeout=900):
    """Run a child (nice 15, RLIMIT_AS=2 GiB when `cap`). Returns
    (rc, stdout, stderr, wall_s, vmpeak_kib, maxrss_kib)."""
    def limit() -> None:
        os.nice(15)
        if cap:
            resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
    t0 = time.monotonic()
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, preexec_fn=limit)
    peak = [0]
    stop = threading.Event()

    def sample() -> None:
        status = pathlib.Path(f"/proc/{p.pid}/status")
        while not stop.is_set():
            try:
                for row in status.read_text().splitlines():
                    if row.startswith("VmPeak:"):
                        peak[0] = max(peak[0], int(row.split()[1]))
            except (OSError, ValueError):
                pass
            stop.wait(0.02)
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, err = [], []
    readers = [threading.Thread(target=lambda: out.append(p.stdout.read())),
               threading.Thread(target=lambda: err.append(p.stderr.read()))]
    for r in readers:
        r.start()
    try:
        p.stdin.write(stdin)
    except BrokenPipeError:
        pass
    p.stdin.close()
    deadline = time.monotonic() + timeout
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            break
        if time.monotonic() > deadline:
            p.kill()
            pid, status, ru = os.wait4(p.pid, 0)
            break
        time.sleep(0.01)
    p.returncode = os.waitstatus_to_exitcode(status)
    stop.set()
    for r in readers:
        r.join()
    return (p.returncode, out[0], err[0], time.monotonic() - t0, peak[0], ru.ru_maxrss)


def v2_image(user: bytes | None) -> bytes:
    """[v2 hdr][user_lo][src_len][combined]; `None` = the whole compiler source."""
    combined, _, user_lo = build_seed()
    if user is not None:
        combined = combined[:user_lo] + user
    return (struct.pack("<qqqq", LAYOUT_MAGIC, *BOUNDED)
            + struct.pack("<qq", user_lo, len(combined)) + combined)


def rw_loads(elf: bytes) -> list[tuple[int, int]]:
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", elf, 54)
    out = []
    for i in range(phnum):
        ptype, flags, _, vaddr, _, _, memsz, _ = struct.unpack_from(
            "<IIQQQQQQ", elf, phoff + i * phentsize)
        if ptype == 1 and flags & 2:
            out.append((vaddr, memsz))
    return out


def write_exec(d: pathlib.Path, name: str, data: bytes) -> pathlib.Path:
    p = d / name
    p.write_bytes(data)
    p.chmod(0o755)
    return p


def describe(tag: str, r) -> str:
    rc, out, err, wall, vmpeak, rss = r
    return (f"{tag} rc={rc} out={len(out)}B wall={wall:.1f}s "
            f"VmPeak={vmpeak} KiB maxrss={rss} KiB")


def derive(td: pathlib.Path) -> pathlib.Path | None:
    img = v2_image(None)
    r = run([str(DEFAULT_SEED)], img, cap=False)
    print("        " + describe("B1 (frozen default seed + v2, uncapped)", r), flush=True)
    if r[0] != 0 or not is_static_elf(r[1]):
        report(False, "derive/B1", f"rc={r[0]} err={r[2][-200:]!r}")
        return None
    exe, stages, peaks = write_exec(td, "B1.elf", r[1]), [], []
    for k in (2, 3):
        r = run([str(exe)], img)
        peaks.append(r[4])
        print("        " + describe(f"B{k} (under AS cap)", r) + f" sha256={sha(r[1])[:16]}",
              flush=True)
        if r[0] != 0 or not is_static_elf(r[1]):
            report(False, f"derive/B{k}", f"rc={r[0]} err={r[2][-200:]!r}")
            return None
        stages.append(r[1])
        exe = write_exec(td, f"B{k}.elf", r[1])
    want = [(0x100_0000, BOUNDED[0]), (0x100_0000 + BOUNDED[0] + BOUNDED[2], BOUNDED[1])]
    report(stages[0] == stages[1], "derive/fixed-point",
           f"B2 == B3 under RLIMIT_AS=2 GiB ({len(stages[0])} B sha256={sha(stages[0])[:16]})")
    report(rw_loads(stages[0]) == want, "derive/layout", f"rw PT_LOADs {rw_loads(stages[0])}")
    # B2 is emitted by B1, whose image still runs the historical allocator; B3
    # is emitted by B2, which runs the bounded allocator. The self-compile must
    # not need more address space under the new allocator (it may not even
    # spill into the overflow region: the arena checkpoints keep reclaiming).
    report(0 < peaks[1] <= peaks[0] + 65536, "derive/self-compile-footprint",
           f"VmPeak B2-run {peaks[0]} KiB, B3-run {peaks[1]} KiB (limit B2-run + 64 MiB)")
    return td / "B2.elf"


def mlir_oracle(src: pathlib.Path, td: pathlib.Path):
    mindc = os.environ.get("MINDC_BIN")
    if not mindc:
        return None
    # Build an isolated copy: a fixture inside the repo would resolve the repo's
    # Mind.toml project and link every sibling fixture's object with it.
    proj = td / f"mlir_{src.stem}"
    proj.mkdir()
    (proj / src.name).write_bytes(src.read_bytes())
    art = proj / (src.stem + ".mlir.elf")
    r = subprocess.run([str(pathlib.Path(mindc).resolve()), "build", src.name,
                        "--emit", "binary", "--out", str(art)],
                       cwd=proj, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if r.returncode != 0 or not art.exists():
        return ("build-failed", r.returncode, r.stderr.decode(errors="replace")[-200:])
    return run([str(art)], b"")


def main(argv: list[str]) -> int:
    print(f"native bounded allocator gate (arena={BOUNDED[0]} stack={BOUNDED[1]} "
          f"guard={BOUNDED[2]}, AS cap={AS_CAP})", flush=True)
    for name in CASES:
        if not (FIX / f"{name}.mind").exists():
            print(f"  BLOCKED  missing fixture {name}.mind")
            return 2
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        if "--compiler" in argv:
            comp = pathlib.Path(argv[argv.index("--compiler") + 1]).resolve()
            print(f"        compiler (given): {comp} sha256={sha(comp.read_bytes())[:16]}")
        else:
            if not DEFAULT_SEED.exists():
                print("  BLOCKED  no frozen default seed")
                return 2
            comp = derive(td)
            if comp is None:
                print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}")
                return 1
            if "--keep" in argv:
                keep = pathlib.Path(argv[argv.index("--keep") + 1])
                keep.mkdir(parents=True, exist_ok=True)
                for k in (1, 2, 3):
                    write_exec(keep, f"B{k}.elf", (td / f"B{k}.elf").read_bytes())
        for name, (want_rc, want_out, portable) in CASES.items():
            src = FIX / f"{name}.mind"
            c = run([str(comp)], v2_image(src.read_bytes()))
            if c[0] != 0 or not c[1].startswith(b"\x7fELF"):
                report(False, f"case/{name}", describe("compile", c) + f" err={c[2][-160:]!r}")
                continue
            exe = write_exec(td, f"{name}.elf", c[1])
            r = run([str(exe)], b"")
            want_err = OOM_DIAG if want_rc == 70 else b""
            ok = (r[0], r[1], r[2]) == (want_rc, want_out, want_err)
            detail = (describe("compile", c) + "; " + describe("run", r)
                      + f" stdout={r[1]!r} stderr={r[2][-60:]!r} (want rc={want_rc} "
                      f"stdout={want_out!r})")
            if name == "oom_exit" and ok:
                r2 = run([str(exe)], b"")
                ok = (r2[0], r2[1], r2[2]) == (r[0], r[1], r[2])
                detail += f"; rerun rc={r2[0]} (deterministic={ok})"
            report(ok, f"case/{name}", detail)
            if portable:
                m = mlir_oracle(src, td)
                if m is None:
                    print(f"  SKIP  mlir/{name}: MINDC_BIN not set")
                elif m[0] == "build-failed":
                    report(False, f"mlir/{name}", f"MLIR build rc={m[1]} {m[2]!r}")
                else:
                    report((m[0], m[1]) == (want_rc, want_out), f"mlir/{name}",
                           describe("MLIR run", m) + f" stdout={m[1]!r}")
    if FAILS:
        print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
