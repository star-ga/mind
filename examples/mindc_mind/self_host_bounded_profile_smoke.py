"""Bounded native memory profile gate: the compiler seed AND its programs load
and run under RLIMIT_AS = 2 GiB.

The default native layout reserves a 1 GiB arena and a 4 GiB stack (5,371,158,528
page-rounded load bytes), so neither the frozen seed nor any image it emits can
even be exec'd under a 2 GiB address-space limit. The bounded profile is the
same compiler emitting a 768 MiB arena + 1 MiB unmapped guard + 384 MiB stack,
selected by the v2 stdin header (selfhost_driver.mind / `nb_layout_check`) or by
`mindc build --backend native --native-profile bounded`.

Checks, in order (each prints PASS/FAIL; any FAIL exits 1):
  1. pin       testdata/selfhost_loop_bounded/{stage1.elf,MANIFEST.txt} agree
  2. layout    the pinned seed's arena/stack PT_LOADs are exactly the bounded layout
  3. derive    the frozen DEFAULT seed, given the v2 header, emits the pinned
               bounded seed byte-identically (one source, one compiler)
  4. loop      under prlimit AS=2 GiB: bounded seed -> stage1 -> stage2 -> stage3,
               all equal to the pinned bounded seed
  5. consumer  under AS=2 GiB + CPU cap: minimal `return 42` and the 8-case
               scalar harness over 4 candidates (3 pass, 1 wrong answer refuses)
  6. refusals  address-space exhaustion -> exit 70 + diagnostic; the fill_0..fill_4
               arena-end boundary grows into the overflow region and returns 7;
               unbounded recursion -> stack guard SIGSEGV; malformed layout request -> exit 64+reason + diagnostic;
               the DEFAULT seed under the cap still dies (the original blocker)
  7. bridge    (when MINDC_BIN is set) `mindc build --backend native
               --native-profile bounded` under AS=2 GiB builds a runnable artifact

Wall time, peak child RSS and the /proc/vmstat oom_kill delta are printed.

Modes:
  (no flag)  verify (all checks above)
  --bless    derive the bounded seed from the frozen default seed (check 3's
             output), require stage1==stage2==stage3 under the cap, and publish
             stage1.elf + MANIFEST.txt; then verify.
"""

import hashlib
import os
import pathlib
import resource
import struct
import subprocess
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).parent.resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from self_host_loop_smoke import build_seed, is_static_elf  # noqa: E402

AS_CAP = 2 * 1024**3
CPU_CAP_S = 12  # consumer compiles and runs (the evaluator's child CPU ceiling)
SELF_CPU_CAP_S = 600  # one whole self-compile (~70 s CPU measured)
LAYOUT_MAGIC = int.from_bytes(b"MINDLAY\x01", "little")
ARENA_ADDR = 0x100_0000
BOUNDED = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard
DEFAULT_SEED = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
BOUNDED_DIR = _HERE / "testdata" / "selfhost_loop_bounded"
BOUNDED_SEED = BOUNDED_DIR / "stage1.elf"
BOUNDED_MANIFEST = BOUNDED_DIR / "MANIFEST.txt"
CONSUMER_HARNESS = """\
// Fixed finite scalar correctness cases (consumer evaluator shape).
fn main() -> i64 {
    if candidate(0 - 100) != 9801 { return 1; }
    if candidate(0 - 2) != 1 { return 1; }
    if candidate(0 - 1) != 0 { return 1; }
    if candidate(0) != 1 { return 1; }
    if candidate(1) != 4 { return 1; }
    if candidate(2) != 9 { return 1; }
    if candidate(17) != 324 { return 1; }
    if candidate(100) != 10201 { return 1; }
    return 0;
}
"""
CANDIDATES = {
    "baseline": ("fn add_one(x: i64) -> i64 { return x + 1; }\n"
                 "fn candidate(x: i64) -> i64 {\n"
                 "    let y: i64 = add_one(x);\n    return y * y;\n}\n", 0),
    "keep": ("fn candidate(x: i64) -> i64 {\n"
             "    let y: i64 = x + 1;\n    return y * y;\n}\n", 0),
    "discard": ("fn add_one(x: i64) -> i64 { return x + 1; }\n"
                "fn square(x: i64) -> i64 { return x * x; }\n"
                "fn candidate(x: i64) -> i64 { return square(add_one(x)); }\n", 0),
    "wrong": ("fn candidate(x: i64) -> i64 { return 0; }\n", 1),
}
FILL_DIAG = b"error: native arena exhausted\n"
FAILS: list[str] = []


def fill_source(k: int) -> bytes:
    """Fill the bounded arena to 8*k bytes short of its end, then alloc(0) and
    alloc(8). The arena's first 8 bytes hold the bump cursor; each block is an
    8-byte header + its rounded payload."""
    first = BOUNDED[0] - 16 - 8 * k
    return (f"fn main() -> i64 {{\n"
            f"    let a: i64 = __mind_alloc({first});\n"
            f"    let b: i64 = __mind_alloc(0);\n"
            f"    let c: i64 = __mind_alloc(8);\n"
            f"    __mind_store_i64(c, 7);\n"
            f"    return __mind_load_i64(c);\n"
            f"}}\n").encode()


def report(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def oom_kills() -> int:
    for row in pathlib.Path("/proc/vmstat").read_text().splitlines():
        if row.startswith("oom_kill "):
            return int(row.split()[1])
    return -1


def v2_header(layout=BOUNDED) -> bytes:
    return struct.pack("<qqqq", LAYOUT_MAGIC, *layout)


def user_image(user: bytes, layout=BOUNDED) -> bytes:
    """[v2 hdr][user_lo][src_len][std ++ user] with the seeded std blob."""
    combined, _, user_lo = build_seed()
    std = combined[:user_lo]
    comb = std + user
    hdr = v2_header(layout) if layout else b""
    return hdr + struct.pack("<qq", len(std), len(comb)) + comb


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


def want_segments(layout=BOUNDED) -> list[tuple[int, int]]:
    arena, stack, guard = layout
    return [(ARENA_ADDR, arena), (ARENA_ADDR + arena + guard, stack)]


def capped(argv, stdin: bytes, *, cap=True, cpu=CPU_CAP_S, timeout=300):
    """Run under RLIMIT_AS=2 GiB + RLIMIT_CPU=`cpu` s; returns
    (rc, out, err, wall, peak_child_rss_kib)."""
    def limit() -> None:
        if cap:
            resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    t0 = time.monotonic()
    r = subprocess.run(argv, input=stdin, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout, preexec_fn=limit)
    wall = time.monotonic() - t0
    return r.returncode, r.stdout, r.stderr, wall, resource.getrusage(
        resource.RUSAGE_CHILDREN).ru_maxrss


def write_exec(d: pathlib.Path, name: str, data: bytes) -> pathlib.Path:
    p = d / name
    p.write_bytes(data)
    p.chmod(0o755)
    return p


def derive_from_default(stdin_v2: bytes) -> bytes:
    """Frozen DEFAULT seed + v2 header -> bounded compiler (uncapped: the
    default seed itself reserves 5 GiB)."""
    rc, out, err, wall, _ = capped([str(DEFAULT_SEED)], stdin_v2, cap=False)
    if rc != 0 or not is_static_elf(out):
        raise RuntimeError(f"default seed exited {rc}, {len(out)}B, err={err[:200]!r}")
    print(f"        derive: {len(out)} B in {wall:.1f}s sha256={sha(out)}")
    return out


def loop(seed: pathlib.Path, stdin_v2: bytes, td: pathlib.Path) -> list[bytes]:
    stages, exe = [], seed
    for k in (1, 2, 3):
        rc, out, err, wall, rss = capped([str(exe)], stdin_v2, cpu=SELF_CPU_CAP_S)
        print(f"        stage{k}: rc={rc} {len(out)} B wall={wall:.1f}s "
              f"peak_child_rss={rss} KiB sha256={sha(out)[:16]}")
        if rc != 0 or not is_static_elf(out):
            raise RuntimeError(f"stage{k} exited {rc} under AS cap: {err[:200]!r}")
        stages.append(out)
        exe = write_exec(td, f"stage{k}.elf", out)
    return stages


def manifest_text(seed: bytes, receipt: dict[str, str]) -> str:
    lines = [
        "# Frozen BOUNDED-profile self-host compiler seed: the pure-MIND compiler",
        "# emitted with an explicit layout (arena/stack/guard below), so it and every",
        "# image it emits load under RLIMIT_AS = 2 GiB. Derived from the frozen",
        "# default seed (testdata/selfhost_loop/stage1.elf) with the v2 header, and",
        "# reproduces itself (stage1==stage2==stage3) under the cap.",
        "# published-by\tself_host_bounded_profile_smoke.py --bless",
        "#",
    ]
    lines += [f"# {k}\t{v}" for k, v in receipt.items()]
    lines += ["#", "# name\tsize_bytes\tsha256", f"stage1.elf\t{len(seed)}\t{sha(seed)}"]
    return "\n".join(lines) + "\n"


def bless(stdin_v2: bytes, combined: bytes) -> int:
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        s1 = derive_from_default(stdin_v2)
        p1 = write_exec(td, "derived.elf", s1)
        stages = loop(p1, stdin_v2, td)
        if not all(s == s1 for s in stages):
            raise SystemExit("BLOCKED: derived bounded seed is not a fixed point under the cap")
    if BOUNDED_SEED.exists() and BOUNDED_SEED.read_bytes() == s1:
        print("BLOCKED: --bless would republish the current bounded seed unchanged; "
              "nothing written.")
        return 2
    BOUNDED_DIR.mkdir(exist_ok=True)
    receipt = {
        "derived_from_seed_sha256": sha(DEFAULT_SEED.read_bytes()),
        "combined_source_sha256": sha(combined),
        "combined_source_bytes": str(len(combined)),
        "layout_arena_bytes": str(BOUNDED[0]),
        "layout_stack_bytes": str(BOUNDED[1]),
        "layout_guard_bytes": str(BOUNDED[2]),
        "stage1_sha256": sha(s1),
        "stage2_sha256": sha(stages[1]),
        "stage3_sha256": sha(stages[2]),
    }
    BOUNDED_SEED.write_bytes(s1)
    BOUNDED_SEED.chmod(0o755)
    BOUNDED_MANIFEST.write_text(manifest_text(s1, receipt))
    print(f"  BLESSED  {BOUNDED_SEED.relative_to(_REPO)} sha256={sha(s1)}")
    return 0


def main(argv: list[str]) -> int:
    combined, stdin_v1, user_lo = build_seed()
    stdin_v2 = v2_header() + stdin_v1
    oom0 = oom_kills()
    t_start = time.monotonic()
    if "--bless" in argv and bless(stdin_v2, combined) != 0:
        return 2

    print("bounded native profile gate "
          f"(arena={BOUNDED[0]} stack={BOUNDED[1]} guard={BOUNDED[2]}, AS cap={AS_CAP})")
    if not BOUNDED_SEED.exists():
        print("  BLOCKED  no pinned bounded seed; run with --bless")
        return 2
    seed = BOUNDED_SEED.read_bytes()

    # 1. pin
    rows = [r.split("\t") for r in BOUNDED_MANIFEST.read_text().splitlines()
            if r and not r.startswith("#")]
    report(rows == [["stage1.elf", str(len(seed)), sha(seed)]], "pin",
           f"stage1.elf {len(seed)} B sha256={sha(seed)}")

    # 2. layout
    got = rw_loads(seed)
    report(got == want_segments(), "layout", f"rw PT_LOADs {got}")

    # 3. derive
    derived = derive_from_default(stdin_v2)
    report(derived == seed, "derive", "frozen default seed + v2 header == pinned bounded seed")

    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        # 4. loop under the cap
        try:
            stages = loop(BOUNDED_SEED, stdin_v2, td)
            report(all(s == seed for s in stages), "loop",
                   "stage1==stage2==stage3==pinned under RLIMIT_AS=2 GiB")
        except RuntimeError as e:
            report(False, "loop", str(e))

        # 5. consumer programs under the cap
        def build_run(name: str, src: bytes, layout=BOUNDED, seed_path=BOUNDED_SEED):
            rc, out, err, wall, rss = capped([str(seed_path)], user_image(src, layout))
            if rc != 0 or not out.startswith(b"\x7fELF"):
                return rc, None, err, wall, rss
            exe = write_exec(td, name, out)
            return rc, exe, err, wall, rss

        rc, exe, err, wall, rss = build_run("ret42", b"fn main() -> i64 { return 42; }\n")
        run = capped([str(exe)], b"") if exe else (None,) * 5
        report(exe is not None and run[0] == 42 and rw_loads(exe.read_bytes()) == want_segments(),
               "consumer/minimal", f"compile rc={rc} wall={wall:.2f}s; run rc={run[0]}")
        for name, (cand, want_rc) in CANDIDATES.items():
            src = cand.encode() + b"\n" + CONSUMER_HARNESS.encode()
            rc, exe, err, wall, rss = build_run(name, src)
            run = capped([str(exe)], b"") if exe else (None,) * 5
            size = exe.stat().st_size if exe else 0
            report(exe is not None and run[0] == want_rc, f"consumer/{name}",
                   f"compile rc={rc} wall={wall:.2f}s elf={size} B; harness exit={run[0]} "
                   f"(want {want_rc})")

        # 6. refusals
        big = b"fn main() -> i64 {\n    let p: i64 = __mind_alloc(1500000000);\n    return 3;\n}\n"
        rc, exe, err, wall, rss = build_run("bigalloc", big)
        run = capped([str(exe)], b"") if exe else (None, b"", b"", 0, 0)
        report(run[0] == 70 and run[2] == FILL_DIAG,
               "refusal/arena", f"exit={run[0]} stderr={run[2]!r}")
        # Arena-cap boundary: one block sized to end within 8*k bytes of the
        # arena end, then alloc(0) + alloc(8). The bounded allocator reserves
        # the arena's last page for its state and grows on demand: a block that
        # does not fit the primary region moves the cursor to the overflow
        # region above the stack (mapped with MAP_FIXED_NOREPLACE), alloc(0)
        # returns 0 (C-runtime parity), and alloc(8) lands next to it, so every
        # k returns the stored 7 with no write into the guard gap. The
        # address-space ceiling itself is the bigalloc case above (exit 70).
        for k in range(5):
            want = (7, b"")
            rc, exe, err, wall, rss = build_run(f"fill_{k}", fill_source(k))
            run = capped([str(exe)], b"") if exe else (None, b"", b"", 0, 0)
            report((run[0], run[2]) == want, f"refusal/fill_{k}",
                   f"compile rc={rc}; exit={run[0]} stderr={run[2]!r} (want {want[0]})")
        deep = (b"fn f(n: i64) -> i64 {\n    if n == 0 {\n        return 0;\n    }\n"
                b"    return 1 + f(n - 1);\n}\nfn main() -> i64 {\n    return f(100000000);\n}\n")
        rc, exe, err, wall, rss = build_run("deeprec", deep)
        run = capped([str(exe)], b"") if exe else (None,) * 5
        report(run[0] == -11, "refusal/stack-guard", f"exit={run[0]} (SIGSEGV at the guard)")
        bad = (BOUNDED[0] + 1, BOUNDED[1], BOUNDED[2])
        rc, out, err, wall, rss = capped([str(BOUNDED_SEED)], user_image(b"fn main() -> i64 { return 1; }\n", bad))
        report(rc == 64 + 7 and out == b"" and err.startswith(b"error[native-layout]"),
               "refusal/layout", f"exit={rc} stdout={len(out)} B stderr={err[:40]!r}")
        rc, out, err, wall, rss = capped([str(DEFAULT_SEED)], user_image(b"fn main() -> i64 { return 1; }\n", None))
        report(rc < 0 and out == b"", "refusal/default-under-cap",
               f"default seed exit={rc} (cannot load under the cap: the original blocker)")

        # 7. bridge
        mindc = os.environ.get("MINDC_BIN")
        if mindc:
            src = td / "main.mind"
            src.write_bytes(CANDIDATES["keep"][0].encode() + b"\n" + CONSUMER_HARNESS.encode())
            art = td / "candidate.elf"
            env = {"PATH": "/nonexistent", "LANG": "C", "LC_ALL": "C",
                   "MINDC_STD_DIR": str(_REPO / "std")}
            def limit() -> None:
                resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
            r = subprocess.run([str(pathlib.Path(mindc).resolve()), "build", str(src),
                                "--backend", "native", "--native-profile", "bounded",
                                "--emit", "binary", "--out", str(art)],
                               cwd=_REPO, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=120, preexec_fn=limit)
            run = capped([str(art)], b"") if art.exists() else (None,) * 5
            report(r.returncode == 0 and run[0] == 0
                   and rw_loads(art.read_bytes()) == want_segments(),
                   "bridge", f"mindc rc={r.returncode} artifact run rc={run[0]} "
                   f"stderr={r.stderr.decode(errors='replace').strip()[-160:]!r}")
        else:
            print("  SKIP  bridge: MINDC_BIN not set")

    oom1 = oom_kills()
    report(oom1 == oom0, "oom", f"/proc/vmstat oom_kill {oom0} -> {oom1}")
    print(f"  wall={time.monotonic() - t_start:.1f}s peak_child_rss="
          f"{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss} KiB")
    if FAILS:
        print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
