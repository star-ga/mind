#!/usr/bin/env python3
"""Native stage1: general control flow (HOME lowering) — three-way behavioural gate.

What it proves
--------------
The stage1 compiler built from THIS tree's main.mind compiles arbitrary nesting
of if / else / else-if / while / break / continue / return and value-if, with
`&&` / `||` / `!` short-circuit, in functions with and without parameters, and
the programs it emits behave exactly like the MLIR backend and like an
independent Python interpreter of the same program (exit status AND stdout).

Before this change stage1 refused every such program (rc 3): its PT_NOTE is the
IR trace hash, computed by the pure-MIND mic@3 twin, and the twin covers fewer
function shapes than the native emitter. selftest_native_elf_stage now keeps
the IR-hash note when the twin can compute it and otherwise lowers exactly the
functions the twin rejected on the HOME path (main.mind: "HOME-SLOT LOWERING"),
marking the note as a SOURCE note (n_type 2). See the census rules A1 A6 A7 A9
L1 L2 L3 O1 O2 O3 in the fixtures.

Legs (each a separate, non-shared implementation):
  native  stage1 ELF derived from this tree (frozen seed -> tree source), fed
          the program through the stdin image protocol, then run.
  MLIR    `mindc build --backend mlir` of the same program, then run.
  Python  _cf_corpus.Interp over the generator's own AST (corpus), or the
          hand-derived `// expect:` header (fixtures).

Steps
  0. harness   the gate's own exec helpers (run_bin, the bounded run) survive
               ETXTBSY: exec of an ELF while another process holds a write
               descriptor on it (deterministic), and 600 pooled write+exec cases
               with forking siblings (the thread-pool race). Every exec goes
               through exec_run, which retries ETXTBSY for up to 5 s.
               `--harness-only` runs just this step (no compiler needed).
  1. fixtures  testdata/native_cf/*.mind. Header keys:
               `// expect: exit=N`, `// stdout: TEXT` (hand-derived),
               `// note: 1|2` (required PT_NOTE n_type),
               `// mlir: nobuild` (pinned MLIR limitation: must not build),
               `// mlir: nobuild=TEXT` (pinned MLIR build failure whose
               error tail contains TEXT),
               `// mlir: exit=N` (pinned MLIR defect: MLIR must give N).
  2. corpus    deterministic random programs (default seeds 11,12 x 160 = 320,
               fed unseeded: they call no std function);
               native == Python == MLIR. An MLIR build failure is accepted only
               for the pinned "break/continue outside a loop" limitation.
               A second, "shadow" profile (default seed 13 x 80) lets half of
               the new bindings re-use a visible name, so lets shadow outer
               variables and parameters inside loop bodies, if branches and
               value-if branches; native == Python strictly, and MLIR may
               differ only where a loop body re-binds a name (pinned:
               mlir_shadow_exit_carry.mind).
  3. capacity  census X1, every case success (== Python) or a clean rc-3
               refusal with a diagnostic and no ELF; never a signal, never the
               arena trap (exit 70); generators in _cf_giant.py. Default
               profile: a 2500-arm else-if chain in a loop, 800/2800-statement
               let/if-return functions, 520/300/600 reassignments past the SSA
               let-env. Bounded production profile (the frozen seed + v2 header:
               768 MiB arena, 384 MiB stack; RLIMIT_AS 2 GiB like the MIND-Lab
               consumer harness): dense if/else trees (2048 leaves, 1024 + 2600
               comment lines); giant FLAT statement lists in every statement
               position (body, then, else, while, value-if, calls, if sequence,
               chained / shadowing lets, let/assign pairs, three-term `&&` and
               `||` conditions) at a mid size and at the token-budget boundary,
               under RLIMIT_CPU (`--flat-only`); DEEP trees (40000-term chains,
               parentheses, calls, `as`, unary minus, `&&` conditions, nested
               ifs / value-ifs / loops; `--deep-only`), also through the bounded
               compiler fed a v1 image and the default compiler; whole-program
               CAPACITY (`--capacity-only`): 8100 small functions and 10000 x
               100-let functions (27 MB), whose SSA tables / token stream ran
               the arena out (exit 70); a 1 MB string literal and 4 MB
               identifier, digit run and comment line (the lexer walked a token
               one stack frame per byte: SIGSEGV); a balanced 65536-leaf `+`
               tree in one expression (the mic@3 descriptor walk took a frame per
               node: SIGSEGV); smaller twins of each must compile. Whole-program
               extremes (a 65536-leaf 18.8 MB tree, a 21 MB refused program, a
               1.96 MB unseeded tree) in both profiles and with the unseeded
               image.
  4. refusal   a HOME-lowered function that takes an address refuses cleanly
               (rc 3, no ELF), never a crash.
  5. twin      with MINDC_SO set, the oracle .so's selftest_native_elf_stage
               emits the SAME bytes as stage1 for every fixture.
  6. oracle    (_cf_oracle.py; `--oracle-only`) every __mind_free in main.mind
               frees a __mind_alloc result; with MINDC_SO, the .so runs every
               fixture and a corpus slice, and emits this tree's own combined
               source (the self-host loop's oracle leg), under glibc's free
               checking (glibc.malloc.check=3): the self-emit must equal stage1.
               A free of an nb_ntab table's interior pointer aborted the .so on
               its own source ("double free or corruption") while every
               native gate stayed green (the arena ignores frees).

Usage:
  MINDC_BIN=target/release/mindc [MINDC_SO=/tmp/oracle.so] \\
    python3 self_host_native_cf_smoke.py [--stage1 PATH] [--stage1-bounded PATH] [--seeds 11,12]
Without --stage1 / --stage1-bounded each compiler is derived from the frozen seed
and this tree (two whole self-compiles).
Exit: 0 ALL PASS, 1 FAIL, 2 BLOCKED (no MLIR-capable mindc / no seed).
"""

import argparse
import ctypes
import errno
import os
import pathlib
import re
import resource
import signal
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
import _cf_corpus as C  # noqa: E402
import _cf_oracle as O  # noqa: E402
from _cf_giant import (  # noqa: E402
    FLAT_CPU,
    REFUSE_ADDR,
    bounded_cases,
    capacity_cases,
    deep_cases,
    deep_default_cases,
    deep_v1_cases,
    default_capacity_cases,
    extreme_cases,
    flat_cases,
    flat_matrix_cases,
    giant_program,
    reassign_program,
    straight_line_program,
)

STD_MODULES = [
    "arena", "async", "blas", "cli", "fs", "io", "io_canon", "iouring",
    "json", "map", "net", "process", "reactor", "regex", "ring", "sha256",
    "string", "time", "toml", "tui", "vec",
]
SEED = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
FIXTURES = _HERE / "testdata" / "native_cf"
MLIR_LIMIT = "break/continue outside a loop"


def std_blob() -> bytes:
    return b"\n".join((_REPO / "std" / f"{m}.mind").read_bytes() for m in STD_MODULES) + b"\n"


BLOB = std_blob()


def image(src: bytes, seeded: bool = True) -> bytes:
    """stdin image: [user_lo][len][std blob ++ program]. The generated corpus
    calls no std function, so it is fed UNSEEDED (user_lo = 0, every function a
    root) to keep the gate inside run_gate's time budget; every fixture and
    capacity case takes the production seeded image."""
    blob = BLOB if seeded else b""
    comb = blob + src
    return struct.pack("<qq", len(blob), len(comb)) + comb


def nice():
    os.nice(15)


ETXTBSY_TRIES, ETXTBSY_WAIT = 250, 0.02  # up to 5 s for a writer to let go
_etxtbsy_retries = [0]


def exec_run(argv, **kw):
    """subprocess.run for every exec this gate performs, retried while the target
    is busy (ETXTBSY). The gate writes ELFs (stage1, the bounded compiler, every
    program) and execs them from a thread pool; a sibling thread's fork holds a
    copy of a just-written file's write descriptor until its child execs, and an
    exec of that file in the window fails with ETXTBSY. The window closes within
    milliseconds, so the exec is retried; any other OSError, or a writer that
    never lets go, still fails the gate."""
    for _ in range(ETXTBSY_TRIES - 1):
        try:
            return subprocess.run(argv, **kw)
        except OSError as e:
            if e.errno != errno.ETXTBSY:
                raise
            _etxtbsy_retries[0] += 1
            time.sleep(ETXTBSY_WAIT)
    return subprocess.run(argv, **kw)


def derive_stage1(td: pathlib.Path) -> pathlib.Path:
    comb = BLOB + (_HERE / "main.mind").read_bytes() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    img = struct.pack("<qq", len(BLOB), len(comb)) + comb
    r = exec_run([str(SEED)], input=img, capture_output=True, timeout=1800, preexec_fn=nice)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise SystemExit(f"BLOCKED: frozen seed could not compile this tree (rc={r.returncode}): "
                         f"{r.stderr[-300:]!r}")
    out = td / "stage1.elf"
    out.write_bytes(r.stdout)
    out.chmod(0o755)
    return out


from _cf_giant import BOUNDED  # noqa: E402  (arena, stack, guard)
CAP_CPU = 600  # CPU seconds per whole-program capacity compile
AS_CAP = 2 << 30  # the MIND-Lab consumer harness runs every compile and program under this


def v2_header() -> bytes:
    magic = int.from_bytes(b"MINDLAY\x01", "little", signed=True)
    return b"".join(int(v).to_bytes(8, "little", signed=True) for v in (magic, *BOUNDED))


def derive_stage1_bounded(td: pathlib.Path) -> pathlib.Path:
    """The bounded production compiler: the frozen DEFAULT seed given the v2 header
    emits this tree's compiler with the bounded layout (768 MiB arena, 384 MiB stack)."""
    comb = BLOB + (_HERE / "main.mind").read_bytes() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    img = v2_header() + struct.pack("<qq", len(BLOB), len(comb)) + comb
    r = exec_run([str(SEED)], input=img, capture_output=True, timeout=1800, preexec_fn=nice)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise SystemExit(f"BLOCKED: frozen seed could not emit the bounded compiler (rc={r.returncode}): "
                         f"{r.stderr[-300:]!r}")
    out = td / "stage1_bounded.elf"
    out.write_bytes(r.stdout)
    out.chmod(0o755)
    return out


def capped():
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def refusal_result(rc: int, out: bytes, err: str):
    """Only a diagnosed rc-3 exit with no emitted bytes is a clean refusal."""
    if rc == 3 and (out or not err.strip()):
        return ("invalid-refusal", rc, f"stdout={len(out)} B stderr={err.strip()[-120:]!r}")
    return ("refused", rc, err.strip()[-120:])


def native_bounded(stage1b, src: bytes, td: pathlib.Path, seeded: bool = True, v2: bool = True,
                   cpu: int = 0):
    """Compile with the bounded compiler (v2 header, RLIMIT_AS 2 GiB) and run the
    program under the same cap. ('crash', signal) for a compiler killed by a
    signal, ('refused', rc, stderr tail) for a compiler that wrote no ELF.
    v2=False feeds a default-layout (v1) image instead of the bounded request.
    cpu > 0 also caps the compile at that many CPU seconds (RLIMIT_CPU, so a
    loaded box does not trip it): ('compile-cpu-budget', cpu) past it."""
    img = (v2_header() if v2 else b"") + image(src, seeded)

    def capped_cpu():
        capped()
        if cpu > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 10))
    try:
        r = exec_run([str(stage1b)], input=img, capture_output=True, timeout=900, preexec_fn=capped_cpu)
    except subprocess.TimeoutExpired:
        return ("compile-timeout", 900)
    if r.returncode in (-signal.SIGXCPU, -signal.SIGKILL) and cpu > 0:
        return ("compile-cpu-budget", cpu)
    if r.returncode < 0:
        return ("crash", -r.returncode)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        return refusal_result(r.returncode, r.stdout, r.stderr.decode(errors="replace"))
    p = td / "b.elf"
    p.write_bytes(r.stdout)
    p.chmod(0o755)
    return run_bin(p, timeout=120, preexec_fn=capped)


def native_compile(stage1: pathlib.Path, src: bytes, seeded: bool = True):
    r = exec_run([str(stage1)], input=image(src, seeded), capture_output=True, timeout=600,
                       preexec_fn=nice)
    ok = r.returncode == 0 and r.stdout[:4] == b"\x7fELF"
    return ok, r.returncode, r.stdout, r.stderr.decode(errors="replace")


def run_bin(path: pathlib.Path, timeout: int = 60, preexec_fn=None):
    try:
        r = exec_run([str(path)], capture_output=True, timeout=timeout, preexec_fn=preexec_fn)
    except subprocess.TimeoutExpired:
        return ("timeout", b"")
    return (r.returncode & 255 if r.returncode >= 0 else r.returncode, r.stdout)


def native(stage1, src: bytes, td: pathlib.Path, seeded: bool = True):
    ok, rc, elf, err = native_compile(stage1, src, seeded)
    if not ok:
        return refusal_result(rc, elf, err), None
    p = td / "n.elf"
    p.write_bytes(elf)
    p.chmod(0o755)
    return run_bin(p), elf


def mlir(mindc: str, src: str, td: pathlib.Path):
    p = td / "p.mind"
    p.write_text(src)
    o = td / "m.elf"
    r = exec_run([mindc, "build", str(p), "--backend", "mlir", "--emit", "binary", "--out", str(o)],
                       capture_output=True, timeout=600, preexec_fn=nice, cwd=str(td))
    if r.returncode != 0 or not o.exists():
        return ("nobuild", r.stderr.decode(errors="replace").strip()[-200:])
    return run_bin(o)


def header(src: str, key: str):
    m = re.search(r"^// " + key + r":[ ]?(.*)$", src, re.M)
    return None if m is None else m.group(1).strip()


class Report:
    def __init__(self):
        self.fails = []
        self.passed = 0

    def check(self, cond: bool, msg: str) -> bool:
        """One verdict line per checked case."""
        if cond:
            self.passed += 1
            print(f"  PASS  {msg}")
        else:
            print(f"  FAIL  {msg}")
            self.fails.append(msg)
        return cond


def check_fixture(f: pathlib.Path, stage1, mindc, so):
    src = f.read_text()
    want = (int(header(src, "expect").split("=")[1]), header(src, "stdout").encode())
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        got, elf = native(stage1, src.encode(), td)
        mres = mlir(mindc, src, td)
    tag = f.name
    if got != want:
        return False, f"fixture {tag}: native={got} want={want}"
    note = header(src, "note")
    if note is not None and elf[-44] != int(note):
        return False, f"fixture {tag}: PT_NOTE n_type={elf[-44]} want {note}"
    mspec = header(src, "mlir")
    if mspec is not None and mspec.startswith("nobuild"):
        why = mspec[len("nobuild="):] if mspec.startswith("nobuild=") else MLIR_LIMIT
        if mres[0] != "nobuild" or why not in mres[1]:
            return False, f"fixture {tag}: pinned MLIR build failure ({why!r}) changed: mlir={mres}"
        mtxt = f"MLIR refuses to build (pinned: {why!r})"
    elif mspec is not None:
        pinned = int(mspec.split("=")[1])
        if mres[0] != pinned:
            return False, f"fixture {tag}: pinned MLIR defect changed: mlir={mres} pinned exit={pinned}"
        mtxt = f"MLIR exit={pinned} (pinned MLIR defect)"
    else:
        if mres != want:
            return False, f"fixture {tag}: mlir={mres} want={want}"
        mtxt = "== MLIR"
    if so is not None:
        so_elf = so_stage(so, src.encode())
        if so_elf != elf:
            return False, f"fixture {tag}: oracle .so stage bytes ({len(so_elf)}B) != stage1 bytes ({len(elf)}B)"
        mtxt += " ; .so == stage1 bytes"
    return True, f"fixture {tag}: exit={want[0]} stdout={want[1].decode()!r} native == hand-derived {mtxt}"


def so_stage(so, src: bytes) -> bytes:
    comb = BLOB + src
    buf = ctypes.create_string_buffer(comb, len(comb))
    es = so.selftest_native_elf_stage(ctypes.cast(buf, ctypes.c_void_p).value, len(comb), len(BLOB))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]  # noqa: E731
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def tiny_elf(code: int) -> bytes:
    """A 132-byte static x86-64 ELF that exits with `code` (harness self-check payload)."""
    base, off = 0x400000, 64 + 56
    text = b"\xbf" + struct.pack("<i", code) + b"\xb8\x3c\x00\x00\x00\x0f\x05"  # mov edi; mov eax,60; syscall
    size = off + len(text)
    eh = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + bytes(9), 2, 0x3E, 1, base + off,
                     64, 0, 0, 64, 56, 1, 0, 0, 0)
    ph = struct.pack("<IIQQQQQQ", 1, 5, 0, base, base, size, size, 0x1000)
    return eh + ph + text


def write_elf(path: pathlib.Path, elf: bytes) -> pathlib.Path:
    path.write_bytes(elf)
    path.chmod(0o755)
    return path


def harness_checks(rep: Report, jobs: int, td: pathlib.Path) -> None:
    """The gate writes ELFs and execs them from a thread pool. Any process that
    holds a write descriptor on a file makes exec of that file fail with ETXTBSY
    (Text file busy); a sibling thread's fork (subprocess with preexec_fn) holds
    copies of every descriptor open at fork time until its child execs. Both
    checks drive the gate's own helpers (run_bin, native_bounded's run)."""
    print("[cf] harness: refusal requires rc 3, a diagnostic, and empty stdout")
    for label, output, diagnostic, accepted in (
        ("clean", "", "printf 'capacity refusal\\n' >&2", True),
        ("silent", "", "", False),
        ("whitespace", "", "printf ' \\n' >&2", False),
        ("elf", "printf '\\177ELFjunkjunk'", "printf 'refused\\n' >&2", False),
        ("text", "printf 'unexpected output'", "printf 'refused\\n' >&2", False),
    ):
        stub = td / f"refusal-{label}.sh"
        stub.write_text(f"#!/bin/sh\ncat >/dev/null\n{output}\n{diagnostic}\nexit 3\n")
        stub.chmod(0o755)
        src = b"fn main() -> i64 { 0 }"
        default, _ = native(stub, src, td)
        bounded = native_bounded(stub, src, td)
        for profile, got in (("default", default), ("bounded", bounded)):
            rep.check((got[:2] == ("refused", 3)) == accepted,
                      f"harness {profile} {label} refusal: {got[:2]}, accepted={accepted}")
    print("[cf] harness: exec of a freshly written ELF while a writer holds it (ETXTBSY)")
    h = write_elf(td / "held.elf", tiny_elf(7))
    for label, kw in (("run_bin", {}), ("bounded run", {"timeout": 120, "preexec_fn": capped})):
        holder = subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'r+b'); "
                                   "print('held', flush=True); time.sleep(0.4)", str(h)],
                                  stdout=subprocess.PIPE)
        held = holder.stdout.readline().strip() == b"held"
        try:
            got = run_bin(h, **kw)
        except OSError as e:
            got = ("OSError", e.errno, e.strerror)
        holder.wait()
        rep.check(held and got == (7, b""), f"harness {label}: exec while another process holds a write "
                                            f"descriptor (writer released after 0.4 s): got={got} want=(7, b'')")
    sib = write_elf(td / "sibling.elf", tiny_elf(0))
    n = 600

    def case(k: int):
        with tempfile.TemporaryDirectory() as cdn:
            exec_run([str(sib)], capture_output=True, preexec_fn=nice)  # a sibling fork, like native_compile
            p = write_elf(pathlib.Path(cdn) / "n.elf", tiny_elf(k % 200))  # like native(): write the ELF
            try:
                got = run_bin(p)  # then exec it
            except OSError as e:
                got = ("OSError", e.errno)
            exec_run([str(sib)], capture_output=True, preexec_fn=nice)  # like mlir()
        return got == (k % 200, b""), got

    before = _etxtbsy_retries[0]
    with ThreadPoolExecutor(max(4, jobs)) as ex:
        res = list(ex.map(case, range(n)))
    bad = [g for ok, g in res if not ok]
    rep.check(not bad, f"harness pooled write+exec ({n} cases, {max(4, jobs)} threads, forking siblings): "
                       f"{len(bad)} failed{(' e.g. ' + str(bad[:3])) if bad else ''}")
    print(f"  ETXTBSY retries absorbed: {_etxtbsy_retries[0]} ({_etxtbsy_retries[0] - before} in the pool)")


def corpus_case(args):
    name, src, exp, ba, stage1, mindc = args
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        got, _ = native(stage1, src.encode(), td, seeded=False)
        mres = mlir(mindc, src, td)
    return name, src, exp, ba, got, mres


def parallel_bounded(rep: Report, stage1b: pathlib.Path, td: pathlib.Path, jobs: int, cases, what: str,
                     cpu: int = FLAT_CPU, vacuous_on_budget: bool = False) -> None:
    """Run (label, (source, expected), may_refuse, seeded) cases on the bounded
    compiler from a thread pool, RLIMIT_AS 2 GiB and RLIMIT_CPU `cpu`: success
    (the Python value) or a clean rc 3; a signal, the arena trap (exit 70), a
    CPU-budget kill or a wrong value is a FAIL. vacuous_on_budget: a case sized
    under the stack budget that the budget refuses never ran the walks it is
    there to exercise, so that refusal FAILs too."""
    def one(case):
        label, (src, exp), may_refuse, seeded = case
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=td) as sub:
            got = native_bounded(stage1b, src.encode(), pathlib.Path(sub), seeded, cpu=cpu)
        ok = got == exp or (may_refuse and got[0] == "refused" and got[1] == 3)
        if vacuous_on_budget and got[0] == "refused" and "stack budget" in str(got[2:]):
            ok = False
            label += " VACUOUS: refused by the token budget, the walks never ran"
        return ok, (f"bounded {what} {label} ({len(src)} B): native={got[:2]} Python={exp}"
                    + (" (a clean rc-3 refusal is accepted)" if may_refuse else "")
                    + f" [{time.time() - t0:.0f} s]")
    with ThreadPoolExecutor(jobs) as ex:
        for ok, msg in ex.map(one, cases):
            rep.check(ok, msg)


def flat_matrix_section(rep: Report, stage1b: pathlib.Path, td: pathlib.Path, jobs: int) -> None:
    """Giant flat statement lists in every statement position."""
    cases = [(label, prog, may_refuse, True) for label, prog, may_refuse in flat_matrix_cases()]
    parallel_bounded(rep, stage1b, td, jobs, cases, "flat", vacuous_on_budget=True)


def oracle_section(rep: Report, so_path, stage1, td: pathlib.Path) -> None:
    """The oracle .so's heap (see _cf_oracle): the free contract of main.mind,
    every fixture and a corpus slice through both .so entries under glibc's
    free checking, and the .so's emit of this tree's own source (the self-host
    loop's oracle leg) == the stage1 the frozen seed derives from it."""
    bad = O.free_lint((_HERE / "main.mind").read_text())
    rep.check(not bad, "oracle free contract: every __mind_free in main.mind frees a __mind_alloc result"
                       + (f" -- VIOLATIONS: {bad}" if bad else ""))
    if so_path is None:
        print("  (MINDC_SO not set: the .so heap and self-emit checks need the oracle .so)")
        return
    ok, why = O.run_checked_child(pathlib.Path(__file__), "heap", str(so_path), td / "heap.out")
    rep.check(ok, f"oracle .so heap: fixtures + corpus slice through selftest_native_elf_stage and "
                  f"selftest_native_elf_u under glibc.malloc.check=3: {why}")
    if stage1 is None:
        return
    out = td / "so_self.elf"
    ok, why = O.run_checked_child(pathlib.Path(__file__), "self", str(so_path), out)
    got = out.read_bytes() if out.exists() else b""
    want = stage1.read_bytes()
    rep.check(ok and got == want,
              f"oracle .so self-emit (selftest_native_elf_u on this tree's combined source, malloc-checked): "
              f"{len(got)} B {'==' if got == want else '!='} frozen-seed stage1 {len(want)} B; {why}")


def so_child(mode: str, so_path: str, out: str) -> int:
    fixtures = sorted(FIXTURES.glob("*.mind"))
    corpus = [src.encode() for _, src, _, _ in C.corpus(11, 40)]
    self_src = (_HERE / "main.mind").read_bytes() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    return O.child(mode, so_path, out, BLOB, fixtures, corpus, self_src)


def bounded_section(rep: Report, stage1b: pathlib.Path, td: pathlib.Path, cases) -> None:
    for label, (bsrc, bexp), may_refuse, seeded in cases:
        t0 = time.time()
        got = native_bounded(stage1b, bsrc.encode(), td, seeded)
        if bexp[0] == "refused":
            ok = got[:2] == bexp  # an expected clean refusal: rc 3, no ELF
        else:
            ok = got == bexp or (may_refuse and got[0] == "refused" and got[1] == 3)
        rep.check(ok, f"bounded {label} ({len(bsrc)} B): native={got[:2]} Python={bexp}"
                      + (" (a clean rc-3 refusal is accepted)" if may_refuse else "")
                      + f" [{time.time() - t0:.0f} s]")


def deep_extra_sections(rep: Report, stage1b, stage1, td: pathlib.Path) -> None:
    for label, (dsrc, dexp), seeded in deep_v1_cases():
        t0 = time.time()
        got = native_bounded(stage1b, dsrc.encode(), td, seeded, v2=False)
        rep.check(got[:2] == dexp, f"{label} ({len(dsrc)} B): native={got[:2]} (must refuse, rc 3)"
                                   f" [{time.time() - t0:.0f} s]")
    if stage1 is None:
        return
    for label, (dsrc, dexp), may_refuse, seeded in deep_default_cases() + default_capacity_cases():
        t0 = time.time()
        try:
            got, _ = native(stage1, dsrc.encode(), td, seeded)
        except subprocess.TimeoutExpired:
            got = ("compile-timeout", 600)
        ok = got == dexp or (may_refuse and got[0] == "refused" and got[1] == 3)
        rep.check(ok, f"default-profile {label} ({len(dsrc)} B): native={got[:2]} Python={dexp}"
                      + (" (a clean rc-3 refusal is accepted)" if may_refuse else "")
                      + f" [{time.time() - t0:.0f} s]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=os.environ.get("CF_STAGE1"))
    ap.add_argument("--stage1-bounded", default=os.environ.get("CF_STAGE1_BOUNDED"))
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--shadow-seeds", default="13")
    ap.add_argument("--shadow-count", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--capacity-only", action="store_true",
                    help="run only the whole-program capacity cases (many functions, giant tokens, "
                         "wide expressions) in the bounded profile")
    ap.add_argument("--oracle-only", action="store_true",
                    help="run only the oracle .so heap checks (needs MINDC_SO)")
    ap.add_argument("--so-child", nargs=3, metavar=("MODE", "SO", "OUT"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--harness-only", action="store_true",
                    help="run only the harness self-checks (no compiler needed)")
    ap.add_argument("--flat-only", action="store_true",
                    help="run only the giant flat statement-list cases in the bounded profile")
    ap.add_argument("--deep-only", action="store_true",
                    help="run only the deep-tree cases (bounded, bounded fed a v1 image, and with "
                         "--stage1 the default profile)")
    a = ap.parse_args()
    if a.so_child:
        return so_child(*a.so_child)
    if a.capacity_only or a.oracle_only:
        rep = Report()
        with tempfile.TemporaryDirectory() as tdn:
            td = pathlib.Path(tdn)
            if not SEED.exists() and not (a.stage1_bounded and a.stage1):
                print(f"BLOCKED: frozen seed {SEED} missing")
                return 2
            if a.oracle_only:
                if not os.environ.get("MINDC_SO"):
                    print("BLOCKED: --oracle-only needs MINDC_SO (the oracle .so)")
                    return 2
                from _selfhost_so import resolve_so
                stage1 = pathlib.Path(a.stage1).resolve() if a.stage1 else derive_stage1(td)
                print(f"[cf] oracle .so heap: stage1 = {stage1}")
                oracle_section(rep, resolve_so(), stage1, td)
            if a.capacity_only:
                stage1b = (pathlib.Path(a.stage1_bounded).resolve() if a.stage1_bounded
                           else derive_stage1_bounded(td))
                print(f"[cf] whole-program capacity, bounded profile: stage1 (bounded) = {stage1b}")
                parallel_bounded(rep, stage1b, td, a.jobs, capacity_cases(), "capacity", cpu=CAP_CPU)
        print(f"\nSDLC-GATE self_host_native_cf ran={rep.passed + len(rep.fails)} fail={len(rep.fails)}")
        print("FAIL" if rep.fails else "ALL PASS  (oracle .so heap / whole-program capacity)")
        return 1 if rep.fails else 0
    if a.deep_only:
        rep = Report()
        with tempfile.TemporaryDirectory() as tdn:
            td = pathlib.Path(tdn)
            if a.stage1_bounded:
                stage1b = pathlib.Path(a.stage1_bounded).resolve()
            elif SEED.exists():
                stage1b = derive_stage1_bounded(td)
            else:
                print(f"BLOCKED: frozen seed {SEED} missing")
                return 2
            stage1 = pathlib.Path(a.stage1).resolve() if a.stage1 else None
            print(f"[cf] deep trees: stage1 (bounded) = {stage1b}, stage1 (default) = {stage1}")
            bounded_section(rep, stage1b, td, deep_cases())
            deep_extra_sections(rep, stage1b, stage1, td)
        print(f"\nSDLC-GATE self_host_native_cf ran={rep.passed + len(rep.fails)} fail={len(rep.fails)}")
        print("FAIL" if rep.fails else "ALL PASS  (deep trees: success or a clean rc-3 refusal, never a signal)")
        return 1 if rep.fails else 0
    if a.flat_only:
        rep = Report()
        with tempfile.TemporaryDirectory() as tdn:
            td = pathlib.Path(tdn)
            if a.stage1_bounded:
                stage1b = pathlib.Path(a.stage1_bounded).resolve()
            elif SEED.exists():
                stage1b = derive_stage1_bounded(td)
            else:
                print(f"BLOCKED: frozen seed {SEED} missing")
                return 2
            print(f"[cf] giant flat statement lists, bounded profile: stage1 (bounded) = {stage1b}")
            bounded_section(rep, stage1b, td, flat_cases())
            flat_matrix_section(rep, stage1b, td, a.jobs)
        print(f"\nSDLC-GATE self_host_native_cf ran={rep.passed + len(rep.fails)} fail={len(rep.fails)}")
        print("FAIL" if rep.fails else "ALL PASS  (giant flat statement lists, bounded profile)")
        return 1 if rep.fails else 0
    if a.harness_only:
        rep = Report()
        with tempfile.TemporaryDirectory() as tdn:
            harness_checks(rep, a.jobs, pathlib.Path(tdn))
        print(f"\nSDLC-GATE self_host_native_cf ran={rep.passed + len(rep.fails)} fail={len(rep.fails)}")
        print("FAIL" if rep.fails else "ALL PASS  (harness self-checks)")
        return 1 if rep.fails else 0
    mindc = os.environ.get("MINDC_BIN") or str(_REPO / "target" / "release" / "mindc")
    if not pathlib.Path(mindc).exists():
        print(f"BLOCKED: MINDC_BIN (an MLIR-capable mindc) not found at {mindc}")
        return 2
    so = so_path = None
    if os.environ.get("MINDC_SO"):
        # Strict freshness (an oracle older than the sources is refused). The
        # twin leg is optional: without MINDC_SO it is skipped, never rebuilt.
        from _selfhost_so import resolve_so
        so_path = resolve_so()
        so = ctypes.CDLL(str(so_path))
        so.selftest_native_elf_stage.restype = ctypes.c_int64
        so.selftest_native_elf_stage.argtypes = [ctypes.c_int64] * 3
    rep = Report()
    with tempfile.TemporaryDirectory() as tdn, ThreadPoolExecutor(1) as bex:
        td = pathlib.Path(tdn)
        harness_checks(rep, a.jobs, td)
        # The bounded compiler is a second whole self-compile: start it now so it
        # overlaps the default derivation, the fixtures and the corpus (wall time
        # ~ one derivation, inside run_gate's budget).
        if not a.stage1_bounded and not SEED.exists():
            print(f"BLOCKED: frozen seed {SEED} missing")
            return 2
        bfut = None if a.stage1_bounded else bex.submit(derive_stage1_bounded, td)
        if a.stage1:
            stage1 = pathlib.Path(a.stage1).resolve()
            print(f"[cf] stage1 = {stage1} (given)")
        else:
            if not SEED.exists():
                print(f"BLOCKED: frozen seed {SEED} missing")
                return 2
            stage1 = derive_stage1(td)
            print(f"[cf] stage1 = frozen seed -> this tree's main.mind ({stage1.stat().st_size} B)")

        print("[cf] oracle .so heap (free contract, malloc-checked fixtures, self-emit == stage1)")
        oracle_section(rep, so_path, stage1, td)

        print("[cf] fixtures (hand-derived expectations, census rules A1 A6 A7 A9 L1 L2 L3 O1 O2 O3)")
        for f in sorted(FIXTURES.glob("*.mind")):
            cond, msg = check_fixture(f, stage1, mindc, so)
            rep.check(cond, msg)

        seeds = [int(x) for x in a.seeds.split(",") if x]
        cases = []
        for sd in seeds:
            cases += [(n, s, e, ba, stage1, mindc) for n, s, e, ba in C.corpus(sd, a.count)]
        shadow_seeds = [int(x) for x in a.shadow_seeds.split(",") if x]
        for sd in shadow_seeds:
            cases += [(n, s, e, ba, stage1, mindc) for n, s, e, ba in C.corpus(sd, a.shadow_count, "shadow")]
        print(f"[cf] corpus: {len(cases)} generated programs (seeds {seeds} x {a.count}, shadow profile "
              f"seeds {shadow_seeds} x {a.shadow_count}), native vs MLIR vs Python")
        st = {"three_way": 0, "native_eq_python_mlir_limit": 0, "native_eq_python_mlir_defect": 0}
        with ThreadPoolExecutor(a.jobs) as ex:
            for name, src, exp, ba, got, mres in ex.map(corpus_case, cases):
                if not rep.check(got == exp, f"corpus {name}: native={got} python={exp}"):
                    if len(rep.fails) <= 3:
                        print(src)
                    continue
                if mres[0] == "nobuild":
                    rep.check(MLIR_LIMIT in mres[1],
                              f"corpus {name}: native == Python; MLIR build refused: {mres[1][-90:]!r}")
                    if MLIR_LIMIT in mres[1]:
                        st["native_eq_python_mlir_limit"] += 1
                    continue
                if mres != exp and ba:
                    # pinned MLIR defect families (see _cf_corpus.has_branch_assign
                    # and, for the shadow profile, _cf_corpus.has_loop_shadow)
                    why = ("MLIR loses a value-if branch assignment or carries a loop-body shadow"
                           if name.startswith("cfshadow_") else "MLIR loses a value-if branch assignment")
                    rep.check(True, f"corpus {name}: native == Python {exp}; MLIR {mres} (pinned: {why})")
                    st["native_eq_python_mlir_defect"] += 1
                    continue
                if not rep.check(mres == exp, f"corpus {name}: native == MLIR == Python {exp}"
                                 if mres == exp else
                                 f"corpus {name}: mlir={mres} python={exp} (native agrees with python)"):
                    if len(rep.fails) <= 3:
                        print(src)
                    continue
                st["three_way"] += 1
        print(f"  corpus: {st['three_way']} three-way equal (native == MLIR == Python), "
              f"{st['native_eq_python_mlir_limit']} native == Python where MLIR refuses "
              f"'{MLIR_LIMIT}', {st['native_eq_python_mlir_defect']} native == Python where MLIR "
              f"hits a pinned defect (value-if branch assignment / loop-body shadow), "
              f"{len(cases) - sum(st.values())} divergent")

        print("[cf] capacity (census X1: giant function)")
        gsrc, gexp = giant_program(2500)
        got, _ = native(stage1, gsrc.encode(), td)
        rep.check(got == gexp, f"giant fn (2500-arm else-if chain in a loop, {len(gsrc)} B): "
                               f"native={got} Python={gexp}")

        for blocks in (200, 700):
            ssrc, sexp = straight_line_program(blocks)
            got, _ = native(stage1, ssrc.encode(), td)
            rep.check(got == sexp, f"giant straight-line fn ({blocks} x let/if-return/store = "
                                   f"{4 * blocks} statements): native={got} Python={sexp}")

        for n, shape in ((520, "straight"), (300, "pairs"), (600, "loop")):
            rsrc, rexp = reassign_program(n, shape)
            got, _ = native(stage1, rsrc.encode(), td)
            rep.check(got == rexp, f"let-env capacity ({n} {shape} reassignments, past the SSA "
                                   f"path's 512-entry let-env): native={got} Python={rexp}")

        for label, (xsrc, xexp), may_refuse, seeded in extreme_cases():
            got, _ = native(stage1, xsrc.encode(), td, seeded)
            if xexp[0] == "refused":
                ok = got[:2] == xexp  # an expected clean refusal: rc 3, no ELF
            else:
                ok = got == xexp or (may_refuse and got[0] == "refused" and got[1] == 3)
            rep.check(ok, f"default-profile {label} ({len(xsrc)} B): native={got[:2]} Python={xexp}"
                          + (" (a clean rc-3 refusal is accepted)" if may_refuse else ""))

        print("[cf] capacity, bounded production profile (768 MiB arena / 384 MiB stack, "
              "RLIMIT_AS 2 GiB)")
        if a.stage1_bounded:
            stage1b = pathlib.Path(a.stage1_bounded).resolve()
            print(f"  stage1 (bounded) = {stage1b} (given)")
        else:
            stage1b = bfut.result()
            print(f"  stage1 (bounded) = frozen seed + v2 header -> this tree ({stage1b.stat().st_size} B)")
        bounded_section(rep, stage1b, td, bounded_cases())
        flat_matrix_section(rep, stage1b, td, a.jobs)
        parallel_bounded(rep, stage1b, td, a.jobs, capacity_cases(), "capacity", cpu=CAP_CPU)
        deep_extra_sections(rep, stage1b, stage1, td)

        print("[cf] clean refusal")
        got, _ = native(stage1, REFUSE_ADDR.encode(), td)
        rep.check(got[0] == "refused" and got[1] == 3,
                  f"address-taking HOME function: clean rc-3 refusal, no ELF (got {got[:2]})")

    print(f"\n  ETXTBSY retries absorbed over the whole run: {_etxtbsy_retries[0]}")
    print(f"SDLC-GATE self_host_native_cf ran={rep.passed + len(rep.fails)} fail={len(rep.fails)}")
    if rep.fails:
        print(f"FAIL ({len(rep.fails)}): " + "; ".join(rep.fails[:6]))
        return 1
    print("ALL PASS  (stage1 general control flow: fixtures == hand-derived == MLIR, "
          "corpus native == MLIR == Python, giant fn compiles in the default and the "
          "bounded profile, clean refusal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
