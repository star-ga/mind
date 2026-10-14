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
  3. capacity  giant functions (a 2500-arm else-if chain inside a loop;
               800- and 2800-statement straight-line let/if-return functions,
               the census X1 shape, past the mic@3 planner's old 512-record
               table and the SSA path's 512-entry jump / let tables; 520
               straight-line reassignments, 300 let/assign pairs and 600
               reassignments in a loop body, past the 512-entry let-env) must
               compile and run correctly — never SIGSEGV.
               Bounded production profile: the compiler the frozen seed emits
               with the v2 layout header (768 MiB arena, 384 MiB stack), run
               under RLIMIT_AS 2 GiB like the MIND-Lab consumer harness, must
               compile and run a 2048-leaf if/else tree (~425 KB, the
               cs_ucd_mask shape at 7x) and a comment-padded 1024-leaf tree;
               the HOME tables were sized from the source span and initialised
               one stack frame per slot, which overflowed the 384 MiB stack.
               Giant FLAT functions in the same profile (40000 straight-line
               statements and 4000 chained lets must compile and run; 20000
               chained lets, 40000 statements past the 4 MiB code window and
               30000 statements in one while body may instead be refused
               cleanly, rc 3): every statement-list walk took one stack frame per
               statement (the mic@3 twin's planner alone ~14 KB) and the
               compiler died with SIGSEGV at ~26k statements. `--flat-only`
               runs just these cases.
               A function past the arena budget, or a program past the stack
               budget (a 65536-leaf, 18.8 MB tree), may be refused (rc 3),
               never a signal; a 21 MB program with an unsupported construct
               must be refused (rc 3): the refusal reporter's per-byte scan
               overflowed the stack there. Both extremes also run through the
               default-profile compiler (1 GiB arena / 4 GiB stack) and with the
               UNSEEDED image (no std prefix, so no intrinsic literal in the
               source: the trace-hash front end's per-byte source scan and copy
               overflowed both stacks), plus a 1.96 MB unseeded tree that must
               compile: success or rc 3, never a signal.
  4. refusal   a HOME-lowered function that takes an address refuses cleanly
               (rc 3, no ELF), never a crash.
  5. twin      with MINDC_SO set, the oracle .so's selftest_native_elf_stage
               emits the SAME bytes as stage1 for every fixture.

Usage:
  MINDC_BIN=target/release/mindc [MINDC_SO=/tmp/oracle.so] \\
    python3 self_host_native_cf_smoke.py [--stage1 PATH] [--stage1-bounded PATH] [--seeds 11,12]
Without --stage1 / --stage1-bounded each compiler is derived from the frozen seed
and this tree (two whole self-compiles).
Exit: 0 ALL PASS, 1 FAIL, 2 BLOCKED (no MLIR-capable mindc / no seed).
"""

import argparse
import bisect
import ctypes
import errno
import os
import pathlib
import re
import resource
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


BOUNDED = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard (self_host_bounded_profile_smoke)
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


def native_bounded(stage1b, src: bytes, td: pathlib.Path, seeded: bool = True):
    """Compile with the bounded compiler (v2 header, RLIMIT_AS 2 GiB) and run the
    program under the same cap. ('crash', signal) for a compiler killed by a
    signal, ('refused', rc, stderr tail) for a compiler that wrote no ELF."""
    img = v2_header() + image(src, seeded)
    try:
        r = exec_run([str(stage1b)], input=img, capture_output=True, timeout=900, preexec_fn=capped)
    except subprocess.TimeoutExpired:
        return ("compile-timeout", 900)
    if r.returncode < 0:
        return ("crash", -r.returncode)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        return ("refused", r.returncode, r.stderr.decode(errors="replace").strip()[-120:])
    p = td / "b.elf"
    p.write_bytes(r.stdout)
    p.chmod(0o755)
    return run_bin(p, timeout=120, preexec_fn=capped)


def native_compile(stage1: pathlib.Path, src: bytes, seeded: bool = True):
    r = exec_run([str(stage1)], input=image(src, seeded), capture_output=True, timeout=600,
                       preexec_fn=nice)
    ok = r.returncode == 0 and r.stdout[:4] == b"\x7fELF"
    return ok, r.returncode, (r.stdout if ok else b""), r.stderr.decode(errors="replace")


def run_bin(path: pathlib.Path, timeout: int = 60, preexec_fn=None):
    try:
        r = exec_run([str(path)], capture_output=True, timeout=timeout, preexec_fn=preexec_fn)
    except subprocess.TimeoutExpired:
        return ("timeout", b"")
    return (r.returncode & 255 if r.returncode >= 0 else r.returncode, r.stdout)


def native(stage1, src: bytes, td: pathlib.Path, seeded: bool = True):
    ok, rc, elf, err = native_compile(stage1, src, seeded)
    if not ok:
        return ("refused", rc, err.strip()[-160:]), None
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


def giant_program(arms: int) -> tuple:
    """X1 shape: one function holding a huge else-if chain inside a loop."""
    lines = ["fn pick(x: i64) -> i64 {", "    let mut r: i64 = 0;"]
    for k in range(arms):
        kw = "if" if k == 0 else "} else if"
        lines.append(f"    {kw} x == {k} {{")
        lines.append(f"        r = {(k * 7 + 3) % 1000};")
    lines += ["    } else {", "        r = 0 - 1;", "    }", "    r", "}",
              "fn main() -> i64 {", "    let mut i: i64 = 0;", "    let mut s: i64 = 0;",
              f"    while i < {arms + 3} {{",
              "        s = s + pick(i * 3);",
              "        if s > 1000000 && i > 2 {", "            return 0 - 2;", "        }",
              "        i = i + 1;", "    }", "    s % 251", "}"]
    s = 0
    for i in range(arms + 3):
        x = i * 3
        s += (x * 7 + 3) % 1000 if x < arms else -1
        if s > 1000000 and i > 2:
            return "\n".join(lines) + "\n", (C.wrap(-2) & 255, b"")
    return "\n".join(lines) + "\n", (C.tmod(s, 251) & 255, b"")


def straight_line_program(blocks: int) -> tuple:
    """Census X1 shape (MIND-Lab nr_literals): one straight-line function of
    `blocks` x {let, early-return if, store} — 600+ statements. The mic@3 twin's
    statement planner had a fixed 512-record table and segfaulted the compiler
    on it."""
    lines = ["fn space(c: i64, n: i64) -> i64 {", "    if n > 0 {", "        return __mind_alloc(n);",
             "    }", "    0", "}", "fn table(c: i64) -> i64 {",
             f"    let t: i64 = space(c, {8 * blocks});", "    if t == 0 { return 0; }"]
    for k in range(blocks):
        lines.append(f"    let p{k}: i64 = space(c, 8);")
        lines.append(f"    if p{k} == 0 {{ return 0; }}")
        lines.append(f"    __mind_store_i64(p{k}, {k * 3 + 1});")
        lines.append(f"    __mind_store_i64(t + {8 * k}, __mind_load_i64(p{k}));")
    lines += ["    t", "}", "fn main() -> i64 {", "    let t: i64 = table(1);", "    let mut s: i64 = 0;",
              "    let mut i: i64 = 0;", f"    while i < {blocks} {{",
              "        s = s + __mind_load_i64(t + i * 8);", "        i = i + 1;", "    }", "    s % 251", "}"]
    return "\n".join(lines) + "\n", (sum(k * 3 + 1 for k in range(blocks)) % 251, b"")


def reassign_program(n: int, shape: str) -> tuple:
    """Census X1 let-env capacity: every parameter, `let` and reassignment of
    the SSA path appends one entry to its 512-entry let-env, which had no bound
    check — `n` straight-line reassignments (n >= 511), n let/assign pairs
    (n >= 256) or n reassignments in a loop body overran it and segfaulted
    stage1. Such a function must compile (HOME) and run correctly."""
    lines = ["fn f(a: i64) -> i64 {", "    let mut s: i64 = a;"]
    s, reps = 1, 1
    if shape == "loop":
        lines.append("    let mut i: i64 = 0;")
        lines.append("    while i < 3 {")
        reps = 3
    for k in range(n):
        if shape == "pairs":
            lines += [f"    let x{k}: i64 = s + {k};", f"    s = x{k} % 1000;"]
        else:
            lines.append(f"    s = (s + {k}) % 1000;")
    if shape == "loop":
        lines += ["        i = i + 1;", "    }"]
    lines += ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]
    for _ in range(reps):
        for k in range(n):
            s = (s + k) % 1000
    return "\n".join(lines) + "\n", (s % 251, b"")


def dense_tree_program(depth: int, pad: int = 0) -> tuple:
    """Census X1 shape at scale (MIND-Lab cs_ucd_mask is a 2006-node if/else tree
    over a code point): one function holding a balanced if/else tree of `depth`
    levels (2**depth leaf returns, 4-space indentation), optionally padded with
    `pad` comment lines inside the body. Its source span, not its node count, is
    what used to size the HOME tables, so the indentation alone pushed the
    bounded compiler's stack past its guard."""
    lim = 1114112
    leaves = []

    def leaf(k: int) -> int:
        return (k * 2654435761 + 97) % 251

    lines = ["fn mask(cp: i64) -> i64 {", "    if cp < 0 || cp > 1114111 { return 0; }"]
    lines += [f"    // capacity padding line {i}: words if while let && || in a comment" for i in range(pad)]

    def tree(lo: int, hi: int, d: int, ind: int) -> None:
        sp = "    " * ind
        if d == 0:
            leaves.append((lo, hi))
            lines.append(f"{sp}return {leaf(len(leaves) - 1)};")
            return
        mid = (lo + hi) // 2
        lines.append(f"{sp}if cp < {mid} {{")
        tree(lo, mid, d - 1, ind + 1)
        lines.append(f"{sp}}} else {{")
        tree(mid, hi, d - 1, ind + 1)
        lines.append(f"{sp}}}")

    tree(0, lim, depth, 1)
    lines += ["}", "fn main() -> i64 {", "    let mut cp: i64 = 0 - 3;", "    let mut acc: i64 = 0;",
              "    while cp < 1114130 {", "        acc = (acc * 131 + mask(cp) + 7) % 1000003;",
              "        cp = cp + 37;", "    }", "    return acc % 251;", "}"]

    lows = [lo for lo, _ in leaves]

    def mask(cp: int) -> int:
        if cp < 0 or cp > 1114111:
            return 0
        k = bisect.bisect_right(lows, cp) - 1
        assert leaves[k][0] <= cp < leaves[k][1]
        return leaf(k)

    acc, cp = 0, -3
    while cp < 1114130:
        acc = (acc * 131 + mask(cp) + 7) % 1000003
        cp += 37
    return "\n".join(lines) + "\n", (acc % 251, b"")


def flat_program(n: int, shape: str) -> tuple:
    """Census X1 giant FLAT function: one block of `n` statements.
    inc:    `s = s + 1;` x n (straight-line reassignments of one name)
    assign: `s = (s + k) % 1000;` x n (the same, ~183 B of code per statement:
            40000 of them pass the image's 4 MiB code window, a clean rc 3)
    lets:   `let xK: i64 = xJ + k % 7;` x n (a chain of n bindings)
    loop:   the assign list as the body of a 2-trip while loop.
    Every statement-list walk of the compiler took one stack frame per
    statement (the mic@3 twin's planner frame alone is ~14 KB, and its env
    lookup at statement i recursed over the i bindings before it): the bounded
    compiler (384 MiB stack) died with SIGSEGV at ~26k straight-line statements."""
    trips = 2 if shape == "loop" else 1
    ind = "        " if shape == "loop" else "    "
    lines = ["fn f(a: i64) -> i64 {", "    let mut s: i64 = a;"]
    if shape == "loop":
        lines += ["    let mut i: i64 = 0;", "    while i < 2 {"]
    v = 1
    if shape == "lets":
        lines.append("    let x0: i64 = s;")
        for k in range(1, n):
            lines.append(f"    let x{k}: i64 = x{k - 1} + {k % 7};")
            v += k % 7
        lines.append(f"    s = x{n - 1} % 1000;")
        v %= 1000
    elif shape == "inc":
        lines += ["    s = s + 1;"] * n
        v += n
    else:
        for k in range(n):
            lines.append(f"{ind}s = (s + {k % 997}) % 1000;")
        for _ in range(trips):
            for k in range(n):
                v = (v + k % 997) % 1000
    if shape == "loop":
        lines += ["        i = i + 1;", "    }"]
    lines += ["    s", "}", "fn main() -> i64 { f(1) % 251 }"]
    return "\n".join(lines) + "\n", (v % 251, b"")


def flat_cases():
    """(label, (source, expected), may_refuse, seeded): giant flat statement
    lists in the bounded production profile. Each must compile and run to the
    Python value, or (may_refuse) be refused cleanly (rc 3); never a signal."""
    return [
        ("40000 straight-line statements in one function (census X1 giant flat list)",
         flat_program(40000, "inc"), False, True),
        # 20000 bindings pass the native alias environment's 4096-binding lookup
        # depth (alias_cap: a structured refusal), so this one may refuse; 4000
        # stay under it and must compile (the mic@3 twin's let-chain tables were
        # a fixed 512 bindings / 4096 descriptors and silently overran).
        ("20000 chained lets in one function", flat_program(20000, "lets"), True, True),
        ("4000 chained lets in one function", flat_program(4000, "lets"), False, True),
        ("40000 heavier straight-line statements (past the 4 MiB code window)",
         flat_program(40000, "assign"), True, True),
        ("30000 statements in one while body", flat_program(30000, "loop"), True, True),
    ]


def bounded_cases():
    """(label, (source, expected), may_refuse, seeded) for the bounded production compiler.
    Each must compile and run to the Python value; a may_refuse case may instead
    be refused cleanly (rc 3). A compiler killed by a signal is always a FAIL."""
    return [
        ("dense if/else tree, 2048 leaves (census X1 cs_ucd_mask shape at 7x)",
         dense_tree_program(11), False, True),
        ("dense if/else tree, 1024 leaves + 2600 comment lines in the body",
         dense_tree_program(10, pad=2600), False, True),
    ] + flat_cases() + extreme_cases()


def extreme_cases():
    """(label, (source, expected), may_refuse, seeded) whole-program extremes, run
    in BOTH the default and the bounded profile: each must compile and run to the
    Python value or be refused cleanly (rc 3); an expected ("refused", 3) must
    refuse. Each SIGSEGVed stage1 before nb_stack_short, the halving
    nb_region_has and the halving source scanners (census X1). The UNSEEDED image
    (user_lo = 0, no std blob) matters: without the std prefix the source holds
    no `__mind_alloc` / `__mind_store_i64` / `__mind_load_i64` literal, so the
    trace-hash front end scanned (and copied) the whole source one stack frame
    per byte (find_alloc_name_span: 1.57M frames at the bounded 384 MiB stack's
    end on a 21 MB program, 3M+ frames on the 18.8 MB tree in the default
    profile; a 1.96 MB tree already overflowed the bounded stack). The 18.8 MB
    tree runs seeded only: unseeded it reaches the same scanners as the 21 MB
    unseeded case below, at twice the compile time."""
    pad = refused_padded_program(320000)
    return [
        # ~786k tokens (18.8 MB): past the bounded profile's whole-program stack
        # budget (nb_stack_short); the front end's per-token walk overflowed the
        # 384 MiB stack here (SIGSEGV), so that compiler must refuse it cleanly.
        ("dense if/else tree, 65536 leaves (18.8 MB, past the bounded stack budget)",
         dense_tree_program(16), True, True),
        # A 21 MB program that is refused (a Tensor parameter, after 320k comment
        # lines): the refusal reporter scanned the user region one stack frame
        # per byte and overflowed the stack (SIGSEGV) instead of exiting rc 3.
        ("21 MB comment-padded program with an unsupported construct (must refuse)",
         pad, False, True),
        ("21 MB comment-padded program with an unsupported construct, unseeded image (must refuse)",
         pad, False, False),
        # 8192 leaves (1.96 MB): fits every budget, so it must compile and run.
        ("dense if/else tree, 8192 leaves (1.96 MB), unseeded image", dense_tree_program(13), False, False),
    ]


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


def refused_padded_program(lines: int):
    pad = "// padding comment line for the refusal reporter scan xxxxxxxxxxxx\n" * lines
    src = ("fn main() -> i64 { return 0; }\n" + pad
           + "fn sink(x: Tensor<f32, [4]>) -> i64 { return 0; }\n")
    return src, ("refused", 3)


REFUSE_ADDR = """fn f(n: i64) -> i64 {
    let mut x: i64 = 0;
    let r: i64 = &x;
    while n > 0 {
        if n == 3 || n == 5 {
            return x;
        }
        n = n - 1;
    }
    x
}
fn main() -> i64 { return f(4); }
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=os.environ.get("CF_STAGE1"))
    ap.add_argument("--stage1-bounded", default=os.environ.get("CF_STAGE1_BOUNDED"))
    ap.add_argument("--seeds", default="11,12")
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--shadow-seeds", default="13")
    ap.add_argument("--shadow-count", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--harness-only", action="store_true",
                    help="run only the harness self-checks (no compiler needed)")
    ap.add_argument("--flat-only", action="store_true",
                    help="run only the giant flat statement-list cases in the bounded profile")
    a = ap.parse_args()
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
    so = None
    if os.environ.get("MINDC_SO"):
        # Strict freshness (an oracle older than the sources is refused). The
        # twin leg is optional: without MINDC_SO it is skipped, never rebuilt.
        from _selfhost_so import resolve_so
        so = ctypes.CDLL(str(resolve_so()))
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
