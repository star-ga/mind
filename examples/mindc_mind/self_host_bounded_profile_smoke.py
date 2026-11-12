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
  --stack-rows [--compiler ELF] [--self-only]
             the stack-budget rows (stack_rows below): the bounded compiler of
             this tree (or ELF) runs as a ptrace tracee and its stack high-water
             must stay inside the budget nb_stack_short gives the program.
             --self-only stops after the self-compile rows.
"""

import hashlib
import os
import pathlib
import re
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
from _selfhost_sources import DRIVER  # noqa: E402

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

# ---- stack-budget rows (--stack-rows) --------------------------------------
# The budget constants are read from the part that owns them (one source).
STACK_PART = _HERE / "nb_stack_budget.mind"
# The lexer's token rules (main.mind scan_one): blanks and `//` comments are
# no token; an identifier, a number, a two-byte operator, a char literal and a
# string literal are one token each; any other byte is one token.
_TOKEN = re.compile(
    rb"[ \t\r\n]+|//[^\n]*|([A-Za-z_][A-Za-z0-9_]*|[0-9][0-9_]*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    rb"|->|==|<=|>=|!=|<<|>>|&&|\|\||'\\.'|'[^'\\]'|\"(?:\\.|[^\"\\])*\"|.)", re.S)
STACK_SLACK = 2000  # tokens kept between a row and the limit (the Python count is a model of the lexer)
RECURSIVE_WALK = """\
pub fn const_find_from(toks: Vec, buf: i64, pos: i64, nlo: i64, nhi: i64) -> i64 {
    if at_eof(toks, pos) == 1 {
        return 0 - 1;
    }
    if tok_kind(toks, pos) == tk_ident() {
        if is_const_kw(buf, tok_lo(toks, pos), tok_hi(toks, pos)) == 1 {
            if tok_kind(toks, pos + 1) == tk_ident() {
                if tok_kind(toks, pos + 2) == tk_colon() {
                    if span_bytes_equal(buf, nlo, nhi, tok_lo(toks, pos + 1), tok_hi(toks, pos + 1)) == 1 {
                        return pos + 1;
                    }
                }
            }
        }
    }
    const_find_from(toks, buf, pos + 1, nlo, nhi)
}
"""


def stack_consts() -> tuple[int, int, int]:
    """(bytes per token, bytes per longest-token unit, base) of nb_stack_short."""
    text = STACK_PART.read_text()
    return tuple(int(re.search(rf"pub fn {name}\(\) -> i64 \{{\n +(\d+)\n\}}", text).group(1))
                 for name in ("nb_stack_tok_bytes", "nb_stack_len_bytes", "nb_stack_base"))


def token_stats(src: bytes) -> tuple[int, int]:
    """(tokens incl. EOF, longest token in budget units: bytes, a string literal's / 8)."""
    count, longest = 1, 0
    for m in _TOKEN.finditer(src):
        tok = m.group(1)
        if tok is not None:
            count += 1
            longest = max(longest, len(tok) // 8 if tok[:1] == b'"' and len(tok) > 1 else len(tok))
    return count, longest


def stack_budget(src: bytes) -> tuple[int, int]:
    """(tokens, stack bytes nb_stack_short budgets for `src`): the bound a row measures against."""
    per_tok, per_len, base = stack_consts()
    count, longest = token_stats(src)
    return count, count * per_tok + longest * per_len + base


def stack_token_limit(longest: int = 8) -> int:
    """Most tokens the per-token rule admits on the bounded stack."""
    per_tok, per_len, base = stack_consts()
    return (BOUNDED[1] - base - longest * per_len) // per_tok


def traced(compiler: pathlib.Path, image: bytes, td: pathlib.Path, name: str):
    """Run `compiler` on `image` as a ptrace tracee under the AS/CPU caps. Returns
    (rc, stdout, stderr, high_water): high_water = stack top - the lowest non-zero
    word of the stack PT_LOAD (zero-fill, never unmapped), read at the exit stop
    or at a fatal signal stop."""
    import array
    import ctypes
    elf = compiler.read_bytes()[:4096]
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    phnum = struct.unpack_from("<H", elf, 56)[0]
    loads = [(struct.unpack_from("<Q", elf, phoff + i * 56 + 16)[0], struct.unpack_from("<Q", elf, phoff + i * 56 + 40)[0])
             for i in range(phnum) if struct.unpack_from("<I", elf, phoff + i * 56)[0] == 1]
    va, size = max(loads)
    top = va + size
    inp, out, err = td / f"{name}.in", td / f"{name}.out", td / f"{name}.err"
    inp.write_bytes(image)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    libc.ptrace.restype = ctypes.c_long
    pid = os.fork()
    if pid == 0:
        try:
            os.dup2(os.open(inp, os.O_RDONLY), 0)
            os.dup2(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o755), 1)
            os.dup2(os.open(err, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), 2)
            resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
            resource.setrlimit(resource.RLIMIT_CPU, (SELF_CPU_CAP_S, SELF_CPU_CAP_S))
            libc.ptrace(0, 0, None, None)  # PTRACE_TRACEME
            os.execv(str(compiler), [str(compiler)])
        finally:
            os._exit(127)
    os.waitpid(pid, 0)
    libc.ptrace(0x4200, pid, None, ctypes.c_void_p(0x40))  # PTRACE_SETOPTIONS, PTRACE_O_TRACEEXIT

    def high_water() -> int:
        pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
        mem = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        try:
            pages, chunk = size >> 12, 1 << 15
            for first in range(0, pages, chunk):
                entries = array.array("Q")
                entries.frombytes(os.pread(pm, 8 * min(chunk, pages - first), ((va >> 12) + first) * 8))
                for k, entry in enumerate(entries):
                    if entry >> 62:  # present or swapped
                        page = va + ((first + k) << 12)
                        words = array.array("Q")
                        words.frombytes(os.pread(mem, 4096, page))
                        used = [j for j, word in enumerate(words) if word]
                        if used:
                            return top - (page + used[0] * 8)
            return 0
        finally:
            os.close(pm)
            os.close(mem)

    hw, rc = None, None
    libc.ptrace(7, pid, None, None)  # PTRACE_CONT
    while True:
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status):
            rc = os.WEXITSTATUS(status)
            break
        if os.WIFSIGNALED(status):
            rc = -os.WTERMSIG(status)
            break
        sig = os.WSTOPSIG(status)
        exit_stop = (status >> 16) == 6
        if (exit_stop or sig != 5) and hw is None:
            hw = high_water()
        libc.ptrace(7, pid, None, ctypes.c_void_p(0 if exit_stop or sig == 5 else sig))
    return rc, out.read_bytes(), err.read_bytes(), hw or 0


def grown_source(combined: bytes, user_lo: int) -> bytes:
    """The compiler with 25% more functions: every 4th `pub fn` of its library
    again under a new name, placed before the driver."""
    at = combined.rfind((_REPO / DRIVER).read_bytes())
    if at <= user_lo:
        raise RuntimeError("driver source not found at the end of the compiler stream")
    fns = re.findall(r"(?ms)^(pub fn (\w+)\(.*?^}\n)", combined[user_lo:at].decode())
    extra = "".join(body.replace(f"pub fn {name}(", f"pub fn {name}_g25x(", 1)
                    for i, (body, name) in enumerate(fns) if i % 4 == 0 and name != "main")
    return combined[:at] + extra.encode() + b"\n" + combined[at:]


def mutant_source(combined: bytes, kind: str) -> bytes:
    """`recursive-walk`: const_find_from recurses once per token again.
    `no-precheck`: nb_stack_short admits everything and the lexer cap is off."""
    text = combined.decode()

    def swap(name: str, body: str | None, new: str | None = None) -> None:
        nonlocal text
        heads = [m.start() for m in re.finditer(rf"(?m)^pub fn {name}\(", text)]
        if len(heads) != 1:
            raise RuntimeError(f"{name}: {len(heads)} definitions")
        end = text.index("\n}\n", heads[0]) + 3
        head = text[heads[0]:text.index("\n", heads[0]) + 1]
        text = text[:heads[0]] + (new if new is not None else head + body + "}\n") + text[end:]

    if kind == "recursive-walk":
        swap("const_find_from", None, RECURSIVE_WALK)
    elif kind == "no-precheck":
        swap("nb_stack_short", "    0\n")
        swap("nb_stack_tok_cap", "    4611686018427387904\n")
    else:
        raise ValueError(kind)
    return text.encode()


def stack_rows(argv: list[str]) -> int:
    """Rows (each PASS/FAIL): the bounded compiler's stack high-water on
      self          the whole compiler
      growth25      the compiler with 25% more functions
      limit-user    a program of USER functions STACK_SLACK tokens under the limit
      beyond        the same shape past the limit: refused rc 3, stack-budget diagnostic
      unknown-item  one unrecognized top-level item as long as the limit allows
      inert-attrs   stacked `#[inline]` lines on one fn, as many as the limit allows
      semis         top-level `;` items, as many as the limit allows
    must be > 0 and <= the budget nb_stack_short gives that program (tokens x
    per-token bytes + longest token x per-unit bytes + base, constants read from
    nb_stack_budget.mind); and two mutants of the compiler source must each turn
    a row red: `recursive-walk` (growth25 passes its budget or dies) and
    `no-precheck` (the 4 MB identifier dies by a signal, not rc 3)."""
    from _cf_giant import long_token_program
    combined, stdin_v1, user_lo = build_seed()
    std = combined[:user_lo]
    std_tokens = token_stats(std)[0] - 1
    limit = stack_token_limit()
    print(f"stack-budget rows (stack={BOUNDED[1]} B, constants {stack_consts()}, per-token limit {limit} tokens, "
          f"std seed {std_tokens} tokens)")
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        if "--compiler" in argv:
            compiler = pathlib.Path(argv[argv.index("--compiler") + 1]).resolve()
        else:
            compiler = write_exec(td, "compiler.elf", derive_from_default(v2_header() + stdin_v1))

        def image(src: bytes, lo: int) -> bytes:
            return v2_header() + struct.pack("<qq", lo, len(src)) + src

        def row(name: str, src: bytes, lo: int, want_rc: int | None, exe=compiler, run_to: int | None = None) -> tuple:
            tokens, budget = stack_budget(src)
            rc, out, err, hw = traced(exe, image(src, lo), td, name.replace("/", "_"))
            ok = rc >= 0 and 0 < hw <= budget and (want_rc is None or rc == want_rc)
            ran = None
            if ok and run_to is not None:
                ran = capped([str(write_exec(td, name.replace("/", "_") + ".elf", out))], b"")[0] if is_static_elf(out) else None
                ok = ran == run_to
            report(ok, name, f"{tokens} tokens: rc={rc} out={len(out)} B high-water={hw} B <= budget {budget} B"
                   + (f"; program exit={ran} (want {run_to})" if run_to is not None else "")
                   + (f" stderr={err[:80]!r}" if rc != 0 else ""))
            return rc, out, err, hw, budget

        self_rc, self_out, _, _, _ = row("stack/self", combined, user_lo, 0)
        report(is_static_elf(self_out), "stack/self-elf", f"the compiler emitted itself: {len(self_out)} B")
        if "--self-only" in argv:  # one leg: this tree's compiler source (a scratch tree with another std seed)
            print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}" if FAILS else "ALL PASS")
            return 1 if FAILS else 0
        grown = grown_source(combined, user_lo)
        row("stack/growth25", grown, user_lo, 0)

        def fns(count: int) -> bytes:
            return "".join(f"fn zq{k}() -> i64 {{ 0 }}\n" for k in range(count)).encode() + b"fn main() -> i64 { 7 }\n"

        room = limit - STACK_SLACK - std_tokens - 16
        row("stack/limit-user", std + fns(room // 9), user_lo, 0, run_to=7)
        over = std + fns((limit + STACK_SLACK - std_tokens) // 9 + 1)
        rc, out, err, hw = traced(compiler, image(over, user_lo), td, "beyond")
        report(rc == 3 and out == b"" and b"stack budget" in err, "stack/beyond",
               f"{token_stats(over)[0]} tokens: rc={rc} out={len(out)} B stderr={err[:80]!r}")
        main_fn = b"fn main() -> i64 { 7 }\n"
        row("stack/unknown-item", std + b"zq" + b" q" * (room - 2) + b";\n" + main_fn, user_lo, None)
        row("stack/inert-attrs", std + b"#[inline]\n" * (room // 4) + main_fn, user_lo, None)
        row("stack/semis", std + b";\n" * room + main_fn, user_lo, None)

        # Mutants: each must turn its row red (a row that cannot fail checks nothing).
        ident = long_token_program("ident", 4000000)[0].encode()
        rc, out, err, hw = traced(compiler, image(std + ident, user_lo), td, "ident4m")
        report(rc == 3 and out == b"" and b"stack budget" in err, "stack/long-token",
               f"4 MB identifier: rc={rc} stderr={err[:80]!r}")
        for kind in ("recursive-walk", "no-precheck"):
            rc, out, err, _ = traced(compiler, image(mutant_source(combined, kind), user_lo), td, "mutant_" + kind)
            if rc != 0 or not is_static_elf(out):
                report(False, f"stack/mutant-{kind}", f"the mutant compiler did not build: rc={rc} stderr={err[:120]!r}")
                continue
            bad = write_exec(td, f"mutant_{kind}.elf", out)
            if kind == "recursive-walk":
                tokens, budget = stack_budget(grown)
                rc, out, err, hw = traced(bad, image(grown, user_lo), td, "mutant_walk_run")
                report(rc < 0 or hw > budget, f"stack/mutant-{kind}",
                       f"KILLED iff growth25 leaves its budget: rc={rc} high-water={hw} B, budget {budget} B")
            else:
                rc, out, err, hw = traced(bad, image(std + ident, user_lo), td, "mutant_ident_run")
                report(rc < 0, f"stack/mutant-{kind}",
                       f"KILLED iff the 4 MB identifier dies by a signal: rc={rc} high-water={hw} B")
    if FAILS:
        print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        return 1
    print("ALL PASS")
    return 0


def main(argv: list[str]) -> int:
    if "--stack-rows" in argv:
        return stack_rows(argv)
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
