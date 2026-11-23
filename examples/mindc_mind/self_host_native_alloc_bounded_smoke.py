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

Allocator runtime (testdata/native_alloc_bounded/rt/): bounded images start
with one shared runtime blob (alloc/free/realloc, boundary-tag merging, bins
with a non-empty bitmap, top-of-heap return, in-place realloc growth). The gate
assembles alloc_rt.s with the system assembler, (a) runs the C stress harness
alloc_rt_harness.c over it (random alloc/free/realloc traces against a model
with a full heap walk after every operation, the C-runtime edge cases and the
grow/free/interleaved loops) and (b) checks that B2 and the first fixture image
start with exactly that blob, patched for the bounded layout (the byte chunks
in main.mind are generated from it by rt/gen_mind.py). Missing as/objcopy or a
C compiler prints SKIP for those checks only.

Default-profile rows (testdata/native_alloc_default/): each program is fed
directly (v1 header) to the compiler under test and the image must have the class
that nb_alloc_select.mind chooses: RECLAIM starts with the runtime blob (default
layout constants, overflow base = arena base) and BUMP does not and equals the base
compiler's image. Each image runs to its expected exit status and output; a RECLAIM
image holds its peak resident size under 64 MiB, ends a full arena with exit status
70 and the line `error: native arena exhausted`, and repeats its result on a second
run (a row that ends with 70 on a second and a third run, the count of blocks included). With MINDC_BIN set, the portable rows also run through the MLIR backend
(a cross-check, not the reference). A REFUSED row is a program with a free in a loop that the
native lowering refuses: exit status 3, no image, and exactly the generic refusal line. A row of STD_UNIT_ROWS is a std module's source with a driver after it
(the module is no native seed module); its image's own peak resident size (VmHWM) stays under the row's bound.

Reclaim: the compiler's own arena mark/reclaim code (nb_arena_reclaim_free_list
and helpers, extracted from main.mind by name) is compiled with a driver and
run natively over the runtime: after each reclaim the cursor must sit exactly
at the mark (no stale word between blocks), the heap must tile, and no binned
block may survive above the checkpoint (exit 42; 1..12 name the failed check).

Peak VmPeak (sampled from /proc/<pid>/status) and ru_maxrss (wait4) are
printed for every compile and run.

Flags:
  --compiler PATH  skip the derivation and use PATH as the bounded compiler
                   (e.g. the pinned testdata/selfhost_loop_bounded/stage1.elf)
  --base-compiler PATH  a compiler minted from the base sources: the default-profile
                   rows of class BUMP must give its image byte for byte
  --mutant NAME    mint a compiler from a scratch copy of the sources with ONE planted change
                   (MUTANTS in _selfhost_alloc_rows.py), run the default-profile rows with it and exit 0 only when
                   a row named for that change in MUTANT_ROWS goes red (the mutant is killed)
  --keep DIR       copy the derived B1/B2/B3 compilers into DIR
Exit 0 = ALL PASS, 1 = a FAIL, 2 = BLOCKED (missing fixture/seed).
"""

import hashlib
import os
import pathlib
import re
import resource
import signal
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
from _selfhost_alloc_rows import MUTANT_ROWS, MUTANTS, generated_row  # noqa: E402
from _selfhost_sources import capture_bundle, self_mode_image, start_child  # noqa: E402

AS_CAP = 2 * 1024**3
LAYOUT_MAGIC = int.from_bytes(b"MINDLAY\x01", "little")
BOUNDED = (768 << 20, 384 << 20, 1 << 20)  # arena, stack, guard
DEFAULT_SEED = _HERE / "testdata" / "selfhost_loop" / "stage1.elf"
FIX = _HERE / "testdata" / "native_alloc_bounded"
OOM_DIAG = b"error: native arena exhausted\n"
REFUSED_DIAG = b"error[backend-native]: unsupported construct; use --backend=mlir\n"
# A whole self-compile takes ~1 min on an idle machine but more than 15 min
# on a heavily shared one; the derivation stages get an hour each so load
# alone cannot kill them (rc -9) and turn the gate red.
DERIVE_TIMEOUT = 3600
# name -> (exit, stdout, portable: also checked against the MLIR backend)
CASES = {
    "r18_free_loop": (33, b"", True),
    "alloc_nonpos": (40, b"np\n", True),
    "grow_big": (30, b"big\n", True),
    "vec_grow": (7, b"vec\n", True),
    "vec_std": (9, b"std\n", True),
    "oom_exit": (70, b"", False),
    "afl_inc": (12, b"", True),
    "grow8": (11, b"", True),
    "big_increasing": (15, b"", True),
    "realloc_null_zero": (104, b"", True),
    "interleaved_growth": (21, b"", True),
    "bin_fit_search": (23, b"", True),
    "realloc_free_neighbour": (25, b"", True),
    "dup_struct_name": (7, b"", False),
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
    (rc, stdout, stderr, wall_s, vmpeak_kib, maxrss_kib, vmhwm_kib)."""
    def limit() -> None:
        os.nice(15)
        if cap:
            resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
    t0 = time.monotonic()
    p = start_child(argv, stdin, popen=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, preexec_fn=limit)
    peak = [0, 0]  # VmPeak, VmHWM (KiB) of the running image
    stop = threading.Event()

    def sample() -> None:
        status = pathlib.Path(f"/proc/{p.pid}/status")
        while not stop.is_set():
            try:
                for row in status.read_text().splitlines():
                    for i, key in enumerate(("VmPeak:", "VmHWM:")):
                        if row.startswith(key):
                            peak[i] = max(peak[i], int(row.split()[1]))
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
    return (p.returncode, out[0], err[0], time.monotonic() - t0, peak[0], ru.ru_maxrss, peak[1])


def v2_image(user: bytes) -> bytes:
    """[v2 hdr][user_lo][src_len][std prefix + user]: a user program. Compiler source goes through v2_compiler_image."""
    combined, _, user_lo = build_seed()
    return _v2_frame(combined[:user_lo] + user, user_lo)


def v2_compiler_image() -> bytes:
    """The v2 image of the whole compiler source: send it through self_mode_image."""
    combined, _, user_lo = build_seed()
    return _v2_frame(combined, user_lo)


def _v2_frame(combined: bytes, user_lo: int) -> bytes:
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
    rc, out, err, wall, vmpeak, rss, hwm = r
    return (f"{tag} rc={rc} out={len(out)}B wall={wall:.1f}s "
            f"VmPeak={vmpeak} KiB maxrss={rss} KiB VmHWM={hwm} KiB")


def derive(td: pathlib.Path) -> pathlib.Path | None:
    img = v2_compiler_image()
    r = run([str(DEFAULT_SEED)], self_mode_image(img, DEFAULT_SEED), cap=False, timeout=DERIVE_TIMEOUT)
    print("        " + describe("B1 (frozen default seed + v2, uncapped)", r), flush=True)
    if r[0] != 0 or not is_static_elf(r[1]):
        report(False, "derive/B1", f"rc={r[0]} err={r[2][-200:]!r}")
        return None
    exe, stages, peaks = write_exec(td, "B1.elf", r[1]), [], []
    for k in (2, 3):
        r = run([str(exe)], self_mode_image(img, exe), timeout=DERIVE_TIMEOUT)
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


# ---- the allocator runtime blob (testdata/native_alloc_bounded/rt/) ----
RT = FIX / "rt"
ARENA_ADDR = 0x100_0000
LOAD_ADDR = 0x40_0000


def rt_layout() -> dict[int, int]:
    """Placeholder byte -> value for the bounded layout (see alloc_rt.s)."""
    arena, stack, guard = BOUNDED
    st = ARENA_ADDR + arena - 4096
    ob = ARENA_ADDR + arena + guard + stack + guard
    return {1: ARENA_ADDR, 2: ARENA_ADDR + 8, 3: st, 4: ob, 5: ARENA_ADDR + 16, 6: ob + 8}


def rt_assemble(td: pathlib.Path) -> bytes | None:
    """Assemble alloc_rt.s with the system assembler; None when unavailable."""
    obj, binf = td / "alloc_rt.o", td / "alloc_rt.bin"
    try:
        a = subprocess.run(["as", "--64", "-o", str(obj), str(RT / "alloc_rt.s")],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if a.returncode != 0:
            report(False, "rt/assemble", a.stderr.decode(errors="replace")[-300:])
            return None
        o = subprocess.run(["objcopy", "-O", "binary", "-j", ".text", str(obj), str(binf)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except FileNotFoundError:
        print("  SKIP  rt/assemble: no GNU as/objcopy on PATH")
        return None
    if o.returncode != 0:
        report(False, "rt/assemble", o.stderr.decode(errors="replace")[-300:])
        return None
    return binf.read_bytes()


def rt_layout_default() -> dict[int, int]:
    """The same for the default layout: arena 1 GiB, no overflow region (OB = A)."""
    st = ARENA_ADDR + (1 << 30) - 4096
    return {1: ARENA_ADDR, 2: ARENA_ADDR + 8, 3: st, 4: ARENA_ADDR, 5: ARENA_ADDR + 16, 6: ARENA_ADDR + 8}


def rt_patch(raw: bytes, vals: dict[int, int] | None = None) -> bytes:
    vals, out, i = vals or rt_layout(), bytearray(raw), 0
    while i + 8 <= len(out):
        if out[i] in vals and out[i + 1:i + 8] == b"\x11" * 7:
            out[i:i + 8] = struct.pack("<Q", vals[out[i]])
            i += 8
        else:
            i += 1
    return bytes(out)


def check_blob(tag: str, elf: bytes, raw: bytes | None, vals: dict[int, int] | None = None) -> None:
    """The emitted image must start (at its ELF entry) with the assembled,
    layout-patched runtime blob, byte-for-byte."""
    if raw is None:
        return
    want = rt_patch(raw, vals)
    entry = struct.unpack_from("<Q", elf, 24)[0]
    off = entry - LOAD_ADDR
    got = elf[off:off + len(want)]
    first = next((k for k in range(len(want)) if k >= len(got) or got[k] != want[k]), None)
    report(first is None, f"rt/blob/{tag}",
           f"{len(want)} B at entry 0x{entry:x}" + ("" if first is None else
           f" first difference at blob offset {first}"))


def run_rt_capped(exe: pathlib.Path, binf: pathlib.Path) -> None:
    """Default layout: the blob patched with OB = A. 40 allocations of a quarter arena
    (10 arena sizes) must fit by reclaiming; the next loop exhausts the arena and the
    blob must end the process with exit status 70 and its diagnostic."""
    r = run([str(exe), str(binf), hex(0x1000_0000), "0", "0", "1", "cap"], b"", cap=False, timeout=600)
    report(r[0] == 70 and r[2] == OOM_DIAG and b"CAP reclaim OK" in r[1], "rt/harness/capped",
           f"rc={r[0]} stdout={r[1][-60:]!r} stderr={r[2][-80:]!r} (want rc=70 and the exhaustion line)")


def run_rt_harness(td: pathlib.Path, raw: bytes | None) -> None:
    """Compile and run the C stress harness over the assembled blob: random
    alloc/free/realloc traces against a model, with a full heap walk (tiling,
    boundary tags, bins, bitmap, no free block below the cursor) after every
    operation, plus the C-runtime edge cases and the grow/free/interleaved loops."""
    if raw is None:
        return
    binf, exe = td / "alloc_rt_raw.bin", td / "alloc_rt_harness"
    binf.write_bytes(raw)
    cc = os.environ.get("CC", "cc")
    try:
        c = subprocess.run([cc, "-O1", "-o", str(exe), str(RT / "alloc_rt_harness.c")],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    except FileNotFoundError:
        print(f"  SKIP  rt/harness: no C compiler ({cc})")
        return
    if c.returncode != 0:
        report(False, "rt/harness-build", c.stderr.decode(errors="replace")[-300:])
        return
    run_rt_capped(exe, binf)
    # (arena bytes, steps, size mode, seed): small arenas force the overflow region.
    for arena, steps, mode, seed in ((0x100_0000, 20000, 0, 1), (0x100_0000, 20000, 0, 2),
                                     (0x1000_0000, 20000, 0, 3), (0x400_0000, 3000, 1, 4)):
        r = run([str(exe), str(binf), hex(arena), str(steps), str(mode), str(seed)], b"",
                cap=False, timeout=600)
        tail = r[1].decode(errors="replace").strip().splitlines()[-1:] or [""]
        report(r[0] == 0 and tail[0].startswith("OK"),
               f"rt/harness/arena={arena:#x},seed={seed}",
               f"rc={r[0]} wall={r[3]:.1f}s {tail[0]} {r[2].decode(errors='replace')[-200:]}")


# ---- arena mark/reclaim (the compiler's own nb_arena_reclaim_free_list) ----
# The fixture is the REAL main.mind code (extracted by name) plus a driver: it
# runs natively over the runtime blob and checks the cursor after each reclaim
# (exactly the mark's position: no stale word between blocks), the heap tiling
# and that no binned block survives above the checkpoint.
RECLAIM_FNS = ("nb_load_addr", "nb_arena_addr", "nb_running_arena_bytes",
               "nb_running_arena_scan", "nb_arena_reclaim_free_list", "nb_rfl_purge",
               "nb_rfl_purge_list", "nb_rfl_settle", "nb_rfl_find", "nb_rfl_find_list",
               "nb_rfl_unlink", "nb_rfl_set_prev", "nb_rfl_bsr")
RECLAIM_DRIVER = """
fn mark() -> i64 {
    nb_arena_addr() + __mind_load_i64(nb_arena_addr()) + 16
}
fn cur() -> i64 {
    __mind_load_i64(nb_arena_addr())
}
fn want(m: i64) -> i64 {
    m - nb_arena_addr() - 16
}
fn bits() -> i64 {
    __mind_load_i64(nb_arena_addr() + nb_running_arena_bytes() - 4096 + 520)
}
fn walk(h: i64, top: i64, pf: i64) -> i64 {
    if h == top {
        if pf == 1 {
            return 0;
        }
        return 1;
    }
    if h > top {
        return 0;
    }
    let hd: i64 = __mind_load_i64(h);
    let s: i64 = hd & (0 - 8);
    if ((hd / 2) & 1) != pf {
        return 0;
    }
    let fr: i64 = hd & 1;
    if fr == 1 {
        if __mind_load_i64(h + s) != s {
            return 0;
        }
    }
    walk(h + 8 + s, top, fr)
}
fn heap_ok() -> i64 {
    walk(nb_arena_addr() + 8, nb_arena_addr() + 8 + cur(), 0)
}
fn main() -> i64 {
    let k1: i64 = __mind_alloc(100);
    let m1: i64 = mark();
    let a: i64 = __mind_alloc(1000);
    let b: i64 = __mind_alloc(64);
    let fa: i64 = __mind_free(a);
    let r1: i64 = nb_arena_reclaim_free_list(m1);
    if cur() != want(m1) {
        return 1;
    }
    if heap_ok() != 1 {
        return 2;
    }
    if ((bits() / 512) & 1) != 0 {
        return 3;
    }
    let c: i64 = __mind_alloc(5000);
    if c != m1 {
        return 4;
    }
    let fc: i64 = __mind_free(c);
    let p1: i64 = __mind_alloc(200);
    let m2: i64 = mark();
    let q: i64 = __mind_alloc(300);
    let fp: i64 = __mind_free(p1);
    let r2: i64 = nb_arena_reclaim_free_list(m2);
    if cur() != want(p1) {
        return 5;
    }
    if heap_ok() != 1 {
        return 6;
    }
    let u: i64 = __mind_alloc(64);
    let m3: i64 = mark();
    let fu: i64 = __mind_free(u);
    let w: i64 = __mind_alloc(500);
    let z: i64 = __mind_alloc(64);
    let fw: i64 = __mind_free(w);
    let r3: i64 = nb_arena_reclaim_free_list(m3);
    if cur() != want(w) {
        return 7;
    }
    if heap_ok() != 1 {
        return 8;
    }
    let m4: i64 = mark();
    let v: i64 = __mind_alloc(64);
    let fv: i64 = __mind_free(v);
    let r4: i64 = nb_arena_reclaim_free_list(m4);
    if cur() != want(m4) {
        return 9;
    }
    let m5: i64 = mark();
    let e1: i64 = __mind_alloc(2000);
    let e2: i64 = __mind_alloc(64);
    let e3: i64 = __mind_alloc(3000);
    let e4: i64 = __mind_alloc(64);
    let f1: i64 = __mind_free(e1);
    let f3: i64 = __mind_free(e3);
    let r5: i64 = nb_arena_reclaim_free_list(m5);
    if ((bits() / 1024) & 3) != 0 {
        return 10;
    }
    if cur() != want(m5) {
        return 11;
    }
    if heap_ok() != 1 {
        return 12;
    }
    __mind_store_i64(k1, 30);
    __mind_load_i64(k1) + 12 + fa + fc + fp + fu + fw + fv + f1 + f3 + r1 + r2 + r3 + r4 + r5 + b * 0 + q * 0 + z * 0 + e2 * 0 + e4 * 0
}
"""


def extract_fns(src: str, names) -> str:
    """Top-level `pub fn NAME(` ... first column-0 `}` for each name."""
    lines, out = src.splitlines(), []
    for name in names:
        hits = [i for i, row in enumerate(lines) if row.startswith(f"pub fn {name}(")]
        if len(hits) != 1:
            raise KeyError(f"{name}: {len(hits)} definitions")
        end = next(j for j in range(hits[0], len(lines)) if lines[j] == "}")
        out.append("\n".join(lines[hits[0]:end + 1]))
    return "\n".join(out) + "\n"


def reclaim_fixture() -> bytes:
    src = (_HERE / "main.mind").read_text()
    return (extract_fns(src, RECLAIM_FNS) + RECLAIM_DRIVER).encode()


DEFAULT_FIX = FIX.parent / "native_alloc_default"
# name -> (class, exit status, stdout, arguments, also checked through the MLIR backend).
# 1,100 x 1 MB overflows a 1 GiB bump arena, so a RECLAIM row that runs is the proof.
_R = ("RECLAIM", 33, b"", (), True)
DEFAULT_ROWS = {
    **{name: _R for name in (
        "arm_let", "arm_expr_stmt", "arm_assign", "arm_binop", "arm_paren", "arm_call_arg", "arm_neg",
        "arm_not", "arm_if_cond", "arm_if_then", "arm_if_else", "arm_value_if", "arm_return",
        "arm_while_cond", "arm_while_cond_only", "arm_while_body", "arm_struct_lit", "arm_array_lit", "arm_index",
        "arm_index_assign", "arm_field_assign", "arm_field", "arm_cast", "arm_for_body", "calls1",
        "calls2", "calls3", "std_sha256_loop", "mutual_recursion", "both_ways", "after_early_return",
        "realloc_free_loop", "realloc_growth", "arm_enum_ctor", "ctl_directive_late",
        "dyn_free_loop")},
    # C malloc does not hand out zeroed memory: the zero-read check has no MLIR leg
    "align_zero": ("RECLAIM", 33, b"", (), False),
    "argv_free_loop": ("RECLAIM", 33, b"", ("a", "b"), False),
    "interleaved_free": ("RECLAIM", 235, b"", (), True),
    "exhaust_live": ("RECLAIM", 70, b"", (), False),
    # a libc call that allocates (getaddrinfo) first: the arena still stops at the same count, one byte per block
    "libc_first_exhaust": ("RECLAIM", 70, b"." * 1073, (), False),
    # RFC 7748 section 5.2: k after 1000 iterations (STD_UNIT_ROWS: std/x25519.mind in the user unit)
    "x25519_chain": ("RECLAIM", 0, bytes.fromhex("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51"), (), False),
    **{name: ("BUMP", 33, b"", (), True) for name in (
        "ctl_no_free", "ctl_free_outside", "ctl_for_end_bound", "ctl_directive", "ctl_both_directives",
        "ctl_recursion_no_loop")},
    # generated rows have no fixture file and no MLIR leg (portable is false: there is no file for the
    # MLIR oracle to read). cell_zero dirties the bump and calls the tree's nbwm_cell. The two
    # dereference-assignment rows store through an allocated pointer; the free is the stored value.
    # The other generated rows name one allocator decision each: a blocked realloc, the directive's
    # exact spelling, a full arena filled from a literal site, a mutual recursion with no free, and a
    # walk over a pre-dirtied mark.
    "cell_zero": ("BUMP", 33, b"", (), False),
    **{name: ("RECLAIM", 33, b"", (), False) for name in (
        "arm_deref_assign", "arm_deref_assign_callee",
        "realloc_blocked", "ctl_bumpy", "ctl_bumx", "ctl_after_code")},
    **{name: ("RECLAIM", 70, b"", (), False) for name in (
        "exhaust_struct", "exhaust_array", "exhaust_field")},
    **{name: ("BUMP", 33, b"", (), False) for name in (
        "ctl_crlf", "mutrec_nofree", "walk_mark")},
    # the two ends of a full bump arena; doc_bump_ends() holds the doc to the signal of each
    "bump_overshoot": ("BUMP", -4, b"", (), False),
    "bump_exact_fill": ("BUMP", -11, b"", (), False),
    # a free stored through a reference: the native lowering refuses the prefix `&`, so no image exists
    "ref_store_free": ("REFUSED", 3, b"", (), False),
}
GENERATED_ROWS = (
    "cell_zero", "arm_deref_assign", "arm_deref_assign_callee", "realloc_blocked", "ctl_bumpy", "ctl_bumx", "ctl_after_code",
    "ctl_crlf", "mutrec_nofree", "exhaust_struct", "exhaust_array", "exhaust_field", "walk_mark",
    "bump_overshoot", "bump_exact_fill", "ref_store_free",
)
RECLAIM_MAXRSS_KIB = 65536
# row -> (std module, driver composed after it, bound of the image's VmHWM in KiB; x25519_chain measured 2080 to 2088)
STD_UNIT_ROWS = {"x25519_chain": ("std/x25519.mind", "tests/native_alloc_std_drivers/x25519_chain_driver.mind", 4096)}


def row_source(name: str, src_dir: pathlib.Path) -> bytes:
    if name in GENERATED_ROWS:
        return generated_row(name, src_dir)
    if name in STD_UNIT_ROWS:
        return b"\n".join((_REPO / path).read_bytes() for path in STD_UNIT_ROWS[name][:2])
    return (DEFAULT_FIX / f"{name}.mind").read_bytes()


def program_image(source: bytes) -> bytes:
    """The v1 (default layout) direct-feed protocol: [user_lo][src_len][std prefix + program]."""
    std = capture_bundle(_REPO).std_prefix
    return struct.pack("<qq", len(std), len(std) + len(source)) + std + source


def entry_bytes(elf: bytes, n: int) -> bytes:
    entry = struct.unpack_from("<Q", elf, 24)[0]
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    for i in range(struct.unpack_from("<H", elf, 56)[0]):
        ptype, _, off, vaddr, _, filesz, _, _ = struct.unpack_from("<IIQQQQQQ", elf, phoff + i * 56)
        if ptype == 1 and vaddr <= entry < vaddr + filesz:
            return elf[off + entry - vaddr:off + entry - vaddr + n]
    return b""


def default_rows(comp: pathlib.Path, base: pathlib.Path | None, td: pathlib.Path, raw: bytes | None,
                 src_dir: pathlib.Path = _HERE) -> None:
    prologue = b"\x49\xb9" + struct.pack("<Q", rt_layout_default()[3])
    names = sorted([p.stem for p in DEFAULT_FIX.glob("*.mind")] + list(GENERATED_ROWS) + list(STD_UNIT_ROWS))
    drivers = {_REPO / row[1] for row in STD_UNIT_ROWS.values()}  # a driver directory holds these drivers only
    if sorted(DEFAULT_ROWS) != names or {q for d in drivers for q in d.parent.glob("*.mind")} != drivers:
        report(False, "default/rows", f"fixtures {names} or drivers {sorted(map(str, drivers))} differ from the row table")
        return
    for name, (cls, want_rc, want_out, args, portable) in DEFAULT_ROWS.items():
        src = DEFAULT_FIX / f"{name}.mind"
        source = row_source(name, src_dir)
        c = run([str(comp)], program_image(source), cap=False, timeout=900)
        if cls == "REFUSED":
            report((c[0], c[1], c[2]) == (want_rc, b"", REFUSED_DIAG), f"default/{name}",
                   describe("compile", c) + f" err={c[2][-80:]!r} (want a refusal, exit status {want_rc})")
            continue
        if c[0] != 0 or not c[1].startswith(b"\x7fELF"):
            report(False, f"default/{name}", describe("compile", c) + f" err={c[2][-160:]!r}")
            continue
        elf = c[1]
        report((entry_bytes(elf, 10) == prologue) == (cls == "RECLAIM"), f"default/{name}/class",
               f"{cls}: entry bytes {entry_bytes(elf, 10).hex()} (the runtime prologue is {prologue.hex()})")
        if cls == "RECLAIM" and name == "arm_let":
            check_blob("default-RECLAIM", elf, raw, rt_layout_default())
        if cls == "BUMP" and base is None:
            print(f"  NOTE  default/{name}/base-identity: no --base-compiler given; not asserted")
        if cls == "BUMP" and base is not None:
            b = run([str(base)], program_image(source), cap=False, timeout=900)
            report(b[1] == elf, f"default/{name}/base-identity",
                   f"{len(elf)} B, sha256={sha(elf)[:16]} vs base {len(b[1])} B sha256={sha(b[1])[:16]}")
        exe = write_exec(td, f"default_{name}.elf", elf)
        r = run([str(exe), *args], b"", cap=False, timeout=900)
        want_err = OOM_DIAG if want_rc == 70 else b""
        ok = (r[0], r[1], r[2]) == (want_rc, want_out, want_err)
        if cls == "RECLAIM":
            # a child starts with the harness's own resident size; 1 MB blocks that stay live touch whole huge pages
            if want_rc != 70:
                ok = ok and r[5] < resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + RECLAIM_MAXRSS_KIB * (4 if name == "dyn_free_loop" else 1)
            if name in STD_UNIT_ROWS:  # the image's own peak, sampled; 0 = the sampler never read it
                ok = ok and 0 < r[6] < STD_UNIT_ROWS[name][2]
            for _ in range(2 if want_rc == 70 else 1):  # a row that ends at the cap repeats its result in 3 of 3 runs
                r2 = run([str(exe), *args], b"", cap=False, timeout=900)
                ok = ok and (r2[0], r2[1], r2[2]) == (r[0], r[1], r[2])
        report(ok, f"default/{name}/run", describe("run", r) + f" stdout={r[1]!r} stderr={r[2][-60:]!r} (want rc={want_rc})")
        if portable and name != "dyn_free_loop":
            m = mlir_oracle(src, td)
            if m is None:
                print(f"  SKIP  default/{name}/mlir: MINDC_BIN not set")
            elif m[0] == "build-failed":
                print(f"  NOTE  default/{name}/mlir: the MLIR backend refuses the build (rc={m[1]}); no cross-check")
            else:
                report((m[0], m[1]) == (want_rc, want_out), f"default/{name}/mlir", describe("MLIR run", m) + f" stdout={m[1]!r}")


def selfcompile_default(comp: pathlib.Path) -> None:
    """The compiler's own source carries the opt-out line: compiled with the default layout it is a
    BUMP image (no runtime prologue at its entry), so the compiler stays on the bump allocator."""
    combined, _, user_lo = build_seed()
    c = run([str(comp)], self_mode_image(struct.pack("<qq", user_lo, len(combined)) + combined, comp),
            cap=False, timeout=DERIVE_TIMEOUT)
    if c[0] != 0 or not c[1].startswith(b"\x7fELF"):
        report(False, "default/selfcompile", describe("compile", c) + f" err={c[2][-160:]!r}")
        return
    head = entry_bytes(c[1], 2)
    report(head != b"\x49\xb9", "default/selfcompile/class",
           f"the compiler's own image ({len(c[1])} B, sha256={sha(c[1])[:16]}) starts with {head.hex()}; no runtime prologue (49b9) expected")


def mutant_run(name: str) -> int:
    """Mint a compiler from a scratch copy of the tree with the planted change, run the default-profile rows
    (and the self-compile class check) with it. 0 = a row went red (killed), 1 = every row passed (survived)."""
    import shutil
    path, old, new = MUTANTS[name]
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        scratch = td / "repo"
        (scratch / "examples/mindc_mind/testdata/selfhost_loop").mkdir(parents=True)
        shutil.copytree(_REPO / "std", scratch / "std")
        shutil.copy2(_REPO / "Mind.toml", scratch / "Mind.toml")
        for f in _HERE.glob("*.mind"):
            shutil.copy2(f, scratch / "examples/mindc_mind" / f.name)
        shutil.copy2(DEFAULT_SEED, scratch / "examples/mindc_mind/testdata/selfhost_loop/stage1.elf")
        target = scratch / "examples/mindc_mind" / path
        text = target.read_text()
        if text.count(old) != 1:
            print(f"BLOCKED  mutant {name}: the planted text matches {text.count(old)} times in {path}")
            return 2
        target.write_text(text.replace(old, new))
        from _selfhost_sources import mint_stage2
        mint_stage2(td / "mutant.elf", repo=scratch)
        default_rows(td / "mutant.elf", None, td, None, scratch / "examples/mindc_mind")
        if name in ("directive_ignored", "select_always"):
            selfcompile_default(td / "mutant.elf")
    red = [f for f in FAILS if f.startswith("default/")]
    hit = [f for f in red if f.split("/")[1] in MUTANT_ROWS[name]]
    print(f"MUTANT {name}: {'KILLED' if hit else 'SURVIVED'} (a row of {sorted(MUTANT_ROWS[name])} must go red; "
          f"{len(hit)} of {len(red)} red verdicts are in that set: {', '.join(hit[:6])}{' ...' if len(hit) > 6 else ''})")
    return 0 if hit else 1


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
    r = subprocess.run([str(pathlib.Path(mindc).resolve()), "build", "--backend", "mlir", src.name,
                        "--emit", "binary", "--out", str(art)],
                       cwd=proj, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if r.returncode != 0 or not art.exists():
        return ("build-failed", r.returncode, r.stderr.decode(errors="replace")[-200:])
    return run([str(art)], b"")


_PARTS_SITES = ("main.mind", "native_emission_context.mind", "native_scoped_source_image.mind")


def parts_record_fits() -> bool:
    """The route and both development drivers allocate 8 bytes past the largest offset that any
    compiler module reads or writes through `parts + K` / `finished + K` (comments do not count)."""
    code = {p.name: re.sub(r"//[^\n]*", "", p.read_text()) for p in _HERE.glob("*.mind")}
    need = 8 + max(int(k) for t in code.values() for k in re.findall(r"\b(?:parts|finished) \+ (\d+)\b", t))
    return all(re.findall(r"\blet parts: i64 = __mind_alloc\((\d+)\);", code[n]) == [str(need)] for n in _PARTS_SITES)


def doc_bump_ends() -> bool:
    """The BUMP bullet of docs/native-closure-lane.md names the signal that each bump end row wants."""
    text = " ".join((_REPO / "docs" / "native-closure-lane.md").read_text().split())
    bullet = text.partition(" - BUMP: ")[2].partition(" - RECLAIM: ")[0]
    sig = {n: signal.Signals(-DEFAULT_ROWS[n][1]).name for n in ("bump_overshoot", "bump_exact_fill")}
    return (f"does not fit runs a trap instruction ({sig['bump_overshoot']})" in bullet
            and f"one more allocation dies on the guard gap ({sig['bump_exact_fill']})" in bullet)


def main(argv: list[str]) -> int:
    if not parts_record_fits():
        print("FAIL parts record: the route and both development drivers must allocate the largest offset + 8")
        return 1
    if not doc_bump_ends():
        print("FAIL doc bump ends: the BUMP bullet must name the signal of bump_overshoot and bump_exact_fill")
        return 1
    if "--mutant" in argv:
        return mutant_run(argv[argv.index("--mutant") + 1])
    print(f"native bounded allocator gate (arena={BOUNDED[0]} stack={BOUNDED[1]} "
          f"guard={BOUNDED[2]}, AS cap={AS_CAP})", flush=True)
    for name in CASES:
        if not (FIX / f"{name}.mind").exists():
            print(f"  BLOCKED  missing fixture {name}.mind")
            return 2
    with tempfile.TemporaryDirectory() as tdn:
        td = pathlib.Path(tdn)
        raw = rt_assemble(td)
        run_rt_harness(td, raw)
        base = pathlib.Path(argv[argv.index("--base-compiler") + 1]).resolve() if "--base-compiler" in argv else None
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
            # B2 was emitted by the tree's emitter: it starts with the blob.
            check_blob("B2", (td / "B2.elf").read_bytes(), raw)
        for name, (want_rc, want_out, portable) in CASES.items():
            src = FIX / f"{name}.mind"
            c = run([str(comp)], v2_image(src.read_bytes()))
            if c[0] != 0 or not c[1].startswith(b"\x7fELF"):
                report(False, f"case/{name}", describe("compile", c) + f" err={c[2][-160:]!r}")
                continue
            exe = write_exec(td, f"{name}.elf", c[1])
            if name == next(iter(CASES)):
                check_blob(name, c[1], raw)
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
        c = run([str(comp)], self_mode_image(v2_image(reclaim_fixture()), comp))
        if c[0] != 0 or not c[1].startswith(b"\x7fELF"):
            report(False, "reclaim", describe("compile", c) + f" err={c[2][-160:]!r}")
        else:
            r = run([str(write_exec(td, "reclaim.elf", c[1]))], b"")
            report((r[0], r[1], r[2]) == (42, b"", b""), "reclaim",
                   describe("run", r) + " (want rc=42; 1..12 name the failed check)")
        default_rows(comp, base, td, raw)
        selfcompile_default(comp)
    if FAILS:
        print(f"FAIL ({len(FAILS)}): {', '.join(FAILS)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
