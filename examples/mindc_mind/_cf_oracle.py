#!/usr/bin/env python3
"""Oracle .so heap checks of self_host_native_cf_smoke (not a gate).

The oracle .so is main.mind built by mindc as a malloc-backed shared object.
In it __mind_free is glibc free(); in the native self-host compiler it is a
no-op on the bump arena. A free of a pointer __mind_alloc did not return is
therefore invisible to every native-ELF gate and corrupts the .so's heap:
count_nonparam_nodes freed the table nb_ntab returns (its allocation + 16), and
the .so aborted with "double free or corruption (!prev)" on this tree's own
source, so the self-host loop's oracle leg and every .so gate died with
SIGABRT. Three checks:

  free_lint      static: every __mind_free(X) in main.mind frees a name that
                 the same function bound to a __mind_alloc(...) result; a table
                 nb_ntab built is released only through nb_ntab_free.
  heap_child     runs the oracle's stage and IR-note entries on every fixture
                 and a corpus slice, one fresh child per entry pair. glibc
                 checks every free (libc_malloc_debug, glibc.malloc.check=3):
                 an invalid free aborts its child at once, on any input size.
  self_emit      the oracle's selftest_native_elf_u on this tree's combined
                 self-host source (the self-host loop's oracle leg), in the same
                 checked child; the ELF must equal the stage1 the frozen seed
                 derives from the same source.

Importing it runs nothing; the smoke's --so-child mode is the child entry.
"""
import ctypes
import glob
import os
import pathlib
import re
import resource
import subprocess
import sys

MARK = "cf-oracle-child-ok"
# Leave headroom for the parent inside its 2 GiB cgroup. An oracle allocation
# failure is a failed check, never a reason to exceed the caller's memory cap.
ORACLE_AS_LIMIT = 1536 << 20


def child_limits():
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (ORACLE_AS_LIMIT, ORACLE_AS_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def strip_comments(src: str) -> str:
    return re.sub(r"//[^\n]*", "", src)


def free_lint(main_src: str) -> list:
    """Violations of the __mind_free contract in main.mind (empty when clean)."""
    bad = []
    for chunk in re.split(r"^(?=pub fn |fn )", strip_comments(main_src), flags=re.M):
        m = re.match(r"(?:pub )?fn (\w+)", chunk)
        name = m.group(1) if m else "<top level>"
        for fm in re.finditer(r"__mind_free\(([^()]*)\)", chunk):
            arg = fm.group(1).strip()
            if name == "nb_ntab_free" and arg == "nodes - 16":
                continue  # the allocation nb_ntab made
            lm = re.search(r"let (?:mut )?" + re.escape(arg) + r"\s*:\s*\w+\s*=\s*([^;]*);", chunk)
            init = lm.group(1).strip() if lm else ""
            if not re.fullmatch(r"\w+", arg) or not init.startswith("__mind_alloc("):
                bad.append(f"{name}: __mind_free({arg}) frees {init or 'an unbound value'!r}, "
                           f"not a __mind_alloc result")
    return bad


def malloc_debug_lib() -> str:
    """glibc's malloc-check library (glibc >= 2.34), or '' when absent."""
    for pat in ("/lib/*-linux-gnu/libc_malloc_debug.so.0", "/usr/lib/*-linux-gnu/libc_malloc_debug.so.0",
                "/lib64/libc_malloc_debug.so.0", "/usr/lib64/libc_malloc_debug.so.0"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return ""


def run_checked_child(smoke: pathlib.Path, mode: str, so: str, out: pathlib.Path, timeout: int = 600):
    """Run `smoke --so-child MODE SO OUT` with glibc checking every free.
    Returns (ok, detail)."""
    lib = malloc_debug_lib()
    if not lib:
        return False, "libc_malloc_debug.so.0 not found: the heap check cannot run (fail closed)"
    env = dict(os.environ, LD_PRELOAD=lib, GLIBC_TUNABLES="glibc.malloc.check=3")
    try:
        r = subprocess.run([sys.executable, str(smoke), "--so-child", mode, so, str(out)], env=env,
                           capture_output=True, timeout=timeout, preexec_fn=child_limits)
    except subprocess.TimeoutExpired:
        return False, f"child timed out after {timeout} s"
    tail = (r.stdout + r.stderr).decode(errors="replace").strip().splitlines()[-3:]
    ok = r.returncode == 0 and any(line.startswith(MARK) for line in tail)
    return ok, f"rc={r.returncode} {' | '.join(tail)}"


def load_so(path: str):
    so = ctypes.CDLL(path)
    for fn in ("selftest_native_elf_stage", "selftest_native_elf_u"):
        getattr(so, fn).restype = ctypes.c_int64
        getattr(so, fn).argtypes = [ctypes.c_int64] * 3
    return so


def emit(so, entry: str, blob: bytes, src: bytes) -> bytes:
    comb = blob + src
    buf = ctypes.create_string_buffer(comb, len(comb))
    es = getattr(so, entry)(ctypes.cast(buf, ctypes.c_void_p).value, len(comb), len(blob))
    rd = lambda a, o=0: ctypes.cast(a + o, ctypes.POINTER(ctypes.c_int64))[0]  # noqa: E731
    sh = rd(es, 0)
    if not sh or rd(sh, 8) <= 0:
        return b""
    return ctypes.string_at(rd(sh, 0), rd(sh, 8))


def child(mode: str, so_path: str, out: str, blob: bytes, fixtures, corpus_srcs, self_src: bytes) -> int:
    """The child body (runs under the malloc check)."""
    programs = [f.read_bytes() for f in fixtures] + corpus_srcs
    if mode == "heap":
        # The malloc-backed research oracle does not reclaim its native bump-
        # allocator's scratch between requests. Preserve every listed case and
        # both entry calls, but give each pair a fresh checked process. Keeping
        # the whole corpus in one process exceeded the 2 GiB cgroup.
        for i in range(len(programs)):
            ok, why = run_checked_child(pathlib.Path(sys.argv[0]), f"heap:{i}", so_path,
                                       pathlib.Path(out), timeout=120)
            if not ok:
                print(f"FAIL oracle heap case={i}: {why}", flush=True)
                return 1
        print(f"{MARK} heap programs={len(programs)} isolated=true", flush=True)
        return 0
    so = load_so(so_path)
    if mode.startswith("heap:"):
        i = int(mode.split(":", 1)[1])
        if i < 0 or i >= len(programs):
            raise SystemExit(f"invalid heap case index {i}")
        emit(so, "selftest_native_elf_stage", blob, programs[i])
        emit(so, "selftest_native_elf_u", blob, programs[i])
        print(f"{MARK} heap case={i} entries=2", flush=True)
        return 0
    if mode == "stage-file":
        src = pathlib.Path(out).with_suffix(".mind").read_bytes()
        elf = emit(so, "selftest_native_elf_stage", blob, src)
        pathlib.Path(out).write_bytes(elf)
        print(f"{MARK} stage bytes={len(elf)}", flush=True)
        return 0
    if mode == "self":
        elf = emit(so, "selftest_native_elf_u", blob, self_src)
        pathlib.Path(out).write_bytes(elf)
        print(f"{MARK} self-emit bytes={len(elf)}", flush=True)
        return 0
    raise SystemExit(f"unknown child mode {mode}")


def twin_child(so_path: str, out: str, blob: bytes, self_src: bytes) -> int:
    """Write the twin's refusal count, then each refused function's name."""
    so = ctypes.CDLL(so_path)
    fn = so.selftest_twin_refusals
    fn.restype = ctypes.c_int64
    fn.argtypes = [ctypes.c_int64] * 5
    comb = blob + self_src
    buf = ctypes.create_string_buffer(comb, len(comb))
    cap = 64
    spans = (ctypes.c_int64 * cap)()
    n = fn(ctypes.cast(buf, ctypes.c_void_p).value, len(comb), len(blob), ctypes.addressof(spans), cap)
    names = []
    for i in range(min(max(n, 0), cap)):
        m = re.match(rb"[A-Za-z_][A-Za-z_0-9]*", comb[spans[i]:spans[i] + 128])
        names.append(m.group(0).decode() if m else f"@{spans[i]}")
    pathlib.Path(out).write_text("\n".join([str(n)] + names) + "\n")
    print(f"{MARK} twin refusals={n}", flush=True)
    return 0
