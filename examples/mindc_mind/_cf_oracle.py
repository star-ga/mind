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
                 and a corpus slice inside a child process whose glibc malloc
                 checks every free (libc_malloc_debug, glibc.malloc.check=3):
                 an invalid free aborts the child at once, on any input size.
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
import subprocess
import sys

MARK = "cf-oracle-child-ok"


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
                           capture_output=True, timeout=timeout, preexec_fn=lambda: os.nice(15))
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
    so = load_so(so_path)
    if mode == "heap":
        n = 0
        for src in [f.read_bytes() for f in fixtures] + corpus_srcs:
            emit(so, "selftest_native_elf_stage", blob, src)
            emit(so, "selftest_native_elf_u", blob, src)
            n += 1
        print(f"{MARK} heap programs={n}", flush=True)
        return 0
    if mode == "self":
        elf = emit(so, "selftest_native_elf_u", blob, self_src)
        pathlib.Path(out).write_bytes(elf)
        print(f"{MARK} self-emit bytes={len(elf)}", flush=True)
        return 0
    raise SystemExit(f"unknown child mode {mode}")
