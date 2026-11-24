"""Low-level helpers of self_host_native_cf_smoke.py: the stdin image, the
frozen-seed compiler derivations, the native/MLIR runners, the exec retry, the
verdict counter and the fixture header reader. Importing this module runs
nothing; the gate script owns the sections and the rows."""

import errno
import os
import pathlib
import re
import resource
import signal
import struct
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
from _selfhost_sources import compiler_library, self_mode_image, start_child  # noqa: E402
from _cf_giant import BOUNDED  # noqa: E402

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


def exec_run(argv, payload=None, **kw):
    """start_child for every exec this gate performs, retried while the target
    is busy (ETXTBSY). The gate writes ELFs (stage1, the bounded compiler, every
    program) and execs them from a thread pool; a sibling thread's fork holds a
    copy of a just-written file's write descriptor until its child execs, and an
    exec of that file in the window fails with ETXTBSY. The window closes within
    milliseconds, so the exec is retried; any other OSError, or a writer that
    never lets go, still fails the gate."""
    for _ in range(ETXTBSY_TRIES - 1):
        try:
            return start_child(argv, payload, **kw)
        except OSError as e:
            if e.errno != errno.ETXTBSY:
                raise
            _etxtbsy_retries[0] += 1
            time.sleep(ETXTBSY_WAIT)
    return start_child(argv, payload, **kw)


def derive_stage1(td: pathlib.Path) -> pathlib.Path:
    comb = BLOB + compiler_library() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    img = struct.pack("<qq", len(BLOB), len(comb)) + comb
    r = exec_run([str(SEED)], self_mode_image(img, SEED), capture_output=True, timeout=1800, preexec_fn=nice)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise SystemExit(f"BLOCKED: frozen seed could not compile this tree (rc={r.returncode}): "
                         f"{r.stderr[-300:]!r}")
    out = td / "stage1.elf"
    out.write_bytes(r.stdout)
    out.chmod(0o755)
    return out


CAP_CPU = 600  # CPU seconds per whole-program capacity compile
AS_CAP = 2 << 30  # the downstream consumer harness runs every compile and program under this


def v2_header() -> bytes:
    magic = int.from_bytes(b"MINDLAY\x01", "little", signed=True)
    return b"".join(int(v).to_bytes(8, "little", signed=True) for v in (magic, *BOUNDED))


def derive_stage1_bounded(td: pathlib.Path) -> pathlib.Path:
    """The bounded production compiler: the frozen DEFAULT seed given the v2 header
    emits this tree's compiler with the bounded layout (768 MiB arena, 384 MiB stack)."""
    comb = BLOB + compiler_library() + b"\n" + (_HERE / "selfhost_driver.mind").read_bytes()
    img = v2_header() + struct.pack("<qq", len(BLOB), len(comb)) + comb
    r = exec_run([str(SEED)], self_mode_image(img, SEED), capture_output=True, timeout=1800, preexec_fn=nice)
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
        r = exec_run([str(stage1b)], img, capture_output=True, timeout=900, preexec_fn=capped_cpu)
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
    r = exec_run([str(stage1)], image(src, seeded), capture_output=True, timeout=600,
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
