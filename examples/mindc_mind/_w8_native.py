"""Shared harness for the W8 float smokes (self_host_native_f64_*_smoke.py).

Builds a stage1 compiler ELF from a main.mind by running the FROZEN pure-MIND seed
(testdata/selfhost_loop/stage1.elf) on [std blob + main.mind + selfhost_driver.mind]
-- no Rust, no MLIR in that step -- then feeds user programs to that stage1 directly
(stdin protocol [user_lo][src_len][std blob + user source], see selfhost_driver.mind)
and runs the emitted ELF. The MLIR oracle builds the same program with
`mindc build` (default MLIR backend) and runs it.

Env:
  W8_MAIN_MIND   main.mind to build stage1 from (default: this tree's main.mind).
                 Used to show RED on a base-tree copy.
  MINDC_BIN      mindc for the MLIR oracle (default: shared resolver).
"""
import hashlib
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _stdlib_manifest import seed_blob  # noqa: E402

FROZEN_SEED = HERE / "testdata" / "selfhost_loop" / "stage1.elf"
DRIVER = HERE / "selfhost_driver.mind"
CACHE = pathlib.Path(tempfile.gettempdir()) / "w8_native_stage1_cache"


def main_mind_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("W8_MAIN_MIND", str(HERE / "main.mind")))


def std_blob() -> bytes:
    return seed_blob()


def _write_exec(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    os.chmod(path, 0o755)
    return path


def build_stage1() -> pathlib.Path:
    """stage1 = frozen seed run on the chosen main.mind. Cached by input sha256."""
    blob = std_blob()
    combined = blob + main_mind_path().read_bytes() + b"\n" + DRIVER.read_bytes()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    key = hashlib.sha256(FROZEN_SEED.read_bytes() + img).hexdigest()[:24]
    CACHE.mkdir(exist_ok=True)
    out = CACHE / f"stage1_{key}.elf"
    if out.exists():
        return out
    r = subprocess.run([str(FROZEN_SEED)], input=img, capture_output=True, timeout=1200)
    if r.returncode != 0 or r.stdout[:4] != b"\x7fELF":
        raise RuntimeError(f"frozen seed failed to build stage1: rc={r.returncode} "
                           f"stderr={r.stderr[-400:]!r}")
    tmp = CACHE / f".{key}.{os.getpid()}"
    _write_exec(tmp, r.stdout)
    os.replace(tmp, out)
    return out


def native_compile(stage1: pathlib.Path, src: str, blob: bytes | None = None) -> tuple[int, bytes, str]:
    """Feed one user program to stage1. Returns (rc, elf_bytes, stderr_tail).

    `blob` overrides the std blob (default: the pinned seed blob)."""
    if blob is None:
        blob = std_blob()
    combined = blob + src.encode()
    img = struct.pack("<qq", len(blob), len(combined)) + combined
    r = subprocess.run([str(stage1)], input=img, capture_output=True, timeout=600)
    elf = r.stdout if r.returncode == 0 and r.stdout[:4] == b"\x7fELF" else b""
    return r.returncode, elf, r.stderr.decode(errors="replace")[-400:]


def run_elf_bytes(elf: bytes, workdir: pathlib.Path, name: str) -> tuple[int, bytes]:
    p = _write_exec(workdir / name, elf)
    r = subprocess.run([str(p)], capture_output=True, timeout=120)
    return r.returncode, r.stdout


def mindc_bin() -> str:
    if os.environ.get("MINDC_BIN"):
        return os.environ["MINDC_BIN"]
    from _selfhost_so import resolve_mindc
    return resolve_mindc()


def mlir_build_run(src: str, workdir: pathlib.Path, name: str) -> tuple[str, int, bytes]:
    """MLIR oracle: ('ok', rc, stdout) or ('build-failed', rc, stderr)."""
    f = workdir / f"{name}.mind"
    f.write_text(src)
    out = workdir / f"{name}.mlir.bin"
    b = subprocess.run([mindc_bin(), "build", str(f), "--release", "--out", str(out)],
                       capture_output=True, timeout=900, cwd=str(workdir))
    if b.returncode != 0 or not out.exists():
        return "build-failed", b.returncode, (b.stdout + b.stderr)[-600:]
    r = subprocess.run([str(out)], capture_output=True, timeout=120)
    return "ok", r.returncode, r.stdout


def bits_of(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def u64(v: int) -> int:
    return v & 0xFFFFFFFFFFFFFFFF


def words(stdout: bytes) -> list[int]:
    n = len(stdout) // 8
    return list(struct.unpack(f"<{n}q", stdout[: n * 8]))


def i64_lit(v: int) -> str:
    """A MIND i64 expression for v (INT64_MIN cannot be a positive literal)."""
    if v == -(1 << 63):
        return "((0 - 9223372036854775807) - 1)"
    if v < 0:
        return f"(0 - {-v})"
    return str(v)
