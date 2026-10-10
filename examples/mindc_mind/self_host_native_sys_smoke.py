#!/usr/bin/env python3
"""Linux x86-64 filesystem parity: current-source stage1 versus Rust/MLIR.

Never updates bootstrap fixtures. Derives stage1 from this tree's main.mind
using the frozen pure-MIND seed. Every program runs in a fresh temporary
working directory, with independent expected exit/stdout and disk assertions.
MINDC_BIN selects the freshly built Rust/MLIR oracle.
"""
import hashlib
import os
import pathlib
import platform
import struct
import subprocess
import tempfile

from _selfhost_so import resolve_mindc
from _stdlib_manifest import seed_blob
from self_host_loop_smoke import build_seed, _FROZEN

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def c_string(name: str, value: bytes) -> str:
    body = [f"fn {name}() -> i64 {{", f"    let p: i64 = __mind_alloc({len(value) + 1});"]
    body += [f"    __mind_store_i8(p + {i}, {b});" for i, b in enumerate(value + b"\0")]
    return "\n".join(body + ["    p", "}", ""])


PATHS = c_string("input_path", b"input") + c_string("old_path", b"data") + c_string("new_path", b"renamed")
POSITIONED = PATHS + """
fn read_at(fd: i64, p: i64, off: i64) -> i64 {
    return __mind_read(fd, p, 1, off);
}
fn main() -> i64 {
    let fd: i64 = __mind_open(input_path());
    let p: i64 = __mind_alloc(8);
    let mut off: i64 = 3;
    let mut total: i64 = 0;
    let mut packed: i64 = 0;
    while off >= 0 {
        total = total + read_at(fd, p, off);
        packed = packed * 256 + __mind_load_i8(p);
        off = off - 1;
    }
    if total != 4 { return 10; }
    if packed != 1145258561 { return 11; }
    if read_at(fd, p, 0 - 1) != 1 { return 12; }
    if __mind_load_i8(p) != 65 { return 13; }
    if read_at(fd, p, 0 - 2) != 1 { return 14; }
    if __mind_load_i8(p) != 66 { return 15; }
    if __mind_write(1, p, 1, 0 - 1) != 1 { return 16; }
    return 0;
}
"""
FILESYSTEM = PATHS + """
fn write_at(fd: i64, p: i64, off: i64) -> i64 {
    return __mind_write(fd, p, 1, off);
}
fn main() -> i64 {
    let old: i64 = old_path();
    let new: i64 = new_path();
    let p: i64 = __mind_alloc(256);
    let saved: i64 = umask(63);
    let fd: i64 = open(old, 578, 438);
    if fd < 0 { return 20; }
    if umask(saved) != 63 { return 21; }
    if getuid() != EXPECTED_UID { return 22; }
    if fstat(fd, p) != 0 { return 23; }
    if (__mind_load_i32(p + 24) & 511) != 384 { return 24; }
    if fcntl(fd, 1, 0) != 0 { return 25; }
    if fcntl(fd, 2, 1) != 0 { return 26; }
    if fcntl(fd, 1, 0) != 1 { return 27; }
    if flock(fd, 6) != 0 { return 28; }
    __mind_store_i8(p, 65);
    if write_at(fd, p, 0 - 1) != 1 { return 29; }
    let mut off: i64 = 3;
    let mut total: i64 = 0;
    while off >= 0 {
        __mind_store_i8(p, 97 + off);
        total = total + write_at(fd, p, off);
        off = off - 1;
    }
    if total != 4 { return 30; }
    if lseek(fd, 0, 1) != 1 { return 31; }
    __mind_store_i8(p, 90);
    if write_at(fd, p, 0 - 1) != 1 { return 32; }
    if fsync(fd) != 0 { return 33; }
    if fstat(fd, p) != 0 { return 34; }
    if __mind_load_i64(p + 48) != 4 { return 35; }
    if fstatat(0 - 100, old, p, 0) != 0 { return 36; }
    if __mind_load_i64(p + 48) != 4 { return 37; }
    if flock(fd, 8) != 0 { return 38; }
    if close(fd) != 0 { return 39; }
    if renameat(0 - 100, old, 0 - 100, new) != 0 { return 40; }
    let rd: i64 = openat(0 - 100, new, 0, 0);
    if rd < 0 { return 41; }
    if __mind_read(rd, p, 4, 0) != 4 { return 42; }
    if __mind_load_i32(p) != 1684232801 { return 43; }
    if __mind_write(1, p, 4, 0 - 1) != 4 { return 44; }
    if close(rd) != 0 { return 45; }
    if unlinkat(0 - 100, new, 0) != 0 { return 46; }
    return 0;
}
"""
ERRORS = PATHS + """
fn main() -> i64 {
    let p: i64 = __mind_alloc(256);
    let path: i64 = old_path();
    if close(0 - 1) != 0 - 9 { return 50; }
    if fstat(0 - 1, p) != 0 - 9 { return 51; }
    if fsync(0 - 1) != 0 - 9 { return 52; }
    if flock(0 - 1, 2) != 0 - 9 { return 53; }
    if fcntl(0 - 1, 1, 0) != 0 - 9 { return 54; }
    if lseek(0 - 1, 0, 0) != 0 - 9 { return 55; }
    if open(path, 0, 0) != 0 - 2 { return 56; }
    if openat(0 - 100, path, 0, 0) != 0 - 2 { return 57; }
    if fstatat(0 - 100, path, p, 0) != 0 - 2 { return 58; }
    if unlinkat(0 - 100, path, 0) != 0 - 2 { return 59; }
    if renameat(0 - 100, path, 0 - 100, new_path()) != 0 - 2 { return 60; }
    if posix_errno(0 - 9) != 9 { return 61; }
    if posix_errno(0) != 0 { return 62; }
    if posix_errno(7) != 0 { return 63; }
    // Existing read/write keep the C twin's -1 error sentinel and no-op guards.
    if __mind_read(0 - 1, p, 1, 0) != 0 - 1 { return 64; }
    if __mind_write(0 - 1, p, 1, 0 - 1) != 0 - 1 { return 65; }
    if __mind_read(0 - 1, 0, 1, 0) != 0 { return 66; }
    if __mind_write(0 - 1, p, 0 - 1, 0) != 0 { return 67; }
    return 0;
}
"""


def run_elf(exe: pathlib.Path, source: bytes) -> bytes:
    # Full compiler derivation can exceed two minutes on shared CI hosts.
    result = subprocess.run([str(exe)], input=source, capture_output=True, timeout=600)
    if result.returncode or not result.stdout.startswith(b"\x7fELF"):
        raise RuntimeError(f"{exe.name}: exit={result.returncode}, stderr={result.stderr[:300]!r}")
    elf = result.stdout
    if struct.unpack_from("<HH", elf, 16) != (2, 62):
        raise RuntimeError("native output is not x86-64 ET_EXEC")
    phoff = struct.unpack_from("<Q", elf, 32)[0]
    entsize, count = struct.unpack_from("<HH", elf, 54)
    for i in range(count):
        if struct.unpack_from("<I", elf, phoff + i * entsize)[0] in (2, 3):
            raise RuntimeError("native output has a dynamic loader/dependency segment")
    return elf


def main() -> int:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        print("FAIL: requires Linux x86-64")
        return 1
    mindc = resolve_mindc()
    _, seed, _ = build_seed()
    failed = False
    checked = 0
    with tempfile.TemporaryDirectory(prefix="mind-sys-") as td:
        tmp = pathlib.Path(td)
        stage1 = tmp / "stage1.elf"
        stage1.write_bytes(run_elf(_FROZEN, seed))
        stage1.chmod(0o755)
        print(f"stage1 from current main.mind: sha256={hashlib.sha256(stage1.read_bytes()).hexdigest()}", flush=True)
        for name, body, expected, posix in [
            ("positioned_read", POSITIONED, b"B", False),
            ("filesystem", FILESYSTEM, b"aZcd", True),
            ("errors", ERRORS, b"", True),
        ]:
            try:
                surface = (REPO / "std/posix.mind").read_text() if posix else ""
                prefix = seed_blob()
                source = prefix + (surface + body.replace("EXPECTED_UID", str(os.getuid()))).encode()
                native = tmp / f"{name}.elf"
                native.write_bytes(run_elf(stage1, struct.pack("<qq", len(prefix), len(source)) + source))
                native.chmod(0o755)
                src = tmp / f"{name}.mind"
                src.write_text(("import std.posix;\n" if posix else "") + body.replace("EXPECTED_UID", str(os.getuid())))
                mlir = tmp / f"{name}.mlir-bin"
                build = subprocess.run([str(mindc), "--allow-nondeterministic", "build", str(src), "--release", "--backend", "mlir",
                                        "--emit=binary", f"--out={mlir}"],
                                       cwd=tmp, capture_output=True, timeout=300)
                if build.returncode:
                    raise RuntimeError(f"MLIR build exit={build.returncode}: {build.stderr[-2000:]!r}")
                results = []
                for backend, exe in [("native", native), ("mlir", mlir)]:
                    work = tmp / f"{name}-{backend}"
                    work.mkdir()
                    (work / "input").write_bytes(b"ABCD")
                    proc = subprocess.run([str(exe)], cwd=work, capture_output=True, timeout=15)
                    checked += 1
                    results.append((proc.returncode, proc.stdout))
                    ok = results[-1] == (0, expected) and not (work / "data").exists() and not (work / "renamed").exists()
                    if ok:
                        print(f"PASS {name}/{backend}: exit={proc.returncode} stdout={proc.stdout!r}", flush=True)
                    else:
                        print(f"FAIL {name}/{backend}: exit={proc.returncode} stdout={proc.stdout!r}", flush=True)
                    failed |= not ok
                if results[0] != results[1]:
                    raise RuntimeError(f"backend mismatch: {results!r}")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                print(f"FAIL {name}: {exc}", flush=True)
                failed = True
    if checked != 6:
        print(f"FAIL: checked {checked} backend cases, expected 6")
        failed = True
    print("FAIL native syscall parity" if failed else "ALL PASS native syscall parity")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
