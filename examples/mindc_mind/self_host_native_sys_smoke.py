#!/usr/bin/env python3
"""Linux x86-64 filesystem parity: current-source stage1 versus Rust/MLIR.

Never updates bootstrap fixtures. Derives stage1 from this tree's main.mind
using the frozen pure-MIND seed. Every program runs in a fresh temporary
working directory, with independent expected exit/stdout and disk assertions.
MINDC_BIN selects the freshly built Rust/MLIR oracle.

The seed-neutral checks guard the self-host fixed point: read/write calls with
a literal -1 offset, and every emitter function this substrate changed, must
compile to the same bytes (code and trace note) under the frozen seed and the
current stage1.
"""
import hashlib
import os
import pathlib
import platform
import re
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
fn write_dyn(fd: i64, p: i64, n: i64, off: i64) -> i64 {
    return __mind_write(fd, p, n, off);
}
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
    // Non-sentinel offsets keep the C twin's -1 error sentinel and no-op guards.
    if __mind_read(0 - 1, p, 1, 0) != 0 - 1 { return 64; }
    if write_dyn(0 - 1, p, 1, 0 - 1) != 0 - 1 { return 65; }
    if __mind_read(0 - 1, 0, 1, 0) != 0 { return 66; }
    if __mind_write(0 - 1, p, 0 - 1, 0) != 0 { return 67; }
    // A literal -1 offset is the bootstrap stream sequence: failure is negative
    // on both backends (raw errno natively, -1 from the C twin).
    if __mind_write(0 - 1, p, 1, 0 - 1) >= 0 { return 68; }
    return 0;
}
"""
# Literal -1 offsets in every spelling, and nothing else. The frozen seed and
# the current stage1 must emit this program byte for byte (code and trace note).
SENTINEL_ONLY = """
fn main() -> i64 {
    let p: i64 = __mind_alloc(8);
    __mind_store_i8(p, 65);
    let a: i64 = __mind_write(1, p, 1, 0 - 1);
    let b: i64 = __mind_write(1, p, 1, -1);
    let c: i64 = __mind_write(1, p, 1, (0 - 1));
    let d: i64 = __mind_read(0, p, 1, 0 - 1);
    let e: i64 = __mind_read(0, p, 1, 1 - 2);
    return a + b + c + d + e;
}
"""
# Sentinel and non-sentinel sites side by side: 3 legacy writes, 1 legacy read,
# 3 runtime dispatches (a runtime offset, the constant -2, the constant 0).
SENTINEL_MIXED = """
fn put(p: i64, off: i64) -> i64 {
    return __mind_write(1, p, 1, off);
}
fn main() -> i64 {
    let p: i64 = __mind_alloc(8);
    __mind_store_i8(p, 65);
    if __mind_write(1, p, 1, 0 - 1) != 1 { return 1; }
    __mind_store_i8(p, 66);
    if __mind_write(1, p, 1, -1) != 1 { return 2; }
    __mind_store_i8(p, 67);
    if __mind_write(1, p, 1, (0 - 1)) != 1 { return 3; }
    __mind_store_i8(p, 68);
    if put(p, 0 - 1) != 1 { return 4; }
    __mind_store_i8(p, 69);
    if __mind_write(1, p, 1, 0 - 2) != 1 { return 5; }
    if __mind_read(0 - 1, p, 1, 1 - 2) >= 0 { return 6; }
    if __mind_read(0 - 1, p, 1, 0) != 0 - 1 { return 7; }
    return 0;
}
"""
LEGACY_IO = {
    nr: re.compile(rb"\x48\x8b\xbd.{4}\x48\x8b\xb5.{4}\x48\x8b\x95.{4}\x48\xc7\xc0"
                   + bytes([nr]) + rb"\x00\x00\x00\x0f\x05", re.DOTALL)
    for nr in (0, 1)
}
DISPATCH_IO = re.compile(rb"\x31\xc0\x48\x85\xf6\x74\x22\x48\x85\xd2\x7e\x1d\xb8")
# Every main.mind function this lane adds or changes is reachable from these
# roots. Their transitive closure, compiled by the frozen seed and by stage1,
# must give identical ELFs: any shape the seed's emitters or trace-note builder
# treat differently breaks the self-host fixed point (stage1 != stage2).
PARITY_ROOTS = [
    "nb_expr", "nb_emit_intrinsic", "nb_intrinsic_id", "nb_intrinsic_site_kind",
    "tc_is_std_surface_intrinsic", "nb_emit_write_stream",
]
_ITEM = re.compile(r"^(?:pub )?(?:fn|struct|enum|const)\s+([A-Za-z_][A-Za-z0-9_]*)")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def closure_program(source: str, roots: list[str]) -> tuple[str, list[str]]:
    """(program, missing_roots): the roots' transitive top-level closure plus a
    main that references every present root behind a runtime-false guard."""
    lines = source.split("\n")
    items: dict[str, str] = {}
    order: list[str] = []
    i = 0
    while i < len(lines):
        m = _ITEM.match(lines[i])
        if not m:
            i += 1
            continue
        j = i
        if not lines[i].rstrip().endswith(";"):
            while not lines[j].startswith("}"):
                j += 1
        items[m.group(1)] = "\n".join(lines[i:j + 1])
        order.append(m.group(1))
        i = j + 1
    missing = [r for r in roots if r not in items]
    need: set[str] = set()
    stack = [r for r in roots if r in items]
    while stack:
        name = stack.pop()
        if name in need:
            continue
        need.add(name)
        body = re.sub(r"//[^\n]*", "", items[name])
        stack.extend(t for t in set(_IDENT.findall(body)) if t in items and t not in need)
    calls = []
    for k, root in enumerate(r for r in roots if r in items):
        params = re.search(r"fn\s+" + root + r"\((.*?)\)", items[root]).group(1)
        args = ["string_new()" if p.split(":")[1].strip() == "String" else "0"
                for p in params.split(",") if p.strip()]
        calls.append(f"        let _r{k} = {root}({', '.join(args)});")
    main = ("fn main() -> i64 {\n    let z: i64 = __mind_argc();\n    if z == 99 {\n"
            + "\n".join(calls) + "\n    }\n    return 0;\n}\n")
    return "\n\n".join(items[n] for n in order if n in need) + "\n\n" + main, missing


def emit(exe: pathlib.Path, user: str) -> bytes:
    prefix = seed_blob()
    source = prefix + user.encode()
    return run_elf(exe, struct.pack("<qq", len(prefix), len(source)) + source)


def seed_neutral(stage1: pathlib.Path, tmp: pathlib.Path, mindc) -> list[str]:
    """Failures of the bootstrap byte-neutrality checks (empty list = pass)."""
    fails: list[str] = []
    seed_bytes, new_bytes = emit(_FROZEN, SENTINEL_ONLY), emit(stage1, SENTINEL_ONLY)
    if seed_bytes != new_bytes:
        fails.append(f"sentinel_seed_bytes: stage1 {len(new_bytes)}B != frozen seed {len(seed_bytes)}B")
    else:
        print(f"PASS sentinel_seed_bytes: stage1 == frozen seed ({len(new_bytes)}B)", flush=True)
    mixed = emit(stage1, SENTINEL_MIXED)
    counts = (len(LEGACY_IO[1].findall(mixed)), len(LEGACY_IO[0].findall(mixed)),
              len(DISPATCH_IO.findall(mixed)))
    if counts != (3, 1, 3):
        fails.append(f"sentinel_dispatch: legacy write/read, dispatch = {counts}, expected (3, 1, 3)")
    else:
        print(f"PASS sentinel_dispatch: legacy write/read, dispatch = {counts}", flush=True)
    native = tmp / "sentinel_mixed.elf"
    native.write_bytes(mixed)
    native.chmod(0o755)
    src = tmp / "sentinel_mixed.mind"
    src.write_text(SENTINEL_MIXED)
    mlir = tmp / "sentinel_mixed.mlir-bin"
    build = subprocess.run([str(mindc), "--allow-nondeterministic", "build", str(src), "--release",
                            "--backend", "mlir", "--emit=binary", f"--out={mlir}"],
                           cwd=tmp, capture_output=True, timeout=300)
    if build.returncode:
        fails.append(f"sentinel_mixed: MLIR build exit={build.returncode}: {build.stderr[-500:]!r}")
    else:
        for backend, exe in [("native", native), ("mlir", mlir)]:
            proc = subprocess.run([str(exe)], cwd=tmp, capture_output=True, timeout=15)
            if (proc.returncode, proc.stdout) != (0, b"ABCDE"):
                fails.append(f"sentinel_mixed/{backend}: exit={proc.returncode} stdout={proc.stdout!r}")
            else:
                print(f"PASS sentinel_mixed/{backend}: exit=0 stdout={proc.stdout!r}", flush=True)
    program, missing = closure_program((HERE / "main.mind").read_text(), PARITY_ROOTS)
    if missing:
        fails.append(f"emitter_seed_parity: roots missing from main.mind: {missing}")
    seed_elf, new_elf = emit(_FROZEN, program), emit(stage1, program)
    if seed_elf != new_elf:
        part = "code" if seed_elf[:-32] != new_elf[:-32] else "trace note"
        fails.append(f"emitter_seed_parity: stage1 != frozen seed ({part} differs; "
                     f"{len(new_elf)}B vs {len(seed_elf)}B)")
    else:
        print(f"PASS emitter_seed_parity: {len(PARITY_ROOTS)} roots, stage1 == frozen seed "
              f"({len(new_elf)}B, sha256={hashlib.sha256(new_elf).hexdigest()[:16]})", flush=True)
    return fails


def run_elf(exe: pathlib.Path, source: bytes) -> bytes:
    # Full compiler derivation can exceed two minutes on shared CI hosts.
    result = subprocess.run(
        [str(exe)], input=source, capture_output=True,
        timeout=int(os.environ.get("MIND_SELFHOST_TIMEOUT_SECS", "600")),
    )
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
        try:
            neutral_fails = seed_neutral(stage1, tmp, mindc)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            neutral_fails = [f"seed_neutral: {exc}"]
        for line in neutral_fails:
            print(f"FAIL {line}", flush=True)
        failed |= bool(neutral_fails)
    if checked != 6:
        print(f"FAIL: checked {checked} backend cases, expected 6")
        failed = True
    print("FAIL native syscall parity" if failed else "ALL PASS native syscall parity")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
