#!/usr/bin/env python3
"""Linux x86-64 process/clock/directory syscall parity, in disposable directories.

Checks observable behavior, never equality of pids or clock readings. Native
ELFs are emitted by current-source stage1 and must have no dynamic loader.
"""
import hashlib
import pathlib
import struct
import subprocess
import tempfile

from _selfhost_so import resolve_mindc
from _stdlib_manifest import seed_blob
from self_host_loop_smoke import build_seed, _FROZEN
from self_host_native_sys_smoke import run_elf, c_string

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

CASES = {
    'pipe_dup': ('''fn main() -> i64 {
    let p: i64 = __mind_alloc(32);
    if pipe2(p, 524288) != 0 { return 1; }
    let rd: i64 = __mind_load_i32(p);
    let wr: i64 = __mind_load_i32(p + 4);
    if dup2(wr, 99) != 99 { return 2; }
    __mind_store_i8(p + 8, 81);
    if __mind_write(99, p + 8, 1, 0 - 1) != 1 { return 3; }
    if __mind_read(rd, p + 9, 1, 0 - 1) != 1 { return 4; }
    if __mind_load_i8(p + 9) != 81 { return 5; }
    if close_range(99, 99, 0) != 0 { return 6; }
    if close(99) != 0 - 9 { return 7; }
    close(rd); close(wr);
    if pipe2(p, 0 - 1) != 0 - 22 { return 8; }
    if dup2(0 - 1, 99) != 0 - 9 { return 9; }
    if close_range(100, 99, 0) != 0 - 22 { return 10; }
    return 0;
}''', 0),
    'fork_wait': ('''fn main() -> i64 {
    let status: i64 = __mind_alloc(128);
    let child: i64 = fork();
    if child < 0 { return 1; }
    if child == 0 { _exit(37); return 99; }
    if waitpid(child, status, 0) != child { return 2; }
    if __mind_load_i32(status) != 9472 { return 3; }
    if waitpid(child, status, 0) != 0 - 10 { return 4; }
    let second: i64 = fork();
    if second < 0 { return 5; }
    if second == 0 { _exit(29); return 99; }
    if waitid(1, second, status, 4) != 0 { return 6; }
    if __mind_load_i32(status) != 17 { return 7; }
    if __mind_load_i32(status + 24) != 29 { return 8; }
    if __mind_load_i32(status + 16) != second { return 10; }
    if waitid(1, second, status, 4) != 0 - 10 { return 9; }
    return 0;
}''', 0),
    'process_control': ('''fn main() -> i64 {
    let p: i64 = __mind_alloc(128);
    let child: i64 = fork();
    if child < 0 { return 1; }
    if child == 0 {
        if setpgid(0, 0) != 0 { _exit(11); }
        if getpgid(0) <= 0 { _exit(12); }
        if kill(0, 0) != 0 { _exit(13); }
        if prctl(38, 1, 0, 0, 0) != 0 { _exit(14); }
        if prctl(39, 0, 0, 0, 0) != 1 { _exit(15); }
        if signal(17, 0) < 0 { _exit(16); }
        if signal(10, 1) < 0 { _exit(17); }
        if signal(10, 0) != 1 { _exit(18); }
        _exit(0);
    }
    if waitpid(child, p, 0) != child { return 2; }
    if __mind_load_i32(p) != 0 { return 3; }
    if kill(child, 0) != 0 - 3 { return 4; }
    if getpgid(child) != 0 - 3 { return 5; }
    if setpgid(child, 0) != 0 - 3 { return 6; }
    if prctl(0 - 1, 0, 0, 0, 0) != 0 - 22 { return 7; }
    if signal(9, 0) != 0 - 22 { return 8; }
    if signal(17, 2) != 0 - 22 { return 9; }
    return 0;
}''', 0),
    'kill_child': ('''fn main() -> i64 {
    let p: i64 = __mind_alloc(32);
    if signal(15, 0) < 0 { return 1; }
    let child: i64 = fork();
    if child < 0 { return 2; }
    if child == 0 {
        __mind_store_i64(p, 10);
        __mind_store_i64(p + 8, 0);
        nanosleep(p, 0);
        _exit(91);
    }
    let sent: i64 = kill(child, 15);
    if waitpid(child, p, 0) != child { return 3; }
    if sent != 0 { return 4; }
    if __mind_load_i32(p) != 15 { return 5; }
    return 0;
}''', 0),
    'clock_limits': ('''fn main() -> i64 {
    let p: i64 = __mind_alloc(128);
    if getrlimit(7, p) != 0 { return 1; }
    if setrlimit(7, p) != 0 { return 2; }
    if getrlimit(0 - 1, p) != 0 - 22 { return 3; }
    if setrlimit(0 - 1, p) != 0 - 22 { return 4; }
    if clock_gettime(1, p) != 0 { return 5; }
    let before: i64 = __mind_load_i64(p) * 1000000000 + __mind_load_i64(p + 8);
    __mind_store_i64(p + 16, 0);
    __mind_store_i64(p + 24, 1000000);
    if nanosleep(p + 16, 0) != 0 { return 6; }
    if clock_gettime(1, p) != 0 { return 7; }
    let after: i64 = __mind_load_i64(p) * 1000000000 + __mind_load_i64(p + 8);
    if after - before < 1000000 { return 8; }
    if clock_gettime(0 - 1, p) != 0 - 22 { return 9; }
    __mind_store_i64(p + 24, 1000000000);
    if nanosleep(p + 16, 0) != 0 - 22 { return 10; }
    return 0;
}''', 0),
    'directory_memfd': (c_string('path', b'.') + '''fn main() -> i64 {
    let fd: i64 = open(path(), 65536, 0);
    if fd < 0 { return 1; }
    if fchdir(fd) != 0 { return 2; }
    if fchdir(0 - 1) != 0 - 9 { return 3; }
    let d: i64 = fdopendir(fd);
    if d <= 0 { return 4; }
    let mut n: i64 = 0;
    let mut entry: i64 = readdir(d);
    while entry > 0 {
        n = n + 1;
        entry = readdir(d);
    }
    if entry != 0 { return 5; }
    if n != 402 { return 6; }
    if readdir(d) != 0 { return 7; }
    if closedir(d) != 0 { return 8; }
    if close(fd) != 0 - 9 { return 9; }
    if fdopendir(0 - 1) != 0 - 9 { return 10; }
    let mem: i64 = memfd_create(path(), 1);
    if mem < 0 { return 11; }
    if fdopendir(mem) != 0 - 20 { return 12; }
    if memfd_create(path(), 0 - 1) != 0 - 22 { return 13; }
    if fexecve(0 - 1, 0, 0) != 0 - 9 { return 14; }
    if fexecve(mem, 0, 0) != 0 - 8 { return 15; }
    close(mem);
    return 0;
}''', 0),
    'exec_exit': (c_string('path', b'child') + '''fn main() -> i64 {
    let p: i64 = __mind_alloc(32);
    __mind_store_i64(p, path());
    __mind_store_i64(p + 8, 0);
    __mind_store_i64(p + 16, 0);
    let fd: i64 = open(path(), 0, 0);
    if fd < 0 { return 1; }
    let child: i64 = fork();
    if child < 0 { return 2; }
    if child == 0 { fexecve(fd, p, p + 16); _exit(98); }
    if waitpid(child, p + 24, 0) != child { return 3; }
    if __mind_load_i32(p + 24) != 0 { return 4; }
    close(fd);
    _exit(23);
    return 99;
}''', 23),
}


def check(stage1, mindc, tmp):
    checked = 0
    failed = False
    prefix = seed_blob()
    surface = (REPO / 'std/posix.mind').read_bytes()
    for name, (body, expected) in CASES.items():
        try:
            source = prefix + surface + body.encode()
            native = tmp / (name + '.elf')
            native.write_bytes(run_elf(stage1, struct.pack('<qq', len(prefix), len(source)) + source))
            native.chmod(0o755)
            src = tmp / (name + '.mind')
            src.write_text('import std.posix;\n' + body)
            mlir = tmp / (name + '.mlir')
            result = subprocess.run([str(mindc), '--allow-nondeterministic', 'build', str(src), '--release', '--backend', 'mlir', '--emit=binary', f'--out={mlir}'], cwd=tmp, capture_output=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f'MLIR build exit={result.returncode}: {result.stderr[-2000:]!r}')
            results = []
            for backend, exe in [('native', native), ('mlir', mlir)]:
                work = tmp / (name + '-' + backend)
                work.mkdir()
                if name == 'directory_memfd':
                    for i in range(400):
                        (work / f'entry-{i:04d}').write_bytes(b'')
                if name == 'exec_exit':
                    child_source = prefix + b'fn main() -> i64 { return 0; }\n'
                    child = work / 'child'
                    child.write_bytes(run_elf(stage1, struct.pack('<qq', len(prefix), len(child_source)) + child_source))
                    child.chmod(0o755)
                proc = subprocess.run([str(exe)], cwd=work, capture_output=True, timeout=20)
                value = (proc.returncode, proc.stdout, proc.stderr)
                results.append(value)
                checked += 1
                ok = value == (expected, b'', b'')
                failed |= not ok
                if ok:
                    print(f"PASS {name}/{backend}: {value!r}", flush=True)
                else:
                    print(f"FAIL {name}/{backend}: {value!r}", flush=True)
            if results[0] != results[1]:
                raise RuntimeError(f'backend mismatch: {results!r}')
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f'FAIL {name}: {exc}', flush=True)
            failed = True
    failed |= checked != 2 * len(CASES)
    print(f'{"FAIL" if failed else "ALL PASS"} process syscall parity: {checked}/{2 * len(CASES)} backend cases', flush=True)
    return int(failed)


def main():
    if len(CASES) != 7:
        print('FAIL: expected seven process cases', flush=True)
        return 1
    with tempfile.TemporaryDirectory(prefix='mind-process-') as td:
        tmp = pathlib.Path(td)
        _, seed, _ = build_seed()
        stage1 = tmp / 'stage1.elf'
        stage1.write_bytes(run_elf(_FROZEN, seed))
        stage1.chmod(0o755)
        print('stage1 sha256=' + hashlib.sha256(stage1.read_bytes()).hexdigest(), flush=True)
        return check(stage1, resolve_mindc(), tmp)


if __name__ == '__main__':
    raise SystemExit(main())
