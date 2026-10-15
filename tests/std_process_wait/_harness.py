# Author: STARGA Inc.
# Shared helpers for the std/process.mind native acceptance cases.
# Split out of accept.py; the helper bodies are unchanged.

import ctypes
import os
import signal

LOGFD = 200000
I64 = ctypes.c_int64


def fdset():
    # listdir opens /proc/<pid>/fd and, on CPython, may dup that directory
    # while the names are collected. Both show up in the listing and then
    # close. Skip that target so a snapshot does not invent or hide a pipe.
    snap = f"/proc/{os.getpid()}/fd"
    found = set()
    for name in os.listdir(snap):
        num = int(name)
        try:
            target = os.readlink(f"{snap}/{name}")
        except FileNotFoundError:
            continue
        if target == snap:
            continue
        found.add(num)
    return found


def emit(line):
    data = (line + "\n").encode()
    try:
        os.write(LOGFD, data)
        return
    except OSError:
        pass
    try:
        os.write(1, data)
    except OSError:
        pass


def finish(ok, **fields):
    parts = " ".join(f"{key}={fields[key]}" for key in fields)
    emit(("RESULT PASS " if ok else "RESULT FAIL ") + parts)
    os._exit(0 if ok else 1)


def bind(so, names):
    lib = ctypes.CDLL(so, use_errno=True)
    have = {}
    for name in names:
        try:
            fn = getattr(lib, name)
        except AttributeError:
            have[name] = None
            continue
        fn.restype = I64
        if name in ("proc_pid",):
            fn.argtypes = []
        elif name in ("wait", "exit_code", "exit_signal", "exit_error", "exit_core",
                      "exit_stopped", "child_pid", "child_stdin", "child_stdout",
                      "child_stderr", "child_exec_errno", "child_spawn_errno"):
            fn.argtypes = [I64]
        elif name in ("proc_kill", "wait_flags"):
            fn.argtypes = [I64, I64]
        elif name in ("spawn", "spawn_capture"):
            fn.argtypes = [I64, I64, I64]
        elif name == "__mind_load_i64":
            fn.argtypes = [I64]
        have[name] = fn
    return lib, have


class Keep:
    def __init__(self):
        self.refs = []

    def cstr(self, text):
        buf = ctypes.create_string_buffer(text.encode())
        self.refs.append(buf)
        return ctypes.addressof(buf)

    def argv(self, items):
        if not items:
            return 0, 0
        bufs = [ctypes.create_string_buffer(item.encode()) for item in items]
        self.refs.append(bufs)
        arr = (I64 * len(bufs))(*(ctypes.addressof(buf) for buf in bufs))
        self.refs.append(arr)
        return ctypes.addressof(arr), len(bufs)


def load_i64(lib, addr, off):
    return lib.__mind_load_i64(addr + off)


def status_of(lib, have, st):
    out = {
        "code": have["exit_code"](st),
        "signal": have["exit_signal"](st),
    }
    if have.get("exit_error"):
        out["error"] = have["exit_error"](st)
    if have.get("exit_core"):
        out["core"] = have["exit_core"](st)
    if have.get("exit_stopped"):
        out["stopped"] = have["exit_stopped"](st)
    return out


def spawn_prog(lib, have, keep, fn, prog, args):
    prog_addr = keep.cstr(prog)
    argv_addr, argv_len = keep.argv(args)
    return fn(prog_addr, argv_addr, argv_len)


def child_pid(lib, child):
    return load_i64(lib, child, 0)


def alive(pid):
    if pid <= 0:
        return 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return 0
    except PermissionError:
        return 1
    return 1


def reap(pid):
    if pid <= 0:
        return
    if alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def arm_log(path):
    if not path:
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    if fd == LOGFD:
        return
    os.dup2(fd, LOGFD)
    os.close(fd)


def require(have, *names):
    missing = [name for name in names if not have.get(name)]
    if missing:
        finish(False, missing=",".join(missing))
