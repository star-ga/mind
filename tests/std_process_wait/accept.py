#!/usr/bin/env python3
# Author: STARGA Inc.
# Native acceptance for std/process.mind wait and spawn_capture.
# --mode red records the defects on an unfixed library.
# --mode green checks the repaired contract.
# Destructive cases write a single result line to --out and _exit.

import ctypes
import os
import resource
import signal
import sys
import time

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


def case_baseline_exits(lib, have, _mode):
    require(have, "spawn", "wait", "exit_code", "exit_signal")
    keep = Keep()
    specs = [
        ("/bin/true", [], 0),
        ("/bin/false", [], 1),
        ("/bin/sh", ["-c", "exit 255"], 255),
        ("/bin/sh", ["-c", "exit 0"], 0),
    ]
    got = []
    ok = True
    for prog, args, expect in specs:
        child = spawn_prog(lib, have, keep, have["spawn"], prog, args)
        pid = child_pid(lib, child)
        st = have["wait"](child)
        code = have["exit_code"](st)
        sig = have["exit_signal"](st)
        got.append(f"{expect}:{code}/{sig}/pid{pid}")
        if pid <= 0 or code != expect or sig != 0:
            ok = False
        reap(pid)
    finish(ok, observed=",".join(got))


def case_baseline_signals(lib, have, _mode):
    require(have, "spawn", "wait", "proc_kill", "exit_code", "exit_signal")
    keep = Keep()
    observed = []
    ok = True
    for sig in (15, 9):
        child = spawn_prog(lib, have, keep, have["spawn"], "/bin/sleep", ["30"])
        pid = child_pid(lib, child)
        killed = have["proc_kill"](child, sig)
        st = have["wait"](child)
        code = have["exit_code"](st)
        got = have["exit_signal"](st)
        observed.append(f"sig{sig}:code{code}:signal{got}:kill{killed}:pid{pid}")
        if pid <= 0 or killed != 1 or code == 0 or got != sig:
            ok = False
        reap(pid)
    finish(ok, observed=",".join(observed))


def case_baseline_capture(lib, have, _mode):
    require(have, "spawn_capture", "wait", "exit_code", "child_stdout")
    keep = Keep()
    before = fdset()
    child = spawn_prog(lib, have, keep, have["spawn_capture"], "/bin/echo", ["HI"])
    pid = child_pid(lib, child)
    out_fd = have["child_stdout"](child)
    data = b""
    read_err = 0
    if out_fd >= 0:
        try:
            data = os.read(out_fd, 64)
        except OSError as exc:
            read_err = exc.errno
    st = have["wait"](child)
    code = have["exit_code"](st)
    after = fdset()
    extra = len(after - before)
    ok = pid > 0 and out_fd >= 3 and data == b"HI\n" and code == 0 and read_err == 0 and extra == 3
    finish(ok, pid=pid, stdout_fd=out_fd, data=data, code=code, read_err=read_err, fd_extra=extra)
    reap(pid)


def case_echild(lib, have, mode):
    require(have, "wait", "exit_code", "exit_signal")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.waitpid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    libc.waitpid.restype = ctypes.c_int
    status = ctypes.c_int(0)
    ctypes.set_errno(0)
    raw = libc.waitpid(os.getpid(), ctypes.byref(status), 0)
    err = ctypes.get_errno()
    child = (I64 * 7)(os.getpid(), -1, -1, -1, 0, 0, 0)
    st = have["wait"](ctypes.addressof(child))
    info = status_of(lib, have, st)
    if mode == "red":
        ok = (raw, err) == (-1, 10) and info["code"] == 0 and info["signal"] == 0
    else:
        require(have, "exit_error")
        ok = (raw, err) == (-1, 10) and info["code"] == -1 and info["signal"] == 0 and info["error"] == 10
    finish(ok, libc_rc=raw, libc_errno=err, **info)


def case_echild_sentinel(lib, have, mode):
    # pid <= 0 must not become waitpid(-1) and reap some other child.
    require(have, "spawn", "wait", "exit_code", "exit_signal", "proc_kill")
    keep = Keep()
    sleeper = spawn_prog(lib, have, keep, have["spawn"], "/bin/sleep", ["30"])
    spid = child_pid(lib, sleeper)
    fake = (I64 * 7)(-1, -1, -1, -1, 0, 0, 0)
    t0 = time.monotonic()
    st = have["wait"](ctypes.addressof(fake))
    dt = time.monotonic() - t0
    info = status_of(lib, have, st)
    still = alive(spid)
    if mode == "red":
        # Unfixed wait calls waitpid(-1) and blocks. The driver timeout is the
        # observation; if it returns, record what it did.
        ok = dt >= 5.0 or still == 1
    else:
        require(have, "exit_error")
        ok = dt < 1.0 and still == 1 and info["code"] == -1 and info["error"] == 10 and info["signal"] == 0
    reap(spid)
    finish(ok, dt=f"{dt:.3f}", sleeper_alive=still, sleeper=spid, **info)


def case_eintr(lib, have, mode):
    require(have, "spawn", "wait", "exit_code", "exit_signal")
    keep = Keep()
    child = spawn_prog(lib, have, keep, have["spawn"], "/bin/sleep", ["2"])
    pid = child_pid(lib, child)
    fired = {"n": 0}

    def handler(signum, _frame):
        fired["n"] += 1

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    t0 = time.monotonic()
    st = have["wait"](child)
    dt = time.monotonic() - t0
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    info = status_of(lib, have, st)
    still = alive(pid)
    if mode == "red":
        ok = fired["n"] >= 1 and dt < 1.0 and info["code"] == 0 and info["signal"] == 0 and still == 1
    else:
        require(have, "exit_error")
        ok = (
            fired["n"] >= 1
            and dt >= 1.2
            and info["code"] == 0
            and info["signal"] == 0
            and info.get("error") == 0
            and still == 0
        )
    reap(pid)
    finish(ok, dt=f"{dt:.3f}", fired=fired["n"], pid=pid, alive=still, **info)


def case_exec_distinguish(lib, have, mode):
    require(have, "spawn", "wait", "exit_code", "exit_signal")
    keep = Keep()
    missing = "/no/such/mind-process-wait-bin"
    bad = spawn_prog(lib, have, keep, have["spawn"], missing, [])
    bad_pid = child_pid(lib, bad)
    bad_st = have["wait"](bad)
    bad_code = have["exit_code"](bad_st)
    good = spawn_prog(lib, have, keep, have["spawn"], "/bin/sh", ["-c", "exit 127"])
    good_pid = child_pid(lib, good)
    good_st = have["wait"](good)
    good_code = have["exit_code"](good_st)
    if mode == "red":
        ok = bad_pid > 0 and good_pid > 0 and bad_code == 127 and good_code == 127
        finish(
            ok,
            bad_pid=bad_pid,
            bad_code=bad_code,
            good_pid=good_pid,
            good_code=good_code,
            distinguishable=0,
        )
    require(have, "child_exec_errno", "exit_error")
    bad_exec = have["child_exec_errno"](bad)
    good_exec = have["child_exec_errno"](good)
    bad_info = status_of(lib, have, bad_st)
    good_info = status_of(lib, have, good_st)
    ok = (
        bad_pid > 0
        and good_pid > 0
        and bad_code == 127
        and good_code == 127
        and bad_exec == 2
        and good_exec == 0
        and bad_info["error"] == 0
        and good_info["error"] == 0
        and bad_info["signal"] == 0
        and good_info["signal"] == 0
    )
    reap(bad_pid)
    reap(good_pid)
    finish(
        ok,
        bad_pid=bad_pid,
        bad_code=bad_code,
        bad_exec=bad_exec,
        good_pid=good_pid,
        good_code=good_code,
        good_exec=good_exec,
    )


def case_exec_capture(lib, have, mode):
    require(have, "spawn_capture", "wait", "exit_code", "child_stdout", "child_pid")
    keep = Keep()
    bad = spawn_prog(
        lib, have, keep, have["spawn_capture"], "/no/such/mind-process-wait-bin", []
    )
    bad_pid = child_pid(lib, bad)
    bad_out = have["child_stdout"](bad)
    good = spawn_prog(
        lib,
        have,
        keep,
        have["spawn_capture"],
        "/bin/sh",
        ["-c", "/bin/echo PURPOSE; exit 127"],
    )
    good_pid = child_pid(lib, good)
    good_out = have["child_stdout"](good)
    data = b""
    if good_out >= 0:
        try:
            data = os.read(good_out, 64)
        except OSError as exc:
            data = f"ERR{exc.errno}".encode()
    bad_code = have["exit_code"](have["wait"](bad))
    good_code = have["exit_code"](have["wait"](good))
    if mode == "red":
        ok = bad_code == 127 and good_code == 127 and bad_out >= 0 and data == b"PURPOSE\n"
        finish(ok, bad_out=bad_out, good_out=good_out, data=data, bad_code=bad_code, good_code=good_code)
    require(have, "child_exec_errno")
    bad_exec = have["child_exec_errno"](bad)
    good_exec = have["child_exec_errno"](good)
    ok = (
        bad_pid > 0
        and good_pid > 0
        and bad_code == 127
        and good_code == 127
        and bad_exec == 2
        and good_exec == 0
        and bad_out == -1
        and good_out >= 0
        and data == b"PURPOSE\n"
    )
    reap(bad_pid)
    reap(good_pid)
    finish(
        ok,
        bad_pid=bad_pid,
        bad_out=bad_out,
        bad_exec=bad_exec,
        bad_code=bad_code,
        good_pid=good_pid,
        good_out=good_out,
        good_exec=good_exec,
        good_code=good_code,
        data=data,
    )


def case_partial_kill(lib, have, mode):
    require(have, "spawn_capture", "wait", "proc_kill", "exit_code", "exit_signal", "child_stdout")
    keep = Keep()
    child = spawn_prog(
        lib,
        have,
        keep,
        have["spawn_capture"],
        "/bin/sh",
        ["-c", "/bin/echo -n PART; /bin/sleep 30"],
    )
    pid = child_pid(lib, child)
    out_fd = have["child_stdout"](child)
    data = b""
    if out_fd >= 0:
        try:
            data = os.read(out_fd, 64)
        except OSError as exc:
            data = f"ERR{exc.errno}".encode()
    killed = have["proc_kill"](child, 9)
    st = have["wait"](child)
    info = status_of(lib, have, st)
    ok = pid > 0 and data == b"PART" and killed == 1 and info["code"] == -1 and info["signal"] == 9
    if mode == "green":
        ok = ok and info.get("error") == 0 and info.get("core") == 0 and info.get("stopped") == 0
    reap(pid)
    finish(ok, pid=pid, data=data, killed=killed, stdout_fd=out_fd, **info)


def case_high_fd(lib, have, mode):
    require(have, "spawn_capture", "wait", "exit_code", "child_stdout", "child_stdin", "child_stderr")
    devnull = os.open("/dev/null", os.O_RDONLY)
    for fd in range(3, 65536):
        if fd == LOGFD:
            continue
        os.dup2(devnull, fd)
    keep = Keep()
    before = fdset()
    child = spawn_prog(lib, have, keep, have["spawn_capture"], "/bin/echo", ["HI"])
    pid = child_pid(lib, child)
    out_fd = have["child_stdout"](child)
    err_fd = have["child_stderr"](child)
    in_fd = have["child_stdin"](child)
    after = fdset()
    new_fds = sorted(after - before)
    data = b""
    read_err = 0
    if mode == "green" and out_fd >= 3:
        try:
            data = os.read(out_fd, 64)
        except OSError as exc:
            read_err = exc.errno
    code = -999
    if pid > 0 and mode == "green":
        code = have["exit_code"](have["wait"](child))
    high_new = [fd for fd in new_fds if fd >= 65536]
    if mode == "red":
        ok = pid > 0 and out_fd < 65536 and len(high_new) > 0
    else:
        ok = (
            pid > 0
            and out_fd >= 65536
            and in_fd >= 65536
            and err_fd >= 65536
            and data == b"HI\n"
            and code == 0
            and read_err == 0
            and set(new_fds) == {in_fd, out_fd, err_fd}
        )
    reap(pid)
    finish(
        ok,
        pid=pid,
        stdout_fd=out_fd,
        stdin_fd=in_fd,
        stderr_fd=err_fd,
        new_high=len(high_new),
        new_sample=",".join(str(fd) for fd in new_fds[:8]),
        data=data,
        code=code,
        read_err=read_err,
    )


def case_pipe_leak(lib, have, mode):
    require(have, "spawn_capture", "child_pid")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    openfds = fdset()
    free = 0
    limit = None
    fd = 0
    while fd < min(hard, 100000):
        if fd not in openfds:
            free += 1
            if free == 2:
                limit = fd + 1
                break
        fd += 1
    if limit is None or limit > hard:
        finish(False, reason="no_limit", hard=hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    except (ValueError, OSError) as exc:
        finish(False, reason="setrlimit", err=getattr(exc, "errno", exc))
    before = fdset()
    keep = Keep()
    child = spawn_prog(
        lib, have, keep, have["spawn_capture"], "/bin/true", []
    )
    pid = child_pid(lib, child)
    try:
        after = fdset()
        delta = len(after - before)
    except OSError as exc:
        # The unrepaired path can consume the last descriptor slots, so the
        # snapshot itself fails. A negative delta is that errno.
        delta = 0 - exc.errno
    spawned = have["child_spawn_errno"](child) if have.get("child_spawn_errno") else -1
    if mode == "red":
        ok = pid < 0 and (delta >= 2 or delta < 0)
    else:
        ok = pid < 0 and delta == 0 and spawned == 24
    finish(ok, pid=pid, fd_delta=delta, limit=limit, spawn_errno=spawned, nofile_was=soft)


def case_fork_leak(lib, have, mode):
    require(have, "spawn_capture", "spawn", "child_pid")
    _soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (1, hard))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.fork.restype = ctypes.c_int
    ctypes.set_errno(0)
    probe = libc.fork()
    probe_err = ctypes.get_errno()
    if probe == 0:
        os._exit(0)
    if probe > 0:
        os.waitpid(probe, 0)
        finish(False, reason="fork_still_works", probe=probe)
    before = fdset()
    keep = Keep()
    cap = spawn_prog(lib, have, keep, have["spawn_capture"], "/bin/true", [])
    cap_pid = child_pid(lib, cap)
    after_cap = fdset()
    cap_delta = len(after_cap - before)
    plain = spawn_prog(lib, have, keep, have["spawn"], "/bin/true", [])
    plain_pid = child_pid(lib, plain)
    after_plain = fdset()
    plain_delta = len(after_plain - after_cap)
    cap_err = have["child_spawn_errno"](cap) if have.get("child_spawn_errno") else -1
    plain_err = have["child_spawn_errno"](plain) if have.get("child_spawn_errno") else -1
    if mode == "red":
        ok = probe_err == 11 and cap_pid < 0 and cap_delta >= 6
    else:
        ok = (
            probe_err == 11
            and cap_pid < 0
            and plain_pid < 0
            and cap_delta == 0
            and plain_delta == 0
            and cap_err == 11
            and plain_err == 11
        )
    finish(
        ok,
        probe_errno=probe_err,
        cap_pid=cap_pid,
        cap_delta=cap_delta,
        cap_errno=cap_err,
        plain_pid=plain_pid,
        plain_delta=plain_delta,
        plain_errno=plain_err,
    )


def case_core(lib, have, mode):
    require(have, "spawn", "wait", "exit_code", "exit_signal")
    if mode == "green":
        require(have, "exit_core", "exit_error", "exit_stopped")
    keep = Keep()
    tmp = "/tmp/mind-process-core-" + str(os.getpid())
    os.mkdir(tmp)
    child = spawn_prog(
        lib,
        have,
        keep,
        have["spawn"],
        "/bin/sh",
        ["-c", f"cd {tmp}; ulimit -c unlimited; kill -ABRT $$"],
    )
    pid = child_pid(lib, child)
    st = have["wait"](child)
    info = status_of(lib, have, st)
    names = os.listdir(tmp)
    nbytes = 0
    for name in names:
        path = os.path.join(tmp, name)
        nbytes += os.path.getsize(path)
        os.remove(path)
    os.rmdir(tmp)
    if mode == "red":
        ok = pid > 0 and info["signal"] == 6 and info["code"] == -1
    else:
        ok = (
            pid > 0
            and info["signal"] == 6
            and info["code"] == -1
            and info["core"] == 1
            and info["error"] == 0
            and info["stopped"] == 0
            and len(names) >= 1
        )
    finish(ok, pid=pid, cores=",".join(names), core_bytes=nbytes, **info)


def case_nocore(lib, have, mode):
    require(have, "spawn", "wait", "exit_code", "exit_signal")
    if mode == "green":
        require(have, "exit_core")
    keep = Keep()
    child = spawn_prog(
        lib,
        have,
        keep,
        have["spawn"],
        "/bin/sh",
        ["-c", "ulimit -c 0; kill -ABRT $$"],
    )
    pid = child_pid(lib, child)
    st = have["wait"](child)
    info = status_of(lib, have, st)
    if mode == "red":
        ok = pid > 0 and info["signal"] == 6
    else:
        ok = pid > 0 and info["signal"] == 6 and info["core"] == 0 and info["code"] == -1
    reap(pid)
    finish(ok, pid=pid, **info)


def case_stopped(lib, have, mode):
    if mode == "red":
        # options are fixed at 0, so a stopped child would block the waiter.
        ok = have.get("wait_flags") is None
        finish(ok, wait_flags="absent" if ok else "present")
    require(have, "spawn", "wait_flags", "wait", "proc_kill", "exit_code", "exit_signal", "exit_stopped", "exit_error")
    keep = Keep()
    child = spawn_prog(
        lib, have, keep, have["spawn"], "/bin/sh", ["-c", "kill -STOP $$"]
    )
    pid = child_pid(lib, child)
    st = have["wait_flags"](child, 2)
    info = status_of(lib, have, st)
    still = alive(pid)
    killed = have["proc_kill"](child, 9)
    st2 = have["wait"](child)
    info2 = status_of(lib, have, st2)
    ok = (
        pid > 0
        and still == 1
        and info["stopped"] == 1
        and info["signal"] == 19
        and info["code"] == -1
        and info["error"] == 0
        and info["core"] == 0
        and killed == 1
        and info2["signal"] == 9
        and info2["stopped"] == 0
        and info2["error"] == 0
    )
    reap(pid)
    finish(ok, pid=pid, still=still, killed=killed, stop_signal=info["signal"], stop_flag=info["stopped"], kill_signal=info2["signal"], kill_code=info2["code"])


def case_wnohang(lib, have, mode):
    if mode == "red":
        ok = have.get("wait_flags") is None
        finish(ok, wait_flags="absent" if ok else "present")
    require(have, "spawn", "wait_flags", "wait", "proc_kill", "exit_code", "exit_error", "exit_signal")
    keep = Keep()
    child = spawn_prog(lib, have, keep, have["spawn"], "/bin/sleep", ["30"])
    pid = child_pid(lib, child)
    t0 = time.monotonic()
    st = have["wait_flags"](child, 1)
    dt = time.monotonic() - t0
    info = status_of(lib, have, st)
    still = alive(pid)
    have["proc_kill"](child, 9)
    st2 = have["wait"](child)
    info2 = status_of(lib, have, st2)
    ok = (
        dt < 1.0
        and still == 1
        and info["code"] == -1
        and info["error"] == -1
        and info["signal"] == 0
        and info2["signal"] == 9
        and info2["error"] == 0
    )
    reap(pid)
    finish(ok, dt=f"{dt:.3f}", alive=still, pid=pid, hang_code=info["code"], hang_error=info["error"], kill_signal=info2["signal"])


def case_spawn_fd_balance(lib, have, mode):
    require(have, "spawn", "wait", "exit_code", "child_pid")
    keep = Keep()
    before = fdset()
    child = spawn_prog(lib, have, keep, have["spawn"], "/bin/true", [])
    mid = fdset()
    pid = child_pid(lib, child)
    code = have["exit_code"](have["wait"](child))
    after = fdset()
    if mode == "red":
        ok = pid > 0 and code == 0 and len(mid - before) == 0
    else:
        require(have, "child_exec_errno", "child_spawn_errno")
        ok = (
            pid > 0
            and code == 0
            and len(mid - before) == 0
            and len(after - before) == 0
            and have["child_exec_errno"](child) == 0
            and have["child_spawn_errno"](child) == 0
        )
    reap(pid)
    finish(ok, pid=pid, code=code, mid_delta=len(mid - before), end_delta=len(after - before))


CASES = {
    "baseline_exits": case_baseline_exits,
    "baseline_signals": case_baseline_signals,
    "baseline_capture": case_baseline_capture,
    "spawn_fd_balance": case_spawn_fd_balance,
    "echild": case_echild,
    "echild_sentinel": case_echild_sentinel,
    "eintr": case_eintr,
    "exec_distinguish": case_exec_distinguish,
    "exec_capture": case_exec_capture,
    "partial_kill": case_partial_kill,
    "core": case_core,
    "nocore": case_nocore,
    "stopped": case_stopped,
    "wnohang": case_wnohang,
    "high_fd": case_high_fd,
    "pipe_leak": case_pipe_leak,
    "fork_leak": case_fork_leak,
}

SYMBOLS = [
    "__mind_load_i64",
    "spawn",
    "spawn_capture",
    "wait",
    "wait_flags",
    "proc_kill",
    "proc_pid",
    "exit_code",
    "exit_signal",
    "exit_error",
    "exit_core",
    "exit_stopped",
    "child_pid",
    "child_stdin",
    "child_stdout",
    "child_stderr",
    "child_exec_errno",
    "child_spawn_errno",
]


def main():
    args = sys.argv[1:]
    mode = "green"
    so = ""
    case = ""
    out = ""
    i = 0
    while i < len(args):
        if args[i] == "--mode":
            mode = args[i + 1]
            i += 2
        elif args[i] == "--so":
            so = args[i + 1]
            i += 2
        elif args[i] == "--case":
            case = args[i + 1]
            i += 2
        elif args[i] == "--out":
            out = args[i + 1]
            i += 2
        else:
            sys.stderr.write(f"unknown arg {args[i]}\n")
            sys.exit(2)
    if mode not in ("red", "green") or not so or not case:
        sys.stderr.write("usage: accept.py --mode red|green --so PATH --case NAME [--out PATH]\n")
        sys.exit(2)
    arm_log(out)
    lib, have = bind(so, SYMBOLS)
    fn = CASES.get(case)
    if fn is None:
        finish(False, reason="unknown_case", case=case)
    fn(lib, have, mode)


if __name__ == "__main__":
    main()
