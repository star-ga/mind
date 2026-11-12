# Author: STARGA Inc.
# Wait-status cases: exit, signal, ECHILD, EINTR, core, stopped, WNOHANG.
# Split out of accept.py; the case bodies are unchanged.

import ctypes
import os
import signal
import time

from _harness import (
    fdset,
    finish,
    Keep,
    status_of,
    spawn_prog,
    child_pid,
    alive,
    reap,
    require,
    I64,
)


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
