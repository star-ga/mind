# Author: STARGA Inc.
# Spawn cases: exec-failure reporting, descriptor balance and leaks.
# Split out of accept.py; the case bodies are unchanged.

import ctypes
import os
import resource

from _harness import (
    fdset,
    finish,
    Keep,
    status_of,
    spawn_prog,
    child_pid,
    reap,
    require,
    LOGFD,
)


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
