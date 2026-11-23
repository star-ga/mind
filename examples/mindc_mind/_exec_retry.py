"""Retry of a busy exec (ETXTBSY) for a smoke that writes a program file and runs
it from a thread pool. Imported by self_host_native_shadow_scope_smoke.py.

A fork in a sibling thread holds a copy of the write descriptor of a file that
was just written until its child execs. An exec of that file in this window
fails with ETXTBSY ("Text file busy"). The window closes within milliseconds, so
the exec is retried. Any other OSError, or a writer that never lets go, still
fails the caller. Same rule as exec_run of self_host_native_cf_smoke.py and
run_exe of self_host_native_misc_census_smoke.py.

Not a gate: running it directly is a no-op.
"""
from __future__ import annotations

import errno
import pathlib
import subprocess
import sys
import threading
import time

from _selfhost_sources import start_child

ETXTBSY_TRIES, ETXTBSY_WAIT = 250, 0.02  # up to 5 s for a writer to let go
_retries = [0]


def exec_run(argv, payload=None, **kw):
    """start_child (stdin = `payload`), retried while the target is busy (ETXTBSY) and only then."""
    for _ in range(ETXTBSY_TRIES - 1):
        try:
            return start_child(argv, payload, **kw)
        except OSError as e:
            if e.errno != errno.ETXTBSY:
                raise
            _retries[0] += 1
            time.sleep(ETXTBSY_WAIT)
    return start_child(argv, payload, **kw)


def _hold(path: pathlib.Path):
    """A child that keeps `path` open for writing until its stdin closes."""
    p = subprocess.Popen([sys.executable, "-c", "import sys; f = open(sys.argv[1], 'r+b'); "
                          "print('held', flush=True); sys.stdin.read()", str(path)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    return p, p.stdout.readline().strip() == b"held"


def _let_go(p) -> None:
    p.stdin.close()
    p.wait()
    p.stdout.close()


def _try(path: pathlib.Path):
    """Exit status of `path` through exec_run, or the errno of the OSError it raised."""
    try:
        return exec_run([str(path)], capture_output=True, timeout=60).returncode
    except OSError as e:
        return ("OSError", e.errno)


def check_exec_retry(wd: pathlib.Path) -> int:
    """Controls for exec_run. Prints one verdict line each; returns the number failed.
    They change ETXTBSY_TRIES for a moment, so call them before any thread pool starts.

    1. A file that a writer holds runs once the writer lets go, and at least one
       retry is counted. The writer lets go only after the first retry is counted
       (or after 3 s when none comes), so the busy exec is certain and no sleep
       length decides the verdict.
    2. A writer that never lets go fails: with 3 tries the exec raises ETXTBSY after
       exactly 2 retries. The writer is let go after 3 s at the latest, so a retry
       without a bound shows as a run, not as a hang.
    3. An error other than ETXTBSY is raised at once, with no retry."""
    global ETXTBSY_TRIES
    h = wd / "held.sh"
    h.write_text("#!/bin/sh\nexit 7\n")
    h.chmod(0o755)

    holder, held = _hold(h)
    before = _retries[0]

    def release():
        deadline = time.monotonic() + 3.0
        while _retries[0] == before and time.monotonic() < deadline:
            time.sleep(0.005)
        holder.stdin.close()

    rel = threading.Thread(target=release)
    rel.start()
    got = _try(h)
    rel.join()
    _let_go(holder)
    retried = _retries[0] - before
    ok_busy = held and got == 7 and retried >= 1
    print(f"  {'PASS' if ok_busy else 'FAIL'}  harness: exec while another process holds a write descriptor "
          f"(released at the first retry): got={got} want=7 retries={retried}")

    holder, held = _hold(h)
    before, keep = _retries[0], ETXTBSY_TRIES
    late = threading.Timer(3.0, holder.stdin.close)
    late.start()
    ETXTBSY_TRIES = 3
    try:
        stuck = _try(h)
    finally:
        ETXTBSY_TRIES = keep
        late.cancel()
        late.join()
        _let_go(holder)
    retried = _retries[0] - before
    ok_stuck = held and stuck == ("OSError", errno.ETXTBSY) and retried == 2
    print(f"  {'PASS' if ok_stuck else 'FAIL'}  harness: a writer that never lets go fails after the last try: "
          f"got={stuck} want=('OSError', {errno.ETXTBSY}) retries={retried} want=2")

    before = _retries[0]
    other = _try(wd / "absent.bin")
    ok_other = other == ("OSError", errno.ENOENT) and _retries[0] == before
    print(f"  {'PASS' if ok_other else 'FAIL'}  harness: an error other than ETXTBSY is raised at once: "
          f"got={other} want=('OSError', {errno.ENOENT}) retries={_retries[0] - before}")
    return (not ok_busy) + (not ok_stuck) + (not ok_other)
