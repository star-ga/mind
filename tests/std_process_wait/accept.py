#!/usr/bin/env python3
# Author: STARGA Inc.
# Native acceptance for std/process.mind wait and spawn_capture.
# --mode red records the defects on an unfixed library.
# --mode green checks the repaired contract.
# Destructive cases write a single result line to --out and _exit.
#
# The cases live in cases_status.py and cases_spawn.py, the shared helpers in
# _harness.py; this file is the driver (case table, symbol list, argv).

import sys

from _harness import arm_log, bind, finish
from cases_spawn import (
    case_exec_capture,
    case_exec_distinguish,
    case_fork_leak,
    case_high_fd,
    case_partial_kill,
    case_pipe_leak,
    case_spawn_fd_balance,
)
from cases_status import (
    case_baseline_capture,
    case_baseline_exits,
    case_baseline_signals,
    case_core,
    case_echild,
    case_echild_sentinel,
    case_eintr,
    case_nocore,
    case_stopped,
    case_wnohang,
)


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
