#!/bin/bash
# Author: STARGA Inc.
# Run accept.py cases one at a time under a 2 GiB address cap.
# usage: run_cases.sh red|green /path/to/lib.so /path/to/outdir
set -u
MODE="${1:?mode}"
SO="${2:?so}"
OUT="${3:?outdir}"
PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/accept.py"
mkdir -p "$OUT"
chmod +x "$PY" || true

if [ "$MODE" = "red" ]; then
  CASES=(
    baseline_exits
    baseline_signals
    baseline_capture
    spawn_fd_balance
    echild
    eintr
    exec_distinguish
    exec_capture
    partial_kill
    core
    nocore
    stopped
    wnohang
    high_fd
    pipe_leak
    fork_leak
    echild_sentinel
  )
else
  CASES=(
    baseline_exits
    baseline_signals
    baseline_capture
    spawn_fd_balance
    echild
    echild_sentinel
    eintr
    exec_distinguish
    exec_capture
    partial_kill
    core
    nocore
    stopped
    wnohang
    high_fd
    pipe_leak
    fork_leak
  )
fi

fail=0
for case in "${CASES[@]}"; do
  timeout=20
  case "$case" in
    high_fd) timeout=60 ;;
    eintr) timeout=15 ;;
    echild_sentinel) timeout=8 ;;
  esac
  log="$OUT/${case}.log"
  timef="$OUT/${case}.time"
  # Process-group isolation: a timeout must not leave the sleep child behind.
  set +e
  /usr/bin/time -f "TIME case=${case} wall_sec=%e max_rss_kb=%M exit=%x" \
    -o "$timef" \
    nice -n 15 prlimit --as=2147483648 \
    python3 - "$MODE" "$SO" "$case" "$log" "$timeout" "$PY" << 'PY'
import os, signal, subprocess, sys
mode, so, case, log, timeout, accept = sys.argv[1:]
timeout = int(timeout)
out = log + ".result"
proc = subprocess.Popen(
    [sys.executable, accept,
     "--mode", mode, "--so", so, "--case", case, "--out", out],
    start_new_session=True,
)
try:
    rc = proc.wait(timeout=timeout)
    timed_out = 0
except subprocess.TimeoutExpired:
    timed_out = 1
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    rc = 124
text = ""
if os.path.exists(out):
    text = open(out).read()
line = text.strip().replace("\n", " | ")
print(f"CASE {case} rc={rc} timeout={timed_out} {line}")
sys.exit(0 if (rc == 0 or (mode == "red" and case == "echild_sentinel" and timed_out)) else 1)
PY
  rc=$?
  set -e
  cat "$log" 2>/dev/null || true
  echo "DRIVER_RC case=${case} rc=${rc}"
  if [ "$rc" -ne 0 ]; then
    fail=1
  fi
done
echo "SUITE mode=${MODE} fail=${fail}"
exit "$fail"
