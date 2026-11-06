# `std.procgroup`: run many child processes and wait for all of them

`std.procgroup` starts a group of child processes, captures each child's
stdout, and waits for all of them with one call that has a deadline. It is
pure MIND in two modules: `std.procgroup` keeps the slots, states and output
buffers and makes no OS call of its own, and `std.procgroup_sys` holds every
OS call it needs (`pipe`, `fork`, `execvp`, `waitpid`, `kill`, `read`,
`poll`, the monotonic clock of the deadline, the environment), one small
primitive each. Importing `std.procgroup` links both into `mindc build`
executables (the MLIR path), like the other bundled `std` modules.

The common use is to run the same program many times at once, for example to
play 25 games in parallel: the program spawns itself 25 times, each child
plays one game and writes its result to stdout, and the parent reads all 25
results when the group is done.

## Example

```mind
import std.procgroup;

fn play(game: i64) -> i64 {
    // ... play one game, return its score ...
    return game * 10;
}

fn main() -> i64 {
    let me: i64 = procgroup_task_index();
    if me >= 0 {
        // A child: `me` is its slot index, 0..24.
        print(play(me));
        return 0;
    }
    let g: i64 = procgroup_new(25);
    let exe: i64 = procgroup_self_exe();
    let mut i: i64 = 0;
    while i < 25 {
        let _ = procgroup_spawn(g, exe, 0, 0);
        i = i + 1;
    }
    let failed: i64 = procgroup_wait_all(g, 600000);
    // procgroup_state / procgroup_exit_code / procgroup_stdout_addr and
    // procgroup_stdout_len give each child's result, by slot index.
    let _ = procgroup_free(g);
    return failed;
}
```

## How it behaves

- **Every child is accounted for.** `procgroup_spawn` returns the child's slot
  index. Each slot ends in one state, and `procgroup_wait_all` returns the
  number of slots that are not `procgroup_state_ok()`. A pipe or fork that
  fails still takes a slot, in `procgroup_state_spawn_failed()`, so the
  failure is counted. Only a full group (or a 0 handle, or a negative argument
  count) makes `procgroup_spawn` return -1; then nothing is started.
- **The deadline is bounded.** With `timeout_ms > 0`, every child still running
  that many milliseconds after the call is killed with `SIGKILL`, reaped, and
  reported as `procgroup_state_timed_out()`. The output it wrote before the
  deadline is kept. `timeout_ms <= 0` waits without a deadline.
- **Pipes never fill up.** The parent reads every child's stdout while the
  children run, so a child that writes more than a pipe holds does not block.
  Each child keeps at most `procgroup_max_output()` bytes (16 MiB); the rest is
  read and dropped, and `procgroup_truncated(g, i)` reports 1.
- **Results are in spawn order.** All results are read by slot index, never in
  the order the children finished, so a group of deterministic children reads
  back the same states and bytes on every run.
- **Children are isolated.** Each child's stdin is `/dev/null` and its stderr
  is the parent's stderr. A child does not keep the pipes of its siblings open.

## States

| Function | Value | Meaning |
|---|---|---|
| `procgroup_state_ok()` | 0 | exited with code 0 |
| `procgroup_state_exit_nonzero()` | 1 | exited with another code (127: the program could not be executed) |
| `procgroup_state_signaled()` | 2 | ended by a signal that the deadline did not send |
| `procgroup_state_timed_out()` | 3 | still running at the deadline; killed with `SIGKILL` |
| `procgroup_state_spawn_failed()` | 4 | the pipe or the fork failed; no child ran |
| `procgroup_state_running()` | 5 | started, not reaped yet |
| `procgroup_state_lost()` | 6 | `waitpid` refused the pid (the child was reaped somewhere else) |

## Functions

| Function | Returns |
|---|---|
| `procgroup_new(capacity)` | a handle for up to `capacity` children (1 to `procgroup_max_children()`, 4096), or 0 |
| `procgroup_spawn(g, prog, argv, argc)` | the slot index, or -1 when nothing was started |
| `procgroup_wait_all(g, timeout_ms)` | the number of slots not ok, or -1 for a 0 handle |
| `procgroup_free(g)` | 1, or 0 while a child is still running |
| `procgroup_len(g)`, `procgroup_failures(g)` | slots taken; slots not ok |
| `procgroup_state(g, i)`, `procgroup_exit_code(g, i)`, `procgroup_signal(g, i)`, `procgroup_pid(g, i)` | the slot's result (-1 for a slot that does not exist) |
| `procgroup_stdout_addr(g, i)`, `procgroup_stdout_len(g, i)`, `procgroup_truncated(g, i)` | the captured stdout bytes, owned by the group until `procgroup_free` |
| `procgroup_task_index()` | in a child, its slot index; in a process not started by a group, -1 |
| `procgroup_self_exe()` | `/proc/self/exe` as a C string: the path that runs the current executable again |
| `procgroup_cstr(addr, len)`, `procgroup_decimal(n)` | C strings for a program name or an argument |

`prog` and each argument are NUL-terminated C strings. `argv` is the address
of `argc` packed 8-byte string addresses; they become `argv[1..]` (`argv[0]`
is `prog`), and `prog` is found through `PATH` when it has no `/`. To pass a
`String`, use `procgroup_cstr(s.addr, s.len)`; to pass a number, use
`procgroup_decimal(n)`.

## The task index

A child learns its slot index from the `MIND_PROCGROUP_TASK` environment
variable, which `procgroup_spawn` sets for that child only (the parent's own
value is put back after each spawn). The environment is used because an
executable built through the MLIR path cannot read its `argv` yet. Arguments
are still passed, so a program that reads `argv` gets them too. A process that
a child starts some other way inherits the child's index.

## Limits

- POSIX only. `procgroup_self_exe()` names `/proc/self/exe`, so spawning the
  current executable works on Linux; on other systems that exec fails and the
  child is reported with exit code 127. Pass the program's own path instead.
- Do not set `SIGCHLD` to ignored: the kernel then reaps the children itself
  and they are reported as `procgroup_state_lost()`.
- Without a deadline, a grandchild that keeps a child's stdout open keeps
  `procgroup_wait_all` waiting. With a deadline, the pipe is closed at the
  deadline.
- After `SIGKILL`, the reap waits for the child to exit. A child stuck in
  uninterruptible sleep delays the return until the kernel lets it end.
- `std.procgroup` reads the world: `procgroup_spawn` forks and reads the
  environment, `procgroup_wait_all` reads the clock for its deadline, and
  `procgroup_task_index` reads the environment. The flat `mindc <file>` path
  (`--emit-shared`, `--emit-obj`, `--emit-evidence`) therefore refuses a
  program that calls one of them unless `--allow-nondeterministic` is given,
  and evidence for such a program is attested `nondeterministic`. `mindc
  build`, the path the example above uses, does not apply this refusal. A
  program that imports `std.procgroup_sys` directly is held to the same rule
  for each primitive that forks, waits, reads a pipe, reads the clock or reads
  the environment.
