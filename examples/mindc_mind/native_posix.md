# Linux x86-64 POSIX substrate

`std.posix` is import-scoped. A file's explicit C extern declarations take
precedence over imported wrappers. MLIR objects use `mind_posix_*` symbols,
so linking the substrate cannot interpose a program's libc calls.

The native intrinsic emitter and the C twins use the same Linux x86-64 ABI:
`rax` holds the syscall number; `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9` hold
arguments; `rcx` and `r11` are clobbered. Results preserve negative errno.
The C twins execute inline syscall assembly and do not resolve libc symbols.
Other platforms return `-ENOSYS` from these Linux-specific twins.

## Export census and admission boundary

The 35 C spellings provided as pure-MIND **word-ABI** wrappers are:

```
open openat close fstat fstatat fsync flock fcntl lseek unlinkat renameat
umask getuid fork pipe2 dup2 close_range fexecve waitpid waitid kill setpgid
getpgid prctl getrlimit setrlimit signal nanosleep clock_gettime memfd_create
fchdir fdopendir readdir closedir _exit
```

The remaining census function is `__errno_location`. It is **not implemented**.
The wrappers above are not automatic aliases for user extern declarations.
They take/return i64 words, including addresses. The consumers instead declare
C i32/u32/pointer signatures, `_exit` with no return value, and sometimes
inspect libc errno. Native admission must explicitly adapt those declarations
and their error convention; importing std.posix must never silently override
an extern. Raw negative errno is not interchangeable with libc's `-1` and TLS
errno, or with NULL on pointer-returning errors.

`__mind_read`/`__mind_write` retain their existing contract: stream I/O for
negative offsets, positioned I/O otherwise, `-1` on failure, and zero for a
null buffer or nonpositive count. They do not supply an errno location.

## Tier 2 kernel adaptations

IDs 32–51 append to the existing native IDs without renumbering:

| ID | Intrinsic suffix | Syscall | Arguments |
|---:|---|---:|---:|
| 32 | fork | 57 | 0 |
| 33 | pipe2 | 293 | 2 |
| 34 | dup2 | 33 | 2 |
| 35 | close_range | 436 | 3 |
| 36 | execveat | 322 | 5 |
| 37 | wait4 | 61 | 4 |
| 38 | waitid | 247 | 5 |
| 39 | kill | 62 | 2 |
| 40 | setpgid | 109 | 2 |
| 41 | getpgid | 121 | 1 |
| 42 | prctl | 157 | 5 |
| 43 | getrlimit | 97 | 2 |
| 44 | setrlimit | 160 | 2 |
| 45 | rt_sigaction | 13 | 4 |
| 46 | nanosleep | 35 | 2 |
| 47 | clock_gettime | 228 | 2 |
| 48 | memfd_create | 319 | 2 |
| 49 | fchdir | 81 | 1 |
| 50 | getdents64 | 217 | 3 |
| 51 | exit_group | 231 | 1 |

`fexecve` uses `execveat(fd, empty, argv, envp, AT_EMPTY_PATH)`.
`waitpid` uses `wait4` without rusage; `waitid` likewise passes NULL rusage.
`_exit` uses `exit_group`. `signal` supports SIG_DFL and SIG_IGN, including
returning the previous disposition; function handlers return `-EINVAL` because
no signal-restorer trampoline is supplied. Campaign only requests SIG_DFL.

`fdopendir` validates a directory fd and transfers ownership on success.
`readdir` refills a 4096-byte getdents64 buffer, returns a borrowed Linux
dirent64 address, zero at EOF, and negative errno on error. The borrowed address
is valid until the next refill or `closedir`. `closedir` closes the fd and frees
the handle. A handle is invalid after close; callers must not reuse it.

## Tests

`self_host_native_sys_smoke.py` covers filesystem and positioned-I/O parity.
`self_host_native_process_smoke.py` covers process, wait, descriptor, directory,
clock, resource-limit, exec and signal-disposition behavior with independent
expected results. Both use temporary directories and require static native
ELFs. Pids and clock values are checked behaviorally rather than compared byte
for byte. Both smokes are wired in `SMOKE_WIRING.tsv` and CI.

Bootstrap fixture publication belongs to integration. A local temporary seed
can be derived through two pure-MIND hops before running the existing loop
gate against it. The repository's frozen seed and manifest must remain intact.

## Consumer declarations

The materialized ten-program corpus declares the following ABIs. These are
recorded verbatim; they are not the word-ABI wrapper signatures above.

```mind
safe fn _exit(code: i32);
unsafe fn __errno_location() -> *mut i32;
unsafe fn clock_gettime(clock: i32, value: *mut u8) -> i32;
unsafe fn close(fd: i32) -> i32;
unsafe fn close_range(first: u32, last: u32, flags: i32) -> i32;
unsafe fn closedir(directory: *mut u8) -> i32;
unsafe fn dup2(oldfd: i32, newfd: i32) -> i32;
unsafe fn fchdir(fd: i32) -> i32;
unsafe fn fcntl(fd: i32, command: i32, value: i64) -> i32;
unsafe fn fdopendir(fd: i32) -> *mut u8;
unsafe fn fexecve(fd: i32, argv: *const *const u8, envp: *const *const u8) -> i32;
unsafe fn flock(fd: i32, operation: i32) -> i32;
unsafe fn fork() -> i32;
unsafe fn fstat(fd: i32, buffer: *mut u8) -> i32;
unsafe fn fstatat(dir: i32, path: *const u8, buffer: *mut u8, flags: i32) -> i32;
unsafe fn fsync(fd: i32) -> i32;
unsafe fn getpgid(pid: i32) -> i32;
unsafe fn getrlimit(resource: i32, limits: *mut u8) -> i32;
unsafe fn getuid() -> i32;
unsafe fn kill(pid: i32, signal: i32) -> i32;
unsafe fn lseek(fd: i32, offset: i64, whence: i32) -> i64;
unsafe fn memfd_create(name: *const u8, flags: i32) -> i32;
unsafe fn nanosleep(value: *const u8, remainder: *mut u8) -> i32;
unsafe fn open(path: *const u8, flags: i32, mode: i32) -> i32;
unsafe fn openat(dir: i32, path: *const u8, flags: i32, mode: i32) -> i32;
unsafe fn pipe2(pipefd: *mut i32, flags: i32) -> i32;
unsafe fn prctl(option: i32, value: i64, a: i64, b: i64, c: i64) -> i32;
unsafe fn readdir(directory: *mut u8) -> *mut u8;
unsafe fn renameat(fromdir: i32, from: *const u8, todir: i32, to: *const u8) -> i32;
unsafe fn setpgid(pid: i32, group: i32) -> i32;
unsafe fn setrlimit(resource: i32, limits: *const u8) -> i32;
unsafe fn signal(number: i32, handler: *const u8) -> *mut u8;
unsafe fn umask(mask: i32) -> i32;
unsafe fn unlinkat(dir: i32, path: *const u8, flags: i32) -> i32;
unsafe fn waitid(kind: i32, pid: i32, info: *mut u8, options: i32) -> i32;
unsafe fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
```
