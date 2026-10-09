// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Supervision of the frozen native compiler child.
//!
//! The bridge streams the composed source image into the child's stdin and
//! reads the artifact from its stdout. A failure on any of those pipes is a
//! SYMPTOM: a child that died in its loader (execve `ENOMEM` under an
//! address-space cap ends in SIGSEGV), faulted, or exited early shows up on
//! this side only as a broken pipe. So on a stream failure the supervisor
//! closes the pipe, reaps the child -- bounded wait, then SIGKILL and reap if
//! it is still running -- and returns the process's actual outcome together
//! with a bounded capture of its stderr.
//!
//! Both output pipes are drained on their own threads WHILE the image is
//! written. Writing first and reading afterwards deadlocks as soon as the child
//! fills a pipe buffer before it has consumed its input.
//!
//! The successful path is unbounded exactly as before: a long compile is not a
//! hang. Only the post-failure wait has a deadline.

use std::fmt;
use std::io::{self, Read, Write};
use std::process::{Child, ExitStatus};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Most stderr bytes kept from the child. The child keeps being drained past
/// this point (so it never blocks); the excess is counted and reported.
pub(super) const STDERR_CAPTURE_LIMIT: usize = 64 * 1024;

/// How long a child may keep running after a stream failure before it is
/// killed. A child that stopped consuming its input is not compiling it.
pub(super) const REAP_GRACE: Duration = Duration::from_secs(5);

/// After the child is reaped, how long to wait for its pipes to reach EOF. A
/// descendant that inherited them can hold them open indefinitely; the
/// outcome is reported without the capture rather than waiting on it.
const DRAIN_AFTER_REAP: Duration = Duration::from_secs(2);

const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Stack for a drain thread. Each holds at most one 8 KiB buffer, so the
/// 2 MiB default would only spend address space -- the resource that is
/// capped when this path matters most.
const DRAIN_STACK: usize = 64 * 1024;

/// The head of a stream plus how many bytes past the limit were discarded.
#[derive(Debug, Default)]
pub(super) struct BoundedCapture {
    pub(super) bytes: Vec<u8>,
    pub(super) dropped: u64,
}

/// Which pipe failed, and how. The first failure observed wins.
#[derive(Debug)]
pub(super) enum StreamFailure {
    /// Writing the source image to the child's stdin failed.
    Write(io::Error),
    /// Reading the child's stdout (the artifact) failed.
    ReadStdout(io::Error),
    /// Reading the child's stderr failed.
    ReadStderr(io::Error),
    /// A pipe the bridge requested was not provided by the spawn.
    MissingPipe(&'static str),
    /// The thread that drains this output pipe could not be started; the
    /// pipe was closed instead.
    DrainThread(&'static str, io::Error),
}

impl fmt::Display for StreamFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamFailure::Write(e) => {
                write!(f, "failed to stream source image to compiler: {e}")
            }
            StreamFailure::ReadStdout(e) => write!(f, "failed to read compiler output: {e}"),
            StreamFailure::ReadStderr(e) => write!(f, "failed to read compiler stderr: {e}"),
            StreamFailure::MissingPipe(which) => {
                write!(f, "compiler {which} pipe was not captured at spawn")
            }
            StreamFailure::DrainThread(which, e) => {
                write!(f, "could not start the compiler {which} reader: {e}")
            }
        }
    }
}

/// What became of the child process.
#[derive(Debug)]
pub(super) enum Reaped {
    /// It terminated on its own and was reaped.
    Exited(ExitStatus),
    /// It was still running when the post-failure grace expired; it was sent
    /// SIGKILL and reaped with this status.
    KilledAfterGrace { grace: Duration, status: ExitStatus },
    /// The OS refused to report or deliver the outcome.
    WaitFailed(io::Error),
}

/// Everything known about one compiler run.
#[derive(Debug)]
pub(super) struct ChildRun {
    /// `None` iff the image was fully written and both outputs reached EOF.
    pub(super) failure: Option<StreamFailure>,
    pub(super) reaped: Reaped,
    /// The complete stdout, or `None` if it could not be collected.
    pub(super) stdout: Option<Vec<u8>>,
    /// The bounded stderr, or `None` if it could not be collected.
    pub(super) stderr: Option<BoundedCapture>,
}

enum Drained {
    Stdout(io::Result<Vec<u8>>),
    Stderr(io::Result<BoundedCapture>),
}

/// Stream `image` into `child`, collect its output, and reap it.
///
/// Never returns with the child unreaped: every path ends in `wait`, and a
/// post-failure wait is bounded by `grace` before a SIGKILL.
pub(super) fn supervise(mut child: Child, image: &[u8], grace: Duration) -> ChildRun {
    let (tx, rx) = mpsc::channel::<Drained>();
    let mut pending = 0usize;
    let mut failure: Option<StreamFailure> = None;

    match child.stdout.take() {
        Some(mut out) => {
            let tx = tx.clone();
            match spawn_drain("stdout", move || {
                let mut buf = Vec::new();
                let res = out.read_to_end(&mut buf).map(|_| buf);
                let _ = tx.send(Drained::Stdout(res));
            }) {
                Ok(()) => pending += 1,
                Err(e) => failure = Some(StreamFailure::DrainThread("stdout", e)),
            }
        }
        None => failure = Some(StreamFailure::MissingPipe("stdout")),
    }
    match child.stderr.take() {
        Some(err) => {
            let tx = tx.clone();
            match spawn_drain("stderr", move || {
                let _ = tx.send(Drained::Stderr(drain_bounded(err, STDERR_CAPTURE_LIMIT)));
            }) {
                Ok(()) => pending += 1,
                Err(e) => {
                    failure.get_or_insert(StreamFailure::DrainThread("stderr", e));
                }
            }
        }
        None => {
            failure.get_or_insert(StreamFailure::MissingPipe("stderr"));
        }
    }
    drop(tx);

    // Dropping the handle closes the write end: EOF on success, and on failure
    // the child sees a closed input instead of a pipe the bridge still holds.
    // If an output cannot be drained, nothing is written: the child could
    // block on that output while the bridge blocks on its input.
    match child.stdin.take() {
        Some(mut stdin) => {
            if failure.is_none() {
                if let Err(e) = stdin.write_all(image) {
                    failure = Some(StreamFailure::Write(e));
                }
            }
        }
        None => {
            failure.get_or_insert(StreamFailure::MissingPipe("stdin"));
        }
    }

    let mut stdout = None;
    let mut stderr = None;
    let mut absorb = |msg: Drained, failure: &mut Option<StreamFailure>| match msg {
        Drained::Stdout(Ok(b)) => stdout = Some(b),
        Drained::Stdout(Err(e)) => {
            failure.get_or_insert(StreamFailure::ReadStdout(e));
        }
        Drained::Stderr(Ok(c)) => stderr = Some(c),
        Drained::Stderr(Err(e)) => {
            failure.get_or_insert(StreamFailure::ReadStderr(e));
        }
    };

    // Healthy so far: wait for both outputs to close, unbounded, as before.
    // Any read failure switches to the bounded reap below.
    while failure.is_none() && pending > 0 {
        match rx.recv() {
            Ok(msg) => {
                pending -= 1;
                absorb(msg, &mut failure);
            }
            Err(_) => break,
        }
    }

    let reaped = if failure.is_none() {
        match child.wait() {
            Ok(status) => Reaped::Exited(status),
            Err(e) => Reaped::WaitFailed(e),
        }
    } else {
        reap_bounded(&mut child, grace)
    };

    // Collect whatever the drain threads still owe, bounded: the child is
    // gone, so only a descendant holding the pipes can keep them open.
    let deadline = Instant::now() + DRAIN_AFTER_REAP;
    while pending > 0 {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok(msg) => {
                pending -= 1;
                absorb(msg, &mut failure);
            }
            Err(_) => break,
        }
    }

    ChildRun {
        failure,
        reaped,
        stdout,
        stderr,
    }
}

/// Start a small-stack thread that drains one output pipe. On failure the
/// closure -- and with it the pipe's read end -- is dropped, closing the pipe.
fn spawn_drain(which: &'static str, f: impl FnOnce() + Send + 'static) -> io::Result<()> {
    thread::Builder::new()
        .name(format!("native-compiler-{which}"))
        .stack_size(DRAIN_STACK)
        .spawn(f)
        .map(|_| ())
}

/// Wait at most `grace` for the child to exit, then SIGKILL and reap it.
fn reap_bounded(child: &mut Child, grace: Duration) -> Reaped {
    let deadline = Instant::now() + grace;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Reaped::Exited(status),
            Ok(None) if Instant::now() >= deadline => break,
            Ok(None) => thread::sleep(POLL_INTERVAL),
            Err(e) => return Reaped::WaitFailed(e),
        }
    }
    // `kill` on a child that exited in the meantime is harmless: it is still
    // unreaped, so its pid cannot have been reused.
    if let Err(e) = child.kill() {
        return Reaped::WaitFailed(e);
    }
    match child.wait() {
        Ok(status) => Reaped::KilledAfterGrace { grace, status },
        Err(e) => Reaped::WaitFailed(e),
    }
}

/// Read `r` to EOF, keeping the first `limit` bytes and counting the rest.
fn drain_bounded(mut r: impl Read, limit: usize) -> io::Result<BoundedCapture> {
    let mut cap = BoundedCapture::default();
    let mut buf = [0u8; 8192];
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) => return Ok(cap),
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        let keep = n.min(limit - cap.bytes.len());
        cap.bytes.extend_from_slice(&buf[..keep]);
        cap.dropped += (n - keep) as u64;
    }
}

/// Exit code the bridge reports for this outcome: the child's own nonzero
/// status, `128 + signo` for a signal death, and `fallback` when the child
/// "succeeded" or its status is unknown (the run still failed).
pub(super) fn exit_code_for(status: &ExitStatus, fallback: i32) -> i32 {
    if let Some(code) = status.code() {
        return if code == 0 { fallback } else { code };
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if let Some(sig) = status.signal() {
            return 128 + sig;
        }
    }
    fallback
}

/// Human description of how the process ended, e.g. `exited with status 3`
/// or `terminated by signal 11 (SIGSEGV), core dumped`.
pub(super) fn describe_status(status: &ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exited with status {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if let Some(sig) = status.signal() {
            let name = signal_name(sig)
                .map(|n| format!(" ({n})"))
                .unwrap_or_default();
            let core = if status.core_dumped() {
                ", core dumped"
            } else {
                ""
            };
            return format!("terminated by signal {sig}{name}{core}");
        }
    }
    format!("ended with an unrecognized status ({status})")
}

/// Describe the reaped outcome as one line.
pub(super) fn describe_reaped(reaped: &Reaped) -> String {
    match reaped {
        Reaped::Exited(status) => describe_status(status),
        Reaped::KilledAfterGrace { grace, status } => format!(
            "did not exit within {} ms after the stream failure; killed with SIGKILL and \
             reaped: {}",
            grace.as_millis(),
            describe_status(status)
        ),
        Reaped::WaitFailed(e) => format!("could not be reaped: {e}"),
    }
}

/// Signal names for the numbering this target actually uses. Only numbers
/// whose meaning is certain are named; anything else is reported by number.
#[cfg(unix)]
fn signal_name(sig: i32) -> Option<&'static str> {
    // Fixed by POSIX/XSI or identical on every Unix this crate targets.
    let common = match sig {
        1 => Some("SIGHUP"),
        2 => Some("SIGINT"),
        3 => Some("SIGQUIT"),
        4 => Some("SIGILL"),
        5 => Some("SIGTRAP"),
        6 => Some("SIGABRT"),
        8 => Some("SIGFPE"),
        9 => Some("SIGKILL"),
        11 => Some("SIGSEGV"),
        13 => Some("SIGPIPE"),
        14 => Some("SIGALRM"),
        15 => Some("SIGTERM"),
        _ => None,
    };
    common.or_else(|| platform_signal_name(sig))
}

/// The generic Linux numbering (x86, Arm, RISC-V; not MIPS/SPARC).
#[cfg(all(
    target_os = "linux",
    not(any(
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "sparc",
        target_arch = "sparc64"
    ))
))]
fn platform_signal_name(sig: i32) -> Option<&'static str> {
    match sig {
        7 => Some("SIGBUS"),
        10 => Some("SIGUSR1"),
        12 => Some("SIGUSR2"),
        16 => Some("SIGSTKFLT"),
        17 => Some("SIGCHLD"),
        18 => Some("SIGCONT"),
        19 => Some("SIGSTOP"),
        20 => Some("SIGTSTP"),
        21 => Some("SIGTTIN"),
        22 => Some("SIGTTOU"),
        23 => Some("SIGURG"),
        24 => Some("SIGXCPU"),
        25 => Some("SIGXFSZ"),
        26 => Some("SIGVTALRM"),
        27 => Some("SIGPROF"),
        28 => Some("SIGWINCH"),
        29 => Some("SIGIO"),
        30 => Some("SIGPWR"),
        31 => Some("SIGSYS"),
        _ => None,
    }
}

/// Other Unix numberings differ above the common set; report those by number.
#[cfg(all(
    unix,
    not(all(
        target_os = "linux",
        not(any(
            target_arch = "mips",
            target_arch = "mips64",
            target_arch = "sparc",
            target_arch = "sparc64"
        ))
    ))
))]
fn platform_signal_name(_sig: i32) -> Option<&'static str> {
    None
}

#[cfg(all(test, unix))]
#[path = "child_tests.rs"]
mod tests;
