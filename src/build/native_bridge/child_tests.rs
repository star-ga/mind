// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Unit controls for the compiler-child supervisor. Each fixture is a
//! `/bin/sh -c` program, so no executable file is written (no ETXTBSY race).

use super::*;
use std::os::unix::process::ExitStatusExt as _;
use std::process::{Command, Stdio};

/// Larger than any pipe buffer, so a child that never reads it guarantees
/// the write fails rather than landing in the kernel buffer.
const BIG: usize = 1 << 20;

fn spawn_sh(script: &str) -> Child {
    Command::new("/bin/sh")
        .args(["-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn /bin/sh")
}

fn stderr_text(run: &ChildRun) -> String {
    String::from_utf8_lossy(&run.stderr.as_ref().expect("stderr captured").bytes).into_owned()
}

#[test]
fn early_exit_is_reaped_with_its_status_and_stderr() {
    let child = spawn_sh("echo early >&2; exit 3");
    let run = supervise(child, &vec![b'a'; BIG], Duration::from_secs(5));
    assert!(
        matches!(run.failure, Some(StreamFailure::Write(ref e)) if e.kind() == io::ErrorKind::BrokenPipe),
        "expected a broken-pipe write failure, got {:?}",
        run.failure
    );
    let Reaped::Exited(status) = &run.reaped else {
        panic!("expected a reaped exit, got {:?}", run.reaped);
    };
    assert_eq!(status.code(), Some(3));
    assert_eq!(exit_code_for(status, 2), 3);
    assert_eq!(describe_status(status), "exited with status 3");
    assert_eq!(stderr_text(&run), "early\n");
}

#[test]
fn signal_death_is_named_and_maps_to_128_plus_signo() {
    let child = spawn_sh("exec 0<&-; kill -SEGV $$");
    let run = supervise(child, &vec![b'a'; BIG], Duration::from_secs(5));
    assert!(matches!(run.failure, Some(StreamFailure::Write(_))));
    let Reaped::Exited(status) = &run.reaped else {
        panic!("expected a reaped exit, got {:?}", run.reaped);
    };
    assert_eq!(status.signal(), Some(11));
    assert_eq!(exit_code_for(status, 2), 139);
    assert!(
        describe_status(status).starts_with("terminated by signal 11 (SIGSEGV)"),
        "{}",
        describe_status(status)
    );
}

#[test]
fn hung_child_is_killed_after_the_grace_and_reaped() {
    let child = spawn_sh("exec 0<&-; exec sleep 30");
    let start = Instant::now();
    let run = supervise(child, &vec![b'a'; BIG], Duration::from_millis(200));
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "bounded reap took {:?}",
        start.elapsed()
    );
    let Reaped::KilledAfterGrace { grace, status } = &run.reaped else {
        panic!("expected kill-after-grace, got {:?}", run.reaped);
    };
    assert_eq!(*grace, Duration::from_millis(200));
    assert_eq!(status.signal(), Some(9));
    assert!(describe_reaped(&run.reaped).contains("killed with SIGKILL and reaped"));
}

#[test]
fn descendant_holding_the_pipes_cannot_stall_the_report() {
    // The direct child exits at once; a background descendant keeps stdout and
    // stderr open for 30 s. The outcome must still be reported promptly.
    let child = spawn_sh("exec 0<&-; sleep 30 & exit 4");
    let start = Instant::now();
    let run = supervise(child, &vec![b'a'; BIG], Duration::from_millis(500));
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "report stalled on a descendant's pipes: {:?}",
        start.elapsed()
    );
    let Reaped::Exited(status) = &run.reaped else {
        panic!("expected a reaped exit, got {:?}", run.reaped);
    };
    assert_eq!(status.code(), Some(4));
}

#[test]
fn stderr_is_bounded_and_the_excess_counted() {
    let child = spawn_sh("head -c 200000 /dev/zero >&2; cat >/dev/null; exit 1");
    let run = supervise(child, &vec![b'a'; BIG], Duration::from_secs(5));
    assert!(run.failure.is_none(), "{:?}", run.failure);
    let cap = run.stderr.expect("stderr captured");
    assert_eq!(cap.bytes.len(), STDERR_CAPTURE_LIMIT);
    assert_eq!(cap.dropped, (200_000 - STDERR_CAPTURE_LIMIT) as u64);
}

#[test]
fn success_path_returns_the_complete_stdout() {
    let image = vec![b'z'; BIG];
    let child = spawn_sh("cat");
    let run = supervise(child, &image, Duration::from_secs(5));
    assert!(run.failure.is_none(), "{:?}", run.failure);
    let Reaped::Exited(status) = &run.reaped else {
        panic!("expected a reaped exit, got {:?}", run.reaped);
    };
    assert!(status.success());
    assert_eq!(run.stdout.as_deref(), Some(&image[..]));
    assert_eq!(run.stderr.expect("stderr").bytes.len(), 0);
}

#[test]
fn signal_names_follow_the_target_numbering() {
    assert_eq!(signal_name(11), Some("SIGSEGV"));
    assert_eq!(signal_name(9), Some("SIGKILL"));
    #[cfg(target_os = "linux")]
    assert_eq!(signal_name(7), Some("SIGBUS"));
    assert_eq!(signal_name(200), None);
}
