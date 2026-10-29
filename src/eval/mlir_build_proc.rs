// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Subprocess plumbing shared by `run_command` and the concurrent
//! mlir-opt -> mlir-translate stage (`mlir_build_pipe.rs`): spawn a build tool,
//! drain its pipes on background threads, wait with a timeout, kill and reap.

use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::BuildError;

/// Spawn one build tool with piped stdout/stderr and the given stdin.
pub(super) fn spawn_tool(
    program: &str,
    args: &[String],
    stdin: Stdio,
    label: &'static str,
) -> Result<Child, BuildError> {
    let mut cmd = Command::new(program);
    cmd.args(args.iter().map(|s| s.as_str()));
    cmd.stdin(stdin);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.spawn().map_err(|err| match err.kind() {
        std::io::ErrorKind::NotFound => BuildError::ToolMissing(label),
        _ => BuildError::Io(err),
    })
}

/// Read `pipe` to EOF on a background thread. A read error keeps the bytes
/// received before it, as `read_to_end` does.
pub(super) fn drain_pipe<R: std::io::Read + Send + 'static>(
    pipe: Option<R>,
) -> Option<JoinHandle<Vec<u8>>> {
    pipe.map(|mut pipe| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            buf
        })
    })
}

pub(super) fn join_drain(handle: Option<JoinHandle<Vec<u8>>>) -> Vec<u8> {
    handle.and_then(|t| t.join().ok()).unwrap_or_default()
}

/// Poll `child` until it exits, killing it once `timeout` has elapsed since
/// `start`.
pub(super) fn wait_with_timeout(
    child: &mut Child,
    start: Instant,
    timeout: Duration,
    label: &'static str,
) -> Result<ExitStatus, BuildError> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                if start.elapsed() > timeout {
                    kill_and_reap(child);
                    return Err(BuildError::Timeout(label));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(err) => return Err(BuildError::Io(err)),
        }
    }
}

pub(super) fn kill_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
