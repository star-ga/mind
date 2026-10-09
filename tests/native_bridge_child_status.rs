// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Native bridge controls: the compiler CHILD's real outcome is reported.
//!
//! The bridge streams a ~380 KB source image into the frozen compiler's stdin.
//! When that child dies before reading it (a loader failure under an address
//! space cap, a signal, an early exit), the write fails with a broken pipe.
//! A broken pipe is a symptom, not a cause: these controls require the bridge to
//! close the pipe, reap the child, and report what the process actually did --
//! exit status or terminating signal, plus its bounded stderr -- while still
//! exiting nonzero and writing no artifact.
//!
//! Every fixture compiler here is a `/bin/sh` script, so the file is Unix-only.
//! The address-space case executes the committed Linux x86-64 seed itself.
#![cfg(all(unix, feature = "std-surface"))]

mod native_bridge_support;
use native_bridge_support::{MINDC, Project, repo_path};

use fs2::FileExt;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Generous ceiling on one mindc invocation. A bridge that deadlocks or never
/// reaps its child must turn this test RED, not hang the suite.
const MINDC_DEADLINE: Duration = Duration::from_secs(120);

/// One finished mindc run: exit code (or `128 + signal` if mindc itself died),
/// stderr, the artifact if one was written, and wall time.
struct Run {
    code: i32,
    stderr: String,
    artifact: Option<Vec<u8>>,
    elapsed: Duration,
}

/// Write an executable fixture without leaking a writable descriptor into a
/// concurrent fork (which makes the later execve fail with ETXTBSY). Same
/// handoff as `native_bridge_support::run`.
fn write_executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("create fixture");
    writer.write_all(body.as_bytes()).expect("write fixture");
    let mut perm = writer.metadata().expect("meta").permissions();
    perm.set_mode(0o755);
    writer.set_permissions(perm).expect("chmod fixture");
    writer.lock_exclusive().expect("lock writable fixture");
    drop(writer);
    let reader = OpenOptions::new()
        .read(true)
        .open(path)
        .expect("reopen fixture");
    reader.lock_exclusive().expect("lock readable fixture");
}

/// A project whose entry is the minimal scalar program the consumer used.
fn return42_project(name: &str) -> Project {
    let p = Project::new(name);
    p.write("src/main.mind", "fn main() -> i64 {\n    return 42;\n}\n");
    p
}

/// A fixture compiler script inside the project directory.
fn fixture_compiler(p: &Project, name: &str, body: &str) -> PathBuf {
    let path = p.root().join(name);
    write_executable(&path, body);
    path
}

/// Run `mindc build --backend native` with `compiler` as the child. `prefix`
/// wraps mindc itself (e.g. `prlimit --as=...`), exactly as a resource-capped
/// caller would.
fn build_with(p: &Project, compiler: &Path, prefix: &[&str]) -> Run {
    let out = p.root().join("out.elf");
    let _ = std::fs::remove_file(&out);
    let mut cmd = if let Some((head, rest)) = prefix.split_first() {
        let mut c = Command::new(head);
        c.args(rest).arg(MINDC);
        c
    } else {
        Command::new(MINDC)
    };
    cmd.args(["build", "src/main.mind", "--release", "--backend", "native"])
        .arg("--out")
        .arg(&out)
        .current_dir(p.root())
        .env("MINDC_STD_DIR", repo_path("std"))
        .env("MINDC_NATIVE_ELF", compiler)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let start = Instant::now();
    let mut child = cmd.spawn().expect("spawn mindc");
    let mut stderr_pipe = child.stderr.take().expect("stderr");
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });
    let status = loop {
        if let Some(status) = child.try_wait().expect("try_wait mindc") {
            break status;
        }
        if start.elapsed() > MINDC_DEADLINE {
            let _ = child.kill();
            let _ = child.wait();
            panic!("mindc did not finish within {MINDC_DEADLINE:?}: the bridge hung");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let elapsed = start.elapsed();
    let stderr = String::from_utf8_lossy(&reader.join().expect("stderr reader")).into_owned();
    let code = status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0));
    Run {
        code,
        stderr,
        artifact: std::fs::read(&out).ok(),
        elapsed,
    }
}

fn assert_refused_without_artifact(run: &Run, context: &str) {
    assert_ne!(run.code, 0, "{context}: must exit nonzero: {}", run.stderr);
    assert!(
        run.artifact.is_none(),
        "{context}: no artifact may be written: {}",
        run.stderr
    );
}

/// Checked LAST in each case, so a RED run on an unfixed bridge fails on the
/// missing child outcome rather than on this wording.
fn assert_says_no_artifact(run: &Run, context: &str) {
    assert!(
        run.stderr.contains("no artifact written"),
        "{context}: the refusal must say no artifact was written: {}",
        run.stderr
    );
}

/// A child that exits 3 before reading stdin. The pipe breaks; the report must
/// carry the child's own status and stderr, and mindc must propagate the code.
#[test]
fn early_exit_before_reading_reports_exit_status_and_stderr() {
    let p = return42_project("early_exit");
    let cc = fixture_compiler(
        &p,
        "early_exit.sh",
        "#!/bin/sh\necho 'fixture: refusing before reading stdin' >&2\nexit 3\n",
    );
    let run = build_with(&p, &cc, &[]);
    assert_refused_without_artifact(&run, "early exit");
    assert!(
        run.stderr
            .contains("failed to stream source image to compiler"),
        "the stream failure itself must still be reported: {}",
        run.stderr
    );
    assert!(
        run.stderr
            .contains("compiler process: exited with status 3"),
        "the child's real exit status must be reported: {}",
        run.stderr
    );
    assert!(
        run.stderr
            .contains("fixture: refusing before reading stdin"),
        "the child's stderr must be surfaced: {}",
        run.stderr
    );
    assert_eq!(run.code, 3, "mindc must propagate the child's exit status");
    assert_says_no_artifact(&run, "early exit");
}

/// A child killed by SIGSEGV after closing stdin: the signal is the outcome.
#[test]
fn signal_death_reports_signal_name_and_number() {
    let p = return42_project("segv");
    let cc = fixture_compiler(
        &p,
        "segv.sh",
        "#!/bin/sh\nexec 0<&-\necho 'fixture: about to fault' >&2\nkill -SEGV $$\n",
    );
    let run = build_with(&p, &cc, &[]);
    assert_refused_without_artifact(&run, "SIGSEGV child");
    assert!(
        run.stderr
            .contains("compiler process: terminated by signal 11 (SIGSEGV)"),
        "the terminating signal must be named and numbered: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("fixture: about to fault"),
        "stderr written before the fault must be surfaced: {}",
        run.stderr
    );
    assert_eq!(run.code, 128 + 11, "a signal death exits 128 + signo");
    assert_says_no_artifact(&run, "SIGSEGV child");
}

/// THE CONSUMER CASE: mindc under RLIMIT_AS = 2 GiB with the real committed
/// seed. The seed's load segments exceed the cap, the kernel fails execve with
/// ENOMEM past the point of no return and delivers SIGSEGV. Before this fix the
/// only report was "Broken pipe (os error 32)".
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn address_space_cap_reports_loader_signal_not_broken_pipe() {
    assert!(
        Command::new("prlimit")
            .arg("--version")
            .stdout(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false),
        "prlimit (util-linux) is required for the address-space control; \
         refusing to skip it silently"
    );
    let p = return42_project("as_cap");
    let seed = repo_path("examples/mindc_mind/testdata/selfhost_loop/stage1.elf");
    let run = build_with(&p, &seed, &["prlimit", "--as=2147483648:2147483648", "--"]);
    assert_refused_without_artifact(&run, "seed under a 2 GiB address-space cap");
    assert!(
        run.stderr
            .contains("compiler process: terminated by signal 11 (SIGSEGV)"),
        "the loader's real outcome must be reported, not only the broken pipe: {}",
        run.stderr
    );
    assert_eq!(run.code, 128 + 11);
    assert_says_no_artifact(&run, "seed under a 2 GiB address-space cap");
}

/// A child that closes stdin and then never exits. The bridge must not wait
/// forever: bounded grace, then SIGKILL, then reap, and say so.
#[test]
fn hung_child_is_killed_and_reaped_after_bounded_grace() {
    let p = return42_project("hang");
    // `exec sleep` keeps the fixture the bridge's direct child, so the kill
    // reaches the process that holds the pipes. 30 s bounds the orphan an
    // unfixed bridge would leave behind.
    let cc = fixture_compiler(
        &p,
        "hang.sh",
        "#!/bin/sh\nexec 0<&-\necho 'fixture: hanging' >&2\nexec sleep 30\n",
    );
    let run = build_with(&p, &cc, &[]);
    assert_refused_without_artifact(&run, "hung child");
    assert!(
        run.stderr.contains("did not exit within"),
        "the bounded wait must be reported: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("killed with SIGKILL and reaped"),
        "the bridge must kill and reap the hung child: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("terminated by signal 9 (SIGKILL)"),
        "the reaped status must be the kill: {}",
        run.stderr
    );
    assert!(
        run.stderr.contains("fixture: hanging"),
        "stderr written before the hang must be surfaced: {}",
        run.stderr
    );
    assert!(
        run.elapsed < Duration::from_secs(25),
        "the grace period must be bounded well below the child's own lifetime, \
         took {:?}",
        run.elapsed
    );
    assert_eq!(run.code, 128 + 9);
    assert_says_no_artifact(&run, "hung child");
}

/// A child that floods stderr BEFORE reading stdin. Both pipes must be drained
/// while the image is streamed, otherwise the child blocks on a full stderr
/// pipe and the bridge blocks on a full stdin pipe: a deadlock. The surfaced
/// stderr is bounded and the truncation is stated.
#[test]
fn stderr_flood_before_reading_does_not_deadlock_and_is_bounded() {
    let p = return42_project("flood");
    let cc = fixture_compiler(
        &p,
        "flood.sh",
        "#!/bin/sh\nhead -c 1048576 /dev/zero | tr '\\0' 'x' >&2\ncat >/dev/null\nexit 5\n",
    );
    let run = build_with(&p, &cc, &[]);
    assert_refused_without_artifact(&run, "stderr flood");
    assert_eq!(run.code, 5, "the child's rejection status must propagate");
    assert_says_no_artifact(&run, "stderr flood");
    assert!(
        run.stderr.contains("bytes of compiler stderr truncated"),
        "a truncated capture must say so: {}",
        &run.stderr[run.stderr.len().saturating_sub(600)..]
    );
    assert!(
        run.stderr.len() < 256 * 1024,
        "surfaced stderr must be bounded, got {} bytes",
        run.stderr.len()
    );
}

/// The success path is unchanged: a child that drains the full image and
/// emits an ELF-framed artifact produces exactly those bytes, exit 0.
#[test]
fn draining_child_success_path_is_unchanged() {
    let p = return42_project("drain_ok");
    let cc = fixture_compiler(
        &p,
        "drain.sh",
        "#!/bin/sh\nwc -c >\"$0.len\"\nprintf '\\177ELF'\nhead -c 393 /dev/zero\n",
    );
    let run = build_with(&p, &cc, &[]);
    assert_eq!(run.code, 0, "success path must stay green: {}", run.stderr);
    let bytes = run.artifact.expect("artifact written");
    assert_eq!(bytes.len(), 397);
    assert_eq!(&bytes[..4], b"\x7fELF");
    assert!(bytes[4..].iter().all(|&b| b == 0));
    let consumed: usize = std::fs::read_to_string(cc.with_extension("sh.len"))
        .expect("fixture recorded the image length")
        .trim()
        .parse()
        .expect("byte count");
    assert!(
        consumed > 64 * 1024,
        "the child must have received the whole composed image (> one pipe \
         buffer), got {consumed} bytes"
    );
    assert!(
        !run.stderr.contains("compiler process:"),
        "a clean build must not print child-outcome diagnostics: {}",
        run.stderr
    );
}

/// The real seed still builds and the artifact still runs (Linux x86-64).
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn real_seed_success_path_is_unchanged() {
    let p = return42_project("seed_ok");
    let seed = repo_path("examples/mindc_mind/testdata/selfhost_loop/stage1.elf");
    let run = build_with(&p, &seed, &[]);
    assert_eq!(run.code, 0, "seed build must succeed: {}", run.stderr);
    let bytes = run.artifact.expect("artifact written");
    assert_eq!(native_bridge_support::run(&bytes), 42);
}
