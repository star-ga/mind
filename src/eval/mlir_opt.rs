// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

use std::fmt;

/// Result of running mlir-opt.
#[derive(Debug)]
pub struct MlirOptOutput {
    pub stdout: String,
    pub stderr: String,
    pub status_ok: bool,
}

impl MlirOptOutput {
    pub fn from_error<E: fmt::Display>(err: E) -> Self {
        Self {
            stdout: String::new(),
            stderr: err.to_string(),
            status_ok: false,
        }
    }
}

#[cfg(feature = "mlir-subprocess")]
use std::time::Duration;

#[cfg(feature = "mlir-subprocess")]
use std::io::Write;
#[cfg(feature = "mlir-subprocess")]
use std::process::Child;
#[cfg(feature = "mlir-subprocess")]
use std::process::Command;
#[cfg(feature = "mlir-subprocess")]
use std::process::Stdio;
#[cfg(feature = "mlir-subprocess")]
use std::thread;

#[cfg(feature = "mlir-subprocess")]
pub fn run_mlir_opt(
    mlir_input: &str,
    bin: &str,
    passes: &[String],
    timeout_ms: u64,
) -> std::io::Result<MlirOptOutput> {
    let pipeline = passes.join(",");

    let mut cmd = Command::new(bin);
    if !pipeline.is_empty() {
        cmd.arg(format!("--pass-pipeline={pipeline}"));
    }

    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(mlir_input.as_bytes())?;
    }

    let start = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let (stdout, stderr) = collect_child_output(&mut child);
            return Ok(MlirOptOutput {
                stdout,
                stderr,
                status_ok: status.success(),
            });
        }

        if start.elapsed() > Duration::from_millis(timeout_ms) {
            let _ = child.kill();
            let _ = child.wait();
            let (stdout, stderr) = collect_child_output(&mut child);
            return Ok(MlirOptOutput {
                stdout,
                stderr: if stderr.is_empty() {
                    "mlir-opt timed out".into()
                } else {
                    format!("mlir-opt timed out: {stderr}")
                },
                status_ok: false,
            });
        }

        thread::sleep(Duration::from_millis(15));
    }
}

#[cfg(feature = "mlir-subprocess")]
fn collect_child_output(child: &mut Child) -> (String, String) {
    use std::io::Read;

    let mut stdout = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }

    let mut stderr = String::new();
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }

    (stdout, stderr)
}

#[cfg(not(feature = "mlir-subprocess"))]
pub fn run_mlir_opt(
    _mlir_input: &str,
    _bin: &str,
    _passes: &[String],
    _timeout_ms: u64,
) -> std::io::Result<MlirOptOutput> {
    Ok(MlirOptOutput::from_error(
        "mlir-subprocess feature is disabled",
    ))
}
