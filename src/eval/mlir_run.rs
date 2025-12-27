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

#[cfg(feature = "mlir-exec")]
use std::path::PathBuf;

#[cfg(feature = "mlir-exec")]
use std::time::Duration;
use std::time::Instant;

#[cfg(feature = "mlir-exec")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlirExecConfig {
    pub mlir_opt: Option<PathBuf>,
    pub mlir_cpu_runner: Option<PathBuf>,
    pub opt_passes: Vec<String>,
    pub timeout: Duration,
}

#[cfg(feature = "mlir-exec")]
impl Default for MlirExecConfig {
    fn default() -> Self {
        Self {
            mlir_opt: None,
            mlir_cpu_runner: None,
            opt_passes: vec![String::from("--canonicalize"), String::from("--cse")],
            timeout: Duration::from_secs(15),
        }
    }
}

#[cfg(feature = "mlir-exec")]
pub fn exec_mlir_text(mlir: &str, cfg: &MlirExecConfig) -> Result<String, String> {
    use std::env;
    use std::io::Write;
    use std::process::Child;
    use std::process::Command;
    use std::process::Stdio;

    use tempfile::NamedTempFile;

    fn resolve(
        env_var: &str,
        configured: &Option<PathBuf>,
        fallback: &str,
    ) -> Result<PathBuf, String> {
        if let Some(path) = configured {
            return Ok(path.clone());
        }
        if let Ok(value) = env::var(env_var) {
            if !value.trim().is_empty() {
                return Ok(PathBuf::from(value));
            }
        }
        which::which(fallback)
            .map_err(|_| format!("Cannot find {fallback} (set {env_var} or adjust PATH)"))
    }

    fn wait_with_timeout(
        child: &mut Child,
        limit: Duration,
    ) -> Result<std::process::ExitStatus, String> {
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {
                    if start.elapsed() >= limit {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(format!(
                            "mlir-cpu-runner timed out after {}s",
                            limit.as_secs()
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(err) => return Err(err.to_string()),
            }
        }
    }

    let mlir_opt = resolve("MLIR_OPT", &cfg.mlir_opt, "mlir-opt")?;
    let mlir_cpu_runner = resolve("MLIR_CPU_RUNNER", &cfg.mlir_cpu_runner, "mlir-cpu-runner")?;

    let mut temp = NamedTempFile::new().map_err(|e| e.to_string())?;
    temp.write_all(mlir.as_bytes()).map_err(|e| e.to_string())?;
    let input_path = temp.path().to_path_buf();

    let mut opt_cmd = Command::new(&mlir_opt);
    opt_cmd.arg(input_path);
    for pass in &cfg.opt_passes {
        if !pass.trim().is_empty() {
            opt_cmd.arg(pass);
        }
    }
    opt_cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let opt_output = opt_cmd
        .output()
        .map_err(|e| format!("mlir-opt failed to spawn: {e}"))?;
    if !opt_output.status.success() {
        return Err(format!(
            "mlir-opt failed: {}\n{}",
            opt_output.status,
            String::from_utf8_lossy(&opt_output.stderr)
        ));
    }

    let optimized_mlir = String::from_utf8(opt_output.stdout.clone())
        .unwrap_or_else(|_| String::from_utf8_lossy(&opt_output.stdout).to_string());

    let mut runner_cmd = Command::new(&mlir_cpu_runner);
    runner_cmd
        .arg("-O0")
        .arg("--entry-point-result=void")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = runner_cmd
        .spawn()
        .map_err(|e| format!("mlir-cpu-runner failed to spawn: {e}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or("Failed to open stdin for mlir-cpu-runner")?;
        stdin
            .write_all(optimized_mlir.as_bytes())
            .map_err(|e| e.to_string())?;
    }

    let mut child_stdout = child
        .stdout
        .take()
        .ok_or("Failed to capture mlir-cpu-runner stdout")?;
    let mut child_stderr = child
        .stderr
        .take()
        .ok_or("Failed to capture mlir-cpu-runner stderr")?;

    use std::io::Read;

    // Spawn threads to read stdout/stderr to avoid blocking.
    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = child_stdout.read_to_end(&mut buf);
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = child_stderr.read_to_end(&mut buf);
        buf
    });

    let status = wait_with_timeout(&mut child, cfg.timeout)?;

    let stdout_data = stdout_handle.join().unwrap_or_default();
    let stderr_data = stderr_handle.join().unwrap_or_default();

    if !status.success() {
        let msg = if !stderr_data.is_empty() {
            String::from_utf8_lossy(&stderr_data).to_string()
        } else {
            String::new()
        };
        return Err(format!("mlir-cpu-runner failed: {}\n{}", status, msg));
    }

    let stdout_string = if !stdout_data.is_empty() {
        match String::from_utf8(stdout_data) {
            Ok(s) => s,
            Err(err) => {
                let bytes = err.into_bytes();
                String::from_utf8_lossy(&bytes).to_string()
            }
        }
    } else {
        String::new()
    };

    Ok(stdout_string.trim().to_string())
}
