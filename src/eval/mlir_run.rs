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
use std::path::Path;
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
            // The runner JITs only the `llvm` dialect: lower the scalar program to it.
            opt_passes: [
                "--canonicalize",
                "--cse",
                "--convert-scf-to-cf",
                "--convert-cf-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
            ]
            .map(String::from)
            .to_vec(),
            timeout: Duration::from_secs(15),
        }
    }
}

/// The JIT runner's tool name for the LLVM release of `mlir_opt`: LLVM 20
/// renamed `mlir-cpu-runner` to `mlir-runner`. An unreadable version takes the
/// current name, so a missing tool is reported under the name a user installs.
#[cfg(feature = "mlir-exec")]
pub fn runner_tool_name(mlir_opt: &Path) -> &'static str {
    let version = std::process::Command::new(mlir_opt)
        .arg("--version")
        .output();
    let text = version.map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    match text.ok().as_deref().and_then(llvm_major) {
        Some(major) if major < 20 => "mlir-cpu-runner",
        _ => "mlir-runner",
    }
}

/// The major release in `--version` text (`Ubuntu LLVM version 20.1.8` -> 20).
#[cfg(feature = "mlir-exec")]
fn llvm_major(text: &str) -> Option<u32> {
    let rest = text.split("LLVM version ").nth(1)?;
    rest.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// The C runner utilities (`printI64`, `printNewline`) of the runner's own LLVM
/// install, `<prefix>/lib/libmlir_c_runner_utils.so*`; the unversioned name
/// sorts first. `None` leaves the runner to report the unresolved symbol.
#[cfg(feature = "mlir-exec")]
fn c_runner_utils(runner: &Path) -> Option<PathBuf> {
    let bin = std::fs::canonicalize(runner).ok()?;
    let lib = bin.parent()?.parent()?.join("lib");
    let is_utils = |p: &PathBuf| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("libmlir_c_runner_utils.so"))
    };
    let mut found: Vec<PathBuf> = std::fs::read_dir(lib)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(is_utils)
        .collect();
    found.sort();
    found.into_iter().next()
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
                        return Err(format!("MLIR runner timed out after {}s", limit.as_secs()));
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(err) => return Err(err.to_string()),
            }
        }
    }

    let mlir_opt = resolve("MLIR_OPT", &cfg.mlir_opt, "mlir-opt")?;
    let runner_name = runner_tool_name(&mlir_opt);
    let mlir_cpu_runner = resolve("MLIR_CPU_RUNNER", &cfg.mlir_cpu_runner, runner_name)?;

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
        .args(
            c_runner_utils(&mlir_cpu_runner).map(|lib| format!("--shared-libs={}", lib.display())),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = runner_cmd
        .spawn()
        .map_err(|e| format!("{runner_name} failed to spawn: {e}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or("Failed to open stdin for the MLIR runner")?;
        stdin
            .write_all(optimized_mlir.as_bytes())
            .map_err(|e| e.to_string())?;
    }

    let mut child_stdout = child
        .stdout
        .take()
        .ok_or("Failed to capture the MLIR runner stdout")?;
    let mut child_stderr = child
        .stderr
        .take()
        .ok_or("Failed to capture the MLIR runner stderr")?;

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
        return Err(format!("{runner_name} failed: {}\n{}", status, msg));
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

#[cfg(all(test, feature = "mlir-exec"))]
mod tests {
    #[test]
    fn llvm_major_reads_the_release() {
        assert_eq!(super::llvm_major("Ubuntu LLVM version 20.1.8\n"), Some(20));
        assert_eq!(super::llvm_major("  LLVM version 18.1.8"), Some(18));
        assert_eq!(super::llvm_major("LLVM version 23.0.0git"), Some(23));
        assert_eq!(super::llvm_major("mlir-opt: no version"), None);
    }
}
