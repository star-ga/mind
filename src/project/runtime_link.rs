// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Selects the optional installed runtime used by the executable linker.
//!
//! Native CPU objects are self-contained with the bundled public runtime-support
//! shim. Every other backend retains the installed-runtime requirement.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use crate::diagnostics::capability::FallbackReason;
use crate::diagnostics::refusal::CodedRefusal;

pub(crate) struct RuntimeLink {
    pub dir: PathBuf,
    pub file_name: &'static str,
    pub link_name: &'static str,
}

pub(super) fn resolve(backend: &str) -> Result<Option<RuntimeLink>> {
    if !installed_runtime_required(backend) {
        return Ok(None);
    }

    let discovery_name = runtime_file_name(backend);
    let (file_name, link_name) = runtime_names(backend);
    let dir = find_installed_runtime(backend, discovery_name)?;
    Ok(Some(RuntimeLink {
        dir,
        file_name,
        link_name,
    }))
}

fn installed_runtime_required(backend: &str) -> bool {
    backend != "cpu"
}

fn runtime_file_name(backend: &str) -> &'static str {
    match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            "libmind_cuda_linux-x64.so"
        }
        "rocm" | "rocm-mi300" => "libmind_rocm_linux-x64.so",
        "metal" | "metal-m4" => "libmind_metal_macos-arm64.dylib",
        "webgpu" => "libmind_webgpu_linux-x64.so",
        "cerebras" | "wse3" | "cs3" => "libmind_cerebras_wse3_linux-x64.so",
        "wse2" | "cs2" => "libmind_cerebras_wse2_linux-x64.so",
        _ => "libmind_cpu_linux-x64.so",
    }
}

fn find_installed_runtime(backend: &str, file_name: &str) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot determine home directory"))?;
    let canonical = home.join(".mind").join("lib");

    if let Ok(env_dir) = std::env::var("MIND_LIB_DIR") {
        let env_path = PathBuf::from(env_dir);
        if env_path.join(file_name).exists() {
            return Ok(env_path);
        }
    }

    if canonical.join(file_name).exists() || canonical.exists() {
        return Ok(canonical);
    }

    Err(anyhow::Error::new(CodedRefusal::new(
        FallbackReason::RuntimeLibraryAbsent,
        format!(
            "MIND runtime not found for backend '{backend}'. \
             See https://mindlang.dev/enterprise for licensing."
        ),
    )))
}

fn runtime_names(backend: &str) -> (&'static str, &'static str) {
    match backend {
        "cuda" | "cuda-ampere" | "cuda-hopper" | "cuda-blackwell" | "cuda-rubin" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cuda_windows-x64.dll", "mind_cuda_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_cuda_linux-x64.so", "mind_cuda_linux-x64")
            }
        }
        "rocm" | "rocm-mi300" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_rocm_windows-x64.dll", "mind_rocm_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_rocm_linux-x64.so", "mind_rocm_linux-x64")
            }
        }
        "metal" | "metal-m4" => ("libmind_metal_macos-arm64.dylib", "mind_metal_macos-arm64"),
        "webgpu" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                (
                    "libmind_webgpu_macos-arm64.dylib",
                    "mind_webgpu_macos-arm64",
                )
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_webgpu_windows-x64.dll", "mind_webgpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_webgpu_linux-x64.so", "mind_webgpu_linux-x64")
            }
        }
        "directx" => ("mind_directx_windows-x64.dll", "mind_directx_windows-x64"),
        "oneapi" => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_oneapi_windows-x64.dll", "mind_oneapi_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                ("libmind_oneapi_linux-x64.so", "mind_oneapi_linux-x64")
            }
        }
        _ => {
            #[cfg(target_os = "linux")]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
            #[cfg(target_os = "macos")]
            {
                ("libmind_cpu_macos-arm64.dylib", "mind_cpu_macos-arm64")
            }
            #[cfg(target_os = "windows")]
            {
                ("mind_cpu_windows-x64.dll", "mind_cpu_windows-x64")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                ("libmind_cpu_linux-x64.so", "mind_cpu_linux-x64")
            }
        }
    }
}

/// Fallback to a launcher shell script when native linking fails. This is the
/// executable-path analogue of the cdylib false-green guarded by #306: the
/// emitted artifact is a `#!/bin/bash` launcher that defers to the installed
/// MIND runtime, NOT a natively-linked binary. Reporting "Finished" with a
/// launcher in place of a real binary — silently — is exactly the kind of
/// false-green that must never happen. Surface it: name the artifact for what
/// it is and show why the native link failed, so a degraded build cannot pass
/// for a clean one.
pub(super) fn warn_launcher(output: &Path, link_result: Result<()>) {
    let link_err = link_result
        .err()
        .map(|e| {
            e.to_string()
                .lines()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("native link failed")
                .trim()
                .to_string()
        })
        .unwrap_or_else(|| "native link failed".to_string());
    eprintln!(
        "[WARN] {}: native link failed — emitting a launcher script that requires the \
         installed MIND runtime, not a native binary ({link_err})",
        output.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn only_explicit_cpu_uses_the_public_shim_only_policy() {
        assert!(!installed_runtime_required("cpu"));
        for backend in [
            "gpu", "cuda", "rocm", "metal", "webgpu", "oneapi", "cerebras",
        ] {
            assert!(installed_runtime_required(backend), "{backend}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_launcher_reads_the_root_the_build_locked() {
        let root = tempfile::tempdir().unwrap();
        let manifest = super::super::build_lock::temporary_manifest("locked", "src/main.mind");
        fs::write(root.path().join("Mind.toml"), manifest).unwrap();
        let out = root.path().join("launcher");
        let runtime = RuntimeLink {
            dir: root.path().to_path_buf(),
            file_name: "libmind_cuda.so",
            link_name: "mind_cuda",
        };
        super::super::write_launcher(&out, "cuda", &runtime, root.path()).unwrap();
        let script = fs::read_to_string(&out).unwrap();
        let entry = root.path().join("src/main.mind");
        assert!(script.contains("# locked v0.1.0"), "{script}");
        assert!(
            script.contains(&format!("MIND_ROOT=\"{}\"", root.path().display())),
            "{script}"
        );
        assert!(
            script.contains(&format!("MIND_ENTRY=\"{}\"", entry.display())),
            "{script}"
        );
    }
}
