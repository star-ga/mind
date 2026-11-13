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

use std::fs;
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

/// The launcher-script fallback of a failed native link: a script that runs the
/// installed runtime. `project_root` is the root whose build lock the caller holds;
/// a search from the current directory could read another standalone build's
/// temporary manifest (`build_lock`).
pub(super) fn write_launcher(
    output: &Path,
    backend: &str,
    runtime: &RuntimeLink,
    project_root: &Path,
) -> Result<()> {
    let manifest = super::load_manifest(project_root)?;
    let entry_path = project_root.join(&manifest.build.entry);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let script = format!(
            r#"#!/bin/bash
# {name} v{version} - Built with MIND
# Backend: {backend}
# Runtime: {runtime}

MIND_ROOT="{project_root}"
MIND_ENTRY="{entry}"
MIND_BACKEND="{backend}"
MIND_LIB_DIR="{lib_dir}"

export LD_LIBRARY_PATH="$MIND_LIB_DIR:$LD_LIBRARY_PATH"
export MIND_SOURCE_ROOT="$MIND_ROOT/src"
export MIND_BACKEND="$MIND_BACKEND"

# Execute the runtime interpreter
if [ -f "$MIND_LIB_DIR/mind-runtime" ]; then
    exec "$MIND_LIB_DIR/mind-runtime" --backend "$MIND_BACKEND" --entry "$MIND_ENTRY" "$@"
elif [ -f "$MIND_LIB_DIR/{runtime}" ]; then
    # Direct library execution via ld.so
    exec /lib64/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    exec /lib/ld-linux-x86-64.so.2 --library-path "$MIND_LIB_DIR" "$MIND_LIB_DIR/{runtime}" --entry "$MIND_ENTRY" "$@" 2>/dev/null || \
    {{
        echo "Error: MIND runtime not executable"
        echo "See https://mindlang.dev/enterprise for licensing."
        exit 1
    }}
else
    echo "Error: MIND runtime not found at $MIND_LIB_DIR/{runtime}"
    echo "See https://mindlang.dev/enterprise for licensing."
    exit 1
fi
"#,
            name = manifest.package.name,
            version = manifest.package.version,
            backend = backend,
            runtime = runtime.file_name,
            project_root = project_root.display(),
            entry = entry_path.display(),
            lib_dir = runtime.dir.display(),
        );

        fs::write(output, script)?;

        let mut perms = fs::metadata(output)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(output, perms)?;
    }

    #[cfg(not(unix))]
    {
        let script = format!(
            "@echo off\r\nrem {} v{} - Built with MIND\r\nset MIND_BACKEND={}\r\nset PATH=%PATH%;{}\r\n\"{}/mind-runtime.exe\" --backend {} --entry \"{}\" %*\r\n",
            manifest.package.name,
            manifest.package.version,
            backend,
            runtime.dir.display(),
            runtime.dir.display(),
            backend,
            entry_path.display(),
        );
        fs::write(output, script)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        write_launcher(&out, "cuda", &runtime, root.path()).unwrap();
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
