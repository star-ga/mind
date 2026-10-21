// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Host LLVM triple for a project build that did not name an explicit target.
//!
//! The default triple is a function of the host architecture and operating
//! system (`std::env::consts::{ARCH, OS}`). An explicit `[targets.*].target`
//! value is applied by the caller and never reaches this module. Apple GPU
//! backends keep one fixed triple: their launcher is not a host object.
//!
//! Windows rows keep the `pc-windows-msvc` triple this path already emitted.
//! Cross-compilation Windows targets use the GNU triple in `crate::target`.

/// LLVM triple for one `(arch, os)` pair.
///
/// `arch` and `os` are the `std::env::consts` spellings (`x86_64` / `aarch64`,
/// `linux` / `macos` / `windows`). The six rows are the hosts `crate::target`
/// accepts. Any other pair keeps the historical fallback.
pub(super) fn default_host_triple(arch: &str, os: &str) -> &'static str {
    match (arch, os) {
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu",
        ("aarch64", "linux") => "aarch64-unknown-linux-gnu",
        ("x86_64", "macos") => "x86_64-apple-darwin",
        ("aarch64", "macos") => "aarch64-apple-darwin",
        ("x86_64", "windows") => "x86_64-pc-windows-msvc",
        ("aarch64", "windows") => "aarch64-pc-windows-msvc",
        _ => "x86_64-unknown-linux-gnu",
    }
}

/// Backend triple. Metal launchers stay on their explicit Darwin triple.
/// Every other backend, including GPU launchers, uses the host triple.
#[cfg_attr(not(feature = "mlir-build"), allow(dead_code))]
pub(super) fn get_target_triple(backend: &str) -> &'static str {
    match backend {
        "metal" | "metal-m4" => "aarch64-apple-darwin",
        _ => default_host_triple(std::env::consts::ARCH, std::env::consts::OS),
    }
}

#[cfg(test)]
mod tests {
    use super::default_host_triple;
    use super::get_target_triple;

    const PAIRS: &[(&str, &str, &str)] = &[
        ("x86_64", "linux", "x86_64-unknown-linux-gnu"),
        ("aarch64", "linux", "aarch64-unknown-linux-gnu"),
        ("x86_64", "macos", "x86_64-apple-darwin"),
        ("aarch64", "macos", "aarch64-apple-darwin"),
        ("x86_64", "windows", "x86_64-pc-windows-msvc"),
        ("aarch64", "windows", "aarch64-pc-windows-msvc"),
    ];

    #[test]
    fn default_host_triple_is_pinned_per_arch_and_os() {
        for (arch, os, triple) in PAIRS {
            assert_eq!(default_host_triple(arch, os), *triple, "{arch} {os}");
        }
        let live = default_host_triple(std::env::consts::ARCH, std::env::consts::OS);
        assert!(
            PAIRS.iter().any(|(arch, os, triple)| {
                *arch == std::env::consts::ARCH && *os == std::env::consts::OS && *triple == live
            }),
            "host {} {} -> {live} is outside the pinned pairs",
            std::env::consts::ARCH,
            std::env::consts::OS
        );
    }

    #[test]
    fn metal_backends_keep_the_explicit_triple() {
        assert_eq!(get_target_triple("metal"), "aarch64-apple-darwin");
        assert_eq!(get_target_triple("metal-m4"), "aarch64-apple-darwin");
    }

    #[test]
    fn other_backends_use_the_host_pair() {
        let host = default_host_triple(std::env::consts::ARCH, std::env::consts::OS);
        for backend in ["cpu", "cuda", "rocm", "webgpu", "directx", "oneapi"] {
            assert_eq!(get_target_triple(backend), host, "{backend}");
        }
    }
}
