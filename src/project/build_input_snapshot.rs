// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Immutable manifest inputs that can alter a project artifact.
//!
//! The command driver captures this beside the MIND source snapshot, before a
//! cache lookup. The project compiler then consumes the same target selection
//! and export list. Native C sources remain path inputs to an external compiler,
//! so builds that declare them are deliberately cache-ineligible until their
//! complete preprocessor input graph can be captured.

use std::fs;
use std::path::Path;
#[cfg(feature = "mlir-build")]
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

use super::{EmitKind, ProjectManifest};

#[derive(Debug, Clone)]
struct NativeSourceSnapshot {
    declared_path: String,
    #[cfg(feature = "mlir-build")]
    absolute_path: PathBuf,
    bytes: Vec<u8>,
}

/// Manifest and native-file inputs captured for one selected target block.
#[derive(Debug, Clone)]
pub(crate) struct BuildInputSnapshot {
    target_name: String,
    backend: String,
    target_triple: Option<String>,
    entry: String,
    sources: Option<Vec<String>>,
    exports: Vec<String>,
    native_sources: Vec<NativeSourceSnapshot>,
}

impl BuildInputSnapshot {
    /// Capture the effective entry, selected target configuration, and every
    /// declared native C file exactly once. The effective entry is already
    /// resolved by the command driver, so a CLI source override remains the
    /// entry consumed by both the cache key and compilation. Missing native
    /// files fail before cache lookup.
    pub(crate) fn capture(
        project_root: &Path,
        manifest: &ProjectManifest,
        target_name: &str,
        emit: EmitKind,
        effective_entry: &str,
    ) -> Result<Self> {
        let target = manifest.targets.get(target_name);
        let backend = target
            .map(|config| config.backend.clone())
            .unwrap_or_else(|| target_name.to_string());
        // Before the cache lookup, so a cached MLIR artifact is never served for a
        // manifest the MLIR route cannot honor.
        super::toolchain_pin::admit_mlir_route(manifest.build.backend, target_name, &backend)?;
        let target_triple = target.and_then(|config| config.target.clone());
        let sources = target.and_then(|config| config.sources.clone());
        let native_paths = if emit == EmitKind::Cdylib {
            &[][..]
        } else {
            target
                .and_then(|config| config.native_sources.as_deref())
                .unwrap_or_default()
        };
        let mut native_sources = Vec::with_capacity(native_paths.len());
        for declared_path in native_paths {
            let absolute_path = project_root.join(declared_path);
            let bytes = fs::read(&absolute_path).with_context(|| {
                format!(
                    "native_sources: cannot read declared C source {}",
                    absolute_path.display()
                )
            })?;
            if !absolute_path.is_file() {
                return Err(anyhow!(
                    "native_sources: declared C source is not a file: {}",
                    absolute_path.display()
                ));
            }
            native_sources.push(NativeSourceSnapshot {
                declared_path: declared_path.clone(),
                #[cfg(feature = "mlir-build")]
                absolute_path,
                bytes,
            });
        }

        Ok(Self {
            target_name: target_name.to_string(),
            backend,
            target_triple,
            entry: effective_entry.to_string(),
            sources,
            exports: manifest.exports.c_abi.clone(),
            native_sources,
        })
    }

    pub(crate) fn target_name(&self) -> &str {
        &self.target_name
    }

    pub(crate) fn backend(&self) -> &str {
        &self.backend
    }

    pub(crate) fn target_triple(&self) -> Option<&str> {
        self.target_triple.as_deref()
    }

    pub(crate) fn entry(&self) -> &str {
        &self.entry
    }

    pub(crate) fn sources(&self) -> Option<&[String]> {
        self.sources.as_deref()
    }

    pub(crate) fn exports(&self) -> &[String] {
        &self.exports
    }

    #[cfg(feature = "mlir-build")]
    pub(crate) fn native_source_paths(&self) -> impl Iterator<Item = &Path> {
        self.native_sources
            .iter()
            .map(|source| source.absolute_path.as_path())
    }

    /// Native sources are compiled by clang from filesystem paths. Their
    /// transitive includes and compiler environment are not immutable yet, so
    /// publishing a whole-artifact cache entry would make a stronger claim
    /// than the compiler can prove. Cdylibs do not consume native_sources.
    pub(crate) fn cache_eligible(&self, emit: EmitKind) -> bool {
        emit == EmitKind::Cdylib || self.native_sources.is_empty()
    }

    /// Canonical, length-delimited bytes for the cache-key dependency set.
    /// Length framing makes embedded separators and non-UTF-8 C bytes
    /// unambiguous; declaration order remains artifact-visible.
    pub(crate) fn identity_bytes(&self) -> Vec<u8> {
        let mut encoded = Vec::new();
        push_field(&mut encoded, b"format", b"mind-build-inputs-v1");
        push_field(&mut encoded, b"target-name", self.target_name.as_bytes());
        push_field(&mut encoded, b"backend", self.backend.as_bytes());
        match &self.target_triple {
            Some(triple) => push_field(&mut encoded, b"target-triple", triple.as_bytes()),
            None => push_field(&mut encoded, b"target-triple-none", b""),
        }
        push_field(&mut encoded, b"entry", self.entry.as_bytes());
        match &self.sources {
            Some(sources) => {
                push_count(&mut encoded, b"sources", sources.len());
                for source in sources {
                    push_field(&mut encoded, b"source-path", source.as_bytes());
                }
            }
            None => push_field(&mut encoded, b"sources-none", b""),
        }
        push_count(&mut encoded, b"exports", self.exports.len());
        for export in &self.exports {
            push_field(&mut encoded, b"export", export.as_bytes());
        }
        push_count(&mut encoded, b"native-sources", self.native_sources.len());
        for source in &self.native_sources {
            push_field(
                &mut encoded,
                b"native-path",
                source.declared_path.as_bytes(),
            );
            push_field(&mut encoded, b"native-bytes", &source.bytes);
        }
        encoded
    }
}

fn push_count(encoded: &mut Vec<u8>, label: &[u8], count: usize) {
    push_field(encoded, label, &(count as u64).to_le_bytes());
}

fn push_field(encoded: &mut Vec<u8>, label: &[u8], value: &[u8]) {
    encoded.extend_from_slice(&(label.len() as u64).to_le_bytes());
    encoded.extend_from_slice(label);
    encoded.extend_from_slice(&(value.len() as u64).to_le_bytes());
    encoded.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::BuildInputSnapshot;
    use crate::project::{EmitKind, load_manifest};

    fn write_manifest(root: &Path, backend: &str, triple: &str, exports: &str, native: &str) {
        fs::write(
            root.join("Mind.toml"),
            format!(
                "[package]\nname = \"inputs\"\nversion = \"0.1.0\"\n\n\
                 [targets.cpu]\nbackend = \"{backend}\"\ntarget = \"{triple}\"\n\
                 native_sources = [\"{native}\"]\n\n[exports]\nc_abi = [\"{exports}\"]\n"
            ),
        )
        .unwrap();
    }

    fn capture_binary(root: &Path) -> BuildInputSnapshot {
        let manifest = load_manifest(root).unwrap();
        BuildInputSnapshot::capture(
            root,
            &manifest,
            "cpu",
            EmitKind::Binary,
            &manifest.build.entry,
        )
        .unwrap()
    }

    use std::path::Path;

    #[test]
    fn identity_changes_with_each_manifest_and_native_input() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("shim.c"), b"long shim(void){return 1;}\n").unwrap();
        fs::write(tmp.path().join("other.c"), b"long shim(void){return 1;}\n").unwrap();
        write_manifest(
            tmp.path(),
            "cpu",
            "x86_64-unknown-linux-gnu",
            "export_a",
            "shim.c",
        );
        let first = capture_binary(tmp.path());

        fs::write(tmp.path().join("shim.c"), b"long shim(void){return 2;}\n").unwrap();
        let changed_bytes = capture_binary(tmp.path());
        assert_ne!(first.identity_bytes(), changed_bytes.identity_bytes());

        fs::write(tmp.path().join("shim.c"), b"long shim(void){return 1;}\n").unwrap();
        write_manifest(
            tmp.path(),
            "cpu",
            "x86_64-unknown-linux-gnu",
            "export_b",
            "shim.c",
        );
        let changed_export = capture_binary(tmp.path());
        assert_ne!(first.identity_bytes(), changed_export.identity_bytes());

        write_manifest(
            tmp.path(),
            "llvm",
            "x86_64-unknown-linux-gnu",
            "export_a",
            "shim.c",
        );
        let changed_backend = capture_binary(tmp.path());
        assert_ne!(first.identity_bytes(), changed_backend.identity_bytes());

        write_manifest(
            tmp.path(),
            "cpu",
            "aarch64-unknown-linux-gnu",
            "export_a",
            "shim.c",
        );
        let changed_triple = capture_binary(tmp.path());
        assert_ne!(first.identity_bytes(), changed_triple.identity_bytes());

        write_manifest(
            tmp.path(),
            "cpu",
            "x86_64-unknown-linux-gnu",
            "export_a",
            "other.c",
        );
        let changed_native_path = capture_binary(tmp.path());
        assert_ne!(first.identity_bytes(), changed_native_path.identity_bytes());

        write_manifest(
            tmp.path(),
            "cpu",
            "x86_64-unknown-linux-gnu",
            "export_a",
            "shim.c",
        );
        let mut changed_entry_manifest = load_manifest(tmp.path()).unwrap();
        changed_entry_manifest.build.entry = "src/other.mind".to_string();
        let changed_entry = BuildInputSnapshot::capture(
            tmp.path(),
            &changed_entry_manifest,
            "cpu",
            EmitKind::Binary,
            &changed_entry_manifest.build.entry,
        )
        .unwrap();
        assert_ne!(first.identity_bytes(), changed_entry.identity_bytes());

        let mut changed_sources_manifest = load_manifest(tmp.path()).unwrap();
        changed_sources_manifest
            .targets
            .get_mut("cpu")
            .unwrap()
            .sources = Some(vec!["src/main.mind".to_string()]);
        let changed_sources = BuildInputSnapshot::capture(
            tmp.path(),
            &changed_sources_manifest,
            "cpu",
            EmitKind::Binary,
            &changed_sources_manifest.build.entry,
        )
        .unwrap();
        assert_ne!(first.identity_bytes(), changed_sources.identity_bytes());

        assert_eq!(first.backend(), "cpu");
        assert_eq!(first.target_triple(), Some("x86_64-unknown-linux-gnu"));
        assert_eq!(first.exports(), &["export_a"]);
        assert!(!first.cache_eligible(EmitKind::Binary));
        assert!(first.cache_eligible(EmitKind::Cdylib));
    }

    #[test]
    fn identity_preserves_export_and_native_declaration_order() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.c"), b"long a(void){return 1;}\n").unwrap();
        fs::write(tmp.path().join("b.c"), b"long b(void){return 2;}\n").unwrap();
        fs::write(
            tmp.path().join("Mind.toml"),
            "[package]\nname=\"inputs\"\nversion=\"0.1.0\"\n\n\
             [targets.cpu]\nbackend=\"cpu\"\nnative_sources=[\"a.c\",\"b.c\"]\n\n\
             [exports]\nc_abi=[\"a\",\"b\"]\n",
        )
        .unwrap();
        let manifest = load_manifest(tmp.path()).unwrap();
        let first = BuildInputSnapshot::capture(
            tmp.path(),
            &manifest,
            "cpu",
            EmitKind::Binary,
            &manifest.build.entry,
        )
        .unwrap();

        fs::write(
            tmp.path().join("Mind.toml"),
            "[package]\nname=\"inputs\"\nversion=\"0.1.0\"\n\n\
             [targets.cpu]\nbackend=\"cpu\"\nnative_sources=[\"b.c\",\"a.c\"]\n\n\
             [exports]\nc_abi=[\"b\",\"a\"]\n",
        )
        .unwrap();
        let manifest = load_manifest(tmp.path()).unwrap();
        let second = BuildInputSnapshot::capture(
            tmp.path(),
            &manifest,
            "cpu",
            EmitKind::Binary,
            &manifest.build.entry,
        )
        .unwrap();
        assert_ne!(first.identity_bytes(), second.identity_bytes());
    }

    #[cfg(all(
        unix,
        feature = "mlir-build",
        feature = "std-surface",
        feature = "cross-module-imports",
        feature = "ffi-c-user"
    ))]
    #[test]
    fn manifest_edit_after_key_compiles_the_captured_exports() {
        use crate::build::{BuildOpts, run_build};
        use crate::project::source_snapshot;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir(root.join("src")).unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(
            root.join("src/main.mind"),
            "fn export_a() -> i64 { 11 }\nfn export_b() -> i64 { 22 }\n",
        )
        .unwrap();
        let manifest = |export: &str| {
            format!(
                "[package]\nname=\"captured_exports\"\nversion=\"0.1.0\"\n\n\
                 [build]\nentry=\"src/main.mind\"\nemit=\"cdylib\"\n\n\
                 [targets.cpu]\nbackend=\"cpu\"\n\n[exports]\nc_abi=[\"{export}\"]\n"
            )
        };
        fs::write(root.join("Mind.toml"), manifest("export_a")).unwrap();

        let manifest_path = root.join("Mind.toml");
        source_snapshot::install_test_hook(move || {
            fs::write(manifest_path, manifest("export_b")).unwrap();
        });
        let first = root.join("first.so");
        let entry = root.join("src/main.mind");
        let first_build = run_build(&BuildOpts {
            paths: vec![entry.clone()],
            target: Some("cpu".to_string()),
            emit: Some(EmitKind::Cdylib),
            out: Some(first.clone()),
            ..BuildOpts::default()
        })
        .unwrap();
        assert_eq!(first_build.cache_stats.misses, 1);
        unsafe {
            let lib = libloading::Library::new(&first).unwrap();
            let has_a = lib
                .get::<unsafe extern "C" fn()>(b"mind_fn_export_a_v1_invoke\0")
                .is_ok();
            let has_b = lib
                .get::<unsafe extern "C" fn()>(b"mind_fn_export_b_v1_invoke\0")
                .is_ok();
            assert_eq!(
                (has_a, has_b),
                (true, false),
                "post-key manifest edit changed the compiled export set"
            );
        }

        let second = root.join("second.so");
        let second_build = run_build(&BuildOpts {
            paths: vec![entry],
            target: Some("cpu".to_string()),
            emit: Some(EmitKind::Cdylib),
            out: Some(second.clone()),
            ..BuildOpts::default()
        })
        .unwrap();
        assert_eq!(second_build.cache_stats.misses, 1);
        unsafe {
            let lib = libloading::Library::new(&second).unwrap();
            let _: libloading::Symbol<
                unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i32,
            > = lib.get(b"mind_fn_export_b_v1_invoke\0").unwrap();
        }
    }

    #[cfg(all(
        unix,
        feature = "mlir-build",
        feature = "std-surface",
        feature = "cross-module-imports",
        feature = "ffi-c-user"
    ))]
    #[test]
    fn entry_edit_after_key_compiles_the_captured_entry() {
        use crate::build::{BuildOpts, run_build};
        use crate::project::source_snapshot;

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir(root.join("src")).unwrap();
        fs::create_dir(root.join(".git")).unwrap();
        let first_entry = root.join("src/first.mind");
        let second_entry = root.join("src/second.mind");
        fs::write(&first_entry, "fn main() -> i64 { 11 }\n").unwrap();
        fs::write(&second_entry, "fn main() -> i64 { 22 }\n").unwrap();
        let manifest = |entry: &str| {
            format!(
                "[package]\nname=\"captured_entry\"\nversion=\"0.1.0\"\n\n\
                 [build]\nentry=\"{entry}\"\n\n\
                 [targets.cpu]\nbackend=\"cpu\"\n\
                 sources=[\"src/first.mind\",\"src/second.mind\"]\n"
            )
        };
        fs::write(root.join("Mind.toml"), manifest("src/first.mind")).unwrap();

        let manifest_path = root.join("Mind.toml");
        source_snapshot::install_test_hook(move || {
            fs::write(manifest_path, manifest("src/second.mind")).unwrap();
        });
        let first_out = root.join("first.so");
        let first_build = run_build(&BuildOpts {
            paths: vec![first_entry],
            target: Some("cpu".to_string()),
            emit: Some(EmitKind::Cdylib),
            out: Some(first_out.clone()),
            ..BuildOpts::default()
        })
        .unwrap();
        assert_eq!(first_build.cache_stats.misses, 1);
        unsafe {
            let lib = libloading::Library::new(&first_out).unwrap();
            let main: libloading::Symbol<unsafe extern "C" fn() -> i64> =
                lib.get(b"main\0").unwrap();
            assert_eq!(main(), 11);
        }

        let second_out = root.join("second.so");
        let second_build = run_build(&BuildOpts {
            paths: vec![second_entry],
            target: Some("cpu".to_string()),
            emit: Some(EmitKind::Cdylib),
            out: Some(second_out.clone()),
            ..BuildOpts::default()
        })
        .unwrap();
        assert_eq!(second_build.cache_stats.misses, 1);
        unsafe {
            let lib = libloading::Library::new(&second_out).unwrap();
            let main: libloading::Symbol<unsafe extern "C" fn() -> i64> =
                lib.get(b"main\0").unwrap();
            assert_eq!(main(), 22);
        }
    }
}
