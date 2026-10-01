// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Manifest-declared native C-ABI runtime sources (`[targets.*].native_sources`).
//!
//! A project whose `.mind` modules call C-ABI runtime shims (registered in
//! `STD_SURFACE_INTRINSICS`, lowered to `func.call @__mind_nerve_*`) lists the C
//! files that define them; their objects join the executable link, so those
//! symbols are DEFINED at link time instead of failing the native link and
//! dropping to a launcher stub. An absent or empty list adds no object, which
//! keeps the link byte-identical to the historical one (keystone-safe).
//!
//! They are compiled before the `.mind` sources: the names the C code calls
//! feed the internal-linkage plan (`private_linkage`), so a fn C calls keeps
//! its global symbol.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use super::TargetConfig;
use super::build_input_snapshot::BuildInputSnapshot;
use crate::eval::mlir_build;

/// Compile each declared native source to `target/obj/__native_<stem>.o` and
/// return the objects in declaration order. The sources come from the build's
/// input snapshot when one was captured, else from the target's manifest list
/// (relative to `project_root`).
pub(super) fn compile(
    project_root: &Path,
    supplied: Option<&BuildInputSnapshot>,
    target_config: Option<&TargetConfig>,
) -> Result<Vec<PathBuf>> {
    let sources: Vec<PathBuf> = match supplied {
        Some(inputs) => inputs
            .native_source_paths()
            .map(Path::to_path_buf)
            .collect(),
        None => target_config
            .and_then(|cfg| cfg.native_sources.as_deref())
            .unwrap_or_default()
            .iter()
            .map(|rel| project_root.join(rel))
            .collect(),
    };
    let mut objects = Vec::new();
    if sources.is_empty() {
        return Ok(objects);
    }
    let tools = mlir_build::resolve_tools()
        .map_err(|e| anyhow!("native_sources: MLIR build tools unavailable: {e}"))?;
    let obj_dir = project_root.join("target").join("obj");
    fs::create_dir_all(&obj_dir)?;
    for src in sources {
        if !src.exists() {
            return Err(anyhow!(
                "native_sources: declared C source not found: {}",
                src.display()
            ));
        }
        let stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("native_sources: invalid path {}", src.display()))?;
        let obj = obj_dir.join(format!("__native_{stem}.o"));
        mlir_build::compile_native_c_obj(&tools, &src, &obj, None)
            .map_err(|e| anyhow!("native_sources: compile failed for {}: {e}", src.display()))?;
        objects.push(obj);
    }
    Ok(objects)
}
