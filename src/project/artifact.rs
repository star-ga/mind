// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! WHERE a build's artifact goes and WHAT it is called — the one owner.
//!
//! Both facts had two owners that disagreed, and both splits reached disk: the
//! NAME was derived from `[package] name` by `build::run_build` and from
//! `[build] output` (serde default `"app"`) by
//! [`crate::project::build_project`], so the spelling depended on whether the
//! host had a native backend; the `target/<profile>/` DIRECTORY was spelled out
//! separately in `build::default_artifact_path` and in that same function, so a
//! change to the layout had to be made twice or not at all.
//!
//! Everything below the artifact's own extension lives here. The emit-aware
//! caller still owns the decoration (`lib….so` / `.o`), because that is a fact
//! about the emit kind, not about the project.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use super::ProjectManifest;

/// The `[targets.<name>]` block a build reads when the caller named no target.
///
/// One owner for the fallback block name: [`crate::project::build_project`] and
/// [`artifact_stem`] must agree on WHICH block is consulted, or a
/// `[targets.cpu] output` would be honoured by one and not the other — the same
/// class of split the artifact name itself suffered.
pub const DEFAULT_TARGET_BLOCK: &str = "cpu";

/// The artifact NAME STEM — no directory, no extension. The ONE owner.
///
/// RFC 0008 §3 precedence, highest first:
///
/// 1. `[targets.<target_name>] output` — a block may rename its own artifact;
/// 2. `[build] output` — the project-wide declaration;
/// 3. `[package] name` — the documented default.
///
/// `target_name` is the caller's `--target` / selected block name; `None` means
/// none was selected and [`DEFAULT_TARGET_BLOCK`] is consulted, exactly as
/// [`crate::project::build_project`] does.
///
/// # Why this exists
///
/// The stem had two owners that disagreed. `build::run_build` derived it from
/// `package.name` and ignored `[build] output` outright; `build_project`
/// derived it from `[build] output`, whose default was the literal `"app"`.
/// Both names reached disk from one manifest — the orchestrator renamed the
/// compile path's file to its own spelling on success, while a build refused
/// before that rename (no native backend, `E5003`) left the OTHER spelling
/// behind. So the artifact's name depended on a host capability, a declared
/// `[build] output` silently did nothing, and every downstream reader had to
/// pick one of two rules to hand-type.
pub fn artifact_stem<'a>(manifest: &'a ProjectManifest, target_name: Option<&str>) -> &'a str {
    manifest
        .targets
        .get(target_name.unwrap_or(DEFAULT_TARGET_BLOCK))
        .and_then(|block| block.output.as_deref())
        .or(manifest.build.output.as_deref())
        .unwrap_or(&manifest.package.name)
}

/// The directory a build of this profile writes its artifact into —
/// `<project_root>/target/{release,debug}`, cargo's convention so both can
/// coexist in one tree.
///
/// One owner for the same reason [`artifact_stem`] is one: the compile path and
/// the `mindc build` orchestrator must resolve the SAME directory, or the file
/// one of them writes is not the file the other reports. Creating the directory
/// stays with the caller — a resolver that touches the filesystem cannot be
/// asked "where would this go" without side effects.
pub fn artifact_dir(project_root: &Path, release: bool) -> PathBuf {
    project_root.join("target").join(profile_name(release))
}

/// The profile directory's NAME — the leaf [`artifact_dir`] appends, and the
/// string a verbose build REPORTS.
///
/// Split out so the reported profile and the written directory cannot disagree:
/// a build that prints `debug` and writes `release/` is the same two-owner
/// defect as the artifact name, one level down.
pub fn profile_name(release: bool) -> &'static str {
    if release { "release" } else { "debug" }
}

/// Module an import path links. Exact key, then the `crate.` key, then (only
/// for one segment) `unique_stem`. A longer path that is not itself a module
/// uses the longest prefix that is a module, so an item path links the module
/// that defines the item instead of matching nothing.
pub(super) fn resolve_import_module(
    path: &[String],
    has_module: impl Fn(&str) -> bool,
    unique_stem: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    if path.is_empty() || matches!(path.first().map(String::as_str), Some("std")) {
        return None;
    }
    let module_key = |key: &str| -> Option<String> {
        if has_module(key) {
            return Some(key.to_string());
        }
        if !key.starts_with("crate.") {
            let prefixed = format!("crate.{key}");
            if has_module(&prefixed) {
                return Some(prefixed);
            }
        }
        None
    };
    if let Some(found) = module_key(&path.join(".")) {
        return Some(found);
    }
    if path.len() == 1 {
        return unique_stem(&path[0]);
    }
    for end in (1..path.len()).rev() {
        if let Some(found) = module_key(&path[..end].join(".")) {
            return Some(found);
        }
    }
    None
}

/// Add `pub import` targets of `start` to `owner`'s collision map. A direct
/// import of the same path keeps the target already recorded. Cycles stop.
pub(super) fn extend_public_import_targets(
    owner: &str,
    start: &str,
    public_imports: &BTreeMap<String, Vec<Vec<String>>>,
    resolve: impl Fn(&[String]) -> Option<String>,
    resolved: &mut BTreeMap<(String, Vec<String>), String>,
) {
    let mut seen = BTreeSet::from([start.to_string()]);
    let mut queue = vec![start.to_string()];
    while let Some(module_key) = queue.pop() {
        let Some(paths) = public_imports.get(&module_key) else {
            continue;
        };
        for path in paths {
            let Some(next) = resolve(path) else {
                continue;
            };
            resolved
                .entry((owner.to_string(), path.clone()))
                .or_insert_with(|| next.clone());
            if seen.insert(next.clone()) {
                queue.push(next);
            }
        }
    }
}
