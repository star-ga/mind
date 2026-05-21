// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! RFC 0008 Phase C — workspace integration tests.
//!
//! Gate: `cargo test --release --features "mlir-build std-surface cross-module-imports" mindc_workspace_phase_c`

use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

use libmind::workspace::{
    resolve_workspace_members, toposort_members, WorkspaceError, WorkspaceOpts,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write a minimal `Mind.toml` for a package with an optional path dep table.
fn write_manifest(dir: &Path, name: &str, path_deps: &[(&str, &str)]) {
    let mut content = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n"
    );
    if !path_deps.is_empty() {
        content.push_str("\n[dependencies]\n");
        for (dep_name, dep_path) in path_deps {
            content.push_str(&format!(
                "{dep_name} = {{ path = \"{dep_path}\" }}\n"
            ));
        }
    }
    fs::write(dir.join("Mind.toml"), &content).unwrap();
}

/// Write a minimal `.mind` source at `src/main.mind`.
fn write_source(dir: &Path, content: &str) {
    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("main.mind"), content).unwrap();
}

/// Minimal valid MIND source.
const HELLO_MIND: &str = "fn main() -> i64 { 42 }\n";

/// Write a workspace root `Mind.toml` with `[workspace]` block only (virtual manifest).
fn write_workspace_manifest(dir: &Path, members: &[&str], exclude: &[&str]) {
    let members_toml = members
        .iter()
        .map(|m| format!("    \"{m}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    let exclude_toml = exclude
        .iter()
        .map(|e| format!("    \"{e}\""))
        .collect::<Vec<_>>()
        .join(",\n");

    let mut content = format!("[workspace]\nmembers = [\n{members_toml}\n]\n");
    if !exclude.is_empty() {
        content.push_str(&format!("exclude = [\n{exclude_toml}\n]\n"));
    }
    fs::write(dir.join("Mind.toml"), &content).unwrap();
}

/// Write a workspace root that also carries a `[package]` block (non-virtual).
#[allow(dead_code)]
fn write_workspace_with_package(dir: &Path, name: &str, members: &[&str]) {
    let members_toml = members
        .iter()
        .map(|m| format!("    \"{m}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    let content = format!(
        "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"src/main.mind\"\n\n[workspace]\nmembers = [\n{members_toml}\n]\n"
    );
    fs::write(dir.join("Mind.toml"), &content).unwrap();
}

// ---------------------------------------------------------------------------
// Test 1: two members, no cross-deps — build resolves both
// ---------------------------------------------------------------------------

#[test]
fn workspace_two_members_no_deps_resolves_both() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Workspace root (virtual manifest)
    write_workspace_manifest(root, &["crates/alpha", "crates/beta"], &[]);

    // Member alpha
    let alpha = root.join("crates/alpha");
    fs::create_dir_all(&alpha).unwrap();
    write_manifest(&alpha, "alpha", &[]);
    write_source(&alpha, HELLO_MIND);

    // Member beta
    let beta = root.join("crates/beta");
    fs::create_dir_all(&beta).unwrap();
    write_manifest(&beta, "beta", &[]);
    write_source(&beta, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("should resolve workspace members");
    assert_eq!(members.len(), 2, "expected 2 members, got {}", members.len());

    let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"alpha"), "alpha missing from members");
    assert!(names.contains(&"beta"), "beta missing from members");
}

// ---------------------------------------------------------------------------
// Test 2: topology — b depends on a, build order is a then b
// ---------------------------------------------------------------------------

#[test]
fn workspace_toposort_dep_order_a_before_b() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["crates/a", "crates/b"], &[]);

    let a_dir = root.join("crates/a");
    fs::create_dir_all(&a_dir).unwrap();
    write_manifest(&a_dir, "a", &[]);
    write_source(&a_dir, HELLO_MIND);

    let b_dir = root.join("crates/b");
    fs::create_dir_all(&b_dir).unwrap();
    // b depends on a via relative path
    write_manifest(&b_dir, "b", &[("a", "../a")]);
    write_source(&b_dir, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("resolve");
    let sorted = toposort_members(&members).expect("toposort should succeed");

    let names: Vec<&str> = sorted.iter().map(|m| m.name.as_str()).collect();
    let pos_a = names.iter().position(|&n| n == "a").expect("a in sorted");
    let pos_b = names.iter().position(|&n| n == "b").expect("b in sorted");
    assert!(pos_a < pos_b, "a must come before b in topo order; got {:?}", names);
}

// ---------------------------------------------------------------------------
// Test 3: --package b builds only b (and a as its prerequisite)
// ---------------------------------------------------------------------------

#[test]
fn workspace_package_filter_selects_member_and_deps() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["crates/a", "crates/b", "crates/c"], &[]);

    let a_dir = root.join("crates/a");
    fs::create_dir_all(&a_dir).unwrap();
    write_manifest(&a_dir, "a", &[]);
    write_source(&a_dir, HELLO_MIND);

    let b_dir = root.join("crates/b");
    fs::create_dir_all(&b_dir).unwrap();
    write_manifest(&b_dir, "b", &[("a", "../a")]);
    write_source(&b_dir, HELLO_MIND);

    // c has no dep on a or b
    let c_dir = root.join("crates/c");
    fs::create_dir_all(&c_dir).unwrap();
    write_manifest(&c_dir, "c", &[]);
    write_source(&c_dir, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("resolve");

    let opts = WorkspaceOpts {
        package_filter: Some("b".to_string()),
    };
    let sorted = toposort_members(&members).expect("topo");
    let selected = opts.filter_members(&members, &sorted);

    let names: Vec<&str> = selected.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"b"), "b must be in selected");
    assert!(names.contains(&"a"), "a must be in selected (dep of b)");
    assert!(!names.contains(&"c"), "c must not be in selected (unrelated)");
}

// ---------------------------------------------------------------------------
// Test 4: cycle detection returns exit-2 diagnostic (no panic)
// ---------------------------------------------------------------------------

#[test]
fn workspace_cycle_returns_error_not_panic() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["crates/p", "crates/q"], &[]);

    let p_dir = root.join("crates/p");
    fs::create_dir_all(&p_dir).unwrap();
    write_manifest(&p_dir, "p", &[("q", "../q")]);
    write_source(&p_dir, HELLO_MIND);

    let q_dir = root.join("crates/q");
    fs::create_dir_all(&q_dir).unwrap();
    write_manifest(&q_dir, "q", &[("p", "../p")]);
    write_source(&q_dir, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("resolve should still work");
    let result = toposort_members(&members);

    assert!(result.is_err(), "expected cycle error, got Ok");
    match result.unwrap_err() {
        WorkspaceError::DependencyCycle(msg) => {
            assert!(
                msg.contains("p") || msg.contains("q"),
                "cycle message should name the packages: {msg}"
            );
        }
        other => panic!("expected DependencyCycle, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 5: [workspace.exclude] correctly excludes matching paths
// ---------------------------------------------------------------------------

#[test]
fn workspace_exclude_removes_matching_members() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Declare three members but exclude one
    write_workspace_manifest(
        root,
        &["crates/keep1", "crates/keep2", "scratch/ignore-me"],
        &["scratch"],
    );

    let keep1 = root.join("crates/keep1");
    fs::create_dir_all(&keep1).unwrap();
    write_manifest(&keep1, "keep1", &[]);
    write_source(&keep1, HELLO_MIND);

    let keep2 = root.join("crates/keep2");
    fs::create_dir_all(&keep2).unwrap();
    write_manifest(&keep2, "keep2", &[]);
    write_source(&keep2, HELLO_MIND);

    let scratch = root.join("scratch/ignore-me");
    fs::create_dir_all(&scratch).unwrap();
    write_manifest(&scratch, "ignore-me", &[]);
    write_source(&scratch, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("resolve");
    let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();

    assert!(names.contains(&"keep1"), "keep1 should be included");
    assert!(names.contains(&"keep2"), "keep2 should be included");
    assert!(
        !names.contains(&"ignore-me"),
        "ignore-me should be excluded by scratch/ pattern"
    );
}

// ---------------------------------------------------------------------------
// Test 6: virtual manifest (no [package]) builds successfully
// ---------------------------------------------------------------------------

#[test]
fn workspace_virtual_manifest_resolves_members() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Root has ONLY [workspace], no [package] block
    write_workspace_manifest(root, &["libs/core", "libs/utils"], &[]);

    let core = root.join("libs/core");
    fs::create_dir_all(&core).unwrap();
    write_manifest(&core, "core", &[]);
    write_source(&core, HELLO_MIND);

    let utils = root.join("libs/utils");
    fs::create_dir_all(&utils).unwrap();
    write_manifest(&utils, "utils", &[]);
    write_source(&utils, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("virtual manifest should resolve");
    assert_eq!(members.len(), 2, "expected 2 members");

    let sorted = toposort_members(&members).expect("toposort");
    assert_eq!(sorted.len(), 2);
}

// ---------------------------------------------------------------------------
// Test 7: member with missing Mind.toml reports a clear error
// ---------------------------------------------------------------------------

#[test]
fn workspace_missing_member_manifest_errors() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["crates/present", "crates/absent"], &[]);

    let present = root.join("crates/present");
    fs::create_dir_all(&present).unwrap();
    write_manifest(&present, "present", &[]);
    write_source(&present, HELLO_MIND);

    // "absent" directory exists but has no Mind.toml
    fs::create_dir_all(root.join("crates/absent")).unwrap();

    let result = resolve_workspace_members(root);
    assert!(result.is_err(), "expected error for missing member manifest");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("absent") || msg.contains("Mind.toml"),
        "error message should mention the missing member: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Test 8: WorkspaceMember carries the correct manifest root path
// ---------------------------------------------------------------------------

#[test]
fn workspace_member_root_paths_are_canonical() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["alpha"], &[]);
    let alpha = root.join("alpha");
    fs::create_dir_all(&alpha).unwrap();
    write_manifest(&alpha, "alpha", &[]);
    write_source(&alpha, HELLO_MIND);

    let members = resolve_workspace_members(root).expect("resolve");
    assert_eq!(members.len(), 1);
    let m = &members[0];

    // The member root must be an absolute path pointing to the alpha dir.
    assert!(m.root.is_absolute(), "member root should be absolute: {}", m.root.display());
    assert_eq!(
        m.root.file_name().unwrap().to_str().unwrap(),
        "alpha"
    );
}

// ---------------------------------------------------------------------------
// Test 9: path dep pointing outside the workspace is accepted in resolve
//         but flagged as an external dep (not built in Phase C)
// ---------------------------------------------------------------------------

#[test]
fn workspace_external_path_dep_is_recognised() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["inner"], &[]);

    let inner = root.join("inner");
    fs::create_dir_all(&inner).unwrap();
    // Points to a path outside the workspace root
    write_manifest(&inner, "inner", &[("outside", "../../somewhere")]);
    write_source(&inner, HELLO_MIND);

    // Resolution should succeed; the dep resolution just records it.
    let members = resolve_workspace_members(root).expect("resolve");
    let m = &members[0];
    let dep = m.path_deps.iter().find(|d| d.name == "outside");
    assert!(dep.is_some(), "outside dep should be recorded in path_deps");
    let dep = dep.unwrap();
    assert!(
        !dep.is_workspace_member,
        "outside dep should NOT be flagged as a workspace member"
    );
}

// ---------------------------------------------------------------------------
// Test 10: toposort of three members (a←b←c chain) respects full order
// ---------------------------------------------------------------------------

#[test]
fn workspace_toposort_three_member_chain() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    write_workspace_manifest(root, &["a", "b", "c"], &[]);

    for (name, deps) in [("a", vec![]), ("b", vec![("a", "../a")]), ("c", vec![("b", "../b")])] {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        write_manifest(&dir, name, &deps);
        write_source(&dir, HELLO_MIND);
    }

    let members = resolve_workspace_members(root).expect("resolve");
    let sorted = toposort_members(&members).expect("toposort");

    let names: Vec<&str> = sorted.iter().map(|m| m.name.as_str()).collect();
    let pos = |n: &str| names.iter().position(|&x| x == n).unwrap();
    assert!(pos("a") < pos("b"), "a < b");
    assert!(pos("b") < pos("c"), "b < c");
}

// ---------------------------------------------------------------------------
// Test 11: WorkspaceError has correct exit_code() values
// ---------------------------------------------------------------------------

#[test]
fn workspace_error_exit_codes() {
    let cycle = WorkspaceError::DependencyCycle("a -> b -> a".into());
    assert_eq!(cycle.exit_code(), 2, "cycle is a usage/config error → exit 2");

    let missing = WorkspaceError::MissingMemberManifest(PathBuf::from("crates/absent"));
    assert_eq!(missing.exit_code(), 1, "missing manifest is a build error → exit 1");
}

// ---------------------------------------------------------------------------
// Test 12: glob pattern in members expands directories
// ---------------------------------------------------------------------------

#[test]
fn workspace_glob_members_expands() {
    let tmp = TempDir::new().unwrap();
    let root = tmp.path();

    // Use a glob pattern to match all crates/* members
    write_workspace_manifest(root, &["crates/*"], &[]);

    let crates_dir = root.join("crates");
    for name in ["lib_one", "lib_two"] {
        let d = crates_dir.join(name);
        fs::create_dir_all(&d).unwrap();
        write_manifest(&d, name, &[]);
        write_source(&d, HELLO_MIND);
    }

    let members = resolve_workspace_members(root).expect("glob expand");
    let names: Vec<&str> = members.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"lib_one"), "lib_one should be expanded");
    assert!(names.contains(&"lib_two"), "lib_two should be expanded");
}
