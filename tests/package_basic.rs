#![cfg(feature = "pkg")]

// Copyright (c) 2025 STARGA Inc. and MIND Language Contributors
// SPDX-License-Identifier: MIT
// Part of the MIND project (Machine Intelligence Native Design).

use std::fs;

use mind::package::build_package;

use mind::package::inspect_package;

use mind::package::MindManifest;

use tempfile::tempdir;

#[test]
fn build_and_inspect_roundtrip() {
    let dir = tempdir().expect("create temp directory");
    let model_path = dir.path().join("model.mlir");
    fs::write(&model_path, "module @main {}").expect("write model");

    let manifest = MindManifest {
        name: "demo".into(),
        version: "0.1.0".into(),
        authors: vec!["mind".into()],
        description: Some("Test package".into()),
        license: None,
        dependencies: None,
        files: vec!["model.mlir".into()],
        checksums: None,
    };

    let package_path = dir.path().join("demo.mindpkg");
    let artifact = model_path.to_string_lossy().into_owned();
    build_package(
        package_path.to_str().expect("package path"),
        &[artifact.as_str()],
        &manifest,
    )
    .expect("package build");

    let parsed =
        inspect_package(package_path.to_str().expect("package path")).expect("inspect package");
    assert_eq!(parsed.name, "demo");
    assert!(parsed
        .checksums
        .as_ref()
        .expect("checksums present")
        .contains_key("model.mlir"));
}
