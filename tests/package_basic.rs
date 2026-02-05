#![cfg(feature = "pkg")]

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

use std::fs;

use libmind::package::build_package;

use libmind::package::inspect_package;

use libmind::package::MindManifest;

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
