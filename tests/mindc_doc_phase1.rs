// Copyright 2025 STARGA Inc.
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

//! Integration tests for `mindc doc` (Phase 1).
//!
//! These tests invoke the `mindc` binary directly via `CARGO_BIN_EXE_mindc`
//! and assert on the generated HTML artefacts.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tempfile::tempdir;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

/// Returns the repository root (the directory containing `Cargo.toml`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ---------------------------------------------------------------------------
// Helper: a minimal MIND source with `///` doc-comments
// ---------------------------------------------------------------------------

const DOC_SRC: &str = r#"
/// Creates an empty vector with zero allocation.
///
/// Returns a `Vec` with `len = 0` and `cap = 0`.
pub fn vec_new() -> Vec {
    Vec { addr: 0, len: 0, cap: 0 }
}

/// Return the current logical length.
pub fn vec_len(v: Vec) -> i64 {
    v.len
}

struct Vec {
    addr: i64,
    len: i64,
    cap: i64,
}
"#;

// ---------------------------------------------------------------------------
// Test 1: basic doc generation on a file with `///` comments
// ---------------------------------------------------------------------------

#[test]
fn doc_generates_html_with_doc_comments() {
    let dir = tempdir().expect("tempdir");
    let src_file = dir.path().join("myvec.mind");
    let out_dir = dir.path().join("doc_out");

    fs::write(&src_file, DOC_SRC).expect("write source");

    let status = mindc()
        .arg("doc")
        .arg(src_file.to_str().unwrap())
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert!(status.success(), "mindc doc exited with non-zero status");

    // There should be an index.html
    assert!(out_dir.join("index.html").exists(), "index.html must exist");

    // And an html file named after the source (myvec.html in the root of out_dir)
    let html_path = out_dir.join("myvec.html");
    assert!(
        html_path.exists(),
        "myvec.html must exist at {}",
        html_path.display()
    );

    let html = fs::read_to_string(&html_path).expect("read myvec.html");

    // Must contain the fn name
    assert!(html.contains("vec_new"), "expected 'vec_new' in myvec.html");

    // Must contain the signature fragment
    assert!(
        html.contains("() -&gt; Vec") || html.contains("() -> Vec"),
        "expected signature params in myvec.html"
    );

    // Must contain the doc-comment body text (plain-comment content)
    assert!(
        html.contains("Creates an empty vector"),
        "expected doc comment text in myvec.html"
    );

    // Must contain second function
    assert!(html.contains("vec_len"), "expected 'vec_len' in myvec.html");
}

// ---------------------------------------------------------------------------
// Test 2: `--out=<custom>` writes to the custom directory
// ---------------------------------------------------------------------------

#[test]
fn doc_custom_out_dir() {
    let dir = tempdir().expect("tempdir");
    let src_file = dir.path().join("test.mind");
    let out_dir = dir.path().join("custom_output");

    fs::write(&src_file, "pub fn simple() -> i64 { 0 }").expect("write source");

    let status = mindc()
        .arg("doc")
        .arg(src_file.to_str().unwrap())
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert!(status.success(), "mindc doc failed");
    assert!(
        out_dir.join("index.html").exists(),
        "index.html must be in custom dir"
    );
    assert!(
        out_dir.join("test.html").exists(),
        "test.html must be in custom dir"
    );
}

// ---------------------------------------------------------------------------
// Test 3: `--no-deps` on user paths does not include stdlib
// ---------------------------------------------------------------------------

#[test]
fn doc_no_deps_restricts_to_given_paths() {
    let dir = tempdir().expect("tempdir");
    let src_file = dir.path().join("mylib.mind");
    let out_dir = dir.path().join("doc_no_deps");

    fs::write(&src_file, "pub fn mylib_fn() -> i64 { 1 }").expect("write source");

    let status = mindc()
        .arg("doc")
        .arg(src_file.to_str().unwrap())
        .arg("--no-deps")
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert!(status.success(), "mindc doc --no-deps failed");

    let index_html = fs::read_to_string(out_dir.join("index.html")).expect("index.html");
    // Only the user-supplied file should appear; no std paths
    assert!(index_html.contains("mylib"), "index should mention mylib");
    // No std/vec or std/io items should appear since we only gave mylib.mind
    assert!(
        !index_html.contains("vec_new"),
        "stdlib items must not appear with --no-deps and explicit paths"
    );
}

// ---------------------------------------------------------------------------
// Test 4: search-index.json is emitted and contains item entries
// ---------------------------------------------------------------------------

#[test]
fn doc_emits_search_index() {
    let dir = tempdir().expect("tempdir");
    let src_file = dir.path().join("indexed.mind");
    let out_dir = dir.path().join("doc_search");

    fs::write(&src_file, "pub fn search_target() -> i64 { 42 }").expect("write source");

    let status = mindc()
        .arg("doc")
        .arg(src_file.to_str().unwrap())
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert!(status.success(), "mindc doc failed");

    let json_path = out_dir.join("search-index.json");
    assert!(json_path.exists(), "search-index.json must exist");

    let json_text = fs::read_to_string(&json_path).expect("read search-index.json");
    let entries: serde_json::Value =
        serde_json::from_str(&json_text).expect("valid JSON in search-index.json");

    let arr = entries.as_array().expect("search-index is a JSON array");
    assert!(!arr.is_empty(), "search-index must have at least one entry");

    // At least one entry with name = "search_target"
    let has_entry = arr.iter().any(|e| {
        e.get("name")
            .and_then(|v| v.as_str())
            .map(|s| s == "search_target")
            .unwrap_or(false)
    });
    assert!(has_entry, "search-index must contain 'search_target'");
}

// ---------------------------------------------------------------------------
// Test 5: exit code 0 on success
// ---------------------------------------------------------------------------

#[test]
fn doc_exit_code_zero_on_success() {
    let dir = tempdir().expect("tempdir");
    let src_file = dir.path().join("ok.mind");
    let out_dir = dir.path().join("doc_exit0");

    fs::write(&src_file, "pub fn ok_fn() -> i64 { 0 }").expect("write source");

    let status = mindc()
        .arg("doc")
        .arg(src_file.to_str().unwrap())
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert_eq!(status.code(), Some(0), "exit code must be 0 on success");
}

// ---------------------------------------------------------------------------
// Test 6: exit code 2 on invalid CLI args (unrecognised flag)
// ---------------------------------------------------------------------------

#[test]
fn doc_exit_code_2_on_bad_args() {
    let status = mindc()
        .arg("doc")
        .arg("--this-flag-does-not-exist")
        .status()
        .expect("spawn mindc doc");

    // clap exits with code 2 for usage errors
    assert_eq!(status.code(), Some(2), "exit code must be 2 for bad args");
}

// ---------------------------------------------------------------------------
// Test 7: `std/vec.mind` (repo file) produces vec.html with fn signatures
// ---------------------------------------------------------------------------

#[test]
fn doc_std_vec_mind_produces_vec_html() {
    let std_vec = repo_root().join("std/vec.mind");
    if !std_vec.exists() {
        eprintln!("skipping: std/vec.mind not found");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("doc_std_vec");

    let status = mindc()
        .arg("doc")
        .arg(std_vec.to_str().unwrap())
        .arg(format!("--out={}", out_dir.display()))
        .status()
        .expect("spawn mindc doc");

    assert!(status.success(), "mindc doc std/vec.mind failed");

    // vec.html should exist somewhere under out_dir
    let vec_html = find_html_containing(&out_dir, "vec_new");
    assert!(
        vec_html.is_some(),
        "expected a .html file containing 'vec_new' under {}",
        out_dir.display()
    );

    let html_content = fs::read_to_string(vec_html.unwrap()).unwrap();
    assert!(
        html_content.contains("vec_push"),
        "vec.html must contain vec_push signature"
    );
}

/// Walk `dir` recursively; return the path of the first `.html` file containing `needle`.
fn find_html_containing(dir: &std::path::Path, needle: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() {
            if let Some(found) = find_html_containing(&path, needle) {
                return Some(found);
            }
        } else if path.extension().is_some_and(|e| e == "html") {
            if let Ok(text) = fs::read_to_string(&path) {
                if text.contains(needle) {
                    return Some(path);
                }
            }
        }
    }
    None
}
