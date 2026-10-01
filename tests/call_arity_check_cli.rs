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

// Part of the MIND project (Machine Intelligence Native Design).

//! Call arity (E2005) through the real `mindc check` CLI, in every feature set.
//!
//! Under `cross-module-imports` the CLI installs a project module table even for
//! a single file, so a call to a same-file function resolves through that table
//! and its arity error is phrased "imported `f` expects ...". That message used
//! to classify as the generic type-error code, and the fn-body pass keeps only a
//! whitelist of sound codes, so every arity error inside a function body was
//! dropped and `check` exited 0. The library-level tests in
//! `intra_module_call_arity.rs` never install the project table and could not
//! see it; these drive the binary.

mod common;

use std::path::Path;
use std::process::{Command, Output};

fn check(path: &Path) -> Output {
    Command::new(common::require_mindc())
        .arg("check")
        .arg("--no-fmt")
        .arg(path)
        .output()
        .expect("spawn mindc check")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn wrong_arity_inside_fn_body_is_e2005() {
    let dir = common::scratch_dir("call_arity_body");
    let src = dir.join("body.mind");
    std::fs::write(
        &src,
        "fn loc(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    let x: i64 = 1;\n    return loc(x, 1);\n}\n",
    )
    .expect("write fixture");
    let out = check(&src);
    assert_eq!(
        out.status.code(),
        Some(1),
        "a 2-argument call to a 1-parameter fn inside a body must fail check; stdout: {} stderr: {}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("E2005"),
        "the failure must be the arity code E2005; stdout: {}",
        stdout(&out)
    );
}

#[test]
fn correct_arity_inside_fn_body_passes() {
    // Negative control: the same program with the right argument count is clean,
    // so the test above fails for the arity and nothing else.
    let dir = common::scratch_dir("call_arity_body_ok");
    let src = dir.join("body_ok.mind");
    std::fs::write(
        &src,
        "fn loc(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    let x: i64 = 1;\n    return loc(x);\n}\n",
    )
    .expect("write fixture");
    let out = check(&src);
    assert_eq!(
        out.status.code(),
        Some(0),
        "a correct call must pass; stdout: {} stderr: {}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(feature = "cross-module-imports")]
#[test]
fn wrong_arity_to_imported_fn_is_e2005() {
    let dir = common::scratch_dir("call_arity_import");
    std::fs::write(
        dir.join("Mind.toml"),
        "[package]\nname = \"arity\"\nversion = \"0.1.0\"\n\n[build]\nentry = \"main.mind\"\nemit = \"cdylib\"\n\n[targets.cpu]\nbackend = \"cpu\"\nsources = [\"main.mind\", \"util.mind\"]\n",
    )
    .expect("write manifest");
    std::fs::write(
        dir.join("util.mind"),
        "pub fn inc(a: i64) -> i64 {\n    return a + 1;\n}\n",
    )
    .expect("write util");
    std::fs::write(
        dir.join("main.mind"),
        "import util;\n\nfn main() -> i64 {\n    return util.inc(1, 2);\n}\n",
    )
    .expect("write main");
    let out = check(&dir.join("main.mind"));
    assert_eq!(
        out.status.code(),
        Some(1),
        "a 2-argument call to an imported 1-parameter fn must fail check; stdout: {} stderr: {}",
        stdout(&out),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout(&out).contains("E2005"),
        "the failure must be the arity code E2005; stdout: {}",
        stdout(&out)
    );

    // Negative control: the right argument count passes.
    std::fs::write(
        dir.join("main.mind"),
        "import util;\n\nfn main() -> i64 {\n    return util.inc(1);\n}\n",
    )
    .expect("rewrite main");
    let ok = check(&dir.join("main.mind"));
    assert_eq!(
        ok.status.code(),
        Some(0),
        "a correct imported call must pass; stdout: {} stderr: {}",
        stdout(&ok),
        String::from_utf8_lossy(&ok.stderr)
    );
}
