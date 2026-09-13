// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Deref-assign field-first CROSS-MODULE owner-exactness gate (D4 carve-out).
//!
//! Two properties of the field-first mutable-reference subset across a
//! multi-module project:
//!
//!  * POSITIVE — cross-module struct resolution + identity replacement. The
//!    `Pair` / `H` records live in a SIBLING module; the deref-assign functions
//!    live in the entry module. `replace(&mut h.pt, b)` still lowers `&mut h.pt`
//!    to the field's real declared cell offset (resolved through the whole-project
//!    struct registry, not `idx*8`) and does identity-preserving place
//!    replacement. Same 10_001_077 probe as the single-module identity gate.
//!
//!  * NEGATIVE — owner exactness is SOUND because same-name owners cannot
//!    coexist. MIND uses a TRANSPARENT (flat) module namespace: two modules that
//!    both declare `Pair` are rejected and NO artifact is produced (same file →
//!    E2035 duplicate-struct; across files → the ambiguity fails field-layout
//!    resolution during lowering). Because the language deterministically
//!    forbids two same-name / different-owner structs in one compilation, a
//!    `*p = v` that mixes a same-named foreign owner is UNREACHABLE — the bare
//!    canonical-owner comparison in `deref_check` is sound by construction, not
//!    by luck. This fixture proves the ambiguity yields no artifact.
//!
//! MIND has no per-module struct qualifier (`mod.Struct`) — the namespace is
//! transparent — so the "qualified positive control" reduces to the unique-name
//! cross-module form exercised by the positive case above.
//!
//! Gate: `cargo test --features "std-surface mlir-build cross-module-imports"
//!                   --test deref_assign_cross_module_owner`
//! Generic types only — no private consumer source.

#![cfg(all(
    unix,
    feature = "mlir-build",
    feature = "std-surface",
    feature = "cross-module-imports"
))]

mod common;
use common::mindc_bin;

use std::process::Command;

// ── POSITIVE: structs cross-module, deref-assign fns in the entry module ──
const POS_TYPES: &str = r#"
struct Pair { x: i64, y: i64 }
struct H { pt: Pair }
"#;

const POS_COMPUTE: &str = r#"
fn replace(p: &mut Pair, new: Pair) {
    *p = new
}
fn do_replace(h: H, nb: Pair) {
    replace(&mut h.pt, nb)
}
pub fn probe() -> i64 {
    let a = Pair { x: 1, y: 2 }
    let h = H { pt: a }
    let mut b = Pair { x: 10, y: 20 }
    do_replace(h, b)
    let after = h.pt.x
    let old = a.x
    b.x = 77
    let alias = h.pt.x
    return after * 1000000 + old * 1000 + alias
}
"#;

const MANIFEST: &str = r#"[package]
name = "derefxmod"
version = "0.1.0"

[build]
entry = "src/compute.mind"
output = "derefxmod"

[targets.cpu]
backend = "cpu"

[exports]
c_abi = ["probe"]
"#;

// ── NEGATIVE: two sibling modules each declaring `Pair` (same-name owners) ──
const NEG_TYPES_A: &str = "struct Pair { x: i64, y: i64 }\nstruct H { pt: Pair }\n";
const NEG_TYPES_B: &str = "struct Pair { a: i64, b: i64 }\n";
const NEG_COMPUTE: &str = r#"
pub fn probe() -> i64 {
    let h = H { pt: Pair { x: 1, y: 2 } }
    return h.pt.x
}
"#;

fn build_project(name: &str, files: &[(&str, &str)]) -> (std::process::Output, std::path::PathBuf) {
    let root = common::scratch_dir(name).join("proj");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).expect("mkdir project");
    std::fs::write(root.join("Mind.toml"), MANIFEST).expect("write manifest");
    for (rel, body) in files {
        std::fs::write(root.join("src").join(rel), body).expect("write src");
    }
    let so = root.join("out.so");
    let out = Command::new(mindc_bin())
        .current_dir(&root)
        .args([
            "build",
            "--emit",
            "cdylib",
            "--no-cache",
            "--out",
            so.to_str().unwrap(),
        ])
        .output()
        .expect("run mindc build");
    (out, so)
}

#[test]
fn cross_module_deref_identity_runs() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "deref_assign_cross_module_owner",
            "cross-module-deref-identity: mindc not found; skipping",
        );
        return;
    }
    let (out, so) = build_project(
        "deref-xmod-positive",
        &[("types.mind", POS_TYPES), ("compute.mind", POS_COMPUTE)],
    );
    if !crate::common::gate::compiled("deref_assign_cross_module_owner", &out) {
        return;
    }
    assert!(
        so.exists(),
        "cross-module positive compiled but emitted no `.so`:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // identity => 10_001_077 (a field copy would give 10_001_010).
    let py = format!(
        "import ctypes\n\
         lib = ctypes.CDLL(r'{}')\n\
         lib.probe.restype = ctypes.c_int64\n\
         r = lib.probe()\n\
         assert r == 10001077, 'cross-module identity expected 10001077; got ' + str(r)\n\
         print('ok')\n",
        so.to_string_lossy()
    );
    let py_out = Command::new("python3")
        .args(["-c", &py])
        .output()
        .expect("python3");
    assert!(
        py_out.status.success(),
        "cross-module identity check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&py_out.stdout),
        String::from_utf8_lossy(&py_out.stderr),
    );
}

#[test]
fn cross_module_same_name_owner_yields_no_artifact() {
    let mindc = mindc_bin();
    if !mindc.exists() {
        crate::common::gate::skipped(
            "deref_assign_cross_module_owner",
            "cross-module-same-name-owner: mindc not found; skipping",
        );
        return;
    }
    // A control build with a SINGLE `Pair` owner must succeed through the exact
    // same entrypoint — so this test proves the negative is caused by the
    // duplicate owner, not by a broken project/toolchain (positive control is
    // the sibling test above; here we assert the ambiguity specifically fails).
    let (out, so) = build_project(
        "deref-xmod-negative",
        &[
            ("types_a.mind", NEG_TYPES_A),
            ("types_b.mind", NEG_TYPES_B),
            ("compute.mind", NEG_COMPUTE),
        ],
    );
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out.status.success(),
        "two same-name `Pair` owners must NOT compile (transparent namespace):\n{text}"
    );
    assert!(
        !so.exists(),
        "no artifact may be emitted for an ambiguous same-name owner:\n{text}"
    );
    // The rejection must reference the ambiguous struct owner (E2035 same-scope
    // duplicate, or the field-layout resolution failure the ambiguity causes) —
    // not an unrelated CLI/toolchain error.
    assert!(
        text.contains("Pair")
            || text.contains("E2035")
            || text.contains("duplicate")
            || text.contains("struct field"),
        "the rejection must be about the ambiguous struct owner:\n{text}"
    );
}
