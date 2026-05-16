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

use libmind::eval;
use libmind::parser;

#[test]
fn lower_and_eval_add_ints() {
    let src = "1 + 2 * 3";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert_eq!(rendered, "7");
}

#[test]
fn lower_tensor_preview() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("Tensor["), "{rendered}");
}

// RFC 0002 deliverable 1 — `Node::Export` lowers into `IRModule.exports`.
#[test]
fn lower_export_block_populates_ir_exports() {
    let src = "export { foo, bar }";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    assert_eq!(ir.exports.len(), 2);
    assert!(ir.exports.contains("foo"));
    assert!(ir.exports.contains("bar"));
    // The lowering MUST NOT add an Output instruction for the export block;
    // it's metadata, not a value.
    assert!(
        !ir.instrs
            .iter()
            .any(|i| matches!(i, libmind::ir::Instr::Output(_))),
        "export block must not produce an Output instr"
    );
}

#[test]
fn lower_no_export_keeps_exports_empty() {
    let src = "1 + 2";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    assert!(
        ir.exports.is_empty(),
        "default code path must leave IRModule.exports empty"
    );
}

// RFC 0002 deliverable 3 — `Mind.toml [exports] c_abi` reaches IR via
// `CompileOptions.manifest_exports`, alongside any in-source `export {}`.
#[test]
fn compile_pipeline_merges_manifest_exports() {
    use libmind::pipeline::{compile_source, CompileOptions};
    let opts = CompileOptions {
        manifest_exports: vec!["from_manifest".to_string()],
        ..Default::default()
    };
    let products = compile_source("export { from_source }", &opts).unwrap();
    assert!(products.ir.exports.contains("from_source"));
    assert!(products.ir.exports.contains("from_manifest"));
    assert_eq!(products.ir.exports.len(), 2);
}

// v0.2.9 hardening — manifest exports list is bounded and identifier-checked.
#[test]
fn manifest_exports_reject_oversized_list() {
    use libmind::pipeline::{compile_source, CompileError, CompileOptions};
    let opts = CompileOptions {
        manifest_exports: (0..2048).map(|i| format!("name_{i}")).collect(),
        ..Default::default()
    };
    let err = compile_source("1 + 1", &opts).expect_err("oversized list must error");
    assert!(matches!(err, CompileError::InvalidManifestExport { .. }));
}

#[test]
fn manifest_exports_reject_non_identifier() {
    use libmind::pipeline::{compile_source, CompileError, CompileOptions};
    for bad in [
        "",
        "../../evil",
        "with space",
        "0starts_with_digit",
        "has\0null",
    ] {
        let opts = CompileOptions {
            manifest_exports: vec![bad.to_string()],
            ..Default::default()
        };
        let err = compile_source("1 + 1", &opts).expect_err(&format!("expected error for `{bad}`"));
        assert!(
            matches!(err, CompileError::InvalidManifestExport { .. }),
            "wrong error variant for {bad:?}: {err:?}"
        );
    }
    // Make sure a valid identifier still passes through cleanly.
    let opts = CompileOptions {
        manifest_exports: vec!["valid_name_1".to_string()],
        ..Default::default()
    };
    let products = compile_source("export { in_source }", &opts).unwrap();
    assert!(products.ir.exports.contains("valid_name_1"));
    assert!(products.ir.exports.contains("in_source"));
}

// RFC 0002 deliverable 5 — `ProfileTag` parses the three canonical names
// case-insensitively; unknown names fall back to Default. The default
// CompileOptions reports Default. The field reaches CompileOptions.
#[test]
fn profile_tag_parse_and_default() {
    use libmind::cache::ProfileTag;
    use libmind::pipeline::CompileOptions;

    assert_eq!(ProfileTag::parse("default"), ProfileTag::Default);
    assert_eq!(ProfileTag::parse("SYSTEMS"), ProfileTag::Systems);
    assert_eq!(ProfileTag::parse("Embedded"), ProfileTag::Embedded);
    assert_eq!(ProfileTag::parse("xyz"), ProfileTag::Default);

    let opts = CompileOptions::default();
    assert_eq!(opts.profile, ProfileTag::Default);

    let with_profile = CompileOptions {
        profile: ProfileTag::Systems,
        ..Default::default()
    };
    assert_eq!(with_profile.profile, ProfileTag::Systems);
}
