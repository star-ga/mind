// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Pin: a decoded mic@3 module does not reach C-export wrapping.
//!
//! `imported_extern_names` is not on the wire. After `parse_mic3` it is `None`
//! (unknown), and `emit_c_export_wrappers` refuses any foreign export of such
//! a module (`mlir::c_export` tests). This source pin is the second line: the
//! entry points that decode an artifact must not lower it at all.

#[cfg(feature = "std-surface")]
use crate::ir::{IRModule, Instr};

#[test]
fn mic_decoded_module_does_not_reach_c_export() {
    let mindc = include_str!("bin/mindc.rs");
    let mic3_output = include_str!("bin/mindc/mic3_output.rs");
    assert_decoded_body_does_not_lower(
        mic3_output,
        "pub(super) fn run_inspect(",
        "parse_mic3_envelope(",
    );
    assert_decoded_body_does_not_lower(mindc, "fn run_verify(", "parse_mic3_envelope(");
    for sig in [
        "fn mindc_main(",
        "fn run_mindc_build(",
        "fn emit_mlir_if_requested(",
        "fn emit_obj_if_requested(",
        "fn emit_shared_if_requested(",
    ] {
        assert_body_does_not_decode(mindc, sig);
    }

    let conformance = include_str!("conformance.rs");
    assert_decoded_body_does_not_lower(conformance, "pub fn run_value_oracle(", "parse_mic3_body(");

    let ir_load = include_str!("ir/mic_load.rs");
    assert_decoded_body_does_not_lower(ir_load, "pub fn load(", "parse_mic3_envelope(");

    let lowering = include_str!("mlir/lowering.rs");
    assert!(
        lowering.contains("emit_c_export_wrappers("),
        "the pass must stay called from MLIR lowering"
    );
    assert!(
        !lowering.contains("parse_mic3"),
        "MLIR lowering must not decode mic@3"
    );

    for (label, src) in [
        ("project build", include_str!("project/mod.rs")),
        ("substrate link", include_str!("project/substrate_link.rs")),
        ("object cache", include_str!("build/cache.rs")),
        ("project build driver", include_str!("build/mod.rs")),
        ("compilation cache", include_str!("cache/mod.rs")),
        ("mlir object emit", include_str!("eval/mlir_build.rs")),
        ("pipeline", include_str!("pipeline.rs")),
        ("mind binary", include_str!("main.rs")),
        ("session binary", include_str!("bin/mind-ai.rs")),
    ] {
        assert!(
            !src.contains("parse_mic3"),
            "{label} decodes mic@3; it must stay on source IR"
        );
        assert!(
            !src.contains("emit_c_export_wrappers"),
            "{label} names the C-export pass"
        );
    }

    #[cfg(feature = "std-surface")]
    assert_mic_round_trip_drops_import_provenance();
}

fn each_item<'a>(src: &'a str, sig: &str) -> Vec<&'a str> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(rel) = src[from..].find(sig) {
        let at = from + rel;
        let at_line_start = at == 0 || src.as_bytes()[at - 1] == b'\n';
        if !at_line_start {
            from = at + sig.len();
            continue;
        }
        let rest = &src[at..];
        let tail = &rest[sig.len()..];
        let end = ["\nfn ", "\npub fn ", "\npub(crate) fn ", "\npub(super) fn "]
            .iter()
            .filter_map(|mark| tail.find(mark).map(|i| sig.len() + i))
            .min()
            .unwrap_or(rest.len());
        found.push(&rest[..end]);
        from = at + end;
    }
    assert!(!found.is_empty(), "missing `{sig}`");
    found
}

fn assert_decoded_body_does_not_lower(src: &str, sig: &str, decode_call: &str) {
    for body in each_item(src, sig) {
        assert!(
            body.contains(decode_call),
            "`{sig}` no longer decodes via `{decode_call}`"
        );
        for pass in [
            "emit_c_export_wrappers(",
            "lower_ir_to_mlir(",
            "lower_ir_to_mlir_with_entry(",
            "lower_to_mlir(",
            "lower_to_mlir_with_entry(",
            "lower_to_mlir_compat(",
            "compile_ir_to_mlir_text(",
        ] {
            assert!(
                !body.contains(pass),
                "`{sig}` reaches C export via `{pass}`"
            );
        }
    }
}

fn assert_body_does_not_decode(src: &str, sig: &str) {
    for body in each_item(src, sig) {
        assert!(
            !body.contains("parse_mic3"),
            "`{sig}` decodes mic@3; that IR must not be lowered"
        );
        assert!(
            !body.contains("emit_c_export_wrappers("),
            "`{sig}` names the C-export pass"
        );
    }
}

#[cfg(feature = "std-surface")]
fn assert_mic_round_trip_drops_import_provenance() {
    let mut live = IRModule::new();
    live.exports.insert("open".to_string());
    live.imported_extern_names
        .get_or_insert_default()
        .insert("open".to_string());
    live.instrs.push(Instr::ExternFnDecl {
        name: "open".to_string(),
        param_types: vec![
            "!llvm.ptr".to_string(),
            "i32".to_string(),
            "i32".to_string(),
        ],
        ret_type: Some("i32".to_string()),
        is_varargs: false,
        vararg_hints: Vec::new(),
        callconv: crate::ast::CallConv::C,
        binding: Default::default(),
    });
    let bytes = crate::ir::compact::emit_mic3(&live);
    let decoded = crate::ir::compact::parse_mic3(&bytes).expect("mic@3 round trip");
    assert!(
        decoded.imported_extern_names.is_none(),
        "decode must mark import provenance unknown, not invent it"
    );
    assert!(decoded.exports.contains("open"));
    assert!(
        decoded
            .instrs
            .iter()
            .any(|instr| { matches!(instr, Instr::ExternFnDecl { name, .. } if name == "open") })
    );
    assert_eq!(
        crate::ir::compact::emit_mic3(&decoded),
        bytes,
        "a decoded module must re-emit the same mic@3 bytes"
    );
    live.imported_extern_names = Some(Default::default());
    assert_eq!(
        crate::ir::compact::emit_mic3(&live),
        bytes,
        "import provenance must not change mic@3 bytes"
    );
}
