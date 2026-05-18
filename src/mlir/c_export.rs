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

//! RFC 0002 deliverable 2 — C-ABI export wrapper codegen.
//!
//! For every name in [`IRModule::exports`] this pass appends an
//! `llvm.func @mind_fn_<name>_invoke` symbol to the lowered MLIR module.
//! The signature is the stable `MindIO` calling convention from
//! `mind-spec` RFC-0007 / `spec/v1.0/ffi.md`:
//!
//! ```c
//! int32_t mind_fn_<name>_invoke(const MindIO *inputs,  size_t in_count,
//!                                     MindIO *outputs, size_t out_count);
//! ```
//!
//! A host runtime `dlopen`s the cdylib and `dlsym`s this symbol instead
//! of embedding and re-parsing MIND source — that embedded-parser path
//! is the largest residual drift surface in the ecosystem (RFC 0002
//! Motivation).
//!
//! Scope (mindc 0.3.0): the wrapper is a real exported C-ABI symbol with
//! the correct prototype, returning rc=0 (`MIND_OK`). The unpack →
//! dispatch-into-user-body → repack is RFC 0003 deliverable D3 (symbol
//! versioning + AOT per-fn emission); the single-`@main` lowering this
//! module sits on has no per-fn bodies to call into yet. Until then the
//! symbol is link-time stable so downstream `dlsym` wiring (mind-nerve,
//! MindLLM, rfn-mind) can be built and tested against a frozen ABI.
//!
//! Gated entirely behind `feature = "ffi-c-user"`. The default build
//! never compiles this file and emits byte-identical MLIR to before —
//! the compile-speed moat is module-level-gated, never per-statement.

use crate::ir::IRModule;

/// `mind-spec` RFC-0007: an identifier admitted into a C symbol name.
/// The pipeline already validates `Mind.toml [exports] c_abi` and
/// `export { ... }` names upstream (`pipeline.rs`), so this is a
/// defence-in-depth guard, not the primary validator.
fn is_c_symbol_safe(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Append `mind_fn_<name>_invoke` wrappers for every export to `out`.
///
/// `out` is the assembled MLIR module text *with the closing `}` not yet
/// written* — wrappers are emitted as sibling top-level symbols inside
/// the `module { ... }` block, after `@main`.
///
/// Deterministic: exports are emitted in sorted order so the MLIR text
/// (and therefore `model_hash`) is stable across runs regardless of the
/// `HashSet` iteration order.
pub fn emit_c_export_wrappers(out: &mut String, module: &IRModule) -> Result<(), String> {
    if module.exports.is_empty() {
        return Ok(());
    }

    let mut names: Vec<&String> = module.exports.iter().collect();
    names.sort();

    for name in names {
        if !is_c_symbol_safe(name) {
            return Err(format!(
                "export `{name}` is not a valid C symbol; reject before codegen \
                 (mind-spec RFC-0007 §symbol-names)"
            ));
        }
        out.push_str(&format!(
            "  llvm.func @mind_fn_{name}_invoke(%inputs: !llvm.ptr, %in_count: i64, \
             %outputs: !llvm.ptr, %out_count: i64) -> i32 {{\n"
        ));
        out.push_str("    %rc = llvm.mlir.constant(0 : i32) : i32\n");
        out.push_str("    llvm.return %rc : i32\n");
        out.push_str("  }\n");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn module_with(exports: &[&str]) -> IRModule {
        let mut m = IRModule::new();
        m.exports = exports
            .iter()
            .map(|s| s.to_string())
            .collect::<HashSet<_>>();
        m
    }

    #[test]
    fn empty_exports_emits_nothing() {
        let m = IRModule::new();
        let mut out = String::new();
        emit_c_export_wrappers(&mut out, &m).unwrap();
        assert!(out.is_empty(), "no exports must emit no wrappers");
    }

    #[test]
    fn single_export_emits_invoke_symbol() {
        let m = module_with(&["preselect_pre_tokenized"]);
        let mut out = String::new();
        emit_c_export_wrappers(&mut out, &m).unwrap();
        assert!(out.contains("llvm.func @mind_fn_preselect_pre_tokenized_invoke("));
        assert!(out.contains("%inputs: !llvm.ptr, %in_count: i64"));
        assert!(out.contains("-> i32"));
        assert!(out.contains("llvm.return %rc : i32"));
    }

    #[test]
    fn exports_emitted_in_sorted_order_for_stable_hash() {
        let m = module_with(&["zeta", "alpha", "mid"]);
        let mut out = String::new();
        emit_c_export_wrappers(&mut out, &m).unwrap();
        let a = out.find("mind_fn_alpha_invoke").unwrap();
        let mid = out.find("mind_fn_mid_invoke").unwrap();
        let z = out.find("mind_fn_zeta_invoke").unwrap();
        assert!(
            a < mid && mid < z,
            "wrappers must be sorted for deterministic MLIR"
        );
    }

    #[test]
    fn invalid_symbol_is_rejected() {
        let m = module_with(&["bad-name"]);
        let mut out = String::new();
        let err = emit_c_export_wrappers(&mut out, &m).unwrap_err();
        assert!(err.contains("not a valid C symbol"));
    }

    #[test]
    fn leading_digit_rejected() {
        assert!(!is_c_symbol_safe("9fn"));
        assert!(is_c_symbol_safe("_ok"));
        assert!(is_c_symbol_safe("fn9"));
    }
}
