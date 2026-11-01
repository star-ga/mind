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

//! Imported `extern "C"` declarations as `ExternFnDecl` instructions.
//!
//! These are not MIND functions. Recording them in `fn_signatures` would
//! `func.call` them and widen a narrow C return to i64. `ExternFnDecl` is
//! what selects `llvm.call` and the C return map.
//!
//! Only an imported extern that this module calls is prepended. The owner's
//! own extern block is already in the instruction stream.

use super::{extern_ret_abi_token_for, extern_type_to_mlir_multi_for, resolve_callconv};
use crate::ir::{IRModule, Instr};
use std::collections::BTreeSet;

/// Prepend called import-scoped extern declarations. A local function of the
/// same name keeps the MIND signature. A declaration already in the stream
/// (the module's own extern block) is left as it is.
pub(super) fn prepend_imported_extern_decls(ir: &mut IRModule) {
    #[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
    prepend_called_imports(ir);
    #[cfg(not(all(feature = "std-surface", feature = "cross-module-imports")))]
    {
        let _ = ir;
    }
}

#[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
fn prepend_called_imports(ir: &mut IRModule) {
    let called = called_names(&ir.instrs);
    let already = declared_names(&ir.instrs);
    let empty = std::collections::BTreeMap::new();
    let imported_externs = crate::project::active_module_table::visible_externs(
        crate::qualified_enums::current_module_path().as_deref(),
    );
    let mut decls = Vec::new();
    for ext in imported_externs {
        if ir.fn_signatures.contains_key(&ext.name) || already.contains(&ext.name) {
            continue;
        }
        if !called.contains(&ext.name) || !signature_is_primitive(&ext) {
            continue;
        }
        let effective_callconv = resolve_callconv(ext.callconv);
        let param_types: Vec<String> = ext
            .param_types
            .iter()
            .flat_map(|ty| extern_type_to_mlir_multi_for(ty, &empty, effective_callconv))
            .collect();
        let ret_type = ext
            .ret_type
            .as_ref()
            .map(|ty| extern_ret_abi_token_for(ty, &empty, effective_callconv));
        ir.imported_extern_names.insert(ext.name.clone());
        decls.push(Instr::ExternFnDecl {
            name: ext.name,
            param_types,
            ret_type,
            is_varargs: ext.is_varargs,
            vararg_hints: Vec::new(),
            callconv: effective_callconv,
        });
    }
    for decl in decls.into_iter().rev() {
        ir.instrs.insert(0, decl);
    }
}

#[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
fn signature_is_primitive(ext: &crate::project::module_table::ExportedExtern) -> bool {
    ext.param_types
        .iter()
        .chain(ext.ret_type.as_ref())
        .all(crate::type_checker::extern_declarations::imported_type_is_primitive)
}

#[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
fn declared_names(instrs: &[Instr]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    fn walk(instrs: &[Instr], names: &mut BTreeSet<String>) {
        for instr in instrs {
            if let Instr::ExternFnDecl { name, .. } = instr {
                names.insert(name.clone());
            }
            for body in crate::ir::instr_bodies(instr) {
                walk(body, names);
            }
        }
    }
    walk(instrs, &mut names);
    names
}

#[cfg(all(feature = "std-surface", feature = "cross-module-imports"))]
fn called_names(instrs: &[Instr]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    fn walk(instrs: &[Instr], names: &mut BTreeSet<String>) {
        for instr in instrs {
            if let Instr::Call { name, .. } = instr {
                names.insert(name.clone());
            }
            for body in crate::ir::instr_bodies(instr) {
                walk(body, names);
            }
        }
    }
    walk(instrs, &mut names);
    names
}
