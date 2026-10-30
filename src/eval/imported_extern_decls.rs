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

use super::{extern_ret_type_to_mlir_for, extern_type_to_mlir_multi_for, resolve_callconv};
use crate::ir::{IRModule, Instr};

/// Prepend import-scoped extern declarations. A local function of the same
/// name keeps the MIND signature. A local extern block later in this module
/// overwrites this decl when it is lowered.
pub(super) fn prepend_imported_extern_decls(ir: &mut IRModule) {
    let imported_externs = crate::project::active_module_table::visible_externs(
        crate::qualified_enums::current_module_path().as_deref(),
    );
    let repr_c = ir.repr_c_structs.clone();
    let mut decls = Vec::new();
    for ext in imported_externs {
        if ir.fn_signatures.contains_key(&ext.name) {
            continue;
        }
        let effective_callconv = resolve_callconv(ext.callconv);
        let param_types: Vec<String> = ext
            .param_types
            .iter()
            .flat_map(|ty| extern_type_to_mlir_multi_for(ty, &repr_c, effective_callconv))
            .collect();
        let ret_type = ext
            .ret_type
            .as_ref()
            .map(|ty| extern_ret_type_to_mlir_for(ty, &repr_c, effective_callconv));
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
