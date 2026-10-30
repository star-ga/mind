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

//! Exported `extern "C"` declarations kept off the ordinary function surface.

use std::collections::BTreeMap;

use crate::ast::{CallConv, ExternFn, TypeAnn};

/// An exported `extern "C"` declaration, including the calling convention
/// the ordinary function signature does not carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedExtern {
    pub name: String,
    pub callconv: CallConv,
    pub is_varargs: bool,
    pub param_types: Vec<TypeAnn>,
    pub ret_type: Option<TypeAnn>,
}

/// Keep one record per distinct declaration. Identical declarations coalesce.
/// Conflicting records stay, so a caller cannot recover a first-wins ABI.
pub(super) fn remember(
    exported_externs: &mut Vec<ExportedExtern>,
    aliases: &crate::eval::type_aliases::LocalTypeAliases,
    callconv: CallConv,
    function: &ExternFn,
) {
    let candidate = ExportedExtern {
        name: function.name.clone(),
        callconv: crate::eval::lower::resolve_callconv(callconv),
        is_varargs: function.is_varargs,
        param_types: function
            .params
            .iter()
            .map(|param| aliases.resolve(&param.ty))
            .collect(),
        ret_type: function.ret_type.as_ref().map(|ty| aliases.resolve(ty)),
    };
    if !exported_externs.contains(&candidate) {
        exported_externs.push(candidate);
    }
}

/// One signature per name. A name with disagreeing declarations is omitted.
/// `groups` must already be in deterministic module order; within a consistent
/// name the cloned record does not depend on that order.
pub(super) fn unique_consistent<'a>(
    groups: impl IntoIterator<Item = &'a [ExportedExtern]>,
) -> Vec<ExportedExtern> {
    let mut grouped: BTreeMap<&str, Vec<&ExportedExtern>> = BTreeMap::new();
    for decls in groups {
        for ext in decls {
            grouped.entry(ext.name.as_str()).or_default().push(ext);
        }
    }
    grouped
        .into_iter()
        .filter_map(|(_, group)| {
            let first = *group.first()?;
            group.iter().all(|ext| *ext == first).then(|| first.clone())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{build_module_table, collect_module_exports};
    use crate::ast::TypeAnn;
    use crate::parser::parse;

    #[test]
    fn extern_decl_return_abi_is_part_of_the_export_surface() {
        let src = "extern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.numerics", &ast);
        assert_eq!(ex.exported, vec!["reduce_sum"]);
        assert!(ex.exported_fns.is_empty());
        assert_eq!(ex.exported_externs.len(), 1);
        assert_eq!(ex.exported_externs[0].name, "reduce_sum");
        assert_eq!(ex.exported_externs[0].param_types, vec![TypeAnn::ScalarI64]);
        assert_eq!(ex.exported_externs[0].ret_type, Some(TypeAnn::ScalarI32));
        assert!(!ex.exported_externs[0].is_varargs);
    }

    #[test]
    fn explicit_export_keeps_listed_extern_return() {
        let src = "export { reduce_sum }\nextern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n    fn hidden(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let ex = collect_module_exports("crate.numerics", &ast);
        assert_eq!(ex.exported, vec!["reduce_sum"]);
        assert!(ex.exported_fns.is_empty());
        assert_eq!(ex.exported_externs.len(), 1);
        assert_eq!(ex.exported_externs[0].ret_type, Some(TypeAnn::ScalarI32));
    }

    #[test]
    fn imported_extern_lookup_keeps_declared_return() {
        let src = "extern \"C\" {\n    fn reduce_sum(x: i64) -> i32\n}\n";
        let ast = parse(src).expect("parse");
        let table = build_module_table(&[("crate.numerics".to_string(), &ast)]);
        assert!(table.lookup_imported_fn("reduce_sum").is_none());
        let found = table
            .unique_exported_externs()
            .into_iter()
            .find(|ext| ext.name == "reduce_sum")
            .expect("extern record");
        assert_eq!(found.ret_type, Some(TypeAnn::ScalarI32));
        assert_eq!(found.param_types, vec![TypeAnn::ScalarI64]);
    }
}
