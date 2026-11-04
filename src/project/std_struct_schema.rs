// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

//! Struct schemas as lowering records them, and the bundled std structs'.
//!
//! A project build gives lowering the schema of every struct in the project
//! and the bundled std through the whole-project registry
//! (`build_global_enums`). A single translation unit has no registry, so it
//! reads a std struct's schema from `bundled_struct_schema` instead. Both
//! build each schema with `struct_schema` from the same bundled sources, so
//! the two paths cannot disagree about a std layout.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::ast::{Field, Node, TypeAnn};
use crate::eval::type_aliases::LocalTypeAliases;

/// One struct's lowering schema: field names and field types in declaration
/// order, each type resolved through the declaring module's aliases.
pub(crate) type StructSchema = (Vec<String>, Vec<TypeAnn>);

/// The schema of one `StructDef` whose module has `aliases`.
pub(crate) fn struct_schema(fields: &[Field], aliases: &LocalTypeAliases) -> StructSchema {
    (
        fields.iter().map(|f| f.name.clone()).collect(),
        fields.iter().map(|f| aliases.resolve(&f.ty)).collect(),
    )
}

/// The schema of the bundled std struct `name`, as the whole-project registry
/// records it: every std module, `module { … }` blocks transparent, the last
/// declaration of a name wins. Built once per process, like
/// `bundled_std_fn_signatures`.
pub(crate) fn bundled_struct_schema(name: &str) -> Option<&'static StructSchema> {
    static CACHE: OnceLock<BTreeMap<String, StructSchema>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut schemas = BTreeMap::new();
            for (_, module) in super::stdlib::parsed_stdlib_modules() {
                let aliases = LocalTypeAliases::new(&module.items);
                collect(&module.items, &aliases, &mut schemas);
            }
            schemas
        })
        .get(name)
}

fn collect(items: &[Node], aliases: &LocalTypeAliases, out: &mut BTreeMap<String, StructSchema>) {
    for item in items {
        match item {
            Node::StructDef { name, fields, .. } => {
                out.insert(name.clone(), struct_schema(fields, aliases));
            }
            Node::Block { stmts, .. } => collect(stmts, aliases, out),
            _ => {}
        }
    }
}

#[cfg(all(test, feature = "cross-module-imports"))]
mod tests {
    use super::*;

    /// Every bundled std struct has the same schema on both paths.
    #[test]
    fn bundled_struct_schemas_equal_the_project_registry() {
        let parsed = super::super::stdlib::parsed_stdlib_modules();
        let registry = super::super::build_global_enums(&parsed);
        assert!(
            registry.structs.contains_key("String"),
            "the bundled std must declare String"
        );
        for (name, schema) in &registry.structs {
            assert_eq!(
                bundled_struct_schema(name),
                Some(schema),
                "bundled schema of std struct {name} differs from the project registry"
            );
        }
        let names = bundled_struct_schema("String").map(|(names, _)| names.as_slice());
        assert_eq!(
            names,
            Some(["addr", "len", "cap"].map(String::from).as_slice())
        );
    }
}
