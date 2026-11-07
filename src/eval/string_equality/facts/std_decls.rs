// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The declarations a bundled std body is written against.
//!
//! `mindc test` evaluates one module: the test file with the bodies of the
//! std functions it reaches merged in, but not their module's `struct` and
//! `extern "C"` items. The build compiles each std module with those items in
//! scope, so its string pass reads `v.addr` in std.vec as the declared `i64`
//! field and `read(...)` in std.fs as the libc call's `i64` result. Without
//! them the interpreted pass read both as of unknown type, an unknown value
//! stored into a declared `i64` slot made that slot maybe a string, and
//! integer comparisons the build compiles were refused with E2038.
//!
//! So the interpreted analysis also reads the `struct` and `extern "C"` items
//! of every bundled std module one of whose function bodies the module holds
//! unchanged, after the module's own declarations (see `Decls::new`).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crate::ast::{Module, Node};

use super::{Decls, Ret, merge};

/// One bundled std module: its functions by name, and its `struct` and
/// `extern "C"` items.
struct StdModule {
    fns: BTreeMap<String, Node>,
    decls: Vec<Node>,
}

/// Every item of `items`, through `module name { ... }` blocks.
fn flatten(items: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    let mut pending: Vec<&Node> = items.iter().rev().collect();
    while let Some(item) = pending.pop() {
        match item {
            Node::Block { stmts, .. } => pending.extend(stmts.iter().rev()),
            other => out.push(other),
        }
    }
    out
}

/// The bundled std modules as this process sees them (the override
/// directory included, see `project::stdlib::std_sources`), parsed once.
fn bundled() -> &'static [StdModule] {
    static MODULES: OnceLock<Vec<StdModule>> = OnceLock::new();
    MODULES.get_or_init(|| {
        crate::project::stdlib::parsed_stdlib_modules()
            .iter()
            .map(|(_, module)| {
                let mut fns = BTreeMap::new();
                let mut decls = Vec::new();
                for item in flatten(&module.items) {
                    match item {
                        Node::FnDef(fd, _) => {
                            fns.insert(fd.name.clone(), item.clone());
                        }
                        Node::StructDef { .. } | Node::ExternBlock { .. } => {
                            decls.push(item.clone())
                        }
                        _ => {}
                    }
                }
                StdModule { fns, decls }
            })
            .collect()
    })
}

/// The `struct` and `extern "C"` items of every bundled std module that
/// defines one of `module`'s functions with the same body.
pub(super) fn declarations(module: &Module) -> Vec<&'static Node> {
    let held: Vec<&Node> = flatten(&module.items)
        .into_iter()
        .filter(|item| matches!(item, Node::FnDef(..)))
        .collect();
    if held.is_empty() {
        return Vec::new();
    }
    bundled()
        .iter()
        .filter(|std| {
            held.iter().any(|item| match item {
                Node::FnDef(fd, _) => std.fns.get(&fd.name) == Some(*item),
                _ => false,
            })
        })
        .flat_map(|std| std.decls.iter())
        .collect()
}

impl Decls {
    /// The `struct` and `extern "C"` items of `std` (see `std_decls`) for
    /// names this module leaves undeclared; two std modules that disagree
    /// leave the name of unknown type.
    pub(super) fn seed_std(&mut self, std: &[&Node]) {
        let (mut structs, mut returns) = (BTreeMap::new(), BTreeMap::new());
        for item in std {
            match item {
                Node::StructDef { name, fields, .. } => {
                    let fields = fields.iter().map(|f| (f.name.clone(), f.ty.clone()));
                    merge(&mut structs, name, Some(fields.collect()));
                }
                Node::ExternBlock { fns, .. } => {
                    for f in fns {
                        merge(
                            &mut returns,
                            &f.name,
                            Some(Ret::of(f.ret_type.as_ref(), &[])),
                        );
                        let types = f.params.iter().map(|p| p.ty.clone()).collect();
                        self.opaque_params.entry(f.name.clone()).or_insert(types);
                    }
                }
                _ => {}
            }
        }
        for (name, fields) in structs {
            self.structs.entry(name).or_insert(fields);
        }
        for (name, ret) in returns {
            self.returns.entry(name).or_insert(ret);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::declarations;
    use crate::ast::Node;

    fn names(src: &str) -> Vec<String> {
        let module = crate::parser::parse(src).expect("parse");
        declarations(&module)
            .into_iter()
            .flat_map(|item| match item {
                Node::StructDef { name, .. } => vec![name.clone()],
                Node::ExternBlock { fns, .. } => fns.iter().map(|f| f.name.clone()).collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// A std.vec body merged unchanged brings `struct Vec`; a std.fs body
    /// brings the libc externs it calls.
    #[test]
    fn merged_std_bodies_bring_their_modules_structs_and_externs() {
        let vec_src = crate::project::stdlib::std_source("std.vec").expect("std.vec");
        assert!(names(vec_src).iter().any(|n| n == "Vec"), "struct Vec");
        let fs_src = crate::project::stdlib::std_source("std.fs").expect("std.fs");
        let fs = names(fs_src);
        assert!(fs.iter().any(|n| n == "read") && fs.iter().any(|n| n == "lseek"));
    }

    /// A user function that only shares a std function's name brings
    /// nothing: the body must be the std one.
    #[test]
    fn a_function_named_like_a_std_one_brings_nothing() {
        assert!(names("fn vec_push(v: i64, x: i64) -> i64 {\n    return v + x\n}\n").is_empty());
        assert!(names("fn score() -> i64 {\n    return 1\n}\n").is_empty());
    }
}
