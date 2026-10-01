// Copyright 2026 STARGA Inc.
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

//! Internal linkage for module-private functions whose names collide.
//!
//! A multi-module executable compiles each source to its own object, and every
//! `fn` became a global symbol. Two modules that each define a non-`pub`
//! `fn helper` therefore failed to link ("multiple definition of `helper`"),
//! although neither `helper` is meant to be seen outside its module.
//!
//! The fix here is deliberately the narrowest one that cannot break a program
//! that links today. `pub` carries no visibility rule in the language yet
//! (`docs/type-system.md`), and `check` accepts a call from one module to
//! another module's non-`pub` fn, so marking every non-`pub` fn internal would
//! turn such calls into undefined symbols. Instead a non-`pub` fn gets internal
//! linkage only when BOTH hold:
//!
//! * another user module of the build defines a fn of the same name, so the
//!   program cannot link today anyway; and
//! * no other module refers to that name without defining it itself, so no
//!   reference can be meant for this module's copy. A module that defines the
//!   name resolves its own uses to its own definition.
//!
//! The attribute is `llvm.linkage = #llvm.linkage<internal>` on the
//! `func.func` definition; `func.func private` alone does not change the
//! linkage `convert-func-to-llvm` emits.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ast::{Literal, Module, Node};

/// The function names, per source, that get internal linkage.
pub(crate) type LinkagePlan = BTreeMap<PathBuf, BTreeSet<String>>;

/// One parsed module's top-level fn definitions and the names it refers to.
struct ModuleNames {
    /// `(name, is_candidate)`: a candidate is non-`pub`, not `main`, and
    /// carries no attribute (an attribute may export or test it).
    defs: Vec<(String, bool)>,
    defined: BTreeSet<String>,
    referenced: BTreeSet<String>,
}

fn module_names(module: &Module) -> ModuleNames {
    let mut names = ModuleNames {
        defs: Vec::new(),
        defined: BTreeSet::new(),
        referenced: BTreeSet::new(),
    };
    collect_defs(&module.items, &mut names);
    for item in &module.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| match node {
            Node::Call { callee, .. } => {
                names.referenced.insert(callee.clone());
            }
            Node::MethodCall { method, .. } => {
                names.referenced.insert(method.clone());
            }
            Node::Lit(Literal::Ident(name), _) => {
                names.referenced.insert(name.clone());
            }
            _ => {}
        });
    }
    names
}

/// Top-level fn definitions, including those inside `module NAME { … }`
/// wrappers (which parse to a transparent block).
fn collect_defs(items: &[Node], names: &mut ModuleNames) {
    for item in items {
        match item {
            Node::FnDef(fd, _) => {
                let candidate = !fd.is_pub && fd.name != "main" && fd.attrs.is_empty();
                names.defs.push((fd.name.clone(), candidate));
                names.defined.insert(fd.name.clone());
            }
            Node::Block { stmts, .. } => collect_defs(stmts, names),
            _ => {}
        }
    }
}

/// Decide which non-`pub` fns get internal linkage (see the module docs).
///
/// A source that does not parse contributes nothing and constrains nothing;
/// its own compile reports the parse error.
pub(crate) fn plan<'a>(sources: impl IntoIterator<Item = (&'a Path, &'a str)>) -> LinkagePlan {
    let sources: Vec<(&Path, &str)> = sources.into_iter().collect();
    if sources.len() < 2 {
        // One object cannot collide with itself; skip the parse.
        return LinkagePlan::new();
    }
    let modules: Vec<(PathBuf, ModuleNames)> = sources
        .into_iter()
        .filter_map(|(path, text)| {
            crate::parser::parse(text)
                .ok()
                .map(|m| (path.to_path_buf(), module_names(&m)))
        })
        .collect();
    let mut plan = LinkagePlan::new();
    for (i, (path, names)) in modules.iter().enumerate() {
        for (name, candidate) in &names.defs {
            if !candidate {
                continue;
            }
            let others = modules.iter().enumerate().filter(|(j, _)| *j != i);
            let collides = others.clone().any(|(_, (_, n))| n.defined.contains(name));
            let referenced_elsewhere = others
                .clone()
                .any(|(_, (_, n))| n.referenced.contains(name) && !n.defined.contains(name));
            if collides && !referenced_elsewhere {
                plan.entry(path.clone()).or_default().insert(name.clone());
            }
        }
    }
    plan
}

thread_local! {
    /// The internal-linkage names for the source being compiled on this thread.
    static CURRENT: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
}

/// Scope guard for [`install`]: clears the set when the source's compile ends.
pub(crate) struct LinkageGuard;

impl Drop for LinkageGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| c.borrow_mut().clear());
    }
}

/// Make `names` the internal-linkage set for the compile that follows.
pub(crate) fn install(names: Option<&BTreeSet<String>>) -> LinkageGuard {
    CURRENT.with(|c| *c.borrow_mut() = names.cloned().unwrap_or_default());
    LinkageGuard
}

/// Mark the installed names' `func.func` definitions in `mlir` internal.
/// Everything else, and every line when nothing is installed, is unchanged.
pub(crate) fn apply(mlir: &str) -> String {
    CURRENT.with(|c| {
        let names = c.borrow();
        if names.is_empty() {
            return mlir.to_string();
        }
        mark_internal(mlir, &names)
    })
}

const INTERNAL: &str = "llvm.linkage = #llvm.linkage<internal>";

fn mark_internal(mlir: &str, names: &BTreeSet<String>) -> String {
    let mut out = String::with_capacity(mlir.len() + 64 * names.len());
    for line in mlir.split_inclusive('\n') {
        let (body, eol) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        match mark_header(body, names) {
            Some(marked) => {
                out.push_str(&marked);
                out.push_str(eol);
            }
            None => out.push_str(line),
        }
    }
    out
}

/// The marked header for a definition of one of `names`, or `None` to keep the
/// line. A definition is a `func.func @name(` header ending in the `{` that
/// opens its body; an existing `attributes {…}` dictionary is extended.
fn mark_header(line: &str, names: &BTreeSet<String>) -> Option<String> {
    let rest = line.trim_start().strip_prefix("func.func @")?;
    let name = &rest[..rest.find('(')?];
    if !names.contains(name) || line.contains(INTERNAL) {
        return None;
    }
    let head = line.trim_end().strip_suffix('{')?.trim_end();
    if let Some(at) = head.find(" attributes {") {
        let (before, dict) = head.split_at(at + " attributes {".len());
        let sep = if dict.starts_with('}') { "" } else { ", " };
        return Some(format!("{before}{INTERNAL}{sep}{dict} {{"));
    }
    Some(format!("{head} attributes {{{INTERNAL}}} {{"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_of(sources: &[(&str, &str)]) -> LinkagePlan {
        plan(sources.iter().map(|(p, t)| (Path::new(*p), *t)))
    }

    fn names(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    const A: &str = "fn helper(x: i64) -> i64 {\n    return x + 1;\n}\n\npub fn fa(x: i64) -> i64 {\n    return helper(x);\n}\n";
    const B: &str = "fn helper(x: i64) -> i64 {\n    return x * 10;\n}\n\npub fn fb(x: i64) -> i64 {\n    return helper(x);\n}\n";

    #[test]
    fn colliding_private_fns_are_internal_in_both_modules() {
        let main = "import a;\nimport b;\n\nfn main() -> i64 {\n    return a.fa(1) + b.fb(2);\n}\n";
        let plan = plan_of(&[("main.mind", main), ("a.mind", A), ("b.mind", B)]);
        assert_eq!(plan.get(Path::new("a.mind")), Some(&names(&["helper"])));
        assert_eq!(plan.get(Path::new("b.mind")), Some(&names(&["helper"])));
        assert_eq!(plan.get(Path::new("main.mind")), None);
    }

    #[test]
    fn a_private_fn_called_from_another_module_stays_global() {
        // `main` calls `helper` without defining it, so it may mean either
        // module's copy: neither is made internal (the link fails as before).
        let main = "import a;\nimport b;\n\nfn main() -> i64 {\n    return a.helper(1);\n}\n";
        let plan = plan_of(&[("main.mind", main), ("a.mind", A), ("b.mind", B)]);
        assert!(plan.is_empty(), "{plan:?}");
    }

    #[test]
    fn a_private_fn_without_a_collision_is_left_alone() {
        let main = "import a;\n\nfn main() -> i64 {\n    return a.fa(1);\n}\n";
        assert!(plan_of(&[("main.mind", main), ("a.mind", A)]).is_empty());
    }

    #[test]
    fn pub_fns_and_main_are_never_internal() {
        let a =
            "pub fn helper() -> i64 {\n    return 1;\n}\n\nfn main() -> i64 {\n    return 0;\n}\n";
        let b =
            "pub fn helper() -> i64 {\n    return 2;\n}\n\nfn main() -> i64 {\n    return 0;\n}\n";
        assert!(plan_of(&[("a.mind", a), ("b.mind", b)]).is_empty());
    }

    #[test]
    fn marks_only_the_named_definitions() {
        let mlir = "module {\n  func.func private @ext(i64) -> i64\n  func.func @helper(%0: i64) -> i64 {\n    return %0 : i64\n  }\n  func.func @helper2(%0: i64) -> i64 {\n    %1 = func.call @helper(%0) : (i64) -> i64\n    return %1 : i64\n  }\n}\n";
        let got = mark_internal(mlir, &names(&["helper"]));
        let want = "module {\n  func.func private @ext(i64) -> i64\n  func.func @helper(%0: i64) -> i64 attributes {llvm.linkage = #llvm.linkage<internal>} {\n    return %0 : i64\n  }\n  func.func @helper2(%0: i64) -> i64 {\n    %1 = func.call @helper(%0) : (i64) -> i64\n    return %1 : i64\n  }\n}\n";
        assert_eq!(got, want);
        assert_eq!(mark_internal(&got, &names(&["helper"])), got, "idempotent");
    }

    #[test]
    fn merges_into_an_existing_attribute_dictionary() {
        assert_eq!(
            mark_internal(
                "  func.func @h() -> i64 attributes {llvm.emit_c_interface} {",
                &names(&["h"])
            ),
            "  func.func @h() -> i64 attributes {llvm.linkage = #llvm.linkage<internal>, llvm.emit_c_interface} {"
        );
    }

    #[test]
    fn apply_is_the_identity_until_a_set_is_installed() {
        let mlir = "  func.func @helper() -> i64 {\n";
        assert_eq!(apply(mlir), mlir);
        {
            let set = names(&["helper"]);
            let _guard = install(Some(&set));
            assert!(apply(mlir).contains("#llvm.linkage<internal>"));
        }
        assert_eq!(apply(mlir), mlir, "the guard clears the set");
    }
}
