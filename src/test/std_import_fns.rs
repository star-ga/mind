// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Which bundled std functions `mindc test` admits, and what a std function's
//! calls resolve to.
//!
//! Admission is the build's link set ([`StdLinkSet`]): every fn of a std
//! module the build compiles and links, plus the runtime-support names of the
//! modules the shim stands for. A module the file imports by name is admitted
//! in full as well, as before.
//!
//! Resolution follows the link. Each std module's definitions are installed
//! under names owned by that module, so a call inside std code binds to the
//! module's own definition first (a colliding helper gets internal linkage in
//! the build), and a call into another std module binds to the definition the
//! link supplies. A file fn of the same name never replaces either: the file
//! reaches std only through forwarders under the bare names it may call.

#[path = "../project/std_link_set.rs"]
mod std_link_set;

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{FnDefData, Literal, Node, Span};
use crate::eval::eval_owned_symbol;
use crate::parser;

use super::top_level;
use std_link_set::{AMBIENT_STD, StdLinkSet};

/// One loaded std module: its fns and consts, and whether all its fns are
/// admitted under their bare names (else only the runtime-support ones).
struct StdModule {
    name: &'static str,
    fns: BTreeMap<String, FnDefData>,
    consts: Vec<Node>,
    callees: BTreeSet<String>,
    whole: bool,
}

/// Resolve the std definitions a test module may call: the build's link set
/// for `program` (the module's source text), plus every bundled module the
/// file imports at module scope. Returns the items to install ahead of the
/// file's own: each module's fns and consts under module-owned names, the
/// forwarders a std module's cross-module calls go through, the consts under
/// their bare names, and one forwarder per admitted bare fn name.
///
/// A `std` import that names no shipped module, or one inside a body, is
/// refused as `mindc check` refuses it; an import of a module the project
/// supplies (`project_std`) contributes nothing here. A bundled module that
/// fails to parse is a fail-loud error. A bare name two admitted modules
/// define is left out (the build would hold two global definitions), and so
/// is a name the file binds through its own `extern "C"` block, and a bare
/// `string_eq` / `string_trim`: the build links the runtime support's, which
/// the evaluator runs itself (`eval::extern_c_intrinsics`).
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(super) fn resolve_std_import_fns(
    module: &crate::ast::Module,
    program: &str,
    project_std: Option<&BTreeSet<String>>,
) -> Result<Vec<Node>, String> {
    let supplied = |path: &[String]| project_std.is_some_and(|own| own.contains(&path.join(".")));

    // A `std` import inside a body binds nothing; `mindc check` refuses it at
    // the import, and so does this runner, before any test body runs.
    if let Some((path, _)) = crate::project::stdlib::misplaced_std_imports(module).first() {
        let refusal = crate::project::stdlib::unknown_std_import(path)
            .filter(|_| !supplied(path))
            .unwrap_or_else(|| crate::project::stdlib::misplaced_std_import(path));
        return Err(format!("error[type-check][E2002]: {refusal}"));
    }
    let mut explicit: BTreeSet<String> = BTreeSet::new();
    for item in top_level::refs(&module.items) {
        if let Node::Import { path, .. } = item {
            if supplied(path) {
                continue;
            }
            if let Some(refusal) = crate::project::stdlib::unknown_std_import(path) {
                return Err(format!("error[type-check][E2002]: {refusal}"));
            }
            if path.len() == 2 && path[0] == "std" {
                explicit.insert(path.join("."));
            }
        }
    }

    let link = StdLinkSet::of_program(program, project_std);
    let mut names: BTreeSet<&'static str> = link.modules().collect();
    for (name, _) in crate::project::stdlib::std_sources() {
        if explicit.contains(*name) || AMBIENT_STD.contains(name) {
            names.insert(*name);
        }
    }
    let modules = names
        .into_iter()
        .map(|name| load_module(name, link.links_module(name) || explicit.contains(name)))
        .collect::<Result<Vec<_>, String>>()?;

    // Per bare name, the modules whose linked definition answers it.
    let mut definers: BTreeMap<&str, Vec<&StdModule>> = BTreeMap::new();
    for m in &modules {
        for name in m.fns.keys() {
            if m.whole || link.links(m.name, name) {
                definers.entry(name.as_str()).or_default().push(m);
            }
        }
    }
    let unique = |name: &str| match definers.get(name).map(Vec::as_slice) {
        Some([only]) => Some(*only),
        _ => None,
    };

    let mut out: Vec<Node> = Vec::new();
    for m in &modules {
        for (name, fd) in &m.fns {
            let mut owned = fd.clone();
            owned.name = eval_owned_symbol(m.name, name);
            out.push(Node::FnDef(Box::new(owned), Span::new(0, 0)));
        }
        for item in &m.consts {
            let mut owned = item.clone();
            if let Node::Const { name, .. } = &mut owned {
                *name = eval_owned_symbol(m.name, name);
            }
            out.push(owned);
        }
        // A call to a name the module does not define binds to the definition
        // the link supplies, through a forwarder the module's own scope finds
        // before any bare name.
        for callee in &m.callees {
            let bare = callee.rsplit("::").next().unwrap_or(callee);
            let target = if m.fns.contains_key(bare) {
                Some(m)
            } else {
                unique(bare)
            };
            if let Some(t) = target.filter(|t| t.name != m.name || callee.as_str() != bare) {
                out.push(forwarder(
                    eval_owned_symbol(m.name, callee),
                    &t.fns[bare],
                    eval_owned_symbol(t.name, bare),
                ));
            }
        }
    }
    let extern_names = file_extern_names(module);
    for m in &modules {
        out.extend(m.consts.iter().cloned());
    }
    for name in definers.keys() {
        let runtime = crate::eval::extern_c_intrinsics::is_runtime_string_fn(name);
        if let Some(m) = unique(name).filter(|_| !extern_names.contains(name) && !runtime) {
            out.push(forwarder(
                name.to_string(),
                &m.fns[*name],
                eval_owned_symbol(m.name, name),
            ));
        }
    }
    Ok(out)
}

/// Parse bundled module `name` into its fns, consts and call targets.
fn load_module(name: &'static str, whole: bool) -> Result<StdModule, String> {
    let src = crate::project::stdlib::std_source(name)
        .ok_or_else(|| format!("bundled std module '{name}' has no source"))?;
    let parsed = parser::parse(src).map_err(|errs| {
        format!(
            "failed to parse bundled std module '{name}': {}",
            errs.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    let mut module = StdModule {
        name,
        fns: BTreeMap::new(),
        consts: Vec::new(),
        callees: BTreeSet::new(),
        whole,
    };
    for item in top_level::refs(&parsed.items) {
        match item {
            Node::FnDef(fd, _) => {
                for stmt in &fd.body {
                    crate::type_checker::nerve_walk::walk(stmt, &mut |node| {
                        if let Node::Call { callee, .. } = node {
                            module.callees.insert(callee.clone());
                        }
                    });
                }
                module.fns.insert(fd.name.clone(), (**fd).clone());
            }
            Node::Const { .. } => module.consts.push(item.clone()),
            _ => {}
        }
    }
    Ok(module)
}

/// A fn `name` with `target`'s signature whose body calls `symbol` with its
/// parameters: entering `symbol` switches the evaluator to its owner's scope.
fn forwarder(name: String, target: &FnDefData, symbol: String) -> Node {
    let span = Span::new(0, 0);
    let args = target
        .params
        .iter()
        .map(|p| Node::Lit(Literal::Ident(p.name.clone()), span))
        .collect();
    let call = Node::Call {
        callee: symbol,
        args,
        span,
    };
    let body = match target.ret_type {
        Some(_) => vec![Node::Return {
            value: Some(Box::new(call)),
            span,
        }],
        None => vec![call],
    };
    Node::FnDef(
        Box::new(FnDefData {
            is_pub: target.is_pub,
            is_test: false,
            name,
            type_params: target.type_params.clone(),
            params: target.params.clone(),
            ret_type: target.ret_type.clone(),
            body,
            reap_threshold: None,
            attrs: Vec::new(),
        }),
        span,
    )
}

/// The names the file binds through its own `extern "C"` blocks: such a call
/// reaches that symbol, never a std definition of the same name.
fn file_extern_names(module: &crate::ast::Module) -> BTreeSet<&str> {
    top_level::refs(&module.items)
        .into_iter()
        .filter_map(|item| match item {
            Node::ExternBlock { fns, .. } => Some(fns.iter().map(|f| f.name.as_str())),
            _ => None,
        })
        .flatten()
        .collect()
}

#[cfg(test)]
fn bare_fn_names(items: &[Node]) -> BTreeSet<&str> {
    items
        .iter()
        .filter_map(|item| match item {
            Node::FnDef(fd, _) if !fd.name.starts_with('\0') => Some(fd.name.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
fn resolve_program(src: &str) -> Vec<Node> {
    let module = parser::parse(src).expect("parse");
    resolve_std_import_fns(&module, src, None).expect("resolve")
}

/// An explicit `import std.x509` compiles `std.sha256` from source, including
/// the helpers `sha256` calls. Those helpers are not runtime-support exports.
/// The ambient `std.vec` name `vec_free` stays refused: nothing imported it.
#[cfg(test)]
#[test]
fn explicit_import_admits_the_modules_it_imports() {
    let src = "import std.x509;\nfn t() -> i64 {\n    return 0;\n}\n";
    let module = parser::parse(src).expect("parse");
    let items = resolve_std_import_fns(&module, src, None).expect("resolve");
    let names: BTreeSet<&str> = items
        .iter()
        .filter_map(|item| match item {
            Node::FnDef(fd, _) => Some(fd.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        names.contains("sha256") && names.contains("rotr32"),
        "std.sha256 and its helpers must be admitted through import std.x509, got {names:?}"
    );
    assert!(
        !names.contains("vec_free"),
        "ambient vec_free must stay refused"
    );
}

/// For every shipped module, `import <it>;` admits exactly the build's link
/// set: every fn of each module in the substrate closure (and of the imported
/// module), plus the runtime-support names of the ambient modules. A bare
/// `string_eq` / `string_trim` call reaches the runtime support's definition,
/// which the evaluator runs itself, so no std body answers it.
#[cfg(all(test, feature = "cross-module-imports", feature = "mlir-build"))]
#[test]
fn admission_is_the_build_link_set_for_every_shipped_module() {
    use crate::project::substrate_link::substrate_closure;
    let exports = std_link_set::runtime_support_exports();
    let fns_of = |name: &str| -> Vec<String> {
        let src = crate::project::stdlib::std_source(name).expect("bundled source");
        top_level::refs(&parser::parse(src).expect("bundled parse").items)
            .into_iter()
            .filter_map(|item| match item {
                Node::FnDef(fd, _) => Some(fd.name.clone()),
                _ => None,
            })
            .collect()
    };
    for (name, _) in crate::project::stdlib::STDLIB_MIND_SOURCES {
        let src = format!("import {name};\n");
        let mut whole = substrate_closure(std::iter::once(src.as_str()));
        whole.insert(*name);
        let mut count: BTreeMap<String, usize> = BTreeMap::new();
        for module in &whole {
            for f in fns_of(module) {
                *count.entry(f).or_default() += 1;
            }
        }
        for ambient in AMBIENT_STD.iter().filter(|m| !whole.contains(*m)) {
            for f in fns_of(ambient)
                .into_iter()
                .filter(|f| exports.contains(f.as_str()))
            {
                *count.entry(f).or_default() += 1;
            }
        }
        let runtime = crate::eval::extern_c_intrinsics::is_runtime_string_fn;
        let expected: BTreeSet<&str> = count
            .iter()
            .filter(|(f, n)| **n == 1 && !runtime(f))
            .map(|(f, _)| f.as_str())
            .collect();
        let items = resolve_program(&src);
        if *name == "std.string" {
            // Positive control: the module defines the name the runtime answers.
            assert!(
                count.contains_key("string_eq"),
                "std.string defines string_eq"
            );
        }
        assert_eq!(
            bare_fn_names(&items),
            expected,
            "{name}: mindc test must admit exactly what the build links"
        );
    }
}

/// The review case on #365: ambient admission never depends on the order of
/// the imports or the depth at which an ambient module is reached.
#[cfg(test)]
#[test]
fn ambient_admission_ignores_import_order_and_depth() {
    let deep_first = resolve_program("import std.tls13_handshake;\nimport std.toml;\n");
    let deep_last = resolve_program("import std.toml;\nimport std.tls13_handshake;\n");
    let (a, b) = (bare_fn_names(&deep_first), bare_fn_names(&deep_last));
    assert_eq!(a, b, "admission must not depend on import order");
    assert!(a.contains("vec_len") && a.contains("map_insert") && a.contains("ch"));
    for unlinkable in ["vec_free", "vec_zeroed", "map_str_key_eq"] {
        assert!(!a.contains(unlinkable), "{unlinkable} is not linkable");
    }
}

/// A std module's own calls resolve inside it, and its calls into another
/// std module go through a forwarder to that module's owned definition, never
/// through a bare name a file fn can take.
#[cfg(test)]
#[test]
fn std_calls_bind_to_the_definitions_the_link_supplies() {
    let items = resolve_program("import std.hkdf;\n");
    let def = |name: &str| {
        items.iter().find_map(|item| match item {
            Node::FnDef(fd, _) if fd.name == name => Some((**fd).clone()),
            _ => None,
        })
    };
    let callee = |fd: &FnDefData| match fd.body.as_slice() {
        [
            Node::Return {
                value: Some(call), ..
            },
        ] => match call.as_ref() {
            Node::Call { callee, .. } => callee.clone(),
            other => panic!("forwarder body is not a call: {other:?}"),
        },
        other => panic!("forwarder body is not a return: {other:?}"),
    };
    let hkdf_sha256 =
        def(eval_owned_symbol("std.hkdf", "sha256").as_str()).expect("hkdf -> sha256 forwarder");
    assert_eq!(
        callee(&hkdf_sha256),
        eval_owned_symbol("std.sha256", "sha256")
    );
    let bare = def("hmac_sha256").expect("bare hmac_sha256");
    assert_eq!(callee(&bare), eval_owned_symbol("std.hkdf", "hmac_sha256"));
    let src = crate::project::stdlib::std_source("std.sha256").expect("bundled sha256");
    let own_ch = top_level::refs(&parser::parse(src).expect("sha256 parses").items)
        .into_iter()
        .find_map(|item| match item {
            Node::FnDef(fd, _) if fd.name == "ch" => Some(fd.body.clone()),
            _ => None,
        })
        .expect("std.sha256 defines ch");
    let owned_ch =
        def(eval_owned_symbol("std.sha256", "ch").as_str()).expect("sha256 keeps its own ch");
    assert_eq!(
        owned_ch.body, own_ch,
        "std.sha256's ch is its own definition"
    );
}

/// std.json_canon and std.toml each define a private `dump_value` with a
/// different signature. The bare name is not admitted (a call to it is a
/// call the link cannot bind to one definition), and each module's own calls
/// keep reaching its own `dump_value`.
#[cfg(test)]
#[test]
fn a_name_two_linked_modules_define_is_not_admitted_bare() {
    let items = resolve_program("import std.json_canon;\nimport std.toml;\n");
    let bare = bare_fn_names(&items);
    assert!(
        !bare.contains("dump_value"),
        "an ambiguous bare name must be refused"
    );
    assert!(bare.contains("jv_dump_canonical") && bare.contains("toml_max_depth"));
    for owner in ["std.json_canon", "std.toml"] {
        let own = eval_owned_symbol(owner, "dump_value");
        assert!(
            items
                .iter()
                .any(|item| matches!(item, Node::FnDef(fd, _) if fd.name == own)),
            "{owner} keeps its own dump_value"
        );
    }
}
