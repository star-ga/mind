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
//! A build that links std substrate modules ([`plan_with_std`]) extends the
//! rule in the direction that keeps the program's own symbols as they were:
//!
//! * a std module's colliding non-`pub` helper yields: it is the one made
//!   internal, and the program's fn keeps its global symbol;
//! * a program fn is made internal on std's account only for a std member the
//!   link actually PULLS from the archive (see [`pulled_std`]): when that
//!   member defines the same name as a global symbol (a `pub` fn), or calls
//!   the name without defining it (std.toml's `vec_push`, meant for the
//!   runtime), so the program fn can neither collide with nor stand in for the
//!   definition std was built against. A std module that is imported but never
//!   pulled constrains nothing.
//!
//! A name a native (C) object of the build calls is called from outside the
//! `.mind` sources, so it is never made internal. A shared library also keeps
//! its exported surface global: the manifest's `[exports] c_abi` names and each
//! module's own `export { … }` list. An executable has no such surface, so
//! there those lists change nothing.
//!
//! The attribute is `llvm.linkage = #llvm.linkage<internal>` on the
//! `func.func` definition; `func.func private` alone does not change the
//! linkage `convert-func-to-llvm` emits.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ast::{Module, Node};

/// The function names, per source, that get internal linkage.
pub(crate) type LinkagePlan = BTreeMap<PathBuf, BTreeSet<String>>;

/// What a build's link exposes, and what its native objects call, beyond its
/// `.mind` sources.
pub(crate) struct LinkSurface<'a> {
    /// `Some(c_abi)` for a shared library, whose exported surface (these names
    /// and each module's `export { … }` list) stays global; `None` for an
    /// executable.
    pub shared_exports: Option<&'a BTreeSet<String>>,
    /// The undefined symbols of the build's native objects. `None` when they
    /// could not be read: then every program fn stays global and every std
    /// module counts as pulled, which can only leave a collision to fail the
    /// link loudly, never bind a call to the wrong definition.
    pub native_refs: Option<&'a BTreeSet<String>>,
}

static NO_NAMES: BTreeSet<String> = BTreeSet::new();

impl<'a> LinkSurface<'a> {
    /// An executable whose native objects call `native_refs`.
    pub(crate) fn executable(native_refs: Option<&'a BTreeSet<String>>) -> Self {
        Self {
            shared_exports: None,
            native_refs,
        }
    }

    /// A shared library exporting `c_abi` (and every `export { … }` name); it
    /// links no native objects.
    pub(crate) fn shared_library(c_abi: &'a BTreeSet<String>) -> Self {
        Self {
            shared_exports: Some(c_abi),
            native_refs: Some(&NO_NAMES),
        }
    }
}

/// One parsed module's top-level fn definitions and the names it calls.
struct ModuleNames {
    /// `(name, is_candidate)`: a candidate is non-`pub`, not `main`, and carries
    /// no attribute but `#[test]` (another attribute may export it; a test fn is
    /// run by `mindc test`'s evaluator, never referenced by symbol).
    defs: Vec<(String, bool)>,
    defined: BTreeSet<String>,
    /// Call and method-call targets. Plain identifiers are not counted: there
    /// are no first-class functions, so a bare name in value position is a
    /// local or a constant, and a std module's local `n` says nothing about a
    /// program fn named `n`.
    referenced: BTreeSet<String>,
    /// Names in the module's own `export { … }` list.
    exported: BTreeSet<String>,
}

fn module_names(module: &Module) -> ModuleNames {
    let mut names = ModuleNames {
        defs: Vec::new(),
        defined: BTreeSet::new(),
        referenced: BTreeSet::new(),
        exported: BTreeSet::new(),
    };
    collect_defs(&module.items, &mut names);
    for item in &module.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| match node {
            Node::Call { callee, .. } => {
                names.referenced.insert(callee.clone());
                // A path call (`m::f`) keeps its qualifier in the callee; count
                // the bare name too, so a reference is never missed.
                if let Some(last) = callee.rsplit([':', '.']).next() {
                    if last.len() < callee.len() {
                        names.referenced.insert(last.to_string());
                    }
                }
            }
            Node::MethodCall { method, .. } => {
                names.referenced.insert(method.clone());
            }
            Node::Export { names: list, .. } => names.exported.extend(list.iter().cloned()),
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
                let candidate = !fd.is_pub
                    && fd.name != "main"
                    && fd.attrs.iter().all(|attr| attr.name == "test");
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
/// its own compile reports the parse error. A build that links std substrate
/// modules uses [`plan_with_std`] instead.
#[cfg_attr(
    all(feature = "cross-module-imports", feature = "mlir-build"),
    allow(dead_code)
)]
pub(crate) fn plan<'a>(
    sources: impl IntoIterator<Item = (&'a Path, &'a str)>,
    link: &LinkSurface<'_>,
) -> LinkagePlan {
    plan_with_std(sources, std::iter::empty(), link)
}

/// One module of a build: its plan key, names, and whether it is a std
/// substrate module (keyed by module name) or one of the build's own sources.
struct Unit {
    key: PathBuf,
    names: ModuleNames,
    is_std: bool,
}

/// Per name, the modules defining it and the modules calling it without
/// defining it, so every rule below is a lookup rather than a scan.
#[derive(Default)]
struct NameIndex<'u> {
    definers: BTreeMap<&'u str, Vec<usize>>,
    foreign: BTreeMap<&'u str, Vec<usize>>,
}

impl<'u> NameIndex<'u> {
    fn new(units: &'u [Unit]) -> Self {
        let mut index = Self::default();
        for (i, unit) in units.iter().enumerate() {
            for name in &unit.names.defined {
                index.definers.entry(name).or_default().push(i);
            }
            for name in unit.names.referenced.difference(&unit.names.defined) {
                index.foreign.entry(name).or_default().push(i);
            }
        }
        index
    }

    fn definers(&self, name: &str) -> &[usize] {
        self.definers.get(name).map_or(&[], Vec::as_slice)
    }

    fn foreign(&self, name: &str) -> &[usize] {
        self.foreign.get(name).map_or(&[], Vec::as_slice)
    }
}

/// [`plan`] for a build that also links the std modules `std_sources` (keyed by
/// module name), under the rules in the module docs.
pub(crate) fn plan_with_std<'a>(
    sources: impl IntoIterator<Item = (&'a Path, &'a str)>,
    std_sources: impl IntoIterator<Item = (&'a Path, &'a str)>,
    link: &LinkSurface<'_>,
) -> LinkagePlan {
    let tagged: Vec<(&Path, &str, bool)> = sources
        .into_iter()
        .map(|(p, t)| (p, t, false))
        .chain(std_sources.into_iter().map(|(p, t)| (p, t, true)))
        .collect();
    if tagged.len() < 2 {
        // One object cannot collide with itself; skip the parse.
        return LinkagePlan::new();
    }
    let units: Vec<Unit> = tagged
        .into_iter()
        .filter_map(|(path, text, is_std)| {
            crate::parser::parse(text).ok().map(|m| Unit {
                key: path.to_path_buf(),
                names: module_names(&m),
                is_std,
            })
        })
        .collect();
    let index = NameIndex::new(&units);
    let mut plan = LinkagePlan::new();

    // A std module's colliding non-`pub` helper yields, unless another module
    // calls that name without defining it (the call may mean this copy).
    for (i, unit) in units.iter().enumerate().filter(|(_, u)| u.is_std) {
        for (name, candidate) in &unit.names.defs {
            if !candidate || unit.names.exported.contains(name) {
                continue;
            }
            let collides = index.definers(name).iter().any(|&j| j != i);
            if collides && index.foreign(name).is_empty() {
                plan.entry(unit.key.clone())
                    .or_default()
                    .insert(name.clone());
            }
        }
    }

    let pulled = pulled_std(&units, &index, &plan, link.native_refs);
    let std_global = |j: usize, name: &str| {
        pulled[j]
            && !plan
                .get(&units[j].key)
                .is_some_and(|set| set.contains(name))
    };
    let mut program_plan = LinkagePlan::new();
    for (i, unit) in units.iter().enumerate().filter(|(_, u)| !u.is_std) {
        for (name, candidate) in &unit.names.defs {
            let exported = link
                .shared_exports
                .is_some_and(|c_abi| c_abi.contains(name) || unit.names.exported.contains(name));
            let native_call = link.native_refs.is_none_or(|refs| refs.contains(name));
            if !candidate || exported || native_call {
                continue;
            }
            let foreign = index.foreign(name);
            if foreign.iter().any(|&j| !units[j].is_std) {
                // Another program module calls it: that call may mean this one.
                continue;
            }
            let definers = index.definers(name);
            let program_collides = definers.iter().any(|&j| j != i && !units[j].is_std);
            let std_defines = definers
                .iter()
                .any(|&j| units[j].is_std && std_global(j, name));
            let std_calls = foreign.iter().any(|&j| units[j].is_std && pulled[j]);
            if program_collides || std_defines || std_calls {
                program_plan
                    .entry(unit.key.clone())
                    .or_default()
                    .insert(name.clone());
            }
        }
    }
    plan.append(&mut program_plan);
    plan
}

/// Which std modules the link pulls from the substrate archive, by unit index.
///
/// The archive is linked after every object, so the linker extracts a member
/// exactly when it defines, as a global symbol, a name still undefined at that
/// point, and repeats until nothing new is extracted. This computes the same
/// least fixed point over the names: the seed is every name a program module
/// or native object calls that no program module defines (a program
/// definition satisfies it first), and an extracted member adds every name it
/// calls without defining. That last step counts a call the program may
/// satisfy too, so the set can only over-approximate the linker's, which errs
/// toward making a program fn internal, never toward a std call binding to it.
/// A worklist over the name index keeps it linear in the names.
fn pulled_std(
    units: &[Unit],
    index: &NameIndex<'_>,
    plan: &LinkagePlan,
    native_refs: Option<&BTreeSet<String>>,
) -> Vec<bool> {
    let Some(native_refs) = native_refs else {
        return units.iter().map(|u| u.is_std).collect();
    };
    let program_defines = |name: &str| index.definers(name).iter().any(|&j| !units[j].is_std);
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut queue: Vec<&str> = Vec::new();
    let program_calls = units
        .iter()
        .filter(|u| !u.is_std)
        .flat_map(|u| u.names.referenced.iter())
        .chain(native_refs.iter());
    for name in program_calls {
        if !program_defines(name) && seen.insert(name) {
            queue.push(name);
        }
    }
    let mut pulled = vec![false; units.len()];
    while let Some(name) = queue.pop() {
        for &j in index.definers(name) {
            let internal = plan
                .get(&units[j].key)
                .is_some_and(|set| set.contains(name));
            if !units[j].is_std || pulled[j] || internal {
                continue;
            }
            pulled[j] = true;
            let names = &units[j].names;
            for called in names.referenced.difference(&names.defined) {
                if seen.insert(called) {
                    queue.push(called);
                }
            }
        }
    }
    pulled
}

/// The undefined symbols of the native objects `objs` (`nm -P -u`), or `None`
/// when a listing fails. No objects: the empty set.
#[cfg(feature = "mlir-build")]
pub(crate) fn native_undefined(objs: &[PathBuf]) -> Option<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    if objs.is_empty() {
        return Some(names);
    }
    let out = std::process::Command::new("nm")
        .args(["-P", "-u"])
        .args(objs)
        .output()
        .ok()
        .filter(|out| out.status.success())?;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        // `name U` per symbol; a multi-object listing adds `file.o:` headers.
        let mut fields = line.split_whitespace();
        if let (Some(name), Some("U")) = (fields.next(), fields.next()) {
            names.insert(name.to_string());
        }
    }
    Some(names)
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
        plan(
            sources.iter().map(|(p, t)| (Path::new(*p), *t)),
            &LinkSurface::executable(Some(&NO_NAMES)),
        )
    }

    /// The plan for the program `program` linking the std modules `std`.
    fn std_plan(program: &[(&str, &str)], std: &[(&str, &str)], link: &LinkSurface) -> LinkagePlan {
        plan_with_std(
            program.iter().map(|(p, t)| (Path::new(*p), *t)),
            std.iter().map(|(p, t)| (Path::new(*p), *t)),
            link,
        )
    }

    fn exe() -> LinkSurface<'static> {
        LinkSurface::executable(Some(&NO_NAMES))
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
    fn colliding_test_fns_are_internal() {
        let a = "#[test]\nfn t_one() {\n    assert 1 == 1;\n}\n";
        let b = "#[test]\nfn t_one() {\n    assert 2 == 2;\n}\n";
        let plan = plan_of(&[("a.mind", a), ("b.mind", b)]);
        assert_eq!(plan.get(Path::new("a.mind")), Some(&names(&["t_one"])));
        assert_eq!(plan.get(Path::new("b.mind")), Some(&names(&["t_one"])));
    }

    #[test]
    fn pub_fns_and_main_are_never_internal() {
        let a =
            "pub fn helper() -> i64 {\n    return 1;\n}\n\nfn main() -> i64 {\n    return 0;\n}\n";
        let b =
            "pub fn helper() -> i64 {\n    return 2;\n}\n\nfn main() -> i64 {\n    return 0;\n}\n";
        assert!(plan_of(&[("a.mind", a), ("b.mind", b)]).is_empty());
    }

    // std.toml-like: a `pub` entry point and a private helper.
    const TOML: &str = "fn is_digit(c: i64) -> i64 {\n    return c;\n}\n\npub fn toml_new() -> i64 {\n    return vec_push(0, is_digit(1));\n}\n";
    // std.json-like: a `pub fn push`.
    const JSON: &str = "pub fn push(a: i64) -> i64 {\n    return a;\n}\n\npub fn json_new() -> i64 {\n    return push(0);\n}\n";

    #[test]
    fn a_program_helper_a_pulled_std_module_calls_is_internal() {
        // std code calling `vec_push` means the runtime's, never the program's.
        let main = "fn vec_push(a: i64, b: i64) -> i64 {\n    return 99;\n}\n\nfn main() -> i64 {\n    return toml.toml_new() + vec_push(1, 2);\n}\n";
        let plan = std_plan(&[("main.mind", main)], &[("std.toml", TOML)], &exe());
        assert_eq!(
            plan.get(Path::new("main.mind")),
            Some(&names(&["vec_push"]))
        );
        // A program fn another PROGRAM module calls stays global.
        let user =
            "import a;\n\nfn main() -> i64 {\n    return a.vec_push(1, 2) + toml.toml_new();\n}\n";
        let a = "fn vec_push(a: i64, b: i64) -> i64 {\n    return 99;\n}\n";
        let plan = std_plan(
            &[("main.mind", user), ("a.mind", a)],
            &[("std.toml", TOML)],
            &exe(),
        );
        assert!(!plan.contains_key(Path::new("a.mind")), "{plan:?}");
    }

    #[test]
    fn a_std_module_nothing_pulls_constrains_nothing() {
        // Imported, never called: the archive member is not linked, so neither
        // its calls nor its `pub` names concern the program's fns.
        let main = "fn vec_push(a: i64, b: i64) -> i64 {\n    return 99;\n}\n\nfn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return vec_push(1, 2) + push(3);\n}\n";
        let plan = std_plan(
            &[("main.mind", main)],
            &[("std.toml", TOML), ("std.json", JSON)],
            &exe(),
        );
        assert!(!plan.contains_key(Path::new("main.mind")), "{plan:?}");
        // Once the program calls into std.json, its `pub fn push` is linked and
        // the program's own `push` must yield.
        let main = "fn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return json.json_new() + push(3);\n}\n";
        let plan = std_plan(&[("main.mind", main)], &[("std.json", JSON)], &exe());
        assert_eq!(plan.get(Path::new("main.mind")), Some(&names(&["push"])));
    }

    #[test]
    fn a_member_is_pulled_through_another_member() {
        // main -> http_get (std.http) -> push (std.json): std.json is linked.
        let http = "pub fn http_get() -> i64 {\n    return json.push(1);\n}\n";
        let main = "fn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return http.http_get() + push(3);\n}\n";
        let plan = std_plan(
            &[("main.mind", main)],
            &[("std.http", http), ("std.json", JSON)],
            &exe(),
        );
        assert_eq!(plan.get(Path::new("main.mind")), Some(&names(&["push"])));
    }

    #[test]
    fn a_path_call_counts_its_last_segment() {
        let main = "fn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return json::json_new() + push(3);\n}\n";
        let plan = std_plan(&[("main.mind", main)], &[("std.json", JSON)], &exe());
        assert_eq!(plan.get(Path::new("main.mind")), Some(&names(&["push"])));
    }

    #[test]
    fn a_std_helper_yields_to_a_program_fn() {
        // Pulled or not, std.toml's private `is_digit` is the one made internal.
        for main in [
            "fn is_digit(c: i64) -> i64 {\n    return c + 1000;\n}\n\nfn main() -> i64 {\n    return is_digit(1);\n}\n",
            "fn is_digit(c: i64) -> i64 {\n    return c + 1000;\n}\n\nfn main() -> i64 {\n    return is_digit(1) + toml.toml_new();\n}\n",
        ] {
            let plan = std_plan(&[("main.mind", main)], &[("std.toml", TOML)], &exe());
            assert!(!plan.contains_key(Path::new("main.mind")), "{plan:?}");
            assert_eq!(plan.get(Path::new("std.toml")), Some(&names(&["is_digit"])));
        }
    }

    #[test]
    fn a_name_a_native_object_calls_stays_global() {
        let main = "fn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return json.json_new() + push(3);\n}\n";
        let c_calls = names(&["push"]);
        let plan = std_plan(
            &[("main.mind", main)],
            &[("std.json", JSON)],
            &LinkSurface::executable(Some(&c_calls)),
        );
        assert!(!plan.contains_key(Path::new("main.mind")), "{plan:?}");
        // A C call into std pulls the member: here std.json, whose `pub fn
        // push` the program's private `push` must then yield to.
        let main = "fn push(a: i64) -> i64 {\n    return a;\n}\n\nfn main() -> i64 {\n    return push(3);\n}\n";
        let c_calls = names(&["json_new"]);
        let plan = std_plan(
            &[("main.mind", main)],
            &[("std.json", JSON)],
            &LinkSurface::executable(Some(&c_calls)),
        );
        assert_eq!(plan.get(Path::new("main.mind")), Some(&names(&["push"])));
        // Unreadable native symbols: no program fn is made internal.
        let plan = std_plan(
            &[("main.mind", main), ("a.mind", A), ("b.mind", B)],
            &[("std.json", JSON)],
            &LinkSurface::executable(None),
        );
        assert!(
            ["main.mind", "a.mind", "b.mind"]
                .iter()
                .all(|k| !plan.contains_key(Path::new(k))),
            "{plan:?}"
        );
    }

    #[test]
    fn a_shared_library_keeps_its_exported_names_global() {
        // Listed in the module's own `export { }` or in the manifest's C-ABI
        // exports: never internal, even when a pulled std member defines it.
        let listed = "export { push }\n\nfn push(a: i64) -> i64 {\n    return a;\n}\n\npub fn go() -> i64 {\n    return json.json_new() + push(1);\n}\n";
        let plan = std_plan(
            &[("main.mind", listed)],
            &[("std.json", JSON)],
            &LinkSurface::shared_library(&NO_NAMES),
        );
        assert!(!plan.contains_key(Path::new("main.mind")), "{plan:?}");
        let unlisted = listed.replace("export { push }\n\n", "");
        let c_abi = names(&["push"]);
        let plan = std_plan(
            &[("main.mind", &unlisted)],
            &[("std.json", JSON)],
            &LinkSurface::shared_library(&c_abi),
        );
        assert!(!plan.contains_key(Path::new("main.mind")), "{plan:?}");
    }

    #[test]
    fn an_executable_ignores_export_lists() {
        // Three private `helper`s, one listed in `export { }`: an executable
        // has no exported surface (the build never hands it `[exports]`), so
        // all three are internal and the program links.
        let main = "fn helper(x: i64) -> i64 {\n    return x;\n}\n\nfn main() -> i64 {\n    return helper(1) + a.fa(1) + b.fb(2);\n}\n";
        let a = format!("export {{ helper, fa }}\n\n{A}");
        let plan = plan_of(&[("main.mind", main), ("a.mind", &a), ("b.mind", B)]);
        for key in ["main.mind", "a.mind", "b.mind"] {
            assert_eq!(plan.get(Path::new(key)), Some(&names(&["helper"])), "{key}");
        }
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
