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

//! The std link set of one program: the bundled std modules a native
//! `mindc build` of it compiles from MIND source and links, and the names the
//! runtime support links into every artifact.
//!
//! `mindc test` admits std definitions through this set
//! (`src/test/std_import_fns.rs`), so a test can call exactly what the built
//! program can link. The linked modules are the build's own answer, not a
//! second walk: `substrate_link::substrate_closure` (the closure `mindc build`
//! compiles and archives) minus the modules the project supplies itself
//! (`substrate_link::drop_project_supplied`).
//! `std.vec`, `std.map` and `std.string` are never in that closure: the build
//! links the runtime-support shim for them, so only the names the shim exports
//! ([`runtime_support_exports`]) are linkable, whichever import reached them.
//!
//! A mindc built without the native linker (`mlir-build` and
//! `cross-module-imports` not both on) links no std module, so it has no link
//! set to read. There the set is the program's std import closure without the
//! runtime-support modules; a native-profile test pins it equal to the
//! substrate closure for every shipped module.

use std::collections::{BTreeSet, HashSet};

/// The std modules a program calls WITHOUT an import: the compiled build
/// resolves their `string_*` / `vec_*` / `map_*` functions through the
/// runtime-support library it links into every artifact, and the std modules
/// themselves call each other this way (std.io builds strings with
/// `string_push_byte` and no import line).
pub(crate) const AMBIENT_STD: [&str; 3] = ["std.string", "std.vec", "std.map"];

/// The runtime-support source every artifact links: the authority on which
/// std names the compiled build can call without compiling their module.
const MIND_RUNTIME_SUPPORT_C: &str = include_str!("../../runtime-support/mind_intrinsics.c");

/// The function names the runtime-support library exports (`MIND_EXPORT` /
/// `MIND_EXPORT_WEAK` definitions in [`MIND_RUNTIME_SUPPORT_C`]).
pub(crate) fn runtime_support_exports() -> &'static HashSet<&'static str> {
    static EXPORTS: std::sync::OnceLock<HashSet<&'static str>> = std::sync::OnceLock::new();
    EXPORTS.get_or_init(|| {
        MIND_RUNTIME_SUPPORT_C
            .lines()
            .filter_map(|line| {
                // `MIND_EXPORT int64_t name(...)` / `MIND_EXPORT_WEAK ... name(...)`.
                // The `#  define MIND_EXPORT ...` macro lines start with `#`
                // and never match.
                let rest = line.trim_start().strip_prefix("MIND_EXPORT")?;
                let rest = rest.strip_prefix("_WEAK").unwrap_or(rest);
                let paren = rest.find('(')?;
                rest[..paren]
                    .trim()
                    .rsplit(char::is_whitespace)
                    .next()
                    .filter(|name| !name.is_empty())
            })
            .collect()
    })
}

/// What a native build of one program links from std.
pub(crate) struct StdLinkSet {
    linked: BTreeSet<&'static str>,
}

impl StdLinkSet {
    /// The link set of a build whose sources are `program`, where the project
    /// supplies the std modules `project_std` itself.
    pub(crate) fn of_program(program: &str, project_std: Option<&BTreeSet<String>>) -> Self {
        let own = project_std.into_iter().flatten().map(String::as_str);
        Self {
            linked: linked_modules(program, own),
        }
    }

    /// True when the build compiles and links `module` from its MIND source,
    /// so every fn it defines is a linkable symbol.
    pub(crate) fn links_module(&self, module: &str) -> bool {
        self.linked.contains(module)
    }

    /// True when the build can link `name` as defined by std module `module`:
    /// the module is linked, or the runtime support exports the name.
    pub(crate) fn links(&self, module: &str, name: &str) -> bool {
        self.links_module(module) || runtime_support_exports().contains(name)
    }

    /// The linked modules, in name order.
    pub(crate) fn modules(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.linked.iter().copied()
    }
}

/// The modules the native build links for `program`: its substrate closure.
#[cfg(all(feature = "cross-module-imports", feature = "mlir-build"))]
fn linked_modules<'a>(
    program: &str,
    project_std: impl IntoIterator<Item = &'a str>,
) -> BTreeSet<&'static str> {
    use crate::project::substrate_link;
    let mut closure = substrate_link::substrate_closure(std::iter::once(program));
    substrate_link::drop_project_supplied(&mut closure, project_std);
    closure
}

/// No native linker in this mindc: [`import_closure`].
// deferred: would call substrate_link::substrate_closure, stubbed because that
// module is compiled only with the native linker — upgrade path: build
// substrate_link's closure walk in every profile that has the std bundle.
#[cfg(not(all(feature = "cross-module-imports", feature = "mlir-build")))]
fn linked_modules<'a>(
    program: &str,
    project_std: impl IntoIterator<Item = &'a str>,
) -> BTreeSet<&'static str> {
    import_closure(program, project_std)
}

/// The program's std import closure without the runtime-support modules and
/// the project-supplied ones: the link set of a mindc with no native linker.
#[cfg_attr(
    all(feature = "cross-module-imports", feature = "mlir-build"),
    allow(dead_code)
)]
fn import_closure<'a>(
    program: &str,
    project_std: impl IntoIterator<Item = &'a str>,
) -> BTreeSet<&'static str> {
    use crate::project::stdlib::{bundled_module_ast, std_import_roots, std_sources};
    let own: BTreeSet<&str> = project_std.into_iter().collect();
    let mut linked = BTreeSet::new();
    let Ok(ast) = crate::parser::parse(program) else {
        return linked;
    };
    let mut pending = std_import_roots(&ast);
    while let Some(key) = pending.pop() {
        let Some(&(name, _)) = std_sources().iter().find(|(name, _)| *name == key) else {
            continue;
        };
        if AMBIENT_STD.contains(&name) || own.contains(name) || !linked.insert(name) {
            continue;
        }
        if let Some(module) = bundled_module_ast(name) {
            pending.extend(std_import_roots(module));
        }
    }
    linked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runtime-support parse finds the shim's names and none of the
    /// ambient fns it does not define.
    #[test]
    fn runtime_support_exports_are_the_shim_names() {
        let exports = runtime_support_exports();
        for name in [
            "vec_new",
            "vec_len",
            "vec_push",
            "string_push_byte",
            "map_insert",
        ] {
            assert!(exports.contains(name), "{name} must be a shim export");
        }
        for name in ["vec_free", "vec_zeroed", "map_str_key_eq"] {
            assert!(!exports.contains(name), "{name} is not a shim export");
        }
    }

    /// The set never links an ambient module, at any depth, and never a
    /// project-supplied one; a deep substrate chain is linked whole.
    #[test]
    fn link_set_excludes_ambient_and_project_supplied_modules() {
        let set = StdLinkSet::of_program("import std.toml;\nimport std.tls13_handshake;\n", None);
        for ambient in AMBIENT_STD {
            assert!(
                !set.links_module(ambient),
                "{ambient} is linked by the shim only"
            );
        }
        for module in ["std.toml", "std.tls13_handshake", "std.hkdf", "std.sha256"] {
            assert!(set.links_module(module), "{module} must be linked");
        }
        assert!(set.links("std.vec", "vec_len") && !set.links("std.vec", "vec_free"));
        let own: BTreeSet<String> = ["std.sha256".to_string()].into_iter().collect();
        let set = StdLinkSet::of_program("import std.hkdf;\n", Some(&own));
        assert!(set.links_module("std.hkdf") && !set.links_module("std.sha256"));
    }

    /// The native link set IS the build's substrate closure for every shipped
    /// module, and the no-native-linker closure agrees with it apart from the
    /// imported module itself (std.cli and std.io are bundled non-members,
    /// which an explicit import admits anyway).
    #[cfg(all(feature = "cross-module-imports", feature = "mlir-build"))]
    #[test]
    fn link_set_is_the_substrate_closure_for_every_shipped_module() {
        use crate::project::substrate_link::{SUBSTRATE_MODULES, substrate_closure};
        for (name, _) in crate::project::stdlib::STDLIB_MIND_SOURCES {
            let program = format!("import {name};\n");
            let native = substrate_closure(std::iter::once(program.as_str()));
            let set: BTreeSet<&str> = StdLinkSet::of_program(&program, None).modules().collect();
            assert_eq!(
                set, native,
                "{name}: the link set must be the substrate closure"
            );
            assert_eq!(
                set.contains(name),
                SUBSTRATE_MODULES.contains(name),
                "{name}: linked exactly when it is a substrate member"
            );
            let mut fallback = import_closure(&program, std::iter::empty());
            let mut native = native;
            fallback.remove(name);
            native.remove(name);
            assert_eq!(fallback, native, "{name}: the fallback closure must agree");
        }
    }
}
