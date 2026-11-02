// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The std-surface names a module's resolver sees.
//!
//! Every module sees the names of the implicit std modules
//! (`IMPLICIT_STD_MODULES`). A module that imports another bundled std module
//! also sees that module's names, and those of the std modules it imports in
//! turn. The name of a module nothing imported stays unresolved, exactly as
//! before that module was bundled, and resolving it costs no parse.
//!
//! The imports are the file's: a function inside a `module` block is resolved
//! as a one-item module, so the checker installs the whole file's imported
//! names (`FileScope`) for its functions, as the loader and the module table
//! read the file's module-scope imports, `module` blocks included.

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

use super::{FxSet, fn_values::only_fns};
use crate::ast::Module;

/// The name-level facts one std module, or a union of several, declares.
#[derive(Default)]
pub(super) struct StdNames {
    /// Every top-level declaration name: the resolvable set.
    pub(super) names: FxSet,
    /// Function and extern-function names.
    fns: FxSet,
    /// Non-function declaration names plus enum variant names.
    pub(super) values: FxSet,
}

impl StdNames {
    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    fn of(module: &Module) -> Self {
        let mut out = Self::default();
        super::collect_decl_names_into(module, &mut out.names);
        super::collect_non_fn_decl_names(module, &mut out.values, &mut out.fns);
        let mut enums = super::FxMap::default();
        super::collect_enum_variants(module, &mut enums);
        out.values.extend(enums.into_values().flatten());
        out
    }

    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    fn absorb(&mut self, other: &Self) {
        self.names.extend(other.names.iter().cloned());
        self.fns.extend(other.fns.iter().cloned());
        self.values.extend(other.values.iter().cloned());
    }

    /// Names this set declares as functions and as nothing else, with `hidden`
    /// (values declared elsewhere) also excluded.
    pub(super) fn fn_only(&self, hidden: &FxSet) -> FxSet {
        only_fns(&self.fns, &self.values)
            .into_iter()
            .filter(|name| !hidden.contains(name))
            .collect()
    }

    /// For a module's resolver: the function-only names (a value an implicit
    /// module declares hides a function of the same name) and the value names.
    pub(super) fn fns_and_values(&self) -> (FxSet, FxSet) {
        (self.fn_only(&implicit().names.values), self.values.clone())
    }
}

/// The implicit modules' names and their function-only subset, built once.
pub(super) struct Implicit {
    pub(super) names: StdNames,
    pub(super) fn_only: FxSet,
}

pub(super) fn implicit() -> &'static Implicit {
    static CACHE: OnceLock<Implicit> = OnceLock::new();
    CACHE.get_or_init(|| {
        let names = implicit_union();
        Implicit {
            fn_only: names.fn_only(&FxSet::default()),
            names,
        }
    })
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn implicit_union() -> StdNames {
    let mut all = StdNames::default();
    for module in crate::project::stdlib::IMPLICIT_STD_MODULES {
        if let Some(names) = module_names(module) {
            all.absorb(names);
        }
    }
    all
}

#[cfg(not(any(feature = "cross-module-imports", feature = "std-surface")))]
fn implicit_union() -> StdNames {
    StdNames::default()
}

/// The names `module`'s imports of non-implicit std modules bring in, or
/// `None` when it imports none (nothing is parsed then).
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(super) fn imported(module: &Module) -> Option<StdNames> {
    let extras = crate::project::stdlib::imported_std_extras(module);
    if extras.is_empty() {
        return None;
    }
    let mut all = StdNames::default();
    for name in extras {
        if let Some(names) = module_names(name) {
            all.absorb(names);
        }
    }
    Some(all)
}

#[cfg(not(any(feature = "cross-module-imports", feature = "std-surface")))]
pub(super) fn imported(module: &Module) -> Option<StdNames> {
    let _ = module;
    None
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
thread_local! {
    /// The imported names of the file being checked (see `FileScope`).
    static FILE: RefCell<Option<Rc<Option<StdNames>>>> = const { RefCell::new(None) };
}

/// While alive, every function of the file it was installed for resolves
/// against the file's imported std names; the previous scope comes back on
/// drop, so a nested check of another file keeps its own.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
pub(in crate::type_checker) struct FileScope(Option<Rc<Option<StdNames>>>);

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
impl FileScope {
    pub(in crate::type_checker) fn install(file: &Module) -> Self {
        let names = Rc::new(imported(file));
        Self(FILE.with(|cell| cell.borrow_mut().replace(names)))
    }
}

#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
impl Drop for FileScope {
    fn drop(&mut self) {
        FILE.with(|cell| *cell.borrow_mut() = self.0.take());
    }
}

/// The non-implicit std names `module`'s functions see: the installed file's
/// (see `FileScope`), else those `module`'s own imports bring in.
pub(super) fn visible(module: &Module) -> Rc<Option<StdNames>> {
    #[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
    if let Some(file) = FILE.with(|cell| cell.borrow().clone()) {
        return file;
    }
    Rc::new(imported(module))
}

/// One bundled module's names, built once per process from its cached AST.
#[cfg(any(feature = "cross-module-imports", feature = "std-surface"))]
fn module_names(module: &str) -> Option<&'static StdNames> {
    use crate::project::stdlib::{STDLIB_MIND_SOURCES, bundled_module_ast};
    const COUNT: usize = STDLIB_MIND_SOURCES.len();
    static CACHE: [OnceLock<Option<StdNames>>; COUNT] = [const { OnceLock::new() }; COUNT];
    let index = STDLIB_MIND_SOURCES
        .iter()
        .position(|(name, _)| *name == module)?;
    CACHE[index]
        .get_or_init(|| bundled_module_ast(module).map(StdNames::of))
        .as_ref()
}

#[cfg(all(test, any(feature = "cross-module-imports", feature = "std-surface")))]
mod tests {
    use crate::diagnostics::Diagnostic;

    fn check(src: &str) -> Vec<Diagnostic> {
        let module = crate::parser::parse(src).expect("test program parses");
        crate::type_checker::check_module_types(&module, src, &Default::default())
    }

    /// A single-file check has no module table, so a non-implicit std
    /// module's functions resolve only through [`super::imported`]: with the
    /// import the call checks clean, without it the name stays unresolved.
    #[test]
    fn a_non_implicit_std_function_resolves_through_its_import() {
        let call = "fn f() -> i64 {\n    let out: i64 = __mind_alloc(64);\n    \
                    return keccak_sha3_512(out, 0, out);\n}\n";
        let imported = check(&format!("import std.keccak;\n\n{call}"));
        assert!(
            imported.is_empty(),
            "the call must check clean: {imported:?}"
        );
        let bare = check(call);
        assert!(
            bare.iter()
                .any(|d| d.message.contains("keccak_sha3_512") && d.code.starts_with("E200")),
            "std.keccak is not implicit, so without the import its call must be refused: {bare:?}"
        );
    }

    /// A function in a `module` block is checked as a one-item module; it
    /// still sees the block's import and the file's.
    #[test]
    fn a_function_in_a_module_block_resolves_through_the_file_imports() {
        let body = "    pub fn f() -> i64 {\n        let out: i64 = __mind_alloc(64);\n        \
                    return keccak_sha3_512(out, 0, out);\n    }\n}\n";
        for src in [
            format!("module inner {{\n    import std.keccak;\n\n{body}"),
            format!("import std.keccak;\n\nmodule inner {{\n{body}"),
        ] {
            let diags = check(&src);
            assert!(diags.is_empty(), "must check clean:\n{src}\n{diags:?}");
        }
    }
}
