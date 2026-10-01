// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0

//! Native cross-module substrate linking for shared-library emission.
//!
//! A `.mind` source that `import`s a self-contained `std` substrate module
//! (see [`SUBSTRATE_MODULES`]) calls that module's functions by their bare
//! symbol name. Emitting the consumer alone produces a shared object whose
//! imported symbols are UNDEFINED, so `dlopen` fails at load time:
//!
//! ```text
//! libio_canon.so: undefined symbol: sha256
//! ```
//!
//! The fix is to compile each imported substrate module to an object, pack the
//! objects into one static archive (see [`SubstrateArchive`]) and link it after
//! every other object. This module owns that walk and the list of modules, so
//! every emitter shares one implementation:
//!
//!  * `mindc build` for executables (the project builder, [`crate::project`]),
//!  * `mindc build --emit=cdylib` (manifest path, [`crate::project`]), and
//!  * `mindc <file> --emit-shared <out.so>` (flat single-file path, `mindc.rs`).
//!
//! Linking std code next to a program's must not change what the program's
//! names mean: std compiles against a std-only module table
//! ([`std_only_scope`]), a std module's colliding non-`pub` helper gets
//! internal linkage, and a program fn does only for a std member the link
//! actually pulls ([`linkage_plan`], `private_linkage`).
//!
//! The flat path previously had no walk at all, which is why compiling
//! `std/io_canon.mind` — the canonical-I/O module the evidence anchor hashes
//! through — produced an un-`dlopen`-able artifact. The defect was in the
//! LINKER wiring, never in `std/sha256.mind`, which is a complete pure-MIND
//! FIPS-180-4 implementation.
//!
//! Determinism: the closure is a [`BTreeSet`] of `&'static str` module names,
//! so archive member order is the module-name sort order, never a hash order,
//! and the archive is written with deterministic member headers. An EMPTY
//! closure yields no archive at all, which keeps the link byte-identical to the
//! historical single-entry path (the Phase G keystone imports no substrate
//! module).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use super::private_linkage::LinkSurface;
use crate::eval::mlir_build::{self, BuildTools};
use crate::runtime::types::BackendTarget;

/// Substrate modules whose `.o` is compiled and linked when imported.
///
/// Membership rule: the module must be SELF-CONTAINED at the object level —
/// it may import other members of this list (the closure walk below follows
/// those edges) and `std.vec` / `std.map` / `std.string`, whose symbols the
/// bundled runtime-support shim defines, but nothing else, so its object never
/// drags in an unresolved dependency. Every member must compile natively on
/// its own; `tests/std_substrate_native_link.rs` compiles each one.
///
/// Bundled modules that are deliberately NOT members:
/// * `std.vec`, `std.map`, `std.string`: the runtime-support shim already
///   provides their symbols; linking the MIND objects too would replace the
///   shim's implementations in every program that imports them.
/// * `std.cli`: calls `__mind_argc` / `__mind_argv`, which the Rust/MLIR path
///   does not register yet, so its object cannot be compiled.
/// * `std.io`: defines `print_bytes`, which the shim also defines as a strong
///   symbol, so the two objects collide at link.
///
/// `std.time` is a member but cannot compile until `__mind_now_ns` is
/// registered for the Rust/MLIR path; a program importing it fails the build
/// with that diagnostic, as it already did before this list grew.
pub const SUBSTRATE_MODULES: &[&str] = &[
    "std.arena",
    "std.async",
    "std.blas",
    "std.fs",
    "std.http",
    "std.io_canon",
    "std.iouring",
    "std.json",
    "std.net",
    "std.process",
    "std.reactor",
    "std.regex",
    "std.ring",
    "std.sha256",
    "std.sha512",
    "std.time",
    "std.toml",
    "std.tui",
];

/// Substrate modules directly imported by `src`.
///
/// Keyed on the RESOLVED import path (`path.join(".")`), never on a bare
/// leading segment: only an import naming a member of [`SUBSTRATE_MODULES`]
/// admits that module's object. An unparseable source contributes nothing.
pub fn scan_substrate_imports(src: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    let Ok(ast) = crate::parser::parse(src) else {
        return found;
    };
    for item in &ast.items {
        if let crate::ast::Node::Import { path, .. } = item {
            let key = path.join(".");
            for &name in SUBSTRATE_MODULES {
                if key == name {
                    found.push(name);
                }
            }
        }
    }
    found
}

/// Transitive closure of the substrate imports reachable from `seed_texts`.
///
/// BFS over the import graph: a substrate module imported by a seed pulls in
/// the substrate modules IT imports (e.g. `std.io_canon` imports
/// `std.sha256`, so a consumer that names only `io_canon` still gets
/// `sha256.o`). Without the transitive step the substrate object itself links
/// with an undefined symbol — the same defect one level down.
///
/// The seeds are source TEXTS, so a caller may seed from one entry file (the
/// flat `--emit-shared` path) or from every project source (the manifest
/// cdylib path) without this function knowing the difference.
pub fn substrate_closure<'a, I>(seed_texts: I) -> BTreeSet<&'static str>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut imported: BTreeSet<&'static str> = BTreeSet::new();
    let mut worklist: Vec<&'static str> = Vec::new();
    for text in seed_texts {
        worklist.extend(scan_substrate_imports(text));
    }
    while let Some(modname) = worklist.pop() {
        if imported.insert(modname) {
            if let Some((_, src)) = crate::project::stdlib::STDLIB_MIND_SOURCES
                .iter()
                .find(|(n, _)| *n == modname)
            {
                for dep in scan_substrate_imports(src) {
                    if !imported.contains(dep) {
                        worklist.push(dep);
                    }
                }
            }
        }
    }
    imported
}

/// The substrate objects of one build, packed into one static archive.
///
/// An archive, not loose objects: the linker pulls a module's object only when
/// something in the link references one of its symbols. Loose objects were
/// linked whether used or not, so a program that imported `std.async` only for
/// a constant and defined its own `pub fn run` failed with "multiple
/// definition of `run`" once `std.async` became a member. The archive and its
/// objects live in a private directory per build, so two builds writing next
/// to each other never race on a shared `__std_<name>.o`; the directory is
/// removed when this value is dropped, so keep it alive until the link is done.
pub struct SubstrateArchive {
    _dir: Option<tempfile::TempDir>,
    archive: Option<PathBuf>,
    plan: super::private_linkage::LinkagePlan,
}

impl SubstrateArchive {
    /// The link input: the archive, or nothing when no module was imported.
    /// Pass it AFTER every object that may reference a substrate symbol.
    pub fn link_inputs(&self) -> Vec<PathBuf> {
        self.archive.iter().cloned().collect()
    }

    /// `mlir` for the build's own source `key` (the key it was planned under),
    /// with that source's planned non-`pub` fns given internal linkage, so the
    /// build's helpers neither collide with nor stand in for std definitions.
    pub fn apply_linkage(&self, key: &Path, mlir: &str) -> String {
        let _linkage = super::private_linkage::install(self.plan.get(key));
        super::private_linkage::apply(mlir)
    }
}

/// Pack `objs` into the static archive `archive` (replacing any old one), with
/// a symbol index and deterministic member headers (`ar rcsD`). `llvm-ar` next
/// to `clang` is preferred; the system `ar` is the fallback.
pub(crate) fn pack_archive(objs: &[PathBuf], archive: &Path, tools: &BuildTools) -> Result<()> {
    if archive.exists() {
        std::fs::remove_file(archive)
            .map_err(|e| anyhow!("cannot replace {}: {e}", archive.display()))?;
    }
    let llvm_ar = Path::new(&tools.clang).with_file_name("llvm-ar");
    let candidates = [llvm_ar.as_os_str(), std::ffi::OsStr::new("ar")];
    let mut last_err = String::new();
    for ar in candidates {
        match std::process::Command::new(ar)
            .arg("rcsD")
            .arg(archive)
            .args(objs)
            .output()
        {
            Ok(out) if out.status.success() => return Ok(()),
            Ok(out) => {
                return Err(anyhow!(
                    "{} failed to pack {}: {}",
                    Path::new(ar).display(),
                    archive.display(),
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
            }
            Err(e) => last_err = format!("{}: {e}", Path::new(ar).display()),
        }
    }
    Err(anyhow!(
        "no archiver found for the std substrate objects ({last_err})"
    ))
}

/// The internal-linkage plan for a build that links `closure`: the
/// `private_linkage` rule applied over the build's own `sources` AND the
/// substrate modules, keyed by source path for the build's sources and by
/// module name (`std.toml`) for the substrate modules.
pub(crate) fn linkage_plan<'a>(
    sources: impl IntoIterator<Item = (&'a Path, &'a str)>,
    closure: &BTreeSet<&'static str>,
    link: &LinkSurface<'_>,
) -> super::private_linkage::LinkagePlan {
    let std_sources = crate::project::stdlib::STDLIB_MIND_SOURCES
        .iter()
        .filter(|(name, _)| closure.contains(name))
        .map(|(name, src)| (Path::new(*name), *src));
    super::private_linkage::plan_with_std(sources, std_sources, link)
}

/// Install a module table and enum registry holding ONLY the bundled std
/// modules for the std substrate compiles that follow; the previous ones come
/// back on drop. A std module must resolve its calls against std and its own
/// `extern "C"` blocks: under a build's project table, a program fn named
/// `read` stood in for std.fs's libc `read` and failed std.fs with E2005.
pub(crate) fn std_only_scope() -> super::single_file_scope::ProjectTableGuard {
    let parsed = crate::project::stdlib::parsed_stdlib_modules();
    let refs: Vec<(String, &crate::ast::Module)> =
        parsed.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = super::module_table::build_module_table(&refs);
    super::single_file_scope::ProjectTableGuard::install_with_enums(
        table,
        super::build_global_enums(&parsed),
    )
}

/// Remove from `closure` every std module the build supplies itself: a project
/// source whose module path is `crate.std.<m>` is the build's own `std.<m>`,
/// so the bundled object must not be linked next to it.
pub(crate) fn drop_project_supplied<'a>(
    closure: &mut BTreeSet<&'static str>,
    module_paths: impl IntoIterator<Item = &'a str>,
) {
    let own: BTreeSet<&str> = module_paths
        .into_iter()
        .map(|p| p.strip_prefix("crate.").unwrap_or(p))
        .collect();
    closure.retain(|m| !own.contains(m));
}

/// Compile each module in `closure` to an object in a private directory, give
/// the planned non-`pub` fns internal linkage, and pack the objects into one
/// archive (see [`SubstrateArchive`]). `sources` are the build's own linked
/// sources and `link` its surface, for the internal-linkage plan.
///
/// Every object has its synthetic `@main` localized (`objcopy
/// --localize-symbol=main`): the MLIR emit gives each module object a `main`,
/// which would otherwise collide with the consumer entry's `main` at link
/// time. The module's public symbols (`canon_*`, `ring_*`, `sha256`, …) stay
/// global so the consumer resolves them.
///
/// Fail-loud: a substrate module that fails to compile, lower, `objcopy` or
/// pack aborts the build with the real diagnostics rather than emitting a `.so`
/// with the symbol still undefined.
pub(crate) fn compile_substrate_objects<'a>(
    closure: &BTreeSet<&'static str>,
    sources: impl IntoIterator<Item = (&'a Path, &'a str)>,
    link: &LinkSurface<'_>,
    target: BackendTarget,
    tools: &BuildTools,
) -> Result<SubstrateArchive> {
    use crate::pipeline::{CompileOptions, compile_source_with_name, lower_to_mlir};

    if closure.is_empty() {
        return Ok(SubstrateArchive {
            _dir: None,
            archive: None,
            plan: Default::default(),
        });
    }
    let dir = tempfile::Builder::new()
        .prefix("mind-substrate-")
        .tempdir()
        .map_err(|e| anyhow!("cannot create the substrate build directory: {e}"))?;
    let plan = linkage_plan(sources, closure, link);
    let _std_scope = std_only_scope();
    let sub_opts = CompileOptions {
        func: None,
        enable_autodiff: false,
        target,
        manifest_exports: Vec::new(),
        ..Default::default()
    };

    let mut objs: Vec<PathBuf> = Vec::new();
    for modname in closure {
        let modname = *modname;
        let Some((_, src)) = crate::project::stdlib::STDLIB_MIND_SOURCES
            .iter()
            .find(|(n, _)| *n == modname)
        else {
            continue;
        };
        let prod = compile_source_with_name(src, Some(modname), &sub_opts).map_err(|e| {
            // Render the real diagnostics (file:line:col + message) instead of
            // the opaque CompileError Display, matching the cdylib build path.
            let diags = e.into_diagnostics(Some(modname));
            let rendered = diags
                .iter()
                .map(|d| crate::diagnostics::render(src, d))
                .collect::<Vec<_>>()
                .join("\n");
            if rendered.trim().is_empty() {
                anyhow!("substrate compile failed for {modname}")
            } else {
                anyhow!("substrate compile failed for {modname}:\n{rendered}")
            }
        })?;
        #[cfg(feature = "autodiff")]
        let sub_mlir = lower_to_mlir(&prod.ir, prod.grad.as_ref())
            .map_err(|e| anyhow!("substrate MLIR lowering for {modname}: {e}"))?;
        #[cfg(not(feature = "autodiff"))]
        let sub_mlir = lower_to_mlir(&prod.ir)
            .map_err(|e| anyhow!("substrate MLIR lowering for {modname}: {e}"))?;
        let mlir = {
            let _linkage = super::private_linkage::install(plan.get(Path::new(modname)));
            super::private_linkage::apply(&sub_mlir.primal_mlir)
        };
        let short = modname.rsplit('.').next().unwrap_or(modname);
        let obj_path = dir.path().join(format!("__std_{short}.o"));
        let sub_bo = mlir_build::BuildOptions {
            preset: mlir_build::preset_for_mlir(&mlir),
            emit_mlir_file: None,
            emit_llvm_file: None,
            emit_obj_file: Some(&obj_path),
            emit_shared: None,
            opt_pipeline: None,
            target_triple: None,
        };
        mlir_build::build_all(&mlir, tools, &sub_bo)
            .map_err(|e| anyhow!("substrate object build for {modname}: {e}"))?;
        let st = std::process::Command::new("objcopy")
            .arg("--localize-symbol=main")
            .arg(&obj_path)
            .status()
            .map_err(|e| anyhow!("objcopy spawn failed for {modname}: {e}"))?;
        if !st.success() {
            return Err(anyhow!(
                "objcopy --localize-symbol=main failed for {modname}"
            ));
        }
        objs.push(obj_path);
    }
    let archive = dir.path().join("libmind_std_substrate.a");
    pack_archive(&objs, &archive, tools)?;
    Ok(SubstrateArchive {
        _dir: Some(dir),
        archive: Some(archive),
        plan,
    })
}

/// The substrate archive a flat `mindc <file> --emit-shared` must link.
///
/// Seeds the closure from the entry and the project sources it links
/// (`linked_siblings`, empty for a file outside any project): a source the
/// entry does not import is not in the `.so`, so neither its std imports nor
/// its fn names concern this link. Then removes any module the entry IS:
/// compiling `std/io_canon.mind` itself must link `sha256.o` but must NOT also
/// link an `io_canon.o` that redefines every `canon_*` symbol the entry
/// already emits. Identity is decided by SOURCE TEXT equality against the
/// bundled blob — an exact test, not a filename heuristic — so a user module
/// that merely happens to be named `io_canon.mind` still links the real
/// `std.io_canon` object. The `.so` exports every global fn, so the plan keeps
/// each `export { … }` name global; the entry is planned under [`ENTRY_KEY`],
/// the siblings under their own paths.
pub fn substrate_objects_for_entry<'a>(
    entry_src: &'a str,
    linked_siblings: &[(&'a Path, &'a str)],
    target: BackendTarget,
    tools: &BuildTools,
) -> Result<SubstrateArchive> {
    let mut closure = substrate_closure(
        std::iter::once(entry_src).chain(linked_siblings.iter().map(|(_, t)| *t)),
    );
    closure.retain(|modname| {
        !crate::project::stdlib::STDLIB_MIND_SOURCES
            .iter()
            .any(|(n, s)| n == modname && *s == entry_src)
    });
    let sources =
        std::iter::once((Path::new(ENTRY_KEY), entry_src)).chain(linked_siblings.iter().copied());
    let no_c_abi = BTreeSet::new();
    compile_substrate_objects(
        &closure,
        sources,
        &LinkSurface::shared_library(&no_c_abi),
        target,
        tools,
    )
}

/// The plan key of the entry source in [`substrate_objects_for_entry`]; pass it
/// to [`SubstrateArchive::apply_linkage`] for the entry's MLIR.
pub const ENTRY_KEY: &str = "entry";
