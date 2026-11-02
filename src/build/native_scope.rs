// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Resolving a native build's source set through the existing project scope.
//!
//! The `--backend native` bridge needs an ORDERED, VALIDATED set of sources to
//! flatten into the frozen compiler's stdin image. That set must come from the
//! project resolver every other path already uses — not from a second crawler,
//! and not from argv. Exactly one argv path reaches the bridge and it is the
//! ENTRY; siblings are discovered, because argv order is not a contract and the
//! image's order is.
//!
//! # Why this is its own module
//!
//! `src/bin/mindc.rs` sits at its size ceiling and the bridge file has its own
//! job. More importantly the resolution decision is separately testable: given an
//! entry path and its bytes, it yields either a single translation unit, an
//! ordered project closure, or a named refusal — with no I/O to the frozen
//! compiler in sight.
//!
//! # The flattening hazard this module exists to refuse
//!
//! The frozen compiler receives ONE flat text. Every top-level `fn` in that text
//! is visible to every other, by bare name, with no notion of module. So
//! flattening a project silently promotes every private sibling function to a
//! callable global — a program that could not compile through the normal
//! resolver would compile here, and mean something different.
//!
//! Module visibility is therefore checked BEFORE flattening, in the owning
//! scope, through the resolver that already knows the answer
//! (`ProjectScope::resolves_imported_symbol`). A reference the project resolver
//! would reject is refused here rather than silently admitted by concatenation.

use std::path::Path;

/// Why a native source set could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeRefusal {
    pub kind: &'static str,
    pub detail: String,
}

/// Captured bytes as text, REFUSING rather than substituting.
///
/// `from_utf8_lossy` would replace invalid sequences with U+FFFD, so the text
/// that gets parsed and type-checked would differ from the bytes composed into
/// the wire image -- validating one program and emitting another, which is the
/// exact split this path exists to prevent. Sources reaching here come from
/// `CapturedSource`, whose field is a `String`, so this cannot fire today; the
/// check makes that invariant enforced instead of assumed, and survives the field
/// becoming bytes later.
#[cfg(feature = "cross-module-imports")]
fn captured_text<'a>(bytes: &'a [u8], what: &NativeSource) -> Result<&'a str, ScopeRefusal> {
    std::str::from_utf8(bytes).map_err(|e| ScopeRefusal {
        kind: "scope.module_not_utf8",
        detail: format!(
            "`{}` (module `{}`) is not valid UTF-8: {e}",
            what.path.display(),
            what.module_path
        ),
    })
}

fn refuse<T>(kind: &'static str, detail: impl Into<String>) -> Result<T, ScopeRefusal> {
    Err(ScopeRefusal {
        kind,
        detail: detail.into(),
    })
}

/// One source in the flattening order, with the ownership metadata preserved.
#[derive(Debug, Clone)]
pub struct NativeSource {
    /// Canonical dotted module path, e.g. `crate.helper`. Retained so a
    /// diagnostic can name the OWNING module rather than a byte offset into the
    /// concatenation.
    pub module_path: String,
    pub path: std::path::PathBuf,
    pub source: Vec<u8>,
    /// True for the module the build was entered through. The entry root must
    /// belong to THIS module -- not to whichever flattened sibling happens to
    /// contain a `main`, which the frozen compiler would otherwise accept.
    pub is_entry: bool,
}

/// The resolved shape of a native build's sources.
#[derive(Debug, Clone)]
pub enum NativeSourceSet {
    /// No imports: exactly the bytes the caller passed, unchanged. This is the
    /// path every existing single-file build takes, and keeping it byte-verbatim
    /// is what preserves the pinned artifact corpus.
    SingleFile {
        path: std::path::PathBuf,
        source: Vec<u8>,
    },
    /// An import-reachable closure, entry first, then ascending module path.
    Project { sources: Vec<NativeSource> },
}

/// A resolved source set plus the scope that produced it, when there was one.
///
/// The scope is carried out rather than reconstructed because the lowering that
/// follows needs the project's enum/type table, and rebuilding that table from
/// the captured text re-derives module identities under a different root. The
/// capture's own scope is the only table that cannot disagree with the bytes it
/// captured. `None` for a single translation unit, which installs nothing and so
/// leaves the pinned single-file artifact corpus byte-identical.
///
/// The scope half exists only where the resolver does. Without
/// `cross-module-imports` there is no `single_file_scope` module at all, so the
/// alias degrades to `Option<()>` and is always `None` -- naming the real type
/// unconditionally is what breaks the default-features build that CI runs.
#[cfg(feature = "cross-module-imports")]
pub type ResolvedNative = (
    NativeSourceSet,
    Option<crate::project::single_file_scope::ProjectScope>,
);

#[cfg(not(feature = "cross-module-imports"))]
pub type ResolvedNative = (NativeSourceSet, Option<()>);

impl NativeSourceSet {
    /// The sources in flattening order.
    pub fn ordered(&self) -> Vec<&[u8]> {
        match self {
            NativeSourceSet::SingleFile { source, .. } => vec![source.as_slice()],
            NativeSourceSet::Project { sources } => {
                sources.iter().map(|s| s.source.as_slice()).collect()
            }
        }
    }

    /// The module path owning the entry, for diagnostics and the entry root rule.
    pub fn entry_module(&self) -> Option<&str> {
        match self {
            NativeSourceSet::SingleFile { .. } => None,
            NativeSourceSet::Project { sources } => sources
                .iter()
                .find(|s| s.is_entry)
                .map(|s| s.module_path.as_str()),
        }
    }
}

// Check captured sources under their owning module before flattening. The
// existing checker supplies export visibility and import precedence, including
// transitive dependencies and explicit export blocks. Reusing it keeps native
// admission aligned with `mindc check` without scanning source text for calls.

/// Resolve the native source set for `entry`.
///
/// Feature-gated deliberately. `cross-module-imports` is NOT a default feature —
/// it is kept off so the default build's byte-identity hot path is untouched, and
/// CI builds `mindc` on defaults. Without it the project resolver does not exist,
/// so a program with imports is REFUSED by name rather than silently compiled as
/// a single file. A silent single-file fallback would hand the frozen compiler a
/// program missing its callees and let the second fence produce a worse
/// diagnostic for a cause the first fence already knew.
#[cfg(feature = "cross-module-imports")]
pub fn resolve_native_sources(
    entry: &Path,
    entry_source: &[u8],
    target_block: &str,
) -> Result<ResolvedNative, ScopeRefusal> {
    use crate::project::single_file_scope::{Discovery, discover_with_source};

    let text = match std::str::from_utf8(entry_source) {
        Ok(t) => t,
        Err(e) => return refuse("scope.entry_not_utf8", format!("{}: {e}", entry.display())),
    };

    // Capture ONCE. Every later consumer reads this snapshot, so nothing can
    // re-read a file between resolution and emission and get different bytes.
    let discovery = match discover_with_source(entry, target_block, text) {
        Ok(d) => d,
        Err(e) => {
            return refuse(
                "scope.discovery_failed",
                format!("{}: {e:#}", entry.display()),
            );
        }
    };

    match discovery {
        Discovery::SingleTranslationUnit => {
            // No project scope means no type check below, so the entry's own
            // `std` imports are admitted here; the project branch admits each
            // captured source's below.
            #[cfg(feature = "std-surface")]
            if let Ok((_module, raw_imports)) = crate::parser::parse_with_imports(text) {
                refuse_unknown_std_import(entry, &raw_imports, |_| false)?;
            }
            Ok((
                NativeSourceSet::SingleFile {
                    path: entry.to_path_buf(),
                    source: entry_source.to_vec(),
                },
                None,
            ))
        }
        Discovery::MissingProject(imports) => refuse(
            "scope.missing_project",
            format!(
                "`{}` imports {:?} but no project manifest governs it; the native backend \
                 needs a resolved project to know which siblings belong in the image",
                entry.display(),
                imports
            ),
        ),
        Discovery::Project(scope) => {
            let entry_canonical = scope.entry().to_path_buf();
            let mut sources = Vec::new();
            // `linked_sources` is the import-reachable body closure in entry-first,
            // then ascending module-path order. That order is a WIRE CONTRACT: the
            // frozen compiler resolves duplicate bare names last-definition-wins, so
            // reordering changes which program runs. Preserved exactly, never sorted
            // again, never deduped.
            for captured in scope.linked_sources() {
                sources.push(NativeSource {
                    module_path: captured.module_path().to_string(),
                    path: captured.path().to_path_buf(),
                    source: captured.source().as_bytes().to_vec(),
                    is_entry: captured.path() == entry_canonical,
                });
            }
            if sources.is_empty() {
                return refuse(
                    "scope.empty_closure",
                    format!(
                        "`{}` resolved to a project with no linked sources",
                        entry.display()
                    ),
                );
            }
            // Require every declared import to resolve to the captured closure,
            // including dependencies that are not referenced by a call.
            let linked: std::collections::BTreeSet<&str> =
                sources.iter().map(|s| s.module_path.as_str()).collect();
            for src in &sources {
                // A failed per-module parse cannot establish complete imports.
                let raw_imports =
                    match crate::parser::parse_with_imports(captured_text(&src.source, src)?) {
                        Ok((_module, imports)) => imports,
                        Err(_) => {
                            return refuse(
                                "scope.module_does_not_parse",
                                format!(
                                    "`{}` (module `{}`) is in the linked closure but does not \
                                 parse; refusing rather than admitting a module whose \
                                 imports and definitions were never examined",
                                    src.path.display(),
                                    src.module_path
                                ),
                            );
                        }
                    };
                // A `std` import the project supplies itself (a linked
                // `std.<name>` module) is in the image; any other must be in
                // the frozen compiler's std seed.
                #[cfg(feature = "std-surface")]
                refuse_unknown_std_import(&src.path, &raw_imports, |path| {
                    scope
                        .resolve_import_path(&src.module_path, path)
                        .is_some_and(|target| linked.contains(target))
                })?;
                for path in local_import_segments(&raw_imports) {
                    let resolved = scope.resolve_import_path(&src.module_path, &path);
                    match resolved {
                        Some(target) if linked.contains(target) => {}
                        Some(target) => {
                            return refuse(
                                "scope.import_not_linked",
                                format!(
                                    "`{}` (module `{}`) imports `{}`, which resolved to `{}` \
                                     but is not in the linked closure",
                                    src.path.display(),
                                    src.module_path,
                                    path.join("."),
                                    target
                                ),
                            );
                        }
                        None => {
                            return refuse(
                                "scope.unresolved_import",
                                format!(
                                    "`{}` (module `{}`) imports `{}`, which the project \
                                     resolver could not resolve to any module. An unresolved \
                                     import is an incomplete closure, not a no-op",
                                    src.path.display(),
                                    src.module_path,
                                    path.join(".")
                                ),
                            );
                        }
                    }
                }
            }
            validate_captured(&sources, &scope)?;

            // The entry root must belong to the ENTRY module. Without this an
            // entry with no `main` and a sibling that has one builds and runs the
            // SIBLING's program: admission roots at "main" over the merged image,
            // which cannot tell which module supplied it.
            {
                // MISSING ENTRY IS REFUSED BEFORE ANY DEFAULT. `unwrap_or_default`
                // here would silently substitute an empty module, which has no
                // `main`, turning a missing entry into a misleading
                // "defines no main" instead of naming the real fault.
                let Some(entry_src_owner) = sources.iter().find(|s| s.is_entry) else {
                    return refuse(
                        "scope.entry_not_in_closure",
                        format!(
                            "`{}` is not among its own linked sources",
                            entry_canonical.display()
                        ),
                    );
                };
                let entry_has_main =
                    crate::parser::parse(captured_text(&entry_src_owner.source, entry_src_owner)?)
                        .map(|m| {
                            m.items.iter().any(
                            |it| matches!(it, crate::ast::Node::FnDef(fd, _) if fd.name == "main"),
                        )
                        })
                        .unwrap_or(false);
                if !entry_has_main {
                    return refuse(
                        "scope.entry_module_has_no_main",
                        format!(
                            "entry `{}` defines no `main`; the entry root must belong to the \
                             entry module, not to whichever linked sibling happens to define one",
                            entry_canonical.display()
                        ),
                    );
                }
            }
            if !sources.iter().any(|s| s.is_entry) {
                return refuse(
                    "scope.entry_not_in_closure",
                    format!(
                        "`{}` is not among its own linked sources; the entry root must belong \
                         to the entry module, not to whichever sibling defines a main",
                        entry_canonical.display()
                    ),
                );
            }
            Ok((NativeSourceSet::Project { sources }, Some(scope)))
        }
    }
}

/// Without `cross-module-imports` the project resolver is not compiled in.
///
/// A single-file program is unaffected. A program WITH imports is refused by
/// name — never silently treated as single-file, which is the no-fallback rule
/// the whole backend is built on.
#[cfg(not(feature = "cross-module-imports"))]
pub fn resolve_native_sources(
    entry: &Path,
    entry_source: &[u8],
    _target_block: &str,
) -> Result<ResolvedNative, ScopeRefusal> {
    // The parser's answer, never a text scan. A scan over source lines was
    // measured wrong in BOTH directions: `import\tx;` and `use x;` are real
    // `Node::Import`s the grammar accepts but `starts_with("import ")` misses,
    // so an unresolved dependency compiled and RAN; while `import std.math;`
    // and an import inside a block comment were refused although neither is a
    // local dependency. Lowering treats every `Node::Import` as a compile-time
    // no-op, so nothing downstream recovers the fact this leg drops.
    let text = match std::str::from_utf8(entry_source) {
        Ok(t) => t,
        Err(e) => {
            return refuse("scope.entry_not_utf8", format!("{}: {e}", entry.display()));
        }
    };
    // An entry this build cannot parse is refused rather than guessed at: its
    // imports are not enumerable, and the featured profile already refuses the
    // same source through `discover_with_source`. Matching it keeps the two
    // profiles from disagreeing about what a program even is.
    // A failed parse REFUSES. It is never reclassified as a safe single
    // translation unit: the parser is the only component that knows what this
    // source declares, so if it could not read the source, nothing downstream
    // may claim the source declares no dependency.
    let raw_imports = match crate::parser::parse_with_imports(text) {
        Ok((_module, imports)) => imports,
        Err(_) => {
            return refuse(
                "scope.entry_does_not_parse",
                format!(
                    "`{}` does not parse, so its imports cannot be enumerated; \
                     refusing rather than compiling it as a single file.",
                    entry.display()
                ),
            );
        }
    };
    #[cfg(feature = "std-surface")]
    refuse_unknown_std_import(entry, &raw_imports, |_| false)?;
    let local = local_import_segments(&raw_imports);
    if !local.is_empty() {
        return refuse(
            "scope.imports_need_feature",
            format!(
                "`{}` declares local import(s) {:?}, but this build has no project \
                 resolver (feature `cross-module-imports` is disabled). Refusing rather \
                 than compiling it as a single file with its callees missing.",
                entry.display(),
                local.iter().map(|p| p.join(".")).collect::<Vec<_>>()
            ),
        );
    }
    Ok((
        NativeSourceSet::SingleFile {
            path: entry.to_path_buf(),
            source: entry_source.to_vec(),
        },
        None,
    ))
}

/// Find imported declarations that the current merged lowering cannot bind.
///
/// Lowering has one ambient owner and one local alias table. Qualified enums,
/// record identities and alias-dependent return types therefore require this
/// refusal until lowering can preserve each source owner's semantic context.
/// This is a temporary whole-source capability restriction: even an unused or
/// private imported alias, struct, or enum is refused. Transparent module blocks
/// retain the same declaration ownership; reachability does not narrow the rule.
#[cfg(feature = "cross-module-imports")]
pub(crate) fn owner_sensitive_imported_bodies(sources: &[NativeSource]) -> Vec<String> {
    // A later sibling's `Word = u8` can otherwise replace `Word = i16` in an
    // imported function's return mask. Refuse before this becomes a wrong value
    // (300 -> 44), independently of source ordering or downstream profile limits.
    fn owner_bearing(items: &[crate::ast::Node]) -> Option<&'static str> {
        for item in items {
            let found = match item {
                crate::ast::Node::EnumDef { .. } => Some("enum"),
                crate::ast::Node::TypeAlias { .. } => Some("type alias"),
                crate::ast::Node::StructDef { .. } => Some("struct"),
                // Transparent blocks are `module NAME { ... }`; a declaration
                // nested there is owner-bearing exactly as much as a top-level
                // one, and is invisible to a direct-items scan.
                crate::ast::Node::Block { stmts, .. } => owner_bearing(stmts),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }
    sources
        .iter()
        .filter(|src| !src.is_entry)
        .filter_map(|src| {
            let module = std::str::from_utf8(&src.source)
                .ok()
                .and_then(|text| crate::parser::parse(text).ok())?;
            owner_bearing(&module.items).map(|kind| format!("{} ({kind})", src.module_path))
        })
        .collect()
}

/// Refuse, at the import, a `std` import the frozen compiler cannot satisfy.
///
/// The frozen compiler treats every import as a no-op and sees only the std
/// modules its seed blob was minted from (the manifest's `native_seed` rows),
/// so without this an import of any other module was accepted and each call
/// into it refused later as an unresolved closure edge. A module no std
/// table names is `scope.unknown_std_module` (as `mindc check` refuses it,
/// E2002); a shipped module outside the seed is
/// `scope.std_module_not_in_native_seed`. Builtin modules
/// (`BUILTIN_STD_MODULES`) need no source, and `supplied` admits a
/// `std.<name>` the build's own sources provide.
#[cfg(feature = "std-surface")]
fn refuse_unknown_std_import(
    entry: &Path,
    raw_imports: &[String],
    supplied: impl Fn(&[String]) -> bool,
) -> Result<(), ScopeRefusal> {
    let seed = super::native_bridge::frozen_std_seed_modules();
    for dotted in raw_imports {
        let path: Vec<String> = dotted.split('.').map(str::to_string).collect();
        let Some(module) = dotted.strip_prefix("std.") else {
            continue;
        };
        if supplied(&path) {
            continue;
        }
        if let Some(why) = crate::project::stdlib::unknown_std_import(&path) {
            return refuse(
                "scope.unknown_std_module",
                format!("`{}`: {why}", entry.display()),
            );
        }
        if crate::project::stdlib::BUILTIN_STD_MODULES.contains(&dotted.as_str())
            || seed.contains(&module)
        {
            continue;
        }
        return refuse(
            "scope.std_module_not_in_native_seed",
            format!(
                "`{}` imports `{dotted}`, which ships with mindc but is not in the std seed \
                 the frozen native compiler was minted from (the `native_seed` column of \
                 examples/mindc_mind/testdata/stdlib_manifest.txt), so none of its \
                 functions can be called from a native build. Use `--backend mlir`, which \
                 links every bundled module",
                entry.display()
            ),
        );
    }
    Ok(())
}

/// The parser's recorded import paths, as LOCAL dependency segments.
///
/// Replaces a hand-rolled AST walk. That walk enumerated statement-carrying
/// containers by hand and was measured escaping a real case: it recursed into
/// `Node::Block` only, so `fn main() { import absent_module; }` was invisible to
/// it and the program compiled with its declared dependency unresolved. Widening
/// the enumeration closed that case but left the same shape of hole open for any
/// container added later.
///
/// The parser's own record has no such hole. `parse_import` and `parse_use` both
/// push unconditionally, and `parse_stmt` accepts those keywords wherever a
/// statement is accepted, so nesting depth, container kind, expression wrappers
/// and future `Node` variants are all irrelevant to it.
///
/// Two policies are preserved EXACTLY as the walk had them, because this changes
/// the SOURCE of the answer, not its meaning:
///
/// * `std` paths are dropped -- satisfied by the seed std blob, not local
///   dependencies.
/// * FULL dotted paths are split back into ordered segments. A last-segment view
///   must never become the authority: `a.helper` and `b.helper` are different
///   modules, and collapsing them is the ambiguity the closure check exists to
///   refuse.
fn local_import_segments(raw: &[String]) -> Vec<Vec<String>> {
    raw.iter()
        .map(|dotted| {
            dotted
                .split('.')
                .map(str::to_string)
                .collect::<Vec<String>>()
        })
        .filter(|segs| !matches!(segs.first().map(String::as_str), Some("std")))
        .collect()
}

/// Type-check the CAPTURED sources against the resolver's own installed scope.
///
/// A seam, deliberately: parameterised by the captured set so a test can hand it
/// bytes that differ from what is on disk and observe which the verdict followed.
/// That is the property that matters -- the checker and the wire image must see
/// one snapshot -- and it is not observable through the CLI, where capture and
/// validation happen in the same instant.
///
/// Uses `scope.install()`, the guard the ProjectScope built at capture, carrying
/// its table, enums and canonical module identities. Not `install_for_check`:
/// that rediscovers from disk for a single file and rebuilds module paths from
/// `common_ancestor_dir` for several, so either branch can disagree with the
/// capture this bridge emits.
#[cfg(feature = "cross-module-imports")]
pub(crate) fn validate_captured(
    sources: &[NativeSource],
    scope: &crate::project::single_file_scope::ProjectScope,
) -> Result<(), ScopeRefusal> {
    let _guard = scope.install();
    for src in sources {
        // Each captured source is checked AS ITSELF. Owner-qualified resolution
        // reads an ambient current module, so without this every module in the
        // set would be checked under the ENTRY's owner: a sibling's imports
        // would resolve against the entry's import graph, and a symbol visible
        // only to the entry would appear visible to every module in the image.
        //
        // The guard is RAII because this loop has early returns. A bare
        // set/clear would leak the wrong owner out of the function on any
        // refusal path, which is the failure mode the guard type exists for.
        let _owner = crate::qualified_enums::ModuleGuard::install(src.module_path.clone());
        let text = captured_text(&src.source, src)?;
        let ast = match crate::parser::parse(text) {
            Ok(a) => a,
            Err(_) => {
                return refuse(
                    "scope.module_does_not_parse",
                    format!(
                        "`{}` (module `{}`) does not parse",
                        src.path.display(),
                        src.module_path
                    ),
                );
            }
        };
        let diags = crate::type_checker::check_module_types_in_file(
            &ast,
            text,
            src.path.to_str(),
            &crate::type_checker::TypeEnv::default(),
        );
        if let Some(first) = diags.first() {
            return refuse(
                "scope.rejected_by_type_check",
                format!(
                    "`{}` (module `{}`) is rejected by the type checker over the captured \
                     source set: {}",
                    src.path.display(),
                    src.module_path,
                    first.message.lines().next().unwrap_or("type error")
                ),
            );
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cross-module-imports"))]
#[path = "native_scope_tests.rs"]
mod snapshot_identity_tests;
