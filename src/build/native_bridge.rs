// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Builds native executables with the frozen pure-MIND x86-64 compiler.
//!
//! Project sources are captured in resolver order and checked under their owning
//! modules before composition. Admission examines the complete user image;
//! reachability does not exempt retained bodies from the frozen construct profile.
//! The single-source wire layout and pinned executable bytes are preserved.

#[cfg(feature = "std-surface")]
use std::collections::BTreeSet;
use std::process;

#[cfg(feature = "std-surface")]
mod admission;
#[cfg(feature = "std-surface")]
mod child;
#[cfg(feature = "std-surface")]
mod entry;
#[cfg(feature = "std-surface")]
use entry::{check_entry_binds_main_params, check_entry_is_not_no_main_stub};
#[cfg(feature = "std-surface")]
mod namespace;
#[cfg(feature = "std-surface")]
mod outcome;
#[cfg(feature = "std-surface")]
mod print;
#[cfg(feature = "std-surface")]
mod seed;
#[cfg(feature = "std-surface")]
mod shadow;

#[cfg(feature = "std-surface")]
/// The `native_seed` column of the committed std manifest, in FILE ORDER.
///
/// enforced-by: STDLIB-MANIFEST
///
/// One source of truth, shared with the Python self-host smokes through
/// `examples/mindc_mind/_stdlib_manifest.py`, which parses the same file with the
/// same rules. `include_str!` bakes the manifest in at compile time, so the
/// shipping binary gains NO runtime file dependency (the bridge must keep working
/// from any cwd) and editing the manifest forces a rebuild of this reader.
///
/// The manifest is exhaustive over `std/*.mind` and its row order is the byte
/// order of the std blob the frozen `stage1.elf` was minted from; both properties
/// are gated by `examples/mindc_mind/stdlib_manifest_lint.py`.
fn frozen_std_seed_modules() -> Vec<&'static str> {
    const MANIFEST: &str = include_str!("../../examples/mindc_mind/testdata/stdlib_manifest.txt");
    MANIFEST
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let module = fields.next()?;
            (fields.next()? == "seed").then_some(module)
        })
        .collect()
}

/// Refuse native requests when the build lacks the closure-admission surface.
#[cfg(not(feature = "std-surface"))]
pub fn run_native_backend_bridge(_paths: &[String], _out: &Option<String>, _emit: &Option<String>) {
    eprintln!(
        "error[backend-native]: the native backend is not available in this build \
         (feature `std-surface` is disabled), so closure admission cannot run. \
         Refusing rather than emitting an unadmitted artifact. \
         (backend.native_unavailable)"
    );
    process::exit(3);
}

/// Refuse native requests when the build lacks the closure-admission surface.
#[cfg(not(feature = "std-surface"))]
pub fn run_native_backend_bridge_profile(
    paths: &[String],
    out: &Option<String>,
    emit: &Option<String>,
    _profile: &str,
) {
    run_native_backend_bridge(paths, out, emit)
}

/// The committed BOUNDED-profile compiler seed (see `NativeLayout::BOUNDED`).
pub const BOUNDED_SEED_PATH: &str = "examples/mindc_mind/testdata/selfhost_loop_bounded/stage1.elf";

/// Default-profile entry point (no explicit layout): byte-identical behavior.
#[cfg(feature = "std-surface")]
pub fn run_native_backend_bridge(paths: &[String], out: &Option<String>, emit: &Option<String>) {
    run_native_backend_bridge_profile(paths, out, emit, "default")
}

/// Compose and validate the captured source image, then invoke the pure-MIND
/// native compiler. Its stdout is the executable artifact without wrapper bytes.
/// `MINDC_NATIVE_ELF` overrides the committed seed; unsupported programs refuse.
///
/// `profile` selects the memory layout of the emitted image (`--native-profile`):
/// `default` sends the v1 header (historical layout, byte-identical images);
/// `bounded` sends `NativeLayout::BOUNDED` in a v2 header, runs the committed
/// bounded seed unless `MINDC_NATIVE_ELF` overrides it, and refuses an artifact
/// whose arena/stack segments do not match the requested layout.
#[cfg(feature = "std-surface")]
pub fn run_native_backend_bridge_profile(
    paths: &[String],
    out: &Option<String>,
    emit: &Option<String>,
    profile: &str,
) {
    let layout = match crate::build::native_image::NativeLayout::from_profile(profile) {
        Ok(l) => l,
        Err(msg) => {
            eprintln!("error[backend-native]: {msg}");
            process::exit(2);
        }
    };
    if let Some(l) = &layout {
        if let Err(msg) = l.validate() {
            eprintln!("error[backend-native]: invalid native layout: {msg}");
            process::exit(2);
        }
    }

    // EXECUTABLE TARGETS ONLY. The frozen compiler emits a static ET_EXEC and
    // this bridge asserts that shape before writing. Until it can emit a library
    // container, a library request is REFUSED rather than satisfied with an
    // executable wearing a library name -- measured before this guard:
    // `--emit=cdylib` wrote a 397-byte ET_EXEC to `o.so` and reported success.
    // A library target follows its own contract; it does not silently become an
    // executable, and it must not inherit the entry-`main` requirement either.
    match emit.as_deref() {
        None | Some("binary") => {}
        Some(other) => {
            eprintln!(
                "error[backend-native]: --emit={other} is not supported by the native \
                 backend, which emits a static executable (ET_EXEC). Refusing rather than \
                 writing an executable under a library name. Use --backend mlir for library \
                 emission."
            );
            process::exit(2);
        }
    }

    // EXACTLY ONE argv path, and it is the ENTRY. Siblings are never taken from
    // argv: the image's module order is a wire contract (the frozen compiler
    // resolves duplicate bare names last-definition-wins, so order decides which
    // program runs), and argv order is whatever a caller happened to type.
    // Siblings come from the project resolver instead.
    // NO PATH: resolve the entry the manifest declares, the same way an ordinary
    // `mindc build` in a project does. Previously this exited with "got 0", so a
    // bare `mindc build --backend native` inside a project could not build at all
    // even though the manifest names its entry.
    let resolved_entry: String = if paths.is_empty() {
        match crate::project::find_project_root().and_then(|root| {
            let manifest = crate::project::load_manifest(&root)?;
            Ok(root.join(&manifest.build.entry))
        }) {
            Ok(entry) => entry.display().to_string(),
            Err(e) => {
                eprintln!(
                    "error[backend-native]: no source given and no project entry could be \
                     resolved: {e:#}"
                );
                process::exit(2);
            }
        }
    } else if paths.len() == 1 {
        paths[0].clone()
    } else {
        eprintln!(
            "error[backend-native]: the native backend bridge takes at most one entry \
             source (got {}). Sibling modules are discovered through the project \
             resolver, not passed on the command line -- argv order is not a contract \
             and the image's order is.",
            paths.len()
        );
        process::exit(2);
    };
    let paths: &[String] = std::slice::from_ref(&resolved_entry);
    let entry_path = std::path::Path::new(&resolved_entry);
    let entry_src = match std::fs::read(entry_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "error[backend-native]: cannot read source '{}': {e}",
                paths[0]
            );
            process::exit(2);
        }
    };

    // Resolve the source set through the SAME project scope every other path
    // uses. Single-file programs come back with their bytes verbatim, which is
    // what keeps the pinned artifact corpus byte-identical.
    let (source_set, resolved_scope) =
        match crate::build::native_scope::resolve_native_sources(entry_path, &entry_src, "cpu") {
            Ok(pair) => pair,
            Err(refusal) => {
                eprintln!(
                    "error[backend-native]: {} ({})\n  \
                     This is a refusal, not a fallback: the native backend never compiles \
                     a program whose source set it could not resolve.",
                    refusal.detail, refusal.kind
                );
                process::exit(3);
            }
        };
    let user_sources: Vec<Vec<u8>> = source_set.ordered().iter().map(|s| s.to_vec()).collect();
    if let crate::build::native_scope::NativeSourceSet::Project { sources, .. } = &source_set {
        eprintln!(
            "native: {} linked module(s), entry `{}`",
            sources.len(),
            source_set.entry_module().unwrap_or("?")
        );
    }

    // Compose the std blob exactly as the pure-MIND compiler expects: the seed std
    // modules in the manifest's fixed order, '\n'-joined with a trailing '\n'.
    //
    // enforced-by: STDLIB-MANIFEST
    //
    // This list used to be a hand-copied literal twin of
    // self_host_standalone_driver_smoke.py::_STDLIB_MODULES (and, it turned out, of
    // fifteen more copies across examples/mindc_mind/*.py), with nothing asserting
    // they agreed. Both readers now consume the ONE committed manifest, and
    // examples/mindc_mind/stdlib_manifest_lint.py fails on drift in both directions.
    let std_modules = frozen_std_seed_modules();
    // The frozen stage1.elf was minted from EXACTLY this many modules in EXACTLY
    // this order. Changing the seed set changes the compiled bytes of every native
    // build — a whole-corpus RESEED event, never a routine edit — so it must not be
    // able to arrive as a quiet one-line manifest change. Refuse instead.
    const FROZEN_STD_SEED_COUNT: usize = 21;
    if std_modules.len() != FROZEN_STD_SEED_COUNT {
        eprintln!(
            "error[backend-native]: std manifest declares {} seed modules, but the frozen \
             stage1.elf was minted from {FROZEN_STD_SEED_COUNT}.\n  \
             Changing the seed set is a whole-corpus reseed event: re-mint stage1.elf and \
             update FROZEN_STD_SEED_COUNT in the same change. Refusing rather than \
             emitting bytes against a std blob the frozen compiler was not built on.",
            std_modules.len()
        );
        process::exit(2);
    }
    // `MINDC_STD_DIR` when set, otherwise the copy embedded in this binary,
    // verified against the manifest's SEED_BLOB_SHA256 pin before use.
    let std_sources = seed::std_seed_sources(&std_modules).unwrap_or_else(|e| {
        eprintln!("error[backend-native]: {e}");
        process::exit(2);
    });

    // Composition goes through `native_image` so the single-file path and the
    // coming multi-module path build the image with the SAME code. With one
    // source the joiner contributes nothing, which is what makes "single-file
    // bytes did not move" a mechanical check instead of an argument.
    let std_blob = crate::build::native_image::compose_std_blob(&std_sources);
    // Layout and feature admission are independent: a memory limit is not a
    // compiler capability. Only the selected compiler's distinct lane marker
    // chooses the candidate profile; unmarked seeds retain the old contract.
    let lane_v1 = seed::compiler_supports_native_lane_v1(layout.is_some());

    // Admission and emission use the same captured, composed user region.
    // The exhaustive frozen-profile allowlist checks every retained user body,
    // including uncalled functions. The current bridge does not admit calls into
    // the seed standard library: unresolved edges and seed-name collisions refuse
    // (a user function that is an exact private copy of a private seed helper
    // is not a collision; see `namespace::identical_private_seed_copies`).
    // The seed compiler's own refusal is a second fence, with no backend fallback.
    // ADMISSION over the merged program. The fence must reason about the same
    // text the ELF receives, so everything below parses THIS composition once --
    // never each file separately, and never a splice of independently lowered IR
    // modules (IRModule::default starts next_id at 0, so that would alias
    // ValueIds and corrupt value_types).
    //
    // Reserved names come from the std blob TEXT rather than by parsing std: the
    // backend must keep working on a build whose feature set the Rust parser
    // would reject, and the scan over-approximates, which is the safe direction
    // for a set whose only use is to refuse.
    let reserved = crate::ir::native_closure::reserved_names_from_text(&String::from_utf8_lossy(
        std_blob.as_bytes(),
    ));
    // Numeric statement `print` is not a native-lane intrinsic. Rewrite only
    // the supported i64 shape into a private helper before admission, so the
    // fence and the emitted user image are the same bytes. Sources with no
    // `print` stay byte-identical. Unsupported prints are left in place and
    // still fail the source-profile refusal.
    let user_sources = match print::expand_numeric_statement_prints(&user_sources, &reserved) {
        Ok(print::PrintSources::Same) => user_sources,
        Ok(print::PrintSources::Expanded(sources)) => sources,
        Err(msg) => {
            eprintln!("error[backend-native]: {msg}");
            process::exit(3);
        }
    };
    // Compose only after the print rewrite. Admission and the emitted image
    // must be this same byte string. Composing first left `print` in the
    // fence, so the lane refused it and the helper never reached the compiler.
    let fence_source = crate::build::native_image::compose_user_region(&user_sources);
    // ONE parse, ONE lower. The admitted module is handed straight to the fence
    // rather than re-derived: two lowerings of the same text are two chances to
    // reason about different programs, and the whole point of this path is that
    // the fence and the emitter see one artifact.
    // The merged image is lowered under the CAPTURE'S OWN table, not a bare
    // global one. Without it, lowering cannot resolve a receiver that a sibling
    // module declares -- a cross-module enum variant reaches
    // `lower.rs`'s "refusing to emit const 0" guard and aborts the process
    // instead of producing a diagnostic. Measured: `Color.Red` declared in
    // `src/a/helper.mind` and used from the entry panicked here, while the same
    // construct in ONE file refused cleanly through the frozen compiler.
    //
    // `None` for a single translation unit, so nothing is installed on the path
    // the pinned artifact corpus takes and those bytes cannot move.
    #[cfg(feature = "cross-module-imports")]
    let _scope_guard = resolved_scope.as_ref().map(|s| s.install());
    #[cfg(not(feature = "cross-module-imports"))]
    let _scope_guard = resolved_scope;
    // Refuse BEFORE lowering a body under an owner that is not its own. One
    // merged image means one ambient owner for the whole lowering, and this
    // bridge does not splice per-body SSA fragments, so an owner-sensitive
    // declaration in an imported body cannot be lowered correctly here.
    //
    // Gated with the resolver: without `cross-module-imports` there is no
    // project resolution, `Project` is never constructed, and no imported body
    // can reach the merged image in the first place.
    #[cfg(feature = "cross-module-imports")]
    if let crate::build::native_scope::NativeSourceSet::Project { sources, .. } = &source_set {
        let offenders = crate::build::native_scope::owner_sensitive_imported_bodies(sources);
        if !offenders.is_empty() {
            eprintln!(
                "error[backend-native]: imported module(s) {offenders:?} carry an \
                 owner-bearing declaration, whose meaning depends on which module \
                 declares it. \
                 The merged image is lowered under a single owner, so an imported body \
                 would be lowered under an owner that is not its own. Refusing rather \
                 than binding it to the wrong owner. (scope.owner_sensitive_imported_body)"
            );
            process::exit(3);
        }
    }
    let (merged_ast, merged_ir, omit_seed) =
        match admit_merged_program(&paths[0], &fence_source, &reserved, &std_sources, lane_v1) {
            Ok(lowered) => lowered,
            Err(msg) => {
                eprintln!("error[backend-native]: {msg}");
                process::exit(3);
            }
        };
    // DIALECT FENCE: an import the frozen compiler would read as a DIFFERENT
    // extent than the admission parser did. Admission above reasoned about the
    // Rust parse; the ELF is built from the frozen parse of the same bytes, so
    // where the two disagree on where an item ends they compile different
    // programs. Refused before the frozen compiler ever sees the image.
    let unterminated = unterminated_imports(&merged_ast, &fence_source);
    if !unterminated.is_empty() {
        eprintln!(
            "error[backend-native]: {}",
            unterminated_import_refusal(&paths[0], &unterminated)
        );
        process::exit(3);
    }
    // An entry `main` with parameters would run with them all 0 unless the
    // compiler binds them to argc/argv: refuse every shape but
    // `main(argc: i64, argv: i64)`, whose binding is verified on the artifact.
    let main_params = match admission::refuse_entry_main_with_params(&merged_ast, &paths[0]) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error[backend-native]: {msg}");
            process::exit(3);
        }
    };
    // A nested binding that re-declares an enclosing name would be merged into
    // the outer binding by the frozen compiler: refuse, never miscompile.
    // `admit_merged_program` already proved the image is UTF-8, so this is exact.
    let fence_text = String::from_utf8_lossy(&fence_source);
    let origins = shadow_origins(&source_set, &paths[0]);
    // A compiler carrying the scoped-nested-bindings mark gives each nested
    // `let` its own storage; the proven shadow shapes are then admitted.
    let bounded = layout.is_some();
    let compiler_scopes = || seed::compiler_scopes_nested_bindings(bounded);
    if let Err(msg) =
        admission::refuse_shadowing(&merged_ast, &fence_text, &origins, &compiler_scopes)
    {
        eprintln!("error[backend-native]: {msg}");
        process::exit(3);
    }
    // Whole-module admission over that single lowering. The image is not pruned,
    // so an uncalled out-of-profile body still becomes bytes in the
    // artifact; reachability informs diagnostics, it does not narrow admission.
    // enforced-by: RI-D1-PROFILE
    // The source check covers what the IR cannot carry (a declared f32 width).
    // The candidate path already checked its complete typed profile before
    // namespace selection. Never apply its additions to an unmarked compiler.
    if !lane_v1 {
        match crate::ir::frozen_profile::profile_frozen_admits_source(&merged_ast)
            .and_then(|()| crate::ir::frozen_profile::profile_frozen_admits(&merged_ir))
        {
            Ok(()) => {}
            Err(rejection) => {
                eprintln!(
                    "error[backend-native]: `{}` is not in the frozen native profile \
                 (out-of-profile construct: {}).\n  \
                 The native backend is proven byte-identical only on the frozen \
                 construct allowlist. Rerun with `--backend mlir` for the full \
                 surface. This is a refusal, not a fallback: silently compiling an \
                 unproven construct is the failure this fence exists to prevent.",
                    paths[0], rejection.construct
                );
                process::exit(3);
            }
        }
    }

    // Compose only after namespace and type admission. A closed private-helper
    // collision omits the WHOLE unused seed, never renames a user symbol or
    // drops a user body. The ordinary path keeps the exact historical bytes.
    let selected_std = if omit_seed {
        crate::build::native_image::compose_std_blob(&[])
    } else {
        std_blob
    };
    let image = crate::build::native_image::compose_image_with_layout(
        &selected_std,
        &user_sources,
        layout.as_ref(),
    );
    let image = image.bytes;

    // Resolve the pure-MIND compiler ELF (the RI-D shell-out target):
    // `MINDC_NATIVE_ELF` when set, otherwise the embedded seed of the requested
    // profile, verified against its committed MANIFEST pin before it runs.
    let compiler = seed::compiler_elf(layout.is_some()).unwrap_or_else(|e| {
        eprintln!("error[backend-native]: {e}");
        process::exit(2);
    });
    let elf = compiler.path.display().to_string();

    let child = match std::process::Command::new(&elf)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            // execve failing before its point of no return (ENOMEM, EACCES,
            // ENOEXEC, ...) surfaces here, not as a child status.
            drop(compiler);
            eprintln!(
                "error[backend-native][{}]: failed to spawn compiler ELF '{elf}': {e}; no \
                 artifact written (native backend is fail-closed — no MLIR fallback).",
                child::CHILD_FAILURE_CODE
            );
            process::exit(2);
        }
    };
    // Stream, drain and ALWAYS reap. A broken pipe is only a symptom of the
    // child's fate (a loader ENOMEM under an address-space cap ends in SIGSEGV
    // before stdin is read), so the report carries the reaped outcome.
    let run = child::supervise(child, &image, child::REAP_GRACE);
    // Remove an unpacked embedded seed now: every later exit skips destructors.
    drop(compiler);
    let elf_bytes = match outcome::finish_compiler_run(run) {
        Ok(bytes) => bytes,
        Err(code) => process::exit(code),
    };
    // A real static ELF (magic + ET_EXEC), never an empty/garbage artifact.
    if elf_bytes.len() < 256 || &elf_bytes[0..4] != b"\x7fELF" {
        eprintln!(
            "error[backend-native]: compiler produced a non-ELF/short artifact ({} bytes); \
             refusing to write.",
            elf_bytes.len()
        );
        process::exit(1);
    }
    // BACKSTOP: admission already proved the program defines `main` (a missing
    // root is refused as closure.missing_entry_root), so an artifact whose entry
    // is the frozen compiler's no-`main` library stub means the compiler did not
    // see the `main` that was admitted. Whatever the cause, that artifact is a
    // silent `exit(0)` and never a correct build of this program.
    if let Err(why) = check_entry_is_not_no_main_stub(&elf_bytes, layout.as_ref()) {
        eprintln!(
            "error[backend-native]: `{}`: {why}; the source declares `main`, so this \
             artifact would not run the admitted program. No artifact written. \
             (native.entry_is_no_main_stub)",
            paths[0]
        );
        process::exit(3);
    }
    // W4b: `main(argc: i64, argv: i64)` was admitted on condition that the
    // compiler binds both parameters; a seed without the binding is refused here.
    if main_params == admission::MainParams::ArgcArgv
        && let Err(why) = check_entry_binds_main_params(&elf_bytes, layout.as_ref())
    {
        eprintln!(
            "error[backend-native]: `{}`: the entry `fn main` declares parameter(s) \
             [\"argc\", \"argv\"], but {why} (a silent miscompile). Use a compiler seed \
             that binds `main(argc, argv)`, declare `fn main()`, or rerun with \
             `--backend mlir`. No artifact written. (native.entry_main_has_params)",
            paths[0]
        );
        process::exit(3);
    }

    // The artifact must carry exactly the requested layout. A seed that predates
    // v2 headers (or ignores them) would otherwise hand back a default-layout
    // image for a bounded request -- one that cannot load under the cap.
    if let Some(l) = &layout {
        let want = l.expected_bss_segments().to_vec();
        match crate::build::native_image::rw_load_segments(&elf_bytes) {
            Some(got) if got == want => {}
            got => {
                eprintln!(
                    "error[backend-native]: compiler '{elf}' produced an image whose \
                     arena/stack segments {got:?} do not match the requested `{profile}` \
                     layout {want:?} (vaddr, memsz); refusing to write. Use a seed built \
                     for explicit layouts."
                );
                process::exit(1);
            }
        }
    }

    let out_path = out.clone().unwrap_or_else(|| "a.out".to_string());
    if let Err(e) = std::fs::write(&out_path, &elf_bytes) {
        eprintln!("error[backend-native]: cannot write output '{out_path}': {e}");
        process::exit(2);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(md) = std::fs::metadata(&out_path) {
            let mut perm = md.permissions();
            perm.set_mode(perm.mode() | 0o755);
            let _ = std::fs::set_permissions(&out_path, perm);
        }
    }
    eprintln!(
        "native: wrote {} ({} bytes) via the pure-MIND native-ELF backend \
         (zero MLIR/LLVM/clang)",
        out_path,
        elf_bytes.len()
    );
}

/// Each user source's display path and its start offset in the composed user
/// region (`native_image::compose_user_region` joins them with one `\n`), so a
/// refusal can name the file and line a binding came from.
#[cfg(feature = "std-surface")]
fn shadow_origins(
    set: &crate::build::native_scope::NativeSourceSet,
    entry: &str,
) -> Vec<(String, usize)> {
    match set {
        crate::build::native_scope::NativeSourceSet::SingleFile { .. } => {
            vec![(entry.to_string(), 0)]
        }
        crate::build::native_scope::NativeSourceSet::Project { sources } => {
            let mut start = 0;
            sources
                .iter()
                .map(|s| {
                    let origin = (s.path.display().to_string(), start);
                    start += s.source.len() + 1;
                    origin
                })
                .collect()
        }
    }
}

#[cfg(feature = "std-surface")]
/// The `import`/`use` items whose extent the frozen compiler reads differently.
///
/// The Rust parser treats the `;` after an import path as optional; the frozen
/// compiler's `parse_use` (`examples/mindc_mind/main.mind`, `skip_to_semi`) ends
/// the item at the next `;` TOKEN, wherever that is. The two agree exactly when
/// the first non-whitespace byte at or after the Rust item's end is that `;`.
/// Anything else -- a newline then `fn main`, a comment, end of input -- makes the
/// frozen parse swallow source the Rust parse admitted as separate items.
///
/// Every `Import` node is checked, at any depth (a statement-position import is
/// consumed with the same scan). A comment between the path and its `;` is
/// refused although harmless: over-refusal is the safe direction here.
fn unterminated_imports(ast: &crate::ast::Module, text: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if let crate::ast::Node::Import { span, .. } = node {
                let end = span.end().min(text.len());
                let start = span.start().min(end);
                let next = text[end..].iter().find(|b| !b.is_ascii_whitespace());
                let terminated = text[..end].last() == Some(&b';') || next == Some(&b';');
                if !terminated {
                    out.push(
                        String::from_utf8_lossy(&text[start..end])
                            .trim()
                            .to_string(),
                    );
                }
            }
        });
    }
    out
}

#[cfg(feature = "std-surface")]
fn unterminated_import_refusal(path: &str, items: &[String]) -> String {
    format!(
        "`{path}`: import item(s) {items:?} are not `;`-terminated. The frozen native \
         compiler ends an `import`/`use` item at the next `;` token, so it would swallow \
         the items that follow (including `main`) and emit a different program than the \
         one admitted. Write the `;` (e.g. `import helper;`), or rerun with `--backend mlir`. \
         No artifact written. (native.import_not_semicolon_terminated)"
    )
}

#[cfg(feature = "std-surface")]
/// Admit the merged program: ambiguity first, then the entry-rooted closure.
///
/// Order is load-bearing. `reachable_from` walks the FIRST definition of a
/// duplicated name while the frozen compiler executes the LAST, so ambiguity has
/// to be refused before reachability is computed -- which is why this goes
/// through `admit_native_closure` rather than calling the pieces directly.
fn admit_merged_program(
    path: &str,
    merged: &[u8],
    reserved: &BTreeSet<String>,
    std_sources: &[Vec<u8>],
    lane_v1: bool,
) -> Result<(crate::ast::Module, crate::ir::IRModule, bool), String> {
    let text = std::str::from_utf8(merged)
        .map_err(|e| format!("`{path}`: source is not valid UTF-8: {e}"))?;
    let ast = match crate::parser::parse(text) {
        Ok(ast) => ast,
        Err(_) => {
            // Keep the native refusal fail-closed, but preserve the parser's
            // structured source diagnostic.  A generic merged-image message
            // loses the E1001 code, source line, and caret that the shipping
            // check path already provides for the same input.
            let diagnostics = crate::parser::parse_with_diagnostics_in_file(text, Some(path))
                .err()
                .unwrap_or_default();
            let emitter = crate::diagnostics::DiagnosticEmitter::new(
                crate::diagnostics::DiagnosticFormat::Human,
                crate::diagnostics::ColorChoice::Never,
            );
            let rendered = diagnostics
                .iter()
                .map(|diagnostic| emitter.render_human(diagnostic, Some(text)))
                .collect::<Vec<_>>()
                .join("\n");
            if rendered.is_empty() {
                return Err(format!(
                    "`{path}`: the merged module set does not parse as one translation unit"
                ));
            }
            return Err(rendered);
        }
    };
    // A lowering refusal propagates as an explicit refusal. It is never
    // unwrapped and never becomes an empty successful module.
    //
    // `lower_to_ir` is fallible: it reports a `MaterializationRefusal` where it
    // once aborted the process. That refusal is carried into THIS path's own
    // refusal channel with a stable kind, so a lowering failure reads as a
    // native-backend diagnostic like every other refusal here rather than as a
    // panic or an empty module. `InvalidLoweringReceiver` in particular is the
    // structured form of the old "refusing to emit const 0" abort.
    // The name checks the MLIR path runs before lowering, plus symbolic
    // dimensions read as values: each used to reach a lowering `panic!` (exit
    // 101). Lowering then runs behind a panic boundary for anything else.
    admission::refuse_unresolved_names(&ast, text, path)?;
    admission::refuse_symbolic_dims_read_as_values(&ast, text, path)?;
    let lane_error = |r: crate::ir::native_lane_profile::LaneProfileRejection| {
        format!(
            "`{path}` is outside candidate native-lane-v1 ({}){}. No artifact written.",
            r.construct,
            admission::lane_rejection_site(&r, text)
        )
    };
    let ir = if lane_v1 {
        // Specific guards own their pinned kinds. The lane fence still runs
        // on a program those guards accept, and it still refuses first for
        // every other construct.
        match admission::lane_guards_before_fence(path, &ast, text, merged, std_sources, reserved)?
        {
            admission::LaneLowering::Ready(ir) => {
                crate::ir::native_lane_profile::profile_lane_admits_source(&ast)
                    .map_err(lane_error)?;
                ir
            }
            admission::LaneLowering::FenceMayOwn(lower_error) => {
                crate::ir::native_lane_profile::profile_lane_admits_source(&ast)
                    .map_err(lane_error)?;
                return Err(lower_error);
            }
        }
    } else {
        admission::lower_without_panicking(&ast, path)?
    };
    if lane_v1 {
        crate::ir::native_lane_profile::profile_lane_admits(&ast, &ir).map_err(lane_error)?;
        let bindings = namespace::intrinsic_bindings(&ir);
        let omit_seed = namespace::omit_private_seed(&ast, std_sources, reserved, &bindings)
            .map_err(|r| format!("{} ({})", r.detail, r.kind))?;
        let empty = BTreeSet::new();
        crate::ir::native_closure::admit_native_lane_closure(
            &ast,
            &ir,
            "main",
            if omit_seed { &empty } else { reserved },
            &bindings,
        )
        .map_err(|r| format!("{} ({})", r.detail, r.kind))?;
        return Ok((ast, ir, omit_seed));
    }
    // Closure admission runs for its ownership, ambiguity and unresolved-edge
    // refusals, which the construct fence cannot express. Its restricted module is
    // deliberately DISCARDED: admission is whole-module, so the fence must see the
    // full lowering, not a reachability-narrowed subset.
    //
    // A seed-name collision is an ambiguity only when the two definitions
    // differ. Names whose user definition is an exact private copy of the
    // seed's private definition leave the reserved set; every other collision
    // keeps its refusal. The image itself is unchanged.
    let copies = namespace::identical_private_seed_copies(&ast, text, std_sources, reserved);
    let reserved: BTreeSet<String> = reserved.difference(&copies).cloned().collect();
    crate::ir::native_closure::admit_native_closure(&ast, &ir, "main", &reserved)
        .map_err(|r| historical_closure_refusal(&r))?;
    let _ = path;
    Ok((ast, ir, false))
}

#[cfg(feature = "std-surface")]
/// The historical closure refusal, naming the compiler selection when the
/// unresolved callee is a compiler intrinsic.
///
/// Only a compiler carrying the native-lane-v1 mark binds intrinsics. An
/// unmarked compiler gets the historical frozen profile, which binds none: the
/// closure refuses the edge here, and the frozen fence would refuse the same
/// call as `call.undefined_or_builtin`. That strictness is deliberate and this
/// adds an explanation, never a binding. The unmarked seeds' intrinsic
/// lowering is not proven in every value position. Measured on the d2a716ff
/// embedded bounded seed (sha256 `e9de4480…`): a store whose result is the tail
/// value of a helper (`deep_dirty_then_tail` in
/// `native_lane_value_shapes_smoke.py`) exits 19 natively against 17 on MLIR,
/// because the seed reads a stale stack slot. Binding intrinsics here would
/// turn this refusal into that silent wrong result.
///
/// Without the note the refusal read as a forgotten user definition, and a
/// qualification round filed it against this closure rather than against the
/// compiler selection. The kind label, the first line and the refused set are
/// unchanged. A callee outside the lane's exact intrinsic table gets no note.
fn historical_closure_refusal(r: &crate::ir::native_closure::ClosureRejection) -> String {
    let refusal = format!("{} ({})", r.detail, r.kind);
    // `reachable_from` words an unresolved edge "`caller` calls `callee`, ...";
    // identifiers carry no backtick, so the split is unambiguous.
    let intrinsic = (r.kind == "closure.unresolved_edge")
        .then(|| r.detail.split_once(" calls `"))
        .flatten()
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(callee, _)| callee)
        .filter(|callee| crate::ir::native_lane_profile::intrinsic_arity(callee).is_some());
    match intrinsic {
        None => refusal,
        Some(callee) => format!(
            "{refusal}\n  `{callee}` is a compiler intrinsic. The selected native compiler does \
             not carry the native-lane-v1 compatibility mark, so the historical frozen \
             profile applies, and it binds no compiler intrinsic. Only native-lane-v1 admits \
             intrinsic calls (docs/native-closure-lane.md). This is a refusal, not a missing \
             definition: use a native compiler that carries the mark, or `--backend mlir`. \
             No artifact written."
        ),
    }
}
