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

/// Compose and validate the captured source image, then invoke the pure-MIND
/// native compiler. Its stdout is the executable artifact without wrapper bytes.
/// `MINDC_NATIVE_ELF` overrides the committed seed; unsupported programs refuse.
#[cfg(feature = "std-surface")]
pub fn run_native_backend_bridge(paths: &[String], out: &Option<String>, emit: &Option<String>) {
    use std::io::Write as _;

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
    let std_dir = std::env::var("MINDC_STD_DIR").unwrap_or_else(|_| "std".to_string());
    let mut std_sources: Vec<Vec<u8>> = Vec::with_capacity(std_modules.len());
    for m in std_modules.iter() {
        let p = format!("{std_dir}/{m}.mind");
        match std::fs::read(&p) {
            Ok(b) => std_sources.push(b),
            Err(e) => {
                eprintln!(
                    "error[backend-native]: cannot read std module '{p}': {e} \
                     (set MINDC_STD_DIR to the std/ directory)"
                );
                process::exit(2);
            }
        }
    }

    // Composition goes through `native_image` so the single-file path and the
    // coming multi-module path build the image with the SAME code. With one
    // source the joiner contributes nothing, which is what makes "single-file
    // bytes did not move" a mechanical check instead of an argument.
    let std_blob = crate::build::native_image::compose_std_blob(&std_sources);
    let image = crate::build::native_image::compose_image(&std_blob, &user_sources);

    // Admission and emission use the same captured, composed user region.
    // The exhaustive frozen-profile allowlist checks every retained user body,
    // including uncalled functions. The current bridge does not admit calls into
    // the seed standard library: unresolved edges and seed-name collisions refuse.
    // The seed compiler's own refusal is a second fence, with no backend fallback.
    let fence_source = crate::build::native_image::compose_user_region(&user_sources);

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
    let (merged_ast, merged_ir) = match admit_merged_program(&paths[0], &fence_source, &reserved) {
        Ok(lowered) => lowered,
        Err(msg) => {
            eprintln!("error[backend-native]: {msg}");
            process::exit(3);
        }
    };
    // Whole-module admission over that single lowering. The image is not pruned,
    // so an uncalled out-of-profile body still becomes bytes in the
    // artifact; reachability informs diagnostics, it does not narrow admission.
    // enforced-by: RI-D1-PROFILE
    // The source check covers what the IR cannot carry (a declared f32 width).
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

    // The stdin image was composed above by `native_image::compose_image`.
    let image = image.bytes;

    // Resolve the pure-MIND compiler ELF (the RI-D shell-out target).
    let elf = std::env::var("MINDC_NATIVE_ELF")
        .unwrap_or_else(|_| "examples/mindc_mind/testdata/selfhost_loop/stage1.elf".to_string());
    if !std::path::Path::new(&elf).exists() {
        eprintln!(
            "error[backend-native]: pure-MIND compiler ELF not found at '{elf}' \
             (set MINDC_NATIVE_ELF)"
        );
        process::exit(2);
    }

    let mut child = match std::process::Command::new(&elf)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error[backend-native]: failed to spawn compiler ELF '{elf}': {e}");
            process::exit(2);
        }
    };
    if let Some(mut si) = child.stdin.take() {
        if let Err(e) = si.write_all(&image) {
            eprintln!("error[backend-native]: failed to stream source image to compiler: {e}");
            process::exit(2);
        }
    }
    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error[backend-native]: compiler ELF did not complete: {e}");
            process::exit(2);
        }
    };

    // Fail-closed: propagate the pure-MIND compiler's diagnostic + exit; no artifact.
    if !output.status.success() {
        let code = output.status.code().unwrap_or(1);
        if !output.stderr.is_empty() {
            let _ = std::io::stderr().write_all(&output.stderr);
        }
        eprintln!(
            "error[backend-native]: pure-MIND compiler rejected the program (exit {code}); \
             no artifact written (native backend is fail-closed — no MLIR fallback)."
        );
        process::exit(if code == 0 { 1 } else { code });
    }
    let elf_bytes = output.stdout;
    // A real static ELF (magic + ET_EXEC), never an empty/garbage artifact.
    if elf_bytes.len() < 256 || &elf_bytes[0..4] != b"\x7fELF" {
        eprintln!(
            "error[backend-native]: compiler produced a non-ELF/short artifact ({} bytes); \
             refusing to write.",
            elf_bytes.len()
        );
        process::exit(1);
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
) -> Result<(crate::ast::Module, crate::ir::IRModule), String> {
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
    let ir = crate::eval::lower_to_ir(&ast)
        .map_err(|refusal| format!("`{path}`: {refusal} (lower.materialization_refused)"))?;
    // Closure admission runs for its ownership, ambiguity and unresolved-edge
    // refusals, which the construct fence cannot express. Its restricted module is
    // deliberately DISCARDED: admission is whole-module, so the fence must see the
    // full lowering, not a reachability-narrowed subset.
    crate::ir::native_closure::admit_native_closure(&ast, &ir, "main", reserved)
        .map_err(|r| format!("{} ({})", r.detail, r.kind))?;
    let _ = path;
    Ok((ast, ir))
}
