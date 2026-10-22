// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! What to tell a user whose MLIR toolchain is missing or too old.
//!
//! A host without `mlir-opt` on `PATH` got only the generic "requires every
//! source module to compile natively" refusal, although the tool was often
//! installed under `/usr/lib/llvm-<N>/bin`. A host with MLIR 18 failed some
//! programs inside `mlir-opt` ("could not infer buffer type of block
//! argument", "cannot bufferize a FuncOp ... without a unique ReturnOp"): its
//! one-shot bufferization cannot handle a tensor carried across a loop or
//! returned from several exits, which MLIR 20 and newer can. CI tests MLIR 23
//! (latest) and MLIR 20 (oldest supported). Both now say which tool, which
//! version, and where a usable one is installed.

use std::path::PathBuf;

use crate::diagnostics::capability::FallbackReason;

/// The oldest MLIR whose bufferization handles tensors carried across loops and
/// multiple returns. CI tests 23 and 20; 20 stays the minimum.
pub const MIN_MLIR_MAJOR: u32 = 20;

/// The major version in `<tool> --version` output (`LLVM version 18.1.3`).
pub fn parse_llvm_major(version_output: &str) -> Option<u32> {
    let rest = version_output.split("version ").nth(1)?;
    rest.split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

fn llvm_major(tool: &str) -> Option<u32> {
    let out = std::process::Command::new(tool)
        .arg("--version")
        .output()
        .ok()?;
    parse_llvm_major(&String::from_utf8_lossy(&out.stdout))
}

/// `(major, bin directory)` of every `/usr/lib/llvm-<N>/bin` holding `tool`,
/// newest first.
fn installed(tool: &str) -> Vec<(u32, PathBuf)> {
    let mut found: Vec<(u32, PathBuf)> = std::fs::read_dir("/usr/lib")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let major = name.strip_prefix("llvm-")?.parse().ok()?;
            let bin = entry.path().join("bin");
            bin.join(tool).is_file().then_some((major, bin))
        })
        .collect();
    found.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    found
}

const SWITCH: &str = "put it first on PATH, or set MLIR_OPT, MLIR_TRANSLATE and CLANG";

/// What to do about `tool` missing from `PATH`.
pub fn missing_tool_hint(tool: &str) -> String {
    match installed(tool).into_iter().next() {
        Some((major, bin)) => format!(
            "`{tool}` is not on PATH; LLVM {major} has it in {}: {SWITCH}",
            bin.display()
        ),
        None => format!(
            "`{tool}` is not on PATH; install LLVM/MLIR {MIN_MLIR_MAJOR} \
             (mlir-{MIN_MLIR_MAJOR}-tools, clang-{MIN_MLIR_MAJOR}) and {SWITCH}"
        ),
    }
}

/// The note for an `mlir-opt` failure at `major` that MLIR [`MIN_MLIR_MAJOR`]
/// bufferizes, or `None` when the failure is something else.
pub fn bufferize_note(stderr: &str, major: u32, newer: Option<(u32, PathBuf)>) -> Option<String> {
    let bufferization = stderr.contains("bufferize") || stderr.contains("buffer type");
    if !bufferization || major >= MIN_MLIR_MAJOR {
        return None;
    }
    let switch = match newer {
        Some((m, bin)) => format!("LLVM {m} is installed in {}: {SWITCH}", bin.display()),
        None => format!("install mlir-{MIN_MLIR_MAJOR}-tools and clang-{MIN_MLIR_MAJOR}"),
    };
    Some(format!(
        "note: this is MLIR {major}, which cannot bufferize a tensor carried across a loop \
         or returned from several exits; MLIR {MIN_MLIR_MAJOR} or newer can. {switch}"
    ))
}

/// `stderr` of a failed run of the `mlir-opt` at `path`, with [`bufferize_note`]
/// appended when it applies.
pub fn with_bufferize_hint(path: &str, mut stderr: String) -> String {
    let newer = installed("mlir-opt")
        .into_iter()
        .find(|(m, _)| *m >= MIN_MLIR_MAJOR);
    if let Some(note) = llvm_major(path).and_then(|major| bufferize_note(&stderr, major, newer)) {
        stderr.push('\n');
        stderr.push_str(&note);
    }
    stderr
}

/// [`llvm_major`] of `tool`, probed once per process and path: a project build
/// compiles many modules, and the answer cannot change under a running build.
fn cached_major(tool: &str) -> Option<u32> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashMap<String, Option<u32>>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(major) = seen.lock().ok().and_then(|m| m.get(tool).copied()) {
        return major;
    }
    let major = llvm_major(tool);
    if let Ok(mut m) = seen.lock() {
        m.insert(tool.to_string(), major);
    }
    major
}

/// The E5004 explanation for a tensor-pipeline failure on the `mlir-opt` at
/// `path`, of LLVM `major` (`None` when its `--version` states none), or
/// `None` when that `mlir-opt` is new enough. `newer` is an installed
/// `mlir-opt` that would do.
pub fn old_mlir_opt_note(
    path: &str,
    major: Option<u32>,
    newer: Option<(u32, PathBuf)>,
) -> Option<String> {
    if major.is_some_and(|m| m >= MIN_MLIR_MAJOR) {
        return None;
    }
    let switch = match newer {
        Some((m, bin)) => format!("LLVM {m} is installed in {}: {SWITCH}", bin.display()),
        None => {
            format!("install mlir-{MIN_MLIR_MAJOR}-tools and clang-{MIN_MLIR_MAJOR}, then {SWITCH}")
        }
    };
    Some(format!(
        "[{code}] the tensor pipeline (one-shot-bufferize) needs MLIR/LLVM {MIN_MLIR_MAJOR} \
         or newer; the mlir-opt in use ({path}) is {}. {switch}",
        found(major),
        code = crate::diagnostics::capability::NATIVE_TOOLCHAIN_ABSENT
    ))
}

fn found(major: Option<u32>) -> String {
    match major {
        Some(m) => format!("LLVM {m}"),
        None => "a version its `--version` output does not state".to_string(),
    }
}

/// The one-line warning for running the tensor pipeline on an `mlir-opt`
/// older than [`MIN_MLIR_MAJOR`]. Such a build is NOT refused: MLIR is an
/// optional backend and many tensor programs build correctly on LLVM 18.
pub fn old_mlir_opt_warning(path: &str, major: Option<u32>) -> String {
    format!(
        "warning[build]: mlir-opt {path} is {}; the tensor pipeline (one-shot-bufferize) \
         is supported on MLIR/LLVM {MIN_MLIR_MAJOR} or newer, building anyway",
        found(major)
    )
}

/// For a tensor-pipeline run on the `mlir-opt` at `path`: `None` when it is new
/// enough, else the [`old_mlir_opt_note`] to append if `mlir-opt` then FAILS.
/// The first call per path in a process also prints [`old_mlir_opt_warning`]
/// (once: a project build runs the pipeline for every module).
pub fn tensor_pipeline_note(path: &str) -> Option<String> {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let major = cached_major(path);
    old_mlir_opt_note(path, major, None)?;
    let first = WARNED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut w| w.insert(path.to_string()))
        .unwrap_or(false);
    if first {
        eprintln!("{}", old_mlir_opt_warning(path, major));
    }
    let newer = installed("mlir-opt")
        .into_iter()
        .find(|(m, _)| *m >= MIN_MLIR_MAJOR);
    old_mlir_opt_note(path, major, newer)
}

/// The toolchain a native build ran with — each tool's resolved path and LLVM
/// major — as one line, e.g.
/// `mlir-opt /usr/lib/llvm-20/bin/mlir-opt (LLVM 20); ...`.
#[cfg(feature = "mlir-subprocess")]
pub fn describe_toolchain(tools: &[(&str, &str)]) -> String {
    tools
        .iter()
        .map(|(name, tool)| {
            let path = which::which(tool)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| tool.to_string());
            match cached_major(tool) {
                Some(m) => format!("{name} {path} (LLVM {m})"),
                None => format!("{name} {path} (LLVM version unknown)"),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The refusal for a CPU executable one of whose modules fell back to the
/// runtime-JIT object, naming the missing tool when that was the `cause`.
pub fn fallback_link_refusal(cause: FallbackReason) -> String {
    let message = "public CPU executable requires every source module to compile natively; \
                   refusing to link a runtime-JIT fallback object";
    match missing_tool(cause) {
        Some(tool) => format!("{message} ({})", missing_tool_hint(tool)),
        None => message.to_string(),
    }
}

/// The tool a [`FallbackReason::NativeToolchainAbsent`] fallback is missing.
fn missing_tool(cause: FallbackReason) -> Option<&'static str> {
    #[cfg(feature = "mlir-build")]
    if cause == FallbackReason::NativeToolchainAbsent {
        if let Err(crate::eval::mlir_build::BuildError::ToolMissing(tool)) =
            crate::eval::mlir_build::resolve_tools()
        {
            return Some(tool);
        }
    }
    let _ = cause;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_major_version() {
        let out = "Ubuntu LLVM version 18.1.3\n  Optimized build.\n";
        assert_eq!(parse_llvm_major(out), Some(18));
        assert_eq!(
            parse_llvm_major("LLVM (http://llvm.org/):\n  LLVM version 20.1.2\n"),
            Some(20)
        );
        assert_eq!(parse_llvm_major("no version here"), None);
    }

    #[test]
    fn notes_only_a_bufferization_failure_below_the_minimum() {
        let err = "error: 'func.func' op could not infer buffer type of block argument";
        let note = bufferize_note(err, 18, None).expect("a note on MLIR 18");
        assert!(
            note.contains("MLIR 18") && note.contains("mlir-20-tools"),
            "{note}"
        );
        let newer = Some((20, PathBuf::from("/usr/lib/llvm-20/bin")));
        assert!(
            bufferize_note(err, 18, newer)
                .expect("note")
                .contains("/usr/lib/llvm-20/bin")
        );
        assert_eq!(bufferize_note(err, 20, None), None, "MLIR 20 needs no note");
        assert_eq!(bufferize_note("error: unknown op", 18, None), None);
    }

    #[test]
    fn notes_and_warns_below_the_minimum_and_names_both_versions() {
        let msg = old_mlir_opt_note("/usr/bin/mlir-opt", Some(18), None).expect("noted");
        assert!(msg.starts_with("[E5004] "), "{msg}");
        assert!(msg.contains("needs MLIR/LLVM 20 or newer"), "{msg}");
        assert!(msg.contains("(/usr/bin/mlir-opt) is LLVM 18"), "{msg}");
        assert!(msg.contains("mlir-20-tools"), "{msg}");
        let newer = Some((20, PathBuf::from("/usr/lib/llvm-20/bin")));
        let msg = old_mlir_opt_note("mlir-opt", Some(18), newer).expect("noted");
        assert!(
            msg.contains("LLVM 20 is installed in /usr/lib/llvm-20/bin"),
            "{msg}"
        );
        assert!(
            old_mlir_opt_note("x", None, None)
                .expect("unknown noted")
                .contains("does not state")
        );
        assert_eq!(old_mlir_opt_note("x", Some(20), None), None);
        assert_eq!(old_mlir_opt_note("x", Some(21), None), None);
        let warning = old_mlir_opt_warning("/usr/bin/mlir-opt", Some(18));
        assert!(!warning.contains('\n'), "one line: {warning}");
        assert!(warning.starts_with("warning[build]: "), "{warning}");
        assert!(warning.contains("is LLVM 18") && warning.contains("building anyway"));
    }
}
