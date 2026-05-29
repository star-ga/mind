// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the “License”);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an “AS IS” BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! MIND command-line compiler: parse, type-check, lower to IR/MLIR, and
//! optionally run autodiff.

use std::fs;
use std::process;

use clap::{ArgAction, Parser, Subcommand};

use libmind::build::{BuildOpts, run_build};
use libmind::check::{CheckOptions, ReporterKind, run_check};
use libmind::deps::{CleanOpts, FetchOpts, LockOpts, run_clean, run_fetch, run_lock};
use libmind::doc::{DocOptions, run_doc};
use libmind::fmt::cli as mindc_fmt;
use libmind::test::{ReporterKind as TestReporterKind, TestOptions as MindTestOptions, run_tests};
use libmind::workspace::{WorkspaceOpts, resolve_workspace_members, toposort_members};

use libmind::BackendTarget;
use libmind::diagnostics::{ColorChoice, DiagnosticEmitter, DiagnosticFormat};
use libmind::ops::core_v1;
use libmind::pipeline::{CompileOptions, compile_source_with_name};
use libmind::project::{
    BenchOptions, BuildOptions, BuildTarget, EmitKind, OptimizeLevel, bench_project, run_project,
};
use libmind::{ConformanceOptions, ConformanceProfile, conformance};

#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
use libmind::pipeline::{MlirProducts, lower_to_mlir};

#[cfg(feature = "mlir-build")]
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    author,
    about = None,
    long_about = None,
    disable_version_flag = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    compile: CompileArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a MIND project (reads Mind.toml).
    ///
    /// RFC 0008 Phase A — single-crate orchestrator.
    /// Reads `[build]` from Mind.toml; CLI flags override the manifest.
    Build {
        /// Source files to compile.  When omitted, uses `[build].entry` or
        /// auto-detects src/main.mind / src/lib.mind.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Build in release mode (equivalent to --optimize=release).
        #[arg(long)]
        release: bool,
        /// Target backend (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras).
        /// Overrides `[build].target` in Mind.toml.
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Output artifact type: binary | cdylib | object.
        /// Overrides `[build].emit` in Mind.toml.
        #[arg(long, value_name = "EMIT")]
        emit: Option<String>,
        /// Optimization level: debug | release | size.
        /// Overrides `[build].optimize` in Mind.toml. --release is shorthand.
        #[arg(long, value_name = "LEVEL", conflicts_with = "release")]
        optimize: Option<String>,
        /// Custom output path.  Overrides the default `target/<profile>/<name>`.
        #[arg(long, value_name = "PATH")]
        out: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Build only the named workspace member (and its prerequisites).
        /// Alias: -p.  RFC 0008 Phase C.
        #[arg(long, short = 'p', value_name = "NAME")]
        package: Option<String>,
        /// Explicitly build all workspace members (no-op when at workspace root;
        /// included for parity with cargo).
        #[arg(long)]
        workspace: bool,
        /// Bypass the incremental object cache for this build (RFC 0008 Phase F).
        ///
        /// New objects are still written to cache so subsequent runs benefit.
        /// Use this when you suspect a stale cache entry.
        #[arg(long)]
        no_cache: bool,
    },
    /// Build and run a MIND project.
    Run {
        /// Build in release mode with optimizations.
        #[arg(long)]
        release: bool,
        /// Target backend (cpu, cuda, cuda-ampere, rocm, metal, webgpu, etc.).
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Arguments to pass to the program (after --).
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Run tests marked with `[test]` in MIND source files (RFC 0008 Phase B).
    ///
    /// Discovers all `[test]`-annotated functions in the specified paths (or
    /// the current directory when none are given), compiles and runs each as an
    /// isolated test case, and reports pass/fail in cargo-test–compatible output.
    ///
    /// Exit code 0 = all tests passed.  Exit code 1 = one or more failed.
    Test {
        /// Source files or directories to search for `[test]` functions.
        /// When omitted, walks the current directory for *.mind files.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Run only tests whose name contains this substring.
        #[arg(long, value_name = "SUBSTR")]
        filter: Option<String>,
        /// Do not capture test stdout/stderr; print it immediately.
        #[arg(long)]
        no_capture: bool,
        /// Maximum parallel worker threads (0 = use available parallelism).
        #[arg(long, value_name = "N", default_value = "0")]
        threads: usize,
        /// List test names and exit without running any tests.
        #[arg(long)]
        list: bool,
        /// Diagnostic reporter: human (default) or json.
        #[arg(long, value_name = "REPORTER", default_value = "human",
              value_parser = ["human", "json"])]
        reporter: String,
        /// Run tests for only the named workspace member (and its prerequisites).
        /// Alias: -p.  RFC 0008 Phase C.
        #[arg(long, short = 'p', value_name = "NAME")]
        package: Option<String>,
    },
    /// Run project benchmarks (bench/*.mind).
    Bench {
        /// Target backend (cpu, cuda, etc.).
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
        /// Filter benchmarks by name.
        #[arg(long, value_name = "PATTERN")]
        filter: Option<String>,
        /// Number of iterations.
        #[arg(long, value_name = "N")]
        iterations: Option<u32>,
        /// Output results as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run the Core v1 conformance suite.
    Conformance {
        /// Which profile to execute (cpu|gpu).
        #[arg(long, default_value = "cpu")]
        profile: String,
    },
    /// Run format-check + lint + type-check over MIND source files.
    ///
    /// Exit code 0 = all passes clean; 1 = one or more error-severity
    /// diagnostics detected.
    Check {
        /// Files or directories to check.  Directories are walked recursively
        /// for *.mind files.  Defaults to the current directory when omitted.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Diagnostic reporter: human (default), json, or lsp.
        ///
        /// `lsp` emits LSP-compatible Diagnostic JSON objects (RFC 0007 §C).
        #[arg(long, value_name = "REPORTER", default_value = "human",
              value_parser = ["human", "json", "lsp"])]
        reporter: String,
        /// Skip the format-check pass.
        #[arg(long)]
        no_fmt: bool,
        /// Skip the lint pass.
        #[arg(long)]
        no_lint: bool,
        /// Skip the type-check pass.
        #[arg(long)]
        no_typecheck: bool,
        /// Apply machine-applicable fixes and rewrite files.
        ///
        /// For every fmt::drift diagnostic, writes the formatted file.
        /// For every lint rule with an auto-fix, applies the byte-range edit.
        /// Iterates up to 5 rounds; warns if convergence is not reached.
        /// Prints: "Fixed N files, M unfixable diagnostics remaining."
        #[arg(long)]
        fix: bool,
    },
    /// Format MIND source files (or directories of *.mind files).
    Fmt {
        /// Files or directories to format. Directories are walked recursively
        /// for *.mind files. Defaults to the current directory when omitted.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Check whether files are already formatted; exit 1 if any would
        /// change. No files are written.
        #[arg(long)]
        check: bool,
        /// Print a unified diff between the original and formatted source;
        /// exit 1 if any file would change. No files are written.
        #[arg(long)]
        diff: bool,
        /// Read source from stdin and write the formatted result to stdout.
        /// Cannot be combined with positional PATHS.
        #[arg(long)]
        stdin: bool,
        /// Explicitly format files in-place (same as the default write mode)
        /// and print a summary: "Formatted N files, M unchanged."
        #[arg(long)]
        fix: bool,
    },
    /// Inspect compiler knowledge about Core profiles.
    Ops {
        /// Show the Core v1 operator catalog.
        #[arg(long, default_value_t = true, action = ArgAction::SetTrue)]
        core_v1: bool,
    },
    /// Regenerate Mind.lock from the current Mind.toml (RFC 0008 Phase E).
    ///
    /// Resolves all path and git dependencies, fetches git deps if needed,
    /// and writes a fully pinned Mind.lock. Mandatory before `mindc build`.
    Lock {
        /// Only verify — do not write Mind.lock; exit 1 if stale.
        #[arg(long)]
        check: bool,
        /// Re-resolve only the named package (update its entry in Mind.lock).
        #[arg(long, value_name = "PKG")]
        update: Option<String>,
    },
    /// Populate ~/.mindenv/cache/ from Mind.lock (RFC 0008 Phase E).
    ///
    /// Idempotent: already-cached deps are not re-fetched unless --update is given.
    Fetch {
        /// Re-fetch all git deps even if already cached. Does NOT modify Mind.lock.
        #[arg(long)]
        update: bool,
    },
    /// Generate HTML documentation from `///` doc-comments in MIND source files.
    ///
    /// Walks *.mind files, extracts `pub` items and their preceding `///`
    /// doc-comment blocks, and renders one HTML page per source file plus a
    /// top-level `index.html` and `search-index.json`.
    ///
    /// Exit code 0 = success, 1 = parse or I/O error, 2 = invalid CLI args.
    Doc {
        /// Source files or directories to document.  Directories are walked
        /// recursively for *.mind files.  Defaults to the current directory.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Output directory for generated HTML (default: `./target/doc`).
        #[arg(long, value_name = "DIR", default_value = "target/doc")]
        out: String,
        /// Do not render dependency files; only document the given paths.
        #[arg(long)]
        no_deps: bool,
        /// Open the generated `index.html` in a browser after rendering.
        #[arg(long)]
        open: bool,
    },
    /// Remove build artifacts and/or the dependency cache (RFC 0008 Phase E).
    Clean {
        /// Wipe ~/.mindenv/cache/ entries for this project's deps.
        #[arg(long)]
        cache: bool,
        /// Wipe both target/ and the entire ~/.mindenv/cache/.
        #[arg(long)]
        all: bool,
    },
}

#[derive(Parser, Debug, Default)]
struct CompileArgs {
    /// Print the compiler version and component stability versions.
    #[arg(long, action = ArgAction::SetTrue)]
    version: bool,
    /// Print a short description of the public stability model.
    #[arg(long, action = ArgAction::SetTrue)]
    stability: bool,
    /// Input .mind file to compile.
    #[arg(value_name = "FILE")]
    input: Option<String>,
    /// Emit canonical IR for the module.
    #[arg(long)]
    emit_ir: bool,
    /// Emit MIC (compact serializable IR) for the module.
    #[arg(long)]
    emit_mic: bool,
    /// Emit gradient IR for the selected function (requires --autodiff).
    #[arg(long)]
    emit_grad_ir: bool,
    /// Emit MLIR text for the canonical IR (requires feature mlir-lowering).
    #[arg(long)]
    emit_mlir: bool,
    /// Focus on a specific function (used for autodiff and MLIR).
    #[arg(long, value_name = "NAME")]
    func: Option<String>,
    /// Run autodiff for the selected function and expose the gradient IR/MLIR.
    #[arg(long)]
    autodiff: bool,
    /// Only verify the pipeline without emitting artifacts.
    #[arg(long)]
    verify_only: bool,
    /// Emit MIC@3 binary artifact to the specified path (RFC 0021 step 3).
    ///
    /// Writes the binary mic@3 encoding of the compiled IR module.  The output
    /// is identical to calling `compact::v3::emit_mic3` on the compiled IR.
    #[arg(long, value_name = "PATH")]
    emit_mic3: Option<String>,
    /// Emit MIC@3 binary artifact with RFC 0021 evidence MAP to the specified path.
    ///
    /// Equivalent to `--emit-mic3` plus an appended `evidence_chain.*` MAP
    /// epilogue containing substrate, toolchain, determinism declaration, and
    /// a SHA-256 trace hash of the canonical IR.  Use `mic3_evidence_report`
    /// to verify the artifact offline.
    #[arg(long, value_name = "PATH")]
    emit_evidence: Option<String>,
    /// Emit object file (.o) to the specified path.
    #[arg(long, value_name = "PATH")]
    emit_obj: Option<String>,
    /// Emit a shared library (`.so` on Linux, `.dylib` on macOS) to the
    /// specified path. Equivalent to `--emit-obj` followed by a shared-
    /// library link. Phase 10.8 / mindc 0.3.0 cdylib-emit foundation.
    /// Requires the `mlir-build` feature.
    #[arg(long, value_name = "PATH")]
    emit_shared: Option<String>,
    /// Select the execution target backend (cpu|gpu).
    #[arg(long, value_name = "TARGET", default_value = "cpu")]
    target: String,
    /// Language profile (default|systems|embedded). RFC 0002 deliverable 5:
    /// the same Mind.toml produces a distinct artifact per profile via the
    /// cache fingerprint, so cross-mode rebuilds never hit a stale entry.
    /// Strict on the CLI surface: unknown values are rejected by clap
    /// before reaching `ProfileTag::parse`'s permissive fallback.
    #[arg(
        long,
        value_name = "PROFILE",
        default_value = "default",
        value_parser = ["default", "systems", "embedded"],
    )]
    profile: String,
    /// Diagnostic output format (human|short|json).
    #[arg(long, value_name = "FORMAT", default_value = "human")]
    diagnostic_format: String,
    /// ANSI color handling (auto|always|never).
    #[arg(long, value_name = "WHEN")]
    color: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Command::Build {
            paths,
            release,
            target,
            emit,
            optimize,
            out,
            verbose,
            package,
            workspace: _,
            no_cache,
        }) => {
            run_mindc_build(
                paths,
                *release,
                target,
                emit,
                optimize,
                out,
                *verbose,
                package.as_deref(),
                *no_cache,
            );
            return;
        }
        Some(Command::Run {
            release,
            target,
            verbose,
            args,
        }) => {
            run_run_command(*release, target.clone(), *verbose, args.clone());
            return;
        }
        Some(Command::Test {
            paths,
            filter,
            no_capture: _,
            threads,
            list,
            reporter,
            package,
        }) => {
            run_mindc_test(
                paths,
                filter.as_deref(),
                *threads,
                *list,
                reporter,
                package.as_deref(),
            );
            return;
        }
        Some(Command::Bench {
            target,
            verbose,
            filter,
            iterations,
            json,
        }) => {
            let opts = BenchOptions {
                target: target.clone(),
                verbose: *verbose,
                filter: filter.clone(),
                iterations: *iterations,
                json: *json,
            };
            match bench_project(&opts) {
                Ok(code) => process::exit(code),
                Err(err) => {
                    eprintln!("error: {}", err);
                    process::exit(1);
                }
            }
        }
        Some(Command::Conformance { profile }) => {
            run_conformance(profile);
            return;
        }
        Some(Command::Check {
            paths,
            reporter,
            no_fmt,
            no_lint,
            no_typecheck,
            fix,
        }) => {
            let reporter_kind = match reporter.as_str() {
                "json" => ReporterKind::Json,
                "lsp" => ReporterKind::Lsp,
                _ => ReporterKind::Human,
            };
            let opts = CheckOptions {
                run_fmt: !no_fmt,
                run_lint: !no_lint,
                run_typecheck: !no_typecheck,
                reporter: reporter_kind,
                paths: paths.clone(),
                fix: *fix,
            };
            process::exit(run_check(&opts));
        }
        Some(Command::Fmt {
            paths,
            check,
            diff,
            stdin,
            fix,
        }) => {
            process::exit(mindc_fmt::run_fmt(paths, *check, *diff, *stdin, *fix));
        }
        Some(Command::Doc {
            paths,
            out,
            no_deps,
            open,
        }) => {
            let opts = DocOptions {
                paths: paths.clone(),
                out_dir: std::path::PathBuf::from(out),
                no_deps: *no_deps,
                open: *open,
            };
            process::exit(run_doc(&opts));
        }
        Some(Command::Ops { .. }) => {
            print_ops(&cli.command);
            return;
        }
        Some(Command::Lock { check, update }) => {
            run_mindc_lock(*check, update.as_deref());
            return;
        }
        Some(Command::Fetch { update }) => {
            run_mindc_fetch(*update);
            return;
        }
        Some(Command::Clean { cache, all }) => {
            run_mindc_clean(*cache, *all);
            return;
        }
        None => {}
    }

    if cli.compile.version {
        print_version();
        return;
    }

    if cli.compile.stability {
        print_stability();
        return;
    }

    let input = match &cli.compile.input {
        Some(path) => path.clone(),
        None => {
            eprintln!("error[cli]: expected an input file or subcommand");
            process::exit(1);
        }
    };

    if cli.compile.autodiff && cli.compile.func.is_none() {
        eprintln!("error[autodiff]: --autodiff requires --func <name>");
        process::exit(1);
    }

    let target = match parse_target(&cli.compile.target) {
        Ok(target) => target,
        Err(msg) => {
            eprintln!("error[backend]: {msg}");
            process::exit(1);
        }
    };

    let diagnostic_format =
        DiagnosticFormat::parse(&cli.compile.diagnostic_format).unwrap_or(DiagnosticFormat::Human);
    let color_choice = resolve_color_choice(&cli.compile.color);
    let emitter = DiagnosticEmitter::new(diagnostic_format, color_choice);

    let source = match fs::read_to_string(&input) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("failed to read {}: {err}", input);
            process::exit(1);
        }
    };

    let opts = CompileOptions {
        func: cli.compile.func.clone(),
        enable_autodiff: cli.compile.autodiff,
        target,
        profile: libmind::cache::ProfileTag::parse(&cli.compile.profile),
        ..Default::default()
    };

    let products = match compile_source_with_name(&source, Some(&input), &opts) {
        Ok(products) => products,
        Err(err) => {
            let diags = err.into_diagnostics(Some(&input));
            emitter.emit_all(&diags, Some(&source));
            process::exit(1);
        }
    };

    if cli.compile.verify_only {
        return;
    }

    let emit_ir = cli.compile.emit_ir
        || (!cli.compile.emit_grad_ir
            && !cli.compile.emit_mlir
            && !cli.compile.emit_mic
            && cli.compile.emit_mic3.is_none()
            && cli.compile.emit_evidence.is_none());
    if emit_ir {
        println!("{}", products.ir);
    }

    if cli.compile.emit_mic {
        let mic = libmind::ir::compact::emit_mic(&products.ir);
        println!("{}", mic);
    }

    emit_mic3_if_requested(&cli.compile, &products);
    emit_evidence_if_requested(&cli.compile, &products);

    #[cfg(feature = "autodiff")]
    if cli.compile.autodiff && cli.compile.emit_grad_ir {
        match products.grad.as_ref() {
            Some(grad) => println!("{}", grad.gradient_module),
            None => {
                eprintln!("autodiff did not produce gradient IR");
                process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "autodiff"))]
    if cli.compile.autodiff && cli.compile.emit_grad_ir {
        eprintln!("gradient IR emission requires building with the 'autodiff' feature");
        process::exit(1);
    }

    emit_mlir_if_requested(&cli.compile, &products);
    emit_obj_if_requested(&cli.compile, &products);
    emit_shared_if_requested(&cli.compile, &products);
}

fn run_mindc_build(
    paths: &[String],
    release: bool,
    target: &Option<String>,
    emit: &Option<String>,
    optimize: &Option<String>,
    out: &Option<String>,
    verbose: bool,
    package: Option<&str>,
    no_cache: bool,
) {
    // Workspace detection: if we are at a workspace root and no explicit
    // source paths are given, delegate to the workspace build path.
    if paths.is_empty() {
        if let Some(root) = detect_workspace_root() {
            run_workspace_build(
                &root, release, target, emit, optimize, out, verbose, package,
            );
            return;
        }
    }

    // Parse --target override.
    let eff_target: Option<BuildTarget> = match target {
        None => None,
        Some(t) => match BuildTarget::parse(t) {
            Ok(bt) => Some(bt),
            Err(msg) => {
                eprintln!("error[build]: {}", msg);
                process::exit(2);
            }
        },
    };

    // Parse --emit override.
    let eff_emit: Option<EmitKind> = match emit {
        None => None,
        Some(e) => match EmitKind::parse(e) {
            Ok(ek) => Some(ek),
            Err(msg) => {
                eprintln!("error[build]: {}", msg);
                process::exit(2);
            }
        },
    };

    // --release is shorthand for --optimize=release.
    let eff_optimize: Option<OptimizeLevel> = if release {
        Some(OptimizeLevel::Release)
    } else {
        match optimize {
            None => None,
            Some(o) => match OptimizeLevel::parse(o) {
                Ok(ol) => Some(ol),
                Err(msg) => {
                    eprintln!("error[build]: {}", msg);
                    process::exit(2);
                }
            },
        }
    };

    let opts = BuildOpts {
        paths: paths.iter().map(std::path::PathBuf::from).collect(),
        target: eff_target,
        emit: eff_emit,
        optimize: eff_optimize,
        out: out.as_ref().map(std::path::PathBuf::from),
        verbose,
        no_cache,
    };

    match run_build(&opts) {
        Ok(output) => {
            println!(
                "   Finished {} [{}] {}",
                output.target,
                output.emit.as_str(),
                output.artifact_path.display()
            );
            println!("   Artifact: {} bytes", output.byte_count);
        }
        Err(err) => {
            eprintln!("error[build]: {}", err);
            process::exit(err.exit_code());
        }
    }
}

fn run_mindc_test(
    paths: &[String],
    filter: Option<&str>,
    threads: usize,
    list: bool,
    reporter: &str,
    package: Option<&str>,
) {
    // Workspace detection: if invoked in a workspace root with no explicit
    // source paths, run tests for all members (or the named member).
    if paths.is_empty() {
        if let Some(root) = detect_workspace_root() {
            run_workspace_test(&root, filter, threads, list, reporter, package);
            return;
        }
    }

    let reporter_kind = if reporter == "json" {
        TestReporterKind::Json
    } else {
        TestReporterKind::Human
    };

    let opts = MindTestOptions {
        paths: paths.iter().map(std::path::PathBuf::from).collect(),
        filter: filter.unwrap_or("").to_string(),
        capture: true,
        threads,
        list,
        reporter: reporter_kind,
    };

    match run_tests(&opts) {
        Ok(summary) => {
            if summary.all_passed() {
                process::exit(0);
            } else {
                process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("error[test]: {}", err);
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// RFC 0008 Phase C — workspace dispatch helpers
// ---------------------------------------------------------------------------

/// Detect whether the current working directory (or a parent) is a workspace
/// root (has a `Mind.toml` with a `[workspace]` block).
///
/// Returns `Some(root)` when a workspace root is found, `None` otherwise.
fn detect_workspace_root() -> Option<std::path::PathBuf> {
    use libmind::project::find_project_root;
    let root = find_project_root().ok()?;
    let text = std::fs::read_to_string(root.join("Mind.toml")).ok()?;
    if text.contains("[workspace]") {
        Some(root)
    } else {
        None
    }
}

/// Build all workspace members (or a filtered subset) in topological order.
fn run_workspace_build(
    workspace_root: &std::path::Path,
    release: bool,
    target: &Option<String>,
    emit: &Option<String>,
    optimize: &Option<String>,
    out: &Option<String>,
    verbose: bool,
    package: Option<&str>,
) {
    let members = match resolve_workspace_members(workspace_root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let sorted = match toposort_members(&members) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let ws_opts = WorkspaceOpts {
        package_filter: package.map(|s| s.to_string()),
    };
    let selected = ws_opts.filter_members(&members, &sorted);

    if selected.is_empty() && package.is_some() {
        eprintln!(
            "error[workspace]: package '{}' not found in workspace",
            package.unwrap()
        );
        process::exit(2);
    }

    let mut any_failed = false;
    for member in &selected {
        if verbose {
            eprintln!("   Building workspace member: {}", member.name);
        }
        // Change into the member directory and delegate to the single-crate
        // builder by temporarily pushing the manifest path.
        let member_paths: Vec<String> = vec![];
        let eff_out: Option<String> = if out.is_some() && selected.len() == 1 {
            out.clone()
        } else {
            None // each member uses its own default output path
        };
        let member_out = std::env::current_dir().ok().map(|_| eff_out).flatten();

        // Run the Phase A build for this member's root.
        let mut build_opts = BuildOpts {
            paths: member_paths.iter().map(std::path::PathBuf::from).collect(),
            target: parse_target_opt(target),
            emit: parse_emit_opt(emit),
            optimize: parse_optimize_opt(release, optimize),
            out: member_out.map(std::path::PathBuf::from),
            verbose,
            no_cache: false,
        };
        // Override paths to use the member root's entry point resolution.
        // The member root is passed via a synthetic path pointing to the member.
        build_opts.paths = vec![member.root.clone()];

        // Temporarily change working directory to the member root so that
        // find_project_root() inside run_build picks up the member's Mind.toml.
        let saved_dir = std::env::current_dir().unwrap_or_else(|_| workspace_root.to_path_buf());
        if std::env::set_current_dir(&member.root).is_ok() {
            build_opts.paths = vec![];
        }

        match run_build(&build_opts) {
            Ok(output) => {
                println!(
                    "   Finished {} ({}) [{}] {}",
                    member.name,
                    output.target,
                    output.emit.as_str(),
                    output.artifact_path.display()
                );
            }
            Err(err) => {
                eprintln!("error[workspace][{}]: {}", member.name, err);
                any_failed = true;
            }
        }

        // Restore working directory.
        let _ = std::env::set_current_dir(&saved_dir);
    }

    if any_failed {
        process::exit(1);
    }
}

fn parse_target_opt(target: &Option<String>) -> Option<BuildTarget> {
    match target {
        None => None,
        Some(t) => match BuildTarget::parse(t) {
            Ok(bt) => Some(bt),
            Err(msg) => {
                eprintln!("error[build]: {msg}");
                process::exit(2);
            }
        },
    }
}

fn parse_emit_opt(emit: &Option<String>) -> Option<EmitKind> {
    match emit {
        None => None,
        Some(e) => match EmitKind::parse(e) {
            Ok(ek) => Some(ek),
            Err(msg) => {
                eprintln!("error[build]: {msg}");
                process::exit(2);
            }
        },
    }
}

fn parse_optimize_opt(release: bool, optimize: &Option<String>) -> Option<OptimizeLevel> {
    if release {
        Some(OptimizeLevel::Release)
    } else {
        match optimize {
            None => None,
            Some(o) => match OptimizeLevel::parse(o) {
                Ok(ol) => Some(ol),
                Err(msg) => {
                    eprintln!("error[build]: {msg}");
                    process::exit(2);
                }
            },
        }
    }
}

/// Run tests for all workspace members (or a filtered subset).
fn run_workspace_test(
    workspace_root: &std::path::Path,
    filter: Option<&str>,
    threads: usize,
    list: bool,
    reporter: &str,
    package: Option<&str>,
) {
    let members = match resolve_workspace_members(workspace_root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let sorted = match toposort_members(&members) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error[workspace]: {e}");
            process::exit(e.exit_code());
        }
    };

    let ws_opts = WorkspaceOpts {
        package_filter: package.map(|s| s.to_string()),
    };
    let selected = ws_opts.filter_members(&members, &sorted);

    if selected.is_empty() && package.is_some() {
        eprintln!(
            "error[workspace]: package '{}' not found in workspace",
            package.unwrap()
        );
        process::exit(2);
    }

    let reporter_kind = if reporter == "json" {
        TestReporterKind::Json
    } else {
        TestReporterKind::Human
    };

    let mut any_failed = false;
    let saved_dir = std::env::current_dir().unwrap_or_else(|_| workspace_root.to_path_buf());

    for member in &selected {
        if std::env::set_current_dir(&member.root).is_err() {
            eprintln!(
                "error[workspace]: cannot enter member directory: {}",
                member.root.display()
            );
            any_failed = true;
            continue;
        }

        let opts = MindTestOptions {
            paths: vec![],
            filter: filter.unwrap_or("").to_string(),
            capture: true,
            threads,
            list,
            reporter: reporter_kind.clone(),
        };

        match run_tests(&opts) {
            Ok(summary) => {
                if !summary.all_passed() {
                    any_failed = true;
                }
            }
            Err(err) => {
                eprintln!("error[test][{}]: {}", member.name, err);
                any_failed = true;
            }
        }

        let _ = std::env::set_current_dir(&saved_dir);
    }

    if any_failed {
        process::exit(1);
    }
}

fn run_run_command(release: bool, target: Option<String>, verbose: bool, args: Vec<String>) {
    let opts = BuildOptions {
        release,
        target,
        verbose,
        ..Default::default()
    };

    match run_project(&args, &opts) {
        Ok(code) => {
            process::exit(code);
        }
        Err(err) => {
            eprintln!("error: {}", err);
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// RFC 0008 Phase D + E — lock / fetch / clean handlers
// ---------------------------------------------------------------------------

fn run_mindc_lock(check: bool, update_pkg: Option<&str>) {
    use libmind::project::{find_project_root, load_manifest};
    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[lock]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };
    let manifest = match load_manifest(&root) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error[lock]: {e}");
            process::exit(1);
        }
    };
    let opts = LockOpts {
        check,
        update_pkg: update_pkg.map(|s| s.to_string()),
    };
    match run_lock(&root, &manifest, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[lock]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn run_mindc_fetch(update: bool) {
    use libmind::project::find_project_root;
    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[fetch]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };
    let opts = FetchOpts { update };
    match run_fetch(&root, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[fetch]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn run_mindc_clean(cache: bool, all: bool) {
    use libmind::build::cache::clean_all_caches;
    use libmind::project::find_project_root;

    let root = match find_project_root() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error[clean]: cannot find Mind.toml: {e}");
            process::exit(1);
        }
    };

    // Phase F: --cache wipes the incremental build object cache (.cache/ dirs
    // under target/), leaving the previously linked binaries intact.
    if cache && !all {
        match clean_all_caches(&root) {
            Ok(()) => println!("   Removed incremental cache (target/*/.cache/)."),
            Err(e) => {
                eprintln!("error[clean]: {e}");
                process::exit(1);
            }
        }
        // Also clean the deps git cache via the deps subsystem.
        let opts = CleanOpts {
            cache: true,
            all: false,
        };
        match run_clean(&root, &opts) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("error[clean]: {e}");
                process::exit(e.exit_code());
            }
        }
        return;
    }

    let opts = CleanOpts { cache, all };
    match run_clean(&root, &opts) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error[clean]: {e}");
            process::exit(e.exit_code());
        }
    }
}

fn print_ops(command: &Option<Command>) {
    if let Some(Command::Ops { core_v1 }) = command {
        if *core_v1 {
            println!("Core v1 operators (name | arity | dtypes | autodiff)");
            for op in core_v1::core_v1_ops() {
                let arity = match op.arity {
                    core_v1::Arity::Fixed(n) => format!("{n}"),
                    core_v1::Arity::Variadic { min } => format!("{min}+"),
                };
                let dtypes = if op.allowed_dtypes.is_empty() {
                    "shape-dependent".to_string()
                } else {
                    op.allowed_dtypes
                        .iter()
                        .map(|d| format!("{d:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let autodiff = if op.differentiable { "yes" } else { "no" };
                println!(
                    "{:<18} | {:<6} | {:<24} | {}",
                    op.name, arity, dtypes, autodiff
                );
            }
        }
    }
}

fn print_version() {
    println!("mind {}", env!("CARGO_PKG_VERSION"));

    #[cfg(feature = "mlir-lowering")]
    let components = {
        let mut components = ["core-ir=1.0", "core-autodiff=1.0"].to_vec();
        components.push("mlir-lowering=0.1");
        components
    };

    #[cfg(not(feature = "mlir-lowering"))]
    let components = ["core-ir=1.0", "core-autodiff=1.0"];

    println!("{}", components.join("  "));
}

fn print_stability() {
    println!(
        "MIND Core v1 stability: stable IR/autodiff/CLI surfaces; MLIR lowering is\
         conditionally stable within a minor release; new ops & feature flags are\
         experimental. See docs/versioning.md for details."
    );
}

fn run_conformance(profile: &str) {
    let profile = match profile.to_ascii_lowercase().as_str() {
        "cpu" => ConformanceProfile::CpuBaseline,
        "gpu" => ConformanceProfile::CpuAndGpu,
        other => {
            eprintln!("error[conformance]: unknown profile '{other}' (expected cpu|gpu)");
            process::exit(1);
        }
    };

    match conformance::run_conformance(ConformanceOptions { profile }) {
        Ok(()) => {
            println!("Core v1 conformance passed for profile: {:?}", profile);
        }
        Err(err) => {
            eprintln!("conformance failures detected:");
            for failure in err.0.iter() {
                eprintln!("- {failure}");
            }
            process::exit(1);
        }
    }
}

#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
fn emit_mlir_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    if !cli.emit_mlir {
        return;
    }

    let mlir: MlirProducts = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    println!("{}", mlir.primal_mlir);

    if cli.autodiff {
        if let Some(grad_mlir) = mlir.grad_mlir {
            println!("{}", grad_mlir);
        }
    }
}

/// Thin wrapper around `pipeline::lower_to_mlir` that erases the
/// `autodiff`-feature signature difference for the `mindc` binary.
#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    feature = "autodiff"
))]
fn lower_to_mlir_compat(
    products: &libmind::pipeline::CompileProducts,
) -> Result<MlirProducts, libmind::MlirLowerError> {
    lower_to_mlir(&products.ir, products.grad.as_ref())
}

#[cfg(all(
    any(feature = "mlir-lowering", feature = "mlir-build"),
    not(feature = "autodiff")
))]
fn lower_to_mlir_compat(
    products: &libmind::pipeline::CompileProducts,
) -> Result<MlirProducts, libmind::MlirLowerError> {
    lower_to_mlir(&products.ir)
}

#[cfg(not(any(feature = "mlir-lowering", feature = "mlir-build")))]
fn emit_mlir_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_mlir {
        eprintln!(
            "error[mlir]: MLIR emission requires building with the 'mlir-lowering' or 'mlir-build' feature"
        );
        process::exit(1);
    }
}

fn emit_mic3_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let path = match &cli.emit_mic3 {
        Some(p) => p,
        None => return,
    };
    let bytes = libmind::ir::compact::emit_mic3(&products.ir);
    if let Err(err) = fs::write(path, &bytes) {
        eprintln!("error[emit-mic3]: failed to write {path}: {err}");
        process::exit(1);
    }
    eprintln!("Wrote mic@3 artifact: {path} ({} bytes)", bytes.len());
}

fn emit_evidence_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let path = match &cli.emit_evidence {
        Some(p) => p,
        None => return,
    };
    let substrate = cli.target.as_str();
    // TODO(#289): flip to Nondeterministic when the #[nondeterministic] attr is
    // propagated through the IR; for now the declaration defaults to Deterministic.
    let determinism = libmind::ir::compact::Determinism::Deterministic;
    let toolchain = env!("CARGO_PKG_VERSION");
    let bytes = libmind::ir::compact::emit_mic3_with_evidence(
        &products.ir,
        substrate,
        None,
        determinism,
        toolchain,
    );
    if let Err(err) = fs::write(path, &bytes) {
        eprintln!("error[emit-evidence]: failed to write {path}: {err}");
        process::exit(1);
    }
    eprintln!(
        "Wrote mic@3 evidence artifact: {path} ({} bytes)",
        bytes.len()
    );
}

fn parse_target(raw: &str) -> Result<BackendTarget, String> {
    match raw.to_ascii_lowercase().as_str() {
        "cpu" => Ok(BackendTarget::Cpu),
        "gpu" | "cuda" | "rocm" | "metal" | "webgpu" => Ok(BackendTarget::Gpu),
        "tpu" => Ok(BackendTarget::Tpu),
        "npu" | "ane" | "hexagon" => Ok(BackendTarget::Npu),
        "lpu" | "groq" => Ok(BackendTarget::Lpu),
        "dpu" | "smartnic" | "bluefield" => Ok(BackendTarget::Dpu),
        "fpga" | "hls" => Ok(BackendTarget::Fpga),
        // Wafer-scale: distinct logical target from GPU because the
        // runtime backend lowers to CSL and reasons about a 2-D fabric
        // mesh rather than CUDA-style SMs. Accept all WSE generations
        // here; the wafer generation (WSE-2 / WSE-3) is selected at
        // runtime, not at the source-level target.
        "cerebras" | "wse" | "wse2" | "wse3" => Ok(BackendTarget::Cerebras),
        other => Err(format!(
            "unknown target '{other}' (expected cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras)"
        )),
    }
}

fn resolve_color_choice(flag: &Option<String>) -> ColorChoice {
    if let Some(value) = flag.as_deref() {
        return ColorChoice::parse(value).unwrap_or(ColorChoice::Auto);
    }
    if let Ok(env) = std::env::var("MINDC_COLOR") {
        return ColorChoice::parse(&env).unwrap_or(ColorChoice::Auto);
    }
    ColorChoice::Auto
}

#[cfg(feature = "mlir-build")]
fn emit_obj_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let obj_path = match &cli.emit_obj {
        Some(path) => path,
        None => return,
    };

    // First lower to MLIR
    let mlir = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    // Resolve build tools
    let tools = match libmind::eval::mlir_build::resolve_tools() {
        Ok(tools) => tools,
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    };

    // Build object file
    let opts = libmind::eval::mlir_build::BuildOptions {
        preset: "core",
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: Some(Path::new(obj_path)),
        emit_shared: None,
        opt_pipeline: None,
        target_triple: None,
    };

    match libmind::eval::mlir_build::build_all(&mlir.primal_mlir, &tools, &opts) {
        Ok(_) => {
            eprintln!("Wrote object file: {}", obj_path);
        }
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "mlir-build"))]
fn emit_obj_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_obj.is_some() {
        eprintln!("error[build]: --emit-obj requires building with the 'mlir-build' feature");
        process::exit(1);
    }
}

#[cfg(feature = "mlir-build")]
fn emit_shared_if_requested(cli: &CompileArgs, products: &libmind::pipeline::CompileProducts) {
    let shared_path = match &cli.emit_shared {
        Some(path) => path,
        None => return,
    };

    let mlir = match lower_to_mlir_compat(products) {
        Ok(mlir) => mlir,
        Err(err) => {
            eprintln!("error[mlir]: {err}");
            process::exit(1);
        }
    };

    let tools = match libmind::eval::mlir_build::resolve_tools() {
        Ok(tools) => tools,
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    };

    let opts = libmind::eval::mlir_build::BuildOptions {
        preset: "core",
        emit_mlir_file: None,
        emit_llvm_file: None,
        emit_obj_file: None,
        emit_shared: Some(Path::new(shared_path)),
        opt_pipeline: None,
        target_triple: None,
    };

    match libmind::eval::mlir_build::build_all(&mlir.primal_mlir, &tools, &opts) {
        Ok(_) => {
            eprintln!("Wrote shared library: {}", shared_path);
        }
        Err(err) => {
            eprintln!("error[build]: {err}");
            process::exit(1);
        }
    }
}

#[cfg(not(feature = "mlir-build"))]
fn emit_shared_if_requested(cli: &CompileArgs, _products: &libmind::pipeline::CompileProducts) {
    if cli.emit_shared.is_some() {
        eprintln!("error[build]: --emit-shared requires building with the 'mlir-build' feature");
        process::exit(1);
    }
}
