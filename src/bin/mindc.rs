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

use libmind::build::{run_build, BuildOpts};
use libmind::check::{run_check, CheckOptions, ReporterKind};
use libmind::fmt::cli as mindc_fmt;
use libmind::test::{run_tests, ReporterKind as TestReporterKind, TestOptions as MindTestOptions};

use libmind::diagnostics::{ColorChoice, DiagnosticEmitter, DiagnosticFormat};
use libmind::ops::core_v1;
use libmind::pipeline::{compile_source_with_name, CompileOptions};
use libmind::project::{
    bench_project, run_project, BenchOptions, BuildOptions,
    BuildTarget, EmitKind, OptimizeLevel,
};
use libmind::BackendTarget;
use libmind::{conformance, ConformanceOptions, ConformanceProfile};

#[cfg(any(feature = "mlir-lowering", feature = "mlir-build"))]
use libmind::pipeline::{lower_to_mlir, MlirProducts};

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
    /// Reads [build] from Mind.toml; CLI flags override the manifest.
    Build {
        /// Source files to compile.  When omitted, uses [build].entry or
        /// auto-detects src/main.mind / src/lib.mind.
        #[arg(value_name = "PATHS")]
        paths: Vec<String>,
        /// Build in release mode (equivalent to --optimize=release).
        #[arg(long)]
        release: bool,
        /// Target backend (cpu|gpu|tpu|npu|lpu|dpu|fpga|cerebras).
        /// Overrides [build].target in Mind.toml.
        #[arg(long, value_name = "TARGET")]
        target: Option<String>,
        /// Output artifact type: binary | cdylib | object.
        /// Overrides [build].emit in Mind.toml.
        #[arg(long, value_name = "EMIT")]
        emit: Option<String>,
        /// Optimization level: debug | release | size.
        /// Overrides [build].optimize in Mind.toml. --release is shorthand.
        #[arg(long, value_name = "LEVEL", conflicts_with = "release")]
        optimize: Option<String>,
        /// Custom output path.  Overrides the default target/<profile>/<name>.
        #[arg(long, value_name = "PATH")]
        out: Option<String>,
        /// Show verbose output.
        #[arg(short, long)]
        verbose: bool,
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
        }) => {
            run_mindc_build(paths, *release, target, emit, optimize, out, *verbose);
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
        }) => {
            run_mindc_test(paths, filter.as_deref(), *threads, *list, reporter);
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
                "lsp"  => ReporterKind::Lsp,
                _      => ReporterKind::Human,
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
        Some(Command::Ops { .. }) => {
            print_ops(&cli.command);
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
        || (!cli.compile.emit_grad_ir && !cli.compile.emit_mlir && !cli.compile.emit_mic);
    if emit_ir {
        println!("{}", products.ir);
    }

    if cli.compile.emit_mic {
        let mic = libmind::ir::compact::emit_mic(&products.ir);
        println!("{}", mic);
    }

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
) {
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
) {
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
