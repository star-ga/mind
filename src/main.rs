//! Command-line entry point for MIND.
//!
//! Usage:
//!   mind eval "1 + 2 * 3"
//!   mind repl

#[cfg(feature = "pkg")]
use anyhow::anyhow;
use anyhow::Context;
use anyhow::Result;

use clap::Args;
use clap::Parser;
use clap::Subcommand;

#[cfg(feature = "pkg")]
use mind::package::build_package;
use mind::package::default_install_dir;
use mind::package::inspect_package;
use mind::package::install_package;
use mind::package::MindManifest;

use mind::diagnostics;
use mind::eval;
use mind::parser;

use std::collections::HashMap;
use std::io;
use std::io::Write;

#[cfg(feature = "pkg")]
use std::fs;
use std::path::Path;
#[cfg(any(feature = "mlir-build", feature = "pkg"))]
use std::path::PathBuf;
#[cfg(feature = "mlir-exec")]
use std::time::Duration;

struct EmitOpts {
    emit_mlir_stdout: bool,
    emit_mlir_file: Option<String>,
    emit_llvm_file: Option<String>,
    emit_obj_file: Option<String>,
    emit_shared_lib: Option<String>,
    #[cfg(feature = "ffi-c")]
    emit_ffi_header: Option<String>,
    #[cfg(feature = "ffi-c")]
    emit_ffi_shim: Option<String>,
    mlir_lower: eval::MlirLowerPreset,
    run_mlir_opt: bool,
    mlir_opt_bin: Option<String>,
    mlir_opt_passes: Vec<String>,
    mlir_opt_timeout_ms: u64,
    mlir_pass_pipeline: Option<String>,
    target_triple: Option<String>,
}

impl Default for EmitOpts {
    fn default() -> Self {
        Self {
            emit_mlir_stdout: false,
            emit_mlir_file: None,
            emit_llvm_file: None,
            emit_obj_file: None,
            emit_shared_lib: None,
            #[cfg(feature = "ffi-c")]
            emit_ffi_header: None,
            #[cfg(feature = "ffi-c")]
            emit_ffi_shim: None,
            mlir_lower: eval::MlirLowerPreset::None,
            run_mlir_opt: false,
            mlir_opt_bin: None,
            mlir_opt_passes: default_mlir_opt_passes(),
            mlir_opt_timeout_ms: 5_000,
            mlir_pass_pipeline: None,
            target_triple: None,
        }
    }
}

fn default_mlir_opt_passes() -> Vec<String> {
    vec!["canonicalize".to_string(), "cse".to_string()]
}

impl EmitOpts {
    fn from_eval_args(args: &EvalArgs) -> Self {
        let mut out = EmitOpts {
            emit_mlir_stdout: args.emit_mlir,
            emit_mlir_file: args.emit_mlir_file.clone(),
            emit_llvm_file: args.emit_llvm_file.clone(),
            emit_obj_file: args.emit_obj.clone(),
            emit_shared_lib: args.build_shared.clone(),
            #[cfg(feature = "ffi-c")]
            emit_ffi_header: args.emit_ffi_c.clone(),
            #[cfg(feature = "ffi-c")]
            emit_ffi_shim: args.emit_ffi_shim.clone(),
            ..Default::default()
        };
        if let Some(lower) = &args.mlir_lower {
            out.mlir_lower = lower.parse().unwrap_or(eval::MlirLowerPreset::None);
        }
        if let Some(pipeline) = &args.mlir_passes {
            if !pipeline.trim().is_empty() {
                out.mlir_pass_pipeline = Some(pipeline.trim().to_string());
            }
        }
        if let Some(triple) = &args.target_triple {
            if !triple.trim().is_empty() {
                out.target_triple = Some(triple.trim().to_string());
            }
        }
        if matches!(args.mlir_opt.as_deref(), Some("")) {
            out.run_mlir_opt = true;
        }
        if let Some(path) = &args.mlir_opt {
            if !path.is_empty() {
                out.mlir_opt_bin = Some(path.clone());
            }
        }
        if let Some(bin) = &args.mlir_opt_bin {
            out.mlir_opt_bin = Some(bin.clone());
        }
        if let Some(passes) = &args.mlir_opt_passes {
            let parsed: Vec<String> = passes
                .split(',')
                .filter_map(|p| {
                    let trimmed = p.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .collect();
            if !parsed.is_empty() {
                out.mlir_opt_passes = parsed;
            }
        }
        if let Some(timeout) = args.mlir_opt_timeout_ms {
            out.mlir_opt_timeout_ms = timeout;
        }
        if out.mlir_opt_passes.is_empty() {
            out.mlir_opt_passes = default_mlir_opt_passes();
        }
        out
    }

    fn wants_aot_artifacts(&self) -> bool {
        let base = self.emit_llvm_file.is_some()
            || self.emit_obj_file.is_some()
            || self.emit_shared_lib.is_some();
        #[cfg(feature = "ffi-c")]
        {
            if base || self.emit_ffi_header.is_some() || self.emit_ffi_shim.is_some() {
                return true;
            }
        }
        base
    }
}

#[derive(Parser)]
#[command(author, version, about = None, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    Eval(EvalArgs),
    Repl,
    #[cfg(feature = "pkg")]
    Package {
        #[arg(subcommand)]
        action: PackageAction,
    },
}

#[cfg(feature = "pkg")]
#[derive(Subcommand)]
enum PackageAction {
    Build {
        #[arg(short, long)]
        out: Option<String>,
    },
    Inspect {
        #[arg(short, long)]
        path: String,
    },
    Install {
        #[arg(short, long)]
        path: String,
        #[arg(short, long)]
        target: Option<String>,
    },
}

#[derive(Args)]
struct EvalArgs {
    #[arg(long)]
    exec: bool,
    #[arg(long, conflicts_with = "exec")]
    mlir_exec: bool,
    #[arg(long)]
    jit: bool,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "cuda")]
    gpu_backend: String,
    #[arg(long, value_name = "X,Y,Z")]
    gpu_blocks: Option<String>,
    #[arg(long, value_name = "X,Y,Z")]
    gpu_threads: Option<String>,
    #[arg(long)]
    emit_mlir: bool,
    #[arg(long)]
    emit_mlir_file: Option<String>,
    #[arg(long)]
    emit_llvm_file: Option<String>,
    #[arg(long, value_name = "PATH")]
    emit_obj: Option<String>,
    #[arg(long = "build", value_name = "PATH")]
    build_shared: Option<String>,
    #[cfg(feature = "ffi-c")]
    #[arg(long, value_name = "PATH")]
    emit_ffi_c: Option<String>,
    #[cfg(feature = "ffi-c")]
    #[arg(long, value_name = "PATH")]
    emit_ffi_shim: Option<String>,
    #[arg(long)]
    mlir_lower: Option<String>,
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    mlir_opt: Option<String>,
    #[arg(long)]
    mlir_opt_bin: Option<String>,
    #[arg(long)]
    mlir_opt_passes: Option<String>,
    #[arg(long)]
    mlir_opt_timeout_ms: Option<u64>,
    #[arg(long, value_name = "PATH")]
    mlir_cpu_runner: Option<String>,
    #[arg(long, value_name = "PASSES")]
    mlir_passes: Option<String>,
    #[arg(long, value_name = "MS")]
    mlir_timeout_ms: Option<u64>,
    #[arg(long)]
    target_triple: Option<String>,
    #[arg(value_name = "SRC", num_args = 1..)]
    program: Vec<String>,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Eval(args) => run_eval_command(args),
        Command::Repl => run_repl(),
        #[cfg(feature = "pkg")]
        Command::Package { action } => run_package_command(action),
    }
}

fn run_eval_command(args: EvalArgs) {
    let src = args.program.join(" ");
    if src.trim().is_empty() {
        eprintln!("Evaluation requires source input.");
        std::process::exit(1);
    }

    let emit_opts = EmitOpts::from_eval_args(&args);
    let exec_mode = if args.device.eq_ignore_ascii_case("gpu") {
        #[cfg(feature = "mlir-gpu")]
        {
            let backend = match args.gpu_backend.to_lowercase().as_str() {
                "cuda" => eval::GpuBackend::Cuda,
                "rocm" => eval::GpuBackend::Rocm,
                other => {
                    eprintln!(
                        "Unknown GPU backend '{other}', defaulting to CUDA; available: cuda, rocm"
                    );
                    eval::GpuBackend::Cuda
                }
            };
            let parse_triplet = |value: &Option<String>, label: &str| -> (u32, u32, u32) {
                if let Some(raw) = value {
                    let parts: Vec<_> = raw.split(',').collect();
                    if parts.len() == 3 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[0].trim().parse::<u32>(),
                            parts[1].trim().parse::<u32>(),
                            parts[2].trim().parse::<u32>(),
                        ) {
                            return (x, y, z);
                        }
                    }
                    eprintln!(
                        "Invalid {label} value '{raw}', expected three comma-separated integers"
                    );
                }
                (1, 1, 1)
            };
            let blocks = parse_triplet(&args.gpu_blocks, "gpu-blocks");
            let threads = parse_triplet(&args.gpu_threads, "gpu-threads");
            eval::ExecMode::MlirGpu {
                backend,
                blocks,
                threads,
            }
        }
        #[cfg(not(feature = "mlir-gpu"))]
        {
            eprintln!(
                "--device gpu requested but this binary lacks the mlir-gpu feature; falling back to preview"
            );
            eval::ExecMode::Preview
        }
    } else if args.jit {
        #[cfg(feature = "mlir-jit")]
        {
            eval::ExecMode::MlirJitCpu
        }
        #[cfg(not(feature = "mlir-jit"))]
        {
            eprintln!(
                "--jit requested but this binary was built without the mlir-jit feature; falling back"
            );
            eval::ExecMode::Preview
        }
    } else if args.mlir_exec {
        #[cfg(feature = "mlir-exec")]
        {
            let mut cfg = eval::MlirExecConfig::default();
            if let Some(path) = args.mlir_opt.as_deref() {
                if !path.is_empty() {
                    cfg.mlir_opt = Some(path.into());
                }
            }
            if let Some(path) = args.mlir_cpu_runner.as_deref() {
                cfg.mlir_cpu_runner = Some(path.into());
            }
            if let Some(passes) = args.mlir_passes.as_deref() {
                let parsed: Vec<String> = passes
                    .split_whitespace()
                    .filter(|p| !p.is_empty())
                    .map(|p| p.to_string())
                    .collect();
                if !parsed.is_empty() {
                    cfg.opt_passes = parsed;
                }
            }
            if let Some(timeout_ms) = args.mlir_timeout_ms {
                cfg.timeout = Duration::from_millis(timeout_ms);
            }
            eval::ExecMode::MlirExternal(cfg)
        }
        #[cfg(not(feature = "mlir-exec"))]
        {
            eprintln!(
                "--mlir-exec requested but this binary was built without the mlir-exec feature. Rebuild with --features mlir-exec."
            );
            std::process::exit(1);
        }
    } else if args.exec {
        #[cfg(feature = "cpu-exec")]
        {
            eval::ExecMode::CpuExec
        }
        #[cfg(not(feature = "cpu-exec"))]
        {
            eprintln!("--exec requested but cpu-exec feature is not enabled; running in preview.");
            eval::ExecMode::Preview
        }
    } else {
        eval::ExecMode::Preview
    };

    run_eval_once(&src, emit_opts, exec_mode);
}

fn run_eval_once(src: &str, emit_opts: EmitOpts, exec_mode: eval::ExecMode) {
    match parser::parse_with_diagnostics(src) {
        Ok(module) => {
            let mut env = HashMap::new();
            let mode_for_eval = exec_mode.clone();
            match eval::eval_module_value_with_env_mode(&module, &mut env, Some(src), mode_for_eval)
            {
                Ok(value) => {
                    println!("{}", eval::format_value_human(&value));
                    match &exec_mode {
                        eval::ExecMode::CpuExec => return,
                        #[cfg(feature = "mlir-exec")]
                        eval::ExecMode::MlirExternal(_) => return,
                        #[cfg(feature = "mlir-jit")]
                        eval::ExecMode::MlirJitCpu => return,
                        #[cfg(feature = "mlir-gpu")]
                        eval::ExecMode::MlirGpu { .. } => return,
                        eval::ExecMode::Preview => {}
                    }
                    let ir = eval::lower_to_ir(&module);
                    #[allow(unused_mut)]
                    let mut built_mlir: Option<String> = None;
                    #[allow(unused_mut)]
                    let mut builder_invoked = false;
                    #[allow(unused_mut)]
                    let mut mlir_written_by_builder = false;
                    if emit_opts.wants_aot_artifacts() {
                        #[cfg(feature = "mlir-build")]
                        {
                            let base_mlir =
                                eval::emit_mlir_string(&ir, eval::MlirLowerPreset::None);
                            let tools = match eval::resolve_mlir_build_tools() {
                                Ok(tools) => tools,
                                Err(err) => {
                                    eprintln!("Failed to resolve MLIR build tools: {err}");
                                    std::process::exit(2);
                                }
                            };
                            let mlir_path = emit_opts.emit_mlir_file.as_ref().map(PathBuf::from);
                            let llvm_path = emit_opts.emit_llvm_file.as_ref().map(PathBuf::from);
                            let obj_path = emit_opts.emit_obj_file.as_ref().map(PathBuf::from);
                            let shared_path = emit_opts.emit_shared_lib.as_ref().map(PathBuf::from);
                            let build_opts = eval::MlirBuildOptions {
                                preset: emit_opts.mlir_lower.as_str(),
                                emit_mlir_file: mlir_path.as_deref(),
                                emit_llvm_file: llvm_path.as_deref(),
                                emit_obj_file: obj_path.as_deref(),
                                emit_shared: shared_path.as_deref(),
                                opt_pipeline: emit_opts.mlir_pass_pipeline.as_deref(),
                                target_triple: emit_opts.target_triple.as_deref(),
                            };
                            match eval::build_mlir_artifacts(&base_mlir, &tools, &build_opts) {
                                Ok(products) => {
                                    mlir_written_by_builder = mlir_path.is_some();
                                    built_mlir = Some(products.optimized_mlir.clone());
                                    builder_invoked = true;
                                }
                                Err(err) => {
                                    eprintln!("Failed to build MLIR artifacts: {err}");
                                    std::process::exit(2);
                                }
                            }
                        }
                        #[cfg(not(feature = "mlir-build"))]
                        {
                            eprintln!(
                                "MLIR artifact emission requires the mlir-build feature; rebuild with --features mlir-build",
                            );
                            std::process::exit(2);
                        }
                    }
                    if emit_opts.emit_mlir_stdout || emit_opts.emit_mlir_file.is_some() {
                        let mut mlir_text = if let Some(text) = built_mlir.as_ref() {
                            text.clone()
                        } else {
                            eval::emit_mlir_string(&ir, emit_opts.mlir_lower)
                        };
                        if !builder_invoked && emit_opts.run_mlir_opt {
                            let bin = emit_opts
                                .mlir_opt_bin
                                .clone()
                                .or_else(|| std::env::var("MLIR_OPT").ok())
                                .unwrap_or_else(|| "mlir-opt".to_string());
                            let passes = emit_opts.mlir_opt_passes.clone();
                            let timeout = emit_opts.mlir_opt_timeout_ms;
                            match eval::mlir_opt::run_mlir_opt(&mlir_text, &bin, &passes, timeout) {
                                Ok(output)
                                    if output.status_ok && !output.stdout.trim().is_empty() =>
                                {
                                    mlir_text = output.stdout;
                                }
                                Ok(output) => {
                                    let msg = output.stderr.trim();
                                    if msg.is_empty() {
                                        eprintln!("[warn] mlir-opt failed without diagnostics");
                                    } else {
                                        eprintln!("[warn] mlir-opt failed: {msg}");
                                    }
                                }
                                Err(err) => {
                                    eprintln!("[warn] failed to spawn mlir-opt: {err}");
                                }
                            }
                        }
                        if emit_opts.emit_mlir_stdout {
                            println!("{mlir_text}");
                        }
                        if let Some(path) = emit_opts.emit_mlir_file.as_ref() {
                            if !(emit_opts.wants_aot_artifacts() && mlir_written_by_builder) {
                                let path_ref = std::path::Path::new(path);
                                let parent = path_ref
                                    .parent()
                                    .unwrap_or_else(|| std::path::Path::new("."));
                                if let Err(e) = std::fs::create_dir_all(parent) {
                                    eprintln!("Failed to create directories for {}: {e}", path);
                                } else if let Err(e) = std::fs::write(path_ref, &mlir_text) {
                                    eprintln!("Failed to write MLIR to {}: {e}", path);
                                }
                            }
                        }
                        return;
                    }
                    if builder_invoked {
                        return;
                    }
                    println!("--- Lowered IR ---\n{ir}");
                    let value = eval::eval_ir(&ir);
                    println!("--- Result ---\n{}", eval::format_value_human(&value));
                }
                Err(e) => report_eval_error(e, src, &module),
            }
        }
        Err(diags) => {
            for d in diags {
                let msg = diagnostics::render(src, &d);
                eprintln!("{msg}");
            }
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "pkg")]
fn run_package_command(action: PackageAction) {
    if let Err(err) = handle_package_command(action) {
        eprintln!("Package command failed: {err}");
        std::process::exit(1);
    }
}

#[cfg(feature = "pkg")]
fn handle_package_command(action: PackageAction) -> Result<()> {
    match action {
        PackageAction::Build { out } => package_build(out)?,
        PackageAction::Inspect { path } => package_inspect(&path)?,
        PackageAction::Install { path, target } => package_install(&path, target.as_deref())?,
    }
    Ok(())
}

#[cfg(feature = "pkg")]
fn package_build(out: Option<String>) -> Result<()> {
    let manifest_path = Path::new("package.toml");
    let mut manifest = if manifest_path.exists() {
        let manifest_data =
            fs::read_to_string(manifest_path).context("failed to read existing package.toml")?;
        toml::from_str::<MindManifest>(&manifest_data).context("failed to parse package.toml")?
    } else {
        MindManifest {
            name: "model".into(),
            version: "0.1.0".into(),
            authors: vec![],
            description: None,
            license: None,
            dependencies: None,
            files: Vec::new(),
            checksums: None,
        }
    };

    let files = if manifest.files.is_empty() {
        let discovered = discover_default_artifacts()?;
        manifest.files = discovered.clone();
        discovered
    } else {
        validate_files(&manifest.files)?
    };

    if manifest.files.is_empty() {
        return Err(anyhow!(
            "package manifest declares no files; add entries to the 'files' array"
        ));
    }

    let mut output =
        out.unwrap_or_else(|| format!("{}-{}.mindpkg", manifest.name, manifest.version));
    if Path::new(&output).extension().and_then(|ext| ext.to_str()) != Some("mindpkg") {
        output.push_str(".mindpkg");
    }

    if let Some(parent) = Path::new(&output).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }

    let file_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
    build_package(&output, &file_refs, &manifest)?;

    let packaged_manifest = inspect_package(&output)?;
    let manifest_toml = packaged_manifest.to_toml()?;
    fs::write(manifest_path, manifest_toml).context("failed to update package.toml")?;
    println!("Created {}", output);
    Ok(())
}

#[cfg(feature = "pkg")]
fn package_inspect(path: &str) -> Result<()> {
    let manifest = inspect_package(path)?;
    println!("{}", manifest.to_toml()?);
    Ok(())
}

#[cfg(feature = "pkg")]
fn package_install(path: &str, target: Option<&str>) -> Result<()> {
    let manifest = inspect_package(path)?;
    let target_str = target.unwrap_or("");
    let target_path = if target_str.is_empty() {
        default_install_dir(&manifest)?
    } else {
        PathBuf::from(target_str)
    };

    install_package(path, target_str)?;
    println!(
        "Installed {} {} to {}",
        manifest.name,
        manifest.version,
        target_path.display()
    );
    Ok(())
}

#[cfg(feature = "pkg")]
fn discover_default_artifacts() -> Result<Vec<String>> {
    let candidates = [
        "model.mlir",
        "model.so",
        "mind.h",
        "metadata.json",
        "README.md",
    ];
    let mut found = Vec::new();
    for candidate in candidates {
        if Path::new(candidate).exists() {
            found.push(candidate.to_string());
        }
    }
    if found.is_empty() {
        Err(anyhow!(
            "no artifacts found; specify files in package.toml or place standard outputs in the working directory"
        ))
    } else {
        Ok(found)
    }
}

#[cfg(feature = "pkg")]
fn validate_files(files: &[String]) -> Result<Vec<String>> {
    let mut validated = Vec::new();
    for file in files {
        let path = Path::new(file);
        if path.is_absolute() {
            return Err(anyhow!(
                "listed artifact '{}' must be a relative path",
                file
            ));
        }
        if path.components().count() != 1 {
            return Err(anyhow!(
                "listed artifact '{}' must not contain directory separators",
                file
            ));
        }
        if !path.exists() {
            return Err(anyhow!("listed artifact '{}' does not exist", file));
        }
        validated.push(file.clone());
    }
    Ok(validated)
}

fn run_repl() {
    let mut env: HashMap<String, i64> = HashMap::new();
    let stdin = io::stdin();
    let mut line = String::new();

    println!("MIND REPL — type :quit to exit");
    loop {
        print!("MIND> ");
        let _ = io::stdout().flush();

        line.clear();
        if stdin.read_line(&mut line).is_err() {
            eprintln!("Input error");
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, ":quit" | ":q" | ":exit") {
            break;
        }

        match parser::parse_with_diagnostics(trimmed) {
            Ok(module) => {
                match eval::eval_module_value_with_env(&module, &mut env, Some(trimmed)) {
                    Ok(result) => println!("{}", eval::format_value_human(&result)),
                    Err(e) => report_eval_error(e, trimmed, &module),
                }
            }
            Err(diags) => {
                for d in diags {
                    let msg = diagnostics::render(trimmed, &d);
                    eprintln!("{msg}");
                }
            }
        }
    }
}

fn report_eval_error(err: eval::EvalError, src: &str, module: &mind::ast::Module) {
    match err {
        eval::EvalError::TypeError(rendered) => {
            eprintln!("Evaluation error: type error");
            eprintln!("{rendered}");
            let diags = mind::type_checker::check_module_types(module, src, &HashMap::new());
            for d in diags {
                eprintln!("{}", mind::diagnostics::render(src, &d));
            }
        }
        other => {
            eprintln!("Evaluation error: {other}");
        }
    }
}
