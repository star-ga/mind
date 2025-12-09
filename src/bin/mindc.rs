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

use mind::diagnostics::{ColorChoice, DiagnosticEmitter, DiagnosticFormat};
use mind::pipeline::{compile_source_with_name, CompileOptions};
use mind::BackendTarget;
use mind::{conformance, ConformanceOptions, ConformanceProfile};

#[cfg(feature = "mlir-lowering")]
use mind::pipeline::{lower_to_mlir, MlirProducts};

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
    /// Run the Core v1 conformance suite.
    Conformance {
        /// Which profile to execute (cpu|gpu).
        #[arg(long, default_value = "cpu")]
        profile: String,
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
    /// Select the execution target backend (cpu|gpu).
    #[arg(long, value_name = "TARGET", default_value = "cpu")]
    target: String,
    /// Diagnostic output format (human|short|json).
    #[arg(long, value_name = "FORMAT", default_value = "human")]
    diagnostic_format: String,
    /// ANSI color handling (auto|always|never).
    #[arg(long, value_name = "WHEN")]
    color: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    if let Some(Command::Conformance { profile }) = &cli.command {
        run_conformance(profile);
        return;
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

    let emit_ir = cli.compile.emit_ir || (!cli.compile.emit_grad_ir && !cli.compile.emit_mlir);
    if emit_ir {
        println!("{}", products.ir);
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

#[cfg(feature = "mlir-lowering")]
fn emit_mlir_if_requested(cli: &CompileArgs, products: &mind::pipeline::CompileProducts) {
    if !cli.emit_mlir {
        return;
    }

    let grads = {
        #[cfg(feature = "autodiff")]
        {
            products.grad.as_ref()
        }
        #[cfg(not(feature = "autodiff"))]
        {
            None
        }
    };

    let mlir: MlirProducts = match lower_to_mlir(&products.ir, grads) {
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

#[cfg(not(feature = "mlir-lowering"))]
fn emit_mlir_if_requested(cli: &CompileArgs, _products: &mind::pipeline::CompileProducts) {
    if cli.emit_mlir {
        eprintln!("error[mlir]: MLIR emission requires building with the 'mlir-lowering' feature");
        process::exit(1);
    }
}

fn parse_target(raw: &str) -> Result<BackendTarget, String> {
    match raw.to_ascii_lowercase().as_str() {
        "cpu" => Ok(BackendTarget::Cpu),
        "gpu" => Ok(BackendTarget::Gpu),
        other => Err(format!("unknown target '{other}' (expected cpu|gpu)")),
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
