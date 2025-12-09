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

use clap::{ArgAction, Parser};

use mind::diagnostics;
use mind::pipeline::{compile_source, CompileError, CompileOptions};

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
    /// Print the compiler version and component stability versions.
    #[arg(long, action = ArgAction::SetTrue)]
    version: bool,
    /// Print a short description of the public stability model.
    #[arg(long, action = ArgAction::SetTrue)]
    stability: bool,
    /// Input .mind file to compile.
    #[arg(required_unless_present_any = ["version", "stability"], value_name = "FILE")]
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
}

fn main() {
    let cli = Cli::parse();

    if cli.version {
        print_version();
        return;
    }

    if cli.stability {
        print_stability();
        return;
    }

    let input = cli.input.as_ref().expect("input validated by clap").clone();

    if cli.autodiff && cli.func.is_none() {
        eprintln!("error[autodiff]: --autodiff requires --func <name>");
        process::exit(1);
    }

    let source = match fs::read_to_string(&input) {
        Ok(src) => src,
        Err(err) => {
            eprintln!("failed to read {}: {err}", input);
            process::exit(1);
        }
    };

    let opts = CompileOptions {
        func: cli.func.clone(),
        enable_autodiff: cli.autodiff,
    };

    let products = match compile_source(&source, &opts) {
        Ok(products) => products,
        Err(err) => {
            render_error(&err, &source);
            process::exit(1);
        }
    };

    if cli.verify_only {
        return;
    }

    let emit_ir = cli.emit_ir || (!cli.emit_grad_ir && !cli.emit_mlir);
    if emit_ir {
        println!("{}", products.ir);
    }

    #[cfg(feature = "autodiff")]
    if cli.autodiff && cli.emit_grad_ir {
        match products.grad.as_ref() {
            Some(grad) => println!("{}", grad.gradient_module),
            None => {
                eprintln!("autodiff did not produce gradient IR");
                process::exit(1);
            }
        }
    }

    #[cfg(not(feature = "autodiff"))]
    if cli.autodiff && cli.emit_grad_ir {
        eprintln!("gradient IR emission requires building with the 'autodiff' feature");
        process::exit(1);
    }

    emit_mlir_if_requested(&cli, &products);
}

fn print_version() {
    println!("mind {}", env!("CARGO_PKG_VERSION"));

    #[cfg(feature = "mlir-lowering")]
    let components = {
        let mut components = vec!["core-ir=1.0", "core-autodiff=1.0"];
        components.push("mlir-lowering=0.1");
        components
    };

    #[cfg(not(feature = "mlir-lowering"))]
    let components = vec!["core-ir=1.0", "core-autodiff=1.0"];

    println!("{}", components.join("  "));
}

fn print_stability() {
    println!(
        "MIND Core v1 stability: stable IR/autodiff/CLI surfaces; MLIR lowering is\
         conditionally stable within a minor release; new ops & feature flags are\
         experimental. See docs/versioning.md for details."
    );
}

#[cfg(feature = "mlir-lowering")]
fn emit_mlir_if_requested(cli: &Cli, products: &mind::pipeline::CompileProducts) {
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
fn emit_mlir_if_requested(cli: &Cli, _products: &mind::pipeline::CompileProducts) {
    if cli.emit_mlir {
        eprintln!("error[mlir]: MLIR emission requires building with the 'mlir-lowering' feature");
        process::exit(1);
    }
}

fn render_error(err: &CompileError, source: &str) {
    match err {
        CompileError::ParseError(diags) | CompileError::TypeError(diags) => {
            for diag in diags {
                let prefix = match err {
                    CompileError::ParseError(_) => "error[parse]",
                    _ => "error[type-check]",
                };
                eprintln!("{prefix}: {}", diagnostics::render(source, diag));
            }
        }
        CompileError::IrVerify(e) => eprintln!("error[ir-verify]: {e}"),
        CompileError::MissingFunctionName => {
            eprintln!("error[autodiff]: --autodiff requires --func <name>")
        }
        #[cfg(feature = "autodiff")]
        CompileError::Autodiff(e) => eprintln!("error[autodiff]: {e}"),
        #[cfg(not(feature = "autodiff"))]
        CompileError::AutodiffDisabled => {
            eprintln!(
                "error[autodiff]: autodiff requested but the 'autodiff' feature is not enabled"
            )
        }
    }
}
