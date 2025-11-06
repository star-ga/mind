//! MIND CLI placeholder (core build, no-default-features).
//! This binary compiles cleanly on stable Rust without MLIR/LLVM features.
//! Backends are feature-gated in Cargo.toml (`mlir`, `llvm`).

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use std::path::PathBuf;

// Keep clap minimal; works fine without special features.
use clap::{CommandFactory, Parser, Subcommand};

/// The MIND programming language toolchain (placeholder).
#[derive(Parser, Debug)]
#[command(name = "mind")]
#[command(about = "The MIND programming language compiler", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile a MIND source file
    Build {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file path
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "0")]
        opt_level: u8,
    },

    /// Run a MIND source file
    Run {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Check a MIND source file for errors
    Check {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },

    /// Format MIND source code
    Fmt {
        /// Input file path
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Build {
            input,
            output,
            opt_level,
        }) => {
            println!("Compiling (placeholder)...");
            println!("  input   = {}", input.display());
            if let Some(out) = output {
                println!("  output  = {}", out.display());
            }
            println!("  opt     = O{opt_level}");
            println!(
                "Note: compiler backend is feature-gated (enable with `--features mlir,llvm`)."
            );
        }
        Some(Commands::Run { input, args }) => {
            println!("Running (placeholder)...");
            println!("  file    = {}", input.display());
            if !args.is_empty() {
                println!("  args    = {args:?}");
            }
            println!("Note: runtime/backend not implemented yet.");
        }
        Some(Commands::Check { input }) => {
            println!("Checking (placeholder)...");
            println!("  file    = {}", input.display());
        }
        Some(Commands::Fmt { input }) => {
            println!("Formatting (placeholder)...");
            println!("  file    = {}", input.display());
        }
        None => {
            // No subcommand: show help and exit 0
            let _ = Cli::command().print_help();
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn compiles_and_runs_placeholder() {
        // Trivial assertion to keep `cargo test` green.
        assert_eq!(2 + 2, 4);
    }
}
