//! Command-line entry point for MIND.
//!
//! Usage:
//!   mind eval "1 + 2 * 3"
//!   mind repl

use mind::{diagnostics, eval, parser};
use std::collections::HashMap;
use std::io::{self, Write};

#[derive(Default)]
struct EmitOpts {
    emit_mlir_stdout: bool,
    emit_mlir_file: Option<String>,
    mlir_lower: eval::MlirLowerPreset,
}

fn parse_emit_flags(args: &[String]) -> EmitOpts {
    let mut out = EmitOpts::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--emit-mlir" => out.emit_mlir_stdout = true,
            "--emit-mlir-file" => {
                if i + 1 < args.len() {
                    out.emit_mlir_file = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--mlir-lower" => {
                if i + 1 < args.len() {
                    out.mlir_lower = eval::MlirLowerPreset::from_str(&args[i + 1])
                        .unwrap_or(eval::MlirLowerPreset::None);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 && args[1] == "eval" {
        if args.len() < 3 {
            eprintln!("Usage: mind eval \"<expression or statements>\"");
            std::process::exit(1);
        }
        let src = &args[2];
        let emit_opts = parse_emit_flags(&args[3..]);
        run_eval_once(src, emit_opts);
        return;
    }

    if args.len() >= 2 && args[1] == "repl" {
        run_repl();
        return;
    }

    eprintln!("Usage:");
    eprintln!("  mind eval \"<expression or statements>\"");
    eprintln!("  mind repl");
    std::process::exit(1);
}

fn run_eval_once(src: &str, emit_opts: EmitOpts) {
    match parser::parse_with_diagnostics(src) {
        Ok(module) => {
            let mut env = HashMap::new();
            match eval::eval_module_value_with_env(&module, &mut env, Some(src)) {
                Ok(_) => {
                    let ir = eval::lower_to_ir(&module);
                    if emit_opts.emit_mlir_stdout || emit_opts.emit_mlir_file.is_some() {
                        let txt = eval::emit_mlir_string(&ir, emit_opts.mlir_lower);
                        if emit_opts.emit_mlir_stdout {
                            println!("{txt}");
                        }
                        if let Some(path) = emit_opts.emit_mlir_file.as_ref() {
                            if let Err(e) = eval::emit_mlir_to_file(
                                &ir,
                                emit_opts.mlir_lower,
                                std::path::Path::new(path),
                            ) {
                                eprintln!("Failed to write MLIR to {}: {e}", path);
                            }
                        }
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
