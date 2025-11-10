//! Command-line entry point for MIND.
//!
//! Usage:
//!   mind eval "1 + 2 * 3"
//!   mind repl

use mind::{diagnostics, eval, parser};
use std::collections::HashMap;
use std::io::{self, Write};

struct EmitOpts {
    emit_mlir_stdout: bool,
    emit_mlir_file: Option<String>,
    mlir_lower: eval::MlirLowerPreset,
    run_mlir_opt: bool,
    mlir_opt_bin: Option<String>,
    mlir_opt_passes: Vec<String>,
    mlir_opt_timeout_ms: u64,
}

impl Default for EmitOpts {
    fn default() -> Self {
        Self {
            emit_mlir_stdout: false,
            emit_mlir_file: None,
            mlir_lower: eval::MlirLowerPreset::None,
            run_mlir_opt: false,
            mlir_opt_bin: None,
            mlir_opt_passes: default_mlir_opt_passes(),
            mlir_opt_timeout_ms: 5_000,
        }
    }
}

fn default_mlir_opt_passes() -> Vec<String> {
    vec!["canonicalize".to_string(), "cse".to_string()]
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
            "--mlir-opt" => {
                out.run_mlir_opt = true;
            }
            "--mlir-opt-bin" => {
                if i + 1 < args.len() {
                    out.mlir_opt_bin = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--mlir-opt-passes" => {
                if i + 1 < args.len() {
                    out.mlir_opt_passes = args[i + 1]
                        .split(',')
                        .filter_map(|p| {
                            let trimmed = p.trim();
                            (!trimmed.is_empty()).then(|| trimmed.to_string())
                        })
                        .collect();
                    i += 1;
                }
            }
            "--mlir-opt-timeout-ms" => {
                if i + 1 < args.len() {
                    if let Ok(value) = args[i + 1].parse::<u64>() {
                        out.mlir_opt_timeout_ms = value;
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if out.mlir_opt_passes.is_empty() {
        out.mlir_opt_passes = default_mlir_opt_passes();
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
                        let mut mlir_text = eval::emit_mlir_string(&ir, emit_opts.mlir_lower);
                        if emit_opts.run_mlir_opt {
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
                            let path_ref = std::path::Path::new(path);
                            let parent =
                                path_ref.parent().unwrap_or_else(|| std::path::Path::new("."));
                            if let Err(e) = std::fs::create_dir_all(parent) {
                                eprintln!("Failed to create directories for {}: {e}", path);
                            } else if let Err(e) = std::fs::write(path_ref, &mlir_text) {
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
