//! Command-line entry point for MIND.
//!
//! Usage:
//!   mind eval "1 + 2 * 3"
//!   mind repl

use mind::{parser, eval, diagnostics};
use std::collections::HashMap;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 && args[1] == "eval" {
        if args.len() < 3 {
            eprintln!("Usage: mind eval \"<expression or statements>\"");
            std::process::exit(1);
        }
        let src = &args[2];
        run_eval_once(src);
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

fn run_eval_once(src: &str) {
    match parser::parse_with_diagnostics(src) {
        Ok(module) => match eval::eval_first_expr(&module) {
            Ok(result) => println!("{result}"),
            Err(e) => eprintln!("Evaluation error: {e}"),
        },
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
        if trimmed.is_empty() { continue; }
        if matches!(trimmed, ":quit" | ":q" | ":exit") { break; }

        match parser::parse_with_diagnostics(trimmed) {
            Ok(module) => match eval::eval_module_with_env(&module, &mut env) {
                Ok(result) => println!("{result}"),
                Err(e) => eprintln!("Evaluation error: {e}"),
            },
            Err(diags) => {
                for d in diags {
                    let msg = diagnostics::render(trimmed, &d);
                    eprintln!("{msg}");
                }
            }
        }
    }
}
