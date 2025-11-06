//! Command-line entry point for MIND.
//!
//! Usage:
//! ```bash
//! mind eval "1 + 2 * 3"
//! ```

use mind::{eval, parser};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args[1] != "eval" {
        eprintln!("Usage: mind eval \"<expression>\"");
        std::process::exit(1);
    }

    let expr = &args[2];
    match parser::parse(expr) {
        Ok(module) => match eval::eval_first_expr(&module) {
            Ok(result) => println!("{}", result),
            Err(e) => eprintln!("Evaluation error: {e}"),
        },
        Err(_) => eprintln!("Parse error"),
    }
}
