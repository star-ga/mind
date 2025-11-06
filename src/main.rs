//! Command-line entry point for MIND.
//!
//! Usage:
//! ```bash
//! mind eval "1 + 2 * 3"
//! ```

use mind::{diagnostics, eval, parser};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args[1] != "eval" {
        eprintln!("Usage: mind eval \"<expression or statements>\"");
        std::process::exit(1);
    }

    let src = &args[2];
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
