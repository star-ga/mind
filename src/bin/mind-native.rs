// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! `mind-native` — compile a `.mind` file straight to a runnable x86-64 ELF with
//! **zero LLVM / MLIR / clang / assembler / linker**, via the experimental
//! native backend (`libmind::native`). Built only with `--features
//! native-backend` (see the `required-features` on the `[[bin]]`).
//!
//! Scope: the scalar-i64 subset the native backend supports (functions, the full
//! integer `BinOp` set, `if`-expressions, `while` loops, recursion). Anything
//! outside it fails loud with a typed `NativeError` — never a broken artifact.
//!
//! ```text
//! mind-native examples/fib.mind /tmp/fib && /tmp/fib ; echo $?   # 55
//! ```

use std::os::unix::fs::PermissionsExt;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.mind> <output-elf>", args[0]);
        return ExitCode::from(2);
    }
    let (input, output) = (&args[1], &args[2]);

    let src = match std::fs::read_to_string(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("mind-native: cannot read {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let module = match libmind::parser::parse(&src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mind-native: parse error: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let ir = libmind::eval::lower::lower_to_ir(&module);

    let elf = match libmind::native::compile_to_elf(&ir) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("mind-native: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = std::fs::write(output, &elf) {
        eprintln!("mind-native: cannot write {output}: {e}");
        return ExitCode::FAILURE;
    }
    if let Ok(meta) = std::fs::metadata(output) {
        let mut perms = meta.permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(output, perms);
    }

    eprintln!("mind-native: wrote {output} ({} bytes, zero LLVM)", elf.len());
    ExitCode::SUCCESS
}
