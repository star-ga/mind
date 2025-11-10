#![cfg(feature = "mlir-subprocess")]

use mind::{eval, parser};

#[test]
fn mlir_opt_runs_when_available() {
    let bin = std::env::var("MLIR_OPT").unwrap_or_else(|_| "mlir-opt".into());
    if which::which(&bin).is_err() {
        eprintln!("mlir-opt not found, skipping test");
        return;
    }

    let src = "let x = 1; x + 2";
    let module = match parser::parse_with_diagnostics(src) {
        Ok(module) => module,
        Err(diags) => panic!("failed to parse input: {:?}", diags),
    };
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::emit_mlir_string(&ir, eval::MlirLowerPreset::None);

    let result = eval::mlir_opt::run_mlir_opt(&mlir, &bin, &["canonicalize".into()], 2_000)
        .expect("spawn mlir-opt");

    if !result.status_ok {
        panic!("mlir-opt invocation failed: {}", result.stderr);
    }

    assert!(
        result.stdout.contains("module"),
        "mlir-opt output missing module\nstdout: {}\nstderr: {}",
        result.stdout,
        result.stderr
    );
}
