#[cfg(feature = "mlir-exec")]
#[test]
fn mlir_exec_scalar_add() {
    if which::which("mlir-opt").is_err() || which::which("mlir-cpu-runner").is_err() {
        eprintln!("skipping: mlir tools not found");
        return;
    }
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "mlir-exec",
            "--",
            "eval",
            "--mlir-exec",
            "1",
            "+",
            "2",
        ])
        .output()
        .expect("run mlir exec");
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("3"), "stdout: {}", stdout);
}

#[cfg(all(feature = "mlir-exec", feature = "cpu-exec"))]
#[test]
fn parity_cpu_vs_mlir_exec_simple() {
    if which::which("mlir-opt").is_err() || which::which("mlir-cpu-runner").is_err() {
        eprintln!("skipping: mlir tools not found");
        return;
    }
    let src = "let x: Tensor[f32,(2,2)] = 1; tensor.sum(x + 2)";

    let cpu = std::process::Command::new("cargo")
        .args(["run", "--quiet", "--features", "cpu-exec", "--", "eval", "--exec", src])
        .output()
        .expect("run cpu exec");
    assert!(cpu.status.success(), "cpu stderr: {}", String::from_utf8_lossy(&cpu.stderr));
    let cpu_stdout = String::from_utf8_lossy(&cpu.stdout).trim().to_string();

    let mlir = std::process::Command::new("cargo")
        .args(["run", "--quiet", "--features", "mlir-exec", "--", "eval", "--mlir-exec", src])
        .output()
        .expect("run mlir exec");
    assert!(mlir.status.success(), "mlir stderr: {}", String::from_utf8_lossy(&mlir.stderr));
    let mlir_stdout = String::from_utf8_lossy(&mlir.stdout).trim().to_string();

    assert!(
        !cpu_stdout.is_empty(),
        "cpu stdout empty: stderr={}",
        String::from_utf8_lossy(&cpu.stderr)
    );
    assert_eq!(cpu_stdout, mlir_stdout, "cpu vs mlir mismatch");
}
