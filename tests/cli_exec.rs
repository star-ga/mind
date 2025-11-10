#[cfg(feature = "cpu-exec")]
#[test]
fn cli_runs_exec() {
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "cpu-exec",
            "--",
            "eval",
            "--exec",
            "let x: Tensor[f32,(2,2)] = 1; tensor.matmul(x,x)",
        ])
        .output()
        .expect("run");
    assert!(output.status.success(), "process failed: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Tensor["));
    assert!(stdout.contains("(2,2)"));
    assert!(stdout.contains("materialized"));
}
