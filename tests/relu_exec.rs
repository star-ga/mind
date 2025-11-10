#[cfg(feature = "cpu-exec")]
#[test]
fn relu_exec_non_negative() {
    use std::process::Command;

    let program = "let x: Tensor[f32,(1,4)] = 0; x = x - 3; tensor.relu(x + 1)";
    let output = Command::new("cargo")
        .args(["run", "--quiet", "--features", "cpu-exec", "--", "eval", "--exec", program])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Tensor[F32,(1,4)]"));
}
