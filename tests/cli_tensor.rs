use std::process::Command;

#[test]
fn cli_prints_tensor_preview() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--no-default-features",
            "--",
            "eval",
            "let x: Tensor[f32,(2,3)] = 0; x + 1",
        ])
        .output()
        .expect("cargo run");
    assert!(output.status.success(), "process failed: {:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Tensor["));
    assert!(stdout.contains("(2,3)"));
    assert!(stdout.contains("fill=1"));
}
