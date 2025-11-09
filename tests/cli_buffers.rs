#[cfg(feature = "cpu-buffers")]
#[test]
fn cli_shows_materialized_note() {
    let out = std::process::Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--features",
            "cpu-buffers",
            "--",
            "eval",
            "let x: Tensor[f32,(2,3)] = 1; tensor.materialize(x)",
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("materialized"));
}
