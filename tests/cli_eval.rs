use std::process::Command;

#[cfg(not(debug_assertions))]
#[ignore]
#[test]
fn _ignore_in_release_mode() {}

#[test]
fn mind_eval_basic_expr() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--no-default-features",
            "--",
            "eval",
            "2 + 3 * 4",
        ])
        .output()
        .expect("run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    assert!(trimmed.contains("--- Lowered IR ---"), "{trimmed}");
    assert!(trimmed.contains("--- Result ---"), "{trimmed}");
    assert!(trimmed.ends_with("14"), "{trimmed}");
}
