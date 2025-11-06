use std::process::Command;

#[test]
fn mind_eval_basic_expr() {
    let output = Command::new("cargo")
        .args(["run", "--quiet", "--", "eval", "2 + 3 * 4"])
        .output()
        .expect("run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "14");
}
