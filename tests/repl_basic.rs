use std::process::{Command, Stdio};
use std::io::Write;

#[test]
fn repl_accepts_statements_and_expressions() {
    let mut child = Command::new("cargo")
        .args(&["run", "--quiet", "--"])
        .arg("repl")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");

    let mut stdin = child.stdin.take().expect("stdin");
    // Feed a small session
    writeln!(stdin, "let x = 2;").unwrap();
    writeln!(stdin, "x * 3").unwrap();
    writeln!(stdin, ":quit").unwrap();
    drop(stdin);

    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    // Expect to see the result '6' somewhere in the output
    assert!(stdout.contains("6"), "stdout was: {}", stdout);
}
