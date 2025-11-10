#[cfg(all(feature = "cpu-exec", feature = "cpu-conv"))]
#[test]
fn conv2d_valid_runs() {
    use std::process::Command;

    let src = r#"
        let x: Tensor[f32,(1,3,3,1)] = 1;
        let w: Tensor[f32,(2,2,1,1)] = 1;
        tensor.conv2d(x, w, stride_h=1, stride_w=1, padding="valid")
    "#;

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--features", "cpu-exec cpu-conv", "--", "eval", "--exec", src])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("(1,2,2,1)"));
}
