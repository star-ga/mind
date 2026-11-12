//! Regression tests for the two fail-closed CLI guards that shipped WITHOUT one.
//!
//! Both guards were added because a command reported SUCCESS for work it had not done:
//!
//!   * `mindc build` embedded a module that failed to type-check as a runtime-JIT
//!     fallback and exited 0 (#244). `mindc check` rejects the same source.
//!   * `mindc test` reported PASS for assertions that were never evaluated -- inside a
//!     loop, inside a match arm, inside a `region { }` -- and for test files that did
//!     not parse at all.
//!
//! A behavioural guard with no test is a guard that can be silently reverted, which is
//! the same class of defect the guards themselves close. Each test below asserts the
//! EXIT CODE, because that is what a CI step reads.

use std::path::PathBuf;
use std::process::Command;

fn mindc() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mindc"))
}

fn write_case(stem: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mind_failclosed_{stem}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create case dir");
    for (name, body) in files {
        std::fs::write(dir.join(name), body).expect("write fixture");
    }
    dir
}

/// #244 — source that `check` REJECTS must not `build` at exit 0.
#[test]
fn build_fails_closed_on_source_check_rejects() {
    let dir = write_case("build244", &[("broken.mind", "fn broken( -> { \n")]);
    let src = dir.join("broken.mind");

    let check = mindc()
        .arg("check")
        .arg(&src)
        .output()
        .expect("spawn check");
    assert_eq!(
        check.status.code(),
        Some(1),
        "precondition: `mindc check` must reject this source, else the test proves nothing"
    );

    let build = mindc()
        .args(["build", "--backend", "mlir"])
        .arg(&src)
        .output()
        .expect("spawn build");
    assert_ne!(
        build.status.code(),
        Some(0),
        "`mindc build` exited 0 on source `mindc check` rejects -- the JIT-fallback \
         false green is back. stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
}

/// #244 — a warning-only self-host intrinsic must not become a successful
/// runtime-JIT fallback artifact. The checker owns E2024 as a warning; the
/// build layer owns the structured E5005 refusal.
#[test]
fn build_fails_closed_on_unregistered_mind_intrinsic() {
    let dir = write_case(
        "build244_nerve_route",
        &[(
            "a4.mind",
            "fn main() -> i32 {\n    let h: i64 = __mind_nerve_route(1);\n    return h;\n}\n",
        )],
    );
    let src = dir.join("a4.mind");
    let artifact = dir.join("a4");

    let build = mindc()
        .args(["build", "--backend", "mlir", "--out"])
        .arg(&artifact)
        .arg(&src)
        .output()
        .expect("spawn build");
    let stderr = String::from_utf8_lossy(&build.stderr);

    assert_eq!(
        build.status.code(),
        Some(1),
        "an unsupported __mind intrinsic must fail the build: {stderr}"
    );
    assert!(
        stderr.contains("[E2024]") && stderr.contains("__mind_nerve_route"),
        "the checker warning must identify the actual unsupported intrinsic: {stderr}"
    );
    assert!(
        stderr.contains("error[build][E5005]")
            && stderr.contains("refusing to link a runtime-JIT fallback object"),
        "the build must report the structured source-fallback refusal: {stderr}"
    );
    assert!(
        !artifact.exists(),
        "a refused fallback build must leave no final artifact at {}",
        artifact.display()
    );
}

/// A valid program must still build -- the guard must not be a blanket refusal.
#[test]
fn build_still_succeeds_on_valid_source() {
    let dir = write_case(
        "buildok",
        &[("ok.mind", "fn main() -> i64 { return 42; }\n")],
    );
    let out = mindc()
        .args(["build", "--backend", "mlir"])
        .arg(dir.join("ok.mind"))
        .output()
        .expect("spawn build");
    assert_eq!(
        out.status.code(),
        Some(0),
        "a valid program failed to build: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// An unparseable file in the suite is a broken test, never a skip.
#[test]
fn test_fails_closed_on_unparseable_file() {
    let dir = write_case(
        "unparseable",
        &[
            ("broken.mind", "fn broken( { this is not mind\n"),
            ("good.mind", "#[test]\nfn ok() -> bool { true }\n"),
        ],
    );
    let out = mindc().arg("test").arg(&dir).output().expect("spawn test");
    assert_ne!(
        out.status.code(),
        Some(0),
        "`mindc test` exited 0 with a file in the suite that does not parse"
    );
}

/// An assertion the runner cannot evaluate must not be reported as passed.
#[test]
fn test_fails_closed_on_assert_in_unexecutable_body() {
    for (stem, body) in [
        ("loop", "for i in 0..1 {\n        assert 0 == 1\n    }"),
        ("region", "region {\n        assert 0 == 1\n    }"),
    ] {
        let dir = write_case(
            &format!("hidden_{stem}"),
            &[(
                "t.mind",
                &format!("#[test]\nfn hidden() {{\n    {body}\n}}\n"),
            )],
        );
        let out = mindc().arg("test").arg(&dir).output().expect("spawn test");
        assert_ne!(
            out.status.code(),
            Some(0),
            "`mindc test` exited 0 for a FALSE assertion inside `{stem}` -- the assertion \
             was never evaluated and that read as a pass. stdout: {}",
            String::from_utf8_lossy(&out.stdout)
        );
    }
}

/// Assertions are evaluated AT THEIR POINT, not against final state (#241).
#[test]
fn assertions_see_state_at_their_own_point() {
    let dir = write_case(
        "seq",
        &[(
            "t.mind",
            "#[test]\nfn correct_at_its_point() {\n    let mut x: i64 = 1\n    assert x == 1\n    x = 2\n}\n",
        )],
    );
    let out = mindc().arg("test").arg(&dir).output().expect("spawn test");
    assert_eq!(
        out.status.code(),
        Some(0),
        "a test whose assertion was TRUE when written was reported as failing -- the \
         runner is evaluating against final state again. stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}
