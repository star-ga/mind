use mind::pipeline::{compile_source_with_name, CompileError, CompileOptions};
use mind::runtime::types::BackendTarget;

fn compile_opts() -> CompileOptions {
    CompileOptions {
        func: None,
        enable_autodiff: false,
        target: BackendTarget::Cpu,
    }
}

fn expect_shape_code(src: &str, code: &str) {
    let err = compile_source_with_name(src, Some("shape.mind"), &compile_opts())
        .expect_err("program should fail shape validation");
    match err {
        CompileError::TypeError(diags) => {
            let codes: Vec<&str> = diags.iter().map(|d| d.code).collect();
            assert!(
                codes.iter().any(|c| *c == code),
                "expected diagnostic code {code}, saw {codes:?}"
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn elementwise_broadcast_failure_surfaces_shape_code() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        let y: Tensor[f32,(4,3)] = 0;
        x + y
    "#;
    expect_shape_code(src, "E2101");
}

#[test]
fn matmul_inner_dimension_mismatch_reports_shape_code() {
    let src = r#"
        let a: Tensor[f32,(2,3)] = 0;
        let b: Tensor[f32,(5,4)] = 0;
        tensor.matmul(a, b)
    "#;
    expect_shape_code(src, "E2103");
}

#[test]
fn valid_broadcast_program_compiles() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        let y: Tensor[f32,(1,3)] = 0;
        x + y
    "#;
    let products = compile_source_with_name(src, Some("shape_ok.mind"), &compile_opts())
        .expect("valid shapes should compile");
    assert!(!products.ir.instrs.is_empty());
}
