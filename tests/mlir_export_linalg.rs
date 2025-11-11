use mind::eval;
use mind::parser;

#[test]
fn mlir_export_emits_linalg_dot_and_matmul() {
    let src = r#"
        let a: Tensor[f32,(2)] = 1;
        let b: Tensor[f32,(2)] = 1;
        let dot_val = tensor.dot(a, b);
        let m: Tensor[f32,(2,3)] = 1;
        let n: Tensor[f32,(3,4)] = 1;
        let mat = tensor.matmul(m, n);
        mat
    "#;
    let module = parser::parse(src).expect("parse linalg module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(mlir.contains("linalg.dot"), "expected linalg.dot in {mlir}");
    assert!(
        mlir.contains("linalg.matmul"),
        "expected linalg.matmul in {mlir}"
    );
}
