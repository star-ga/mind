use mind::eval;
use mind::parser;

#[test]
fn mlir_export_basic() {
    let src = "1 + 2 * 3";
    let m = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&m);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(mlir.contains("func.func @main"));
    assert!(mlir.contains("arith.addi"));
    assert!(mlir.contains("return"));
}

#[test]
fn mlir_export_tensor_const() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let m = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&m);
    let mlir = eval::to_mlir(&ir, "main");
    assert!(mlir.contains("tensor.empty"));
    assert!(mlir.contains("linalg.fill"));
}
