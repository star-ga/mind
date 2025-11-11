use mind::eval;
use mind::parser;

#[test]
fn mlir_export_covers_shape_ops() {
    let src = r#"
        let x: Tensor[f32,(2,3)] = 0;
        let reshaped = tensor.reshape(x, (3,2));
        let expanded = tensor.expand_dims(reshaped, axis=1);
        let squeezed = tensor.squeeze(expanded, axes=[1]);
        tensor.transpose(squeezed, axes=[1,0])
    "#;
    let module = parser::parse(src).expect("parse shape module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(
        mlir.contains("tensor.reshape"),
        "expected tensor.reshape in {mlir}"
    );
    assert!(
        mlir.contains("linalg.transpose"),
        "expected linalg.transpose in {mlir}"
    );
}
