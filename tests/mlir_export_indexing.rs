use mind::eval;
use mind::parser;

#[test]
fn mlir_export_handles_index_slice_and_gather() {
    let src = r#"
        let x: Tensor[f32,(4,4)] = 0;
        let idxs: Tensor[i32,(2)] = 0;
        let slice = tensor.slice(x, axis=0, start=1, end=3);
        let strided = tensor.slice_stride(x, axis=1, start=0, end=4, step=2);
        let picked = tensor.index(strided, axis=0, i=1);
        tensor.gather(slice, axis=0, idx=idxs)
    "#;
    let module = parser::parse(src).expect("parse indexing module");
    let ir = eval::lower_to_ir(&module);
    let mlir = eval::to_mlir(&ir, "main");

    assert!(
        mlir.contains("tensor.extract"),
        "expected tensor.extract in {mlir}"
    );
    assert!(
        mlir.contains("tensor.extract_slice"),
        "expected tensor.extract_slice in {mlir}"
    );
    assert!(
        mlir.contains("tensor.insert"),
        "expected tensor.insert in {mlir}"
    );
    assert!(mlir.contains("scf.for"), "expected scf.for in {mlir}");
}
