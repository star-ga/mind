use mind::eval;
use mind::parser;

#[test]
fn lower_and_eval_add_ints() {
    let src = "1 + 2 * 3";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert_eq!(rendered, "7");
}

#[test]
fn lower_tensor_preview() {
    let src = "let x: Tensor[f32,(2,3)] = 0; x + 1";
    let module = parser::parse(src).unwrap();
    let ir = eval::lower_to_ir(&module);
    let value = eval::eval_ir(&ir);
    let rendered = eval::format_value_human(&value);
    assert!(rendered.contains("Tensor["), "{rendered}");
}
