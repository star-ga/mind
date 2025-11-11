use std::collections::HashMap;

use mind::diagnostics;

use mind::parser;

use mind::type_checker;

#[test]
fn unknown_ident_points_to_name() {
    let src = "let n: i32 = x + 1";
    let module = parser::parse_with_diagnostics(src).expect("parse failed");
    let diags = type_checker::check_module_types(&module, src, &HashMap::new());
    assert!(!diags.is_empty(), "expected type error diagnostic");
    let rendered = diagnostics::render(src, &diags[0]);
    assert!(
        rendered.contains("x + 1"),
        "diagnostic missing offending line: {rendered}"
    );
    let line = "let n: i32 = x + 1";
    let x_idx = line.find('x').unwrap();
    let caret_line = rendered.lines().last().unwrap_or("");
    let caret_pos = caret_line.find('^').unwrap_or(usize::MAX);
    assert!(
        caret_pos >= x_idx.saturating_sub(2),
        "caret not near identifier: {rendered}"
    );
}
