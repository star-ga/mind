use mind::parser;

#[test]
fn shows_pretty_error_for_unexpected_paren() {
    let src = ")";
    let Err(diags) = parser::parse_with_diagnostics(src) else { panic!("expected error"); };
    let joined = diags
        .iter()
        .map(|d| mind::diagnostics::render(src, d))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("error"));
    assert!(joined.contains("line 1"));
    assert!(joined.contains("^")); // caret present
}

#[test]
fn shows_error_for_unclosed_paren() {
    let src = "(";
    let Err(diags) = parser::parse_with_diagnostics(src) else { panic!("expected error"); };
    let s = mind::diagnostics::render(src, &diags[0]);
    assert!(s.contains("line 1"));
    assert!(s.contains("^"));
}
