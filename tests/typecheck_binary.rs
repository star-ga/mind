use mind::parser;
use mind::type_checker;
use mind::types::ValueType;

use std::collections::HashMap;

#[test]
fn scalars_ok() {
    let src = "1 + 2 * 3";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    assert!(diags.is_empty());
}

#[test]
fn unknown_ident_reports_error() {
    let src = "y + 1";
    let module = parser::parse(src).unwrap();
    let env: HashMap<String, ValueType> = HashMap::new();
    let diags = type_checker::check_module_types(&module, src, &env);
    assert!(!diags.is_empty());
}
