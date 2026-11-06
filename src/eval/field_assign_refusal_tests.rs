use super::*;

use crate::parser;
use std::collections::HashMap;

fn eval_src(src: &str, types: bool) -> Result<Value, EvalError> {
    let module = parser::parse(src).expect("field-assignment fixture parses");
    let mut env = HashMap::new();
    eval_module_value_with_env(&module, &mut env, types.then_some(src))
}

/// The store must land in the record: a later read sees it. (The evaluator
/// used to refuse field stores; before that it returned the RHS and dropped
/// the write, so a read-back is the property that matters.)
#[test]
fn eval_field_assignment_stores_into_the_record() {
    let src = r#"
struct Point { x: i64 }

fn mutate() -> i64 {
    let mut p: Point = Point { x: 1 };
    p.x = 9;
    return p.x;
}

mutate()
"#;
    assert_eq!(eval_src(src, true).unwrap(), Value::Int(9));
}

#[test]
fn eval_module_level_field_assignment_stores() {
    let src = r#"
struct Point { x: i64 }
let mut p: Point = Point { x: 1 };
p.x = 9;
p.x
"#;
    // This test targets the module executor's dispatch. Its hand-built
    // annotation is outside the evaluator's source type-check surface;
    // bypass that unrelated gate so the FieldAssign arm is exercised.
    assert_eq!(eval_src(src, false).unwrap(), Value::Int(9));
}

/// Record identity: a second binding and a callee see the same record.
#[test]
fn eval_field_store_is_visible_through_every_reference() {
    let src = r#"
struct Point { x: i64, y: i64 }

fn bump(p: &mut Point) {
    p.x = p.x + 10;
}

fn run() -> i64 {
    let mut p: Point = Point { x: 1, y: 2 };
    let mut q: Point = p;
    q.x = 5;
    bump(&mut p);
    return p.x * 10 + q.y;
}

run()
"#;
    assert_eq!(eval_src(src, true).unwrap(), Value::Int(152));
}

/// A fixed-array field is a value container: an element store changes the
/// record's field and not an array copied out of it before the store.
#[test]
fn eval_fixed_array_field_element_store() {
    let src = r#"
struct Bank { xs: [i64; 3] }

fn run() -> i64 {
    let mut b: Bank = Bank { xs: [1, 2, 3] };
    let before: [i64; 3] = b.xs;
    b.xs[1] = 20;
    return b.xs[1] * 10 + before[1];
}

run()
"#;
    assert_eq!(eval_src(src, false).unwrap(), Value::Int(202));
}

#[test]
fn eval_field_assignment_refuses_a_non_struct_receiver_and_a_missing_field() {
    let err = eval_src("let mut n = 5;\nn.x = 1;\n", false).unwrap_err();
    assert!(
        err.to_string()
            .contains("struct field assignment `.x` needs a struct receiver, got an integer"),
        "unexpected refusal: {err}"
    );
    let src = "struct Point { x: i64 }\nlet mut p: Point = Point { x: 1 };\np.z = 9;\n";
    let err = eval_src(src, false).unwrap_err();
    assert!(
        err.to_string().contains("struct `Point` has no field .z"),
        "unexpected refusal: {err}"
    );
}
