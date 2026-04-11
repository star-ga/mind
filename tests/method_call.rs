// Copyright 2025 STARGA Inc.

use libmind::eval;
use libmind::parser;

#[test]
fn array_len_method() {
    let m = parser::parse("let x = [1, 2, 3]\nx.len()").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 3);
}

#[test]
fn array_last_method() {
    let m = parser::parse("let x = [10, 20, 30]\nx.last()").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 30);
}

#[test]
fn clone_method() {
    let m = parser::parse("let x = 42\nx.clone()").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 42);
}

#[test]
fn field_access_len() {
    let m = parser::parse("let x = [1, 2]\nx.len").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 2);
}

#[test]
fn chained_method_calls() {
    // a.clone().len() — clone returns a copy, then .len() returns length
    let m = parser::parse("let items = [1, 2, 3]\nitems.clone().len()").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 3);
}

#[test]
fn string_len_method() {
    let m = parser::parse("let s = \"hello\"\ns.len()").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 5);
}

#[test]
fn tensor_sum_still_works() {
    // Existing tensor.sum(x) syntax must still parse
    let m = parser::parse("let t: Tensor[f32,(2,3)] = 1.0\ntensor.sum(t)");
    assert!(m.is_ok(), "tensor.sum should still parse: {:?}", m.err());
}

#[test]
fn method_call_in_let() {
    let m = parser::parse("let x = [1, 2, 3]\nlet n = x.len()\nn").unwrap();
    let v = eval::eval_module(&m).unwrap();
    assert_eq!(v, 3);
}
