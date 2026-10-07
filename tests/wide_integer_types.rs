// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Integer types wider than 64 bits are refused.
//!
//! `let a: i128 = …` type-checked, and the value was lowered as `i64`, so
//! 128-bit arithmetic silently wrapped at 64 bits: `(a * 4 / 2) / a` with
//! `a = i64::MAX` returned 0 instead of 2. No backend represents a wider
//! integer, so the annotation is now an error naming the limit.

mod common;

use std::process::Command;

fn check(target: &str, src: &str) -> (bool, String) {
    let path = common::scratch_dir(target).join("main.mind");
    std::fs::write(&path, src).expect("write source");
    let out = Command::new(common::mindc_bin())
        .arg("check")
        .arg(&path)
        .output()
        .expect("run mindc check");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

#[test]
fn i128_and_u128_annotations_are_refused() {
    for (target, src, name) in [
        (
            "wide_int_let",
            "fn f() -> i64 {\n    let a: i128 = 9223372036854775807;\n    return 0;\n}\n",
            "i128",
        ),
        (
            "wide_int_param",
            "fn f(a: u128) -> i64 {\n    return 0;\n}\n",
            "u128",
        ),
        (
            "wide_int_ret",
            "fn f() -> i256 {\n    return 0;\n}\n",
            "i256",
        ),
    ] {
        let (ok, text) = check(target, src);
        assert!(!ok, "{target}: `{name}` must be refused:\n{text}");
        assert!(
            text.contains(&format!("`{name}` is not supported"))
                && text.contains("at most 64 bits wide"),
            "{target}:\n{text}"
        );
    }
}

#[test]
fn integer_types_up_to_64_bits_still_check() {
    let (ok, text) = check(
        "wide_int_control",
        "fn f(a: u8, b: i16, c: u32, d: i64, e: u64) -> i64 {\n    return d;\n}\n",
    );
    assert!(ok, "{text}");
}
