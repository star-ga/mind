// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Shared driver source and property checks for the json-canon-kat
//! cross-substrate workload (canonical JSON dump, CRX-04).

/// The json-canon-kat driver: imports std.json + std.json_canon and exposes
/// `json_canon_kat(input_addr, input_len) -> i64`, the polynomial hash
/// (`acc = acc*31 + byte`) of the canonical dump bytes.
pub const DRIVER_SRC: &str = "import std.json;\n\
     import std.json_canon;\n\
     pub fn json_canon_kat(input_addr: i64, input_len: i64) -> i64 {\n\
     \x20   let h: i64 = json.jv_parse(input_addr, input_len);\n\
     \x20   if h == 0 {\n\
     \x20       return 0;\n\
     \x20   }\n\
     \x20   let r: i64 = json_canon.jv_dump_canonical(h);\n\
     \x20   if r == 0 {\n\
     \x20       return 0;\n\
     \x20   }\n\
     \x20   let addr: i64 = __mind_load_i64(r + 0);\n\
     \x20   let len: i64 = __mind_load_i64(r + 8);\n\
     \x20   let mut i: i64 = 0;\n\
     \x20   let mut acc: i64 = 1;\n\
     \x20   while i < len {\n\
     \x20       acc = acc * 31 + __mind_load_i8(addr + i);\n\
     \x20       i = i + 1;\n\
     \x20   }\n\
     \x20   acc\n\
     }\n";

pub type JsonCanonFn = unsafe extern "C" fn(i64, i64) -> i64;

/// True when the MLIR toolchain the driver build needs is on PATH.
/// Reports a skip through the gate harness and returns false otherwise.
pub fn require_toolchain(target: &str) -> bool {
    for tool in ["mlir-opt", "mlir-translate", "clang"] {
        if which::which(tool).is_err() {
            crate::common::gate::skipped(
                target,
                &format!(
                    "{tool} not on PATH; install the MLIR toolchain \
                     (mlir-opt / mlir-translate / clang) on this runner"
                ),
            );
            return false;
        }
    }
    true
}

/// The three pinned properties of the canonical dump. Returns the polynomial
/// hash of the canonical bytes of the shuffled-order reference document.
///
/// 1. shuffled insertion order gives identical bytes
/// 2. depth cap refused (deeply-nested document does not dump)
/// 3. non-integer numbers refused
///
/// `f` is `json_canon_kat(input_addr, input_len)` from the compiled driver.
pub fn check_properties(f: JsonCanonFn, id: &str) -> i64 {
    // Property 1: shuffled insertion order gives identical bytes.
    // Both documents are the same logical object with different key order.
    let input_a = br#"{"z":"\ud83d\ude00","y":2,"m":"\ue000","a":1}"#;
    let input_b = br#"{"a":1,"m":"\ue000","y":2,"z":"\ud83d\ude00"}"#;
    let ha = unsafe { f(input_a.as_ptr() as i64, input_a.len() as i64) };
    let hb = unsafe { f(input_b.as_ptr() as i64, input_b.len() as i64) };
    assert_ne!(
        ha, 0,
        "{id}: shuffled input A must produce a canonical dump (got 0 = refused)"
    );
    assert_ne!(
        hb, 0,
        "{id}: shuffled input B must produce a canonical dump (got 0 = refused)"
    );
    assert_eq!(
        ha, hb,
        "{id}: shuffled insertion order must produce identical canonical bytes \
         (hash_a={ha}, hash_b={hb})"
    );

    // Property 2: depth cap refused — a document nested deeper than the
    // parser/dump cap must not produce a dump.
    let deep = format!("{}{}", "[".repeat(600), "]".repeat(600));
    let hd = unsafe { f(deep.as_ptr() as i64, deep.len() as i64) };
    assert_eq!(
        hd, 0,
        "{id}: depth cap must refuse a deeply-nested document (got {hd}, want 0)"
    );

    // Property 3: non-integer numbers refused — 1.5 is not an integer, so
    // the canonical dump refuses it.
    let non_int = br#"{"n":1.5}"#;
    let hn = unsafe { f(non_int.as_ptr() as i64, non_int.len() as i64) };
    assert_eq!(
        hn, 0,
        "{id}: non-integer number must be refused (got {hn}, want 0)"
    );

    ha
}
