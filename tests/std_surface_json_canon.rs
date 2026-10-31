// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! CRX-04 — canonical JSON dump KATs (`std/json_canon.mind`).
//!
//! The NAMED rule (byte-equal to the documented dump):
//!   - object keys sorted by UTF-16 CODE-UNIT order (not UTF-8 byte order)
//!   - no whitespace (compact) or indent=2 Python pretty (`jv_dump_pretty_py`)
//!   - integers only: a non-integral number makes the dump REFUSE (return 0)
//!   - `ensure_ascii`: every code point > U+007F escapes to `\uXXXX` lowercase
//!     hex with UTF-16 surrogate pairs above U+FFFF; control chars incl. U+007F
//!     escape to `\u00XX` except `\n \r \t \b \f \" \\`; `/` unescaped
//!   - `-0` renders `0`
//!
//! Expected bytes were generated ONCE from Python `json.dumps(...,
//! ensure_ascii=True)` with the key order taken from the NAMED rule (UTF-16
//! code units). For key sets that mix non-BMP code points with U+E000..U+FFFF
//! the NAMED order differs from Python's `sort_keys=True` (which is code-point
//! order, i.e. UTF-8 byte order); the `mixed_utf16_order` KAT pins that
//! distinction and is what makes the "swap the comparator to UTF-8 byte order"
//! mutant go red.
//!
//! Gate: `cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_surface_json_canon`

#![cfg(feature = "std-surface")]

mod common;

/// (input JSON text, expected compact bytes, expected pretty bytes).
/// Pretty = Python `json.dumps(sort_keys-by-utf16, indent=2, ensure_ascii=True)`.
const KATS: &[(&str, &str, &str)] = &[
    // Shuffled insertion order — the same logical object must produce the same
    // bytes regardless of the order the parser saw the keys in.
    (
        r#"{"b":1,"a":2,"c":3}"#,
        r#"{"a":2,"b":1,"c":3}"#,
        "{\n  \"a\": 2,\n  \"b\": 1,\n  \"c\": 3\n}",
    ),
    // Non-ASCII in a key and a value: ensure_ascii escapes both.
    (
        r#"{"\u00e9":"\u00e9"}"#,
        r#"{"\u00e9":"\u00e9"}"#,
        "{\n  \"\\u00e9\": \"\\u00e9\"\n}",
    ),
    // Non-BMP key: surrogate pair in the escape, and in the sort key.
    (
        r#"{"\ud83d\ude00":1}"#,
        r#"{"\ud83d\ude00":1}"#,
        "{\n  \"\\ud83d\\ude00\": 1\n}",
    ),
    // THE comparator fixture: U+1F600 vs U+E000. UTF-16 code-unit order puts
    // the non-BMP key FIRST (its lead surrogate U+D83D < U+E000); UTF-8 byte
    // order (== code-point order == Python sort_keys) puts U+E000 first. The
    // NAMED rule is UTF-16 code-unit order.
    (
        r#"{"\ue000":2,"\ud83d\ude00":1}"#,
        r#"{"\ud83d\ude00":1,"\ue000":2}"#,
        "{\n  \"\\ud83d\\ude00\": 1,\n  \"\\ue000\": 2\n}",
    ),
    // Control character U+001F -> \u001f (lowercase hex).
    (
        r#"{"k":"\u001f"}"#,
        r#"{"k":"\u001f"}"#,
        "{\n  \"k\": \"\\u001f\"\n}",
    ),
    // -0 renders 0 (integer zero is unsigned in the canonical dump).
    (r#"{"n":-0}"#, r#"{"n":0}"#, "{\n  \"n\": 0\n}"),
    // 2^63-1.
    (
        r#"{"n":9223372036854775807}"#,
        r#"{"n":9223372036854775807}"#,
        "{\n  \"n\": 9223372036854775807\n}",
    ),
    // The seven named specials: " \\ / and \n \r \t \b \f. `/` stays literal.
    (
        r#"{"s":"\"\\/\n\r\t\b\f"}"#,
        r#"{"s":"\"\\/\n\r\t\b\f"}"#,
        "{\n  \"s\": \"\\\"\\\\/\\n\\r\\t\\b\\f\"\n}",
    ),
    // U+007F is a control char: \u007f, same as the Python oracle.
    (
        r#"{"s":"\u007f"}"#,
        r#"{"s":"\u007f"}"#,
        "{\n  \"s\": \"\\u007f\"\n}",
    ),
    // Empty containers, no inner newline in pretty.
    (
        r#"{"o":{},"a":[]}"#,
        r#"{"a":[],"o":{}}"#,
        "{\n  \"a\": [],\n  \"o\": {}\n}",
    ),
    // Bools, null, nested sort (both levels), array inside object.
    (
        r#"{"z":{"y":[1,{"x":2}]},"a":true,"m":null,"q":false}"#,
        r#"{"a":true,"m":null,"q":false,"z":{"y":[1,{"x":2}]}}"#,
        "{\n  \"a\": true,\n  \"m\": null,\n  \"q\": false,\n  \"z\": {\n    \"y\": [\n      1,\n      {\n        \"x\": 2\n      }\n    ]\n  }\n}",
    ),
    // Top-level array keeps its order (arrays are not sorted).
    ("[3,1,2]", "[3,1,2]", "[\n  3,\n  1,\n  2\n]"),
    // true/false/null/0 in an array.
    (
        "[true,false,null,0]",
        "[true,false,null,0]",
        "[\n  true,\n  false,\n  null,\n  0\n]",
    ),
    // Nested object key sort at depth 2.
    (
        r#"{"b":{"d":4,"c":3},"a":{"b":2,"a":1}}"#,
        r#"{"a":{"a":1,"b":2},"b":{"c":3,"d":4}}"#,
        "{\n  \"a\": {\n    \"a\": 1,\n    \"b\": 2\n  },\n  \"b\": {\n    \"c\": 3,\n    \"d\": 4\n  }\n}",
    ),
    // Non-BMP VALUE (not key): surrogate pair, no sorting involved.
    (
        r#"{"k":"\ud83d\ude00"}"#,
        r#"{"k":"\ud83d\ude00"}"#,
        "{\n  \"k\": \"\\ud83d\\ude00\"\n}",
    ),
    // U+0000 in a value: \u0000.
    (
        r#"{"k":"\u0000"}"#,
        r#"{"k":"\u0000"}"#,
        "{\n  \"k\": \"\\u0000\"\n}",
    ),
];

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod kat {
    use super::KATS;
    use super::common::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::OnceLock;

    /// Compile one driver .so that exposes the json_canon surface. Returns the
    /// library path, or None when the gate self-skips (no mindc / no toolchain).
    /// Cached so parallel tests share one compilation (avoids a write/read race
    /// on the .so path).
    fn build_driver() -> Option<&'static PathBuf> {
        static SO: OnceLock<Option<PathBuf>> = OnceLock::new();
        SO.get_or_init(|| {
            let mindc = mindc_bin();
            if !mindc.exists() {
                crate::common::gate::skipped(
                    "std_surface_json_canon",
                    "json_canon KAT driver: mindc not found; skipping",
                );
                return None;
            }
            let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("std_surface_json_canon");
            std::fs::create_dir_all(&out_dir).expect("create output dir");
            let driver_src = "import std.json;\n\
                 import std.json_canon;\n\
                 pub fn smoke_jv_parse(buf: i64, buf_len: i64) -> i64 {\n\
                 \x20   json.jv_parse(buf, buf_len)\n\
                 }\n\
                 pub fn smoke_jv_dump_canonical(h: i64) -> i64 {\n\
                 \x20   json_canon.jv_dump_canonical(h)\n\
                 }\n\
                 pub fn smoke_jv_dump_pretty_py(h: i64) -> i64 {\n\
                 \x20   json_canon.jv_dump_pretty_py(h)\n\
                 }\n\
                 pub fn smoke_jv_str_addr(r: i64) -> i64 { __mind_load_i64(r + 0) }\n\
                 pub fn smoke_jv_str_len(r: i64) -> i64 { __mind_load_i64(r + 8) }\n";
            let driver_path = out_dir.join("json_canon_kat.mind");
            let so_path = out_dir.join("libjson_canon_kat.so");
            std::fs::write(&driver_path, driver_src).expect("write driver MIND");
            let compile_out = Command::new(&mindc)
                .args([
                    driver_path.to_str().unwrap(),
                    "--emit-shared",
                    so_path.to_str().unwrap(),
                ])
                .output()
                .expect("run mindc");
            if !crate::common::gate::compiled("std_surface_json_canon", &compile_out) {
                return None;
            }
            Some(so_path)
        })
        .as_ref()
    }

    struct Ctx {
        _lib: libloading::Library,
        parse: unsafe extern "C" fn(i64, i64) -> i64,
        canon: unsafe extern "C" fn(i64) -> i64,
        pretty: unsafe extern "C" fn(i64) -> i64,
        str_addr: unsafe extern "C" fn(i64) -> i64,
        str_len: unsafe extern "C" fn(i64) -> i64,
    }

    fn open_ctx(so: &PathBuf) -> Ctx {
        unsafe {
            let lib = libloading::Library::new(so).expect("dlopen json_canon_kat.so");
            let parse: unsafe extern "C" fn(i64, i64) -> i64 =
                *lib.get(b"smoke_jv_parse\0").unwrap();
            let canon: unsafe extern "C" fn(i64) -> i64 =
                *lib.get(b"smoke_jv_dump_canonical\0").unwrap();
            let pretty: unsafe extern "C" fn(i64) -> i64 =
                *lib.get(b"smoke_jv_dump_pretty_py\0").unwrap();
            let str_addr: unsafe extern "C" fn(i64) -> i64 =
                *lib.get(b"smoke_jv_str_addr\0").unwrap();
            let str_len: unsafe extern "C" fn(i64) -> i64 =
                *lib.get(b"smoke_jv_str_len\0").unwrap();
            Ctx {
                _lib: lib,
                parse,
                canon,
                pretty,
                str_addr,
                str_len,
            }
        }
    }

    impl Ctx {
        fn parse(&self, src: &str) -> i64 {
            let b = src.as_bytes();
            unsafe { (self.parse)(b.as_ptr() as i64, b.len() as i64) }
        }
        fn record_bytes(&self, rec: i64) -> Vec<u8> {
            if rec == 0 {
                return Vec::new();
            }
            let len = unsafe { (self.str_len)(rec) };
            if len == 0 {
                return Vec::new();
            }
            let addr = unsafe { (self.str_addr)(rec) };
            unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) }.to_vec()
        }
        fn canon_bytes(&self, h: i64) -> Option<Vec<u8>> {
            let rec = unsafe { (self.canon)(h) };
            if rec == 0 {
                return None;
            }
            Some(self.record_bytes(rec))
        }
        fn pretty_bytes(&self, h: i64) -> Option<Vec<u8>> {
            let rec = unsafe { (self.pretty)(h) };
            if rec == 0 {
                return None;
            }
            Some(self.record_bytes(rec))
        }
    }

    /// Compact canonical dump: byte-equal to the documented dump on every KAT.
    #[test]
    fn jv_dump_canonical_matches_documented_dump() {
        let Some(so) = build_driver() else {
            return;
        };
        let ctx = open_ctx(so);
        for (i, (input, expect_compact, _)) in KATS.iter().enumerate() {
            let h = ctx.parse(input);
            assert_ne!(h, 0, "KAT {i}: input must parse: {input}");
            let got = ctx.canon_bytes(h).unwrap_or_else(|| {
                panic!("KAT {i}: jv_dump_canonical refused a valid integer-only document: {input}")
            });
            assert_eq!(
                got,
                expect_compact.as_bytes(),
                "KAT {i}: compact bytes mismatch for {input}\n got: {:?}\nwant: {:?}",
                String::from_utf8_lossy(&got),
                expect_compact
            );
        }
    }

    /// Pretty dump: Python indent=2, same key order, same escapes.
    #[test]
    fn jv_dump_pretty_py_matches_documented_dump() {
        let Some(so) = build_driver() else {
            return;
        };
        let ctx = open_ctx(so);
        for (i, (input, _, expect_pretty)) in KATS.iter().enumerate() {
            let h = ctx.parse(input);
            assert_ne!(h, 0, "KAT {i}: input must parse: {input}");
            let got = ctx.pretty_bytes(h).unwrap_or_else(|| {
                panic!("KAT {i}: jv_dump_pretty_py refused a valid integer-only document: {input}")
            });
            assert_eq!(
                got,
                expect_pretty.as_bytes(),
                "KAT {i}: pretty bytes mismatch for {input}\n got: {:?}\nwant: {:?}",
                String::from_utf8_lossy(&got),
                expect_pretty
            );
        }
    }

    /// Key-order independence: re-inserting the same object in a different
    /// order must produce identical bytes (the shuffle half of the acceptance).
    #[test]
    fn shuffled_insertion_order_is_byte_identical() {
        let Some(so) = build_driver() else {
            return;
        };
        let ctx = open_ctx(so);
        let a = ctx.parse(r#"{"z":"\ud83d\ude00","y":2,"m":"\ue000","a":1}"#);
        let b = ctx.parse(r#"{"a":1,"m":"\ue000","y":2,"z":"\ud83d\ude00"}"#);
        assert_ne!(a, 0, "document A must parse");
        assert_ne!(b, 0, "document B must parse");
        let ba = ctx.canon_bytes(a).expect("A must dump");
        let bb = ctx.canon_bytes(b).expect("B must dump");
        assert_eq!(
            ba, bb,
            "shuffled insertion order must not change canonical bytes"
        );
    }

    /// Non-integral numbers are REFUSED (handle 0), never rendered.
    #[test]
    fn non_integral_numbers_are_refused() {
        let Some(so) = build_driver() else {
            return;
        };
        let ctx = open_ctx(so);
        for input in [
            "1.5",
            r#"{"n":1.5}"#,
            "0.5",
            r#"{"a":[1.25]}"#,
            r#"{"a":1,"b":2.0}"#, // 2.0 is non-integral in the jv number model
        ] {
            let h = ctx.parse(input);
            assert_ne!(h, 0, "input must parse: {input}");
            assert_eq!(
                ctx.canon_bytes(h),
                None,
                "jv_dump_canonical must refuse non-integral number in {input}"
            );
            assert_eq!(
                ctx.pretty_bytes(h),
                None,
                "jv_dump_pretty_py must refuse non-integral number in {input}"
            );
        }
    }
}
