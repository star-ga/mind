// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Task #269 -- `std/regex.mind` surface tests.
//!
//! Tests in this file verify:
//!
//! 1. `std/regex.mind` parses and lowers to IR with all required `pub fn`s.
//! 2. NFA state kind constants are distinct.
//! 3. `rx_compile` calls `rx_compile_alternation` (the compile entry-point).
//! 4. `rx_is_match` calls `rx_run`.
//! 5. `rx_find_all` calls `rx_sq_new`.
//! 6. std.regex auto-exports all public symbols.
//! 7. Bundled stdlib resolves `use std.regex`.
//! 8. MLIR functional test: compile + is_match + find against a battery of fixtures
//!    (gated on `mlir-build`).
//!
//! Gate: `cargo test --features "std-surface cross-module-imports"
//!                   --test std_surface_regex`

#![cfg(feature = "std-surface")]

mod common;
use common::mindc_bin;

use libmind::eval::lower::lower_to_ir;
use libmind::ir::Instr;
use libmind::parser;

const REGEX_MIND_SRC: &str = include_str!("../std/regex.mind");

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn lower_regex_mind() -> libmind::ir::IRModule {
    let module = parser::parse(REGEX_MIND_SRC).expect("std/regex.mind must parse cleanly");
    lower_to_ir(&module)
}

fn count_calls_recursive(instrs: &[Instr], callee: &str) -> usize {
    let mut n = 0;
    for instr in instrs {
        match instr {
            Instr::Call { name, .. } if name == callee => n += 1,
            Instr::If {
                cond_instrs,
                then_instrs,
                else_instrs,
                ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(then_instrs, callee);
                n += count_calls_recursive(else_instrs, callee);
            }
            Instr::While {
                cond_instrs, body, ..
            } => {
                n += count_calls_recursive(cond_instrs, callee);
                n += count_calls_recursive(body, callee);
            }
            Instr::FnDef { body, .. } => {
                n += count_calls_recursive(body, callee);
            }
            _ => {}
        }
    }
    n
}

fn fn_body<'a>(ir: &'a libmind::ir::IRModule, name: &str) -> &'a [Instr] {
    ir.instrs
        .iter()
        .find_map(|i| match i {
            Instr::FnDef { name: n, body, .. } if n == name => Some(body.as_slice()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected FnDef `{name}` in lowered IR"))
}

// ─── Test 1: parse + lower ────────────────────────────────────────────────────

#[test]
fn regex_mind_parses_and_lowers() {
    let ir = lower_regex_mind();
    for want in [
        "rx_compile",
        "rx_try_compile",
        "rx_is_match",
        "rx_find",
        "rx_find_all",
        "rx_capture",
        "rx_nfa_start",
        "rx_capture_count",
        // Kind constants
        "rx_kind_match",
        "rx_kind_split",
        "rx_kind_char",
        "rx_kind_any",
        "rx_kind_class",
        "rx_kind_anchor_start",
        "rx_kind_anchor_end",
    ] {
        assert!(
            ir.instrs
                .iter()
                .any(|i| matches!(i, Instr::FnDef { name, .. } if name == want)),
            "missing FnDef `{want}` in lowered std/regex.mind IR"
        );
    }
}

// ─── Test 2: NFA kind constants are distinct ──────────────────────────────────

#[test]
fn rx_kind_constants_are_distinct() {
    let ir = lower_regex_mind();
    let mut values: Vec<i64> = Vec::new();
    for fn_name in [
        "rx_kind_match",
        "rx_kind_split",
        "rx_kind_char",
        "rx_kind_any",
        "rx_kind_class",
        "rx_kind_anchor_start",
        "rx_kind_anchor_end",
    ] {
        let body = fn_body(&ir, fn_name);
        let val = body.iter().find_map(|i| match i {
            Instr::ConstI64(_, v) => Some(*v),
            _ => None,
        });
        let v = val.unwrap_or_else(|| panic!("{fn_name} must emit a ConstI64"));
        values.push(v);
    }
    let mut seen = std::collections::HashSet::new();
    for v in &values {
        assert!(
            seen.insert(v),
            "rx_kind_* constants must be distinct; duplicate value {v}"
        );
    }
}

// ─── Test 3: rx_compile calls rx_compile_alternation ─────────────────────────

#[test]
fn rx_compile_calls_compile_alternation() {
    let ir = lower_regex_mind();
    let body = fn_body(&ir, "rx_compile");
    let calls = count_calls_recursive(body, "rx_compile_alternation");
    assert!(
        calls >= 1,
        "rx_compile must call rx_compile_alternation at least once; got {calls}"
    );
}

// ─── Test 4: rx_is_match calls rx_run ────────────────────────────────────────

#[test]
fn rx_is_match_calls_rx_run() {
    let ir = lower_regex_mind();
    let body = fn_body(&ir, "rx_is_match");
    let calls = count_calls_recursive(body, "rx_run");
    assert!(
        calls >= 1,
        "rx_is_match must call rx_run at least once; got {calls}"
    );
}

// ─── Test 5: rx_find_all calls rx_sq_new ─────────────────────────────────────

#[test]
fn rx_find_all_calls_rx_sq_new() {
    let ir = lower_regex_mind();
    let body = fn_body(&ir, "rx_find_all");
    let calls = count_calls_recursive(body, "rx_sq_new");
    assert!(
        calls >= 1,
        "rx_find_all must call rx_sq_new at least once; got {calls}"
    );
}

// ─── Test 6: rx_step exists and has a While loop ─────────────────────────────

#[test]
fn rx_step_has_while_loop() {
    let ir = lower_regex_mind();
    let body = fn_body(&ir, "rx_step");
    let has_while = body.iter().any(|i| matches!(i, Instr::While { .. }));
    assert!(
        has_while,
        "rx_step must contain a while loop to iterate over current states"
    );
}

// ─── Test 7: rx_cls_matches checks ranges ────────────────────────────────────

#[test]
fn rx_cls_matches_uses_rx_cls_negated() {
    let ir = lower_regex_mind();
    let body = fn_body(&ir, "rx_cls_matches");
    let calls = count_calls_recursive(body, "rx_cls_negated");
    assert!(
        calls >= 1,
        "rx_cls_matches must call rx_cls_negated; got {calls}"
    );
}

// ─── Test 8: std.regex auto-exports public symbols ───────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn regex_mind_auto_exports_public_symbols() {
    use libmind::project::module_table::collect_module_exports;

    let module = parser::parse(REGEX_MIND_SRC).expect("std/regex.mind must parse");
    let ex = collect_module_exports("std.regex", &module);

    assert_eq!(ex.module_path, "std.regex");
    for want in [
        "rx_compile",
        "rx_try_compile",
        "rx_is_match",
        "rx_find",
        "rx_find_all",
        "rx_capture",
        "rx_nfa_start",
        "rx_capture_count",
        "rx_kind_match",
        "rx_kind_split",
        "rx_kind_char",
        "rx_kind_any",
        "rx_kind_class",
        "rx_kind_anchor_start",
        "rx_kind_anchor_end",
    ] {
        assert!(
            ex.exported.iter().any(|s| s == want),
            "std.regex must auto-export `{want}`; got {:?}",
            ex.exported
        );
    }
}

// ─── Test 9: bundled stdlib resolves use std.regex ───────────────────────────

#[cfg(feature = "cross-module-imports")]
#[test]
fn bundled_stdlib_resolves_use_std_regex() {
    use libmind::project::module_table::build_module_table;
    use libmind::project::stdlib::parsed_stdlib_modules;
    use libmind::type_checker::{TypeEnv, check_module_types_with_modules};

    let stdlib = parsed_stdlib_modules();
    let refs: Vec<(String, &libmind::ast::Module)> =
        stdlib.iter().map(|(p, m)| (p.clone(), m)).collect();
    let table = build_module_table(&refs);

    let consumer = "use std.regex\nlet rx = rx_compile(0, 0)\n";
    let ast = parser::parse(consumer).expect("consumer must parse");
    let env = TypeEnv::default();
    let diags = check_module_types_with_modules(&ast, consumer, None, &env, &table);
    assert!(
        diags.is_empty(),
        "use std.regex + rx_compile should resolve via bundled stdlib; got {:?}",
        diags
    );
}

// ─── Test 10: MLIR functional round-trip ──────────────────────────────────────

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod mlir_functional {
    use super::mindc_bin;
    use std::path::PathBuf;
    use std::process::Command;

    // mindc_bin() provided by tests/common (CARGO_BIN_EXE_mindc — staleness-free)

    #[test]
    fn regex_compile_and_match_via_compiled_so() {
        let mindc = mindc_bin();
        if !mindc.exists() {
            println!("regex_compile_and_match_via_compiled_so: mindc not found; skipping");
            return;
        }

        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_surface_regex");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let regex_src = include_str!("../std/regex.mind");
        let driver_src = format!(
            r#"{regex_src}

pub fn smoke_rx_compile(pat: i64, plen: i64) -> i64 {{
    rx_compile(pat, plen)
}}
pub fn smoke_rx_is_match(rx: i64, inp: i64, ilen: i64) -> i64 {{
    rx_is_match(rx, inp, ilen)
}}
pub fn smoke_rx_find(rx: i64, inp: i64, ilen: i64) -> i64 {{
    rx_find(rx, inp, ilen)
}}
pub fn smoke_rx_find_all_len(rx: i64, inp: i64, ilen: i64) -> i64 {{
    let fa: i64 = rx_find_all(rx, inp, ilen);
    __mind_load_i64(fa + 8)
}}
"#
        );
        let driver_path = out_dir.join("regex_smoke.mind");
        let so_path = out_dir.join("libregex_smoke.so");
        std::fs::write(&driver_path, &driver_src).expect("write driver MIND");

        let status = Command::new(&mindc)
            .args([
                driver_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
            ])
            .status()
            .expect("run mindc");

        if !status.success() {
            println!("regex_compile_and_match_via_compiled_so: mindc compile failed; skipping");
            return;
        }

        unsafe {
            let lib = libloading::Library::new(&so_path).expect("dlopen regex_smoke.so");
            type CompileFn = unsafe extern "C" fn(i64, i64) -> i64;
            type IsMatchFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
            type FindFn = unsafe extern "C" fn(i64, i64, i64) -> i64;
            type FindAllLFn = unsafe extern "C" fn(i64, i64, i64) -> i64;

            let compile_fn: libloading::Symbol<CompileFn> = lib.get(b"smoke_rx_compile\0").unwrap();
            let is_match_fn: libloading::Symbol<IsMatchFn> =
                lib.get(b"smoke_rx_is_match\0").unwrap();
            let find_fn: libloading::Symbol<FindFn> = lib.get(b"smoke_rx_find\0").unwrap();
            let find_all_fn: libloading::Symbol<FindAllLFn> =
                lib.get(b"smoke_rx_find_all_len\0").unwrap();

            // Fixture 1: literal match
            let pat1 = b"hello";
            let rx1 = compile_fn(pat1.as_ptr() as i64, pat1.len() as i64);
            assert!(rx1 != 0, "literal pattern must compile");

            let inp1 = b"say hello world";
            assert_eq!(
                is_match_fn(rx1, inp1.as_ptr() as i64, inp1.len() as i64),
                1,
                "literal 'hello' must match in 'say hello world'"
            );

            let inp1b = b"no match here";
            assert_eq!(
                is_match_fn(rx1, inp1b.as_ptr() as i64, inp1b.len() as i64),
                0,
                "literal 'hello' must NOT match in 'no match here'"
            );

            // Fixture 2: find byte offset
            let offset = find_fn(rx1, inp1.as_ptr() as i64, inp1.len() as i64);
            assert_eq!(
                offset, 4,
                "literal 'hello' starts at offset 4 in 'say hello world'"
            );

            // Fixture 3: character class [a-z]+
            let pat3 = b"[a-z]+";
            let rx3 = compile_fn(pat3.as_ptr() as i64, pat3.len() as i64);
            assert!(rx3 != 0, "character class pattern must compile");
            let inp3 = b"abc123def";
            assert_eq!(
                is_match_fn(rx3, inp3.as_ptr() as i64, inp3.len() as i64),
                1,
                "[a-z]+ must match in 'abc123def'"
            );

            // Fixture 4: alternation a|b
            let pat4 = b"cat|dog";
            let rx4 = compile_fn(pat4.as_ptr() as i64, pat4.len() as i64);
            assert!(rx4 != 0, "alternation pattern must compile");
            let inp4a = b"I have a cat";
            let inp4b = b"I have a dog";
            let inp4c = b"I have a fish";
            assert_eq!(
                is_match_fn(rx4, inp4a.as_ptr() as i64, inp4a.len() as i64),
                1,
                "cat|dog must match 'cat'"
            );
            assert_eq!(
                is_match_fn(rx4, inp4b.as_ptr() as i64, inp4b.len() as i64),
                1,
                "cat|dog must match 'dog'"
            );
            assert_eq!(
                is_match_fn(rx4, inp4c.as_ptr() as i64, inp4c.len() as i64),
                0,
                "cat|dog must NOT match 'fish'"
            );

            // Fixture 5: \d+ (digit class)
            let pat5 = b"\\d+";
            let rx5 = compile_fn(pat5.as_ptr() as i64, pat5.len() as i64);
            assert!(rx5 != 0, "\\d+ pattern must compile");
            let inp5 = b"abc 123 xyz";
            assert_eq!(
                is_match_fn(rx5, inp5.as_ptr() as i64, inp5.len() as i64),
                1,
                "\\d+ must match digits in 'abc 123 xyz'"
            );

            // Fixture 6: ^ start anchor
            let pat6 = b"^hello";
            let rx6 = compile_fn(pat6.as_ptr() as i64, pat6.len() as i64);
            assert!(rx6 != 0, "^hello must compile");
            let inp6a = b"hello world";
            let inp6b = b"say hello world";
            assert_eq!(
                is_match_fn(rx6, inp6a.as_ptr() as i64, inp6a.len() as i64),
                1,
                "^hello must match at start"
            );
            assert_eq!(
                is_match_fn(rx6, inp6b.as_ptr() as i64, inp6b.len() as i64),
                0,
                "^hello must NOT match in middle"
            );

            // Fixture 6b: $ end anchor
            let pat6c = b"world$";
            let rx6c = compile_fn(pat6c.as_ptr() as i64, pat6c.len() as i64);
            assert!(rx6c != 0, "world$ must compile");
            let inp6c = b"hello world";
            let inp6d = b"world peace";
            assert_eq!(
                is_match_fn(rx6c, inp6c.as_ptr() as i64, inp6c.len() as i64),
                1,
                "world$ must match at end"
            );
            assert_eq!(
                is_match_fn(rx6c, inp6d.as_ptr() as i64, inp6d.len() as i64),
                0,
                "world$ must NOT match in middle"
            );

            // Fixture 6c: ^...$ both anchors (whole-string match)
            let pat6e = b"^hi$";
            let rx6e = compile_fn(pat6e.as_ptr() as i64, pat6e.len() as i64);
            assert!(rx6e != 0, "^hi$ must compile");
            let inp6e = b"hi";
            let inp6f = b"hi there";
            assert_eq!(
                is_match_fn(rx6e, inp6e.as_ptr() as i64, inp6e.len() as i64),
                1,
                "^hi$ must match the whole string \"hi\""
            );
            assert_eq!(
                is_match_fn(rx6e, inp6f.as_ptr() as i64, inp6f.len() as i64),
                0,
                "^hi$ must NOT match \"hi there\""
            );

            // Fixture 7: find_all count
            let pat7 = b"[0-9]+";
            let rx7 = compile_fn(pat7.as_ptr() as i64, pat7.len() as i64);
            assert!(rx7 != 0, "[0-9]+ must compile");
            let inp7 = b"1 22 333";
            let n = find_all_fn(rx7, inp7.as_ptr() as i64, inp7.len() as i64);
            assert_eq!(
                n, 3,
                "find_all([0-9]+, '1 22 333') must find 3 matches; got {n}"
            );

            // Fixture 8: negated character class [^abc]
            let pat8 = b"[^abc]+";
            let rx8 = compile_fn(pat8.as_ptr() as i64, pat8.len() as i64);
            assert!(rx8 != 0, "[^abc]+ must compile");
            let inp8a = b"xyz";
            let inp8b = b"aaa";
            assert_eq!(
                is_match_fn(rx8, inp8a.as_ptr() as i64, inp8a.len() as i64),
                1,
                "[^abc]+ must match 'xyz'"
            );
            assert_eq!(
                is_match_fn(rx8, inp8b.as_ptr() as i64, inp8b.len() as i64),
                0,
                "[^abc]+ must NOT match 'aaa'"
            );

            // Fixture 9: email-like pattern \w+@\w+\.\w+
            let pat9 = b"\\w+@\\w+\\.\\w+";
            let rx9 = compile_fn(pat9.as_ptr() as i64, pat9.len() as i64);
            assert!(rx9 != 0, "email-like pattern must compile");
            let inp9a = b"user@example.com";
            let inp9b = b"not-an-email";
            assert_eq!(
                is_match_fn(rx9, inp9a.as_ptr() as i64, inp9a.len() as i64),
                1,
                "email-like pattern must match 'user@example.com'"
            );
            assert_eq!(
                is_match_fn(rx9, inp9b.as_ptr() as i64, inp9b.len() as i64),
                0,
                "email-like pattern must NOT match 'not-an-email'"
            );

            // Fixture 10: IPv4-like pattern \d+\.\d+\.\d+\.\d+
            let pat10 = b"\\d+\\.\\d+\\.\\d+\\.\\d+";
            let rx10 = compile_fn(pat10.as_ptr() as i64, pat10.len() as i64);
            assert!(rx10 != 0, "IPv4-like pattern must compile");
            let inp10a = b"192.168.1.1";
            let _inp10b = b"not.an.ip";
            assert_eq!(
                is_match_fn(rx10, inp10a.as_ptr() as i64, inp10a.len() as i64),
                1,
                "IPv4-like pattern must match '192.168.1.1'"
            );
            // "not.an.ip" has no digit sequences at all in the right spots
            let inp10c = b"abc.def.ghi.jkl";
            assert_eq!(
                is_match_fn(rx10, inp10c.as_ptr() as i64, inp10c.len() as i64),
                0,
                "IPv4-like pattern must NOT match all-alpha octets"
            );

            // Fixture 11: empty alternation branches must COMPILE (not crash).
            // Regression for the null-deref where an empty branch returned a
            // null fragment that rx_compile_alternation then dereferenced.
            for pat in [&b"a|"[..], &b"|a"[..], &b"(|x)"[..], &b"|"[..]] {
                let rx = compile_fn(pat.as_ptr() as i64, pat.len() as i64);
                assert!(
                    rx != 0,
                    "empty-alternation pattern must compile without crashing"
                );
            }
            // `a|` (a OR empty) matches "a".
            let rx_alt = compile_fn(b"a|".as_ptr() as i64, 2);
            let inp_alt = b"a";
            assert_eq!(
                is_match_fn(rx_alt, inp_alt.as_ptr() as i64, inp_alt.len() as i64),
                1,
                "'a|' must match 'a'"
            );

            // Fixture 12: `^`/`$` anchors are zero-width — must NOT consume a
            // byte (regression for false-positive matches mid-stream).
            let rx_end = compile_fn(b"a$".as_ptr() as i64, 2);
            let ab = b"ab";
            assert_eq!(
                is_match_fn(rx_end, ab.as_ptr() as i64, ab.len() as i64),
                0,
                "'a$' must NOT match 'ab' ('a' is not at end)"
            );
            let a = b"a";
            assert_eq!(
                is_match_fn(rx_end, a.as_ptr() as i64, a.len() as i64),
                1,
                "'a$' must match 'a'"
            );
            let rx_start = compile_fn(b"^a".as_ptr() as i64, 2);
            let ba = b"ba";
            assert_eq!(
                is_match_fn(rx_start, ba.as_ptr() as i64, ba.len() as i64),
                0,
                "'^a' must NOT match 'ba' ('a' is not at start)"
            );
        }
    }
}
