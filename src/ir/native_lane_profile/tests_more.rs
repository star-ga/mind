// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
use super::*;

/// The refusal classes the downstream consumer lane-admission probe reported, as generic
/// shapes. The scalar ones are admitted together; `print` and a C extern block,
/// which no native-lane compiler implements, refuse at a located source site.
#[test]
fn probe_refusals_are_admitted_or_located() {
    // lane.source.top_level (plain import), lane.ir.discard_result_use (a store
    // result read as a value) and lane.type.mismatch (a store-returning helper,
    // a diverging branch, full-width casts) in one program.
    let scalar = "import helper;\n\
        fn w_set(w: i64, i: i64, v: i64) -> i64 { __mind_store_i64(w + i * 8, v) }\n\
        fn decode(first: i64) -> i64 { let mut n = 0; if first < 128 { n = 1; } \
        else { if first >= 194 { n = 2; } else { return 0 - 1; } } return n; }\n\
        fn ratio(a: i64, b: i64) -> i64 { return ((a as f64) / (b as f64) * 100.0) as i64; }\n\
        fn main() -> i64 { let w = __mind_alloc(64); let bad = w_set(w, 1, 5) + w_set(w, 2, 6); \
        return bad + decode(65) + ratio(1, 4); }\n";
    assert_eq!(admits(scalar), Ok(()), "{scalar}");

    // lane.call.unknown on the evaluator: `print` lowers to runtime calls no
    // native-lane compiler implements. It now refuses at the call, located.
    let evaluator = "fn main(argc: i64, argv: i64) -> i64 {\n    let count: i64 = argc + argv;\n    \
        if count < 1 { return 2; }\n    print(count); 0\n}\n";
    for err in [
        source_admits(evaluator).unwrap_err(),
        admits(evaluator).unwrap_err(),
    ] {
        assert_eq!(err.construct, "lane.source.print");
        assert_eq!(err.function.as_deref(), Some("main"));
        assert_eq!(err.at, Some(evaluator.find("print").unwrap()));
    }
    for source in [
        "fn main() -> i64 { print(1); 0 }",
        "fn main() -> i64 { print(\"label\"); 0 }",
        "fn main() -> i64 { print(1, 2); 0 }",
    ] {
        assert_eq!(
            source_admits(source).unwrap_err().construct,
            "lane.source.print"
        );
    }

    // The C-extern programs: a typable declaration is admitted under the mark
    // and refused by name without it.
    let externs =
        "fn main() -> i64 { return 0; }\nextern \"C\" {\n    fn close(fd: i32) -> i32;\n}\n";
    assert_eq!(source_admits(externs), Ok(()));
    let unmarked = source_admits_with(externs, false).unwrap_err();
    assert_eq!(unmarked.construct, "lane.extern.compiler_unmarked");
    // Spec "Extern bindings": a `bind` clause is refused by name first, with or
    // without the mark, even on a typable declaration; the same block with no
    // clause admits.
    for clause in ["bind(\"libz.so.1\") ", "bind(.address) "] {
        let bound = externs.replace("extern \"C\" ", &format!("extern \"C\" {clause}"));
        for marked in [true, false] {
            let err = source_admits_with(&bound, marked).unwrap_err();
            assert_eq!(err.construct, "lane.extern.bind_clause", "{bound}");
        }
    }
}

#[test]
fn silent_mismatch_admission_refusals() {
    for (source, kind) in [
        (
            "fn main() -> i64 { return __mind_alloc(0); }",
            "lane.alloc.size_positive",
        ),
        (
            "fn main() -> i64 { return __mind_alloc(0 - 16); }",
            "lane.alloc.size_positive",
        ),
        (
            "fn f(n: i64) -> i64 { return __mind_alloc(n); } fn main() -> i64 { return f(0); }",
            "lane.alloc.size_positive",
        ),
        (
            "fn main() -> i64 { return __mind_realloc(0, 0); }",
            "lane.alloc.size_positive",
        ),
        (
            "fn f(b: bool) -> i64 { return !b; } fn main() -> i64 { return f(2); }",
            "lane.bool.width",
        ),
        (
            "fn f(b: bool) -> i64 { return b == true; } fn main() -> i64 { return f(3); }",
            "lane.bool.width",
        ),
        (
            "fn f(b: bool) -> i64 { return b + 10; } fn main() -> i64 { return f(true); }",
            "lane.bool.width",
        ),
        (
            "fn f(b: bool) -> i64 { let c = b; return !c; } fn main() -> i64 { return f(2); }",
            "lane.bool.width",
        ),
        (
            "fn f(b: bool) -> i64 { return -b; } fn main() -> i64 { return f(true); }",
            "lane.bool.width",
        ),
        ("fn main() -> f64 { return 1.0; }", "lane.entry.f64_result"),
    ] {
        assert_eq!(admits(source).unwrap_err().construct, kind, "{source}");
    }
}

// NATTLS B2: `#[protection]` (no arguments) is codegen-inert; any other
// attribute, or protection with arguments, keeps the contract refusal.
#[test]
fn protection_attr_is_admitted_other_attrs_refuse() {
    let ok = "#[protection]\nfn f() -> i64 { return 1; }\nfn main() -> i64 { return f(); }";
    assert_eq!(admits(ok), Ok(()));
    for bad in [
        "#[inline]\nfn f() -> i64 { return 1; }\nfn main() -> i64 { return f(); }",
        "#[protection]\n#[inline]\nfn f() -> i64 { return 1; }\nfn main() -> i64 { return f(); }",
    ] {
        let err = admits(bad).unwrap_err();
        assert_eq!(err.construct, "lane.source.function_contract", "{bad}");
    }
}

// NATTLS B3: the freeaddrinfo and uname declarations std.dns needs are typed.
#[test]
fn freeaddrinfo_and_uname_declarations_admit() {
    let src = "extern \"C\" {\n    fn freeaddrinfo(res: *mut u8);\n    fn uname(buf: *mut u8) -> i32;\n}\nfn main() -> i64 { return 0; }";
    assert_eq!(admits(src), Ok(()));
}

// NATTLS B5: a repeated identical extern declaration is one symbol. A repeat
// with a different signature is the compiler's decision, not the host's: the
// host keeps the first signature (tests/native_lane_extern_parity.rs,
// `the_compiler_refuses_the_externs_the_host_no_longer_names`).
#[test]
fn identical_duplicate_extern_is_one_symbol() {
    let decl = "extern \"C\" { fn memset(ptr: *mut u8, value: i32, num: i64) -> *mut u8; }\n";
    let same = format!("{decl}{decl}fn main() -> i64 {{ return 0; }}");
    assert_eq!(admits(&same), Ok(()));
    let other = format!(
        "{decl}extern \"C\" {{ fn memset(ptr: *mut u8, value: i64, num: i64) -> *mut u8; }}\nfn main() -> i64 {{ return 0; }}"
    );
    assert_eq!(admits(&other), Ok(()));
    let user =
        format!("{decl}fn memset(a: i64) -> i64 {{ return a; }}\nfn main() -> i64 {{ return 0; }}");
    assert_eq!(admits(&user).unwrap_err().construct, "lane.call.ambiguous");
}

// NATTLS B7/B8: __mind_load_i16, __mind_argc, __mind_argv have lane rows.
#[test]
fn load_i16_argc_argv_rows_admit() {
    let src = "fn main() -> i64 { let p = __mind_alloc(8); __mind_store_i64(p, 0); let x = __mind_load_i16(p); __mind_free(p); return x + __mind_argc() + __mind_argv(0); }";
    assert_eq!(
        profile_lane_admits_source(&crate::parser::parse(src).unwrap(), true),
        Ok(())
    );
}

// NATTLS B3/B4/B6/B7/B11 ratchet on the shipped std.dns and std.entropy
// sources read as a user image: the extern block (freeaddrinfo, uname, errno
// as *mut i32), the `0 as i32` fills and flags, and the memset result
// discarded as a statement call are admitted. std.entropy passes whole; the
// first std.dns refusal is now the user-image allocation proof in
// dns_new_record, which a seed body does not take (the seed is typed, not
// re-proved).
#[test]
fn std_dns_and_entropy_sources_pass_the_pointer_discard() {
    let verdict = |module: &str| {
        let path = format!("{}/std/{module}.mind", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap();
        let ast = crate::parser::parse(&text).expect("std module parses");
        match profile_lane_admits_source(&ast, true) {
            Ok(()) => ("ok", String::new()),
            Err(e) => (e.construct, e.function.unwrap_or_default()),
        }
    };
    assert_eq!(
        [verdict("dns"), verdict("entropy")],
        [
            ("lane.alloc.size_positive", "dns_new_record".to_string()),
            ("ok", String::new()),
        ]
    );
}

// B11 mutant control: a `let _` that binds a pointer stays refused, so the
// ratchet above is not a vacuous pass. The pointer result of an extern call
// passes through (the compiler judges it); a pointer cast is the host's.
#[test]
fn a_pointer_discard_binding_stays_refused() {
    let src = "extern \"C\" { unsafe fn memset(ptr: *mut u8, value: i32, num: i64) -> *mut u8; }\n\
               fn z(p: i64, n: i64) { let _ = memset(p as *mut u8, 0 as i32, n); }\n\
               fn main() -> i64 { let p: i64 = __mind_alloc(8); z(p, 8); return 0; }";
    assert_eq!(
        profile_lane_admits_source(&crate::parser::parse(src).unwrap(), true),
        Ok(())
    );
    let cast = src.replace("memset(p as *mut u8, 0 as i32, n)", "p as *mut u8");
    let err = profile_lane_admits_source(&crate::parser::parse(&cast).unwrap(), true).unwrap_err();
    assert_eq!(err.construct, "lane.source.binding_type");
}

// A `let _` discard of an i32 value binds no name (std.net `tcp_close`). A
// named i32 binding still refuses, and a later read of `_` finds nothing. An
// extern result passes through either binding: the compiler judges it.
#[test]
fn an_i32_discard_binds_no_name() {
    let admit = |body: &str| {
        let text = format!(
            "extern \"C\" {{ fn getuid() -> i32; }}\nfn main() -> i64 {{\n{body}\n    return 0;\n}}\n"
        );
        profile_lane_admits_source(&crate::parser::parse(&text).unwrap(), true)
    };
    assert_eq!(admit("    let _ = getuid();"), Ok(()));
    assert_eq!(admit("    let _ = 5 as i32;"), Ok(()));
    assert_eq!(admit("    let u = getuid();"), Ok(()));
    let named = admit("    let u = 5 as i32;").unwrap_err();
    assert_eq!(named.construct, "lane.source.binding_type");
    let read = admit("    let _ = 5;\n    let _ = 5 as i32;\n    let v: i64 = _;").unwrap_err();
    assert_eq!(read.construct, "lane.source.unbound");
}

/// `print` is admitted only for a compiler with the `nbprint1` mark
/// (`LaneSeed::with_print`): string literals, i64 and bool arguments, an
/// effect-free argument after the first, no user or extern helper names.
#[test]
fn print_needs_the_print_mark_and_its_argument_contract() {
    let seeded = |source: &str, print: bool| -> Check<LaneEdges> {
        let (ast, ir) = lower(source);
        let seed = LaneSeed::new(&[], &[]).with_print(print);
        profile_lane_admits_seeded(&ast, &ir, &seed, true)
    };
    for source in [
        "fn main() -> i64 { print(\"hello\"); 0 }",
        "fn main() -> i64 { print(0); print(0 - 9223372036854775807 - 1); 0 }",
        "fn main() -> i64 { let b: bool = 3 > 2; print(b); print(true); 0 }",
        "fn main() -> i64 { let x = 5; print(\"x = \", x, -7, \"\"); 0 }",
        "fn f() -> i64 { return 7; } fn main() -> i64 { print(f(), \"!\"); 0 }",
        "fn main() -> i64 { let mut i = 0; while i < 3 { print(i); i = i + 1; } 0 }",
    ] {
        assert_eq!(
            seeded(source, false).unwrap_err().construct,
            "lane.source.print",
            "unmarked: {source}"
        );
        let edges = seeded(source, true).unwrap_or_else(|e| panic!("{source}: {e:?}"));
        assert!(
            edges.print.keys().all(|n| n.starts_with("print")),
            "{source}"
        );
    }
    for (source, construct) in [
        (
            "fn main() -> i64 { print(1.5); 0 }",
            "lane.source.print_argument",
        ),
        (
            "fn f() -> i64 { return 7; } fn main() -> i64 { print(\"a\", f()); 0 }",
            "lane.source.print_order",
        ),
        (
            "fn main() -> i64 { let x = 1; print(x, x + 1); 0 }",
            "lane.source.print_order",
        ),
        (
            "fn printI64(x: i64) -> i64 { return x; } fn main() -> i64 { print(1); 0 }",
            "lane.source.print_collision",
        ),
        ("fn main() -> i64 { printI64(1); 0 }", "lane.call.unknown"),
    ] {
        assert_eq!(
            seeded(source, true).unwrap_err().construct,
            construct,
            "{source}"
        );
    }
    // A user `print_bytes` without a print statement keeps its own body.
    let own = "fn print_bytes(a: i64, n: i64) -> i64 { return a + n; } \
        fn main() -> i64 { return print_bytes(1, 2); }";
    assert!(!seeded(own, true).unwrap().print.contains_key("print_bytes"));
}

/// This check types `String` (by its exact name), a string literal and a
/// field read as the existing i64 word, and decides nothing more about them: the
/// compiler's own string rule does. The IR leg takes the same programs unchanged.
#[test]
fn string_literal_and_field_read_are_the_i64_word() {
    let calls = |ir: &IRModule, func: &str| -> Vec<String> {
        ir.instrs
            .iter()
            .find_map(|i| match i {
                Instr::FnDef { name, body, .. } if name == func => Some(
                    body.iter()
                        .filter_map(|b| match b {
                            Instr::Call { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .expect("the function is lowered")
    };
    let source =
        "fn f(s: String) -> i64 { return s.len; }\nfn main() -> i64 { return f(\"hi\"); }\n";
    let (ast, ir) = lower(source);
    assert_eq!(profile_lane_admits(&ast, &ir, true), Ok(()), "{source}");
    // Positive controls: the literal and the read are in the IR the walk typed.
    assert!(calls(&ir, "main").contains(&"__mind_alloc".to_string()));
    assert!(calls(&ir, "f").contains(&"__mind_load_i64".to_string()));
}

/// What this check now passes and no longer decides. The compiler's own string
/// rule refuses each of these programs; here the only claim is that the Rust
/// check stays out of it.
#[test]
fn the_rust_check_no_longer_decides_what_the_string_rule_decides() {
    let seed = LaneSeed::new(&[], &[]).with_print(true);
    for source in [
        // an integer in a String slot
        "fn g(s: String) -> i64 { return 1; } fn main() -> i64 { return g(5); }",
        // a field read on an i64, and on a name that re-binds a String parameter
        "fn f(x: i64) -> i64 { return x.len; }",
        "fn f(s: String) -> i64 { let s = 5; return s.len; }",
        // a String const, a String result, a String let, another field
        "const S: String = \"hi\"; fn main() -> i64 { return 0; }",
        "fn mk() -> String { return \"a\"; }",
        "fn main() -> i64 { let s: String = \"hi\"; return 0; }",
        "fn f(s: String) -> i64 { return s.cap; }",
        // print of a String value
        "fn f(s: String) -> i64 { print(s); return 0; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert!(
            profile_lane_admits_source_seeded(&ast, &seed, true).is_ok(),
            "{source}"
        );
    }
}

/// Only the exact name `String` is the i64 word. `Named(_)` must keep refusing
/// every other name, and u8, i16 and usize depend on it.
#[test]
fn only_the_exact_name_string_is_typed_as_the_word() {
    let kind = |source: &str| {
        profile_lane_admits_source(&crate::parser::parse(source).expect(source), true)
            .expect_err(source)
            .construct
    };
    for (ty, expected) in [
        ("u8", "lane.type.int_width"),
        ("i16", "lane.type.int_width"),
        ("usize", "lane.type.int_width"),
        ("u64", "lane.type.int_width"),
        ("isize", "lane.type.int_width"),
        ("string", "lane.type.named"),
        ("Strings", "lane.type.named"),
        ("StringBuf", "lane.type.named"),
        ("Foo", "lane.type.named"),
        ("Vec<i64>", "lane.type.generic"),
        ("Vec<String>", "lane.type.generic"),
    ] {
        for source in [
            format!("fn f(x: {ty}) -> i64 {{ return 0; }}"),
            format!("fn f() -> {ty} {{ return 0; }}"),
            format!("fn main() -> i64 {{ let x: {ty} = 0; return 0; }}"),
        ] {
            assert_eq!(kind(&source), expected, "{source}");
        }
    }
}

/// A field read walks its receiver, so a refusal inside the receiver still
/// names the program. A field WRITE stays refused.
#[test]
fn a_field_read_walks_its_receiver_and_a_field_write_stays_refused() {
    let kind = |source: &str| {
        profile_lane_admits_source(&crate::parser::parse(source).expect(source), true)
            .expect_err(source)
            .construct
    };
    for (source, expected) in [
        ("fn f() -> i64 { return g().len; }", "lane.call.unknown"),
        ("fn f() -> i64 { return y.len; }", "lane.source.unbound"),
        (
            "fn f(s: String) -> i64 { s.len = 3; return 0; }",
            "lane.source.struct",
        ),
        (
            "fn f(s: String) -> i64 { s.len.cap = 3; return 0; }",
            "lane.source.struct",
        ),
    ] {
        assert_eq!(kind(source), expected, "{source}");
    }
    // Positive control: the same shapes with a walkable receiver are admitted.
    for source in [
        "fn f(s: String) -> i64 { return s.len; }",
        "fn f(s: String) -> i64 { return s.len.addr; }",
    ] {
        let ast = crate::parser::parse(source).expect(source);
        assert_eq!(profile_lane_admits_source(&ast, true), Ok(()), "{source}");
    }
}
/// The host types an `extern "C"` declaration by its ABI classes and decides
/// nothing about the symbol: which names exist, and with which signature, is
/// the compiler's decision. Any ABI-typable declaration is admitted here,
/// whatever its name, its arity or its widths.
#[test]
fn an_abi_typable_extern_is_admitted_whatever_its_name_and_widths() {
    let eleven = (0..11)
        .map(|i| format!("p{i}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    for decl in [
        "fn open(path: *const u8, flags: i32) -> i32;".to_string(),
        "fn nosuchsymbol(x: i32) -> i32;".to_string(),
        "fn close() -> i32;".to_string(),
        "fn close(fd: i64) -> i64;".to_string(),
        "fn close(fd: i32);".to_string(),
        "fn _exit(status: i32) -> i32;".to_string(),
        "fn uname(buf: *mut u8) -> i64;".to_string(),
        "fn getuid() -> i64;".to_string(),
        format!("fn wide({eleven}) -> i32;"),
    ] {
        let source = format!("extern \"C\" {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        assert_eq!(source_admits(&source), Ok(()), "{decl}");
    }
}

/// The host judges no extern signature: a calling convention, a varargs
/// marker, a type outside the ABI classes, any result class and any arity pass
/// through it, and the compiler's rows decide (lane.extern.unknown_symbol,
/// lane.extern.arity, lane.extern.signature_mismatch, lane.extern.callconv).
#[test]
fn the_host_passes_every_extern_signature_to_the_compiler() {
    let twelve = (0..12)
        .map(|i| format!("p{i}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    let too_wide = format!("fn close({twelve}) -> i32;");
    for (head, decl) in [
        ("", "fn close(fd: i32, ...) -> i32;"),
        (" callconv(.win64)", "fn close(fd: i32) -> i32;"),
        ("", "fn close(fd: f64) -> i32;"),
        ("", "fn close(fd: i32) -> u32;"),
        ("", "fn getenv(name: *const u8) -> *const u8;"),
        ("", too_wide.as_str()),
    ] {
        let source = format!("extern \"C\"{head} {{ {decl} }}\nfn main() -> i64 {{ return 0; }}\n");
        assert_eq!(source_admits(&source), Ok(()), "{source}");
    }
}

/// A call of an extern is admitted whatever its signature, so the compiler's
/// rows name the refusal. The host judges no extern call and no use of its result.
#[test]
fn a_called_unjudged_extern_is_left_to_the_compiler() {
    for source in [
        "extern \"C\" { fn cbrt(x: f64) -> f64; }\nfn main() -> i64 { let y: f64 = cbrt(8.0); return 7; }\n",
        "extern \"C\" { fn close(fd: f64) -> i32; }\nfn main() -> i64 { close(1.5); return 7; }\n",
        "extern \"C\" { fn getuid() -> f64; }\nfn main() -> i64 { let y: f64 = getuid(); return 7; }\n",
        // An f64 result used as i64 is still the compiler's row, not a host mismatch.
        "extern \"C\" { fn getuid() -> f64; }\nfn main() -> i64 { return getuid(); }\n",
        "extern \"C\" { fn getuid() -> f64; }\nfn main() -> i64 { let y: i64 = getuid(); return y; }\n",
        "extern \"C\" { fn umask(mode: f32) -> i32; }\nfn main() -> i64 { umask(1.5); return 7; }\n",
        "extern \"C\" { fn close(fd: bool) -> i32; }\nfn main() -> i64 { close(1); return 7; }\n",
        "extern \"C\" { fn close(fd: i64) -> i32; }\nextern \"C\" { fn close(fd: i32) -> i32; }\n\
         fn main() -> i64 { close((0 - 1) as i32); return 7; }\n",
        "extern \"C\" { fn close(fd: [u8; 16]) -> i32; }\nfn main() -> i64 { return 7; }\n",
        // The host judged this call before; the compiler names key 66 at it now.
        "extern \"C\" { fn close(fd: i32) -> i32; }\nfn main() -> i64 { close(1.5); return 7; }\n",
        // An integer result passes through each use: a binding, an operand, a store, a negation.
        "extern \"C\" { fn close(fd: i32) -> i32; }\n\
         fn main() -> i64 { let n = close(1); if n < 0 { return 1; } return 0 - n; }\n",
        "extern \"C\" { fn close(fd: i32) -> i32; }\n\
         fn main() -> i64 { let mut n = close(1); n = n + 1; return -n; }\n",
        "extern \"C\" { fn close(fd: i32) -> i32; }\nfn main() -> i64 { return close(1) + close(2); }\n",
    ] {
        assert_eq!(source_admits(source), Ok(()), "{source}");
    }
    // The operand beside an extern result is still judged.
    let unit = "extern \"C\" { fn close(fd: i32) -> i32; }\nfn h() { }\n\
                fn main() -> i64 { let y = close(1) + h(); return y; }\n";
    assert_eq!(
        source_admits(unit).unwrap_err().construct,
        "lane.binop.non_scalar"
    );
    // Source and the lowered IR both pass: the use of an f64 extern result is not a host key.
    for source in [
        "extern \"C\" { fn getuid() -> f64; }\nfn main() -> i64 { return getuid(); }\n",
        "extern \"C\" { fn getuid() -> f64; }\nfn main() -> i64 { let y: i64 = getuid(); return y; }\n",
        "extern \"C\" { fn getuid() -> bool; }\nfn main() -> i64 { let y: i64 = getuid(); return y; }\n",
        "extern \"C\" { fn close(fd: i32) -> i32; }\n\
         fn main() -> i64 { let mut n = close(1); n = n + 1; if 0 > n { return 1; } return n; }\n",
    ] {
        assert_eq!(admits(source), Ok(()), "{source}");
    }
}

/// The host holds no list of names, so an extern named like a compiler
/// intrinsic is admitted as a bare declaration. A call of it stays refused,
/// because the name is then defined in two tiers.
#[test]
fn an_extern_named_like_an_intrinsic_is_refused_only_where_it_is_called() {
    let decl = "extern \"C\" { fn __mind_alloc(x: i64) -> i64; }\n";
    let bare = format!("{decl}fn main() -> i64 {{ return 0; }}\n");
    assert_eq!(source_admits(&bare), Ok(()));
    let called = format!("{decl}fn main() -> i64 {{ return __mind_alloc(8); }}\n");
    assert_eq!(
        source_admits(&called).unwrap_err().construct,
        "lane.call.ambiguous"
    );
}
