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

    // The C-extern programs: the table row is admitted under the mark and
    // refused by name without it (and every off-table name still refuses).
    let externs =
        "fn main() -> i64 { return 0; }\nextern \"C\" {\n    fn close(fd: i32) -> i32;\n}\n";
    assert_eq!(source_admits(externs), Ok(()));
    let unmarked = source_admits_with(externs, false).unwrap_err();
    assert_eq!(unmarked.construct, "lane.extern.compiler_unmarked");
    let off_table =
        "fn main() -> i64 { return 0; }\nextern \"C\" {\n    fn open(x: i32) -> i32;\n}\n";
    let err = source_admits(off_table).unwrap_err();
    assert_eq!(err.construct, "lane.extern.unknown_symbol");
    assert_eq!(err.function.as_deref(), Some("open"));
    // Spec "Extern bindings": a `bind` clause is refused by name first, with or
    // without the mark, even on a table row; the same block with no clause admits.
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
            "fn main() -> i64 { while 1 { __mind_free(0); } return 0; }",
            "lane.alloc.free_in_loop",
        ),
        (
            "fn f() -> i64 { return __mind_free(0); } fn g() -> i64 { return f(); } fn main() -> i64 { while 1 { g(); } return 0; }",
            "lane.alloc.free_in_loop",
        ),
        (
            "fn main() -> i64 { while __mind_free(0) { } return 0; }",
            "lane.alloc.free_in_loop",
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

// NATTLS B3: the freeaddrinfo and uname libc rows (nbx_row_part 87, 88).
#[test]
fn freeaddrinfo_and_uname_rows_admit() {
    let src = "extern \"C\" {\n    fn freeaddrinfo(res: *mut u8);\n    fn uname(buf: *mut u8) -> i32;\n}\nfn main() -> i64 { return 0; }";
    assert_eq!(admits(src), Ok(()));
    let bad = "extern \"C\" { fn uname(buf: *mut u8) -> i64; }\nfn main() -> i64 { return 0; }";
    assert_eq!(
        admits(bad).unwrap_err().construct,
        "lane.extern.signature_mismatch"
    );
}

// NATTLS B5: a repeated identical extern declaration is one symbol; a repeat
// with a different signature stays refused.
#[test]
fn identical_duplicate_extern_is_one_symbol() {
    let decl = "extern \"C\" { fn memset(ptr: *mut u8, value: i32, num: i64) -> *mut u8; }\n";
    let same = format!("{decl}{decl}fn main() -> i64 {{ return 0; }}");
    assert_eq!(admits(&same), Ok(()));
    let other = format!(
        "{decl}extern \"C\" {{ fn memset(ptr: *mut u8, value: i64, num: i64) -> *mut u8; }}\nfn main() -> i64 {{ return 0; }}"
    );
    assert_eq!(
        admits(&other).unwrap_err().construct,
        "lane.extern.signature_mismatch"
    );
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

// B11 mutant control: the old `let _ = memset(..)` form binds a pointer and
// stays refused, so the ratchet above is not a vacuous pass.
#[test]
fn a_pointer_discard_binding_stays_refused() {
    let src = "extern \"C\" { unsafe fn memset(ptr: *mut u8, value: i32, num: i64) -> *mut u8; }\n\
               fn z(p: i64, n: i64) { let _ = memset(p as *mut u8, 0 as i32, n); }\n\
               fn main() -> i64 { let p: i64 = __mind_alloc(8); z(p, 8); return 0; }";
    let err = profile_lane_admits_source(&crate::parser::parse(src).unwrap(), true).unwrap_err();
    assert_eq!(err.construct, "lane.source.binding_type");
}

// A `let _` discard of an i32 result binds no name (std.net `tcp_close`). A
// named i32 binding still refuses, and a later read of `_` finds nothing.
#[test]
fn an_i32_discard_binds_no_name() {
    let admit = |body: &str| {
        let text = format!(
            "extern \"C\" {{ fn getuid() -> i32; }}\nfn main() -> i64 {{\n{body}\n    return 0;\n}}\n"
        );
        profile_lane_admits_source(&crate::parser::parse(&text).unwrap(), true)
    };
    assert_eq!(admit("    let _ = getuid();"), Ok(()));
    let named = admit("    let u = getuid();").unwrap_err();
    assert_eq!(named.construct, "lane.source.binding_type");
    let read = admit("    let _ = 5;\n    let _ = getuid();\n    let v: i64 = _;").unwrap_err();
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
