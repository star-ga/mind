// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
use super::*;

fn parse(source: &str) -> Module {
    crate::parser::parse(source).expect("synthetic fixture must parse before contract checks")
}
fn contract(source: &str) -> Result<NativeCAbi> {
    NativeCAbi::from_ast(
        &parse(source),
        &SymbolContext {
            imported: None,
            reserved: &BTreeSet::new(),
        },
    )
}
fn error(source: &str, kind: &str) {
    assert_eq!(contract(source).unwrap_err().kind, kind, "{source}");
}

#[test]
fn source_widths_survive_where_legacy_ir_erases_them() {
    let source = "extern \"C\" { fn probe(a:i32,b:u32,c:i64,d:u64); }";
    let ast = parse(source);
    let got = contract(source).unwrap();
    assert_eq!(
        got.declarations()[0].parameters,
        [CAbiType::I32, CAbiType::U32, CAbiType::I64, CAbiType::U64]
    );
    let ir = crate::eval::lower_to_ir(&ast).unwrap();
    let params = ir
        .instrs
        .iter()
        .find_map(|i| match i {
            crate::ir::Instr::ExternFnDecl { param_types, .. } => Some(param_types),
            _ => None,
        })
        .unwrap();
    assert_eq!(params, &["i64", "i64", "i64", "i64"]);
}

#[test]
fn all_scalar_spellings_remain_distinct() {
    use CAbiType::*;
    for (spelling, expected) in [
        ("i8", I8),
        ("u8", U8),
        ("i16", I16),
        ("u16", U16),
        ("i32", I32),
        ("u32", U32),
        ("i64", I64),
        ("u64", U64),
        ("isize", Isize),
        ("usize", Usize),
        ("f32", F32),
        ("f64", F64),
        ("bool", Bool),
    ] {
        let got = contract(&format!(
            "extern \"C\" {{fn probe(x:{spelling})->{spelling};}}"
        ))
        .unwrap();
        assert_eq!(got.declarations()[0].parameters, [expected.clone()]);
        assert_eq!(got.declarations()[0].result, expected);
    }
}

#[test]
fn recursive_pointer_pointees_and_mutability_are_preserved() {
    let got =
        contract("extern \"C\" {fn probe(a:*const *mut u8,b:*mut i32)->*const u64;}").unwrap();
    assert_eq!(
        got.declarations()[0].parameters,
        [
            CAbiType::Pointer {
                mutable: false,
                pointee: Box::new(CAbiType::Pointer {
                    mutable: true,
                    pointee: Box::new(CAbiType::U8)
                })
            },
            CAbiType::Pointer {
                mutable: true,
                pointee: Box::new(CAbiType::I32)
            },
        ]
    );
    assert_eq!(
        got.declarations()[0].result,
        CAbiType::Pointer {
            mutable: false,
            pointee: Box::new(CAbiType::U64)
        }
    );
}

#[test]
fn void_means_absent_return_only() {
    assert_eq!(
        contract("extern \"C\" {fn probe();}")
            .unwrap()
            .declarations()[0]
            .result,
        CAbiType::Void
    );
    for source in [
        "extern \"C\" {fn probe()->void;}",
        "extern \"C\" {fn probe(x:void);}",
        "extern \"C\" {fn probe(x:*const void);}",
    ] {
        error(source, "cabi.unsupported_type");
    }
}

#[test]
fn callconv_varargs_and_informational_safety_are_not_erased() {
    for (tag, expected, supported) in [
        ("c", CallConv::C, true),
        ("sysv", CallConv::SysV, true),
        ("win64", CallConv::Win64, false),
        ("aapcs", CallConv::Aapcs, false),
    ] {
        let got = contract(&format!(
            "extern \"C\" callconv(.{tag}) {{safe fn probe(x:i64);}}"
        ))
        .unwrap();
        assert_eq!(got.declarations()[0].callconv, expected);
        assert!(!got.declarations()[0].is_unsafe);
        assert_eq!(got.require_fixed_sysv_amd64().is_ok(), supported);
    }
    let got = contract("extern \"C\" {unsafe fn probe(x:i32,...)->i64;}").unwrap();
    assert!(got.declarations()[0].is_unsafe);
    assert!(got.declarations()[0].is_varargs);
    assert_eq!(got.declarations()[0].parameters.len(), 1);
    assert_eq!(
        got.require_fixed_sysv_amd64().unwrap_err().kind,
        "cabi.varargs_unsupported"
    );
    assert!(
        contract("extern \"C\" {fn probe();}")
            .unwrap()
            .declarations()[0]
            .is_unsafe
    );
}

#[test]
fn duplicate_declarations_and_parameter_names_refuse() {
    for source in [
        "extern \"C\" {fn probe(x:i64);fn probe(x:i64);}",
        "extern \"C\" {fn probe(x:i64);} extern \"C\" {fn probe(x:u64);}",
        "extern \"C\" {fn probe(x:u64);} extern \"C\" {fn probe(x:i64);}",
    ] {
        error(source, "cabi.duplicate_declaration");
    }
    error(
        "extern \"C\" {fn probe(x:i32,x:i64);}",
        "cabi.duplicate_parameter",
    );
}

#[test]
fn local_import_and_intrinsic_conflicts_refuse() {
    for local in [
        "fn probe()->i64{return 0;}",
        "const probe:i64=3;",
        "type probe=i64;",
    ] {
        for source in [
            format!("{local} extern \"C\" {{fn probe();}}"),
            format!("extern \"C\" {{fn probe();}} {local}"),
        ] {
            error(&source, "cabi.local_conflict");
        }
    }
    error(
        "extern \"C\" {fn __mind_future_intrinsic();}",
        "cabi.reserved_symbol",
    );
    let imported = BTreeSet::from(["probe".to_owned()]);
    let reserved = BTreeSet::from(["reserved_entry".to_owned()]);
    let context = SymbolContext {
        imported: Some(&imported),
        reserved: &reserved,
    };
    for (source, kind) in [
        ("extern \"C\" {fn probe();}", "cabi.import_conflict"),
        (
            "extern \"C\" {fn reserved_entry();}",
            "cabi.reserved_symbol",
        ),
        (
            "import other as alias; extern \"C\" {fn alias();}",
            "cabi.import_conflict",
        ),
    ] {
        assert_eq!(
            NativeCAbi::from_ast(&parse(source), &context)
                .unwrap_err()
                .kind,
            kind
        );
    }
}

#[test]
fn aggregates_aliases_and_callbacks_refuse_even_behind_pointers() {
    for ty in [
        "Unknown",
        "[i64;4]",
        "[i64]",
        "&i64",
        "(i64,u32)",
        "Tensor<f64,[4]>",
        "Vec<i64>",
        "*const Unknown",
        "*mut [i64;4]",
    ] {
        error(
            &format!("extern \"C\" {{fn probe(x:{ty});}}"),
            "cabi.unsupported_type",
        );
    }
    error(
        "type Alias=i64; extern \"C\" {fn probe(x:Alias);}",
        "cabi.unsupported_type",
    );
    error(
        "struct Pair {x:i64,y:i64} extern \"C\" {fn probe(x:Pair);}",
        "cabi.unsupported_type",
    );
    for ty in [
        "extern \"C\" fn(i64)->i64",
        "*const extern \"C\" fn(i64)->i64",
    ] {
        error(
            &format!("extern \"C\" {{fn probe(x:{ty});}}"),
            "cabi.callback_unsupported",
        );
    }
}

#[test]
fn aliases_cannot_impersonate_named_scalar_spellings() {
    for (name, ty) in [
        ("u64", "i32"),
        ("usize", "u32"),
        ("i8", "i64"),
        ("i64", "u32"),
        ("f64", "i64"),
    ] {
        for position in [name.to_owned(), format!("*const {name}")] {
            for source in [
                format!("type {name}={ty}; extern \"C\" {{fn probe(x:{position});}}"),
                format!("extern \"C\" {{fn probe()->{position};}} type {name}={ty};"),
            ] {
                error(&source, "cabi.unsupported_type");
            }
        }
    }
    let imported = BTreeSet::from(["u64".to_owned()]);
    let reserved = BTreeSet::new();
    assert_eq!(
        NativeCAbi::from_ast(
            &parse("extern \"C\" {fn probe(x:*const u64);}"),
            &SymbolContext {
                imported: Some(&imported),
                reserved: &reserved
            },
        )
        .unwrap_err()
        .kind,
        "cabi.unsupported_type",
    );
    assert_eq!(
        NativeCAbi::from_ast(
            &parse("import other as u64; extern \"C\" {fn probe(x:u64);}"),
            &SymbolContext {
                imported: Some(&reserved),
                reserved: &reserved
            },
        )
        .unwrap_err()
        .kind,
        "cabi.unsupported_type",
    );
    // An unused alias does not corrupt a separate primitive signature.
    assert!(contract("type u64=i32; extern \"C\" {fn probe(x:i64);}").is_ok());
}

#[test]
fn unresolved_imports_nested_externs_and_data_imports_refuse() {
    error(
        "import other; extern \"C\" {fn probe();}",
        "cabi.unresolved_imports",
    );
    error(
        "fn main()->i64{extern \"C\" {fn probe();} return 0;}",
        "cabi.nested_extern",
    );
    error(
        "module inner {extern \"C\" {fn probe();}}",
        "cabi.nested_extern",
    );
    error(
        "extern const thread_state:i64; extern \"C\" {fn probe();}",
        "cabi.data_import_unsupported",
    );
    assert!(crate::parser::parse("#[thread_local] extern \"C\" {fn probe();}").is_err());
    let empty = BTreeSet::new();
    assert!(
        NativeCAbi::from_ast(
            &parse("import other; extern \"C\" {fn probe();}"),
            &SymbolContext {
                imported: Some(&empty),
                reserved: &empty
            }
        )
        .is_ok()
    );
}

#[test]
fn canonical_wire_has_independently_written_golden_bytes() {
    let source = "extern \"C\" callconv(.sysv) {safe fn probe(a:u32,p:*const u8)->i32;}";
    let got = contract(source).unwrap();
    let expected=b"MCABI001\x01\x00\x00\x00\x05\x00\x00\x00probe\x01\x00\x00\x02\x00\x00\x00\x05\x06\x20\x02";
    assert_eq!(got.descriptor_bytes(), expected);
    assert_eq!(
        got.descriptor_hash(),
        [
            0x73, 0x91, 0x8f, 0x1f, 0x8b, 0xe6, 0x2e, 0x85, 0xdd, 0xae, 0x4d, 0xb7, 0x22, 0xdd,
            0x03, 0xfd, 0x49, 0x5f, 0x6a, 0xff, 0xa4, 0x50, 0xd5, 0xea, 0x59, 0x25, 0x93, 0xbc,
            0x80, 0x8c, 0x2f, 0xc5
        ]
    );
    assert_eq!(contract(" extern \"C\" callconv(.sysv) { safe fn probe(renamed: u32, renamed2: *const u8) -> i32; } ").unwrap().descriptor_bytes(),expected);
    assert_eq!(
        contract(source).unwrap().descriptor_hash(),
        got.descriptor_hash()
    );
    assert_eq!(
        contract("").unwrap().descriptor_bytes(),
        b"MCABI001\0\0\0\0"
    );
}

#[test]
fn identity_tracks_every_abi_field_and_declaration_order() {
    let original = contract("extern \"C\" {fn probe(x:i32)->i64;}").unwrap();
    for variant in [
        "extern \"C\" {fn probe(x:u32)->i64;}",
        "extern \"C\" {fn probe(x:i64)->i64;}",
        "extern \"C\" {fn renamed(x:i32)->i64;}",
        "extern \"C\" {fn probe(x:i32)->u64;}",
        "extern \"C\" {fn probe(x:i32);}",
        "extern \"C\" {fn probe(x:i32,y:i32)->i64;}",
        "extern \"C\" {fn probe(x:i32,...)->i64;}",
        "extern \"C\" {safe fn probe(x:i32)->i64;}",
        "extern \"C\" callconv(.sysv) {fn probe(x:i32)->i64;}",
    ] {
        assert_ne!(
            original.descriptor_hash(),
            contract(variant).unwrap().descriptor_hash(),
            "{variant}"
        );
    }
    let a = contract("extern \"C\" {fn a();fn b();}").unwrap();
    let b = contract("extern \"C\" {fn b();fn a();}").unwrap();
    assert_eq!(
        a.declarations()
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_ne!(a.descriptor_hash(), b.descriptor_hash());
    let p = contract("extern \"C\" {fn probe(x:*const u8);}").unwrap();
    for ty in ["*mut u8", "*const i8", "*const *const u8"] {
        assert_ne!(
            p.descriptor_hash(),
            contract(&format!("extern \"C\" {{fn probe(x:{ty});}}"))
                .unwrap()
                .descriptor_hash()
        );
    }
}

#[test]
fn native_metadata_must_match_exactly_without_pass_flags() {
    let got = contract("extern \"C\" {fn a(x:i32)->u64;fn b(p:*mut u8);}").unwrap();
    let bytes = got.descriptor_bytes();
    got.check_native_descriptor(&bytes).unwrap();
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 1;
        assert_eq!(
            got.check_native_descriptor(&bad).unwrap_err().kind,
            "cabi.native_descriptor_mismatch"
        );
        assert!(got.check_native_descriptor(&bytes[..i]).is_err());
    }
    let mut appended = bytes.clone();
    appended.push(0);
    assert!(got.check_native_descriptor(&appended).is_err());
    for bad in [
        contract("extern \"C\" {fn b(p:*mut u8);fn a(x:i32)->u64;}")
            .unwrap()
            .descriptor_bytes(),
        contract("extern \"C\" {fn a(x:i32)->u64;}")
            .unwrap()
            .descriptor_bytes(),
        b"PASS".to_vec(),
    ] {
        assert!(got.check_native_descriptor(&bad).is_err());
    }
}

#[test]
fn bounded_inventory_has_adjacent_valid_and_invalid_controls() {
    for (count, ok) in [(MAX_DECLARATIONS, true), (MAX_DECLARATIONS + 1, false)] {
        let decls = (0..count)
            .map(|i| format!("fn f{i}();"))
            .collect::<String>();
        let result = contract(&format!("extern \"C\" {{{decls}}}"));
        assert_eq!(result.is_ok(), ok);
        if !ok {
            assert_eq!(result.unwrap_err().kind, "cabi.declaration_limit");
        }
    }
    for (count, ok) in [(MAX_PARAMETERS, true), (MAX_PARAMETERS + 1, false)] {
        let params = (0..count)
            .map(|i| format!("p{i}:i64"))
            .collect::<Vec<_>>()
            .join(",");
        let result = contract(&format!("extern \"C\" {{fn f({params});}}"));
        assert_eq!(result.is_ok(), ok);
        if !ok {
            assert_eq!(result.unwrap_err().kind, "cabi.parameter_limit");
        }
    }
    for (depth, ok) in [(MAX_POINTER_DEPTH, true), (MAX_POINTER_DEPTH + 1, false)] {
        let ty = format!("{}u8", "*const ".repeat(depth));
        let result = contract(&format!("extern \"C\" {{fn f(x:{ty});}}"));
        assert_eq!(result.is_ok(), ok);
        if !ok {
            assert_eq!(result.unwrap_err().kind, "cabi.pointer_depth");
        }
    }
    for (length, ok) in [(MAX_SYMBOL_BYTES, true), (MAX_SYMBOL_BYTES + 1, false)] {
        let result = contract(&format!("extern \"C\" {{fn {}();}}", "f".repeat(length)));
        assert_eq!(result.is_ok(), ok);
        if !ok {
            assert_eq!(result.unwrap_err().kind, "cabi.symbol");
        }
    }
}

#[test]
fn descriptors_do_not_enable_native_build_admission() {
    let ast = parse("extern \"C\" {fn probe(x:i32)->i64;} fn main()->i64{return 0;}");
    let empty = BTreeSet::new();
    NativeCAbi::from_ast(
        &ast,
        &SymbolContext {
            imported: None,
            reserved: &empty,
        },
    )
    .unwrap()
    .require_fixed_sysv_amd64()
    .unwrap();
    assert_eq!(
        crate::ir::native_lane_profile::profile_lane_admits_source(&ast)
            .unwrap_err()
            .construct,
        "lane.extern.unimplemented"
    );
}
