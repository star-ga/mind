// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
//! Rust admission contracts for the parallel wave-three seed batch.
#![cfg(all(feature = "std-surface", feature = "cross-module-imports", unix))]
mod native_bridge_support;
use libmind::ir::frozen_profile::{profile_frozen_admits, profile_frozen_admits_source};
use libmind::ir::native_closure::{StdSeed, admit_native_closure_linked};
use native_bridge_support::{MINDC, Project};
use std::collections::BTreeSet;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

fn admits(src: &str) -> Result<(), String> {
    let ast = libmind::parser::parse(src).unwrap();
    profile_frozen_admits_source(&ast).map_err(|e| e.construct.to_string())?;
    let ir = libmind::eval::lower_to_ir(&ast).unwrap();
    profile_frozen_admits(&ir).map_err(|e| e.construct.to_string())
}

#[test]
fn signed_integer_ops_pass_and_float_ops_refuse() {
    for op in ["/", "%", "<<", ">>"] {
        let src = format!("fn main() -> i64 {{ let x: i64 = -17; return x {op} 65; }}");
        assert_eq!(admits(&src), Ok(()), "{op}");
        let src = format!("fn main() -> f64 {{ return 17.0 {op} 2.0; }}");
        assert!(admits(&src).is_err(), "float {op}");
    }
}

#[test]
fn unsigned_ops_need_explicit_unsigned_operands() {
    for op in ["/", "%", "<<", ">>"] {
        assert_eq!(
            admits(&format!(
                "fn f(x: u64, y: u64) -> u64 {{ return x {op} y; }} fn main() -> i64 {{ return 0; }}"
            )),
            Ok(()),
            "{op}"
        );
        assert!(admits(&format!("fn f(x: u64, y: f64) -> u64 {{ return x {op} y; }} fn main() -> i64 {{ return 0; }}")).is_err());
        assert_eq!(
            admits(&format!(
                "fn main() -> i64 {{ let x: u64 = 17; let y: u64 = 2; let z = x {op} y; return 0; }}"
            )),
            Ok(()),
            "unsigned local {op}"
        );
    }
    use libmind::ir::{BinOp, IRModule, Instr, ValueId};
    let ir = IRModule {
        instrs: vec![Instr::BinOp {
            dst: ValueId(2),
            op: BinOp::Div,
            lhs: ValueId(0),
            rhs: ValueId(1),
        }],
        ..IRModule::default()
    };
    assert_eq!(
        profile_frozen_admits(&ir).unwrap_err().construct,
        "binop.div_operand_type"
    );
}

#[test]
fn census_conversions_pass_but_unknown_and_wrong_arity_refuse() {
    for (name, arg, ret) in [
        ("i32", "7", "i64"),
        ("i64", "7.0", "i64"),
        ("f64", "7", "f64"),
    ] {
        assert_eq!(
            admits(&format!(
                "fn main() -> {ret} {{ return __mind_conv_{name}({arg}); }}"
            )),
            Ok(()),
            "{name}"
        );
        assert!(
            admits(&format!(
                "fn main() -> {ret} {{ return __mind_conv_{name}({arg}, {arg}); }}"
            ))
            .is_err()
        );
    }
    assert!(admits("fn main() -> i64 { return __mind_conv_unknown(1); }").is_err());
    for (expr, ret) in [
        ("7 as i32", "i64"),
        ("7.0 as i64", "i64"),
        ("7 as f64", "f64"),
    ] {
        assert_eq!(
            admits(&format!("fn main() -> {ret} {{ return {expr}; }}")),
            Ok(())
        );
    }
}

#[test]
fn numeric_proofs_follow_callers_and_returns() {
    assert_eq!(
        admits("fn halve(x: i64) -> i64 { return x / 2; } fn main() -> i64 { return halve(8); }"),
        Ok(())
    );
    for src in [
        "fn halve(x: i64) -> i64 { return x / 2; } fn main() -> i64 { return halve(8.0); }",
        "fn bad() -> i64 { return 8.0; } fn main() -> i64 { return bad() / 2; }",
        "fn cast(x: f64) -> i64 { return __mind_conv_i64(x); } fn main() -> i64 { return cast(8); }",
        "fn bad() -> f64 { return 8; } fn main() -> i64 { return __mind_conv_i64(bad()); }",
    ] {
        assert!(admits(src).is_err(), "{src}");
    }
}

#[test]
fn byte_pointer_casts_pass_other_pointees_refuse() {
    for mutability in ["const", "mut"] {
        assert_eq!(
            admits(&format!(
                "fn main() -> i64 {{ let p = 4096 as *{mutability} u8; return 0; }}"
            )),
            Ok(())
        );
        assert!(
            admits(&format!(
                "fn main() -> i64 {{ let p = 4096 as *{mutability} i32; return 0; }}"
            ))
            .is_err()
        );
        assert!(
            admits(&format!(
                "fn main() -> i64 {{ let p = 1.5 as *{mutability} u8; return 0; }}"
            ))
            .is_err()
        );
    }
}

// A draining sentinel proves the Rust fences invoked stage1 without relying on
// the parallel seed batch or treating an unsupported old emitter as success.
fn bridge(src: &str, profile: &str) -> String {
    let p = Project::new("wave3");
    p.write("src/main.mind", src);
    let compiler = p.write(
        "compiler",
        "#!/bin/sh\ncat >/dev/null\necho WAVE3_STAGE1_REACHED >&2\nexit 42\n",
    );
    std::fs::set_permissions(&compiler, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = Command::new(MINDC)
        .current_dir(p.root())
        .args([
            "build",
            "src/main.mind",
            "--backend",
            "native",
            "--native-profile",
            profile,
            "--out",
            "out.elf",
        ])
        .env("MINDC_NATIVE_ELF", compiler)
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    let expected = if stderr.contains("WAVE3_STAGE1_REACHED") {
        42 // Child status is preserved once admission reaches stage1.
    } else {
        3 // A Rust admission refusal never invokes the sentinel.
    };
    assert_eq!(out.status.code(), Some(expected), "{stderr}");
    assert!(!p.root().join("out.elf").exists());
    stderr
}

#[test]
fn argc_argv_entry_passes_only_the_exact_signature() {
    let good = bridge(
        "fn main(argc: i64, argv: i64) -> i64 { return argc; }",
        "default",
    );
    assert!(good.contains("WAVE3_STAGE1_REACHED"), "{good}");
    for sig in ["n: i64", "argc: f64, argv: i64", "argv: i64, argc: i64"] {
        let bad = bridge(&format!("fn main({sig}) -> i64 {{ return 0; }}"), "default");
        assert!(bad.contains("native.entry_main_has_params"), "{bad}");
    }
    let bad = bridge(
        "fn main(argc: i64, argv: i64) -> f64 { return 0.0; }",
        "default",
    );
    assert!(bad.contains("native.entry_main_has_params"), "{bad}");
}

#[test]
fn bounded_allocations_pass_default_stays_fenced() {
    for (src, label) in [
        (
            "fn a(n: i64) -> i64 { return __mind_alloc(n); } fn main() -> i64 { return a(8); }",
            "call.alloc_size_not_proven_positive",
        ),
        (
            "fn main() -> i64 { let mut n: i64 = 0; while n < 2 { let p = __mind_alloc(8); n = n + 1; } return 0; }",
            "alloc.in_loop",
        ),
        (
            "fn main() -> i64 { let p = __mind_alloc(600000000); return 0; }",
            "alloc.total_over_budget",
        ),
    ] {
        let yes = bridge(src, "bounded");
        assert!(yes.contains("WAVE3_STAGE1_REACHED"), "{yes}");
        let no = bridge(src, "default");
        assert!(no.contains(label), "{no}");
    }
    let recursive = bridge(
        "fn a(n: i64) -> i64 { let p = __mind_alloc(8); return a(n); } fn main() -> i64 { return a(1); }",
        "bounded",
    );
    assert!(recursive.contains("alloc.recursive"), "{recursive}");
    let bad = bridge("fn main() -> i64 { return __mind_alloc(8.0); }", "bounded");
    assert!(bad.contains("call.intrinsic_operand_not_i64"), "{bad}");
}

#[test]
fn extern_posix_requires_matching_signature_and_module() {
    let std_src = "pub fn close(fd: i64) -> i64 { return fd; }";
    let std_ast = libmind::parser::parse(std_src).unwrap();
    let std_ir = libmind::eval::lower_to_ir(&std_ast).unwrap();
    for (module, arg, varargs, accepted) in [
        ("posix", "i64", "", true),
        ("posix", "f64", "", false),
        ("io", "i64", "", false),
        ("posix", "i64", ", ...", false),
    ] {
        let seed = StdSeed {
            modules: vec![(module.into(), BTreeSet::from(["close".into()]))],
            structs: BTreeSet::new(),
            lowered: Some((std_ast.clone(), std_ir.clone())),
        };
        let ast = libmind::parser::parse(&format!("extern \"C\" {{ fn close(fd: {arg}{varargs}) -> i64; }} fn main() -> i64 {{ return __mind_conv_i64(close(0)); }}")).unwrap();
        let ir = libmind::eval::lower_to_ir(&ast).unwrap();
        let result = admit_native_closure_linked(&ast, &ir, "main", &seed);
        assert_eq!(result.is_ok(), accepted, "{module}, {arg}, {varargs}");
        if accepted {
            let linked = result.unwrap().std_linked;
            assert!(linked.contains("close"));
            profile_frozen_admits_source(&ast).unwrap();
            libmind::ir::frozen_profile::profile_frozen_admits_linked(&ir, &linked).unwrap();
            libmind::ir::frozen_profile::profile_frozen_admits_program(
                &ir,
                std::slice::from_ref(&std_ir),
            )
            .unwrap();
        } else {
            assert_eq!(
                result.err().unwrap().kind,
                "closure.extern_signature_mismatch"
            );
        }
    }
}

#[test]
fn every_census_posix_name_resolves_to_its_fixture_body() {
    use libmind::ir::native_closure::reserved_names_from_text;
    let src = include_str!("fixtures/native_posix.mind");
    let std_ast = libmind::parser::parse(src).unwrap();
    let std_ir = libmind::eval::lower_to_ir(&std_ast).unwrap();
    let names = reserved_names_from_text(src);
    let seed = StdSeed {
        modules: vec![("posix".into(), names.clone())],
        structs: BTreeSet::new(),
        lowered: Some((std_ast, std_ir)),
    };
    for line in src.lines().filter(|s| s.starts_with("pub fn")) {
        let signature = line
            .strip_prefix("pub ")
            .unwrap()
            .split(" {")
            .next()
            .unwrap();
        let name = signature
            .strip_prefix("fn ")
            .unwrap()
            .split('(')
            .next()
            .unwrap();
        let params = signature
            .split('(')
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap();
        let arity = if params.is_empty() {
            0
        } else {
            params.split(',').count()
        };
        let args = vec!["0"; arity].join(", ");
        let ast = libmind::parser::parse(&format!(
            "extern \"C\" {{ {signature}; }} fn main() -> i64 {{ return {name}({args}); }}"
        ))
        .unwrap();
        let ir = libmind::eval::lower_to_ir(&ast).unwrap();
        let linked = admit_native_closure_linked(&ast, &ir, "main", &seed).unwrap();
        assert_eq!(linked.std_linked, BTreeSet::from([name.to_owned()]));
        let wrong = signature.replace("-> i64", "-> f64");
        let ast = libmind::parser::parse(&format!(
            "extern \"C\" {{ {wrong}; }} fn main() -> i64 {{ return 0; }}"
        ))
        .unwrap();
        let ir = libmind::eval::lower_to_ir(&ast).unwrap();
        assert_eq!(
            admit_native_closure_linked(&ast, &ir, "main", &seed)
                .err()
                .unwrap()
                .kind,
            "closure.extern_signature_mismatch",
            "{name}"
        );
    }
}
