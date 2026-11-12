// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
use super::*;
use crate::ir::native_lane_profile::profile_lane_admits;

const MODULES: [&str; 6] = ["cli", "io", "io_canon", "ring", "sha256", "vec"];

fn std_sources() -> Vec<Vec<u8>> {
    MODULES
        .iter()
        .map(|m| std::fs::read(format!("{}/std/{m}.mind", env!("CARGO_MANIFEST_DIR"))).unwrap())
        .collect()
}

fn admit_with(
    source: &str,
    modules: &[&str],
    sources: &[Vec<u8>],
) -> Check<BTreeMap<String, usize>> {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    let seed = LaneSeed::new(modules, sources);
    profile_lane_admits_source_seeded(&ast, &seed, false)?;
    profile_lane_admits_seeded(&ast, &ir, &seed, false).map(|edges| edges.seed)
}

fn admit(source: &str) -> Check<BTreeMap<String, usize>> {
    admit_with(source, &MODULES, &std_sources())
}

const SHA: &str = "import std.sha256;\nfn main() -> i64 {\n    let m: i64 = __mind_alloc(8);\n    \
    let out: i64 = __mind_alloc(32);\n    let rc: i64 = sha256(m, 0, out);\n    \
    return __mind_load_i8(out) + rc;\n}\n";

#[test]
fn a_seed_call_is_admitted_with_exactly_its_transitive_closure() {
    let reached = admit(SHA).unwrap();
    for (name, arity) in [
        ("sha256", 3),
        ("sha_compress_block", 3),
        ("sha_ch", 3),
        ("sha_mask32", 1),
    ] {
        assert_eq!(reached.get(name), Some(&arity), "{name}: {reached:?}");
    }
    // `hash(input: bytes)` is out of profile and never reached.
    assert!(!reached.contains_key("hash"), "{reached:?}");
    // RED control: the seed-free admission refuses the same call.
    let ast = crate::parser::parse(SHA).unwrap();
    let ir = crate::eval::lower_to_ir(&ast).unwrap();
    let err = profile_lane_admits(&ast, &ir, false).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
}

#[test]
fn a_cross_module_seed_call_reaches_the_other_module() {
    let src = "import std.io_canon;\nfn main() -> i64 {\n    let h: i64 = canon_new(4);\n    \
        let out: i64 = __mind_alloc(32);\n    return canon_anchor(h, out);\n}\n";
    let reached = admit(src).unwrap();
    assert!(reached.contains_key("canon_anchor") && reached.contains_key("sha256"));
}

#[test]
fn a_user_call_binds_only_to_a_public_function_of_an_imported_seed_module() {
    let without_import = SHA.replacen("import std.sha256;\n", "", 1);
    let other_import = SHA.replacen("import std.sha256;", "import std.ring;", 1);
    let private = SHA.replacen(
        "return __mind_load_i8(out) + rc;",
        "return sha_ch(1, 2, 3) + rc;",
        1,
    );
    for source in [
        without_import.as_str(),
        other_import.as_str(),
        private.as_str(),
    ] {
        let err = admit(source).unwrap_err();
        assert_eq!(err.construct, "lane.call.unknown", "{source}");
        assert_eq!(err.seed_module, None, "{source}");
        assert_eq!(err.function.as_deref(), Some("main"), "{source}");
    }
}

#[test]
fn a_user_function_named_like_a_reached_seed_helper_is_ambiguous() {
    let src = format!("{SHA}fn sha_ch(x: i64, y: i64, z: i64) -> i64 {{ return x + y + z; }}\n");
    let err = admit(&src).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module.as_deref(), Some("std/sha256.mind"));
    assert_eq!(err.function.as_deref(), Some("sha_compress_block"));
}

#[test]
fn a_user_function_named_like_a_public_function_of_an_imported_seed_module_is_ambiguous() {
    let sources = vec![b"pub fn f(x: i64) -> i64 { return x; }\n".to_vec()];
    let user = "fn f(x: i64) -> i64 { return x + 1; }\nfn main() -> i64 { return f(1); }\n";
    let err = admit_with(&format!("import std.m;\n{user}"), &["m"], &sources).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module, None);
    assert_eq!(err.function.as_deref(), Some("main"));
    // Control: without the import the seed function is not bindable.
    assert_eq!(admit_with(user, &["m"], &sources), Ok(BTreeMap::new()));
    // A seed signature outside the lane is never typed: still ambiguous.
    let sources = vec![b"pub fn f(v: Vec) -> i64 { return 0; }\n".to_vec()];
    let err = admit_with(&format!("import std.m;\n{user}"), &["m"], &sources).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module, None);
}

#[test]
fn a_reached_seed_body_outside_the_profile_refuses_in_its_own_module() {
    let sources = std_sources();
    let text = |m: &str| {
        String::from_utf8(sources[MODULES.iter().position(|x| *x == m).unwrap()].clone()).unwrap()
    };
    let vec = "import std.vec;\nfn main() -> i64 { return vec_zeroed(4); }\n";
    let err = admit(vec).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
    assert_eq!(err.seed_module.as_deref(), Some("std/vec.mind"));
    assert_eq!(err.function.as_deref(), Some("vec_zeroed"));
    assert_eq!(err.at, text("vec").find("__mind_vec_zeroed(n)"));
    let cli = "import std.cli;\nfn main() -> i64 { return args_get(0, 0); }\n";
    let err = admit(cli).unwrap_err();
    assert_eq!(err.construct, "lane.type.named");
    assert_eq!(err.seed_module.as_deref(), Some("std/cli.mind"));
    assert_eq!(err.function.as_deref(), Some("args_get"));
    assert_eq!(err.at, text("cli").find("fn args_get"));
}

#[test]
fn an_image_without_a_bindable_seed_call_never_parses_the_seed() {
    let garbage = vec![b"fn broken(".to_vec()];
    for source in [
        "fn main() -> i64 { return __mind_alloc(8); }",
        "fn main() -> i64 { return missing(1); }",
        "import std.sha256;\nfn main() -> i64 { return 7; }",
    ] {
        let ast = crate::parser::parse(source).unwrap();
        let seed = LaneSeed::new(&["sha256"], &garbage);
        let _ = profile_lane_admits_source_seeded(&ast, &seed, false);
        assert!(seed.parsed.get().is_none(), "{source}");
    }
    let err = admit_with(SHA, &["sha256"], &garbage).unwrap_err();
    assert_eq!(err.construct, "lane.seed.unparsed");
}

#[test]
fn a_seed_body_never_binds_to_a_user_only_function() {
    let sources = vec![b"pub fn f() -> i64 { return g(); }\n".to_vec()];
    let src = "import std.m;\nfn g() -> i64 { return 1; }\nfn main() -> i64 { return f(); }\n";
    let err = admit_with(src, &["m"], &sources).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    assert_eq!(err.function.as_deref(), Some("f"));
}

#[test]
fn a_name_defined_by_two_seed_modules_is_ambiguous() {
    let sources = vec![
        b"pub fn f(x: i64) -> i64 { return x; }\n".to_vec(),
        b"pub fn f(x: i64) -> i64 { return x + 1; }\n".to_vec(),
    ];
    let src = "import std.a;\nfn main() -> i64 { return f(1); }\n";
    let err = admit_with(src, &["a", "b"], &sources).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module.as_deref(), Some("std/b.mind"));
}

#[test]
fn the_ir_leg_rechecks_every_reached_seed_body() {
    let sources = vec![b"pub fn f(x: i64) -> i64 { return x + 1; }\n".to_vec()];
    let seed = LaneSeed::new(&["m"], &sources);
    let ast = crate::parser::parse("import std.m;\nfn main() -> i64 { return f(1); }\n").unwrap();
    let mut reached = reach(&ast, &seed).unwrap();
    fn callees<'a>(tier: &'a Functions, none: &'a Functions) -> Callees<'a> {
        Callees {
            seed: tier,
            ..Callees::from(none)
        }
    }
    let (none, tier) = (Functions::new(), reached.tier.clone());
    assert_eq!(reached.lowered(callees(&tier, &none), &seed), Ok(()));
    // A signature the lowered body contradicts is caught by the IR leg alone.
    reached.tier.get_mut("f").unwrap().params = vec![super::super::types::Ty::F64];
    let tier = reached.tier.clone();
    let err = reached.lowered(callees(&tier, &none), &seed).unwrap_err();
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    assert_eq!(err.function.as_deref(), Some("f"));
}

/// Source and IR legs for one image with the extern mark given.
fn edges(source: &str, modules: &[&str], sources: &[Vec<u8>], marked: bool) -> Check<LaneEdges> {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    let seed = LaneSeed::new(modules, sources);
    profile_lane_admits_source_seeded(&ast, &seed, marked)?;
    profile_lane_admits_seeded(&ast, &ir, &seed, marked)
}

#[test]
fn an_extern_call_and_a_seed_call_are_admitted_together() {
    let src = SHA.replacen(
        "fn main() -> i64 {",
        "extern \"C\" { fn getuid() -> i32; }\nfn main() -> i64 {\n    let u: i64 = getuid() as i64;",
        1,
    );
    let got = edges(&src, &MODULES, &std_sources(), true).unwrap();
    assert_eq!(got.externs, BTreeMap::from([("getuid".to_string(), 0)]));
    assert_eq!(got.seed.get("sha256"), Some(&3), "{got:?}");
    // RED control: without the extern mark the block refuses first.
    let err = edges(&src, &MODULES, &std_sources(), false).unwrap_err();
    assert_eq!(err.construct, "lane.extern.compiler_unmarked");
}

/// Source and IR legs with the `nbextc01` mark and the `nbsext01` mark given.
fn seed_edges(source: &str, sources: &[Vec<u8>], marked: bool) -> Check<LaneEdges> {
    let ast = crate::parser::parse(source).expect("fixture parses");
    let ir = crate::eval::lower_to_ir(&ast).expect("fixture lowers");
    let seed = LaneSeed::new(&["m", "n"], sources).binding_seed_externs(marked);
    profile_lane_admits_source_seeded(&ast, &seed, true)?;
    profile_lane_admits_seeded(&ast, &ir, &seed, true)
}

/// The seed binds exactly the declarations its reached bodies call.
#[test]
fn a_reached_seed_body_binds_exactly_the_seed_declarations_it_calls() {
    let m = b"extern \"C\" {\n    fn getuid() -> i32;\n    fn close(fd: i32) -> i32;\n    \
        fn open(path: *const u8, flags: i32) -> i32;\n}\n\
        pub fn uid() -> i64 { return getuid() as i64; }\n\
        pub fn shut(fd: i64) -> i64 { return close(fd as i32) as i64; }\n\
        pub fn bad() -> i64 { return open(0 as *const u8, 0 as i32) as i64; }\n\
        pub fn two() -> i64 { return 2; }\n";
    let n = b"extern \"C\" { fn getuid() -> i32; }\npub fn other() -> i64 { return 1; }\n";
    let sources = vec![m.to_vec(), n.to_vec()];
    let main = |body: &str| format!("import std.m;\nfn main() -> i64 {{ return {body}; }}\n");
    let bound = |body: &str| -> Vec<String> {
        let got = seed_edges(&main(body), &sources, true).unwrap();
        assert!(got.externs.is_empty(), "{got:?}");
        got.seed_externs.into_keys().collect()
    };
    assert_eq!(bound("two()"), Vec::<String>::new());
    assert_eq!(bound("uid()"), ["getuid"]);
    assert_eq!(bound("uid() + shut(0 - 1)"), ["close", "getuid"]);
    // A called declaration outside the table refuses in its seed module.
    let err = seed_edges(&main("bad()"), &sources, true).unwrap_err();
    assert_eq!(err.construct, "lane.extern.unknown_symbol");
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    assert_eq!(err.function.as_deref(), Some("open"));
    // RED control: without the mark the seed body's call binds nothing.
    let err = seed_edges(&main("uid()"), &sources, false).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    // A user body never binds a seed declaration.
    let err = seed_edges(&main("getuid() as i64"), &sources, true).unwrap_err();
    assert_eq!(err.construct, "lane.call.unknown");
    assert_eq!(err.seed_module, None);
    // A user function named like a bound seed declaration is ambiguous.
    let clash = format!("{}fn getuid() -> i64 {{ return 7; }}\n", main("uid()"));
    let err = seed_edges(&clash, &sources, true).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    // A user declaration of the same row is the same symbol.
    let same = format!("extern \"C\" {{ fn getuid() -> i32; }}\n{}", main("uid()"));
    let got = seed_edges(&same, &sources, true).unwrap();
    assert_eq!(got.seed_externs.keys().collect::<Vec<_>>(), ["getuid"]);
    // Each seed declaration of a called name must match the table.
    let wrong = vec![
        m.to_vec(),
        b"extern \"C\" { fn getuid() -> i64; }\n".to_vec(),
    ];
    let err = seed_edges(&main("uid()"), &wrong, true).unwrap_err();
    assert_eq!(err.construct, "lane.extern.signature_mismatch");
    assert_eq!(err.seed_module.as_deref(), Some("std/n.mind"));
}

#[test]
fn an_extern_named_like_a_reached_seed_helper_is_ambiguous_in_the_seed_body() {
    let sources = vec![
        b"pub fn f() -> i64 { return getuid(); }\nfn getuid() -> i64 { return 7; }\n".to_vec(),
    ];
    let src =
        "import std.m;\nextern \"C\" { fn getuid() -> i32; }\nfn main() -> i64 { return f(); }\n";
    let err = edges(src, &["m"], &sources, true).unwrap_err();
    assert_eq!(err.construct, "lane.call.ambiguous");
    assert_eq!(err.seed_module.as_deref(), Some("std/m.mind"));
    assert_eq!(err.function.as_deref(), Some("f"));
    // Control: without the extern the seed helper binds alone.
    let src = "import std.m;\nfn main() -> i64 { return f(); }\n";
    let got = edges(src, &["m"], &sources, true).unwrap();
    assert_eq!(got.seed.get("getuid"), Some(&0), "{got:?}");
    assert!(got.externs.is_empty());
}
