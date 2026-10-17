// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! A closed scalar user image may omit an otherwise unused seed prefix.
//!
//! This is deliberately not a name rewrite. Both parsers receive the exact
//! captured user bytes, every user function remains a root, and no seed body
//! remains available to bind a call differently. Only private/private helper
//! collisions qualify. Public names, unresolved calls, implicit library
//! objects, or an unparseable seed remain refusals. Noncolliding images retain
//! the historical prefix and bytes. Type/profile admission is a separate gate.

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Literal, Module, Node, TypeAnn};
use crate::ir::native_closure::{ClosureRejection, ast_definition_counts};

/// Bindings from the same exact table used by mandatory lane type admission.
/// This records arity only AFTER source and IR profile checks have succeeded.
pub(super) fn intrinsic_bindings(ir: &crate::ir::IRModule) -> BTreeMap<String, usize> {
    fn walk(instrs: &[crate::ir::Instr], out: &mut BTreeMap<String, usize>) {
        for instr in instrs {
            if let crate::ir::Instr::Call { name, .. } = instr {
                if let Some(arity) = crate::ir::native_lane_profile::intrinsic_arity(name) {
                    out.insert(name.clone(), arity);
                }
            }
            for body in crate::ir::instr_bodies(instr) {
                walk(body, out);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(&ir.instrs, &mut out);
    out
}

fn reject(detail: impl Into<String>) -> ClosureRejection {
    ClosureRejection {
        kind: "closure.std_symbol_collision",
        detail: detail.into(),
    }
}

fn private_functions(ast: &Module) -> BTreeSet<String> {
    let (mut private, public) = function_visibility(ast);
    private.retain(|name| !public.contains(name));
    private
}

/// Functions defined without `pub`, and names that are `pub` or exported.
/// A name can be in both; `private_functions` is the first minus the second.
fn function_visibility(ast: &Module) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut private = BTreeSet::new();
    let mut public = BTreeSet::new();
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if let Node::FnDef(f, _) = node {
                if f.is_pub {
                    public.insert(f.name.clone());
                } else {
                    private.insert(f.name.clone());
                }
            }
            if let Node::Export { names, .. } = node {
                public.extend(names.iter().cloned());
            }
        });
    }
    (private, public)
}

fn primitive_type(ty: &TypeAnn) -> bool {
    match ty {
        TypeAnn::ScalarI32
        | TypeAnn::ScalarI64
        | TypeAnn::ScalarU32
        | TypeAnn::ScalarF32
        | TypeAnn::ScalarF64
        | TypeAnn::ScalarBool => true,
        TypeAnn::Named(n) => matches!(n.as_str(), "i8" | "u8" | "i16" | "u16" | "u64"),
        TypeAnn::RawPtr { pointee, .. } => primitive_type(pointee),
        _ => false,
    }
}

/// True means omit the COMPLETE seed prefix, after proving all bindings closed.
/// The caller still runs the whole-image lane profile and closure checks.
pub(super) fn omit_private_seed(
    ast: &Module,
    std_sources: &[Vec<u8>],
    reserved: &BTreeSet<String>,
    bindings: &BTreeMap<String, usize>,
) -> Result<bool, ClosureRejection> {
    let defined = ast_definition_counts(ast);
    let collisions: BTreeSet<_> = defined
        .keys()
        .filter(|n| reserved.contains(*n))
        .cloned()
        .collect();
    if collisions.is_empty() {
        return Ok(false);
    }
    let user_private = private_functions(ast);
    if !collisions.is_subset(&user_private) {
        return Err(reject(
            "a public or exported user name collides with the std seed",
        ));
    }
    // Parse, don't infer visibility from text. The old textual reservation set
    // remains conservative; only a positively identified private fn is eligible.
    let mut seed_private = BTreeSet::new();
    let mut seed_public = BTreeSet::new();
    for bytes in std_sources {
        let source = std::str::from_utf8(bytes).map_err(|_| reject("seed is not UTF-8"))?;
        let module =
            crate::parser::parse(source).map_err(|_| reject("seed visibility cannot be parsed"))?;
        let private = private_functions(&module);
        for name in ast_definition_counts(&module).into_keys() {
            if private.contains(&name) {
                seed_private.insert(name);
            } else {
                seed_public.insert(name);
            }
        }
    }
    seed_private.retain(|name| !seed_public.contains(name));
    if !collisions.is_subset(&seed_private) {
        return Err(reject(
            "a user name collides with a public or unclassified std symbol",
        ));
    }

    // Check SOURCE calls too: a lowering may replace surface library syntax
    // with intrinsic IR. The native parser consumes the surface, so an IR-only
    // closed-call check would not establish the source's independence from std.
    let mut failure = None;
    for item in &ast.items {
        crate::type_checker::nerve_walk::walk(item, &mut |node| {
            if failure.is_some() {
                return;
            }
            let types_ok = match node {
                Node::FnDef(f, _) => {
                    f.type_params.is_empty()
                        && f.params.iter().all(|p| primitive_type(&p.ty))
                        && f.ret_type.as_ref().is_none_or(primitive_type)
                }
                Node::Let { ann, .. } => ann.as_ref().is_none_or(primitive_type),
                Node::Const { ty, .. } => ty.as_ref().is_none_or(primitive_type),
                Node::As { ty, .. } => primitive_type(ty),
                Node::ExternBlock { fns, .. } => fns.iter().all(|f| {
                    f.params.iter().all(|p| primitive_type(&p.ty))
                        && f.ret_type.as_ref().is_none_or(primitive_type)
                }),
                // These can introduce objects/functions/types from std without
                // a bare source Call. Refuse rather than assume an inline ABI.
                Node::MethodCall { .. }
                | Node::FieldAccess { .. }
                | Node::Lit(Literal::Str(_), _)
                | Node::MapLit { .. }
                | Node::SetLit { .. }
                | Node::ForEach { .. }
                | Node::StructLit { .. }
                | Node::StructDef { .. }
                | Node::EnumDef { .. }
                | Node::TypeAlias { .. }
                | Node::ImplBlock { .. }
                | Node::TraitDef { .. } => false,
                _ => true,
            };
            if !types_ok {
                failure = Some(
                    "seed omission requires a primitive, explicit-call user image".to_string(),
                );
                return;
            }
            if let Node::Call { callee, args, .. } = node {
                if defined.contains_key(callee) {
                    return;
                }
                if bindings.get(callee) == Some(&args.len()) {
                    return;
                }
                failure = Some(format!(
                    "source call `{callee}` is not closed over the user image and checked bindings"
                ));
            }
        });
    }
    if let Some(detail) = failure {
        return Err(reject(detail));
    }
    Ok(true)
}

/// Colliding names a HISTORICAL (unmarked compiler) image may keep, because
/// the user definition is an exact private copy of the seed's.
///
/// The frozen compiler keeps the LAST definition of each bare name, so a user
/// function named like a seed helper replaces that helper for every caller,
/// std callers included. That is an ambiguity only when the two definitions
/// differ. A reserved name qualifies only when all of these hold, judged on
/// parsed sources rather than on the textual reservation scan:
///
/// * the user image defines it exactly once, as a private, unexported,
///   top-level function with no attributes or type parameters;
/// * every seed module parses, the seed defines the name exactly once in
///   total, as such a function, and no seed module makes it public or exports
///   it;
/// * the two definitions are equal line for line under `comparable_definition`.
///
/// Anything else leaves the name reserved, so the historical
/// `closure.std_symbol_collision` refusal and its message are unchanged, and an
/// image with no qualifying candidate never parses the seed. Nothing is
/// rewritten or omitted: both identical copies stay in the image and the frozen
/// compiler keeps one of them. Calls OUT of a kept copy must still resolve
/// inside the user image; closure admission refuses an open copy as an
/// unresolved edge rather than binding it to a seed body.
pub(super) fn identical_private_seed_copies(
    ast: &Module,
    text: &str,
    std_sources: &[Vec<u8>],
    reserved: &BTreeSet<String>,
) -> BTreeSet<String> {
    let user_counts = ast_definition_counts(ast);
    let user_private = private_functions(ast);
    let mut candidates = plain_private_definitions(ast, text);
    candidates.retain(|name, _| {
        reserved.contains(*name)
            && user_counts.get(*name) == Some(&1)
            && user_private.contains(*name)
    });
    if candidates.is_empty() {
        return BTreeSet::new();
    }
    let mut seed_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut seed_public: BTreeSet<String> = BTreeSet::new();
    let mut seed_definitions: BTreeMap<String, String> = BTreeMap::new();
    for bytes in std_sources {
        let Ok(source) = std::str::from_utf8(bytes) else {
            return BTreeSet::new();
        };
        let Ok(module) = crate::parser::parse(source) else {
            return BTreeSet::new();
        };
        for (name, n) in ast_definition_counts(&module) {
            *seed_counts.entry(name).or_insert(0) += n;
        }
        seed_public.extend(function_visibility(&module).1);
        for (name, definition) in plain_private_definitions(&module, source) {
            if candidates.contains_key(name) {
                seed_definitions.insert(name.to_string(), definition.to_string());
            }
        }
    }
    candidates
        .into_iter()
        .filter(|(name, user)| {
            seed_counts.get(*name) == Some(&1)
                && !seed_public.contains(*name)
                && seed_definitions.get(*name).is_some_and(|seed| {
                    comparable_definition(user)
                        .is_some_and(|u| comparable_definition(seed) == Some(u))
                })
        })
        .map(|(name, _)| name.to_string())
        .collect()
}

/// Private top-level functions with no attributes or type parameters, each
/// paired with the exact source text of its definition (`fn` through `}`).
///
/// A definition whose span does not start at `fn NAME` and end at `}` is left
/// out, which keeps its name reserved.
fn plain_private_definitions<'a>(ast: &'a Module, text: &'a str) -> BTreeMap<&'a str, &'a str> {
    let mut out = BTreeMap::new();
    for item in &ast.items {
        let Node::FnDef(f, span) = item else {
            continue;
        };
        if f.is_pub
            || f.is_test
            || f.reap_threshold.is_some()
            || !f.attrs.is_empty()
            || !f.type_params.is_empty()
        {
            continue;
        }
        let Some(definition) = text.get(span.start()..span.end()) else {
            continue;
        };
        let names_itself = definition
            .strip_prefix("fn")
            .map(str::trim_start)
            .and_then(|rest| rest.strip_prefix(f.name.as_str()))
            .is_some_and(|rest| rest.starts_with(['(', ' ', '\t', '\n']));
        if names_itself && definition.ends_with('}') {
            out.insert(f.name.as_str(), definition);
        }
    }
    out
}

/// The lines of a definition as both native parsers tokenize them, or `None`
/// when that cannot be decided from the bytes alone.
///
/// Each line loses its `//` comment and then its trailing spaces and tabs; the
/// line structure itself is kept, so newline-sensitive parsing sees the same
/// layout. Neither parser has another comment form to recognise (the frozen
/// lexer has none), so the removal is exact once literals are excluded: a
/// string or character literal could hide a `//`, and a block-comment marker
/// could be a comment to one parser and operators to the other. Both are
/// refused rather than interpreted.
fn comparable_definition(definition: &str) -> Option<Vec<&str>> {
    let lines: Vec<&str> = definition
        .split('\n')
        .map(|line| {
            let code = line.find("//").map_or(line, |at| &line[..at]);
            code.trim_end_matches([' ', '\t'])
        })
        .collect();
    let undecidable =
        |line: &&str| line.contains(['"', '\'']) || line.contains("/*") || line.contains("*/");
    (!lines.iter().any(undecidable)).then_some(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::native_image::{compose_image, compose_std_blob};
    use crate::ir::native_closure::reserved_names_from_text;

    const SEED: &str = "fn helper(x: i64) -> i64 { return x + 100; }\npub fn seed_api() -> i64 { return helper(2); }\n";
    fn decision(source: &str, seed: &str) -> Result<bool, ClosureRejection> {
        omit_private_seed(
            &crate::parser::parse(source).unwrap(),
            &[seed.as_bytes().to_vec()],
            &reserved_names_from_text(seed),
            &BTreeMap::new(),
        )
    }
    const USER: &str =
        "fn helper(x: i64) -> i64 { return x + 3; }\nfn main() -> i64 { return helper(4); }\n";

    #[test]
    fn private_collision_omits_prefix_without_rewriting_user_bytes() {
        assert_eq!(decision(USER, SEED), Ok(true));
        let image = compose_image(&compose_std_blob(&[]), &[USER.as_bytes().to_vec()]);
        assert_eq!(image.user_lo, 0);
        assert_eq!(image.src_len as usize, USER.len());
        assert_eq!(&image.bytes[16..], USER.as_bytes());
    }

    #[test]
    fn noncolliding_path_keeps_prefix_even_if_optional_seed_parser_would_fail() {
        assert_eq!(
            decision("fn main() -> i64 { return 7; }", "not valid MIND"),
            Ok(false)
        );
    }

    #[test]
    fn public_and_exported_collisions_remain_refusals() {
        for user in [
            USER.replacen("fn helper", "pub fn helper", 1),
            format!("{USER}\nexport {{ helper }}"),
        ] {
            assert_eq!(
                decision(&user, SEED).unwrap_err().kind,
                "closure.std_symbol_collision"
            );
        }
        assert!(decision(USER, &SEED.replacen("fn helper", "pub fn helper", 1)).is_err());
    }

    #[test]
    fn dead_source_seed_call_also_prevents_omission() {
        let source = format!("{USER}\nfn dead() -> i64 {{ return seed_api(); }}");
        let err = decision(&source, SEED).unwrap_err();
        assert!(err.detail.contains("seed_api"), "{err:?}");
    }

    #[test]
    fn source_to_ir_library_desugaring_cannot_hide_std_dependencies() {
        for suffix in [
            "fn dead() -> i64 { let s: String = 0; return 0; }",
            "fn dead() -> i64 { let s = \"text\"; return 0; }",
            "fn dead() -> i64 { let s: i64 = 0; return s.len(); }",
        ] {
            assert!(
                decision(&format!("{USER}\n{suffix}"), SEED).is_err(),
                "{suffix}"
            );
        }
    }

    #[test]
    fn unparseable_or_only_textually_reserved_seed_cannot_be_omitted() {
        assert!(decision(USER, "fn helper broken seed").is_err());
        assert!(decision(USER, "/*\nfn helper() {}\n*/").is_err());
    }

    #[test]
    fn exact_intrinsic_arity_is_required_for_source_closure() {
        let mut bindings = BTreeMap::new();
        bindings.insert("__mind_alloc".into(), 1);
        for (expression, accepted) in [
            ("__mind_alloc(8)", true),
            ("__mind_alloc()", false),
            ("__mind_forged(8)", false),
        ] {
            let user = format!("{USER}\nfn dead() -> i64 {{ return {expression}; }}");
            let result = omit_private_seed(
                &crate::parser::parse(&user).unwrap(),
                &[SEED.as_bytes().to_vec()],
                &reserved_names_from_text(SEED),
                &bindings,
            );
            assert_eq!(result.is_ok(), accepted, "{result:?}");
        }
    }

    #[test]
    fn actual_seed_private_sha_helper_is_eligible_but_public_api_is_not() {
        let modules = super::super::frozen_std_seed_modules();
        let sources = super::super::seed::std_seed_sources(&modules).unwrap();
        let blob = compose_std_blob(&sources);
        let reserved = reserved_names_from_text(std::str::from_utf8(blob.as_bytes()).unwrap());
        for (name, eligible) in [("bsig0", true), ("sha256", false)] {
            let source = format!(
                "fn {name}(x: i64) -> i64 {{ return x + 3; }}\nfn main() -> i64 {{ return {name}(4); }}"
            );
            let ast = crate::parser::parse(&source).unwrap();
            assert_eq!(
                omit_private_seed(&ast, &sources, &reserved, &BTreeMap::new()).is_ok(),
                eligible
            );
        }
    }

    const COPY_SEED: &str = "fn helper(x: i64) -> i64 {\n    // seed note\n    return x + 100; // tail\n}\npub fn seed_api() -> i64 { return helper(2); }\n";
    const COPY_USER: &str = "fn helper(x: i64) -> i64 {\n    // a different note\n    return x + 100;\n}\nfn main() -> i64 { return helper(4); }\n";
    fn kept(user: &str, seeds: &[&str]) -> BTreeSet<String> {
        let sources: Vec<Vec<u8>> = seeds.iter().map(|s| s.as_bytes().to_vec()).collect();
        identical_private_seed_copies(
            &crate::parser::parse(user).unwrap(),
            user,
            &sources,
            &reserved_names_from_text(&seeds.concat()),
        )
    }
    fn helper_only() -> BTreeSet<String> {
        BTreeSet::from(["helper".to_string()])
    }

    #[test]
    fn exact_private_copy_is_kept_whatever_its_comments_say() {
        assert_eq!(kept(COPY_USER, &[COPY_SEED]), helper_only());
        // A name that does not collide is never reported.
        let other = COPY_USER.replace("helper", "own_helper");
        assert!(kept(&other, &["not valid MIND"]).is_empty());
    }

    #[test]
    fn any_code_or_line_difference_keeps_the_name_reserved() {
        for (from, to) in [
            ("x + 100;", "x + 101;"),
            ("x + 100;", "x  + 100;"),
            ("return x + 100;", "return x +\n        100;"),
            ("(x: i64)", "(x: i32)"),
            ("fn helper(", "pub fn helper("),
            ("fn helper(", "fn helper<T>("),
        ] {
            let user = COPY_USER.replacen(from, to, 1);
            assert!(kept(&user, &[COPY_SEED]).is_empty(), "{to}");
        }
    }

    #[test]
    fn literals_and_block_comment_markers_are_never_compared() {
        for line in ["let s = \"a//b\";", "let c = 'a';"] {
            let seed = COPY_SEED.replace("// seed note", line);
            let user = COPY_USER.replace("// a different note", line);
            assert!(kept(&user, &[&seed]).is_empty(), "{line}");
        }
        // Neither parser accepts block comments; the marker is refused anyway.
        assert_eq!(
            comparable_definition("fn f() -> i64 {\n    /* n */ 1\n}"),
            None
        );
        assert_eq!(comparable_definition("fn f() -> i64 {\n    1 */\n}"), None);
        assert_eq!(
            comparable_definition("fn f() -> i64 {\n    1 // c \"q\" /*\n}"),
            Some(vec!["fn f() -> i64 {", "    1", "}"])
        );
    }

    #[test]
    fn the_seed_side_must_be_one_parsed_private_unexported_definition() {
        let second = "fn helper(x: i64) -> i64 {\n    return x + 100;\n}\n";
        assert!(
            kept(COPY_USER, &[COPY_SEED, second]).is_empty(),
            "defined twice"
        );
        let exported = format!("{COPY_SEED}export {{ helper }}\n");
        assert!(
            kept(COPY_USER, &[&exported]).is_empty(),
            "exported by the seed"
        );
        let public = COPY_SEED.replacen("fn helper", "pub fn helper", 1);
        assert!(kept(COPY_USER, &[&public]).is_empty(), "public in the seed");
        assert!(
            kept(COPY_USER, &[COPY_SEED, "fn broken seed"]).is_empty(),
            "unparseable"
        );
        let nested = format!("module inner {{\n{second}}}\n");
        assert!(
            kept(COPY_USER, &[&nested]).is_empty(),
            "not a top-level item"
        );
    }

    #[test]
    fn the_user_side_must_be_one_private_unexported_top_level_definition() {
        let exported = format!("{COPY_USER}export {{ helper }}\n");
        assert!(
            kept(&exported, &[COPY_SEED]).is_empty(),
            "exported by the user"
        );
        let helper = "fn helper(x: i64) -> i64 {\n    return x + 100;\n}\n";
        let twice = format!("{helper}{helper}fn main() -> i64 {{ return helper(4); }}\n");
        assert!(kept(&twice, &[COPY_SEED]).is_empty(), "defined twice");
        let nested =
            format!("module inner {{\n{helper}}}\nfn main() -> i64 {{ return helper(4); }}\n");
        assert!(
            kept(&nested, &[COPY_SEED]).is_empty(),
            "not a top-level item"
        );
    }

    #[test]
    fn copies_of_the_real_sha256_helpers_are_kept_but_its_public_api_is_not() {
        let modules = super::super::frozen_std_seed_modules();
        let sources: Vec<Vec<u8>> = modules
            .iter()
            .map(|m| std::fs::read(format!("{}/std/{m}.mind", env!("CARGO_MANIFEST_DIR"))).unwrap())
            .collect();
        let blob = compose_std_blob(&sources);
        let reserved = reserved_names_from_text(std::str::from_utf8(blob.as_bytes()).unwrap());
        let sha = modules.iter().position(|m| *m == "sha256").unwrap();
        let sha = std::str::from_utf8(&sources[sha]).unwrap();
        let module = crate::parser::parse(sha).unwrap();
        let seed = plain_private_definitions(&module, sha);
        let user = format!(
            "// vendored\n{}\n\n{}\n\nfn sha256(a: i64, b: i64, c: i64) -> i64 {{ return 0; }}\n\
             fn main() -> i64 {{ return ch(1, 2, 3); }}\n",
            seed["mask32"], seed["ch"]
        );
        let ast = crate::parser::parse(&user).unwrap();
        let found = identical_private_seed_copies(&ast, &user, &sources, &reserved);
        let want: BTreeSet<String> = ["ch", "mask32"].map(String::from).into();
        assert_eq!(found, want);
    }
}
