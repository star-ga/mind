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
    private.retain(|name| !public.contains(name));
    private
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
}
